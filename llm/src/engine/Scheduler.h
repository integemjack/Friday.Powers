// 推理线程与请求调度。照 llama-server 的 update_slots / pre_decode / decode / post_decode（server-context.cpp）改写，
// 去掉投机解码、LoRA、上下文平移；v2.7 起序列由调度器自己分（请求不再指定序列），加上基数树前缀共享（docs/llm-design.md §3）：
//   - 一个推理线程独占 llama_context：每步「生成中的请求各 1 个 token + 预填中的请求按优先级把批填到 n_batch（在检查点处停下）」，
//     一次 llama_process，采样，逐 token 处理。同时在跑的请求最多 nParallel 个，其余排队（优先级高的先，同级先来先走）；
//   - 排到一个请求时给它挑序列（place）：
//       ① 同一会话（x_friday.session）上次用的空闲序列：原地接着用（账本截到公共前缀 / 退回检查点，== v1）；
//       ② 树上的最长公共前缀够长：从经过那里的序列分叉——新序列先 seq_cp 那段前缀（统一 KV 里只给格子加一个序列号，不拷数据），
//          不可回退的模型再恢复分叉点的检查点；只预填剩下的；
//       ③ 正在预填的请求里有和它共用更长前缀的（同时派出去的几个子智能体）：先等那个请求算过分叉点，再从它分叉，不重复算；
//       ④ 都没有：空序列，或者腾掉最久没用的空闲序列；
//   - llama_process 返回 1（KV 池满）：先腾空闲序列（只腾自己独占格子的、非主会话、最久没用的优先），再减半批大小重试，
//     批大小 1 仍放不下就只让一个请求失败（FLR_KV_FULL）；
//   - 取消：调用线程置标志，生成中下一步就撤下；预填中 PART 型模型由 abort 回调中止 llama_process（≤ 1 个 ubatch），
//     不可回退的模型（混合 / 循环 / SWA）不中止，预填每步最多 n_ubatch 项，取消落在步与步之间；
//   - chunk 在调用线程上回调：推理线程只发「这个 token 要发的文字」与停止信息，增量解析、拼 chunk 在调用线程上做（ChatStream）。
#pragma once

#include "ChatFormat.h"
#include "Sampler.h"
#include "internal.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace flr {

class Scheduler {
public:
    struct Config {
        llama_context* ctx = nullptr;
        const llama_vocab* vocab = nullptr;
        MemoryOps* memory = nullptr;
        /// 看图；没有 mmproj 时为空
        Media* media = nullptr;
        /// llama_n_batch（一次 llama_process 最多的 token 数）
        int nBatch = 512;
        /// llama_n_ubatch（不可回退的模型上预填每步最多这么多项，取消不中止 llama_process 也能及时生效）
        int nUbatch = 512;
        /// 单序列上限（== server 的 n_ctx_slot：min(llama_n_ctx_seq, --kv-unified-per-slot, n_ctx_train)）
        int nCtxSeq = 4096;
        /// 能用的序列数（不含工作序列）：同时缓存着的对话 / 前缀最多这么多条
        int nSeq = 1;
        /// llama_n_seq_max（== nSeq + 1，最后一个是工作序列）
        int nSeqMax = 2;
        /// 同时在算的请求最多几个（其余排队）
        int nParallel = 4;
        /// 公共前缀不到这么多项就不分叉（不值得占一个序列号）
        int forkMin = 64;
        /// 正在预填的请求比树上多共用这么多项以上，才等它算完再分叉
        int dedupeMin = 128;
        /// params.special：生成的特殊 token 照样转成文字
        bool special = false;
        /// chunk 的 model 字段
        std::string modelName;
    };

    explicit Scheduler(Config config);
    /// 没调过 stop 时先 stop
    ~Scheduler();
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    /// 起推理线程（llama_set_abort_callback 也在这里设）
    void start();
    /// 卸载：拒绝新请求，排队与在途的请求以 FLR_NOT_READY 结束（正在跑的 llama_process 经 abort 回调中止），推理线程退出。
    /// 阻塞到线程结束；可重复调用
    void stop();

    /// 一次请求（调用线程）：排队、挑序列，阻塞到结束；期间在本线程回调 chunk，每 ≤ 10 ms 调一次 should_stop。
    /// sampler 已经 acceptPrompt。返回 FLR_*，result 是结果（finish_reason / usage / timings / cache，出错时 error）
    int chat(ChatRequest request, Sampler sampler, flr_chunk_fn chunk, flr_should_stop_fn shouldStop, void* user, json& result);

    /// 在推理线程上、等 seq 上没有在途请求时执行 job，阻塞调用线程直到 job 返回。正在卸载 → FLR_NOT_READY
    int withSequence(int seq, const std::function<int(SequenceCache&, MemoryOps&)>& job);
    /// 只在 withSequence 的 job 里调用（推理线程、步与步之间）：清空一个空闲序列（不是 keep），清了返回 true
    bool evictIdleFromJob(int keep);

    /// 排队、在途、序列与基数树的统计（任意线程；推理线程每次排请求、结束请求时更新）
    json stats() const;
    /// 有没有在途或排队的请求（换模型、卸载前看）
    bool busy() const;

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};

} // namespace flr
