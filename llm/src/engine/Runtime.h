// 一个已加载的模型：加载、推理线程、信息、卸载。
//   - 加载流程照 llama-server 的 main 与 load_model：先自己校验参数（不让 common_params_parse 往 stderr 打字、不让它 exit），
//     再 common_params_parse(LLAMA_EXAMPLE_SERVER)，默认值与 llama-server 逐项相同；统一 KV 池（序列之间共用前缀只改元数据）；
//     序列数按模型的记忆类型定（纯注意力的序列不占额外显存，多留一些缓存着；混合 / 循环模型每个序列一份循环状态，少留）；
//     另加一个内部工作序列；common_init_from_params（fit + 权重 + 上下文 + 预热）；mmproj；common_context_can_seq_rm；聊天模板；
//   - 卸载：拒绝新调用，在途请求以 FLR_NOT_READY 结束，等推理线程跑完当前一步、等所有调用方离开，再释放。
#pragma once

#include "ChatFormat.h"
#include "Scheduler.h"
#include "internal.h"

#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace flr {

/// 加载失败的原因
struct LoadError {
    int code = FLR_INTERNAL;
    std::string message;
};

/// 怎么加载（ModelHost 按请求和本机情况填）
struct LoadOptions {
    std::string modelPath;
    /// 视觉投影；空 = 纯文本
    std::string mmprojPath;
    /// chunk 的 model 字段（Friday 的模型 id）
    std::string modelId;
    /// 同时在算的请求最多几个
    int parallel = 4;
    /// 缓存着的序列数（0 = 自动：纯注意力 32；混合 / 循环 / 滑窗 parallel + 4）
    int sequences = 0;
    /// KV 池（-c；0 = 模型的训练上下文，fit 按显存往下调）
    int contextTokens = 0;
    /// 单个序列最多占多少（--kv-unified-per-slot；0 = 不另限，最多整个池子）
    int perSequence = 0;
    /// 追加给 common_params_parse 的参数（llama-server 的写法；环境变量 FRIDAY_LLAMA_ARGS 也追加在这）
    std::vector<std::string> extraArgs;
};

class Runtime final : public SequenceHost {
public:
    /// 失败返回空，error 写原因。progress 0…1，返回 false = 取消
    static std::unique_ptr<Runtime> load(const LoadOptions& options, flr_progress_fn progress, void* user, LoadError& error);
    ~Runtime() override;

    /// 卸载的前半：拒绝新调用、在途请求以 FLR_NOT_READY 结束、推理线程退出、等调用方离开。之后才能析构。可重复调用
    void shutdown();

    /// 进出的调用计数（卸载要等调用方离开）。enter 返回 false = 正在卸载
    bool enter();
    void leave();

    /// /v1/info 里这个模型的部分
    json info() const;
    /// 一次 chat completion（调用线程）：body 是 OpenAI 请求体；流式时逐块回调 chunk，不流式时 result 是整个 chat.completion
    int chat(json body, flr_chunk_fn chunk, flr_should_stop_fn shouldStop, void* user, json& result);
    /// 有没有在途或排队的请求
    bool busy() const;
    const std::string& modelPath() const { return m_modelPath; }
    const std::string& mmprojPath() const { return m_mmprojPath; }
    const std::string& modelId() const { return m_modelId; }

    // MARK: SequenceHost（StateFile 的 seqSave… 用）
    int appSequences() const override { return m_nSeq; }
    int workspaceSequence() const override { return m_nSeqMax - 1; }
    const std::string& modelSha() const override { return m_modelSha; }
    const std::string& memoryKind() const override { return m_memoryKind; }
    int withSequence(int seq, const std::function<int(SequenceCache& cache, MemoryOps& memory)>& job) override;
    bool evictIdleSequence(int keep) override;

private:
    Runtime() = default;

    common_params m_params;
    /// 模型、上下文的生命周期（析构顺序：调度器 → 模板 / 媒体 → 上下文与模型）
    common_init_result_ptr m_init;
    llama_model* m_model = nullptr;
    llama_context* m_ctx = nullptr;
    const llama_vocab* m_vocab = nullptr;
    common_chat_templates_ptr m_templates;
    std::unique_ptr<Media> m_media;
    std::unique_ptr<MemoryOps> m_memory;
    std::unique_ptr<ChatFormat> m_format;
    std::unique_ptr<Scheduler> m_scheduler;

    common_context_seq_rm_type m_seqRm = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    std::map<std::string, bool> m_caps;
    bool m_supportsThinking = false;
    std::string m_modelId;
    std::string m_modelPath;
    std::string m_mmprojPath;
    std::string m_modelSha;
    std::string m_memoryKind;
    int m_nCtx = 0;
    int m_nCtxSeq = 0;
    int m_nCtxTrain = 0;
    int m_nSeq = 1;
    int m_nSeqMax = 2;
    int m_nParallel = 1;
    int m_nBatch = 0;
    int m_nUbatch = 0;
    uint64_t m_kvBytesPerToken = 0;
    uint64_t m_stateBytesPerSeq = 0;
    double m_loadMs = 0;

    std::mutex m_callMutex;
    std::condition_variable m_callsDone;
    int m_calls = 0;
    bool m_closing = false;
};

/// 进出的调用计数（RAII）
class CallGuard {
public:
    explicit CallGuard(Runtime& runtime)
        : m_runtime(runtime)
        , m_entered(runtime.enter())
    {
    }
    ~CallGuard()
    {
        if (m_entered)
            m_runtime.leave();
    }
    CallGuard(const CallGuard&) = delete;
    CallGuard& operator=(const CallGuard&) = delete;
    explicit operator bool() const { return m_entered; }

private:
    Runtime& m_runtime;
    bool m_entered;
};

/// 交给 common_params_parse 之前的检查与整理：先试读 llama.cpp 的系统配置文件（格式或值不对时 common_params_parse 会 exit(1)）；
/// 去掉会让进程退出的参数与只管控制台输出的参数；每个参数按 llama-server 的参数表核对（未知参数、缺值、处理函数抛异常的值 →
/// false + error）；开头加 --log-disable，并保证参数个数不等于本进程命令行的个数（Windows 上 common_params_parse 遇到相等时
/// 会改用进程的命令行）。out[0] 是程序名
bool prepareArguments(const std::vector<std::string>& args, std::vector<std::string>& out, std::string* error);

/// 看 GGUF 的元数据猜记忆类型（加载前定序列数用）：attention | hybrid | recurrent | swa；读不了按 attention
std::string guessMemoryKind(const std::string& ggufPath);

} // namespace flr
