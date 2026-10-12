// 按显存定容量（docs/llm-design.md §4）：缓存几条序列、KV 池多大、同时算几个请求；以及 KV 余量放行用的小算式。
// RTX PRO 6000（96 GB）上实测（Qwen3.6-35B-A3B，16 个智能体各自多轮）：大头是缓存装得下所有会话（序列多、池子大，不用来回腾掉重算，
// 678 → 463 秒）；同时算的请求超过 8 个反而变慢（统一 KV 里每步的注意力要扫到池里最高的已用格子，并发越多越慢），所以自动最多 8 个，
// 要更多由用户在 Friday 的设置里选（--parallel / FRIDAY_LLM_PARALLEL）。
// 纯计算，不碰 llama（显存由 Runtime 用 common_get_device_memory_data 量好交进来），单测直接测。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace flr {

/// 一张卡上量出来的（common_get_device_memory_data，no_alloc 加载）
struct DeviceMeasure {
    std::string name;
    /// 卡上现在空着的字节
    int64_t free = 0;
    /// 权重 + 上下文 + 计算空间（训练上下文、缺省序列数时）
    int64_t used = 0;
    /// KV 池每多一个 token 多占的字节（含随上下文变大的计算空间）
    int64_t perToken = 0;
    /// 每多一条序列多占的字节（混合 / 循环模型的循环状态；纯注意力模型 0）
    int64_t perSequence = 0;
    /// 要留着不用的（fit 的余量：--fit-target、视觉投影、给生图 / 生视频的预留）
    int64_t margin = 0;
    /// 和内存共用的显卡（Apple 芯片、Jetson、核显）：剩下的也是系统内存，少拿一些
    bool unified = false;
};

struct CapacityInput {
    std::vector<DeviceMeasure> devices;
    /// 模型的训练上下文（池子至少这么大，单序列上限也是它）
    int nCtxTrain = 0;
    /// 量 used 时的序列数（缺省规则下的，见 Runtime::load）
    int baseSequences = 0;
    /// 纯注意力模型：序列不占额外显存，缓存的序列多留一些
    bool attentionOnly = true;
    /// 同时算几个请求按显存定（--parallel 0）；否则用 fixedParallel
    bool autoParallel = true;
    int fixedParallel = 4;
    /// 用户给了序列数（--sequences / -np）：不改
    int fixedSequences = 0;
};

struct CapacityPlan {
    /// false：放不下训练上下文（或量不出来），照旧交给 fit 往下压（nCtx = 0，parallel / sequences 按缺省）
    bool grown = false;
    /// KV 池（-c；0 = 不改）
    int nCtx = 0;
    /// 同时在算的请求最多几个
    int parallel = 4;
    /// 缓存的序列数（不含工作序列）
    int sequences = 0;
    /// 日志：怎么定的
    std::string why;
};

/// 一个会话 / 请求平均要的格子（主智能体 / 子智能体各自独有的那段，系统提示和工具说明共用）
constexpr int kCellsPerRequest = 16384;
/// 按显存定的同时请求数的上下限（用户自己选的不受上限管）
constexpr int kMinParallel = 4;
constexpr int kMaxParallel = 8;
/// 按显存定时最多缓存几条序列（会话 / 子智能体）；混合 / 循环模型每条一份循环状态，最多拿多出来的显存的四分之一
constexpr int kCacheSequences = 64;
/// KV 池最多按这么多个会话放大（kCellsPerRequest 格一个）
constexpr int kPoolSessions = 32;
/// 序列号上限（LLAMA_MAX_SEQ = 256，留一个给工作序列）
constexpr int kMaxSequences = 255;

/// 缺省规则下的序列数（放不下训练上下文、不放大时）：纯注意力 32（至少 parallel + 8），混合 / 循环 / 滑窗 parallel + 4
int defaultSequences(int parallel, bool attentionOnly);

/// 显存多出来的那部分拿多少给 KV 池：独立显卡六成，和内存共用的四分之一（剩下的留给别的程序、friday-voice、计算空间的波动）
double growthShare(bool unified);

CapacityPlan planCapacity(const CapacityInput& input);

/// 给一个请求的生成留多少格子（KV 余量放行时算）：max_tokens 有就按它，最多 4096；不超过单序列上限减去提示
int generationReserve(int nPredict, int64_t promptCells, int nCtxSeq);

} // namespace flr
