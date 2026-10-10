// 推理内核的内部接口（来自 v1 的进程内运行时 friday-llama，Friday 7555bc9^:qt/src/runtime/llama；v2.7 搬进 Friday.Powers 的 friday-llm）。
// C++17；唯一包含 llama / common / mtmd 头文件的地方。分工：
//   Runtime / Scheduler / ChatFormat / Sampler / RuntimeLog：加载、推理线程、组批、采样、逐 token 处理、chunk；
//   SequenceCache / StateFile / Eviction：账本、P / B / E 检查点、复用决策、状态文件；
//   RadixTree：所有序列账本的基数树（前缀共享、分叉、腾地方，docs/llm-design.md §3）；
//   Media：data URL → 位图 → mtmd_tokenize → 媒体块编码预填。
// 这个目录里的文件名不要与 llama / common / mtmd 的头文件同名（log.h、sampling.h、chat.h、common.h、arg.h、json.h…）：
// Windows / macOS 不分大小写，#include "log.h" 会先找到同目录的 Log.h（所以日志是 RuntimeLog.*，采样叫 Sampler.*）。
//
// JSON：内核自己的请求解析 / chunk 生成用 llama.cpp vendor 里的 nlohmann::ordered_json；
// 交给 common 的地方（common_chat_msgs_parse_oaicompat 等）转成 common_json（common_json::parse(j.dump())）。
//
// 推理线程上一个请求的生命周期（WP2 Scheduler 按这个顺序调 WP3 / WP4，WP3 / WP4 按这个顺序假设）：
//  0. 调用线程：ChatFormat 解析请求 → media::extract（有媒体时）→ 模板 → 分词（Media::tokenize 或 common_tokenize）
//     → Prepared（items、P / B、采样、解析参数）；提示超过单序列上限 → FLR_CONTEXT_OVERFLOW（不进推理线程）
//  1. 推理线程绑定序列 seq（-1 = 空闲的非主序列；被占就排队）
//  2. plan = cache.prepare(prepared, memory, seq)：复用 / 截断 / 回退检查点 / 清空。之后 cache.ledger() == items[0, plan.keep)，
//     KV 里正好是这些；n_done = plan.keep；timings.cache_n = cache.nTokens()
//  3. 每一步（一次 llama_process）：
//     a. 生成中的序列：batch.add(sampled, cache.nextPos(), seq, true)；cache.append({sampled})
//     b. 预填中的序列（按请求先后，批满为止；长提示每步最多占 n_batch − 生成中的序列数）：
//        - items[n_done] 是媒体块，且本步还没给这个序列加过 token：
//            r = media.eval(prepared, n_done, ctx, seq, cache.nextPos(), n_batch, &next)；r == 0 → cache.append(items[n_done])，++n_done；
//            r != 0 → cache.rollback(memory, seq)；r == 1（KV 满，评审修正）照 §E.4 先腾空闲序列、再选牺牲者（这一步的批算完以后），
//            腾出地方就下一步重试这一块，选中的是它自己才 FLR_KV_FULL；r == 2 / < 0 同下面 c
//        - stop = cache.nextStop(n_done)；limit = stop ? stop->at : items.size()
//        - while (n_done < limit && 批未满 && items[n_done] 不是媒体块)：
//              batch.add(items[n_done].token, cache.nextPos(), seq, false)；cache.append(items[n_done])；++n_done
//        - n_done == items.size() → 最后一个 token 要 logits，转入生成
//     c. r = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)：
//        r == 0：每个预填中的序列 if (stop && n_done == stop->at) 对 stop->kinds 逐个 cache.takeCheckpoint(kind, memory, seq)；
//                采样 → 停止条件（停止词 / UTF-8 / EOG / max_tokens / cache.nTokens() + 1 >= n_ctx_seq → "length"）→ 增量解析 → chunk
//        r == 1：所有涉及的序列 cache.rollback(memory, seq)；先腾空闲序列（chooseIdleEviction → memory.seqClear + cache.clear），
//                再减半 n_batch 重试；n_batch == 1 仍失败 → chooseVictim(在途)：seqClear + clear，它的请求 FLR_KV_FULL，其余继续
//        r == 2：所有涉及的序列 cache.rollback(memory, seq)；已取消的请求撤下（FLR_CANCELLED），其余下一步接着算
//        r <  0：涉及的请求 FLR_COMPUTE_ERROR，所有涉及的序列 cache.rollback(memory, seq)
//        （评审修正）不可回退的模型（!memory.partialRemovable()）上 r == 2 / r < -1 之后不能按 seq_pos_max 对账：llama 删不掉失败那个
//        ubatch 的位置（循环状态的尾 cell 已经改到 ubatch 末尾，seq_rm 返回 false），没算过的位置还算在序列里。改为
//        cache.discardAfter(memory, seq, 这一步之前的账本长度)：回到那之前最近的检查点，没有就清空。所以这类模型上取消不中止
//        llama_process（只在卸载时中止），预填每步最多 n_ubatch 项，取消落在步与步之间也一样及时
//  4. 结束：账本 = 提示 + 已 decode 的生成 token（最后采样出的那个没进 KV，不在账本里）
// flr_seq_* / flr_prefix_*：Abi（WP2）检查参数后调 seqSave…（WP3），它们经 SequenceHost::withSequence 在推理线程上、
// 等这个序列空闲时操作 SequenceCache + MemoryOps；文件读写在调用线程（§E.1「写盘」）。
#pragma once

#include "Codes.h"

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace flr {

using json = nlohmann::ordered_json;

// MARK: - 日志（WP2 RuntimeLog.cpp）

/// 运行时自己的日志：转给 flr_init 的回调（级别 FLR_LOG_*）；回调没设时丢掉。任意线程。绝不写 stdout / stderr
void log(int level, const std::string& text);
/// printf 风格（末尾不用带换行）
void logf(int level, const char* format, ...);

// MARK: - 账本与一次请求的准备结果

/// 账本的一项：一个文本 token，或者一整个媒体块（图片 / 音频，token 记 LLAMA_TOKEN_NULL）。
/// 序列的账本与它在 KV 里的内容严格一致（== server 的 server_tokens）；复用决策按项比较（媒体块比 media_id）
struct LedgerItem {
    llama_token token = LLAMA_TOKEN_NULL;
    /// 媒体块的 id（mtmd_input_chunk_get_id：图片字节的哈希，同一张图在不同请求里相同）；文本 token 为空
    std::string media_id;
    /// 占的 KV 单元数（文本 1；媒体块 = mtmd_input_chunk_get_n_tokens）
    int n_tokens = 1;
    /// 占的位置数（文本 1；M-RoPE 的图片块 = mtmd_input_chunk_get_n_pos，可能远小于 n_tokens，§B.8）。
    /// 注意（WP4 实测，Qwen3.5）：M-RoPE 图片块的 n_tokens 个 KV 单元，位置（t 维）都等于块的起始位置，块后的文本从「起始 + n_pos」
    /// 接着排，所以图片块之后 llama_memory_seq_pos_max 只到块的起始位置（不是起始 + n_pos − 1）。按 seq_pos_max 判断一项是否在 KV 里时
    /// 用「这一项的起始位置 ≤ seq_pos_max」（媒体块只在 Media::eval 成功后才进账本，不会只进一半）
    int n_pos = 1;

    static LedgerItem text(llama_token token) { LedgerItem item; item.token = token; return item; }
    bool isMedia() const { return !media_id.empty(); }
    bool operator==(const LedgerItem& other) const
    {
        return token == other.token && media_id == other.media_id && n_tokens == other.n_tokens && n_pos == other.n_pos;
    }
    bool operator!=(const LedgerItem& other) const { return !(*this == other); }
};

/// 请求里的一个媒体输入（image_url / input_audio 的 data URL 解开之后，按在提示里出现的顺序）
struct MediaInput {
    /// 文件字节（PNG / JPEG / WAV …，mtmd_helper_bitmap_init_from_buf 认得的格式）
    std::vector<uint8_t> data;
    /// "image" | "audio"
    std::string kind;
};

/// 一次请求模板 + 分词之后的全部结果（调用线程上算好，交给推理线程）。ChatFormat 填，有媒体时 Media 填 items / 媒体块
struct Prepared {
    /// 模板渲染出来的提示（common_chat_session::prompt）与生成提示（generation_prompt，B 边界用）
    std::string prompt;
    std::string generation_prompt;
    /// 要进 KV 的全部内容（文本 token + 媒体块）
    std::vector<LedgerItem> items;
    /// 采样参数：server 默认（params.sampling）+ 请求的 temperature / top_p / top_k / min_p / seed + grammar / 触发词 / 保留 token
    common_params_sampling sampling;
    /// 停止词（chat.additional_stops）
    std::vector<std::string> stops;
    /// max_tokens（-1 = 不限；运行时另按单序列上限停）
    int n_predict = -1;
    /// 请求的 chat_template_kwargs.enable_thinking（日志 / 统计用；真正起作用的是模板）
    bool thinking = false;

    // 检查点边界（§F.2）。都是 items 的下标；-1 = 没有
    /// P：系统提示 + 工具说明结束处（探针渲染法，== Swift 12d0450；可用第一条非 system 消息的 span 交叉核对）
    int prefix_boundary = -1;
    /// B：生成提示之前（add_generation_prompt = false 再渲染、分词，必须是 items 的严格前缀）；只在最后一条消息是 user 时设
    int transcript_boundary = -1;
    /// 最后一条消息是 user（工具结果续写为 false）
    bool user_driven = false;
    /// P 的指纹（prefixFingerprint(items, prefix_boundary)；没有 P 时为空）
    std::string prefix_fingerprint;

    // 媒体（WP4 管生命周期）：media_chunks[k] 是 items[media_items[k]] 那个媒体块的 mtmd_input_chunk（shared_ptr 带删除器）
    std::vector<std::shared_ptr<void>> media_chunks;
    std::vector<int> media_items;

    // 计时（日志 / rt_bench）
    double t_template_ms = 0;
    double t_tokenize_ms = 0;
};

// MARK: - 检查点与 KV 操作

/// 一份序列状态的字节（llama_state_seq_get_data_ext 的结果）。不可变、共用：从一个序列分叉出来的序列拿到同一份检查点，不拷贝
using StateBlob = std::shared_ptr<const std::vector<uint8_t>>;

inline StateBlob makeBlob(std::vector<uint8_t> data)
{
    return data.empty() ? StateBlob() : std::make_shared<const std::vector<uint8_t>>(std::move(data));
}

/// 上下文检查点（§F.2）：混合 / 循环 / SWA 模型里不能回退的那部分状态（PARTIAL_ONLY）
struct Checkpoint {
    /// P = 系统前缀，B = 生成提示之前（回合边界），E = 提示末尾 − 4（同一轮里也会分歧，§F.2）
    enum class Kind { Prefix, Boundary, End };
    Kind kind = Kind::End;
    /// 取检查点时账本的项数（文档 §H.1 写作 n_tokens；纯文本时就是 token 数）
    int64_t n_items = 0;
    /// 取检查点时的下一个位置（M-RoPE 图片块之后 ≠ n_items）
    llama_pos pos = 0;
    /// llama_state_seq_get_data_ext(PARTIAL_ONLY)（Qwen3.5-9B 每个 50.25 MiB）；PART 型模型的位置标记为空
    StateBlob partial;

    bool hasData() const { return partial && !partial->empty(); }
    size_t bytes() const { return partial ? partial->size() : 0; }
};

/// llama_context 上的序列操作。WP2 用 llama_context 实现（makeMemoryOps），WP3 的单测用假的。
/// 都在推理线程上调用；失败返回 false / 空，绝不 abort（不用 common_memory::seq_rm 这类会 GGML_ABORT 的辅助函数，§E.6）
class MemoryOps {
public:
    virtual ~MemoryOps() = default;
    /// common_context_can_seq_rm == PART：能从任意位置截断，不需要检查点
    virtual bool partialRemovable() const = 0;
    /// llama_memory_seq_rm(seq, p0, p1)；不可回退的模型从中间删返回 false
    virtual bool seqRemove(int seq, llama_pos p0, llama_pos p1) = 0;
    /// dst 先清空，再 llama_memory_seq_cp(src, dst, -1, -1)（统一池里注意力 KV 只改元数据，循环状态写时复制）
    virtual void seqCopy(int src, int dst) = 0;
    /// llama_memory_seq_rm(seq, -1, -1)
    virtual void seqClear(int seq) = 0;
    /// llama_memory_seq_pos_max；序列为空 = -1
    virtual llama_pos seqPosMax(int seq) const = 0;
    /// llama_state_seq_get_data_ext(…, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
    virtual std::vector<uint8_t> partialState(int seq) = 0;
    /// llama_state_seq_set_data_ext(…, PARTIAL_ONLY)；返回 0 视为失败
    virtual bool restorePartial(int seq, const std::vector<uint8_t>& data) = 0;
    /// 同上，检查点版（空 = 失败）
    bool restorePartial(int seq, const StateBlob& data) { return data && restorePartial(seq, *data); }
    /// （v2.7）src 的前 [0, p1) 位置交给 dst（dst 先清空）：统一 KV 里只给格子添一个序列号，不拷数据
    virtual void seqShare(int src, int dst, llama_pos p1) = 0;
    /// llama_state_seq_get_data_ext(…, LLAMA_STATE_SEQ_FLAGS_NONE)：整条序列
    virtual std::vector<uint8_t> fullState(int seq) = 0;
    /// llama_state_seq_set_data_ext(…, NONE)：替换序列里原有的内容；返回 0（数据不对、放不下）视为失败
    virtual bool restoreFull(int seq, const std::vector<uint8_t>& data) = 0;
};

/// WP2（Runtime.cpp）：llama_context 上的实现。seq_rm_type 只在加载时测一次（common_context_can_seq_rm 会清空 memory）
std::unique_ptr<MemoryOps> makeMemoryOps(llama_context* ctx, common_context_seq_rm_type seqRmType);

// MARK: - 序列账本与缓存（WP3 SequenceCache.cpp）

/// 检查点保留规则（§F.2：每序列 P + 最近 3 个 B + 最近 2 个 E）
struct CachePolicy {
    int max_boundary = 3;
    int max_end = 2;
    /// E = 提示末尾 − end_offset（== server 的 4）
    int end_offset = 4;
    /// （集成时加）E 离它前面最近的检查点 / 停点（这次的 P、B，或者账本里留着的检查点）不到这么多项就不设 E：
    /// 每个停点要多一次 llama_process + 一份 PARTIAL 拷贝（Qwen3.5-9B 约 8 + 7 ms，全算在首 token 上），而 E 只在下一次请求
    /// 在它附近分歧时省下「E − 前一个检查点」这么多项的重算。用户消息驱动的请求里 E 就在 B 后面几个 token（生成提示的长度）；
    /// 新预填不多的工具结果续写退到上一个检查点也只多算一小段。0 = 每个请求都设（WP3 / server 的规则，单测用）
    int min_end_gap = 0;
    /// false：不设检查点（rt_bench 对照用）
    bool checkpoints = true;
};

/// 一个序列的账本 + 检查点 + 复用决策（纯逻辑，经 MemoryOps 动 KV）。只在推理线程上用。
/// 私有状态全在 State 里（WP3 在 SequenceCache.cpp 里定义），改实现不用动这个头文件
class SequenceCache {
public:
    explicit SequenceCache(CachePolicy policy = {});
    ~SequenceCache();
    SequenceCache(SequenceCache&& other) noexcept;
    SequenceCache& operator=(SequenceCache&& other) noexcept;
    SequenceCache(const SequenceCache&) = delete;
    SequenceCache& operator=(const SequenceCache&) = delete;

    /// prepare 的结果：之后从 items[keep] 开始预填
    struct Plan {
        /// 复用的项数（≤ items.size() − 1：至少算 1 个 token 才有 logits）
        int keep = 0;
        /// 回退到的检查点的项数（-1 = 没回退）
        int restored = -1;
        /// 整条序列清空重算
        bool reset = false;
    };
    /// 复用决策（§F.2；新 token T = p.items，账本 L，公共前缀 n）：
    ///   ① n == |L| 且 |T| > |L| → 直接接着算；② PART 型 → seqRemove(seq, pos(n), -1)（n == |T| 时退一个）；
    ///   ③ 不可回退 → 取 n_items ≤ n 的最近检查点 → restorePartial + seqRemove(seq, cp.pos, -1)；没有 → seqClear 重算。
    /// P 只要 T 以它开头就一直有效；其它 > n 的检查点作废。顺带按 p 排好这次预填要停下取检查点的地方（P / B / E）。
    /// 调用后 ledger() == p.items 的前 keep 项，且 KV 里正好是这些
    Plan prepare(const Prepared& p, MemoryOps& memory, int seq);

    /// 预填要停下的地方：在 at 项（== 已进 KV 的项数）处取 kinds 这几种检查点（同一处可能既是 P 又是 E）
    struct Stop {
        int at = 0;
        std::vector<Checkpoint::Kind> kinds;
    };
    /// 这次请求要停下取检查点的全部地方（at 升序；v2.7：调度器看正在预填的请求会在哪里取检查点，好让同前缀的请求等着从那里分叉）
    const std::vector<Stop>& stops() const;
    /// processed（已进 KV 的项数）之后第一个要停下的地方；没有 = nullopt。
    /// PART 型模型不停（WP3：P / B 只记位置，账本长到那里自动记下）
    std::optional<Stop> nextStop(int processed) const;
    /// 预填刚好到 Stop::at（这一批 decode 成功之后）时调用。取 PARTIAL 检查点（按 CachePolicy 淘汰旧的）。
    /// WP3：前缀快照不在这里整份拷贝——flr_prefix_save 时由 P 检查点 + 注意力 KV 现场重建（savePrefixState），known_prefixes 不影响预填
    void takeCheckpoint(Checkpoint::Kind kind, MemoryOps& memory, int seq);

    /// 一项进了批次（预填的 token、生成的 token；媒体块在 Media::eval 成功之后）。账本先于 decode 记，decode 失败时 rollback
    void append(const LedgerItem& item);
    /// llama_process / Media::eval 失败之后：账本截到 KV 里实际有的（memory.seqPosMax），失效的检查点去掉
    void rollback(MemoryOps& memory, int seq);
    /// （评审修正）不可回退的模型上 llama_process 中止 / 出错（返回 2 或 < -1）之后：llama 删不掉失败那个 ubatch 的位置，
    /// seqPosMax 把没算过的位置也算进来，账本前 limit 项（这一步之前的账本）之后的内容都不可信。回到 n_items ≤ limit 的最近检查点
    /// （restorePartial + seqRemove(cp.pos, -1)，顺带删掉那些没算过的注意力 cell），没有可用的检查点就整条清空。不补取停点检查点
    void discardAfter(MemoryOps& memory, int seq, size_t limit);
    /// 序列被清空之后（memory.seqClear）：账本、检查点、前缀快照全清
    void clear();
    /// 从 source 分叉（v2.7，基数树）：这个序列的 KV 已经由调用方准备好（seqCopy 了 source 的前 keep 项，不可回退的模型再恢复了
    /// keep 处的检查点）。账本换成 source 账本的前 keep 项，source 里 n_items ≤ keep 的检查点一起带过来（共用状态字节，不拷贝）
    void fork(const SequenceCache& source, size_t keep);
    /// 账本的「代」：截断、清空、换内容（adopt / fork）时 +1，只追加时不变（RadixTree::sync 据此只接新的那段）
    uint64_t generation() const;

    const std::vector<LedgerItem>& ledger() const;
    /// KV 单元数（Σ n_tokens），单序列上限按它算
    int64_t nTokens() const;
    /// 下一个位置（Σ n_pos）
    llama_pos nextPos() const;
    const std::vector<Checkpoint>& checkpoints() const;

    /// 这条序列现在以哪个前缀开头：P 的指纹（账本以 P 开头，且留着 P 的检查点或模型是 PART 型）；没有 = nullopt
    std::optional<std::string> prefixFingerprint() const;
    /// 回退到 P（flr_prefix_restore 发现序列里就是同一个前缀时，不读盘）。成功后账本 == 前 P 项
    bool rollbackToPrefix(MemoryOps& memory, int seq);
    /// 刚恢复了一份状态（restoreState）：账本换成它，检查点清空，记下它的前缀指纹
    void adopt(std::vector<LedgerItem> ledger, std::optional<std::string> prefix);
    /// （WP3 加）adopt 之后把状态文件里带回来的检查点放回去（restoreState 用：回合边界文件带着 P 的 PARTIAL 状态，
    /// 恢复后折叠历史仍能回退到 P）。n_items 超出账本、pos 与账本对不上、P 与前缀指纹对不上的忽略
    void addCheckpoint(Checkpoint checkpoint);

private:
    struct State;
    std::unique_ptr<State> m_state;
};

// MARK: - 状态文件与快照（WP3 StateFile.cpp，§F.3）

/// 状态文件头（header_json）
struct StateHeader {
    /// modelFingerprint(GGUF)：恢复时与运行时比对，不符 → FLR_MODEL_MISMATCH
    std::string model_sha;
    /// attention | hybrid | recurrent | swa（flr_info.memory）
    std::string memory;
    /// Σ n_tokens / Σ n_pos
    int64_t n_tokens = 0;
    llama_pos pos_next = 0;
    std::vector<LedgerItem> ledger;
    /// P 的指纹；没有为空
    std::string prefix;
    /// "B"（回合边界）| "P"（前缀快照）| "full"（整条序列）
    std::string boundary;
};

/// 一份序列状态：头 + llama_state_seq_get_data(FLAGS_NONE) 的字节
struct StateSnapshot {
    StateHeader header;
    std::vector<uint8_t> data;
    /// （WP3 加）随状态一起存的检查点（不可回退的模型：回合边界文件带上 P 的 PARTIAL 状态）；
    /// 文件里写在 data 之后，头里的 "checkpoints" 记它们的种类、项数、位置、字节数
    std::vector<Checkpoint> checkpoints;
};

/// 写状态文件："FRKV" | u32 version=1 | u32 header_len | header_json | u64 len | data（小端）
/// [| 每个 snapshot.checkpoints：u64 len | partial（WP3 加，头里的 "checkpoints" 描述）]。
/// path 是 UTF-8（Windows 转宽字符打开）。应用侧（KVCacheStore）已经先写 .saving.bin 再改名，这里直接写 path
bool writeStateFile(const std::string& path, const StateSnapshot& snapshot, std::string* error);
/// 读状态文件；headerOnly = true 时只读头（flr_prefix_restore 先比指纹）。不是 FRKV（例如旧的 llama-server 槽位文件 ggsq）→ nullopt + error
std::optional<StateSnapshot> readStateFile(const std::string& path, bool headerOnly, std::string* error);
/// 模型标识："<GGUF 前 1 MiB 的 SHA-256 十六进制>-<文件字节数>"（分片模型取第一片）。加载时算一次（WP2 调用）
std::string modelFingerprint(const std::string& ggufPath);
/// 前缀指纹（== Swift 12d0450）："<n>-<FNV-1a 64 位，16 位小写十六进制>"。哈希的输入逐项：文本 token = int32 小端 4 字节；
/// 媒体块 = media_id 的 UTF-8 字节 + n_tokens、n_pos 各 int32 小端。n > items.size() 时按 items.size() 算
std::string prefixFingerprint(const std::vector<LedgerItem>& items, int n);
/// （评审修正）XXH64（xxHash 的算法，小端读取）：状态文件版本 2 的数据段与检查点校验（头里的 "data_xxh64" / "xxh64"，16 位小写十六进制）
uint64_t xxh64(const void* data, size_t size, uint64_t seed = 0);

/// 回合边界状态（§F.3）：序列在 B（最近一个用户驱动边界；没有 B 时整条序列）处的完整状态，供 flr_seq_save。
/// 混合模型：seqCopy(seq → workspace) → restorePartial(workspace, B) → seqRemove(workspace, B.pos, -1) → fullState(workspace)
/// → seqClear(workspace)；PART 型直接 seqCopy + seqRemove 到 B。推理线程、seq 空闲时调用。失败 → nullopt
std::optional<StateSnapshot> saveBoundaryState(SequenceCache& cache, MemoryOps& memory, int seq, int workspaceSeq);
/// 前缀快照（§F.3）：序列在 P 处的完整状态，供 flr_prefix_save。没有（不以 P 开头、P 的快照 / 检查点已失效）→ nullopt
std::optional<StateSnapshot> savePrefixState(SequenceCache& cache, MemoryOps& memory, int seq, int workspaceSeq);
/// 把一份快照恢复进 seq：memory.restoreFull + cache.adopt。失败时 seq 被清空、返回 false
bool restoreState(SequenceCache& cache, MemoryOps& memory, int seq, const StateSnapshot& snapshot);

// MARK: - 池满（WP3 Eviction.cpp，§E.4）

/// 一个序列此刻的情况（Scheduler 每次要选的时候现填）
struct SequenceStatus {
    int seq = 0;
    /// seq == 0（主对话）
    bool main = false;
    /// 有在途请求（预填或生成中）
    bool active = false;
    /// 在生成（否则在预填；active 为 false 时无意义）
    bool generating = false;
    /// KV 里的单元数（cache.nTokens()）
    int64_t n_tokens = 0;
    /// 在途请求的序号（越大越年轻）；空闲为 0
    uint64_t serial = 0;
};
/// 池满时先腾哪个空闲序列（非主优先、占得多的优先）；没有可腾的（空闲且有 KV）→ -1
int chooseIdleEviction(const std::vector<SequenceStatus>& all);
/// 腾空闲、减半 n_batch 都不行时让哪个在途请求失败：非主序列里 token 最多的；都没有就最年轻的请求 → seq；没有在途请求 → -1
int chooseVictim(const std::vector<SequenceStatus>& active);

// MARK: - 运行时对 WP3 的服务（WP2 的 Runtime 实现）与 flr_seq_* / flr_prefix_* 的主体（WP3）

class SequenceHost {
public:
    virtual ~SequenceHost() = default;
    /// 应用可用的序列数（== -np）；合法的 seq 是 0…appSequences()−1
    virtual int appSequences() const = 0;
    /// 内部工作序列（== appSequences()，n_seq_max − 1）
    virtual int workspaceSequence() const = 0;
    virtual const std::string& modelSha() const = 0;
    /// attention | hybrid | recurrent | swa
    virtual const std::string& memoryKind() const = 0;
    /// 在推理线程上、等 seq 上没有在途请求时执行 job（job 里可以用工作序列），阻塞调用线程直到 job 返回，原样返回 job 的返回值。
    /// 运行时正在卸载 → 不执行，返回 FLR_NOT_READY
    virtual int withSequence(int seq, const std::function<int(SequenceCache& cache, MemoryOps& memory)>& job) = 0;
    /// （评审修正）只在 withSequence 的 job 里调用（推理线程）：清空一个空闲的应用序列给 keep 腾地方（非主优先、占得多的优先，
    /// == 池满时的 chooseIdleEviction；有在途请求的、空的、keep 自己都不动）。清了返回 true；恢复状态装不进 KV 池时用
    virtual bool evictIdleSequence(int keep)
    {
        (void) keep;
        return false;
    }
};

// result：C ABI 的 result_json 内容（成功时见 friday_llama.h，失败时 {"error":{"message":…}}）。调用线程（文件读写在这里）
int seqSave(SequenceHost& host, int seq, const std::string& path, std::string& result);
int seqRestore(SequenceHost& host, int seq, const std::string& path, std::string& result);
int seqErase(SequenceHost& host, int seq);
int prefixSave(SequenceHost& host, int seq, const std::string& path, std::string& result);
int prefixRestore(SequenceHost& host, int seq, const std::string& path, std::string& result);

// MARK: - 媒体（WP4 Media.cpp，§B.8）

namespace media {
/// 调用线程（ChatFormat 渲染模板之前）：messages（OpenAI 格式）里的 image_url / input_audio / input_video / video_url 部件
/// 解码进 out（按出现顺序），原地换成 {"type":"media_marker","text":marker}（== server 的 oaicompat_content_load_media）。
/// 只认 data URL / 纯 base64（不下载 http、不读 file://）。allowImages / allowAudio 为 false 时遇到对应部件 → FLR_UNSUPPORTED_MEDIA；
/// 格式不对 → FLR_INVALID_REQUEST。没有媒体时什么都不改，返回 FLR_OK
int extract(json& messages, const std::string& marker, bool allowImages, bool allowAudio, std::vector<MediaInput>& out,
            std::string* error);
}

/// mtmd 的封装：一个运行时一个（有 mmproj 时）。
/// 释放顺序：Media 要先于 llama_model 释放（mtmd 上下文引用模型的词表，== server 的 unload）；Prepared 里的媒体块是独立的数据，
/// 可以晚于 Media 释放，但只能交给生成它的那个 Media 的 eval
class Media {
public:
    virtual ~Media() = default;

    /// fit 之前：把 mmproj 要占的显存加进 params.fit_params_target（== server-context.cpp:1045-1066）
    static void reserveForFit(const std::string& mmprojPath, common_params& params);
    /// 加载线程，common_init_from_params 之后：mtmd_init_from_file（media_marker 用随机串）。progress 0…1，返回 false = 取消。
    /// 失败返回 nullptr，error 写原因。
    /// （评审修正）nUbatch = 上下文的 n_ubatch（llama_n_ubatch；0 = 不核对）：非因果注意力的投影（Gemma 3 / Gemma 4 26B·31B /
    /// DeepSeek-V4…）整张图要在一个 ubatch 里（llama_context::decode 里 GGML_ASSERT，进程内会让整个应用退出）。这类投影最多出的
    /// 图片 token 超过 nUbatch 时，按 image_max_tokens = nUbatch 重新加载（mtmd 把图缩到这么多 token 以内）；tokenize / eval 再把关
    static std::unique_ptr<Media> load(const std::string& mmprojPath, llama_model* model, const common_params& params,
                                       const std::function<bool(float)>& progress, std::string* error, int nUbatch = 0);

    /// 随机媒体标记（ChatFormat 渲染时放进 media_marker 部件）
    virtual const std::string& marker() const = 0;
    virtual bool supportsImages() const = 0;
    virtual bool supportsAudio() const = 0;
    /// 调用线程：带标记的提示 + 媒体 → out.items（文本 token + 媒体块）、out.media_chunks / out.media_items（mtmd_tokenize，
    /// add_special / parse_special 为真）。返回 FLR_OK / FLR_INVALID_REQUEST（图片解不开）/ FLR_INTERNAL
    /// （WP4 补：还有 FLR_UNSUPPORTED_MEDIA——这个构建不能直接解视频、模型不能听音频；评审修正：非因果注意力的图片块超过 n_ubatch）。
    /// 多个调用线程可以同时调用。
    /// items 的最后一项总是文本 token。B 边界不必带着图片再渲染、分词一遍：生成提示单独分词，正好是 items 的尾巴（test_media.cpp 的 prepare）
    virtual int tokenize(const std::string& prompt, const std::vector<MediaInput>& inputs, Prepared& out, std::string* error) = 0;
    /// 推理线程：把 p.items[item] 这个媒体块编码并预填进 seq（从位置 pos 开始；mtmd_helper_eval_chunk_single，n_batch 分批）。
    /// 返回 0 = 成功（*nextPos = 之后的位置）；1 / 2 / < 0 同 llama_process（KV 满 / 被中止 / 出错）
    /// （WP4 补：n_batch ≤ 0 = 上下文的 n_batch；非因果注意力的模型不按 n_batch 拆；成功时 *nextPos == pos + items[item].n_pos，
    /// 调用方接着 cache.append(items[item])；失败时 *nextPos 不动，KV 里可能留着半张图，按 rollback 处理。
    /// 评审修正：非因果注意力的图片块超过 n_ubatch 时不调 llama_process、返回 -1——那会 GGML_ASSERT）
    virtual int eval(const Prepared& p, int item, llama_context* ctx, int seq, llama_pos pos, int n_batch, llama_pos* nextPos) = 0;
};

} // namespace flr
