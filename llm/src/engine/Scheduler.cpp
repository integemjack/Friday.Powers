// 推理线程与请求调度（见 Scheduler.h）。
// 照 llama.cpp tools/server/server-context.cpp 的 update_slots / pre_decode / decode / post_decode / process_token 改写
// （MIT License，Copyright (c) 2023-2026 The ggml authors）；v2.7 加基数树前缀共享（docs/llm-design.md §3）。
#include "Scheduler.h"

#include "RadixTree.h"
#include "RuntimeLog.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <optional>
#include <thread>

namespace flr {
namespace {

/// 推理线程发给调用线程的一件事
struct StreamEvent {
    enum class Kind { Partial, Final, Failed };
    Kind kind = Kind::Partial;
    /// Partial：这个 token 要发的文字（带 token，可能为空）
    common_chat_input text;
    /// Partial：已生成的 token 数（== server 的 n_decoded）
    uint64_t nDecoded = 0;
    /// Partial：停止词那一块带 timings
    bool withStats = false;
    RequestStats stats;
    /// Final
    StopType stop = StopType::None;
    /// Failed
    int status = FLR_OK;
    std::string message;
    /// 实际用的序列（-1 = 还没排上）
    int seq = -1;
    /// Final：复用决策（日志 / result.cache.how）
    std::string reuse;
};

/// 一个请求（调用线程与推理线程共用）
struct Request {
    uint64_t serial = 0;
    ChatRequest chat;
    std::optional<Sampler> sampler;
    /// 进队列的时刻（ggml_time_us；日志里的排队时间）
    int64_t tSubmit = 0;

    std::mutex mutex;
    std::condition_variable cv;
    std::deque<StreamEvent> events;
    /// 调用方要停（should_stop / chunk 回调返回 false），或解析出错
    std::atomic<bool> cancelled { false };

    // —— 以下只在推理线程上用 ——
    enum class Phase { Queued, Prefill, Generating, Done };
    Phase phase = Phase::Queued;
    int seq = -1;
    /// 在等另一个正在预填的请求算过共用的前缀（只记一次统计）
    bool deferred = false;
    /// 已进 KV 的提示项数
    int nDone = 0;
    /// 这一步加进批的项数
    int added = 0;
    /// 这一步要采样的输出在批里的下标（-1 = 没有）
    int iBatch = -1;
    /// 这一步预填正好停在检查点上
    std::optional<SequenceCache::Stop> stopHere;
    /// 这一步之前账本的项数（不可回退的模型上 llama_process 失败时退回不晚于它的检查点）
    size_t ledgerBefore = 0;
    /// 图片块放不下、又没有空闲序列可腾：这一步的批算完后选牺牲者（resolveMediaRoom）
    bool waitingForRoom = false;
    llama_token sampled = LLAMA_TOKEN_NULL;
    TokenText text;
    RequestStats stats;
    StopType stop = StopType::None;
    bool truncated = false;
    /// 日志：复用决策
    std::string reuse;
    /// 日志：生成阶段 llama_process 与采样各用了多少（微秒）
    int64_t decodeUs = 0;
    int64_t sampleUs = 0;

    void push(StreamEvent event)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            events.push_back(std::move(event));
        }
        cv.notify_one();
    }
};

/// withSequence 排的活
struct Job {
    int seq = 0;
    std::function<int(SequenceCache&, MemoryOps&)> fn;
    std::promise<int> result;
};

struct Pending {
    std::shared_ptr<Request> request;
    std::shared_ptr<Job> job;
};

/// 给一个请求挑的序列
struct Placement {
    /// -1 = 现在没有能用的序列（排着）
    int seq = -1;
    /// 等正在预填的请求算过共用的前缀
    bool wait = false;
    /// 从哪个序列分叉（-1 = 不分叉，原地用 seq）
    int forkFrom = -1;
    /// 分叉时共用的项数
    size_t keep = 0;
    const char* how = "";
};

/// 序列现在归谁（最近一次用它的请求）
struct Owner {
    std::string session;
    bool main = false;
};

/// llama_batch_ext 的包装：加 token 之前自己校验（容量、token、seq、位置），不走 common_batch::add 那样失败就 GGML_ABORT 的路
class Batch {
public:
    Batch(llama_context* ctx, const llama_vocab* vocab)
        : m_batch(llama_batch_ext_init(ctx))
        , m_capacity(int(llama_n_batch(ctx)))
        , m_vocabSize(llama_vocab_n_tokens(vocab))
        , m_seqMax(int(llama_n_seq_max(ctx)))
    {
    }
    ~Batch()
    {
        if (m_batch)
            llama_batch_ext_free(m_batch);
    }
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;

    void clear()
    {
        llama_batch_ext_clear(m_batch);
        m_size = 0;
    }
    int size() const { return m_size; }
    int capacity() const { return m_capacity; }
    bool validToken(llama_token token) const { return token >= 0 && token < m_vocabSize; }

    /// 返回批里的下标；装不下 / token 或 seq 不合法 → -1（不加）
    int add(llama_token token, llama_pos pos, int seq, bool output)
    {
        // llama_batch_ext_add_token 遇到不合法的 token 会先把这一项加进去再返回 -2（留下一个空 token），所以先自己查
        if (m_size >= m_capacity || !validToken(token) || seq < 0 || seq >= m_seqMax || pos < 0)
            return -1;
        const int32_t index = llama_batch_ext_add_token(m_batch, seq, token);
        if (index < 0)
            return -1;
        llama_batch_ext_set_pos(m_batch, index, &pos);
        if (output)
            llama_batch_ext_set_output_logits(m_batch, index, true);
        ++m_size;
        return index;
    }
    bool setOutput(int index) { return llama_batch_ext_set_output_logits(m_batch, index, true); }
    llama_batch_ext* get() const { return m_batch; }

private:
    llama_batch_ext* m_batch = nullptr;
    int m_capacity = 0;
    int m_vocabSize = 0;
    int m_seqMax = 0;
    int m_size = 0;
};

const char* stopName(StopType stop, bool truncated)
{
    switch (stop) {
    case StopType::Eos: return "eos";
    case StopType::Word: return "停止词";
    case StopType::Limit: return truncated ? "单序列上限" : "max_tokens";
    default: return "无";
    }
}

/// a 是不是 b 的前缀（按账本项比较）
bool isPrefix(const std::vector<LedgerItem>& a, const std::vector<LedgerItem>& b)
{
    if (a.size() > b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

size_t commonPrefix(const std::vector<LedgerItem>& a, const std::vector<LedgerItem>& b)
{
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n])
        ++n;
    return n;
}

/// items 前 n 项之后的下一个位置（Σ n_pos）
llama_pos positionAfter(const std::vector<LedgerItem>& items, size_t n)
{
    llama_pos pos = 0;
    for (size_t i = 0; i < n && i < items.size(); ++i)
        pos += items[i].n_pos;
    return pos;
}

/// cache 里 n_items ≤ limit、带状态数据的最近一个检查点（不可回退的模型分叉用）；没有为空
const Checkpoint* restorableAtOrBefore(const SequenceCache& cache, size_t limit)
{
    const Checkpoint* best = nullptr;
    for (const Checkpoint& checkpoint : cache.checkpoints()) {
        if (checkpoint.hasData() && checkpoint.n_items <= int64_t(limit) && (!best || checkpoint.n_items > best->n_items))
            best = &checkpoint;
    }
    return best;
}

json errorResult(int seq, const std::string& message)
{
    return json { { "seq", seq }, { "error", json { { "message", message } } } };
}

} // namespace

struct Scheduler::Impl {
    Config config;
    /// 每个序列一个（含工作序列）；只在推理线程上用
    std::vector<SequenceCache> caches;
    std::unique_ptr<Batch> batch;
    std::thread thread;

    mutable std::mutex mutex;
    std::condition_variable wake;
    std::deque<Pending> pending;
    bool stopRequested = false;
    /// stop() 已经做过（第二次直接返回）
    bool stopped = false;
    std::atomic<bool> stopping { false };
    std::atomic<uint64_t> nextSerial { 1 };
    std::atomic<int> running { 0 };

    // —— 推理线程 ——
    std::vector<std::shared_ptr<Request>> active;
    std::vector<char> busy;
    std::vector<int64_t> lastUsed;
    std::vector<Owner> owners;
    RadixTree tree;
    /// 这一步的批大小上限（KV 满时减半，成功一次就恢复）
    int budget = 0;
    /// 这一步实际的批上限（buildBatch 算：budget；不可回退的模型上预填部分最多 n_ubatch 项）
    int stepLimit = 0;
    /// 取消时中止 llama_process（只有 PART 型模型）：混合 / 循环 / SWA 模型上中止后，llama 对失败那个 ubatch 的
    /// seq_rm 返回 false，没算过的位置还算在序列里（internal.h）。这类模型取消落在步与步之间，预填每步最多 n_ubatch 项
    const bool abortOnCancel;

    // —— 统计（推理线程写，stats() 在任意线程读）——
    mutable std::mutex statsMutex;
    json statsSnapshot = json::object();
    uint64_t requestsDone = 0;
    uint64_t forks = 0;
    uint64_t dedupeWaits = 0;
    uint64_t tokensReused = 0;
    uint64_t tokensShared = 0;
    uint64_t tokensPrefilled = 0;
    uint64_t evictions = 0;

    // —— abort 回调看的：这一批里有哪些请求 ——
    std::vector<Request*> inBatch;
    std::atomic<bool> armed { false };

    explicit Impl(Config value)
        : config(std::move(value))
        , abortOnCancel(config.memory && config.memory->partialRemovable())
    {
        config.nParallel = std::max(1, std::min(config.nParallel, config.nSeq));
        caches.reserve(size_t(config.nSeqMax));
        // E 离前一个检查点不到 256 项就不设（CachePolicy::min_end_gap）：256 项按 6k tok/s 重算约 43 ms，比每个请求的首 token
        // 都多等一次 llama_process + 一份 PARTIAL 拷贝（约 16 ms）划算
        CachePolicy policy;
        policy.min_end_gap = 256;
        for (int i = 0; i < config.nSeqMax; ++i)
            caches.emplace_back(policy);
        busy.assign(size_t(config.nSeqMax), 0);
        lastUsed.assign(size_t(config.nSeqMax), 0);
        owners.assign(size_t(config.nSeqMax), Owner {});
        budget = config.nBatch;
        batch = std::make_unique<Batch>(config.ctx, config.vocab);
        updateStats();
    }

    static bool abortCallback(void* data)
    {
        auto* self = static_cast<Impl*>(data);
        bool abort = self->stopping.load(std::memory_order_relaxed);
        if (!abort && self->abortOnCancel && self->armed.load(std::memory_order_acquire) && !self->inBatch.empty()) {
            // 只在这一批的请求全都取消时中止：中止后已算完的 ubatch 留在 KV，失败的那个 ubatch 被删，
            // 还要继续的请求拿不到 logits，所以有一个没取消就不中止
            abort = true;
            for (const Request* request : self->inBatch) {
                if (!request->cancelled.load(std::memory_order_relaxed)) {
                    abort = false;
                    break;
                }
            }
        }
        // 有意的中止：llama 随后报的 graph_compute failed 之类不算错误
        if (abort)
            setLibraryErrorsQuiet(true);
        return abort;
    }

    void run()
    {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [this] { return stopRequested || !pending.empty() || !active.empty(); });
                if (stopRequested)
                    break;
            }
            try {
                const bool progressed = admit();
                if (!active.empty()) {
                    step();
                } else if (!progressed) {
                    // 排着的请求一个都放不上（不该有：没有在途请求时总能腾出序列）：别空转
                    std::unique_lock<std::mutex> lock(mutex);
                    wake.wait_for(lock, std::chrono::milliseconds(50));
                }
            } catch (const std::exception& exception) {
                recover(exception.what());
            } catch (...) {
                recover("未知异常");
            }
        }
        shutdown();
    }

    /// 推理线程上漏出来的异常（不该有）：在途请求都以 FLR_INTERNAL 结束，账本截到 KV 里实际有的，线程接着跑
    void recover(const std::string& what)
    {
        armed.store(false);
        setLibraryErrorsQuiet(false);
        logf(FLR_LOG_ERROR, "推理线程出错：%s", what.c_str());
        for (const std::shared_ptr<Request>& request : active) {
            if (request->phase == Request::Phase::Done || request->seq < 0)
                continue;
            caches[size_t(request->seq)].rollback(*config.memory, request->seq);
            fail(*request, FLR_INTERNAL, "推理线程出错：" + what);
        }
        compact();
    }

    // MARK: 基数树

    /// 树跟上每个序列现在的账本（只追加的序列只接新的那段）
    void syncTree()
    {
        for (int seq = 0; seq < config.nSeq; ++seq)
            tree.sync(seq, caches[size_t(seq)].ledger(), caches[size_t(seq)].generation());
    }

    /// 清空一个序列（KV、账本、树）
    void clearSequence(int seq)
    {
        config.memory->seqClear(seq);
        caches[size_t(seq)].clear();
        tree.remove(seq);
        owners[size_t(seq)] = Owner {};
    }

    /// 能拿来放新请求的序列：先找空的，再腾最久没用的空闲序列（非主会话优先）。exclude（分叉的来源）不动
    int freeSequence(const std::vector<char>& taken, int exclude)
    {
        int empty = -1;
        int oldest = -1;
        for (int seq = 0; seq < config.nSeq; ++seq) {
            if (busy[size_t(seq)] || taken[size_t(seq)] || seq == exclude)
                continue;
            if (caches[size_t(seq)].ledger().empty()) {
                empty = seq;
                break;
            }
            if (oldest < 0) {
                oldest = seq;
                continue;
            }
            const Owner& a = owners[size_t(seq)];
            const Owner& b = owners[size_t(oldest)];
            if ((b.main && !a.main) || (a.main == b.main && lastUsed[size_t(seq)] < lastUsed[size_t(oldest)]))
                oldest = seq;
        }
        if (empty >= 0)
            return empty;
        return oldest;
    }

    /// 正在预填的请求 A 会在哪里有能分叉的状态（≤ limit）：PART 型模型任何位置都行；不可回退的模型看它已有的检查点和要停的点
    size_t plannedForkPoint(const Request& a, size_t limit) const
    {
        if (config.memory->partialRemovable())
            return limit;
        size_t best = 0;
        const SequenceCache& cache = caches[size_t(a.seq)];
        if (const Checkpoint* checkpoint = restorableAtOrBefore(cache, limit))
            best = size_t(checkpoint->n_items);
        for (const SequenceCache::Stop& stop : cache.stops()) {
            if (stop.at > 0 && size_t(stop.at) <= limit)
                best = std::max(best, size_t(stop.at));
        }
        return best;
    }

    /// 给请求挑序列（见 Scheduler.h 的 ①–④）
    Placement place(const Request& request, const std::vector<char>& taken)
    {
        Placement placement;
        const std::vector<LedgerItem>& items = request.chat.prepared.items;
        const size_t limit = items.empty() ? 0 : items.size() - 1;   // 至少算 1 项才有 logits
        const bool part = config.memory->partialRemovable();
        const std::string& session = request.chat.session;

        // ① 同一会话上次用的空闲序列：原地接着用（它的旧分支这个会话不会再用了）
        if (!session.empty()) {
            int best = -1;
            size_t bestPrefix = 0;
            for (int seq = 0; seq < config.nSeq; ++seq) {
                if (busy[size_t(seq)] || taken[size_t(seq)] || owners[size_t(seq)].session != session || caches[size_t(seq)].ledger().empty())
                    continue;
                const size_t prefix = commonPrefix(caches[size_t(seq)].ledger(), items);
                if (best < 0 || prefix > bestPrefix) {
                    best = seq;
                    bestPrefix = prefix;
                }
            }
            if (best >= 0) {
                // 同一会话的旧序列只剩很短的公共前缀、树上别处能共用得多得多（比如历史被折叠了，系统提示还在别的序列里）：
                // 不原地截断，下面去分叉（旧序列留给 LRU）
                const RadixTree::Match elsewhere = tree.match(items, { best });
                if (!(elsewhere.length >= bestPrefix + size_t(config.dedupeMin) && elsewhere.length >= size_t(config.forkMin))) {
                    placement.seq = best;
                    placement.how = "同一会话";
                    return placement;
                }
            }
        }

        const RadixTree::Match match = tree.match(items);
        const size_t matched = std::min(match.length, limit);

        // ③ 正在预填的请求里有和它共用得更多的：等那个请求算过分叉点
        for (const std::shared_ptr<Request>& other : active) {
            if (other->phase != Request::Phase::Prefill || other->seq < 0)
                continue;
            const size_t shared = std::min(commonPrefix(other->chat.prepared.items, items), limit);
            if (shared < matched + size_t(config.dedupeMin))
                continue;
            const size_t point = plannedForkPoint(*other, shared);
            if (point < matched + size_t(config.dedupeMin))
                continue;
            if (caches[size_t(other->seq)].ledger().size() < point) {
                placement.wait = true;
                return placement;
            }
        }

        // ④' 整个账本就是这次提示的前缀的空闲序列：原地接着算，什么都不丢
        for (const int holder : match.holders) {
            if (busy[size_t(holder)] || taken[size_t(holder)] || holder >= config.nSeq)
                continue;
            const size_t length = caches[size_t(holder)].ledger().size();
            if (length > 0 && length == match.length && length <= limit) {
                placement.seq = holder;
                placement.how = "接着";
                return placement;
            }
        }

        // ② 分叉：从经过公共前缀的序列（空闲的、在算的都行）拿那段前缀
        if (matched >= size_t(config.forkMin)) {
            int source = -1;
            size_t keep = 0;
            for (const int holder : match.holders) {
                if (holder >= config.nSeq)
                    continue;
                size_t k = matched;
                if (!part) {
                    const Checkpoint* checkpoint = restorableAtOrBefore(caches[size_t(holder)], matched);
                    k = checkpoint ? size_t(checkpoint->n_items) : 0;
                }
                if (k > keep) {
                    keep = k;
                    source = holder;
                }
            }
            if (source >= 0 && keep >= size_t(config.forkMin)) {
                placement.seq = freeSequence(taken, source);
                if (placement.seq < 0)
                    return placement;
                placement.forkFrom = source;
                placement.keep = keep;
                placement.how = "分叉";
                return placement;
            }
        }

        // ④ 没有能共用的：空序列，或者腾掉最久没用的
        placement.seq = freeSequence(taken, -1);
        placement.how = "新序列";
        return placement;
    }

    // MARK: 排队与开始

    /// 返回 true = 有进展（开始了请求、执行了活或撤下了取消的请求）
    bool admit()
    {
        std::vector<std::shared_ptr<Request>> cancelledRequests;
        std::vector<std::shared_ptr<Job>> jobs;
        std::vector<std::shared_ptr<Request>> candidates;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (pending.empty())
                return false;
            std::vector<char> jobTaken(busy);
            for (auto it = pending.begin(); it != pending.end();) {
                if (it->request && it->request->cancelled.load()) {
                    cancelledRequests.push_back(it->request);
                    it = pending.erase(it);
                    continue;
                }
                if (it->job) {
                    const size_t seq = size_t(it->job->seq);
                    if (jobTaken[seq]) {
                        ++it;
                        continue;
                    }
                    jobTaken[seq] = 1;
                    jobs.push_back(it->job);
                    it = pending.erase(it);
                    continue;
                }
                candidates.push_back(it->request);
                ++it;
            }
        }
        for (const std::shared_ptr<Request>& request : cancelledRequests)
            request->push(failedEvent(*request, FLR_CANCELLED, "已取消"));
        if (!jobs.empty()) {
            // 上一步的计算可能还在显卡上跑（预填批没有输出时 llama_process 不同步）
            llama_synchronize(config.ctx);
            for (const std::shared_ptr<Job>& job : jobs) {
                int status = FLR_INTERNAL;
                try {
                    status = job->fn(caches[size_t(job->seq)], *config.memory);
                } catch (const std::exception& exception) {
                    logf(FLR_LOG_ERROR, "序列 %d 上的操作出错：%s", job->seq, exception.what());
                } catch (...) {
                    logf(FLR_LOG_ERROR, "序列 %d 上的操作出了未知错误", job->seq);
                }
                job->result.set_value(status);
            }
        }
        if (candidates.empty())
            return !jobs.empty() || !cancelledRequests.empty();

        // 优先级高的先，同级先来先走
        std::stable_sort(candidates.begin(), candidates.end(), [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
            if (a->chat.priority != b->chat.priority)
                return a->chat.priority > b->chat.priority;
            return a->serial < b->serial;
        });
        int runningNow = 0;
        for (const std::shared_ptr<Request>& request : active) {
            if (request->phase != Request::Phase::Done)
                ++runningNow;
        }
        std::vector<char> taken(size_t(config.nSeqMax), 0);
        std::vector<std::shared_ptr<Request>> started;
        if (runningNow < config.nParallel)
            llama_synchronize(config.ctx);
        syncTree();
        for (const std::shared_ptr<Request>& request : candidates) {
            if (runningNow >= config.nParallel)
                break;
            const Placement placement = place(*request, taken);
            if (placement.wait) {
                if (!request->deferred) {
                    request->deferred = true;
                    ++dedupeWaits;
                    logf(FLR_LOG_NOTICE, "请求 #%llu 等正在预填的请求算完共用的前缀再分叉", (unsigned long long) request->serial);
                }
                continue;
            }
            if (placement.seq < 0)
                continue;
            taken[size_t(placement.seq)] = 1;
            startRequest(request, placement);
            started.push_back(request);
            ++runningNow;
            syncTree();
        }
        if (!started.empty()) {
            std::lock_guard<std::mutex> lock(mutex);
            pending.erase(std::remove_if(pending.begin(), pending.end(),
                                         [&started](const Pending& item) {
                                             return item.request && std::find(started.begin(), started.end(), item.request) != started.end();
                                         }),
                          pending.end());
        }
        running.store(runningNow);
        updateStats();
        return !started.empty() || !jobs.empty() || !cancelledRequests.empty();
    }

    /// 从 source 分叉到 seq：seq 拿 source 前 keep 项的 KV（统一池里只加标记）；不可回退的模型再恢复 keep 处的检查点
    bool fork(int seq, int source, size_t keep)
    {
        clearSequence(seq);
        const SequenceCache& from = caches[size_t(source)];
        if (config.memory->partialRemovable()) {
            config.memory->seqShare(source, seq, positionAfter(from.ledger(), keep));
        } else {
            const Checkpoint* checkpoint = nullptr;
            for (const Checkpoint& candidate : from.checkpoints()) {
                if (candidate.hasData() && candidate.n_items == int64_t(keep))
                    checkpoint = &candidate;
            }
            if (!checkpoint)
                return false;
            // 先拿注意力 KV 的那段（循环状态这时跟着 source 的末尾），再用检查点盖掉循环 / 滑窗那部分
            config.memory->seqShare(source, seq, checkpoint->pos);
            if (!config.memory->restorePartial(seq, checkpoint->partial)) {
                clearSequence(seq);
                return false;
            }
        }
        caches[size_t(seq)].fork(from, keep);
        // KV 末位置应该正好接着账本（M-RoPE 的图片块之后 seq_pos_max 只到块的起始位置，那种情况不核对，见 SequenceCache::rollback）
        const std::vector<LedgerItem>& ledger = caches[size_t(seq)].ledger();
        if (!ledger.empty() && !ledger.back().isMedia() && config.memory->seqPosMax(seq) + 1 != caches[size_t(seq)].nextPos()) {
            logf(FLR_LOG_WARN, "序列 %d 从序列 %d 分叉后 KV 末位置 %d 与账本 %d 对不上，改成整段重算", seq, source,
                 int(config.memory->seqPosMax(seq)), int(caches[size_t(seq)].nextPos()));
            clearSequence(seq);
            return false;
        }
        ++forks;
        tokensShared += uint64_t(caches[size_t(seq)].nTokens());
        return true;
    }

    void startRequest(const std::shared_ptr<Request>& request, const Placement& placement)
    {
        const int seq = placement.seq;
        request->seq = seq;
        busy[size_t(seq)] = 1;
        lastUsed[size_t(seq)] = ggml_time_us();
        std::string how = placement.how;
        if (placement.forkFrom >= 0) {
            if (fork(seq, placement.forkFrom, placement.keep)) {
                how = "分叉自序列 " + std::to_string(placement.forkFrom) + "（共用 " + std::to_string(placement.keep) + " 项）";
            } else {
                how = "分叉失败，新序列";
            }
        } else if (owners[size_t(seq)].session != request->chat.session && !caches[size_t(seq)].ledger().empty()
                   && !isPrefix(caches[size_t(seq)].ledger(), request->chat.prepared.items)) {
            // 腾掉别的会话的序列：它独占的格子随 prepare 的截断 / 清空释放
            ++evictions;
        }
        owners[size_t(seq)] = Owner { request->chat.session, request->chat.mainAgent };

        SequenceCache& cache = caches[size_t(seq)];
        const Prepared& prepared = request->chat.prepared;
        SequenceCache::Plan plan;
        try {
            plan = cache.prepare(prepared, *config.memory, seq);
        } catch (const std::exception& exception) {
            logf(FLR_LOG_WARN, "序列 %d 的复用决策出错（整段重算）：%s", seq, exception.what());
            config.memory->seqClear(seq);
            cache.clear();
            plan = SequenceCache::Plan();
            plan.reset = true;
        }
        request->nDone = int(cache.ledger().size());
        // 约定（internal.h）：账本 == items 的前 keep 项，且至少留 1 项来算 logits。不满足就整段重算（不让推理线程卡死）
        if (request->nDone >= int(prepared.items.size()) || !isPrefix(cache.ledger(), prepared.items)) {
            logf(FLR_LOG_WARN, "序列 %d 的账本与提示对不上（%zu / %zu 项），整段重算", seq, cache.ledger().size(), prepared.items.size());
            config.memory->seqClear(seq);
            cache.clear();
            request->nDone = 0;
            plan = SequenceCache::Plan();
            plan.reset = true;
        }
        if (plan.reset)
            request->reuse = how + "，重算";
        else if (plan.restored >= 0)
            request->reuse = how + "，退回检查点 " + std::to_string(plan.restored);
        else
            request->reuse = how;
        request->stats.promptCached = uint64_t(cache.nTokens());
        request->stats.tStart = ggml_time_us();
        request->phase = Request::Phase::Prefill;
        active.push_back(request);
    }

    // MARK: 一步

    void step()
    {
        reapCancelled();
        while (!active.empty()) {
            buildBatch();
            compact();
            if (batch->size() == 0)
                break;
            armed.store(true, std::memory_order_release);
            // KV 满后减小批大小重试时，llama 每次都报一行 failed to find a memory slot：第一次照报，重试的那些降成 DEBUG
            setLibraryErrorsQuiet(budget < config.nBatch);
            const int64_t decodeStart = ggml_time_us();
            const int ret = llama_process(config.ctx, LLAMA_PROCESS_TYPE_DECODE, batch->get());
            armed.store(false, std::memory_order_release);
            setLibraryErrorsQuiet(false);
            if (ret == 0) {
                // 要采样的批（有输出）等计算完：采样本来也要同步，这里先同步，用时记到生成中的请求上（日志）
                bool sampling = false;
                for (const Request* request : inBatch)
                    sampling = sampling || request->iBatch >= 0;
                if (sampling) {
                    llama_synchronize(config.ctx);
                    const int64_t decodeUs = ggml_time_us() - decodeStart;
                    for (Request* request : inBatch) {
                        if (request->phase == Request::Phase::Generating)
                            request->decodeUs += decodeUs;
                    }
                }
                afterDecode();
                // KV 满时减下来的批大小逐步恢复（一下子恢复到 n_batch 的话下一步多半又放不下）
                budget = std::min(config.nBatch, budget * 2);
                compact();
                break;
            }
            // 1：KV 放不下（memory 已恢复到这一步之前）；-1：批次不合法（没动 memory）；2：abort 回调中止；< -1：出错
            // （2 与 < -1 时算完的 ubatch 留在 KV 里）。不可回退的模型上 2 / < -1 之后 seq_pos_max 不可信：涉及的序列退回这一步之前的检查点
            if (!abortOnCancel && (ret == 2 || ret < -1))
                discardBatch();
            else
                rollbackBatch();
            if (ret == 1) {
                if (!makeRoom())
                    break;
                compact();
                continue;
            }
            if (ret == 2) {
                reapCancelled();
                break;
            }
            logf(FLR_LOG_ERROR, "llama_process 返回 %d（%d 个 token 的批）", ret, batch->size());
            for (Request* request : inBatch) {
                if (request->phase != Request::Phase::Done)
                    fail(*request, FLR_COMPUTE_ERROR,
                         ret == -1 ? "推理批次不合法（llama_process 返回 -1）" : "推理计算出错（llama_process 返回 " + std::to_string(ret) + "）");
            }
            compact();
            break;
        }
        // 图片块放不下、又没有空闲序列可腾的请求：这一步的批已经算完，选牺牲者
        resolveMediaRoom();
        compact();
    }

    void buildBatch()
    {
        batch->clear();
        inBatch.clear();
        for (const std::shared_ptr<Request>& request : active) {
            request->iBatch = -1;
            request->added = 0;
            request->stopHere.reset();
            if (request->seq >= 0)
                request->ledgerBefore = caches[size_t(request->seq)].ledger().size();
        }
        // 生成中的请求各 1 个 token（上一步采样出来的）
        int generating = 0;
        for (const std::shared_ptr<Request>& request : active) {
            if (request->phase != Request::Phase::Generating)
                continue;
            SequenceCache& cache = caches[size_t(request->seq)];
            const int index = batch->add(request->sampled, cache.nextPos(), request->seq, true);
            if (index < 0) {
                fail(*request, FLR_INTERNAL, "生成的 token 加不进批次");
                continue;
            }
            cache.append(LedgerItem::text(request->sampled));
            request->iBatch = index;
            request->added = 1;
            inBatch.push_back(request.get());
            ++generating;
        }
        // 预填中的请求按优先级（同级先来先走），把批填到上限（长提示每步最多占 n_batch − 生成中的请求数）。
        // 不可回退的模型上取消不中止 llama_process：预填部分每步最多 n_ubatch 项，取消照样一个 ubatch 内生效
        stepLimit = abortOnCancel ? budget : std::min(budget, generating + std::max(1, config.nUbatch));
        std::vector<Request*> prefilling;
        for (const std::shared_ptr<Request>& request : active) {
            if (request->phase == Request::Phase::Prefill && !request->waitingForRoom)
                prefilling.push_back(request.get());
        }
        std::stable_sort(prefilling.begin(), prefilling.end(), [](const Request* a, const Request* b) {
            if (a->chat.priority != b->chat.priority)
                return a->chat.priority > b->chat.priority;
            return a->serial < b->serial;
        });
        for (Request* request : prefilling) {
            if (batch->size() >= stepLimit)
                break;
            prefill(*request);
        }
    }

    void prefill(Request& request)
    {
        const Prepared& prepared = request.chat.prepared;
        const std::vector<LedgerItem>& items = prepared.items;
        const int total = int(items.size());
        SequenceCache& cache = caches[size_t(request.seq)];

        // 媒体块：本步还没给这个序列加过 token 时单独编码 + 预填（它自己 decode）
        if (request.nDone < total && items[size_t(request.nDone)].isMedia()) {
            if (!config.media) {
                fail(request, FLR_UNSUPPORTED_MEDIA, "这个模型不能看图（没有加载视觉投影 mmproj）");
                return;
            }
            // 已经取消（或正在卸载）就不开始：编码 + 预填一张大图要将近 1 秒，视觉编码本身不能中断
            if (request.cancelled.load() || stopping.load()) {
                fail(request, stopping ? FLR_NOT_READY : FLR_CANCELLED, stopping ? "正在卸载模型" : "已取消");
                return;
            }
            // 这一块单独 decode：abort 回调这时只看这个请求。PART 型模型上取消可以中止图片的预填（rollback 把半张图删干净）
            std::vector<Request*> batchRequests;
            batchRequests.swap(inBatch);
            inBatch.push_back(&request);
            armed.store(true, std::memory_order_release);
            llama_pos next = 0;
            const int result = config.media->eval(prepared, request.nDone, config.ctx, request.seq, cache.nextPos(), config.nBatch, &next);
            armed.store(false, std::memory_order_release);
            setLibraryErrorsQuiet(false);
            inBatch.swap(batchRequests);
            if (result != 0) {
                cache.rollback(*config.memory, request.seq);
                request.nDone = int(cache.ledger().size());
                if (result == 1) {
                    // KV 池放不下这张图：与文字一样先腾空闲序列，下一步重试这一块；没有可腾的，等这一步的批算完再选牺牲者
                    if (!evictIdle("图片", request.seq))
                        request.waitingForRoom = true;
                    return;
                }
                if (result == 2)
                    fail(request, stopping ? FLR_NOT_READY : FLR_CANCELLED, stopping ? "正在卸载模型" : "已取消");
                else
                    fail(request, FLR_COMPUTE_ERROR, "图片编码 / 预填出错（" + std::to_string(result) + "）");
                return;
            }
            const LedgerItem& item = items[size_t(request.nDone)];
            cache.append(item);
            if (next != cache.nextPos())
                logf(FLR_LOG_WARN, "序列 %d：媒体块之后的位置对不上（eval 给 %d，账本 %d）", request.seq, int(next), int(cache.nextPos()));
            request.stats.promptProcessed += uint64_t(item.n_tokens);
            request.stats.tPromptLast = ggml_time_us();
            ++request.nDone;
        }

        std::optional<SequenceCache::Stop> stop = cache.nextStop(request.nDone);
        if (stop && (stop->at <= request.nDone || stop->at > total)) {
            // 约定是 processed 之后的第一个停点；越界的停点不理，免得预填卡住
            logf(FLR_LOG_WARN, "序列 %d 的检查点停点 %d 不在 (%d, %d] 里，忽略", request.seq, stop->at, request.nDone, total);
            stop.reset();
        }
        const int limit = stop ? stop->at : total;
        int last = -1;
        while (request.nDone < limit && batch->size() < stepLimit && !items[size_t(request.nDone)].isMedia()) {
            const LedgerItem& item = items[size_t(request.nDone)];
            if (!batch->validToken(item.token)) {
                fail(request, FLR_INVALID_REQUEST, "提示里有不合法的 token：" + std::to_string(item.token));
                return;
            }
            const int index = batch->add(item.token, cache.nextPos(), request.seq, false);
            if (index < 0)
                break;
            cache.append(item);
            ++request.nDone;
            ++request.added;
            last = index;
        }
        if (request.added == 0)
            return;
        // 提示最后一个 token 要 logits，之后转入生成
        if (request.nDone == total) {
            batch->setOutput(last);
            request.iBatch = last;
        }
        if (stop && request.nDone == stop->at)
            request.stopHere = stop;
        inBatch.push_back(&request);
    }

    void afterDecode()
    {
        const int64_t now = ggml_time_us();
        bool checkpoints = false;
        for (Request* request : inBatch) {
            if (request->phase == Request::Phase::Prefill) {
                request->stats.promptProcessed += uint64_t(request->added);
                request->stats.tPromptLast = now;
                checkpoints = checkpoints || request->stopHere.has_value();
            }
        }
        if (checkpoints) {
            llama_synchronize(config.ctx);
            for (Request* request : inBatch) {
                if (request->phase != Request::Phase::Prefill || !request->stopHere)
                    continue;
                for (const Checkpoint::Kind kind : request->stopHere->kinds) {
                    try {
                        caches[size_t(request->seq)].takeCheckpoint(kind, *config.memory, request->seq);
                    } catch (const std::exception& exception) {
                        logf(FLR_LOG_WARN, "序列 %d 取检查点出错：%s", request->seq, exception.what());
                    }
                }
            }
        }
        for (Request* request : inBatch) {
            if (request->phase == Request::Phase::Done)
                continue;
            // 这一步里被取消的：不再采样（算完的 token 留在 KV、账本里，下一个请求照样复用）
            if (request->cancelled.load()) {
                fail(*request, stopping ? FLR_NOT_READY : FLR_CANCELLED, stopping ? "正在卸载模型" : "已取消");
                continue;
            }
            if (request->iBatch < 0)
                continue;
            if (request->phase == Request::Phase::Prefill)
                request->phase = Request::Phase::Generating;
            llama_token token = LLAMA_TOKEN_NULL;
            const int64_t sampleStart = ggml_time_us();
            try {
                token = request->sampler->sample(config.ctx, request->iBatch);
            } catch (const std::exception& exception) {
                fail(*request, FLR_INTERNAL, std::string("采样出错：") + exception.what());
                continue;
            }
            request->stats.generated += 1;
            const int64_t sampledAt = ggml_time_us();
            request->sampleUs += sampledAt - sampleStart;
            if (request->stats.generated == 1)
                request->stats.tPromptLast = sampledAt;
            request->stats.tGenLast = sampledAt;
            processToken(*request, token);
        }
    }

    /// == server_context_impl::process_token
    void processToken(Request& request, llama_token token)
    {
        const Prepared& prepared = request.chat.prepared;
        const bool eog = llama_vocab_is_eog(config.vocab, token);
        const bool special = config.special || prepared.sampling.preserved_tokens.count(token) > 0;
        const std::string piece = common_token_to_piece(config.ctx, token, special);
        request.sampled = token;

        bool hasNext = true;
        const TokenText::Step text = request.text.push(piece, token, eog, prepared.stops);
        if (text.stopWord) {
            request.stop = StopType::Word;
            hasNext = false;
        }
        if (text.send) {
            StreamEvent event;
            event.kind = StreamEvent::Kind::Partial;
            event.text = text.text;
            event.nDecoded = request.stats.generated;
            event.seq = request.seq;
            // == server：stop 已定（停止词）或请求要 timings_per_token 时这一块带 timings
            if (request.stop != StopType::None || request.chat.timingsPerToken) {
                event.withStats = true;
                event.stats = request.stats;
            }
            request.push(std::move(event));
        }
        // 单序列上限（不做上下文平移）：再生成一个就超了
        const SequenceCache& cache = caches[size_t(request.seq)];
        if (cache.nTokens() + 1 >= config.nCtxSeq) {
            request.truncated = true;
            request.stop = StopType::Limit;
            hasNext = false;
        }
        // max_tokens
        const int limit = prepared.n_predict;
        if (request.stats.generated > 0 && hasNext && limit != -1 && int64_t(request.stats.generated) >= int64_t(limit)) {
            request.stop = StopType::Limit;
            hasNext = false;
        }
        if (eog) {
            request.stop = StopType::Eos;
            hasNext = false;
        }
        if (!hasNext)
            finish(request);
    }

    // MARK: KV 池满

    /// 腾一个空闲序列给 KV 池：只腾自己独占格子的（全是和别的序列共用的格子，腾了也空不出地方）；非主会话、最久没用的优先。
    /// 腾了返回 true。空闲序列不在这一步的批里，组批途中也能清
    bool evictIdle(const char* what, int keep = -1)
    {
        syncTree();
        int best = -1;
        for (int seq = 0; seq < config.nSeq; ++seq) {
            if (busy[size_t(seq)] || seq == keep || caches[size_t(seq)].ledger().empty() || tree.exclusive(seq) == 0)
                continue;
            if (best < 0) {
                best = seq;
                continue;
            }
            const Owner& a = owners[size_t(seq)];
            const Owner& b = owners[size_t(best)];
            if ((b.main && !a.main) || (a.main == b.main && lastUsed[size_t(seq)] < lastUsed[size_t(best)]))
                best = seq;
        }
        if (best < 0)
            return false;
        logf(FLR_LOG_NOTICE, "KV 池满（%s）：腾掉空闲序列 %d（独占 %zu 项，共 %lld 个 token）后重试", what, best, tree.exclusive(best),
             (long long) caches[size_t(best)].nTokens());
        clearSequence(best);
        ++evictions;
        return true;
    }

    std::vector<SequenceStatus> statuses() const
    {
        std::vector<SequenceStatus> all;
        for (int seq = 0; seq < config.nSeq; ++seq) {
            SequenceStatus status;
            status.seq = seq;
            status.main = owners[size_t(seq)].main;
            status.n_tokens = caches[size_t(seq)].nTokens();
            for (const std::shared_ptr<Request>& request : active) {
                if (request->seq == seq && request->phase != Request::Phase::Done) {
                    status.active = true;
                    status.generating = request->phase == Request::Phase::Generating;
                    status.serial = request->serial;
                }
            }
            all.push_back(status);
        }
        return all;
    }

    /// 返回 true = 腾出了地方或换了批大小，重试这一步
    bool makeRoom()
    {
        if (evictIdle("文字"))
            return true;
        if (budget > 1) {
            budget = std::max(1, budget / 2);
            logf(FLR_LOG_DEBUG, "KV 池满：批大小减到 %d 重试", budget);
            return true;
        }
        std::vector<SequenceStatus> running;
        for (const SequenceStatus& status : statuses()) {
            if (status.active)
                running.push_back(status);
        }
        const int victim = chooseVictim(running);
        if (victim < 0)
            return false;
        for (const std::shared_ptr<Request>& request : active) {
            if (request->seq != victim || request->phase == Request::Phase::Done)
                continue;
            logf(FLR_LOG_WARN, "KV 池满：序列 %d 的请求失败（%lld 个 token），其余继续", victim, (long long) caches[size_t(victim)].nTokens());
            clearSequence(victim);
            fail(*request, FLR_KV_FULL, "本机模型的上下文缓存（几个请求共用的 KV 池）被同时处理的请求占满了");
        }
        budget = config.nBatch;
        return true;
    }

    /// 图片块放不下、又没有空闲序列可腾的请求：这一步的批已经算完，选一个在途请求失败——
    /// 选中别的请求（非主会话里 token 最多的）就让它失败、清掉它的 KV，这张图下一步重试；选中的是它自己才 FLR_KV_FULL
    void resolveMediaRoom()
    {
        for (const std::shared_ptr<Request>& request : active) {
            if (request->phase == Request::Phase::Done || !request->waitingForRoom)
                continue;
            request->waitingForRoom = false;
            // 这一步里结束的请求把序列空了出来
            if (evictIdle("图片", request->seq))
                continue;
            std::vector<SequenceStatus> running;
            for (const SequenceStatus& status : statuses()) {
                if (status.active)
                    running.push_back(status);
            }
            const int victim = chooseVictim(running);
            if (victim < 0 || victim == request->seq) {
                fail(*request, FLR_KV_FULL, "KV 池放不下这张图");
                continue;
            }
            for (const std::shared_ptr<Request>& other : active) {
                if (other->seq != victim || other->phase == Request::Phase::Done)
                    continue;
                logf(FLR_LOG_WARN, "KV 池满（图片）：序列 %d 的请求失败（%lld 个 token），序列 %d 的图片下一步重试", victim,
                     (long long) caches[size_t(victim)].nTokens(), request->seq);
                clearSequence(victim);
                fail(*other, FLR_KV_FULL, "本机模型的上下文缓存（几个请求共用的 KV 池）被同时处理的请求占满了");
            }
        }
    }

    /// llama_process 失败之后：账本截到 KV 里实际有的，预填进度跟着账本走
    void rollbackBatch()
    {
        for (Request* request : inBatch) {
            if (request->phase == Request::Phase::Done)
                continue;
            SequenceCache& cache = caches[size_t(request->seq)];
            cache.rollback(*config.memory, request->seq);
            if (request->phase == Request::Phase::Prefill)
                request->nDone = std::min(int(cache.ledger().size()), int(request->chat.prepared.items.size()));
            request->iBatch = -1;
            request->added = 0;
            request->stopHere.reset();
        }
    }

    /// 不可回退的模型上 llama_process 中止 / 出错之后：KV 里这一步之后的内容不可信，涉及的序列退回这一步之前的检查点
    void discardBatch()
    {
        for (Request* request : inBatch) {
            if (request->phase == Request::Phase::Done)
                continue;
            SequenceCache& cache = caches[size_t(request->seq)];
            cache.discardAfter(*config.memory, request->seq, request->ledgerBefore);
            if (request->phase == Request::Phase::Prefill)
                request->nDone = std::min(int(cache.ledger().size()), int(request->chat.prepared.items.size()));
            request->iBatch = -1;
            request->added = 0;
            request->stopHere.reset();
        }
    }

    // MARK: 结束

    StreamEvent failedEvent(const Request& request, int status, const std::string& message) const
    {
        StreamEvent event;
        event.kind = StreamEvent::Kind::Failed;
        event.status = status;
        event.message = message;
        event.stats = request.stats;
        event.seq = request.seq;
        return event;
    }

    void release(Request& request)
    {
        request.phase = Request::Phase::Done;
        if (request.seq >= 0) {
            busy[size_t(request.seq)] = 0;
            lastUsed[size_t(request.seq)] = ggml_time_us();
        }
    }

    void finish(Request& request)
    {
        const RequestStats& stats = request.stats;
        logf(FLR_LOG_NOTICE,
             "序列 %d：提示 %lld = 复用 %llu + 预填 %llu（%.0f ms，%.0f tok/s）| 生成 %llu（%.0f ms，%.1f tok/s；每步计算 %.2f + 采样 %.2f ms）| %s | "
             "准备 %.1f + 采样器 %.1f + 排队 %.1f ms | 停止：%s",
             request.seq, (long long) request.chat.promptTokens, (unsigned long long) stats.promptCached,
             (unsigned long long) stats.promptProcessed, stats.promptMs(), stats.promptPerSecond(), (unsigned long long) stats.generated,
             stats.genMs(), stats.genPerSecond(), stats.genSteps() > 0 ? double(request.decodeUs) / 1000.0 / double(stats.genSteps()) : 0.0,
             stats.generated > 0 ? double(request.sampleUs) / 1000.0 / double(stats.generated) : 0.0, request.reuse.c_str(),
             request.chat.prepareMs, request.chat.samplerMs, double(stats.tStart - request.tSubmit) / 1000.0,
             stopName(request.stop, request.truncated));
        ++requestsDone;
        tokensReused += stats.promptCached;
        tokensPrefilled += stats.promptProcessed;
        StreamEvent event;
        event.kind = StreamEvent::Kind::Final;
        event.stop = request.stop;
        event.stats = request.stats;
        event.seq = request.seq;
        event.reuse = request.reuse;
        release(request);
        request.push(std::move(event));
        updateStats();
    }

    void fail(Request& request, int status, const std::string& message)
    {
        if (request.phase == Request::Phase::Done)
            return;
        if (status != FLR_CANCELLED)
            logf(status == FLR_NOT_READY ? FLR_LOG_INFO : FLR_LOG_WARN, "序列 %d 的请求结束（%d）：%s", request.seq, status, message.c_str());
        StreamEvent event = failedEvent(request, status, message);
        release(request);
        request.push(std::move(event));
    }

    /// 撤下已取消的请求（还没进这一步的批）
    void reapCancelled()
    {
        for (const std::shared_ptr<Request>& request : active) {
            if (request->phase != Request::Phase::Done && (request->cancelled.load() || stopping.load()))
                fail(*request, stopping ? FLR_NOT_READY : FLR_CANCELLED, stopping ? "正在卸载模型" : "已取消");
        }
        compact();
    }

    void compact()
    {
        active.erase(std::remove_if(active.begin(), active.end(),
                                    [](const std::shared_ptr<Request>& request) { return request->phase == Request::Phase::Done; }),
                     active.end());
        running.store(int(active.size()));
    }

    /// 推理线程：统计快照
    void updateStats()
    {
        syncTree();
        const RadixTree::Stats tree = this->tree.stats();
        int used = 0;
        for (int seq = 0; seq < config.nSeq; ++seq)
            used += caches[size_t(seq)].ledger().empty() ? 0 : 1;
        size_t queued = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            queued = pending.size();
        }
        json snapshot {
            { "parallel", config.nParallel },
            { "sequences", config.nSeq },
            { "sequences_used", used },
            { "active", int(active.size()) },
            { "queued", queued },
            { "requests", requestsDone },
            { "forks", forks },
            { "dedupe_waits", dedupeWaits },
            { "evictions", evictions },
            { "tokens_reused", tokensReused },
            { "tokens_shared", tokensShared },
            { "tokens_prefilled", tokensPrefilled },
            { "tree", json { { "nodes", tree.nodes }, { "items", tree.items }, { "logical", tree.logical } } },
        };
        std::lock_guard<std::mutex> lock(statsMutex);
        statsSnapshot = std::move(snapshot);
    }

    /// 线程退出前：在途与排队的请求以 FLR_NOT_READY 结束，排队的活不执行
    void shutdown()
    {
        for (const std::shared_ptr<Request>& request : active)
            fail(*request, FLR_NOT_READY, "正在卸载模型");
        active.clear();
        running.store(0);
        std::deque<Pending> left;
        {
            std::lock_guard<std::mutex> lock(mutex);
            left.swap(pending);
        }
        for (Pending& item : left) {
            if (item.request)
                item.request->push(failedEvent(*item.request, FLR_NOT_READY, "正在卸载模型"));
            if (item.job)
                item.job->result.set_value(FLR_NOT_READY);
        }
    }
};

Scheduler::Scheduler(Config config)
    : d(std::make_unique<Impl>(std::move(config)))
{
}

Scheduler::~Scheduler()
{
    stop();
}

void Scheduler::start()
{
    llama_set_abort_callback(d->config.ctx, &Impl::abortCallback, d.get());
    d->thread = std::thread([this] { d->run(); });
}

void Scheduler::stop()
{
    {
        std::lock_guard<std::mutex> lock(d->mutex);
        if (d->stopped)
            return;
        d->stopped = true;
        d->stopRequested = true;
    }
    d->stopping.store(true);
    d->wake.notify_all();
    if (d->thread.joinable()) {
        d->thread.join();
    } else {
        // 线程没起来过：排队的直接失败
        d->shutdown();
    }
    llama_set_abort_callback(d->config.ctx, nullptr, nullptr);
}

json Scheduler::stats() const
{
    std::lock_guard<std::mutex> lock(d->statsMutex);
    return d->statsSnapshot;
}

bool Scheduler::busy() const
{
    if (d->running.load() > 0)
        return true;
    std::lock_guard<std::mutex> lock(d->mutex);
    for (const Pending& item : d->pending) {
        if (item.request)
            return true;
    }
    return false;
}

int Scheduler::chat(ChatRequest request, Sampler sampler, flr_chunk_fn chunk, flr_should_stop_fn shouldStop, void* user, json& result)
{
    auto pending = std::make_shared<Request>();
    pending->serial = d->nextSerial++;
    pending->chat = std::move(request);
    pending->sampler.emplace(std::move(sampler));
    pending->tSubmit = ggml_time_us();
    // 推理线程用不到解析器：整个交给这边的 ChatStream
    ChatStream stream(std::move(pending->chat.chatSession), "chatcmpl-" + ChatStream::randomId(), d->config.modelName);
    const bool streaming = pending->chat.stream;
    {
        std::lock_guard<std::mutex> lock(d->mutex);
        if (d->stopRequested) {
            result = errorResult(-1, "正在卸载模型");
            return FLR_NOT_READY;
        }
        d->pending.push_back({ pending, nullptr });
    }
    d->wake.notify_one();

    const auto cancel = [this, &pending] {
        pending->cancelled.store(true);
        {
            std::lock_guard<std::mutex> lock(d->mutex);
        }
        d->wake.notify_one();
    };

    bool userStopped = false;
    bool done = false;
    int status = FLR_OK;
    int usedSeq = -1;
    std::string message;
    std::string parseError;
    std::vector<std::string> chunks;
    const auto deliver = [&]() {
        if (streaming) {
            for (const std::string& text : chunks) {
                if (chunk && !chunk(text.c_str(), user)) {
                    userStopped = true;
                    cancel();
                    break;
                }
            }
        }
        chunks.clear();
    };
    while (!done) {
        std::deque<StreamEvent> events;
        {
            std::unique_lock<std::mutex> lock(pending->mutex);
            pending->cv.wait_for(lock, std::chrono::milliseconds(10), [&] { return !pending->events.empty(); });
            events.swap(pending->events);
        }
        for (StreamEvent& event : events) {
            if (event.seq >= 0)
                usedSeq = event.seq;
            if (event.kind == StreamEvent::Kind::Partial) {
                if (userStopped || !parseError.empty())
                    continue;
                try {
                    stream.partial(event.text, event.nDecoded, event.withStats ? &event.stats : nullptr, chunks);
                } catch (const std::exception& exception) {
                    parseError = exception.what();
                    chunks.clear();
                    cancel();
                    continue;
                }
                deliver();
            } else if (event.kind == StreamEvent::Kind::Final) {
                done = true;
                if (userStopped || !parseError.empty())
                    continue;
                std::string reason;
                try {
                    reason = stream.finish(event.stop, pending->chat.promptTokens, event.stats, chunks);
                } catch (const std::exception& exception) {
                    parseError = exception.what();
                    chunks.clear();
                    continue;
                }
                deliver();
                const json cache { { "seq", usedSeq }, { "how", event.reuse }, { "reused", event.stats.promptCached },
                                   { "prefilled", event.stats.promptProcessed } };
                if (streaming) {
                    result = json {
                        { "finish_reason", reason },
                        { "usage", ChatStream::usage(pending->chat.promptTokens, event.stats) },
                        { "timings", event.stats.toJson() },
                    };
                } else {
                    result = stream.completion(reason, pending->chat.promptTokens, event.stats);
                }
                result["x_friday"] = json { { "cache", cache } };
            } else {
                done = true;
                status = event.status;
                message = event.message;
            }
        }
        if (!done && !userStopped && parseError.empty() && shouldStop && shouldStop(user)) {
            userStopped = true;
            cancel();
        }
    }
    if (userStopped) {
        result = errorResult(usedSeq, "已取消");
        return FLR_CANCELLED;
    }
    if (!parseError.empty()) {
        result = errorResult(usedSeq, "解析模型输出出错：" + parseError);
        return FLR_INTERNAL;
    }
    if (status != FLR_OK) {
        result = errorResult(usedSeq, message);
        return status;
    }
    return FLR_OK;
}

bool Scheduler::evictIdleFromJob(int keep)
{
    return d->evictIdle("恢复序列状态", keep);
}

int Scheduler::withSequence(int seq, const std::function<int(SequenceCache&, MemoryOps&)>& job)
{
    if (seq < 0 || seq >= d->config.nSeqMax)
        return FLR_INVALID_REQUEST;
    auto pending = std::make_shared<Job>();
    pending->seq = seq;
    pending->fn = job;
    std::future<int> result = pending->result.get_future();
    {
        std::lock_guard<std::mutex> lock(d->mutex);
        if (d->stopRequested)
            return FLR_NOT_READY;
        d->pending.push_back({ nullptr, pending });
    }
    d->wake.notify_one();
    return result.get();
}

} // namespace flr
