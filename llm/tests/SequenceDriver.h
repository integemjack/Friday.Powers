// WP3 单测的小工具（不需要模型）：照 internal.h 开头写的「推理线程上一个请求的生命周期」驱动 SequenceCache + FakeMemory，
// 以及一个假的 SequenceHost（每个序列一个 SequenceCache，共用一个 FakeMemory，withSequence 直接在本线程执行）。
#pragma once

#include "FakeMemory.h"
#include "internal.h"

#include <string>
#include <utility>
#include <vector>

namespace seqtest {

using flr::Checkpoint;
using flr::LedgerItem;
using flr::Prepared;
using flr::SequenceCache;

/// 文本 token first, first+1, …（count 个）
inline std::vector<LedgerItem> range(llama_token first, int count)
{
    std::vector<LedgerItem> items;
    for (int i = 0; i < count; ++i)
        items.push_back(LedgerItem::text(first + i));
    return items;
}

inline std::vector<LedgerItem> join(std::initializer_list<std::vector<LedgerItem>> parts)
{
    std::vector<LedgerItem> items;
    for (const auto& part : parts)
        items.insert(items.end(), part.begin(), part.end());
    return items;
}

inline LedgerItem media(const std::string& id, int nTokens, int nPos)
{
    LedgerItem item;
    item.media_id = id;
    item.n_tokens = nTokens;
    item.n_pos = nPos;
    return item;
}

/// 一次请求：P、B（B 只在用户驱动时有意义）；E 由 SequenceCache 自己按末尾 − 4 算
inline Prepared request(std::vector<LedgerItem> items, int prefix, int boundary, bool userDriven)
{
    Prepared p;
    p.items = std::move(items);
    p.prefix_boundary = prefix;
    p.transcript_boundary = boundary;
    p.user_driven = userDriven;
    if (prefix > 0)
        p.prefix_fingerprint = flr::prefixFingerprint(p.items, prefix);
    return p;
}

/// 一次请求跑下来的情况
struct Run {
    SequenceCache::Plan plan;
    /// 实际预填的项数（== items.size() − keep）
    int prefilled = 0;
    /// 停下取过检查点的位置
    std::vector<int> stops;
    /// 预填用了几次 decode
    int decodes = 0;
};

/// 照生命周期跑一个请求：prepare → 按 nextStop 分批预填（每批 ≤ nBatch 项，媒体块单独一批）→ 停点处 takeCheckpoint →
/// 生成 generated（最后一个只采样、不进 KV）。FakeMemory::decode 按项的 n_pos 推进位置
inline Run run(SequenceCache& cache, FakeMemory& memory, int seq, const Prepared& p, const std::vector<llama_token>& generated = {},
               int nBatch = 512)
{
    Run result;
    result.plan = cache.prepare(p, memory, seq);
    int done = result.plan.keep;
    result.prefilled = int(p.items.size()) - done;
    while (done < int(p.items.size())) {
        const auto stop = cache.nextStop(done);
        const int limit = stop ? stop->at : int(p.items.size());
        int positions = 0;
        int added = 0;
        if (p.items[size_t(done)].isMedia()) {
            positions = p.items[size_t(done)].n_pos;
            cache.append(p.items[size_t(done)]);
            ++done;
            ++added;
        } else {
            while (done < limit && added < nBatch && !p.items[size_t(done)].isMedia()) {
                cache.append(p.items[size_t(done)]);
                positions += p.items[size_t(done)].n_pos;
                ++done;
                ++added;
            }
        }
        memory.decode(seq, positions);
        ++result.decodes;
        if (stop && done == stop->at) {
            result.stops.push_back(stop->at);
            for (const Checkpoint::Kind kind : stop->kinds)
                cache.takeCheckpoint(kind, memory, seq);
        }
    }
    for (size_t i = 0; i + 1 < generated.size(); ++i) {
        cache.append(LedgerItem::text(generated[i]));
        memory.decode(seq, 1);
    }
    return result;
}

/// 账本里的 token（媒体块记 -1）
inline std::vector<llama_token> tokensOf(const std::vector<LedgerItem>& items)
{
    std::vector<llama_token> tokens;
    for (const LedgerItem& item : items)
        tokens.push_back(item.isMedia() ? -1 : item.token);
    return tokens;
}

/// 序列的检查点位置（n_items，升序）
inline std::vector<int64_t> checkpointItems(const SequenceCache& cache)
{
    std::vector<int64_t> items;
    for (const Checkpoint& checkpoint : cache.checkpoints())
        items.push_back(checkpoint.n_items);
    return items;
}

inline int countKind(const SequenceCache& cache, Checkpoint::Kind kind)
{
    int count = 0;
    for (const Checkpoint& checkpoint : cache.checkpoints())
        count += checkpoint.kind == kind ? 1 : 0;
    return count;
}

inline int countCalls(const FakeMemory& memory, const std::string& prefix)
{
    int count = 0;
    for (const std::string& call : memory.calls)
        count += call.rfind(prefix, 0) == 0 ? 1 : 0;
    return count;
}

/// 假的 SequenceHost：序列 0…n−1 给应用，n 是工作序列
class FakeHost : public flr::SequenceHost {
public:
    FakeHost(bool hybrid, int sequences, flr::CachePolicy policy = {})
        : memory(hybrid)
        , kind(hybrid ? "hybrid" : "attention")
    {
        for (int i = 0; i < sequences; ++i)
            caches.emplace_back(policy);
    }

    int appSequences() const override { return int(caches.size()); }
    int workspaceSequence() const override { return int(caches.size()); }
    const std::string& modelSha() const override { return sha; }
    const std::string& memoryKind() const override { return kind; }
    int withSequence(int seq, const std::function<int(SequenceCache&, flr::MemoryOps&)>& job) override
    {
        ++jobs;
        if (unloading)
            return FLR_NOT_READY;
        return job(caches[size_t(seq)], memory);
    }
    /// 还有 idleSequences 个空闲序列可腾：腾一个（记下 keep），之后 restoreFull 少失败一次（评审修正的用例）
    bool evictIdleSequence(int keep) override
    {
        evictedFor.push_back(keep);
        if (idleSequences <= 0)
            return false;
        --idleSequences;
        return true;
    }

    FakeMemory memory;
    std::vector<SequenceCache> caches;
    std::string sha = "0123abcd-4096";
    std::string kind;
    bool unloading = false;
    int jobs = 0;
    /// evictIdleSequence：还能腾几个空闲序列、每次是给哪个序列腾的
    int idleSequences = 0;
    std::vector<int> evictedFor;
};

} // namespace seqtest
