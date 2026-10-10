// WP3 的用例：SequenceCache 的复用决策表、检查点（P / B / E）与上限、停点、回退、前缀、池满规则（不需要模型）。
// 用一个仿 Qwen3.5 模板的小对话（docs/LOCAL_INFERENCE.md §F.2）：系统提示 + 工具 60 项（P = 60），用户消息带 3 项消息头，
// 生成提示 5 项（<|im_start|> assistant \n <think> \n）；下一轮模板去掉上一轮的思考 → 在上一段回复开头分歧。
// 每个场景 PART 型（普通 Transformer）与混合模型（Qwen3.5）各跑一遍，并核对 KV（FakeMemory）与账本一致。
#include "SequenceDriver.h"
#include "rt_test.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace seqtest;
using flr::Checkpoint;
using flr::LedgerItem;
using flr::Prepared;
using flr::SequenceCache;

namespace {

const std::vector<LedgerItem> kSystem = range(1000, 60);                     // 系统提示 + 工具说明（P = 60）
const std::vector<LedgerItem> kUserHead = { LedgerItem::text(10), LedgerItem::text(11), LedgerItem::text(12) };
const std::vector<LedgerItem> kEnd = { LedgerItem::text(13), LedgerItem::text(14) };
const std::vector<LedgerItem> kAssistantHead = { LedgerItem::text(20), LedgerItem::text(21), LedgerItem::text(22) };
const std::vector<LedgerItem> kGenerationPrompt = join({ kAssistantHead, { LedgerItem::text(23), LedgerItem::text(24) } });

std::vector<LedgerItem> user(llama_token first, int count)
{
    return join({ kUserHead, range(first, count), kEnd });
}

/// 用户驱动的一轮：B = 生成提示之前
Prepared userTurn(const std::vector<LedgerItem>& history, const std::vector<LedgerItem>& userMessage)
{
    const std::vector<LedgerItem> transcript = join({ history, userMessage });
    return request(join({ transcript, kGenerationPrompt }), int(kSystem.size()), int(transcript.size()), true);
}

/// 第一轮的生成：思考 10 + 「\n</think>\n\n」2 + 回答 20 + EOS
std::vector<llama_token> firstReply()
{
    std::vector<llama_token> tokens;
    for (int i = 0; i < 10; ++i)
        tokens.push_back(3000 + i);
    tokens.push_back(25);
    tokens.push_back(26);
    for (int i = 0; i < 20; ++i)
        tokens.push_back(4000 + i);
    tokens.push_back(27);
    return tokens;
}

/// KV 与账本一致：KV 里正好是账本的那些位置；混合模型的循环状态停在最后一个位置
void checkConsistent(const SequenceCache& cache, const FakeMemory& memory, int seq, bool hybrid)
{
    RT_CHECK_EQ(memory.seqPosMax(seq) + 1, cache.nextPos());
    RT_CHECK_EQ(int64_t(memory.cellCount(seq)), int64_t(cache.nextPos()));
    if (hybrid)
        RT_CHECK_EQ(memory.recurrent(seq), cache.nextPos() - 1);
}

void checkLedgerPrefix(const SequenceCache& cache, const Prepared& p, int keep)
{
    RT_CHECK_EQ(cache.ledger().size(), size_t(keep));
    for (int i = 0; i < keep; ++i)
        RT_CHECK(cache.ledger()[size_t(i)] == p.items[size_t(i)]);
}

/// 第一轮跑完（带生成）的一条序列
struct FirstTurn {
    Prepared request;
    Run run;
};

FirstTurn firstTurn(SequenceCache& cache, FakeMemory& memory, int seq)
{
    FirstTurn turn;
    turn.request = userTurn(kSystem, user(2000, 20));
    turn.run = run(cache, memory, seq, turn.request, firstReply());
    return turn;
}

} // namespace

// MARK: - 复用决策表（PART / 混合 × 延续 / 中途分歧 / 相同 / 系统提示变化 / 同轮重分词 / 折叠）

RT_TEST(seqcache_first_turn_stops_at_p_b_e)
{
    for (const bool hybrid : { false, true }) {
        FakeMemory memory(hybrid);
        SequenceCache cache;
        const FirstTurn turn = firstTurn(cache, memory, 0);
        RT_CHECK_EQ(turn.run.plan.keep, 0);
        RT_CHECK(!turn.run.plan.reset);
        RT_CHECK_EQ(turn.request.items.size(), size_t(90));
        if (hybrid) {
            // P = 60、B = 85（生成提示之前）、E = 90 − 4
            RT_CHECK(turn.run.stops == std::vector<int>({ 60, 85, 86 }));
            RT_CHECK(checkpointItems(cache) == std::vector<int64_t>({ 60, 85, 86 }));
            RT_CHECK_EQ(countCalls(memory, "partialState"), 3);
        } else {
            // PART 型不停、不取状态，只记 P、B 的位置
            RT_CHECK(turn.run.stops.empty());
            RT_CHECK(checkpointItems(cache) == std::vector<int64_t>({ 60, 85 }));
            RT_CHECK_EQ(countCalls(memory, "partialState"), 0);
            for (const Checkpoint& checkpoint : cache.checkpoints())
                RT_CHECK(!checkpoint.hasData());
        }
        RT_CHECK_EQ(cache.ledger().size(), size_t(90 + 32));   // 提示 + 生成（最后一个没进 KV）
        checkConsistent(cache, memory, 0, hybrid);
        RT_CHECK(cache.prefixFingerprint() == flr::prefixFingerprint(turn.request.items, 60));
    }
}

RT_TEST(seqcache_end_checkpoint_needs_a_gap)
{
    // CachePolicy::min_end_gap（运行时用 256）：E 离前一个检查点 / 停点太近就不设
    FakeMemory memory(true);
    flr::CachePolicy policy;
    policy.min_end_gap = 256;
    SequenceCache cache(policy);
    // 用户消息驱动的第一轮：E（86）就在 B（85）后面 → 只停 P、B
    const FirstTurn turn = firstTurn(cache, memory, 0);
    RT_CHECK(turn.run.stops == std::vector<int>({ 60, 85 }));
    RT_CHECK(checkpointItems(cache) == std::vector<int64_t>({ 60, 85 }));
    // 工具结果续写、新预填的少（离 B 不到 256 项）：不设 E
    std::vector<LedgerItem> history = cache.ledger();
    history.push_back(LedgerItem::text(27));
    const std::vector<LedgerItem> shortResult = join({ history, range(5000, 100), kGenerationPrompt });
    const Run small = run(cache, memory, 0, request(shortResult, 60, -1, false), { 6000, 6001, 6002 });
    RT_CHECK(small.stops.empty());
    RT_CHECK(checkpointItems(cache) == std::vector<int64_t>({ 60, 85 }));
    // 新预填的多（大段工具结果）：设 E
    history = cache.ledger();
    history.push_back(LedgerItem::text(6003));
    const std::vector<LedgerItem> longResult = join({ history, range(7000, 600), kGenerationPrompt });
    const Prepared longRequest = request(longResult, 60, -1, false);
    const Run large = run(cache, memory, 0, longRequest);
    RT_CHECK(large.stops == std::vector<int>({ int(longRequest.items.size()) - 4 }));
    RT_CHECK_EQ(cache.checkpoints().back().n_items, int64_t(longRequest.items.size()) - 4);
    checkConsistent(cache, memory, 0, true);
    // 缺省（min_end_gap = 0）：每个请求都设 E（WP3 / server 的规则，见 seqcache_first_turn_stops_at_p_b_e）
}

RT_TEST(seqcache_continue_appends_without_touching_kv)
{
    for (const bool hybrid : { false, true }) {
        FakeMemory memory(hybrid);
        SequenceCache cache;
        firstTurn(cache, memory, 0);
        // 模板把生成的内容原样渲染出来（思考关、无工具时）：账本是新提示的前缀 → 直接接着算
        std::vector<LedgerItem> history = cache.ledger();
        history.push_back(LedgerItem::text(27));
        history.insert(history.end(), kEnd.begin(), kEnd.end());
        const Prepared next = userTurn(history, user(2100, 8));
        memory.calls.clear();
        const Run second = run(cache, memory, 0, next);
        RT_CHECK_EQ(second.plan.keep, 122);
        RT_CHECK_EQ(second.plan.restored, -1);
        RT_CHECK(!second.plan.reset);
        RT_CHECK_EQ(second.prefilled, int(next.items.size()) - 122);
        RT_CHECK_EQ(countCalls(memory, "seqRemove") + countCalls(memory, "restorePartial") + countCalls(memory, "seqClear"), 0);
        checkConsistent(cache, memory, 0, hybrid);
    }
}

RT_TEST(seqcache_second_turn_diverges_in_last_reply)
{
    for (const bool hybrid : { false, true }) {
        FakeMemory memory(hybrid);
        SequenceCache cache;
        firstTurn(cache, memory, 0);
        // 第二轮：模板去掉上一轮的思考，在回复开头（生成提示的 <think> 处，第 88 项）分歧
        const std::vector<LedgerItem> history = join({ kSystem, user(2000, 20), kAssistantHead, range(4000, 20), kEnd });
        const Prepared second = userTurn(history, user(2100, 10));
        memory.calls.clear();
        const SequenceCache::Plan plan = cache.prepare(second, memory, 0);
        if (hybrid) {
            // 回退到 E（提示末尾 − 4 = 86 ≤ 公共前缀 88）：恢复循环状态，再删注意力 KV 的尾巴
            RT_CHECK_EQ(plan.keep, 86);
            RT_CHECK_EQ(plan.restored, 86);
            RT_CHECK(memory.calls == std::vector<std::string>({ "restorePartial 0", "seqRemove 0 86 -1" }));
            // 86 之后的检查点作废，P、B 留着
            RT_CHECK(checkpointItems(cache) == std::vector<int64_t>({ 60, 85, 86 }));
        } else {
            // PART 型：截到公共前缀
            RT_CHECK_EQ(plan.keep, 88);
            RT_CHECK_EQ(plan.restored, -1);
            RT_CHECK(memory.calls == std::vector<std::string>({ "seqRemove 0 88 -1" }));
        }
        RT_CHECK(!plan.reset);
        checkLedgerPrefix(cache, second, plan.keep);
        checkConsistent(cache, memory, 0, hybrid);
        // 这一轮的停点：B、E（P 已有）
        if (hybrid) {
            const auto stop = cache.nextStop(plan.keep);
            RT_CHECK(stop.has_value());
            RT_CHECK_EQ(stop->at, int(second.transcript_boundary));
            RT_CHECK(stop->kinds == std::vector<Checkpoint::Kind>({ Checkpoint::Kind::Boundary }));
        } else {
            RT_CHECK(!cache.nextStop(plan.keep).has_value());
        }
    }
}

RT_TEST(seqcache_same_turn_retokenization_uses_end_checkpoint)
{
    for (const bool hybrid : { false, true }) {
        FakeMemory memory(hybrid);
        SequenceCache cache;
        const FirstTurn turn = firstTurn(cache, memory, 0);
        // 工具结果续写：生成的思考在第 5 个 token 处被重新分词成两个不同的 token（采样出来的 token 与同一段文字重新分词的结果不同）
        const std::vector<LedgerItem> continuation = join({ turn.request.items, range(3000, 5), range(31, 2), range(3006, 4),
                                                            range(25, 2), range(5000, 8), kEnd, join({ kUserHead, range(40, 1) }),
                                                            range(6000, 10), kEnd, kGenerationPrompt });
        const Prepared next = request(continuation, int(kSystem.size()), -1, false);
        memory.calls.clear();
        const SequenceCache::Plan plan = cache.prepare(next, memory, 0);
        if (hybrid) {
            // 公共前缀停在上一段输出里（95），B（85）之后还有 E（86）：只多算 9 项
            RT_CHECK_EQ(plan.keep, 86);
            RT_CHECK_EQ(plan.restored, 86);
        } else {
            RT_CHECK_EQ(plan.keep, 95);
        }
        checkLedgerPrefix(cache, next, plan.keep);
        checkConsistent(cache, memory, 0, hybrid);
        // 工具续写不是用户驱动：不设 B，只有 E
        if (hybrid) {
            const auto stop = cache.nextStop(plan.keep);
            RT_CHECK(stop.has_value());
            RT_CHECK_EQ(stop->at, int(next.items.size()) - 4);
            RT_CHECK(stop->kinds == std::vector<Checkpoint::Kind>({ Checkpoint::Kind::End }));
            RT_CHECK(!cache.nextStop(stop->at).has_value());
        }
    }
}

RT_TEST(seqcache_identical_request_recomputes_the_tail)
{
    for (const bool hybrid : { false, true }) {
        FakeMemory memory(hybrid);
        SequenceCache cache;
        const FirstTurn turn = firstTurn(cache, memory, 0);
        // 同一个请求再来一次（重试 / 重新生成）：至少要算 1 项才有 logits
        const Run again = run(cache, memory, 0, turn.request, firstReply());
        RT_CHECK_EQ(again.plan.keep, hybrid ? 86 : 89);
        RT_CHECK_EQ(again.prefilled, hybrid ? 4 : 1);
        checkConsistent(cache, memory, 0, hybrid);
        // 只预填完、没有生成（账本 == 提示）时也一样
        FakeMemory memory2(hybrid);
        SequenceCache cache2;
        run(cache2, memory2, 1, turn.request);
        const SequenceCache::Plan plan = cache2.prepare(turn.request, memory2, 1);
        RT_CHECK_EQ(plan.keep, hybrid ? 86 : 89);
        checkConsistent(cache2, memory2, 1, hybrid);
    }
}

RT_TEST(seqcache_system_prompt_change)
{
    for (const bool hybrid : { false, true }) {
        FakeMemory memory(hybrid);
        SequenceCache cache;
        const FirstTurn turn = firstTurn(cache, memory, 0);
        const std::string oldPrefix = *cache.prefixFingerprint();
        // 系统提示第 30 项起变了（例如工具列表变了）：P 失效
        std::vector<LedgerItem> system = kSystem;
        system[30] = LedgerItem::text(9999);
        const Prepared next = request(join({ system, user(2000, 20), kGenerationPrompt }), int(system.size()), 85, true);
        memory.calls.clear();
        const Run result = run(cache, memory, 0, next);
        if (hybrid) {
            // 公共前缀 30 之前没有检查点：清空重算，并在新的 P、B、E 处取检查点
            RT_CHECK(result.plan.reset);
            RT_CHECK_EQ(result.plan.keep, 0);
            RT_CHECK_EQ(memory.calls.front(), std::string("seqClear 0"));
            RT_CHECK(result.stops == std::vector<int>({ 60, 85, 86 }));
        } else {
            RT_CHECK(!result.plan.reset);
            RT_CHECK_EQ(result.plan.keep, 30);
        }
        const auto prefix = cache.prefixFingerprint();
        RT_CHECK(prefix.has_value());
        RT_CHECK(*prefix != oldPrefix);
        RT_CHECK(*prefix == flr::prefixFingerprint(next.items, 60));
        RT_CHECK_EQ(countKind(cache, Checkpoint::Kind::Prefix), 1);
        checkConsistent(cache, memory, 0, hybrid);
        (void)turn;
    }
}

RT_TEST(seqcache_fold_restores_prefix_checkpoint)
{
    for (const bool hybrid : { false, true }) {
        FakeMemory memory(hybrid);
        SequenceCache cache;
        firstTurn(cache, memory, 0);
        const std::vector<LedgerItem> history2 = join({ kSystem, user(2000, 20), kAssistantHead, range(4000, 20), kEnd });
        run(cache, memory, 0, userTurn(history2, user(2100, 10)), { 7000, 7001, 7002, 7003, 27 });
        // 折叠：去掉第一轮，第二轮的用户消息紧跟系统提示（用户消息头 3 项还相同）
        const Prepared folded = userTurn(kSystem, user(2100, 10));
        memory.calls.clear();
        const SequenceCache::Plan plan = cache.prepare(folded, memory, 0);
        if (hybrid) {
            RT_CHECK_EQ(plan.keep, 60);
            RT_CHECK_EQ(plan.restored, 60);
            RT_CHECK(checkpointItems(cache) == std::vector<int64_t>({ 60 }));
        } else {
            RT_CHECK_EQ(plan.keep, 63);
        }
        checkLedgerPrefix(cache, folded, plan.keep);
        checkConsistent(cache, memory, 0, hybrid);
        RT_CHECK(cache.prefixFingerprint().has_value());
    }
}

RT_TEST(seqcache_cancel_then_resend_continues)
{
    for (const bool hybrid : { false, true }) {
        FakeMemory memory(hybrid);
        SequenceCache cache;
        const Prepared first = userTurn(kSystem, user(2000, 20));
        // 预填到 40 项时取消（已算的留在 KV 里）
        cache.prepare(first, memory, 0);
        for (int i = 0; i < 40; ++i)
            cache.append(first.items[size_t(i)]);
        memory.decode(0, 40);
        // 重发同一个请求：接着算，停点照旧
        const Run again = run(cache, memory, 0, first);
        RT_CHECK_EQ(again.plan.keep, 40);
        RT_CHECK(!again.plan.reset);
        RT_CHECK(again.stops == (hybrid ? std::vector<int>({ 60, 85, 86 }) : std::vector<int>()));
        checkConsistent(cache, memory, 0, hybrid);
    }
}

// MARK: - 检查点的上限与身份

RT_TEST(seqcache_checkpoint_limits)
{
    FakeMemory memory(true);
    SequenceCache cache;
    std::vector<LedgerItem> history = kSystem;
    std::vector<int64_t> boundaries;
    for (int turn = 0; turn < 7; ++turn) {
        const Prepared p = userTurn(history, user(2000 + turn * 100, 12));
        boundaries.push_back(p.transcript_boundary);
        run(cache, memory, 0, p, { 8000 + turn, 27 });
        RT_CHECK(cache.checkpoints().size() <= size_t(1 + 3 + 2));
        RT_CHECK_EQ(countKind(cache, Checkpoint::Kind::Prefix), 1);
        RT_CHECK(countKind(cache, Checkpoint::Kind::Boundary) <= 3);
        RT_CHECK(countKind(cache, Checkpoint::Kind::End) <= 2);
        // 模板原样渲染上一段回复：下一轮接着算
        history = join({ cache.ledger(), { LedgerItem::text(27) }, kEnd });
    }
    // 留下的是 P + 最近 3 个 B + 最近 2 个 E
    std::vector<int64_t> expected = { 60 };
    for (size_t i = boundaries.size() - 3; i < boundaries.size(); ++i)
        expected.push_back(boundaries[i]);
    expected.push_back(boundaries[boundaries.size() - 2] + 1);
    expected.push_back(boundaries.back() + 1);
    std::sort(expected.begin(), expected.end());
    RT_CHECK(checkpointItems(cache) == expected);
    for (const Checkpoint& checkpoint : cache.checkpoints())
        RT_CHECK(checkpoint.hasData());

    // 上限可配
    flr::CachePolicy policy;
    policy.max_boundary = 1;
    policy.max_end = 1;
    FakeMemory memory2(true);
    SequenceCache small(policy);
    run(small, memory2, 0, userTurn(kSystem, user(2000, 12)), { 8000, 27 });
    run(small, memory2, 0, userTurn(join({ small.ledger(), { LedgerItem::text(27) }, kEnd }), user(2100, 12)), { 8001, 27 });
    RT_CHECK_EQ(small.checkpoints().size(), size_t(3));
}

RT_TEST(seqcache_coinciding_boundary_and_end_share_one_state)
{
    // 生成提示只有 4 项时 B == E（提示末尾 − 4）：同一个位置只取一次状态，两个身份
    FakeMemory memory(true);
    SequenceCache cache;
    const std::vector<LedgerItem> generationPrompt = range(20, 4);
    const std::vector<LedgerItem> transcript = join({ kSystem, user(2000, 20) });
    const Prepared p = request(join({ transcript, generationPrompt }), 60, int(transcript.size()), true);
    const Run result = run(cache, memory, 0, p);
    RT_CHECK(result.stops == std::vector<int>({ 60, 85 }));
    RT_CHECK_EQ(countCalls(memory, "partialState"), 2);
    RT_CHECK(checkpointItems(cache) == std::vector<int64_t>({ 60, 85 }));
    RT_CHECK(cache.checkpoints()[1].kind == Checkpoint::Kind::Boundary);
    // 再来两个只带 E 的请求（工具续写）：E 身份被挤掉，B 身份还在，所以 85 留着
    std::vector<LedgerItem> history = cache.ledger();
    for (int i = 0; i < 2; ++i) {
        history = join({ history, range(5000 + i * 10, 6), generationPrompt });
        run(cache, memory, 0, request(history, 60, -1, false));
        history = cache.ledger();
    }
    const auto items = checkpointItems(cache);
    RT_CHECK(std::find(items.begin(), items.end(), int64_t(85)) != items.end());
    RT_CHECK_EQ(countKind(cache, Checkpoint::Kind::End), 2);
}

// MARK: - 停点

RT_TEST(seqcache_next_stop_semantics)
{
    FakeMemory memory(true);
    SequenceCache cache;
    const Prepared p = userTurn(kSystem, user(2000, 20));
    cache.prepare(p, memory, 0);
    RT_CHECK_EQ(cache.nextStop(0)->at, 60);
    RT_CHECK(cache.nextStop(0)->kinds == std::vector<Checkpoint::Kind>({ Checkpoint::Kind::Prefix }));
    RT_CHECK_EQ(cache.nextStop(59)->at, 60);
    RT_CHECK_EQ(cache.nextStop(60)->at, 85);
    RT_CHECK_EQ(cache.nextStop(85)->at, 86);
    RT_CHECK(!cache.nextStop(86).has_value());

    // PART 型不停
    FakeMemory part(false);
    SequenceCache partCache;
    partCache.prepare(p, part, 0);
    RT_CHECK(!partCache.nextStop(0).has_value());

    // 不设检查点（rt_bench 对照）：不停；混合模型分歧时只能清空重算
    flr::CachePolicy off;
    off.checkpoints = false;
    FakeMemory memory2(true);
    SequenceCache plain(off);
    run(plain, memory2, 0, p, firstReply());
    RT_CHECK(plain.checkpoints().empty());
    const std::vector<LedgerItem> history = join({ kSystem, user(2000, 20), kAssistantHead, range(4000, 20), kEnd });
    const SequenceCache::Plan plan = plain.prepare(userTurn(history, user(2100, 10)), memory2, 0);
    RT_CHECK(plan.reset);
    RT_CHECK_EQ(plan.keep, 0);
    RT_CHECK(!plain.prefixFingerprint().has_value());
}

RT_TEST(seqcache_no_checkpoint_right_after_media)
{
    // 图片块之后不建检查点（== server）：B 正好在图片之后时不停
    FakeMemory memory(true);
    SequenceCache cache;
    const std::vector<LedgerItem> transcript = join({ kSystem, kUserHead, range(2000, 5), { media("img-1", 260, 20) } });
    const Prepared p = request(join({ transcript, range(20, 8) }), 60, int(transcript.size()), true);
    const Run result = run(cache, memory, 0, p);
    RT_CHECK(result.stops == std::vector<int>({ 60, int(p.items.size()) - 4 }));
    // 位置按 n_pos 记：图片 260 个 token 只占 20 个位置
    RT_CHECK_EQ(cache.nTokens(), int64_t(p.items.size()) - 1 + 260);
    RT_CHECK_EQ(cache.nextPos(), llama_pos(p.items.size()) - 1 + 20);
    const Checkpoint& end = cache.checkpoints().back();
    RT_CHECK_EQ(end.n_items, int64_t(p.items.size()) - 4);
    RT_CHECK_EQ(end.pos, llama_pos(end.n_items) - 1 + 20);
    checkConsistent(cache, memory, 0, true);
}

// MARK: - 失败之后（rollback）

RT_TEST(seqcache_rollback_after_partial_batch)
{
    for (const bool hybrid : { false, true }) {
        {
            FakeMemory memory(hybrid);
            SequenceCache cache;
            const Prepared p = userTurn(kSystem, user(2000, 20));
            cache.prepare(p, memory, 0);
            // 这一批到 P（60 项）为止；llama_process 返回 1（同一批里别的序列的 ubatch 放不下），这个序列的 60 项都算完了
            for (int i = 0; i < 60; ++i)
                cache.append(p.items[size_t(i)]);
            memory.decode(0, 60);
            cache.rollback(memory, 0);
            RT_CHECK_EQ(cache.ledger().size(), size_t(60));
            checkConsistent(cache, memory, 0, hybrid);
            // 正好停在还没取的 P 上：现在取（KV 正好到这里）
            RT_CHECK(cache.prefixFingerprint().has_value());
            // 接着算：停点还在（nextStop 按已处理数往后找）
            RT_CHECK(!hybrid || cache.nextStop(60)->at == 85);
        }
        {
            // 这一批只算完了前 40 项：账本截到 40，没有检查点，重试时还会停在 P
            FakeMemory memory(hybrid);
            SequenceCache cache;
            const Prepared p = userTurn(kSystem, user(2000, 20));
            cache.prepare(p, memory, 0);
            for (int i = 0; i < 60; ++i)
                cache.append(p.items[size_t(i)]);
            memory.decode(0, 40);
            cache.rollback(memory, 0);
            RT_CHECK_EQ(cache.ledger().size(), size_t(40));
            RT_CHECK(cache.checkpoints().empty());
            RT_CHECK(!cache.prefixFingerprint().has_value());
            RT_CHECK(!hybrid || cache.nextStop(40)->at == 60);
            checkConsistent(cache, memory, 0, hybrid);
        }
    }
}

RT_TEST(seqcache_rollback_half_evaluated_media)
{
    // 媒体块只算了一半（KV 比账本多）：PART 删掉多的；混合模型退到最近的检查点，没有就清空
    for (const int variant : { 0, 1, 2 }) {
        const bool hybrid = variant > 0;
        FakeMemory memory(hybrid);
        SequenceCache cache;
        const Prepared p = request(join({ kSystem, kUserHead, { media("img-2", 300, 30) }, range(20, 8) }), variant == 2 ? 60 : -1, -1, false);
        cache.prepare(p, memory, 0);
        for (int i = 0; i < 60; ++i)
            cache.append(p.items[size_t(i)]);
        memory.decode(0, 60);
        if (variant == 2)
            cache.takeCheckpoint(Checkpoint::Kind::Prefix, memory, 0);
        for (int i = 60; i < 63; ++i)
            cache.append(p.items[size_t(i)]);
        memory.decode(0, 3);
        memory.decode(0, 12);   // 图片的前 12 个位置（之后失败，账本里还没有它）
        cache.rollback(memory, 0);
        if (variant == 0) {
            RT_CHECK_EQ(cache.ledger().size(), size_t(63));
        } else if (variant == 1) {
            RT_CHECK(cache.ledger().empty());
            RT_CHECK_EQ(memory.seqPosMax(0), llama_pos(-1));
        } else {
            RT_CHECK_EQ(cache.ledger().size(), size_t(60));
        }
        checkConsistent(cache, memory, 0, hybrid);
    }
}

RT_TEST(seqcache_discard_after_failed_decode_on_hybrid)
{
    // 评审修正：混合模型上 llama_process 中止 / 出错后，llama 对失败那个 ubatch 的 seq_rm 返回 false（循环状态已经在起点之后），
    // 没算过的位置还算在序列里。rollback 按 seq_pos_max 对账会把它们当成已在 KV 里（这里核对这个坑本身）；
    // discardAfter 退回这一步之前最近的检查点，没有就清空
    const Prepared p = userTurn(kSystem, user(2000, 400));
    {
        FakeMemory memory(true);
        SequenceCache cache;
        cache.prepare(p, memory, 0);
        for (int i = 0; i < 60; ++i)
            cache.append(p.items[size_t(i)]);
        memory.decode(0, 60);
        cache.takeCheckpoint(Checkpoint::Kind::Prefix, memory, 0);   // P@60
        for (int i = 60; i < 200; ++i)
            cache.append(p.items[size_t(i)]);
        memory.decode(0, 140);                                        // 上一步成功：账本 200 项
        const size_t before = cache.ledger().size();
        for (int i = 200; i < 400; ++i)
            cache.append(p.items[size_t(i)]);
        memory.decode(0, 100);       // 这一步前一个 ubatch 成功（200…299）
        memory.failedDecode(0, 100); // 后一个 ubatch 中止：300…399 删不掉
        RT_CHECK_EQ(memory.seqPosMax(0), llama_pos(399));   // 没算过的 300…399 还算在序列里
        cache.discardAfter(memory, 0, before);
        RT_CHECK_EQ(cache.ledger().size(), size_t(60));   // 退回 P（这一步之前最近的检查点）
        checkConsistent(cache, memory, 0, true);
        RT_CHECK(cache.prefixFingerprint().has_value());
        // 接着算：只预填 P 之后的
        const Run next = run(cache, memory, 0, p);
        RT_CHECK_EQ(next.plan.keep, 60);
        checkConsistent(cache, memory, 0, true);
    }
    {
        // 没有检查点：整条清空
        FakeMemory memory(true);
        SequenceCache cache;
        const Prepared noPrefix = request(p.items, -1, -1, false);
        cache.prepare(noPrefix, memory, 0);
        for (int i = 0; i < 300; ++i)
            cache.append(noPrefix.items[size_t(i)]);
        memory.decode(0, 200);
        memory.failedDecode(0, 100);
        cache.discardAfter(memory, 0, 0);
        RT_CHECK(cache.ledger().empty());
        RT_CHECK_EQ(memory.seqPosMax(0), llama_pos(-1));
        RT_CHECK_EQ(memory.recurrent(0), llama_pos(-1));
    }
    {
        // 对照：rollback 在混合模型上会把没算过的位置留在账本里（discardAfter 存在的原因）
        FakeMemory memory(true);
        SequenceCache cache;
        const Prepared noPrefix = request(p.items, -1, -1, false);
        cache.prepare(noPrefix, memory, 0);
        for (int i = 0; i < 300; ++i)
            cache.append(noPrefix.items[size_t(i)]);
        memory.decode(0, 200);
        memory.failedDecode(0, 100);
        cache.rollback(memory, 0);
        RT_CHECK_EQ(cache.ledger().size(), size_t(300));
    }
    {
        // PART 型：llama 的 seq_rm 删得掉，rollback 本来就对（运行时只在不可回退的模型上用 discardAfter）
        FakeMemory memory(false);
        SequenceCache cache;
        cache.prepare(p, memory, 0);
        for (int i = 0; i < 300; ++i)
            cache.append(p.items[size_t(i)]);
        memory.decode(0, 200);
        memory.failedDecode(0, 100);
        cache.rollback(memory, 0);
        RT_CHECK_EQ(cache.ledger().size(), size_t(200));
        checkConsistent(cache, memory, 0, false);
    }
}

RT_TEST(seqcache_rollback_keeps_complete_mrope_media)
{
    // M-RoPE 的图片块（Qwen-VL 系列）：块里单元的位置（t 维）都等于起始位置，整块进了 KV 之后 seq_pos_max 只到起始位置。
    // 图片之后的文字那一批失败（KV 满 / 中止）时，账本里的图片块是完整的，要留着（不退检查点、不重新编码）
    for (const bool hybrid : { false, true }) {
        FakeMemory memory(hybrid);
        SequenceCache cache;
        const Prepared p = request(join({ kSystem, kUserHead, { media("img-3", 260, 20) }, range(20, 8) }), 60, -1, false);
        cache.prepare(p, memory, 0);
        for (int i = 0; i < 63; ++i)
            cache.append(p.items[size_t(i)]);
        memory.decode(0, 63);
        cache.append(p.items[63]);   // Media::eval 成功：图片块进账本
        memory.decode(0, 1);         // KV 里多一个位置：图片的起始位置（63）
        for (int i = 64; i < 68; ++i)
            cache.append(p.items[size_t(i)]);
        memory.calls.clear();
        cache.rollback(memory, 0);   // 图片之后的 4 个文字 token 没进 KV
        RT_CHECK_EQ(cache.ledger().size(), size_t(64));
        RT_CHECK_EQ(cache.nextPos(), llama_pos(63 + 20));
        RT_CHECK_EQ(cache.nTokens(), int64_t(63 + 260));
        RT_CHECK(memory.calls.empty());   // KV 不用动
    }
}

// MARK: - 前缀

RT_TEST(seqcache_prefix_fingerprint_and_rollback_to_prefix)
{
    for (const bool hybrid : { false, true }) {
        FakeMemory memory(hybrid);
        SequenceCache cache;
        const Prepared p = userTurn(kSystem, user(2000, 20));
        cache.prepare(p, memory, 0);
        // 还没预填到 P：没有前缀
        RT_CHECK(!cache.prefixFingerprint().has_value());
        RT_CHECK(!cache.rollbackToPrefix(memory, 0));
        run(cache, memory, 0, p, firstReply());
        RT_CHECK(cache.prefixFingerprint() == p.prefix_fingerprint);
        memory.calls.clear();
        RT_CHECK(cache.rollbackToPrefix(memory, 0));
        RT_CHECK_EQ(cache.ledger().size(), size_t(60));
        RT_CHECK(checkpointItems(cache) == std::vector<int64_t>({ 60 }));
        RT_CHECK(memory.calls == (hybrid ? std::vector<std::string>({ "restorePartial 0", "seqRemove 0 60 -1" })
                                         : std::vector<std::string>({ "seqRemove 0 60 -1" })));
        checkConsistent(cache, memory, 0, hybrid);
        // 已经在 P：什么都不用做
        memory.calls.clear();
        RT_CHECK(cache.rollbackToPrefix(memory, 0));
        RT_CHECK(memory.calls.empty());
        // 新对话：接着算
        const Run next = run(cache, memory, 0, userTurn(kSystem, user(2500, 6)));
        RT_CHECK_EQ(next.plan.keep, 60);
        RT_CHECK_EQ(next.prefilled, 3 + 6 + 2 + 5);
    }
}

RT_TEST(seqcache_request_without_prefix_drops_it)
{
    FakeMemory memory(true);
    SequenceCache cache;
    firstTurn(cache, memory, 0);
    RT_CHECK(cache.prefixFingerprint().has_value());
    // 没有系统消息的请求（例如起标题）：没有 P
    const Prepared plain = request(join({ user(2000, 20), kGenerationPrompt }), -1, 25, true);
    run(cache, memory, 0, plain);
    RT_CHECK(!cache.prefixFingerprint().has_value());
    RT_CHECK_EQ(countKind(cache, Checkpoint::Kind::Prefix), 0);
}

RT_TEST(seqcache_adopt_and_add_checkpoint)
{
    FakeMemory memory(true);
    SequenceCache cache;
    const std::vector<LedgerItem> ledger = join({ kSystem, user(2000, 20) });
    const std::string prefix = flr::prefixFingerprint(ledger, 60);
    cache.adopt(ledger, prefix);
    RT_CHECK_EQ(cache.nTokens(), int64_t(85));
    RT_CHECK(cache.checkpoints().empty());
    // P 的指纹记下了，但还没有 P 的状态
    RT_CHECK(!cache.prefixFingerprint().has_value());
    Checkpoint p;
    p.kind = Checkpoint::Kind::Prefix;
    p.n_items = 60;
    p.pos = 59;   // 对不上：忽略
    p.partial = flr::makeBlob({ 'P', ' ', '5', '9' });
    cache.addCheckpoint(p);
    RT_CHECK(cache.checkpoints().empty());
    p.pos = 60;
    cache.addCheckpoint(p);
    RT_CHECK(cache.prefixFingerprint() == prefix);
    // 超出账本的忽略；不是 P 位置的 P 忽略
    Checkpoint far = p;
    far.kind = Checkpoint::Kind::Boundary;
    far.n_items = 90;
    far.pos = 90;
    cache.addCheckpoint(far);
    Checkpoint wrongPrefix = p;
    wrongPrefix.n_items = 50;
    wrongPrefix.pos = 50;
    cache.addCheckpoint(wrongPrefix);
    RT_CHECK(checkpointItems(cache) == std::vector<int64_t>({ 60 }));
    // 指纹格式不对 / 比账本长：不记前缀
    SequenceCache other;
    other.adopt(ledger, std::string("abc"));
    other.addCheckpoint(p);
    RT_CHECK(!other.prefixFingerprint().has_value());
    other.adopt(ledger, std::string("999-0000000000000000"));
    RT_CHECK(!other.prefixFingerprint().has_value());
}

RT_TEST(seqcache_clear_forgets_everything)
{
    FakeMemory memory(true);
    SequenceCache cache;
    firstTurn(cache, memory, 0);
    memory.seqClear(0);
    cache.clear();
    RT_CHECK(cache.ledger().empty());
    RT_CHECK(cache.checkpoints().empty());
    RT_CHECK(!cache.prefixFingerprint().has_value());
    RT_CHECK(!cache.nextStop(0).has_value());
    RT_CHECK_EQ(cache.nTokens(), int64_t(0));
}

// MARK: - 池满（§E.4）的细则

RT_TEST(eviction_tie_breaks)
{
    using flr::SequenceStatus;
    const auto status = [](int seq, bool active, int64_t tokens, uint64_t serial) {
        SequenceStatus s;
        s.seq = seq;
        s.main = seq == 0;
        s.active = active;
        s.n_tokens = tokens;
        s.serial = serial;
        return s;
    };
    // 牺牲者 token 一样多时选更年轻的请求；空闲的不算
    RT_CHECK_EQ(flr::chooseVictim({ status(0, true, 900, 1), status(1, true, 500, 2), status(2, true, 500, 5), status(3, false, 9000, 0) }), 2);
    RT_CHECK_EQ(flr::chooseVictim({ status(1, true, 500, 9), status(2, true, 500, 5) }), 1);
    RT_CHECK_EQ(flr::chooseVictim({}), -1);
    // 空闲序列一样多时腾排在前面的；只有主对话空闲时腾它
    RT_CHECK_EQ(flr::chooseIdleEviction({ status(1, false, 300, 0), status(2, false, 300, 0) }), 1);
    RT_CHECK_EQ(flr::chooseIdleEviction({ status(0, false, 300, 0), status(1, true, 9000, 4) }), 0);
    RT_CHECK_EQ(flr::chooseIdleEviction({}), -1);
}
