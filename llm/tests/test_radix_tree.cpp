// 基数树（RadixTree）与分叉（SequenceCache::fork）的单测：不需要模型
#include "RadixTree.h"
#include "SequenceDriver.h"
#include "rt_test.h"

#include <set>

using flr::RadixTree;
using seqtest::join;
using seqtest::media;
using seqtest::range;

namespace {

std::set<int> holders(const RadixTree::Match& match)
{
    return std::set<int>(match.holders.begin(), match.holders.end());
}

} // namespace

RT_TEST(radix_shared_prefix_is_stored_once)
{
    RadixTree tree;
    const auto system = range(1000, 600);   // 系统提示 + 工具说明
    tree.sync(0, join({ system, range(1, 50) }), 1);
    tree.sync(1, join({ system, range(2000, 80) }), 1);
    tree.sync(2, join({ system, range(3000, 30) }), 1);

    const RadixTree::Stats stats = tree.stats();
    RT_CHECK_EQ(stats.sequences, size_t(3));
    RT_CHECK_EQ(stats.logical, size_t(600 * 3 + 50 + 80 + 30));
    RT_CHECK_EQ(stats.items, size_t(600 + 50 + 80 + 30));
    RT_CHECK_EQ(stats.nodes, size_t(4));

    // 新请求：同样的系统提示 + 别的任务
    const RadixTree::Match match = tree.match(join({ system, range(4000, 10) }));
    RT_CHECK_EQ(match.length, size_t(600));
    RT_CHECK(holders(match) == std::set<int>({ 0, 1, 2 }));
    RT_CHECK_EQ(tree.exclusive(1), size_t(80));
    RT_CHECK_EQ(tree.sharedWithOthers(1), size_t(600));
    RT_CHECK_EQ(tree.shared(0, 2), size_t(600));
}

RT_TEST(radix_match_inside_an_edge_and_exclude)
{
    RadixTree tree;
    tree.sync(5, range(1, 100), 1);
    // 走到边的中间就分歧
    const RadixTree::Match match = tree.match(join({ range(1, 40), range(900, 5) }));
    RT_CHECK_EQ(match.length, size_t(40));
    RT_CHECK(holders(match) == std::set<int>({ 5 }));
    // 排除自己时什么都匹配不到
    const RadixTree::Match none = tree.match(range(1, 100), { 5 });
    RT_CHECK_EQ(none.length, size_t(0));
    RT_CHECK(none.holders.empty());
}

RT_TEST(radix_append_only_extends_and_generation_change_reinserts)
{
    RadixTree tree;
    auto ledger = range(1, 10);
    tree.sync(0, ledger, 7);
    // 只追加（逐 token 生成）：同一代，接在叶子上
    for (int i = 0; i < 5; ++i) {
        ledger.push_back(flr::LedgerItem::text(500 + i));
        tree.sync(0, ledger, 7);
    }
    RT_CHECK_EQ(tree.stats().nodes, size_t(1));
    RT_CHECK_EQ(tree.match(ledger).length, size_t(15));

    // 截断（代变了）：整条重插
    ledger.resize(6);
    ledger.push_back(flr::LedgerItem::text(77));
    tree.sync(0, ledger, 8);
    RT_CHECK_EQ(tree.match(ledger).length, size_t(7));
    RT_CHECK_EQ(tree.stats().items, size_t(7));
}

RT_TEST(radix_remove_merges_edges)
{
    RadixTree tree;
    const auto shared = range(1, 50);
    tree.sync(0, join({ shared, range(100, 10) }), 1);
    tree.sync(1, join({ shared, range(200, 10) }), 1);
    RT_CHECK_EQ(tree.stats().nodes, size_t(3));
    tree.remove(1);
    // 只剩序列 0：两段合成一条边
    RT_CHECK_EQ(tree.stats().nodes, size_t(1));
    RT_CHECK_EQ(tree.stats().items, size_t(60));
    RT_CHECK_EQ(tree.exclusive(0), size_t(60));
    RT_CHECK(!tree.contains(1));

    // 两个一样的账本：同一个节点，两人都经过
    tree.sync(2, join({ shared, range(100, 10) }), 1);
    RT_CHECK_EQ(tree.stats().nodes, size_t(1));
    RT_CHECK_EQ(tree.exclusive(0), size_t(0));
    tree.remove(0);
    RT_CHECK_EQ(tree.exclusive(2), size_t(60));
}

RT_TEST(radix_one_sequence_extends_another)
{
    RadixTree tree;
    tree.sync(0, range(1, 30), 1);
    // 序列 1 从序列 0 的末尾接着写（同一段对话的下一轮被别的序列接走）
    tree.sync(1, range(1, 45), 1);
    RT_CHECK_EQ(tree.exclusive(0), size_t(0));
    RT_CHECK_EQ(tree.exclusive(1), size_t(15));
    RT_CHECK_EQ(tree.stats().items, size_t(45));
    // 序列 0 又长出自己的一段：在 30 处分叉
    tree.sync(0, join({ range(1, 30), range(900, 4) }), 1);
    RT_CHECK_EQ(tree.shared(0, 1), size_t(30));
    RT_CHECK_EQ(tree.stats().nodes, size_t(3));
}

RT_TEST(radix_media_items_are_keys_too)
{
    RadixTree tree;
    tree.sync(0, join({ range(1, 20), { media("图A", 256, 4) }, range(30, 5) }), 1);
    tree.sync(1, join({ range(1, 20), { media("图B", 256, 4) }, range(30, 5) }), 1);
    const RadixTree::Match a = tree.match(join({ range(1, 20), { media("图A", 256, 4) }, range(30, 2) }));
    RT_CHECK_EQ(a.length, size_t(23));
    RT_CHECK(holders(a) == std::set<int>({ 0 }));
    const RadixTree::Match c = tree.match(join({ range(1, 20), { media("图C", 256, 4) } }));
    RT_CHECK_EQ(c.length, size_t(20));
    RT_CHECK(holders(c) == std::set<int>({ 0, 1 }));
}

RT_TEST(fork_part_model_shares_prefix)
{
    FakeMemory memory(false);
    flr::SequenceCache source;
    const auto system = range(1000, 300);
    const auto items = join({ system, range(1, 40) });
    seqtest::run(source, memory, 0, seqtest::request(items, 300, 300, true));
    RT_CHECK_EQ(source.ledger().size(), items.size());

    // 分叉：前 300 项给序列 3
    memory.seqShare(0, 3, 300);
    flr::SequenceCache target;
    target.fork(source, 300);
    RT_CHECK_EQ(target.ledger().size(), size_t(300));
    RT_CHECK_EQ(target.nextPos(), llama_pos(300));
    RT_CHECK_EQ(memory.seqPosMax(3), llama_pos(299));
    // P 跟着过来（PART 型是位置标记）
    RT_CHECK(target.prefixFingerprint().has_value());

    // 新请求在它上面接着算：账本是提示的前缀
    const auto next = join({ system, range(5000, 20) });
    const flr::SequenceCache::Plan plan = target.prepare(seqtest::request(next, 300, 300, true), memory, 3);
    RT_CHECK_EQ(plan.keep, 300);
    RT_CHECK(!plan.reset);
}

RT_TEST(fork_hybrid_model_restores_checkpoint)
{
    FakeMemory memory(true);
    flr::SequenceCache source;
    const auto system = range(1000, 300);
    const auto items = join({ system, range(1, 400) });
    seqtest::run(source, memory, 0, seqtest::request(items, 300, 300, true));
    const flr::Checkpoint* p = nullptr;
    for (const flr::Checkpoint& checkpoint : source.checkpoints()) {
        if (checkpoint.n_items == 300)
            p = &checkpoint;
    }
    RT_CHECK(p != nullptr);
    RT_CHECK(p->hasData());

    // 先拿注意力的那段（循环状态这时是 source 末尾的），再用 P 的检查点盖掉循环状态
    memory.seqShare(0, 2, p->pos);
    RT_CHECK_EQ(memory.recurrent(2), llama_pos(699));
    RT_CHECK(memory.restorePartial(2, p->partial));
    RT_CHECK_EQ(memory.recurrent(2), llama_pos(299));
    flr::SequenceCache target;
    target.fork(source, 300);
    RT_CHECK_EQ(target.ledger().size(), size_t(300));
    // 检查点共用同一份字节，不拷贝
    bool sharedBlob = false;
    for (const flr::Checkpoint& checkpoint : target.checkpoints())
        sharedBlob = sharedBlob || checkpoint.partial == p->partial;
    RT_CHECK(sharedBlob);

    const auto next = join({ system, range(7000, 50) });
    const flr::SequenceCache::Plan plan = target.prepare(seqtest::request(next, 300, 300, true), memory, 2);
    RT_CHECK_EQ(plan.keep, 300);
    RT_CHECK(!plan.reset);
    RT_CHECK_EQ(plan.restored, -1);
}
