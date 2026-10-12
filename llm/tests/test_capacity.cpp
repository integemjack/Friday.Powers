// 按显存定容量（Capacity）与基数树格子数（RadixTree::cells）的单测：不需要模型
#include "Capacity.h"
#include "RadixTree.h"
#include "SequenceDriver.h"
#include "rt_test.h"

#include <set>

using flr::CapacityInput;
using flr::CapacityPlan;
using flr::DeviceMeasure;
using flr::planCapacity;
using seqtest::join;
using seqtest::media;
using seqtest::range;

namespace {

constexpr int64_t MiB = 1024 * 1024;
constexpr int64_t GiB = 1024 * MiB;

DeviceMeasure card(int64_t freeGiB, int64_t usedGiB, int64_t perTokenKiB, int64_t perSequenceMiB = 0)
{
    DeviceMeasure device;
    device.name = "GPU0";
    device.free = freeGiB * GiB;
    device.used = usedGiB * GiB;
    device.perToken = perTokenKiB * 1024;
    device.perSequence = perSequenceMiB * MiB;
    device.margin = 1 * GiB;
    return device;
}

CapacityInput hybrid(const DeviceMeasure& device, int train)
{
    CapacityInput input;
    input.devices = { device };
    input.nCtxTrain = train;
    input.attentionOnly = false;
    input.baseSequences = flr::defaultSequences(flr::kMinParallel, false);
    return input;
}

} // namespace

RT_TEST(capacity_big_card_caches_sessions_and_caps_parallel)
{
    // RTX PRO 6000（96 GB）跑 Qwen3.6-35B-A3B 这类混合 MoE：权重 + 训练上下文约 25 GB，KV 约 20 KiB/token，循环状态约 60 MiB/序列
    const CapacityPlan plan = planCapacity(hybrid(card(94, 25, 20, 60), 262144));
    RT_CHECK(plan.grown);
    // 缓存的序列放到上限，池子按 32 个会话放大，同时算的自动最多 8 个（再多反而慢）
    RT_CHECK_EQ(plan.sequences, flr::kCacheSequences);
    RT_CHECK_EQ(plan.nCtx, flr::kPoolSessions * flr::kCellsPerRequest);
    RT_CHECK_EQ(plan.parallel, flr::kMaxParallel);
}

RT_TEST(capacity_small_card_keeps_old_behaviour)
{
    // 16 GB 卡放 9B 混合模型：放不下训练上下文 → 照旧交给 fit，同时 4 个
    const CapacityPlan plan = planCapacity(hybrid(card(15, 16, 32, 50), 262144));
    RT_CHECK(!plan.grown);
    RT_CHECK_EQ(plan.nCtx, 0);
    RT_CHECK_EQ(plan.parallel, flr::kMinParallel);
    RT_CHECK_EQ(plan.sequences, flr::kMinParallel + 4);
}

RT_TEST(capacity_dense_model_limited_by_kv_size)
{
    // 稠密注意力模型 KV 大（256 KiB/token）：多出来的 60 GB × 六成 ≈ 144K token，加上 40K 训练上下文
    CapacityInput input;
    input.devices = { card(94, 33, 256) };
    input.nCtxTrain = 40960;
    input.attentionOnly = true;
    input.baseSequences = flr::defaultSequences(flr::kMinParallel, true);
    const CapacityPlan plan = planCapacity(input);
    RT_CHECK(plan.grown);
    RT_CHECK_EQ(plan.nCtx, 40960 + 147456);
    RT_CHECK_EQ(plan.nCtx % 256, 0);
    RT_CHECK_EQ(plan.parallel, flr::kMaxParallel);
    // 纯注意力模型的序列不占显存
    RT_CHECK_EQ(plan.sequences, flr::kCacheSequences);
}

RT_TEST(capacity_recurrent_states_limit_sequences)
{
    // 循环状态很大（每条 1 GiB）：序列最多拿多出来的四分之一（40.8 GiB / 4 ≈ 10 条，加上原来的 8 条）
    const CapacityPlan plan = planCapacity(hybrid(card(94, 25, 20, 1024), 262144));
    RT_CHECK(plan.grown);
    RT_CHECK_EQ(plan.sequences, 18);
    RT_CHECK_EQ(plan.nCtx, 18 * flr::kCellsPerRequest);
    RT_CHECK_EQ(plan.parallel, flr::kMaxParallel);
}

RT_TEST(capacity_user_chosen_parallel)
{
    // 用户在 Friday 里选了同时 8 个、32 个：照选的，序列至少 parallel + 4，池子照样按显存放大
    CapacityInput input = hybrid(card(94, 25, 20, 60), 65536);
    input.autoParallel = false;
    input.fixedParallel = 8;
    input.baseSequences = flr::defaultSequences(8, false);
    CapacityPlan plan = planCapacity(input);
    RT_CHECK(plan.grown);
    RT_CHECK_EQ(plan.parallel, 8);
    RT_CHECK_EQ(plan.sequences, flr::kCacheSequences);
    RT_CHECK_EQ(plan.nCtx, flr::kPoolSessions * flr::kCellsPerRequest);

    input.fixedParallel = 64;
    input.baseSequences = flr::defaultSequences(64, false);
    plan = planCapacity(input);
    RT_CHECK_EQ(plan.parallel, 64);
    RT_CHECK(plan.sequences >= 68);
}

RT_TEST(capacity_unified_memory_takes_less)
{
    DeviceMeasure discrete = card(40, 20, 64);
    DeviceMeasure unified = discrete;
    unified.unified = true;
    CapacityInput input = hybrid(discrete, 32768);
    const CapacityPlan a = planCapacity(input);
    input.devices = { unified };
    const CapacityPlan b = planCapacity(input);
    RT_CHECK(a.nCtx > b.nCtx);
}

RT_TEST(capacity_generation_reserve)
{
    RT_CHECK_EQ(flr::generationReserve(-1, 1000, 262144), 4096);
    RT_CHECK_EQ(flr::generationReserve(256, 1000, 262144), 256);
    RT_CHECK_EQ(flr::generationReserve(-1, 262000, 262144), 144);
    RT_CHECK_EQ(flr::generationReserve(-1, 300000, 262144), 0);
}

RT_TEST(radix_cells_count_shared_once_and_media_tokens)
{
    flr::RadixTree tree;
    const auto system = range(1000, 600);
    tree.sync(0, join({ system, range(1, 50) }), 1);
    tree.sync(1, join({ system, range(2000, 80) }), 1);
    // 图片块占 256 格
    tree.sync(2, join({ system, std::vector<flr::LedgerItem> { media("img", 256, 4) }, range(3000, 30) }), 1);

    RT_CHECK_EQ(tree.cells({ 0 }), size_t(650));
    RT_CHECK_EQ(tree.cells({ 0, 1 }), size_t(600 + 50 + 80));
    RT_CHECK_EQ(tree.cells({ 2 }), size_t(600 + 256 + 30));
    RT_CHECK_EQ(tree.stats().cells, size_t(600 + 50 + 80 + 256 + 30));
    RT_CHECK_EQ(tree.cells({}), size_t(0));

    // 拆边、合边以后格子数照样对
    tree.sync(3, join({ range(1000, 300), range(5000, 10) }), 1);
    RT_CHECK_EQ(tree.stats().cells, size_t(600 + 50 + 80 + 256 + 30 + 10));
    tree.remove(3);
    tree.remove(1);
    RT_CHECK_EQ(tree.stats().cells, size_t(600 + 50 + 256 + 30));
    // 只追加
    auto ledger = join({ system, range(1, 50) });
    for (int i = 0; i < 5; ++i) {
        ledger.push_back(flr::LedgerItem::text(9000 + i));
        tree.sync(0, ledger, 1);
    }
    RT_CHECK_EQ(tree.cells({ 0 }), size_t(655));
}
