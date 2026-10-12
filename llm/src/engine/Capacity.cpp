// 按显存定容量（见 Capacity.h）
#include "Capacity.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <limits>

namespace flr {
namespace {

constexpr int64_t kMiB = 1024 * 1024;

std::string format(const char* pattern, ...)
{
    char buffer[512];
    va_list args;
    va_start(args, pattern);
    std::vsnprintf(buffer, sizeof(buffer), pattern, args);
    va_end(args);
    return buffer;
}

} // namespace

int defaultSequences(int parallel, bool attentionOnly)
{
    const int sequences = attentionOnly ? std::max(32, parallel + 8) : parallel + 4;
    return std::min(sequences, kMaxSequences);
}

double growthShare(bool unified)
{
    return unified ? 0.25 : 0.6;
}

CapacityPlan planCapacity(const CapacityInput& input)
{
    CapacityPlan plan;
    plan.parallel = input.autoParallel ? kMinParallel : std::max(1, input.fixedParallel);
    plan.sequences = input.fixedSequences > 0 ? input.fixedSequences : defaultSequences(plan.parallel, input.attentionOnly);
    if (input.devices.empty() || input.nCtxTrain <= 0) {
        plan.why = "量不出显存";
        return plan;
    }
    // 每张卡放下训练上下文之后多出来的、能拿来放大的那部分
    std::vector<double> budget;
    for (const DeviceMeasure& device : input.devices) {
        const int64_t surplus = device.free - device.used - device.margin;
        if (surplus <= 0) {
            plan.why = format("%s 放不下训练上下文（空闲 %lld MiB，要 %lld + 余量 %lld MiB）", device.name.c_str(),
                              (long long) (device.free / kMiB), (long long) (device.used / kMiB), (long long) (device.margin / kMiB));
            return plan;
        }
        budget.push_back(double(surplus) * growthShare(device.unified));
    }
    // 多 extra 条序列时，KV 池还能比训练上下文多放多少 token（各卡按层分，取最紧的那张）；序列都放不下为 -1
    const auto tokens = [&](int extra) -> int64_t {
        int64_t best = std::numeric_limits<int64_t>::max();
        bool any = false;
        for (size_t i = 0; i < input.devices.size(); ++i) {
            const DeviceMeasure& device = input.devices[i];
            const double left = budget[i] - double(std::max(0, extra)) * double(std::max<int64_t>(0, device.perSequence));
            if (left < 0)
                return -1;
            if (device.perToken <= 0)
                continue;
            any = true;
            best = std::min(best, int64_t(left / double(device.perToken)));
        }
        return any ? best : 0;
    };
    const int minExtra = input.attentionOnly ? 8 : 4;

    // 1. 缓存几条序列（会话 / 子智能体，缓存着就不用整段重算）：纯注意力模型的序列不占显存，留 kCacheSequences 条；
    //    混合 / 循环模型每条一份循环状态，最多拿多出来的四分之一
    int sequences = input.fixedSequences;
    if (sequences <= 0) {
        int64_t room = kCacheSequences;
        if (!input.attentionOnly) {
            for (size_t i = 0; i < input.devices.size(); ++i) {
                if (input.devices[i].perSequence > 0)
                    room = std::min<int64_t>(room, int64_t(budget[i] / 4 / double(input.devices[i].perSequence)));
            }
            room += input.baseSequences;
        }
        sequences = int(std::clamp<int64_t>(room, input.baseSequences, std::max(kCacheSequences, input.baseSequences)));
    }
    int64_t extra = tokens(sequences - input.baseSequences);
    while (extra < 0 && sequences > input.baseSequences)
        extra = tokens(--sequences - input.baseSequences);
    extra = std::max<int64_t>(0, extra);

    // 2. KV 池：装得下这些会话（每个按 kCellsPerRequest 格，最多 kPoolSessions 个）
    const int64_t cap = std::max<int64_t>(input.nCtxTrain, int64_t(std::min(sequences, kPoolSessions)) * kCellsPerRequest);
    int64_t pool = std::min<int64_t>(int64_t(input.nCtxTrain) + extra, cap);
    pool = std::min<int64_t>(pool, std::numeric_limits<int32_t>::max() / 2);
    pool -= pool % 256;
    pool = std::max<int64_t>(pool, input.nCtxTrain);

    // 3. 同时几个：池子按每个请求 kCellsPerRequest 格能放几个，自动最多 kMaxParallel（再多反而慢，见 Capacity.h）
    const int parallel = input.autoParallel ? int(std::clamp<int64_t>(pool / kCellsPerRequest, kMinParallel, kMaxParallel)) : plan.parallel;
    if (input.fixedSequences <= 0)
        sequences = std::min(std::max(sequences, parallel + minExtra), kMaxSequences);

    plan.parallel = parallel;
    plan.sequences = sequences;
    if (pool <= input.nCtxTrain && parallel == (input.autoParallel ? kMinParallel : input.fixedParallel)
        && sequences == input.baseSequences) {
        plan.why = "显存只够训练上下文";
        return plan;
    }
    plan.grown = true;
    plan.nCtx = int(pool);
    const DeviceMeasure& first = input.devices.front();
    plan.why = format("%s 空闲 %lld MiB，训练上下文要 %lld MiB，KV %.1f KiB/token、循环状态 %.1f MiB/序列，余量 %lld MiB",
                      first.name.c_str(), (long long) (first.free / kMiB), (long long) (first.used / kMiB), double(first.perToken) / 1024.0,
                      double(first.perSequence) / double(kMiB), (long long) (first.margin / kMiB));
    return plan;
}

int generationReserve(int nPredict, int64_t promptCells, int nCtxSeq)
{
    const int64_t want = nPredict > 0 ? std::min(nPredict, 4096) : 4096;
    const int64_t room = std::max<int64_t>(0, int64_t(nCtxSeq) - promptCells);
    return int(std::min(want, room));
}

} // namespace flr
