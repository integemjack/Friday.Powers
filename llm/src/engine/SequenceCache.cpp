// 序列账本与缓存（docs/LOCAL_INFERENCE.md §F.2；WP3 拥有）：账本、P / B / E 检查点、复用决策。只在推理线程上用。
// 对应 Swift：MLXLMCommon/ChatSession.swift 的 Friday 补丁（checkpoint / prefixCheckpoint，12d0450）；
// 复用与检查点的细节参考 llama-server b11294（tools/server/server-context.cpp:3196-3417 复用、3500-3616 建检查点）。
//
// 检查点有两种：
//   - 不可回退的模型（混合 / 循环 / SWA，MemoryOps::partialRemovable() == false）：PARTIAL_ONLY 状态（Qwen3.5-9B 每个 50.25 MiB）。
//     prepare 排好这次预填要停的地方（P / B / E），WP2 预填到了调 takeCheckpoint 取；分歧时回退到 n_items ≤ 公共前缀的最近一个。
//   - PART 型（普通 Transformer）：KV 能从任意位置截断，不需要状态，预填也不停；P / B 只记位置（partial 为空的「标记」），
//     账本长到那里才算数，供回合边界落盘、前缀快照、回退到 P 用。
// 同一个位置只存一份数据，身份（P / B / E）记在 roles 里：保留 P 一个 + 最近 3 个 B + 最近 2 个 E（CachePolicy），
// 一份检查点的身份都过期了才删掉。前缀快照不在预填中途整份拷贝：flr_prefix_save 时由 P 检查点 + 注意力 KV 现场重建
// （savePrefixState，StateFile.cpp），所以 known_prefixes 不影响这里。
#include "internal.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <utility>

namespace flr {
namespace {

constexpr unsigned rolePrefix = 1u << 0;
constexpr unsigned roleBoundary = 1u << 1;
constexpr unsigned roleEnd = 1u << 2;

unsigned roleOf(Checkpoint::Kind kind)
{
    switch (kind) {
    case Checkpoint::Kind::Prefix: return rolePrefix;
    case Checkpoint::Kind::Boundary: return roleBoundary;
    case Checkpoint::Kind::End: break;
    }
    return roleEnd;
}

/// 对外的 Checkpoint::kind：身份里最重要的那个（P > B > E）
Checkpoint::Kind primaryKind(unsigned roles)
{
    if (roles & rolePrefix)
        return Checkpoint::Kind::Prefix;
    if (roles & roleBoundary)
        return Checkpoint::Kind::Boundary;
    return Checkpoint::Kind::End;
}

std::string rolesName(unsigned roles)
{
    std::string name;
    for (const auto& [role, text] : { std::pair<unsigned, const char*> { rolePrefix, "P" }, { roleBoundary, "B" }, { roleEnd, "E" } }) {
        if (roles & role)
            name += name.empty() ? text : std::string("+") + text;
    }
    return name;
}

/// 账本与新提示的公共前缀（按项比较：文本比 token，媒体块比 media_id + 单元数 + 位置数）
size_t commonPrefix(const std::vector<LedgerItem>& a, const std::vector<LedgerItem>& b)
{
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i])
        ++i;
    return i;
}

/// items 前 n 项之后的下一个位置（Σ n_pos）
llama_pos positionAfter(const std::vector<LedgerItem>& items, size_t n)
{
    llama_pos pos = 0;
    for (size_t i = 0; i < n && i < items.size(); ++i)
        pos += items[i].n_pos;
    return pos;
}

double elapsedMs(std::chrono::steady_clock::time_point since)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

} // namespace

struct SequenceCache::State {
    CachePolicy policy;
    std::vector<LedgerItem> ledger;
    /// 账本的代（generation()）
    uint64_t generation = 1;
    int64_t tokens = 0;
    llama_pos nextPos = 0;

    /// 检查点（按 n_items 升序）与它们的身份；roleSerial 记每种身份是第几次取的（淘汰最旧的）
    std::vector<Checkpoint> checkpoints;
    struct Roles {
        unsigned roles = 0;
        uint64_t boundarySerial = 0;
        uint64_t endSerial = 0;
    };
    std::vector<Roles> roles;
    uint64_t serial = 0;

    /// 这条序列的 P（这次请求的 P 的指纹与项数）；没有 P 时为空 / -1
    std::string prefix;
    int64_t prefixItems = -1;

    /// 这次请求要停下取检查点的地方（不可回退的模型；at 升序，请求进行中不删：nextStop 按已处理数往后找）
    std::vector<Stop> stops;
    /// 这次请求的位置标记（PART 型：账本长到 at 时记成 partial 为空的检查点）
    struct Mark {
        int64_t at = 0;
        llama_pos pos = 0;
        unsigned role = 0;
    };
    std::vector<Mark> marks;

    void recount()
    {
        tokens = 0;
        nextPos = 0;
        for (const LedgerItem& item : ledger) {
            tokens += item.n_tokens;
            nextPos += item.n_pos;
        }
    }

    /// 账本截到前 n 项；超出的检查点删掉
    void truncate(size_t n)
    {
        if (n < ledger.size()) {
            ledger.resize(n);
            recount();
            ++generation;
        }
        for (size_t i = checkpoints.size(); i-- > 0;) {
            if (checkpoints[i].n_items > int64_t(n)) {
                checkpoints.erase(checkpoints.begin() + std::ptrdiff_t(i));
                roles.erase(roles.begin() + std::ptrdiff_t(i));
            }
        }
    }

    /// 账本、检查点清空；request = true 时这次请求的停点、标记和 P 也清掉（否则请求还要接着从 0 算）
    void reset(bool request)
    {
        ledger.clear();
        recount();
        ++generation;
        checkpoints.clear();
        roles.clear();
        if (request) {
            stops.clear();
            marks.clear();
            prefix.clear();
            prefixItems = -1;
        }
    }

    int find(int64_t nItems) const
    {
        for (size_t i = 0; i < checkpoints.size(); ++i) {
            if (checkpoints[i].n_items == nItems)
                return int(i);
        }
        return -1;
    }

    /// n_items ≤ limit、带状态数据的最近一个检查点（回退用）；没有为 -1
    int latestRestorable(int64_t limit) const
    {
        int best = -1;
        for (size_t i = 0; i < checkpoints.size(); ++i) {
            if (checkpoints[i].n_items <= limit && checkpoints[i].hasData())
                best = int(i);
        }
        return best;
    }

    void addRole(size_t index, unsigned role)
    {
        Roles& r = roles[index];
        r.roles |= role;
        if (role & roleBoundary)
            r.boundarySerial = ++serial;
        if (role & roleEnd)
            r.endSerial = ++serial;
        checkpoints[index].kind = primaryKind(r.roles);
    }

    /// 放进一个检查点（同一位置已有就只加身份；已有的是标记、新的带数据时换成新的数据），然后按 CachePolicy 淘汰
    void insert(Checkpoint checkpoint, unsigned role)
    {
        const int existing = find(checkpoint.n_items);
        if (existing >= 0) {
            if (!checkpoints[size_t(existing)].hasData() && checkpoint.hasData())
                checkpoints[size_t(existing)].partial = std::move(checkpoint.partial);
            addRole(size_t(existing), role);
        } else {
            const auto at = std::upper_bound(checkpoints.begin(), checkpoints.end(), checkpoint.n_items,
                                             [](int64_t n, const Checkpoint& c) { return n < c.n_items; });
            const size_t index = size_t(at - checkpoints.begin());
            checkpoints.insert(at, std::move(checkpoint));
            roles.insert(roles.begin() + std::ptrdiff_t(index), Roles {});
            addRole(index, role);
        }
        enforceLimits();
    }

    void dropRole(unsigned role)
    {
        for (size_t i = 0; i < roles.size(); ++i) {
            roles[i].roles &= ~role;
            checkpoints[i].kind = primaryKind(roles[i].roles);
        }
        removeUnused();
    }

    void removeUnused()
    {
        for (size_t i = checkpoints.size(); i-- > 0;) {
            if (roles[i].roles == 0) {
                checkpoints.erase(checkpoints.begin() + std::ptrdiff_t(i));
                roles.erase(roles.begin() + std::ptrdiff_t(i));
            }
        }
    }

    /// P 只留这条序列现在的那个；B、E 各留最近的 max_boundary / max_end 个
    void enforceLimits()
    {
        for (size_t i = 0; i < roles.size(); ++i) {
            if ((roles[i].roles & rolePrefix) && checkpoints[i].n_items != prefixItems)
                roles[i].roles &= ~rolePrefix;
        }
        const auto keepNewest = [this](unsigned role, int limit, uint64_t Roles::*serialOf) {
            std::vector<size_t> holders;
            for (size_t i = 0; i < roles.size(); ++i) {
                if (roles[i].roles & role)
                    holders.push_back(i);
            }
            std::sort(holders.begin(), holders.end(), [&](size_t a, size_t b) { return roles[a].*serialOf > roles[b].*serialOf; });
            for (size_t k = size_t(std::max(limit, 0)); k < holders.size(); ++k)
                roles[holders[k]].roles &= ~role;
        };
        keepNewest(roleBoundary, policy.max_boundary, &Roles::boundarySerial);
        keepNewest(roleEnd, policy.max_end, &Roles::endSerial);
        for (size_t i = 0; i < roles.size(); ++i)
            checkpoints[i].kind = primaryKind(roles[i].roles);
        removeUnused();
    }

    /// 账本刚好长到某个位置标记（PART 型）：记下来
    void materializeMarks()
    {
        for (const Mark& mark : marks) {
            if (mark.at == int64_t(ledger.size())) {
                Checkpoint checkpoint;
                checkpoint.kind = primaryKind(mark.role);
                checkpoint.n_items = mark.at;
                checkpoint.pos = mark.pos;
                insert(std::move(checkpoint), mark.role);
            }
        }
    }

    const Checkpoint* prefixCheckpoint() const
    {
        if (prefix.empty() || prefixItems <= 0 || prefixItems > int64_t(ledger.size()))
            return nullptr;
        const int index = find(prefixItems);
        if (index < 0 || !(roles[size_t(index)].roles & rolePrefix))
            return nullptr;
        return &checkpoints[size_t(index)];
    }
};

SequenceCache::SequenceCache(CachePolicy policy)
    : m_state(std::make_unique<State>())
{
    m_state->policy = policy;
}

SequenceCache::~SequenceCache() = default;
SequenceCache::SequenceCache(SequenceCache&& other) noexcept = default;
SequenceCache& SequenceCache::operator=(SequenceCache&& other) noexcept = default;

SequenceCache::Plan SequenceCache::prepare(const Prepared& p, MemoryOps& memory, int seq)
{
    State& s = *m_state;
    s.stops.clear();
    s.marks.clear();
    Plan plan;
    const std::vector<LedgerItem>& items = p.items;
    if (items.empty()) {
        // 没有要算的（WP2 不会发空请求）：清空，避免账本与 KV 对不上
        memory.seqClear(seq);
        s.reset(true);
        plan.reset = true;
        return plan;
    }

    const size_t common = commonPrefix(s.ledger, items);
    const size_t last = items.size() - 1;   // 至少算 1 项才有 logits
    const bool part = memory.partialRemovable();
    const size_t cached = s.ledger.size();
    size_t keep = 0;
    bool reset = false;
    if (common == cached && items.size() > cached) {
        // ① 账本是新提示的前缀：直接接着算
        keep = cached;
    } else if (part) {
        // ② 普通 Transformer：截到公共前缀（完全相同时退一项）
        keep = std::min(common, last);
        reset = !memory.seqRemove(seq, positionAfter(s.ledger, keep), -1);
    } else {
        // ③ 不可回退：回到 n_items ≤ 公共前缀的最近检查点（恢复循环状态，再删掉注意力 KV 的尾巴）；没有就清空重算
        const int best = s.latestRestorable(int64_t(std::min(common, last)));
        if (best >= 0 && memory.restorePartial(seq, s.checkpoints[size_t(best)].partial)
            && memory.seqRemove(seq, s.checkpoints[size_t(best)].pos, -1)) {
            keep = size_t(s.checkpoints[size_t(best)].n_items);
            plan.restored = int(keep);
        } else {
            reset = true;
        }
    }
    if (reset) {
        memory.seqClear(seq);
        s.reset(false);
        keep = 0;
        plan.reset = true;
    } else {
        s.truncate(keep);
    }
    plan.keep = int(keep);

    // 这次请求的 P：指纹变了（系统提示 / 工具变了，或这次没有 P）旧的 P 身份作废
    const int prefixAt = p.prefix_boundary;
    if (prefixAt > 0 && size_t(prefixAt) < items.size()) {
        const std::string fingerprint = p.prefix_fingerprint.empty() ? flr::prefixFingerprint(items, prefixAt) : p.prefix_fingerprint;
        if (fingerprint != s.prefix || s.prefixItems != prefixAt) {
            s.prefix = fingerprint;
            s.prefixItems = prefixAt;
            s.enforceLimits();   // 位置不是新 P 的 P 身份去掉
        }
    } else if (!s.prefix.empty()) {
        s.prefix.clear();
        s.prefixItems = -1;
        s.dropRole(rolePrefix);
    }

    if (s.policy.checkpoints) {
        const int boundaryAt = p.user_driven ? p.transcript_boundary : -1;
        const auto valid = [&](int at) { return at > 0 && size_t(at) < items.size(); };
        if (part) {
            // 只记位置（P、B）；已经在 KV 里的马上记，还没到的等账本长到那里（append）
            const auto mark = [&](int at, unsigned role) {
                if (valid(at))
                    s.marks.push_back({ at, positionAfter(items, size_t(at)), role });
            };
            mark(prefixAt, rolePrefix);
            mark(boundaryAt, roleBoundary);
            for (const State::Mark& m : s.marks) {
                if (m.at <= int64_t(keep)) {
                    Checkpoint checkpoint;
                    checkpoint.n_items = m.at;
                    checkpoint.pos = m.pos;
                    s.insert(std::move(checkpoint), m.role);
                }
            }
        } else {
            const auto stop = [&](int at, Checkpoint::Kind kind) {
                // 图片块之后不建（== server：do not checkpoint after mtmd chunks）
                if (!valid(at) || size_t(at) <= keep || items[size_t(at) - 1].isMedia())
                    return;
                for (Stop& existing : s.stops) {
                    if (existing.at == at) {
                        existing.kinds.push_back(kind);
                        return;
                    }
                }
                s.stops.push_back({ at, { kind } });
            };
            stop(prefixAt, Checkpoint::Kind::Prefix);
            stop(boundaryAt, Checkpoint::Kind::Boundary);
            const int endAt = int(items.size()) - s.policy.end_offset;
            if (s.policy.min_end_gap > 0 && valid(endAt)) {
                // E 前面最近的检查点（留着的、带数据的）或这次的停点；离它太近就不设 E（CachePolicy::min_end_gap）
                int64_t previous = 0;
                for (const Checkpoint& checkpoint : s.checkpoints) {
                    if (checkpoint.hasData() && checkpoint.n_items <= endAt)
                        previous = std::max(previous, checkpoint.n_items);
                }
                for (const Stop& existing : s.stops) {
                    if (existing.at <= endAt)
                        previous = std::max(previous, int64_t(existing.at));
                }
                if (endAt - previous >= s.policy.min_end_gap)
                    stop(endAt, Checkpoint::Kind::End);
            } else {
                stop(endAt, Checkpoint::Kind::End);
            }
            std::sort(s.stops.begin(), s.stops.end(), [](const Stop& a, const Stop& b) { return a.at < b.at; });
        }
    }

    if (cached > 0 || plan.restored >= 0) {
        logf(FLR_LOG_NOTICE, "序列 %d：缓存 %zu 项，新提示 %zu 项，公共前缀 %zu → %s，复用 %zu 项", seq, cached, items.size(), common,
             plan.restored >= 0 ? "回退到检查点" : plan.reset ? "清空重算" : keep == cached ? "接着算" : "截断", keep);
    }
    return plan;
}

const std::vector<SequenceCache::Stop>& SequenceCache::stops() const
{
    return m_state->stops;
}

std::optional<SequenceCache::Stop> SequenceCache::nextStop(int processed) const
{
    for (const Stop& stop : m_state->stops) {
        if (stop.at > processed)
            return stop;
    }
    return std::nullopt;
}

void SequenceCache::takeCheckpoint(Checkpoint::Kind kind, MemoryOps& memory, int seq)
{
    State& s = *m_state;
    const int64_t at = int64_t(s.ledger.size());
    if (at <= 0 || !s.policy.checkpoints)
        return;
    const unsigned role = roleOf(kind);
    // P 只认这次请求的 P（指纹是按它算的）
    if (role == rolePrefix && at != s.prefixItems)
        return;
    if (memory.partialRemovable()) {
        // PART 型：只记位置（E 没有用）
        if (role == roleEnd)
            return;
        Checkpoint checkpoint;
        checkpoint.n_items = at;
        checkpoint.pos = s.nextPos;
        s.insert(std::move(checkpoint), role);
        return;
    }
    const int existing = s.find(at);
    if (existing >= 0 && s.checkpoints[size_t(existing)].hasData()) {
        // 同一位置刚取过（例如 B 与 E 重合）：只加身份
        s.addRole(size_t(existing), role);
        s.enforceLimits();
        return;
    }
    // KV 必须正好到账本末尾（这一批 decode 成功之后才能取）；对不上就不取，免得存下错位的状态
    if (memory.seqPosMax(seq) + 1 != s.nextPos) {
        logf(FLR_LOG_WARN, "序列 %d：KV 末位置 %d 与账本 %d 对不上，不取检查点", seq, int(memory.seqPosMax(seq)), int(s.nextPos));
        return;
    }
    const auto started = std::chrono::steady_clock::now();
    Checkpoint checkpoint;
    checkpoint.n_items = at;
    checkpoint.pos = s.nextPos;
    checkpoint.partial = makeBlob(memory.partialState(seq));
    if (!checkpoint.hasData()) {
        logf(FLR_LOG_WARN, "序列 %d：在 %lld 项处取检查点失败", seq, static_cast<long long>(at));
        return;
    }
    const double mib = double(checkpoint.bytes()) / 1048576.0;
    s.insert(std::move(checkpoint), role);
    logf(FLR_LOG_NOTICE, "序列 %d：检查点 %s@%lld，%.2f MiB，%.1f ms（现有 %zu 个）", seq, rolesName(role).c_str(),
         static_cast<long long>(at), mib, elapsedMs(started), s.checkpoints.size());
}

void SequenceCache::append(const LedgerItem& item)
{
    State& s = *m_state;
    s.ledger.push_back(item);
    s.tokens += item.n_tokens;
    s.nextPos += item.n_pos;
    if (!s.marks.empty())
        s.materializeMarks();
}

void SequenceCache::rollback(MemoryOps& memory, int seq)
{
    State& s = *m_state;
    // 账本截到 KV 里实际有的（位置 0…posMax）：一项的起始位置 ≤ posMax 就算在 KV 里。媒体块只在 Media::eval 整块成功之后才进账本，
    // 账本里的媒体块都是完整的；M-RoPE 的图片块（Qwen-VL 系列）里单元的位置（t 维）都等于起始位置，块之后 seq_pos_max 只到起始位置，
    // 不能按「起始 + n_pos − 1 ≤ posMax」判断（那样整张图会被当成只算了一半丢掉，混合模型还得退检查点、重新编码；WP4 实测）。
    // 只进了一半的媒体块不在账本里：下面按「KV 比账本多」处理
    const llama_pos posMax = memory.seqPosMax(seq);
    size_t keep = 0;
    llama_pos end = 0;
    for (const LedgerItem& item : s.ledger) {
        if (end > posMax)
            break;
        end += item.n_pos;
        ++keep;
    }
    s.truncate(keep);
    if (posMax >= end && !memory.seqRemove(seq, end, -1)) {
        // KV 里比账本多（媒体块只算了一半）而模型不能截断：退到最近的检查点，再不行就清空（这次请求还要从头接着算，停点留着）
        const int best = s.latestRestorable(int64_t(keep));
        if (best >= 0 && memory.restorePartial(seq, s.checkpoints[size_t(best)].partial)
            && memory.seqRemove(seq, s.checkpoints[size_t(best)].pos, -1)) {
            s.truncate(size_t(s.checkpoints[size_t(best)].n_items));
        } else {
            memory.seqClear(seq);
            s.reset(false);
        }
    }
    // 这一批里别的部分失败了、这个序列正好停在一个还没取的停点上：现在取（KV 正好到这里）
    const int64_t at = int64_t(s.ledger.size());
    for (const Stop& stop : s.stops) {
        if (stop.at != at)
            continue;
        for (const Checkpoint::Kind kind : stop.kinds) {
            const int existing = s.find(at);
            if (existing < 0 || !(s.roles[size_t(existing)].roles & roleOf(kind)))
                takeCheckpoint(kind, memory, seq);
        }
    }
}

void SequenceCache::discardAfter(MemoryOps& memory, int seq, size_t limit)
{
    State& s = *m_state;
    // 评审修正：混合模型（Qwen3.5）上 llama_process 中止或出错时，llama 对失败那个 ubatch 调 seq_rm(seq, pos_min, -1)，
    // 可循环状态的尾 cell 在 apply() 里已经改到 ubatch 末尾、n_rs_seq = 0，seq_rm 返回 false，注意力部分也不删——
    // 那一段的注意力 cell 占着却没写过 K/V，循环状态缺了这一段，seq_pos_max 却到了 ubatch 末尾。rollback 按 seq_pos_max 对账会把
    // 这些没算过的 token 当成已在 KV 里（实测重发时复用 = 中止处 + 512，答错）。这一步之前的账本（limit 项）KV 是对的，但循环状态
    // 已经被改过，只能回到不晚于它的检查点：先恢复循环状态，再删掉注意力 KV 的尾巴（连同那些没算过的 cell）
    const int best = s.latestRestorable(int64_t(std::min(limit, s.ledger.size())));
    if (best >= 0) {
        const Checkpoint& checkpoint = s.checkpoints[size_t(best)];
        const int64_t items = checkpoint.n_items;
        if (memory.restorePartial(seq, checkpoint.partial) && memory.seqRemove(seq, checkpoint.pos, -1)) {
            s.truncate(size_t(items));
            logf(FLR_LOG_NOTICE, "序列 %d：推理失败后退回检查点 %lld 项（之后的 KV 不可信）", seq, static_cast<long long>(items));
            return;
        }
    }
    memory.seqClear(seq);
    s.reset(false);
    logf(FLR_LOG_NOTICE, "序列 %d：推理失败后清空（之前没有可退回的检查点）", seq);
}

void SequenceCache::clear()
{
    m_state->reset(true);
}

const std::vector<LedgerItem>& SequenceCache::ledger() const
{
    return m_state->ledger;
}

int64_t SequenceCache::nTokens() const
{
    return m_state->tokens;
}

llama_pos SequenceCache::nextPos() const
{
    return m_state->nextPos;
}

const std::vector<Checkpoint>& SequenceCache::checkpoints() const
{
    return m_state->checkpoints;
}

std::optional<std::string> SequenceCache::prefixFingerprint() const
{
    if (!m_state->prefixCheckpoint())
        return std::nullopt;
    return m_state->prefix;
}

bool SequenceCache::rollbackToPrefix(MemoryOps& memory, int seq)
{
    State& s = *m_state;
    const Checkpoint* checkpoint = s.prefixCheckpoint();
    if (!checkpoint)
        return false;
    const size_t prefixItems = size_t(checkpoint->n_items);
    s.stops.clear();
    s.marks.clear();
    if (s.ledger.size() == prefixItems)
        return true;
    const bool restored = (!checkpoint->hasData() || memory.restorePartial(seq, checkpoint->partial))
        && memory.seqRemove(seq, checkpoint->pos, -1);
    if (!restored) {
        // 可能已经动了循环状态：整条清空
        memory.seqClear(seq);
        s.reset(true);
        return false;
    }
    s.truncate(prefixItems);
    return true;
}

void SequenceCache::adopt(std::vector<LedgerItem> ledger, std::optional<std::string> prefix)
{
    State& s = *m_state;
    s.ledger = std::move(ledger);
    s.recount();
    ++s.generation;
    s.checkpoints.clear();
    s.roles.clear();
    s.stops.clear();
    s.marks.clear();
    s.prefix.clear();
    s.prefixItems = -1;
    if (prefix && !prefix->empty()) {
        // 指纹是「<项数>-<哈希>」：项数就是 P 的位置
        char* end = nullptr;
        const long long count = std::strtoll(prefix->c_str(), &end, 10);
        if (end && *end == '-' && count > 0 && count <= static_cast<long long>(s.ledger.size())) {
            s.prefix = *prefix;
            s.prefixItems = count;
        }
    }
}

void SequenceCache::fork(const SequenceCache& source, size_t keep)
{
    State& s = *m_state;
    const State& from = *source.m_state;
    keep = std::min(keep, from.ledger.size());
    s.ledger.assign(from.ledger.begin(), from.ledger.begin() + std::ptrdiff_t(keep));
    s.recount();
    ++s.generation;
    s.checkpoints.clear();
    s.roles.clear();
    s.stops.clear();
    s.marks.clear();
    // P 跟着走（指纹的项数在 keep 以内时）；检查点只带 n_items ≤ keep 的，身份原样
    if (!from.prefix.empty() && from.prefixItems > 0 && from.prefixItems <= int64_t(keep)) {
        s.prefix = from.prefix;
        s.prefixItems = from.prefixItems;
    } else {
        s.prefix.clear();
        s.prefixItems = -1;
    }
    for (size_t i = 0; i < from.checkpoints.size(); ++i) {
        if (from.checkpoints[i].n_items > int64_t(keep))
            continue;
        s.checkpoints.push_back(from.checkpoints[i]);
        s.roles.push_back(from.roles[i]);
    }
    s.serial = std::max(s.serial, from.serial);
    s.enforceLimits();
}

uint64_t SequenceCache::generation() const
{
    return m_state->generation;
}

void SequenceCache::addCheckpoint(Checkpoint checkpoint)
{
    State& s = *m_state;
    if (!s.policy.checkpoints || checkpoint.n_items <= 0 || checkpoint.n_items > int64_t(s.ledger.size())
        || checkpoint.pos != positionAfter(s.ledger, size_t(checkpoint.n_items)))
        return;
    const unsigned role = roleOf(checkpoint.kind);
    if (role == rolePrefix && checkpoint.n_items != s.prefixItems)
        return;
    s.insert(std::move(checkpoint), role);
}

} // namespace flr
