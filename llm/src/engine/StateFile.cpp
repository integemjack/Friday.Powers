// 状态文件、回合边界与前缀快照、flr_seq_* / flr_prefix_* 的主体（docs/LOCAL_INFERENCE.md §F.3；WP3 拥有）。
// 对应 Swift：ChatSession.saveTranscriptCache / savePrefixCheckpoint / init(transcriptCache:prefixCache:)（12d0450）。
//
// 文件（扩展名仍是 .bin，旁路 .json 归应用）：
//   "FRKV" | u32 version=2 | u32 header_len | header_json | u64 len | llama_state_seq_get_data(FLAGS_NONE)
//   [| 每个头里 "checkpoints" 的项：u64 len | PARTIAL_ONLY 状态]                     （小端；之后必须正好到文件尾）
//   header_json = {"model_sha","memory","n_tokens","pos_next","prefix","boundary":"B|P|full","data_bytes","data_xxh64",
//                  "checkpoints":[{"kind":"P","n_items","pos","bytes","xxh64"}]（可选）,"ledger":[token…, {"media","n_tokens","n_pos"}…]}
// 回合边界文件（B）在不可回退的模型上带着 P 的 PARTIAL 状态（约 50 MiB）：恢复后较早的历史被折叠时仍能回退到 P，
// == Swift 恢复会话时同时传入 prefixCache。读的时候什么都核对（魔数、版本、长度、账本与 n_tokens / pos_next、文件正好结束、
// 数据段与检查点的 XXH64），对不上一律当损坏（FLR_IO_ERROR，应用删掉文件、重新预填）；模型标识不符 → FLR_MODEL_MISMATCH。
// 版本 2（评审修正）：加了数据段校验——llama 恢复单序列循环状态时数据不对会 GGML_ASSERT（llama-memory-recurrent.cpp 的 DEBUG CHECK），
// 进程内会让整个应用退出，坏文件还会每次打开这段对话都崩一次；版本 1 的文件一律不认（也顺带作废评审第 1 条修好之前可能写坏的前缀快照）。
// 校验和对得上、llama 还是装不进去：多半是 KV 池被别的序列占着——先腾空闲序列再试，仍不行报 FLR_KV_FULL（应用留着文件，这次从头预填）。
// 文件读写在调用线程；拷状态 / 回退 / 恢复经 SequenceHost::withSequence 在推理线程上做（§E.1「写盘」）。
#include "internal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace flr {
namespace {

constexpr char stateMagic[4] = { 'F', 'R', 'K', 'V' };
/// 2：数据段与检查点带 XXH64（评审修正）
constexpr uint32_t stateVersion = 2;
/// 头的上限（64k token 的账本约 0.5 MB；再大就是坏文件）
constexpr uint64_t maxHeaderBytes = 64ull << 20;
/// modelFingerprint 哈希的长度
constexpr size_t fingerprintBytes = 1u << 20;

// MARK: SHA-256（FIPS 180-4；SDK 里没有 vendor/hash，自己写一个）

class Sha256 {
public:
    Sha256() = default;

    void update(const uint8_t* data, size_t size)
    {
        m_length += uint64_t(size);
        while (size > 0) {
            const size_t take = std::min(size, sizeof(m_buffer) - m_used);
            std::memcpy(m_buffer + m_used, data, take);
            m_used += take;
            data += take;
            size -= take;
            if (m_used == sizeof(m_buffer)) {
                block(m_buffer);
                m_used = 0;
            }
        }
    }

    std::array<uint8_t, 32> finish()
    {
        const uint64_t bits = m_length * 8;
        const uint8_t pad = 0x80;
        update(&pad, 1);
        const uint8_t zero = 0;
        while (m_used != 56)
            update(&zero, 1);
        uint8_t length[8];
        for (int i = 0; i < 8; ++i)
            length[i] = uint8_t(bits >> (56 - 8 * i));
        update(length, 8);
        std::array<uint8_t, 32> digest {};
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 4; ++j)
                digest[size_t(i * 4 + j)] = uint8_t(m_h[i] >> (24 - 8 * j));
        }
        return digest;
    }

private:
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t* p)
    {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
            0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
            0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = uint32_t(p[i * 4]) << 24 | uint32_t(p[i * 4 + 1]) << 16 | uint32_t(p[i * 4 + 2]) << 8 | uint32_t(p[i * 4 + 3]);
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = m_h[0], b = m_h[1], c = m_h[2], d = m_h[3], e = m_h[4], f = m_h[5], g = m_h[6], h = m_h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        m_h[0] += a;
        m_h[1] += b;
        m_h[2] += c;
        m_h[3] += d;
        m_h[4] += e;
        m_h[5] += f;
        m_h[6] += g;
        m_h[7] += h;
    }

    uint32_t m_h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    uint8_t m_buffer[64] = {};
    size_t m_used = 0;
    uint64_t m_length = 0;
};

} // namespace

// MARK: XXH64（数据段校验：几百 MB，SHA-256 太慢）

uint64_t xxh64(const void* bytes, size_t size, uint64_t seed)
{
    // xxHash 的 XXH64（Yann Collet，BSD 2-Clause；照 xxhash.h 的算法写，一次算完整块内存）
    const auto* data = static_cast<const uint8_t*>(bytes);
    constexpr uint64_t p1 = 11400714785074694791ull;
    constexpr uint64_t p2 = 14029467366897019727ull;
    constexpr uint64_t p3 = 1609587929392839161ull;
    constexpr uint64_t p4 = 9650029242287828579ull;
    constexpr uint64_t p5 = 2870177450012600261ull;
    const auto rotl = [](uint64_t x, int r) { return (x << r) | (x >> (64 - r)); };
    // 小端读取（文件格式就是小端；目标平台 x64 / arm64 都是小端，memcpy 编成一次读）
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    const auto read64 = [](const uint8_t* p) {
        uint64_t v = 0;
        for (int i = 7; i >= 0; --i)
            v = v << 8 | uint64_t(p[i]);
        return v;
    };
    const auto read32 = [](const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; };
#else
    const auto read64 = [](const uint8_t* p) {
        uint64_t v;
        std::memcpy(&v, p, sizeof(v));
        return v;
    };
    const auto read32 = [](const uint8_t* p) {
        uint32_t v;
        std::memcpy(&v, p, sizeof(v));
        return v;
    };
#endif
    const auto round = [&](uint64_t acc, uint64_t input) {
        acc += input * p2;
        acc = rotl(acc, 31);
        return acc * p1;
    };
    const auto merge = [&](uint64_t acc, uint64_t value) {
        acc ^= round(0, value);
        return acc * p1 + p4;
    };
    const uint8_t* p = data;
    const uint8_t* const end = data + size;
    uint64_t h = 0;
    if (size >= 32) {
        uint64_t v1 = seed + p1 + p2;
        uint64_t v2 = seed + p2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - p1;
        const uint8_t* const limit = end - 32;
        do {
            v1 = round(v1, read64(p));
            v2 = round(v2, read64(p + 8));
            v3 = round(v3, read64(p + 16));
            v4 = round(v4, read64(p + 24));
            p += 32;
        } while (p <= limit);
        h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        h = merge(h, v1);
        h = merge(h, v2);
        h = merge(h, v3);
        h = merge(h, v4);
    } else {
        h = seed + p5;
    }
    h += uint64_t(size);
    while (end - p >= 8) {
        h ^= round(0, read64(p));
        h = rotl(h, 27) * p1 + p4;
        p += 8;
    }
    if (end - p >= 4) {
        h ^= uint64_t(read32(p)) * p1;
        h = rotl(h, 23) * p2 + p3;
        p += 4;
    }
    while (p < end) {
        h ^= uint64_t(*p) * p5;
        h = rotl(h, 11) * p1;
        ++p;
    }
    h ^= h >> 33;
    h *= p2;
    h ^= h >> 29;
    h *= p3;
    h ^= h >> 32;
    return h;
}

namespace {

std::string xxh64Hex(const std::vector<uint8_t>& data)
{
    char text[17];
    std::snprintf(text, sizeof(text), "%016" PRIx64, xxh64(data.data(), data.size()));
    return text;
}

std::string hex(const uint8_t* data, size_t size)
{
    static const char digits[] = "0123456789abcdef";
    std::string text;
    text.reserve(size * 2);
    for (size_t i = 0; i < size; ++i) {
        text += digits[data[i] >> 4];
        text += digits[data[i] & 0xf];
    }
    return text;
}

// MARK: 小工具

std::filesystem::path fsPath(const std::string& utf8)
{
    return std::filesystem::u8path(utf8);
}

void putU32(std::string& out, uint32_t value)
{
    for (int i = 0; i < 4; ++i)
        out += char((value >> (8 * i)) & 0xff);
}

void putU64(std::string& out, uint64_t value)
{
    for (int i = 0; i < 8; ++i)
        out += char((value >> (8 * i)) & 0xff);
}

uint32_t getU32(const unsigned char* bytes)
{
    return uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 | uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
}

uint64_t getU64(const unsigned char* bytes)
{
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i)
        value = value << 8 | uint64_t(bytes[i]);
    return value;
}

const char* kindName(Checkpoint::Kind kind)
{
    switch (kind) {
    case Checkpoint::Kind::Prefix: return "P";
    case Checkpoint::Kind::Boundary: return "B";
    case Checkpoint::Kind::End: break;
    }
    return "E";
}

std::optional<Checkpoint::Kind> kindFromName(const std::string& name)
{
    if (name == "P")
        return Checkpoint::Kind::Prefix;
    if (name == "B")
        return Checkpoint::Kind::Boundary;
    if (name == "E")
        return Checkpoint::Kind::End;
    return std::nullopt;
}

/// 前缀指纹「<项数>-<哈希>」里的项数；不是这个格式为 -1
int64_t prefixItemsOf(const std::string& fingerprint)
{
    char* end = nullptr;
    const long long count = std::strtoll(fingerprint.c_str(), &end, 10);
    return end && end != fingerprint.c_str() && *end == '-' && count > 0 ? int64_t(count) : -1;
}

llama_pos positionAfter(const std::vector<LedgerItem>& items, size_t n)
{
    llama_pos pos = 0;
    for (size_t i = 0; i < n && i < items.size(); ++i)
        pos += items[i].n_pos;
    return pos;
}

void setCounts(StateHeader& header)
{
    header.n_tokens = 0;
    header.pos_next = 0;
    for (const LedgerItem& item : header.ledger) {
        header.n_tokens += item.n_tokens;
        header.pos_next += item.n_pos;
    }
}

/// seq 的 P 检查点（n_items == 前缀指纹的项数）；没有为 nullptr
const Checkpoint* prefixCheckpointOf(const SequenceCache& cache, const std::string& fingerprint)
{
    const int64_t items = prefixItemsOf(fingerprint);
    for (const Checkpoint& checkpoint : cache.checkpoints()) {
        if (checkpoint.kind == Checkpoint::Kind::Prefix && checkpoint.n_items == items)
            return &checkpoint;
    }
    return nullptr;
}

/// seq 在前 items 项处的完整状态：已经正好在那里就直接拷；否则拷到工作序列 → 回退（PART 直接截断，
/// 不可回退的先恢复 checkpoint 的 PARTIAL 状态）→ 拷 → 清空工作序列。回退不了返回空
std::vector<uint8_t> stateAt(const SequenceCache& cache, MemoryOps& memory, int seq, int workspaceSeq, const Checkpoint& checkpoint)
{
    if (checkpoint.n_items == int64_t(cache.ledger().size()))
        return memory.fullState(seq);
    memory.seqCopy(seq, workspaceSeq);
    std::vector<uint8_t> data;
    const bool rewound = (!checkpoint.hasData() ? memory.partialRemovable() : memory.restorePartial(workspaceSeq, checkpoint.partial))
        && memory.seqRemove(workspaceSeq, checkpoint.pos, -1);
    if (rewound)
        data = memory.fullState(workspaceSeq);
    memory.seqClear(workspaceSeq);
    return data;
}

std::string errorResult(const std::string& message)
{
    return json { { "error", json { { "message", message } } } }.dump();
}

json prefixValue(const std::string& prefix)
{
    return prefix.empty() ? json(nullptr) : json(prefix);
}

double elapsedMs(std::chrono::steady_clock::time_point since)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

uint64_t fileSize(const std::string& path)
{
    std::error_code error;
    const auto size = std::filesystem::file_size(fsPath(path), error);
    return error ? 0 : uint64_t(size);
}

bool validSequence(SequenceHost& host, int seq, std::string& result)
{
    if (seq >= 0 && seq < host.appSequences())
        return true;
    result = errorResult("序列号不对：" + std::to_string(seq));
    return false;
}

/// 文件头里的模型标识与运行时的比对
bool sameModel(SequenceHost& host, const StateHeader& header, std::string& result)
{
    if (!host.modelSha().empty() && header.model_sha == host.modelSha() && header.memory == host.memoryKind())
        return true;
    result = errorResult("状态文件属于别的模型（文件 " + header.model_sha + "，当前 " + host.modelSha() + "）");
    return false;
}

json headerJson(const StateSnapshot& snapshot)
{
    const StateHeader& header = snapshot.header;
    json value {
        { "model_sha", header.model_sha },
        { "memory", header.memory },
        { "n_tokens", header.n_tokens },
        { "pos_next", header.pos_next },
        { "prefix", header.prefix },
        { "boundary", header.boundary },
        { "data_bytes", uint64_t(snapshot.data.size()) },
        { "data_xxh64", xxh64Hex(snapshot.data) },
    };
    if (!snapshot.checkpoints.empty()) {
        json checkpoints = json::array();
        for (const Checkpoint& checkpoint : snapshot.checkpoints) {
            checkpoints.push_back({ { "kind", kindName(checkpoint.kind) }, { "n_items", checkpoint.n_items }, { "pos", checkpoint.pos },
                                    { "bytes", uint64_t(checkpoint.bytes()) },
                                    { "xxh64", checkpoint.partial ? xxh64Hex(*checkpoint.partial) : xxh64Hex(std::vector<uint8_t>()) } });
        }
        value["checkpoints"] = std::move(checkpoints);
    }
    json ledger = json::array();
    for (const LedgerItem& item : header.ledger) {
        if (item.isMedia())
            ledger.push_back({ { "media", item.media_id }, { "n_tokens", item.n_tokens }, { "n_pos", item.n_pos } });
        else
            ledger.push_back(item.token);
    }
    value["ledger"] = std::move(ledger);
    return value;
}

/// 头里一段数据（主状态或一个检查点）的描述：字节数与 XXH64
struct BlockInfo {
    uint64_t bytes = 0;
    std::string xxh64;
};

bool isHash(const std::string& text)
{
    return text.size() == 16 && std::all_of(text.begin(), text.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

/// header_json → StateHeader + 检查点的描述（数据之后再读，partial 先空着）+ 各段的字节数与校验和；不对返回 false。
/// available：头之后文件还剩的字节数——各段（u64 长度 + 数据）加起来必须正好是它（评审修正：先核对再分配，离谱的长度不会先要几十 GB）
bool parseHeader(const json& value, uint64_t available, StateHeader& header, std::vector<Checkpoint>& checkpoints, BlockInfo& data,
                 std::vector<BlockInfo>& blocks, std::string& error)
{
    if (!value.is_object()) {
        error = "头不是 JSON 对象";
        return false;
    }
    const auto text = [&](const char* key, std::string& out, bool required) {
        const auto it = value.find(key);
        if (it == value.end() || it->is_null()) {
            if (required)
                error = std::string("头里缺少 ") + key;
            return !required;
        }
        if (!it->is_string()) {
            error = std::string("头里的 ") + key + " 不是字符串";
            return false;
        }
        out = it->get<std::string>();
        return true;
    };
    if (!text("model_sha", header.model_sha, true) || !text("memory", header.memory, true) || !text("prefix", header.prefix, false)
        || !text("boundary", header.boundary, true))
        return false;
    if (header.boundary != "B" && header.boundary != "P" && header.boundary != "full") {
        error = "头里的 boundary 不对：" + header.boundary;
        return false;
    }
    const auto ledger = value.find("ledger");
    if (ledger == value.end() || !ledger->is_array() || ledger->empty()) {
        error = "头里没有账本";
        return false;
    }
    header.ledger.clear();
    header.ledger.reserve(ledger->size());
    for (const json& entry : *ledger) {
        if (entry.is_number_integer()) {
            const int64_t token = entry.get<int64_t>();
            if (token < 0 || token > INT32_MAX) {
                error = "账本里的 token 不对";
                return false;
            }
            header.ledger.push_back(LedgerItem::text(llama_token(token)));
        } else if (entry.is_object()) {
            LedgerItem item;
            const auto media = entry.find("media");
            const auto tokens = entry.find("n_tokens");
            const auto positions = entry.find("n_pos");
            if (media == entry.end() || !media->is_string() || media->get<std::string>().empty() || tokens == entry.end()
                || !tokens->is_number_integer() || positions == entry.end() || !positions->is_number_integer()) {
                error = "账本里的媒体块不对";
                return false;
            }
            const int64_t nTokens = tokens->get<int64_t>();
            const int64_t nPos = positions->get<int64_t>();
            if (nTokens < 1 || nTokens > INT32_MAX || nPos < 1 || nPos > INT32_MAX) {
                error = "账本里的媒体块不对";
                return false;
            }
            item.media_id = media->get<std::string>();
            item.n_tokens = int(nTokens);
            item.n_pos = int(nPos);
            header.ledger.push_back(std::move(item));
        } else {
            error = "账本里有不认识的项";
            return false;
        }
    }
    const auto number = [&](const char* key, int64_t& out) {
        const auto it = value.find(key);
        if (it == value.end() || !it->is_number_integer()) {
            error = std::string("头里缺少 ") + key;
            return false;
        }
        out = it->get<int64_t>();
        return true;
    };
    int64_t nTokens = 0;
    int64_t posNext = 0;
    if (!number("n_tokens", nTokens) || !number("pos_next", posNext))
        return false;
    StateHeader counted;
    counted.ledger = header.ledger;
    setCounts(counted);
    if (counted.n_tokens != nTokens || int64_t(counted.pos_next) != posNext) {
        error = "头里的 n_tokens / pos_next 与账本对不上";
        return false;
    }
    header.n_tokens = counted.n_tokens;
    header.pos_next = counted.pos_next;
    if (!header.prefix.empty()) {
        const int64_t items = prefixItemsOf(header.prefix);
        if (items <= 0 || items > int64_t(header.ledger.size())) {
            error = "头里的前缀指纹不对";
            return false;
        }
    }
    // 剩下的字节：各段都扣得开才往下（u64 长度 + 数据）
    uint64_t remaining = available;
    const auto take = [&remaining](uint64_t bytes) {
        if (remaining < 8 || bytes > remaining - 8)
            return false;
        remaining -= 8 + bytes;
        return true;
    };
    int64_t dataBytes = 0;
    if (!number("data_bytes", dataBytes) || dataBytes <= 0) {
        error = "头里的 data_bytes 不对";
        return false;
    }
    data.bytes = uint64_t(dataBytes);
    if (!text("data_xxh64", data.xxh64, true))
        return false;
    if (!isHash(data.xxh64)) {
        error = "头里的 data_xxh64 不对";
        return false;
    }
    if (!take(data.bytes)) {
        error = "状态数据比文件还长";
        return false;
    }
    checkpoints.clear();
    blocks.clear();
    if (const auto list = value.find("checkpoints"); list != value.end()) {
        if (!list->is_array()) {
            error = "头里的 checkpoints 不对";
            return false;
        }
        for (const json& entry : *list) {
            Checkpoint checkpoint;
            const auto kind = entry.is_object() ? kindFromName(entry.value("kind", std::string())) : std::nullopt;
            const int64_t items = entry.is_object() ? entry.value("n_items", int64_t(-1)) : -1;
            const int64_t pos = entry.is_object() ? entry.value("pos", int64_t(-1)) : -1;
            const int64_t bytes = entry.is_object() ? entry.value("bytes", int64_t(-1)) : -1;
            const std::string hash = entry.is_object() ? entry.value("xxh64", std::string()) : std::string();
            if (!kind || items <= 0 || items > int64_t(header.ledger.size()) || pos != int64_t(positionAfter(header.ledger, size_t(items)))
                || bytes <= 0 || !isHash(hash)) {
                error = "头里的检查点描述不对";
                return false;
            }
            if (!take(uint64_t(bytes))) {
                error = "检查点数据比文件还长";
                return false;
            }
            checkpoint.kind = *kind;
            checkpoint.n_items = items;
            checkpoint.pos = llama_pos(pos);
            checkpoints.push_back(std::move(checkpoint));
            blocks.push_back({ uint64_t(bytes), hash });
        }
    }
    if (remaining != 0) {
        error = "结尾有多余的数据";
        return false;
    }
    return true;
}

} // namespace

// MARK: - 指纹

std::string prefixFingerprint(const std::vector<LedgerItem>& items, int n)
{
    // FNV-1a 64 位
    uint64_t hash = 14695981039346656037ull;
    const auto feed = [&hash](const void* data, size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
    };
    const auto feedInt = [&feed](int32_t value) {
        const unsigned char bytes[4] = { static_cast<unsigned char>(value & 0xff), static_cast<unsigned char>((value >> 8) & 0xff),
                                         static_cast<unsigned char>((value >> 16) & 0xff), static_cast<unsigned char>((value >> 24) & 0xff) };
        feed(bytes, sizeof(bytes));
    };
    const size_t count = n < 0 ? 0 : std::min(size_t(n), items.size());
    for (size_t i = 0; i < count; ++i) {
        const LedgerItem& item = items[i];
        if (item.isMedia()) {
            feed(item.media_id.data(), item.media_id.size());
            feedInt(item.n_tokens);
            feedInt(item.n_pos);
        } else {
            feedInt(item.token);
        }
    }
    char text[64];
    std::snprintf(text, sizeof(text), "%zu-%016" PRIx64, count, hash);
    return text;
}

std::string modelFingerprint(const std::string& ggufPath)
{
    std::error_code error;
    const std::filesystem::path path = fsPath(ggufPath);
    const auto size = std::filesystem::file_size(path, error);
    if (error)
        return {};
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return {};
    std::vector<char> head(size_t(std::min<uint64_t>(uint64_t(size), fingerprintBytes)));
    in.read(head.data(), std::streamsize(head.size()));
    if (in.gcount() != std::streamsize(head.size()))
        return {};
    Sha256 sha;
    sha.update(reinterpret_cast<const uint8_t*>(head.data()), head.size());
    const std::array<uint8_t, 32> digest = sha.finish();
    return hex(digest.data(), digest.size()) + "-" + std::to_string(uint64_t(size));
}

// MARK: - 文件

bool writeStateFile(const std::string& path, const StateSnapshot& snapshot, std::string* error)
{
    const auto fail = [&](const std::string& message) {
        if (error)
            *error = message;
        std::error_code ignored;
        std::filesystem::remove(fsPath(path), ignored);
        return false;
    };
    if (snapshot.data.empty() || snapshot.header.ledger.empty())
        return fail("没有要存的状态");
    std::string head;
    head.append(stateMagic, sizeof(stateMagic));
    putU32(head, stateVersion);
    std::string header;
    try {
        header = headerJson(snapshot).dump();
    } catch (const std::exception& exception) {
        return fail(std::string("状态头生成失败：") + exception.what());
    }
    putU32(head, uint32_t(header.size()));
    head += header;
    putU64(head, uint64_t(snapshot.data.size()));

    std::ofstream out(fsPath(path), std::ios::binary | std::ios::trunc);
    if (!out)
        return fail("无法写入 " + path);
    out.write(head.data(), std::streamsize(head.size()));
    out.write(reinterpret_cast<const char*>(snapshot.data.data()), std::streamsize(snapshot.data.size()));
    for (const Checkpoint& checkpoint : snapshot.checkpoints) {
        std::string length;
        putU64(length, uint64_t(checkpoint.bytes()));
        out.write(length.data(), std::streamsize(length.size()));
        if (checkpoint.partial)
            out.write(reinterpret_cast<const char*>(checkpoint.partial->data()), std::streamsize(checkpoint.partial->size()));
    }
    out.close();
    if (!out)
        return fail("写入 " + path + " 失败（磁盘满了？）");
    return true;
}

std::optional<StateSnapshot> readStateFile(const std::string& path, bool headerOnly, std::string* error)
{
    const auto fail = [&](const std::string& message) -> std::optional<StateSnapshot> {
        if (error)
            *error = message;
        return std::nullopt;
    };
    std::error_code sizeError;
    const uint64_t total = uint64_t(std::filesystem::file_size(fsPath(path), sizeError));
    if (sizeError)
        return fail("找不到状态文件 " + path);
    std::ifstream in(fsPath(path), std::ios::binary);
    if (!in)
        return fail("无法读取状态文件 " + path);
    unsigned char prologue[12] = {};
    if (total < sizeof(prologue) || !in.read(reinterpret_cast<char*>(prologue), sizeof(prologue)))
        return fail("不是 Friday 的状态文件（太短）");
    if (std::memcmp(prologue, stateMagic, sizeof(stateMagic)) != 0) {
        // llama-server 的槽位文件：LLAMA_STATE_SEQ_MAGIC 'ggsq'（小端写成 "qsgg"）
        if (std::memcmp(prologue, "qsgg", 4) == 0 || std::memcmp(prologue, "ggsq", 4) == 0)
            return fail("旧的 llama-server 槽位文件（ggsq），进程内运行时不认");
        return fail("不是 Friday 的状态文件（魔数不对）");
    }
    const uint32_t version = getU32(prologue + 4);
    if (version != stateVersion)
        return fail("状态文件版本 " + std::to_string(version) + " 不认识");
    const uint64_t headerBytes = getU32(prologue + 8);
    uint64_t remaining = total - sizeof(prologue);
    if (headerBytes == 0 || headerBytes > maxHeaderBytes || headerBytes > remaining)
        return fail("状态文件损坏（头的长度不对）");
    std::string headerText(size_t(headerBytes), '\0');
    if (!in.read(headerText.data(), std::streamsize(headerBytes)))
        return fail("状态文件损坏（读头失败）");
    remaining -= headerBytes;

    StateSnapshot snapshot;
    BlockInfo dataInfo;
    std::vector<BlockInfo> blocks;
    std::vector<Checkpoint> checkpoints;
    try {
        std::string message;
        if (!parseHeader(json::parse(headerText), remaining, snapshot.header, checkpoints, dataInfo, blocks, message))
            return fail("状态文件损坏（" + message + "）");
    } catch (const std::exception& exception) {
        return fail(std::string("状态文件损坏（头不是合法的 JSON：") + exception.what() + "）");
    }
    if (headerOnly)
        return snapshot;

    // 各段的长度 parseHeader 已经按文件大小核对过（正好到文件尾），这里读出来再核对长度前缀与 XXH64：
    // 数据交给 llama 之前必须是存下时的样子（数据不对时它恢复循环状态会 GGML_ASSERT，评审修正）
    const auto readBlock = [&](const BlockInfo& info, std::vector<uint8_t>& out) -> const char* {
        unsigned char length[8];
        if (!in.read(reinterpret_cast<char*>(length), sizeof(length)) || getU64(length) != info.bytes)
            return "长度不对";
        out.resize(size_t(info.bytes));
        if (!in.read(reinterpret_cast<char*>(out.data()), std::streamsize(info.bytes)))
            return "不完整";
        if (xxh64Hex(out) != info.xxh64)
            return "校验和对不上";
        return nullptr;
    };
    if (const char* problem = readBlock(dataInfo, snapshot.data))
        return fail(std::string("状态文件损坏（状态数据") + problem + "）");
    for (size_t i = 0; i < checkpoints.size(); ++i) {
        std::vector<uint8_t> partial;
        if (const char* problem = readBlock(blocks[i], partial))
            return fail(std::string("状态文件损坏（检查点数据") + problem + "）");
        checkpoints[i].partial = makeBlob(std::move(partial));
    }
    snapshot.checkpoints = std::move(checkpoints);
    return snapshot;
}

// MARK: - 快照

std::optional<StateSnapshot> saveBoundaryState(SequenceCache& cache, MemoryOps& memory, int seq, int workspaceSeq)
{
    const std::vector<LedgerItem>& ledger = cache.ledger();
    if (ledger.empty())
        return std::nullopt;
    // 最近一个用户驱动边界 B（取最靠后的；PART 型是位置标记）
    const Checkpoint* boundary = nullptr;
    for (const Checkpoint& checkpoint : cache.checkpoints()) {
        if (checkpoint.kind == Checkpoint::Kind::Boundary && checkpoint.n_items <= int64_t(ledger.size())
            && (!boundary || checkpoint.n_items > boundary->n_items))
            boundary = &checkpoint;
    }
    StateSnapshot snapshot;
    size_t items = ledger.size();
    snapshot.header.boundary = "full";
    if (boundary) {
        snapshot.data = stateAt(cache, memory, seq, workspaceSeq, *boundary);
        if (!snapshot.data.empty()) {
            items = size_t(boundary->n_items);
            snapshot.header.boundary = "B";
        } else {
            logf(FLR_LOG_WARN, "序列 %d：回退到回合边界 B@%lld 失败，改存整条序列", seq, static_cast<long long>(boundary->n_items));
        }
    }
    if (snapshot.data.empty())
        snapshot.data = memory.fullState(seq);
    if (snapshot.data.empty())
        return std::nullopt;
    snapshot.header.ledger.assign(ledger.begin(), ledger.begin() + std::ptrdiff_t(items));
    setCounts(snapshot.header);
    if (const auto prefix = cache.prefixFingerprint(); prefix && prefixItemsOf(*prefix) <= int64_t(items)) {
        snapshot.header.prefix = *prefix;
        // 不可回退的模型带上 P 的 PARTIAL 状态：恢复后较早的历史被折叠时还能回退到 P（== Swift 恢复时的 prefixCache）
        const Checkpoint* checkpoint = prefixCheckpointOf(cache, *prefix);
        if (checkpoint && checkpoint->hasData() && checkpoint->n_items < int64_t(items))
            snapshot.checkpoints.push_back(*checkpoint);
    }
    return snapshot;
}

std::optional<StateSnapshot> savePrefixState(SequenceCache& cache, MemoryOps& memory, int seq, int workspaceSeq)
{
    const auto prefix = cache.prefixFingerprint();
    if (!prefix)
        return std::nullopt;
    const Checkpoint* checkpoint = prefixCheckpointOf(cache, *prefix);
    if (!checkpoint)
        return std::nullopt;
    StateSnapshot snapshot;
    snapshot.data = stateAt(cache, memory, seq, workspaceSeq, *checkpoint);
    if (snapshot.data.empty())
        return std::nullopt;
    snapshot.header.ledger.assign(cache.ledger().begin(), cache.ledger().begin() + std::ptrdiff_t(checkpoint->n_items));
    setCounts(snapshot.header);
    snapshot.header.prefix = *prefix;
    snapshot.header.boundary = "P";
    return snapshot;
}

bool restoreState(SequenceCache& cache, MemoryOps& memory, int seq, const StateSnapshot& snapshot)
{
    const StateHeader& header = snapshot.header;
    const auto fail = [&] {
        memory.seqClear(seq);
        cache.clear();
        return false;
    };
    if (header.ledger.empty() || snapshot.data.empty() || !memory.restoreFull(seq, snapshot.data))
        return fail();
    // 纯文本的账本：KV 的末位置必须正好对上（媒体块的位置另算，不核对）
    const bool textOnly = std::none_of(header.ledger.begin(), header.ledger.end(), [](const LedgerItem& item) { return item.isMedia(); });
    if (textOnly && memory.seqPosMax(seq) != header.pos_next - 1)
        return fail();
    cache.adopt(header.ledger, header.prefix.empty() ? std::nullopt : std::optional<std::string>(header.prefix));
    for (const Checkpoint& checkpoint : snapshot.checkpoints)
        cache.addCheckpoint(checkpoint);
    // 恢复点本身记成检查点：前缀快照 → P；回合边界 → B（接下来的请求在它之后分歧时不用退到更早）
    if (header.boundary == "P")
        cache.takeCheckpoint(Checkpoint::Kind::Prefix, memory, seq);
    else if (header.boundary == "B")
        cache.takeCheckpoint(Checkpoint::Kind::Boundary, memory, seq);
    // PART 型：P 只要位置
    if (memory.partialRemovable() && !header.prefix.empty() && !cache.prefixFingerprint()) {
        Checkpoint mark;
        mark.kind = Checkpoint::Kind::Prefix;
        mark.n_items = prefixItemsOf(header.prefix);
        mark.pos = positionAfter(header.ledger, size_t(mark.n_items));
        cache.addCheckpoint(std::move(mark));
    }
    return true;
}

// MARK: - flr_seq_* / flr_prefix_* 的主体

namespace {

const char* const kvFullMessage = "状态装不进 KV 池（被正在处理的请求占着，或加载参数与存档时不同）：这次从头预填，文件留着";

/// 推理线程（job 里）：恢复一份已核对过校验和的快照。装不进去多半是统一 KV 池被别的序列占着：腾一个空闲序列再试，
/// 腾不出来为止（评审修正；以前直接报 FLR_IO_ERROR，应用把好好的对话缓存删了）
bool restoreIntoPool(SequenceHost& host, SequenceCache& cache, MemoryOps& memory, int seq, const StateSnapshot& snapshot)
{
    while (true) {
        if (restoreState(cache, memory, seq, snapshot))
            return true;
        if (!host.evictIdleSequence(seq))
            return false;
    }
}

} // namespace

int seqSave(SequenceHost& host, int seq, const std::string& path, std::string& result)
{
    const auto started = std::chrono::steady_clock::now();
    if (!validSequence(host, seq, result))
        return FLR_INVALID_REQUEST;
    if (host.modelSha().empty()) {
        result = errorResult("模型标识未知，不能存状态");
        return FLR_INTERNAL;
    }
    std::optional<StateSnapshot> snapshot;
    bool empty = false;
    const int code = host.withSequence(seq, [&](SequenceCache& cache, MemoryOps& memory) {
        empty = cache.ledger().empty();
        if (!empty)
            snapshot = saveBoundaryState(cache, memory, seq, host.workspaceSequence());
        return int(FLR_OK);
    });
    if (code != FLR_OK) {
        result = errorResult("运行时正在卸载");
        return code;
    }
    if (empty) {
        result = errorResult("序列是空的，没有可存的状态");
        return FLR_NOT_READY;
    }
    if (!snapshot) {
        result = errorResult("拷贝序列状态失败");
        return FLR_INTERNAL;
    }
    const double copyMs = elapsedMs(started);
    snapshot->header.model_sha = host.modelSha();
    snapshot->header.memory = host.memoryKind();
    std::string error;
    if (!writeStateFile(path, *snapshot, &error)) {
        result = errorResult(error);
        return FLR_IO_ERROR;
    }
    const uint64_t bytes = fileSize(path);
    const double ms = elapsedMs(started);
    result = json {
        { "n_tokens", snapshot->header.n_tokens },
        { "bytes", bytes },
        { "ms", ms },
        { "prefix", prefixValue(snapshot->header.prefix) },
        { "boundary", snapshot->header.boundary },
        { "state_bytes", uint64_t(snapshot->data.size()) },
    }.dump();
    logf(FLR_LOG_NOTICE, "序列 %d：存回合边界状态（%s，%lld token，%.1f MiB，拷贝 %.0f ms，共 %.0f ms）", seq, snapshot->header.boundary.c_str(),
         static_cast<long long>(snapshot->header.n_tokens), double(bytes) / 1048576.0, copyMs, ms);
    return FLR_OK;
}

int seqRestore(SequenceHost& host, int seq, const std::string& path, std::string& result)
{
    const auto started = std::chrono::steady_clock::now();
    if (!validSequence(host, seq, result))
        return FLR_INVALID_REQUEST;
    // 先只读头：别的模型的文件（可能 1 GB 多）不用整个读进来
    std::string error;
    const std::optional<StateSnapshot> head = readStateFile(path, true, &error);
    if (!head) {
        result = errorResult(error);
        return FLR_IO_ERROR;
    }
    if (!sameModel(host, head->header, result))
        return FLR_MODEL_MISMATCH;
    std::optional<StateSnapshot> snapshot = readStateFile(path, false, &error);
    if (!snapshot) {
        result = errorResult(error);
        return FLR_IO_ERROR;
    }
    if (!sameModel(host, snapshot->header, result))
        return FLR_MODEL_MISMATCH;
    const double readMs = elapsedMs(started);
    bool restored = false;
    const int code = host.withSequence(seq, [&](SequenceCache& cache, MemoryOps& memory) {
        restored = restoreIntoPool(host, cache, memory, seq, *snapshot);
        return int(FLR_OK);
    });
    if (code != FLR_OK) {
        result = errorResult("运行时正在卸载");
        return code;
    }
    if (!restored) {
        // 数据按校验和核对过：装不进去是 KV 池被正在处理的请求占着（或加载参数与存档时不同）——文件留着，这次从头预填（评审修正）
        result = errorResult(kvFullMessage);
        return FLR_KV_FULL;
    }
    const double ms = elapsedMs(started);
    result = json {
        { "n_tokens", snapshot->header.n_tokens },
        { "ms", ms },
        { "prefix", prefixValue(snapshot->header.prefix) },
        { "boundary", snapshot->header.boundary },
    }.dump();
    logf(FLR_LOG_NOTICE, "序列 %d：恢复状态（%s，%lld token，读文件 %.0f ms，共 %.0f ms）", seq, snapshot->header.boundary.c_str(),
         static_cast<long long>(snapshot->header.n_tokens), readMs, ms);
    return FLR_OK;
}

int seqErase(SequenceHost& host, int seq)
{
    if (seq < 0 || seq >= host.appSequences())
        return FLR_INVALID_REQUEST;
    return host.withSequence(seq, [seq](SequenceCache& cache, MemoryOps& memory) {
        memory.seqClear(seq);
        cache.clear();
        return int(FLR_OK);
    });
}

int prefixSave(SequenceHost& host, int seq, const std::string& path, std::string& result)
{
    const auto started = std::chrono::steady_clock::now();
    if (!validSequence(host, seq, result))
        return FLR_INVALID_REQUEST;
    if (host.modelSha().empty()) {
        result = errorResult("模型标识未知，不能存状态");
        return FLR_INTERNAL;
    }
    std::optional<StateSnapshot> snapshot;
    bool available = false;
    const int code = host.withSequence(seq, [&](SequenceCache& cache, MemoryOps& memory) {
        available = cache.prefixFingerprint().has_value();
        if (available)
            snapshot = savePrefixState(cache, memory, seq, host.workspaceSequence());
        return int(FLR_OK);
    });
    if (code != FLR_OK) {
        result = errorResult("运行时正在卸载");
        return code;
    }
    if (!available) {
        result = errorResult("这个序列没有前缀快照");
        return FLR_NOT_READY;
    }
    if (!snapshot) {
        result = errorResult("拷贝前缀状态失败");
        return FLR_INTERNAL;
    }
    snapshot->header.model_sha = host.modelSha();
    snapshot->header.memory = host.memoryKind();
    std::string error;
    if (!writeStateFile(path, *snapshot, &error)) {
        result = errorResult(error);
        return FLR_IO_ERROR;
    }
    const uint64_t bytes = fileSize(path);
    const double ms = elapsedMs(started);
    result = json {
        { "n_tokens", snapshot->header.n_tokens },
        { "bytes", bytes },
        { "ms", ms },
        { "prefix", snapshot->header.prefix },
    }.dump();
    logf(FLR_LOG_NOTICE, "序列 %d：存前缀快照 %s（%lld token，%.1f MiB，%.0f ms）", seq, snapshot->header.prefix.c_str(),
         static_cast<long long>(snapshot->header.n_tokens), double(bytes) / 1048576.0, ms);
    return FLR_OK;
}

int prefixRestore(SequenceHost& host, int seq, const std::string& path, std::string& result)
{
    const auto started = std::chrono::steady_clock::now();
    if (!validSequence(host, seq, result))
        return FLR_INVALID_REQUEST;
    std::string error;
    const std::optional<StateSnapshot> head = readStateFile(path, true, &error);
    if (!head) {
        result = errorResult(error);
        return FLR_IO_ERROR;
    }
    if (!sameModel(host, head->header, result))
        return FLR_MODEL_MISMATCH;
    if (head->header.boundary != "P" || head->header.prefix.empty()) {
        result = errorResult("不是前缀快照文件");
        return FLR_IO_ERROR;
    }
    const std::string& prefix = head->header.prefix;
    // 序列里已经是同一个前缀（通常是上一段对话，系统提示一样）：回退到 P，不读盘
    bool fromMemory = false;
    int64_t tokens = 0;
    int code = host.withSequence(seq, [&](SequenceCache& cache, MemoryOps& memory) {
        if (cache.prefixFingerprint() == prefix && cache.rollbackToPrefix(memory, seq)) {
            fromMemory = true;
            tokens = cache.nTokens();
        }
        return int(FLR_OK);
    });
    if (code != FLR_OK) {
        result = errorResult("运行时正在卸载");
        return code;
    }
    if (!fromMemory) {
        std::optional<StateSnapshot> snapshot = readStateFile(path, false, &error);
        if (!snapshot) {
            result = errorResult(error);
            return FLR_IO_ERROR;
        }
        bool restored = false;
        code = host.withSequence(seq, [&](SequenceCache& cache, MemoryOps& memory) {
            restored = restoreIntoPool(host, cache, memory, seq, *snapshot);
            return int(FLR_OK);
        });
        if (code != FLR_OK) {
            result = errorResult("运行时正在卸载");
            return code;
        }
        if (!restored) {
            result = errorResult(kvFullMessage);
            return FLR_KV_FULL;
        }
        tokens = snapshot->header.n_tokens;
    }
    const double ms = elapsedMs(started);
    result = json {
        { "n_tokens", tokens },
        { "ms", ms },
        { "prefix", prefix },
        { "from", fromMemory ? "memory" : "file" },
    }.dump();
    logf(FLR_LOG_NOTICE, "序列 %d：从前缀 %s 开始（%s，%lld token，%.0f ms）", seq, prefix.c_str(), fromMemory ? "序列里已有，回退" : "读文件",
         static_cast<long long>(tokens), ms);
    return FLR_OK;
}

} // namespace flr
