// 看图 / 听音（mtmd，docs/LOCAL_INFERENCE.md §B.8；WP4 拥有）。
// 请求里的 image_url / input_audio / input_video（data URL 或纯 base64）→ 字节 → mtmd_helper_bitmap_init_from_buf
// （块 id = 文件字节的 SHA-256，同一张图在不同请求里相同）→ mtmd_tokenize（带随机标记的提示 → 文本 token + 媒体块）
// → 账本项（媒体块记 id、KV 单元数、位置数；M-RoPE 的图片块位置数 = max(宽, 高)，远小于 token 数）
// → 推理线程上 mtmd_encode_chunk + mtmd_helper_decode_image_chunk 编码并预填进序列。
// 照着 llama.cpp b11294 的 llama-server 写（MIT）：server-common.cpp 的 handle_media / oaicompat_content_load_media /
// process_mtmd_prompt，server-context.cpp 的 load_model（mmproj 参数、fit 预留）与 process_mtmd_chunk。差别：
//   - 不下载 http(s)、不读 file://（应用只发 data URL）；base64 严格校验（server 遇到非法字符静默截断）；
//   - 一次编码一个媒体块（server 用 mtmd_batch 把同尺寸的几张图一起编码；Qwen-VL 系列的投影不支持批量，效果相同）；
//   - 进程内没有崩溃隔离（§E.6）：会让 mtmd 触发 GGML_ASSERT 的输入（n_batch 越界、seq 非法、太短的音频）先在这里挡掉。
#include "internal.h"
#include "RuntimeLog.h"

#include "ggml-backend.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <random>
#include <system_error>
#include <utility>

namespace flr {
namespace {

using Clock = std::chrono::steady_clock;

double millisecondsSince(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool startsWith(const std::string& text, const char* prefix)
{
    const size_t length = std::strlen(prefix);
    return text.size() >= length && text.compare(0, length, prefix) == 0;
}

bool endsWith(const std::string& text, const char* suffix)
{
    const size_t length = std::strlen(suffix);
    return text.size() >= length && text.compare(text.size() - length, length, suffix) == 0;
}

/// 放进错误信息的片段：太长的截断（data URL 的头可能带着整段数据）。只在 UTF-8 字符边界上截——
/// 半个字符进了 result_json，nlohmann 的 dump 会抛异常
std::string excerpt(const std::string& text, size_t limit = 64)
{
    if (text.size() <= limit)
        return text;
    size_t cut = limit;
    while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80)
        --cut;
    return text.substr(0, cut) + "…";
}

// MARK: - base64

int base64Value(unsigned char c)
{
    if (c >= 'A' && c <= 'Z')
        return c - 'A';
    if (c >= 'a' && c <= 'z')
        return c - 'a' + 26;
    if (c >= '0' && c <= '9')
        return c - '0' + 52;
    if (c == '+' || c == '-')
        return 62;
    if (c == '/' || c == '_')
        return 63;
    return -1;
}

/// text[begin…] 解 base64（标准与 URL 安全两种字母表；跳过空白；'=' 之后只能再有 '=' 或空白；可以不带补齐）。
/// 不合法返回 false（server 的 base64_decode 遇到非法字符静默截断，图片会被截成半张）
bool base64Decode(const std::string& text, size_t begin, std::vector<uint8_t>& out)
{
    out.clear();
    out.reserve((text.size() - std::min(begin, text.size())) / 4 * 3 + 3);
    uint32_t buffer = 0;
    int bits = 0;
    bool padding = false;
    for (size_t i = begin; i < text.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t')
            continue;
        if (c == '=') {
            padding = true;
            continue;
        }
        const int value = base64Value(c);
        if (padding || value < 0)
            return false;
        buffer = (buffer << 6) | uint32_t(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(uint8_t(buffer >> bits));
            buffer &= (1u << bits) - 1;
        }
    }
    // 最后只剩一个字符（6 位）凑不出一个字节
    return bits < 6;
}

// MARK: - 请求里的媒体部件（== server-common.cpp oaicompat_content_load_media）

enum class PartKind { Text, Image, Audio, Video, Unsupported };

PartKind partKind(const std::string& type)
{
    if (type == "text")
        return PartKind::Text;
    if (type == "image_url")
        return PartKind::Image;
    if (type == "input_audio")
        return PartKind::Audio;
    if (type == "input_video" || type == "video_url")
        return PartKind::Video;
    return PartKind::Unsupported;
}

/// 部件里的地址：image_url.url；input_audio / input_video / video_url 的 data，其次 url（== server）。
/// 也收直接写成字符串的（"image_url":"data:…"，有的客户端这么发）
std::string partURL(const json& part, const std::string& type, PartKind kind)
{
    const auto field = part.find(type);
    if (field == part.end())
        return {};
    if (field->is_string())
        return field->get<std::string>();
    if (!field->is_object())
        return {};
    if (kind == PartKind::Image)
        return field->value("url", std::string());
    const std::string data = field->value("data", std::string());
    return data.empty() ? field->value("url", std::string()) : data;
}

/// data URL / 纯 base64 → 字节（== server-common.cpp handle_media，去掉下载 http 与读 file:// 两支）
int decodeURL(const std::string& url, std::vector<uint8_t>& out, std::string& error)
{
    if (url.empty()) {
        error = "媒体部件里没有数据（缺少 url / data）";
        return FLR_INVALID_REQUEST;
    }
    if (startsWith(url, "http://") || startsWith(url, "https://")) {
        error = "不支持网络地址的媒体（" + excerpt(url) + "），运行时不下载：请先存成文件、以 data URL 发送";
        return FLR_INVALID_REQUEST;
    }
    if (startsWith(url, "file://")) {
        error = "不支持 file:// 地址的媒体，请以 data URL 发送";
        return FLR_INVALID_REQUEST;
    }
    size_t begin = 0;
    if (startsWith(url, "data:")) {
        const size_t comma = url.find(',');
        if (comma == std::string::npos) {
            error = "data URL 格式不对（没有逗号）";
            return FLR_INVALID_REQUEST;
        }
        const std::string header = url.substr(0, comma);
        if (!startsWith(header, "data:image/") && !startsWith(header, "data:video/") && !startsWith(header, "data:audio/")) {
            error = "不支持的 data URL 类型：" + excerpt(header);
            return FLR_INVALID_REQUEST;
        }
        if (!endsWith(header, "base64")) {
            error = "data URL 必须是 base64 编码：" + excerpt(header);
            return FLR_INVALID_REQUEST;
        }
        begin = comma + 1;
    }
    if (!base64Decode(url, begin, out)) {
        error = "媒体数据不是合法的 base64";
        return FLR_INVALID_REQUEST;
    }
    if (out.empty()) {
        error = "媒体数据是空的";
        return FLR_INVALID_REQUEST;
    }
    return FLR_OK;
}

/// == mtmd-helper.cpp 的 is_audio_file（WAV / MP3 / FLAC 的文件头）：模型不能听音频时，助手函数对这类数据只回一个空指针，
/// 这里先认出来，好报「不支持」而不是「解不开」
bool looksLikeAudio(const std::vector<uint8_t>& data)
{
    if (data.size() < 12)
        return false;
    const auto* bytes = data.data();
    const bool wav = std::memcmp(bytes, "RIFF", 4) == 0 && std::memcmp(bytes + 8, "WAVE", 4) == 0;
    const bool mp3 = std::memcmp(bytes, "ID3", 3) == 0 || (bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0);
    const bool flac = std::memcmp(bytes, "fLaC", 4) == 0;
    return wav || mp3 || flac;
}

size_t countOccurrences(const std::string& text, const std::string& needle)
{
    if (needle.empty())
        return 0;
    size_t count = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size()))
        ++count;
    return count;
}

/// == server-common.cpp get_media_marker：每个运行时一个随机标记，用户文字里碰巧出现也不会被当成图片位置
std::string randomMarker()
{
    static const char characters[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::random_device device;
    std::mt19937 generator(device());
    std::string suffix(32, ' ');
    for (char& c : suffix)
        c = characters[generator() % (sizeof(characters) - 1)];
    return "<__media_" + suffix + "__>";
}

/// mtmd 的参数（== server-context.cpp load_model：1027-1042）。marker 要活到 mtmd_init_from_file / mtmd_get_memory_usage 返回
mtmd_context_params contextParams(const common_params& params, const std::string& marker)
{
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = params.mmproj_use_gpu;
    mparams.device = params.mmproj_device;
    mparams.print_timings = false;
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.flash_attn_type = params.flash_attn_type;
    mparams.warmup = params.warmup;
    mparams.image_min_tokens = params.image_min_tokens;
    mparams.image_max_tokens = params.image_max_tokens;
    mparams.batch_max_tokens = params.mtmd_batch_max_tokens;
    mparams.media_marker = marker.c_str();
    return mparams;
}

/// UTF-8 路径的文件是否存在（Windows 上 std::filesystem::path(std::string) 会按 ANSI 代码页解释）
bool fileExists(const std::string& utf8Path)
{
    std::error_code error;
#if defined(_WIN32)
    const std::filesystem::path path = std::filesystem::u8path(utf8Path);
#else
    const std::filesystem::path path(utf8Path);
#endif
    return std::filesystem::is_regular_file(path, error);
}

std::string fileName(const std::string& utf8Path)
{
    const size_t slash = utf8Path.find_last_of("/\\");
    return slash == std::string::npos ? utf8Path : utf8Path.substr(slash + 1);
}

/// 这个视觉投影最多出多少个图片 token（评审修正）：拿一张很大的灰图分一次词——mtmd 按投影自己的上限（或 image_max_tokens）把图缩下来，
/// 出来的就是上限。图片块不止一个（切片的投影）时取最大的；分不了词返回 -1
int probeMaxImageTokens(mtmd_context* context, const std::string& marker)
{
    constexpr uint32_t side = 2048;
    const std::vector<unsigned char> pixels(size_t(side) * side * 3, 128);
    const mtmd::bitmap bitmap(side, side, pixels.data());
    if (!bitmap.ptr)
        return -1;
    const mtmd::input_chunks chunks(mtmd_input_chunks_init());
    const mtmd_input_text text { marker.data(), marker.size(), false, true };
    const mtmd_bitmap* pointers[1] = { bitmap.ptr.get() };
    if (mtmd_tokenize(context, chunks.ptr.get(), &text, pointers, 1) != 0)
        return -1;
    int largest = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
        if (mtmd_input_chunk_get_type(chunks[i]) == MTMD_INPUT_CHUNK_TYPE_IMAGE)
            largest = std::max(largest, int(mtmd_input_chunk_get_n_tokens(chunks[i])));
    }
    return largest;
}

// MARK: - 加载 mmproj 时接住 mtmd 的错误日志

/// mtmd_init_from_file 失败只回空指针，原因（「mmproj 与模型的 n_embd 不符，可能用错了 mmproj」这类）只进日志。
/// 加载期间把 mtmd 的日志先经这里过一遍：照常转给 flr::log，同时记下最后一条错误，加载完 installLibraryLogging 换回去
struct LoadLogCapture {
    std::mutex mutex;
    std::string lastError;
    int lastLevel = FLR_LOG_INFO;
};

LoadLogCapture& loadLogCapture()
{
    static LoadLogCapture capture;
    return capture;
}

void captureLoadLog(enum ggml_log_level level, const char* text, void*)
{
    LoadLogCapture& capture = loadLogCapture();
    int mapped = FLR_LOG_INFO;
    {
        std::lock_guard<std::mutex> lock(capture.mutex);
        switch (level) {
        case GGML_LOG_LEVEL_DEBUG: mapped = FLR_LOG_DEBUG; break;
        case GGML_LOG_LEVEL_WARN: mapped = FLR_LOG_WARN; break;
        case GGML_LOG_LEVEL_ERROR: mapped = FLR_LOG_ERROR; break;
        case GGML_LOG_LEVEL_CONT: mapped = capture.lastLevel; break;
        default: mapped = FLR_LOG_INFO; break;
        }
        if (level != GGML_LOG_LEVEL_CONT) {
            capture.lastLevel = mapped;
            if (mapped == FLR_LOG_ERROR)
                capture.lastError.clear();
        }
        if (mapped == FLR_LOG_ERROR && text)
            capture.lastError += text;
    }
    if (text)
        flr::log(mapped, text);
}

/// 「mtmd_init_from_file: error: mismatch between …\nhint: …\n」→「mismatch between … hint: …」
std::string cleanLoadError(std::string text)
{
    const size_t marker = text.find("error: ");
    if (marker != std::string::npos && marker < 64)
        text = text.substr(marker + 7);
    for (char& c : text) {
        if (c == '\n' || c == '\r')
            c = ' ';
    }
    while (!text.empty() && text.back() == ' ')
        text.pop_back();
    return excerpt(text, 400);
}

// MARK: - mtmd 上的实现

class MtmdMedia final : public Media {
public:
    MtmdMedia(mtmd::context_ptr context, std::string marker, const common_params& params, int nUbatch)
        : m_context(std::move(context))
        , m_marker(std::move(marker))
        , m_ffmpegDirectory(params.video_ffmpeg_bin_dir)
        , m_ubatch(nUbatch)
    {
        m_options = mtmd_helper_init_opt_default();
        m_options.video_params.fps_target = params.video_fps;
        m_options.video_params.timestamp_interval_ms = params.video_timestamp_interval_ms;
        m_options.video_params.ffmpeg_bin_dir = m_ffmpegDirectory.empty() ? nullptr : m_ffmpegDirectory.c_str();
        m_vision = mtmd_support_vision(m_context.get());
        m_audio = mtmd_support_audio(m_context.get());
        m_video = mtmd_helper_support_video(m_context.get());
        m_mrope = mtmd_decode_use_mrope(m_context.get());
    }
    // m_options 里指着 m_ffmpegDirectory 的缓冲区：不能拷贝 / 移动
    MtmdMedia(const MtmdMedia&) = delete;
    MtmdMedia& operator=(const MtmdMedia&) = delete;

    const std::string& marker() const override { return m_marker; }
    bool supportsImages() const override { return m_vision; }
    bool supportsAudio() const override { return m_audio; }

    bool supportsVideo() const { return m_video; }
    bool usesMrope() const { return m_mrope; }

    int tokenize(const std::string& prompt, const std::vector<MediaInput>& inputs, Prepared& out, std::string* error) override;
    int eval(const Prepared& p, int item, llama_context* ctx, int seq, llama_pos pos, int n_batch, llama_pos* nextPos) override;

private:
    int fail(int code, const std::string& message, std::string* error) const
    {
        if (error)
            *error = message;
        logf(code == FLR_INTERNAL ? FLR_LOG_ERROR : FLR_LOG_WARN, "看图：%s", message.c_str());
        return code;
    }

    mtmd::context_ptr m_context;
    std::string m_marker;
    std::string m_ffmpegDirectory;
    /// 上下文的 n_ubatch（0 = 不核对）：非因果注意力的媒体块不能超过它（评审修正）
    int m_ubatch = 0;
    mtmd_helper_init_opt m_options {};
    bool m_vision = false;
    bool m_audio = false;
    bool m_video = false;
    bool m_mrope = false;
};

int MtmdMedia::tokenize(const std::string& prompt, const std::vector<MediaInput>& inputs, Prepared& out, std::string* error)
{
    try {
        const auto started = Clock::now();
        // 1. 字节 → 位图（== server-common.cpp process_mtmd_prompt）。块 id 是文件字节的 SHA-256（mtmd 助手函数算的），
        //    同一张图在下一轮请求里 id 相同 → 账本按项比较时整块复用（§B.8）
        mtmd::bitmaps bitmaps;
        std::vector<mtmd_helper::video_ptr> videos;   // 视频的懒加载位图在 mtmd_tokenize 里展开，展开完才能释放
        for (size_t i = 0; i < inputs.size(); ++i) {
            const MediaInput& input = inputs[i];
            const std::string ordinal = "第 " + std::to_string(i + 1) + " 个媒体";
            if (input.data.empty())
                return fail(FLR_INVALID_REQUEST, ordinal + "没有数据", error);
            if (input.kind == "video" && !m_video)
                return fail(FLR_UNSUPPORTED_MEDIA, "当前版本不能直接看视频（没有内置视频解码），请先抽帧成图片再发", error);
            const bool audio = input.kind == "audio" || looksLikeAudio(input.data);
            if (audio && !m_audio)
                return fail(FLR_UNSUPPORTED_MEDIA, "当前模型不能听音频", error);
            if (!audio && !m_vision)
                return fail(FLR_UNSUPPORTED_MEDIA, "当前模型不能看图（视觉投影 mmproj 不含视觉部分）", error);
            const mtmd_helper_bitmap_wrapper wrapped
                = mtmd_helper_bitmap_init_from_buf(m_context.get(), input.data.data(), input.data.size(), false, m_options);
            if (wrapped.video_ctx)
                videos.emplace_back(wrapped.video_ctx);
            if (!wrapped.bitmap) {
                return fail(FLR_INVALID_REQUEST,
                            ordinal + (audio ? "解不开（不是能识别的音频格式：WAV / MP3 / FLAC）"
                                             : "解不开（不是能识别的图片格式：JPEG / PNG / BMP / GIF…）"),
                            error);
            }
            bitmaps.entries.emplace_back(wrapped.bitmap);
            // mtmd 对只有一个采样点的音频直接 GGML_ASSERT（mtmd.cpp add_media），进程内会让整个应用退出
            if (mtmd_bitmap_is_audio(wrapped.bitmap) && mtmd_bitmap_get_n_bytes(wrapped.bitmap) < 160 * sizeof(float))
                return fail(FLR_INVALID_REQUEST, ordinal + "太短（音频不到 160 个采样点）", error);
        }
        // 2. 标记数要与媒体数一致（模板丢掉了某条带图的消息时就会对不上；mtmd_tokenize 也会查，这里给个明确的说法）。
        //    视频在 mtmd_tokenize 里展开成若干帧，标记仍是一个
        const size_t markers = countOccurrences(prompt, m_marker);
        if (markers != inputs.size()) {
            return fail(FLR_INVALID_REQUEST,
                        "提示里的媒体标记（" + std::to_string(markers) + " 个）与媒体（" + std::to_string(inputs.size()) + " 个）对不上",
                        error);
        }
        const double decodeMs = millisecondsSince(started);

        // 3. 带标记的提示 + 位图 → 文本 token 与媒体块（add_special / parse_special 为真，== server）
        const auto tokenizeStarted = Clock::now();
        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        const mtmd_input_text text { prompt.data(), prompt.size(), true, true };
        std::vector<const mtmd_bitmap*> pointers = bitmaps.c_ptr();
        const int32_t tokenized = mtmd_tokenize(m_context.get(), chunks.ptr.get(), &text, pointers.data(), pointers.size());
        if (tokenized == 1)
            return fail(FLR_INVALID_REQUEST, "提示里的媒体标记与媒体数量对不上", error);
        if (tokenized != 0)
            return fail(FLR_INVALID_REQUEST, "图片 / 音频预处理失败（详情见推理日志）", error);

        // 4. → 账本项。KV 单元数 = 块的 token 数，位置数 = mtmd_input_chunk_get_n_pos（M-RoPE 的图片 = max(宽, 高)，
        //    例如 260 个 token 只占 20 个位置）。媒体块不拷贝（预处理后的像素，1568 px 的图约 20 MB）：Prepared 里每个块的
        //    shared_ptr 都共享整个 chunks 容器的所有权（别名构造），最后一个块释放时容器一起释放
        const std::shared_ptr<mtmd_input_chunks> owner(chunks.ptr.release(), &mtmd_input_chunks_free);
        std::vector<LedgerItem> items;
        std::vector<std::shared_ptr<void>> mediaChunks;
        std::vector<int> mediaItems;
        int64_t mediaTokens = 0;
        int64_t mediaPositions = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(owner.get()); ++i) {
            const mtmd_input_chunk* chunk = mtmd_input_chunks_get(owner.get(), i);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t count = 0;
                const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &count);
                for (size_t k = 0; k < count; ++k)
                    items.push_back(LedgerItem::text(tokens[k]));
                continue;
            }
            LedgerItem item;
            item.n_tokens = int(mtmd_input_chunk_get_n_tokens(chunk));
            item.n_pos = int(mtmd_input_chunk_get_n_pos(chunk));
            const char* id = mtmd_input_chunk_get_id(chunk);
            if (id && *id) {
                item.media_id = id;
            } else {
                // 没有 id（不该发生：助手函数总会算 SHA-256）：给一个不会与任何账本项相同的，这块就不复用
                static std::atomic<uint64_t> serial { 0 };
                item.media_id = "media-" + std::to_string(++serial);
            }
            if (item.n_tokens <= 0 || item.n_pos <= 0)
                return fail(FLR_INTERNAL, "媒体块的大小不对（" + std::to_string(item.n_tokens) + " token）", error);
            // 非因果注意力的投影（Gemma 3 / Gemma 4 26B·31B / DeepSeek-V4…）整块要在一个 ubatch 里，否则 llama_context::decode
            // GGML_ASSERT、整个应用退出（评审修正）。加载时已按 n_ubatch 限了图片 token，这里再把一次关（自定义的 image_max_tokens…）
            if (m_ubatch > 0 && item.n_tokens > m_ubatch && mtmd_decode_use_non_causal(m_context.get(), chunk)) {
                return fail(FLR_UNSUPPORTED_MEDIA,
                            "这张图太大：当前模型的视觉投影要整张图一起算，这张图 " + std::to_string(item.n_tokens) + " 个 token，超过了批大小 n_ubatch "
                                + std::to_string(m_ubatch) + "（请缩小图片，或在 FRIDAY_LLAMA_ARGS 里加 -ub " + std::to_string(item.n_tokens) + "）",
                            error);
            }
            mediaChunks.emplace_back(owner, static_cast<void*>(const_cast<mtmd_input_chunk*>(chunk)));
            mediaItems.push_back(int(items.size()));
            mediaTokens += item.n_tokens;
            mediaPositions += item.n_pos;
            items.push_back(std::move(item));
        }
        // 提示必须以文本 token 结尾：采样要最后一项的 logits，而媒体块不出 logits（模板总以生成提示结尾，正常走不到）
        if (items.empty() || items.back().isMedia())
            return fail(FLR_INVALID_REQUEST, "提示不能以图片 / 音频结尾", error);

        out.items = std::move(items);
        out.media_chunks = std::move(mediaChunks);
        out.media_items = std::move(mediaItems);
        logf(FLR_LOG_INFO, "看图：%zu 个媒体 → %zu 个媒体块，共 %lld token / %lld 个位置；解码 %.1f ms，分词 %.1f ms", inputs.size(),
             out.media_chunks.size(), (long long)mediaTokens, (long long)mediaPositions, decodeMs, millisecondsSince(tokenizeStarted));
        return FLR_OK;
    } catch (const std::exception& exception) {
        return fail(FLR_INTERNAL, std::string("媒体分词出错：") + exception.what(), error);
    } catch (...) {
        return fail(FLR_INTERNAL, "媒体分词出错", error);
    }
}

int MtmdMedia::eval(const Prepared& p, int item, llama_context* ctx, int seq, llama_pos pos, int n_batch, llama_pos* nextPos)
{
    try {
        if (!ctx || item < 0 || size_t(item) >= p.items.size() || !p.items[size_t(item)].isMedia()) {
            logf(FLR_LOG_ERROR, "看图：第 %d 项不是媒体块", item);
            return -1;
        }
        const mtmd_input_chunk* chunk = nullptr;
        for (size_t k = 0; k < p.media_items.size() && k < p.media_chunks.size(); ++k) {
            if (p.media_items[k] == item) {
                chunk = static_cast<const mtmd_input_chunk*>(p.media_chunks[k].get());
                break;
            }
        }
        const LedgerItem& ledger = p.items[size_t(item)];
        if (!chunk || mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT
            || int(mtmd_input_chunk_get_n_tokens(chunk)) != ledger.n_tokens) {
            logf(FLR_LOG_ERROR, "看图：第 %d 项没有对应的媒体块", item);
            return -1;
        }
        // mtmd_helper_decode_image_chunk 里 n_batch ≤ 0、超过上下文的批大小、seq 不合法都会 GGML_ASSERT（§E.6：先挡掉）
        const int capacity = int(llama_n_batch(ctx));
        n_batch = n_batch <= 0 ? capacity : std::min(n_batch, capacity);
        // 非因果注意力（Gemma 3 一类）要整张图在一次 llama_process 里：不按调用方给的（KV 满后减半的）n_batch 拆
        const bool nonCausal = mtmd_decode_use_non_causal(m_context.get(), chunk);
        if (nonCausal)
            n_batch = capacity;
        if (n_batch <= 0 || seq < 0 || seq >= int(llama_n_seq_max(ctx)) || pos < 0) {
            logf(FLR_LOG_ERROR, "看图：参数不对（seq %d，pos %d，n_batch %d）", seq, int(pos), n_batch);
            return -1;
        }
        if (nonCausal && ledger.n_tokens > int(llama_n_ubatch(ctx))) {
            // 还要在一个 ubatch 里：llama_context::decode 里 GGML_ASSERT(causal_attn || n_ubatch >= n_tokens_all)，进程内会让整个应用
            // 退出（server 没处理，崩的只是 llama-server）。tokenize 已经挡过，这里不调 llama_process（评审修正）
            logf(FLR_LOG_ERROR, "看图：图片 %d token 超过 n_ubatch %u，非因果注意力的投影要整张图在一个 ubatch 里，不预填", ledger.n_tokens,
                 llama_n_ubatch(ctx));
            return -1;
        }

        // 编码（视觉 / 音频编码器，不可中断；结果在 mtmd 上下文里，只能在推理线程上用）
        const auto encodeStarted = Clock::now();
        if (mtmd_encode_chunk(m_context.get(), chunk) != 0) {
            logf(FLR_LOG_ERROR, "看图：媒体块编码失败（%s）", excerpt(ledger.media_id, 16).c_str());
            return -1;
        }
        const double encodeMs = millisecondsSince(encodeStarted);
        float* embeddings = mtmd_get_output_embd(m_context.get());

        // 预填：按 n_batch 分批 llama_process；M-RoPE 的图片每个 token 带 (t, y, x) 三维位置，t 都是 pos
        const auto decodeStarted = Clock::now();
        llama_pos after = pos;
        const int32_t result
            = mtmd_helper_decode_image_chunk(m_context.get(), ctx, chunk, embeddings, pos, seq, n_batch, &after, nullptr, nullptr);
        if (result != 0) {
            // 1 = KV 放不下、2 = 被 abort 回调中止（已算的 ubatch 留在 KV 里，由调用方 rollback）、< 0 = 出错
            if (result != 1 && result != 2)
                logf(FLR_LOG_ERROR, "看图：媒体块预填失败（%d）", int(result));
            return result == 1 || result == 2 ? result : (result < 0 ? result : -1);
        }
        if (after != pos + ledger.n_pos) {
            logf(FLR_LOG_ERROR, "看图：媒体块之后的位置不对（%d，应为 %d）", int(after), int(pos + ledger.n_pos));
            return -1;
        }
        if (nextPos)
            *nextPos = after;
        logf(FLR_LOG_INFO, "看图：媒体块 %s 进了序列 %d（%d token / %d 个位置，从位置 %d 起），编码 %.1f ms，预填提交 %.1f ms",
             excerpt(ledger.media_id, 12).c_str(), seq, ledger.n_tokens, ledger.n_pos, int(pos), encodeMs,
             millisecondsSince(decodeStarted));
        return 0;
    } catch (const std::exception& exception) {
        logf(FLR_LOG_ERROR, "看图：预填媒体块出错：%s", exception.what());
        return -1;
    } catch (...) {
        logf(FLR_LOG_ERROR, "看图：预填媒体块出错");
        return -1;
    }
}

} // namespace

// MARK: - media::extract

namespace media {

int extract(json& messages, const std::string& marker, bool allowImages, bool allowAudio, std::vector<MediaInput>& out,
            std::string* error)
{
    const size_t original = out.size();
    std::string message;
    int status = FLR_OK;
    try {
        if (messages.is_array()) {
            for (json& entry : messages) {
                if (status != FLR_OK)
                    break;
                if (!entry.is_object())
                    continue;
                const auto content = entry.find("content");
                if (content == entry.end() || !content->is_array())
                    continue;
                for (json& part : *content) {
                    if (!part.is_object()) {
                        message = "消息内容的部件必须是对象";
                        status = FLR_INVALID_REQUEST;
                        break;
                    }
                    const auto typeField = part.find("type");
                    const std::string type = typeField != part.end() && typeField->is_string() ? typeField->get<std::string>() : std::string();
                    const PartKind kind = partKind(type);
                    if (kind == PartKind::Text)
                        continue;
                    if (kind == PartKind::Unsupported) {
                        message = type.empty() ? std::string("消息内容的部件缺少 type") : "不支持的消息内容类型：" + excerpt(type);
                        status = FLR_INVALID_REQUEST;
                        break;
                    }
                    // == server：模型没有对应的能力 → 「不支持」，与「数据不对」分开
                    if ((kind == PartKind::Image || kind == PartKind::Video) && !allowImages) {
                        message = kind == PartKind::Image ? "当前模型不能看图（没有加载视觉投影 mmproj）" : "当前模型不能看视频（没有加载视觉投影 mmproj）";
                        status = FLR_UNSUPPORTED_MEDIA;
                        break;
                    }
                    if (kind == PartKind::Audio && !allowAudio) {
                        message = "当前模型不能听音频";
                        status = FLR_UNSUPPORTED_MEDIA;
                        break;
                    }
                    if (marker.empty()) {
                        message = "媒体标记为空";
                        status = FLR_INTERNAL;
                        break;
                    }
                    MediaInput input;
                    input.kind = kind == PartKind::Image ? "image" : kind == PartKind::Audio ? "audio" : "video";
                    status = decodeURL(partURL(part, type, kind), input.data, message);
                    if (status != FLR_OK)
                        break;
                    out.push_back(std::move(input));
                    // 原地换成标记（== server：之后 common_chat_msgs_parse_oaicompat 认 text / media_marker 两种部件）
                    part = json { { "type", "media_marker" }, { "text", marker } };
                }
            }
        }
    } catch (const std::exception& exception) {
        message = std::string("消息内容格式不对：") + exception.what();
        status = FLR_INVALID_REQUEST;
    }
    if (status != FLR_OK) {
        out.resize(original);
        if (error)
            *error = message;
    }
    return status;
}

} // namespace media

// MARK: - Media

void Media::reserveForFit(const std::string& mmprojPath, common_params& params)
{
    // == server-context.cpp:1045-1066：fit 按「模型 + 上下文 + 余量」挪层，mmproj 要的显存先算进余量（实测约 1.1 GiB）
    if (mmprojPath.empty() || !params.fit_params)
        return;
    try {
        const auto started = Clock::now();
        const std::string marker = "<__media__>";
        const mtmd_memory_usage memory = mtmd_get_memory_usage(mmprojPath.c_str(), contextParams(params, marker));
        const std::map<ggml_backend_dev_t, size_t>& usage = memory.backend_mem_usage;
        if (usage.empty()) {
            logf(FLR_LOG_WARN, "看图：估算视觉投影的显存失败（%s），fit 不为它预留", fileName(mmprojPath).c_str());
            return;
        }
        size_t total = 0;
        for (const auto& [device, size] : usage) {
            total += size;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                if (ggml_backend_dev_get(i) != device)
                    continue;
                if (i < params.fit_params_target.size())
                    params.fit_params_target[i] += size;
                break;
            }
        }
        logf(FLR_LOG_INFO, "看图：为视觉投影预留 %.1f MiB（估算用时 %.0f ms）", double(total) / (1024.0 * 1024.0), millisecondsSince(started));
    } catch (const std::exception& exception) {
        logf(FLR_LOG_WARN, "看图：估算视觉投影的显存出错：%s", exception.what());
    } catch (...) {
        logf(FLR_LOG_WARN, "看图：估算视觉投影的显存出错");
    }
}

std::unique_ptr<Media> Media::load(const std::string& mmprojPath, llama_model* model, const common_params& params,
                                   const std::function<bool(float)>& progress, std::string* error, int nUbatch)
{
    const auto failed = [&](const std::string& message) -> std::unique_ptr<Media> {
        if (error)
            *error = message;
        logf(FLR_LOG_ERROR, "%s", message.c_str());
        return nullptr;
    };
    if (mmprojPath.empty())
        return failed("没有指定视觉投影（mmproj）文件");
    if (!fileExists(mmprojPath))
        return failed("找不到视觉投影（mmproj）文件：" + mmprojPath);
    try {
        const auto started = Clock::now();
        const std::string marker = randomMarker();
        mtmd_context_params mparams = contextParams(params, marker);
        struct ProgressState {
            const std::function<bool(float)>* callback;
            bool cancelled = false;
        } state { progress ? &progress : nullptr };
        if (state.callback) {
            mparams.progress_callback = [](float value, void* user) -> bool {
                auto* s = static_cast<ProgressState*>(user);
                const bool keep = (*s->callback)(value);
                s->cancelled = s->cancelled || !keep;
                return keep;
            };
            mparams.progress_callback_user_data = &state;
        }

        // 加载期间接住 mtmd 的错误日志（见 captureLoadLog），完了换回运行时的日志接管
        {
            std::lock_guard<std::mutex> lock(loadLogCapture().mutex);
            loadLogCapture().lastError.clear();
            loadLogCapture().lastLevel = FLR_LOG_INFO;
        }
        mtmd_helper_log_set(captureLoadLog, nullptr);
        mtmd::context_ptr context(mtmd_init_from_file(mmprojPath.c_str(), model, mparams));
        installLibraryLogging();
        std::string lastError;
        {
            std::lock_guard<std::mutex> lock(loadLogCapture().mutex);
            lastError = std::move(loadLogCapture().lastError);
        }
        if (!context) {
            if (state.cancelled)
                return failed("加载视觉投影已取消");
            return failed("视觉投影（mmproj）加载失败：" + (lastError.empty() ? fileName(mmprojPath) : cleanLoadError(lastError)));
        }
        if (!mtmd_support_vision(context.get()) && !mtmd_support_audio(context.get()))
            return failed("视觉投影（mmproj）里既没有视觉也没有音频部分：" + fileName(mmprojPath));
        // 非因果注意力的视觉投影（Gemma 3 / Gemma 4 26B·31B / DeepSeek-V4…）：整张图要在一个 ubatch 里（评审修正）。
        // 应用不传 -ub（缺省 512），Gemma 4 的图可到 1120 token：这时按 image_max_tokens = n_ubatch 重新加载，mtmd 把图缩到这么多
        // token 以内（图会糊一点，但不会让 llama_context::decode 的 GGML_ASSERT 结束整个应用）。投影的上限拿一张大图试出来；
        // 用户自己给了更小的 --image-max-tokens 就不动
        if (nUbatch > 0 && mtmd_support_vision(context.get()) && mtmd_decode_use_non_causal(context.get(), nullptr)) {
            const int largest = probeMaxImageTokens(context.get(), marker);
            if (largest > nUbatch && (params.image_max_tokens <= 0 || params.image_max_tokens > nUbatch)) {
                mparams.image_max_tokens = nUbatch;
                mparams.warmup = false;   // 刚热过身
                mtmd_helper_log_set(captureLoadLog, nullptr);
                mtmd::context_ptr capped(mtmd_init_from_file(mmprojPath.c_str(), model, mparams));
                installLibraryLogging();
                if (state.cancelled)
                    return failed("加载视觉投影已取消");
                const int cappedLargest = capped ? probeMaxImageTokens(capped.get(), marker) : -1;
                if (capped && cappedLargest > 0 && cappedLargest <= nUbatch) {
                    logf(FLR_LOG_NOTICE, "看图：视觉投影要整张图一起算（非因果注意力），一张图最多 %d token，超过 n_ubatch %d：按 %d token 重新加载，"
                                         "大图会缩小（要原样看大图可在 FRIDAY_LLAMA_ARGS 里加 -ub %d）",
                         largest, nUbatch, cappedLargest, largest);
                    context = std::move(capped);
                } else {
                    // 限不下来（投影不认 image_max_tokens）：留着原来的，超过 n_ubatch 的图在 tokenize 时拒绝
                    logf(FLR_LOG_WARN, "看图：视觉投影要整张图一起算，一张图最多 %d token，超过 n_ubatch %d，限制图片 token 没有生效（%d）；"
                                       "超过的图会被拒绝",
                         largest, nUbatch, cappedLargest);
                }
            }
        }
        auto media = std::make_unique<MtmdMedia>(std::move(context), marker, params, nUbatch);
        logf(FLR_LOG_NOTICE, "看图：视觉投影 %s 已加载，用时 %.0f ms（看图 %s、听音 %s、视频 %s，%s）", fileName(mmprojPath).c_str(),
             millisecondsSince(started), media->supportsImages() ? "是" : "否", media->supportsAudio() ? "是" : "否",
             media->supportsVideo() ? "是" : "否", media->usesMrope() ? "M-RoPE 位置" : "普通位置");
        return media;
    } catch (const std::exception& exception) {
        installLibraryLogging();
        return failed(std::string("视觉投影（mmproj）加载出错：") + exception.what());
    } catch (...) {
        installLibraryLogging();
        return failed("视觉投影（mmproj）加载出错");
    }
}

} // namespace flr
