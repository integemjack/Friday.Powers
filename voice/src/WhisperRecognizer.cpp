#include "WhisperRecognizer.h"

#include "Log.h"

#include "whisper.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace voice {

namespace {

/// whisper.cpp 自己用 ifstream 开模型，Windows 上路径有中文（用户名）时打不开：自己按 UTF-8 路径开文件喂给它
struct FileLoader {
    FILE* file = nullptr;

    static size_t read(void* ctx, void* output, size_t size)
    {
        return std::fread(output, 1, size, static_cast<FileLoader*>(ctx)->file);
    }
    static bool eof(void* ctx) { return std::feof(static_cast<FileLoader*>(ctx)->file) != 0; }
    static void close(void* ctx)
    {
        auto* self = static_cast<FileLoader*>(ctx);
        if (self->file)
            std::fclose(self->file);
        self->file = nullptr;
    }
};

FILE* openUtf8(const std::string& path)
{
#ifdef _WIN32
    return _wfopen(fs::u8path(path).wstring().c_str(), L"rb");
#else
    return std::fopen(path.c_str(), "rb");
#endif
}

/// whisper.cpp 的日志：只留警告和错误（加载时每层张量都打一行）
void whisperLog(ggml_log_level level, const char* text, void*)
{
    if (level != GGML_LOG_LEVEL_WARN && level != GGML_LOG_LEVEL_ERROR)
        return;
    std::string line(text ? text : "");
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
        line.pop_back();
    if (!line.empty())
        VLOG_DEBUG("whisper：%s", line.c_str());
}

/// UTF-8 字符数
size_t codepoints(const std::string& s)
{
    size_t n = 0;
    for (const unsigned char c : s)
        n += (c & 0xC0) != 0x80 ? 1 : 0;
    return n;
}

std::string trim(std::string s)
{
    const auto space = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    while (!s.empty() && space(static_cast<unsigned char>(s.back())))
        s.pop_back();
    size_t i = 0;
    while (i < s.size() && space(static_cast<unsigned char>(s[i])))
        ++i;
    return s.substr(i);
}

} // namespace

Whisper::Whisper(const Backend& backend)
    : m_backend(backend)
{
}

Whisper::~Whisper()
{
    if (m_short)
        whisper_free_state(m_short);
    if (m_long)
        whisper_free_state(m_long);
    if (m_ctx)
        whisper_free(m_ctx);
}

bool Whisper::load(const std::string& path, std::string* error)
{
    whisper_log_set(whisperLog, nullptr);
    FileLoader file;
    file.file = openUtf8(path);
    if (!file.file) {
        if (error)
            *error = "打不开 Whisper 模型：" + path;
        return false;
    }
    whisper_model_loader loader { &file, &FileLoader::read, &FileLoader::eof, &FileLoader::close };

    whisper_context_params params = whisper_context_default_params();
    params.use_gpu = m_backend.usesGpu();
    // Metal 上 CosyVoice 的 LLM 开 flash attention 算错过（voice 0.2.1），Whisper 在 Metal 上也不开，别的卡开
    params.flash_attn = m_backend.usesGpu() && !m_backend.isMetal();
    // whisper.cpp 按顺序数 GPU / 集显（GGML_BACKEND_DEVICE_TYPE_GPU、IGPU），挑第 gpu_device 个：数到我们选的那张
    params.gpu_device = 0;
    if (params.use_gpu) {
        int index = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU)
                continue;
            if (m_backend.device().name == ggml_backend_dev_name(dev)) {
                params.gpu_device = index;
                break;
            }
            ++index;
        }
    }
    const auto started = std::chrono::steady_clock::now();
    m_ctx = whisper_init_with_params_no_state(&loader, params);
    if (file.file)
        FileLoader::close(&file);
    if (!m_ctx) {
        if (error)
            *error = "Whisper 模型加载失败：" + path;
        return false;
    }
    m_short = whisper_init_state(m_ctx);
    m_long = m_short ? whisper_init_state(m_ctx) : nullptr;
    if (!m_short || !m_long) {
        if (m_short)
            whisper_free_state(m_short);
        m_short = nullptr;
        whisper_free(m_ctx);
        m_ctx = nullptr;
        if (error)
            *error = "Whisper 初始化失败（显存不够？）";
        return false;
    }
    m_path = path;
    VLOG_INFO("Whisper 已加载：%s（%s，%.1f 秒）", path.c_str(), params.use_gpu ? m_backend.device().name.c_str() : "CPU",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    return true;
}

Whisper::Result Whisper::transcribe(const float* pcm, size_t count, const std::string& language, const std::string& prompt)
{
    Result result;
    if (!m_ctx) {
        result.error = "没加载 Whisper 模型";
        return result;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto started = std::chrono::steady_clock::now();
    const bool known = language == "zh" || language == "en" || language == "yue" || language == "ja" || language == "ko";
    // 中文：先给一句简体、带标点的上文（不然常出繁体、不带标点），说明中间可能夹着英文；再接客户端给的热词
    std::string initial;
    if (language == "zh")
        initial = "以下是普通话的句子，中间可能夹着英文。";
    if (!prompt.empty())
        initial += (initial.empty() ? "" : " ") + prompt;

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = std::max(1, m_backend.threads());
    params.no_context = true;
    params.no_timestamps = true;
    params.single_segment = true;
    params.print_special = false;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.language = known ? language.c_str() : "auto";
    params.detect_language = false;
    params.initial_prompt = initial.empty() ? nullptr : initial.c_str();
    params.suppress_blank = true;
    params.suppress_nst = true;
    // 不做温度回退（Whisper 觉得这次解码不够好时换高温度、多次采样重来，M1 上一句能拖到 8 秒多）：
    // 结果不像样有 plausible 和 merge 兜底，用 SenseVoice 的；按长度限制出多少 token（每秒 15 个够了），一直重复也很快收住
    params.temperature = 0.0f;
    params.temperature_inc = 0.0f;
    params.greedy.best_of = 1;
    params.max_tokens = std::min(160, int(count * 15 / 16000) + 24);
    // 编码窗口：Whisper 默认按 30 秒算，一句几秒的话 M1 上要 4 秒多；放得下（每秒 50 帧，多留 128 帧）就用 10 秒窗口的那个状态
    // （M1 上约 1 秒），放不下用 30 秒的。每个状态的窗口固定不变（见头文件）
    const bool fits = int(count * 50 / 16000) + 128 <= kShortCtx;
    whisper_state* state = fits ? m_short : m_long;
    params.audio_ctx = fits ? kShortCtx : 0;

    // Whisper 要至少 1 秒：短的后面补静音
    std::vector<float> padded;
    const float* samples = pcm;
    int n = int(count);
    if (count < 16000 + 1600) {
        padded.assign(pcm, pcm + count);
        padded.resize(16000 + 1600, 0.0f);
        samples = padded.data();
        n = int(padded.size());
    }
    if (whisper_full_with_state(m_ctx, state, params, samples, n) != 0) {
        result.error = "Whisper 识别失败";
        return result;
    }
    std::string text;
    const int segments = whisper_full_n_segments_from_state(state);
    for (int i = 0; i < segments; ++i)
        text += whisper_full_get_segment_text_from_state(state, i);
    result.text = trim(text);
    result.ok = true;
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

bool Whisper::plausible(const std::string& whisper, const std::string& sensevoice)
{
    if (whisper.empty())
        return false;
    // 没人说话时 Whisper 爱补的视频字幕套话：SenseVoice 那边没有就当是编的
    static const char* const phrases[] = { "字幕", "订阅", "訂閱", "谢谢观看", "謝謝觀看", "感谢观看", "感謝觀看", "点赞", "點贊",
                                           "打赏", "请不吝", "請不吝", "Amara", "明镜", "Thanks for watching", "Subscribe" };
    for (const char* phrase : phrases)
        if (whisper.find(phrase) != std::string::npos && sensevoice.find(phrase) == std::string::npos)
            return false;
    // 比 SenseVoice 长出好几倍：多半是一直重复
    return codepoints(whisper) <= codepoints(sensevoice) * 3 + 12;
}

namespace {

/// 对齐用的一个单位：一个汉字（假名、谚文）、一个英文词（字母数字连着的）、一个标点
struct Token {
    enum class Kind { Cjk, Latin, Punct };
    Kind kind = Kind::Punct;
    std::string text;
    std::string key;   // 比较用：英文小写、全角标点换成半角
};

std::string punctKey(uint32_t cp, const std::string& text)
{
    switch (cp) {
    case 0xFF0C: case 0x3001: return ",";   // ，、
    case 0x3002: return ".";                // 。
    case 0xFF1F: return "?";
    case 0xFF01: return "!";
    case 0xFF1A: return ":";
    case 0xFF1B: return ";";
    default: return text;
    }
}

std::vector<Token> tokenize(const std::string& s)
{
    std::vector<Token> tokens;
    bool inWord = false;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp = c;
        size_t len = 1;
        if (c >= 0xF0) {
            cp = c & 0x07, len = 4;
        } else if (c >= 0xE0) {
            cp = c & 0x0F, len = 3;
        } else if (c >= 0xC0) {
            cp = c & 0x1F, len = 2;
        }
        for (size_t k = 1; k < len && i + k < s.size(); ++k)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        const std::string ch = s.substr(i, len);
        i += len;
        const bool latin = (cp < 0x80 && (std::isalnum(int(cp)) || ((cp == '\'' || cp == '-') && inWord))) || (cp >= 0xC0 && cp < 0x2000);
        const bool cjk = (cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x3400 && cp <= 0x9FFF) || (cp >= 0xAC00 && cp <= 0xD7AF)
            || (cp >= 0xF900 && cp <= 0xFAFF);
        if (latin) {
            if (!inWord)
                tokens.push_back({ Token::Kind::Latin, {}, {} });
            tokens.back().text += ch;
            for (const char x : ch)
                tokens.back().key += char(std::tolower(static_cast<unsigned char>(x)));
            inWord = true;
            continue;
        }
        inWord = false;
        if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x3000)
            continue;
        tokens.push_back({ cjk ? Token::Kind::Cjk : Token::Kind::Punct, ch, cjk ? ch : punctKey(cp, ch) });
    }
    return tokens;
}

} // namespace

std::string Whisper::merge(const std::string& sensevoice, const std::string& whisper, const std::string& language)
{
    if (language == "en")
        return whisper;
    const std::vector<Token> a = tokenize(sensevoice);
    const std::vector<Token> b = tokenize(whisper);
    const size_t n = a.size(), m = b.size();
    // 编辑距离对齐（一句话几十个单位，O(n·m) 够了）
    std::vector<std::vector<int>> cost(n + 1, std::vector<int>(m + 1, 0));
    for (size_t i = 0; i <= n; ++i)
        cost[i][0] = int(i);
    for (size_t j = 0; j <= m; ++j)
        cost[0][j] = int(j);
    for (size_t i = 1; i <= n; ++i)
        for (size_t j = 1; j <= m; ++j)
            cost[i][j] = std::min({ cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + (a[i - 1].key == b[j - 1].key ? 0 : 1) });
    // 从后往前走出对齐：一样的一对，或者不一样的一段（两边各几个）
    struct Step {
        bool same = false;
        std::vector<const Token*> left, right;   // SenseVoice、Whisper
    };
    std::vector<Step> steps;
    size_t i = n, j = m;
    const auto differ = [&](const Token* left, const Token* right) {
        if (steps.empty() || steps.back().same)
            steps.push_back({});
        if (left)
            steps.back().left.insert(steps.back().left.begin(), left);
        if (right)
            steps.back().right.insert(steps.back().right.begin(), right);
    };
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && a[i - 1].key == b[j - 1].key && cost[i][j] == cost[i - 1][j - 1]) {
            steps.push_back({ true, { &a[i - 1] }, { &b[j - 1] } });
            --i, --j;
        } else if (i > 0 && j > 0 && cost[i][j] == cost[i - 1][j - 1] + 1) {
            differ(&a[i - 1], &b[j - 1]);
            --i, --j;
        } else if (i > 0 && cost[i][j] == cost[i - 1][j] + 1) {
            differ(&a[i - 1], nullptr);
            --i;
        } else {
            differ(nullptr, &b[j - 1]);
            --j;
        }
    }
    std::reverse(steps.begin(), steps.end());
    // 拼回去：英文词之间一个空格，汉字、标点和英文之间不加（同 SenseVoice 的写法）
    std::string out;
    bool lastLatin = false;
    const auto put = [&](const Token& token) {
        const bool latin = token.kind == Token::Kind::Latin;
        if (latin && lastLatin)
            out += ' ';
        out += token.text;
        lastLatin = latin;
    };
    for (const Step& step : steps) {
        if (step.same) {
            put(step.left.front()->kind == Token::Kind::Latin ? *step.right.front() : *step.left.front());
            continue;
        }
        const auto has = [](const std::vector<const Token*>& tokens, Token::Kind kind) {
            return std::any_of(tokens.begin(), tokens.end(), [kind](const Token* t) { return t->kind == kind; });
        };
        const bool rightLatin = has(step.right, Token::Kind::Latin), rightCjk = has(step.right, Token::Kind::Cjk);
        const bool leftLatin = has(step.left, Token::Kind::Latin), leftCjk = has(step.left, Token::Kind::Cjk);
        // Whisper 那边是英文、SenseVoice 那边把它写成了汉字；或者两边都是英文：用 Whisper 的。别的用 SenseVoice 的
        const bool useWhisper = rightLatin && !rightCjk && (leftCjk || (leftLatin && !leftCjk));
        for (const Token* token : useWhisper ? step.right : step.left)
            if (!useWhisper || token->kind != Token::Kind::Punct)
                put(*token);
        // Whisper 那段里的标点不要（半角逗号这些），SenseVoice 那段要是有标点照样留着
        if (useWhisper)
            for (const Token* token : step.left)
                if (token->kind == Token::Kind::Punct)
                    put(*token);
    }
    return out;
}

} // namespace voice
