// Whisper（whisper.cpp v1.9.5，和我们用同一份 ggml 0.26.0）：一句说完后出定稿，中英混说比 SenseVoice 准。
// 用户 2026-10-10「识别不清楚我说的英语」：中文为主的一句里夹着英文词（「帮我看看 Secret Chat」），SenseVoice 按中文听、
// 把英文按读音写成汉字（「C亏欠」）。Whisper large-v3-turbo 夹着的英文照原样写出来，还能用 prompt 偏向热词（项目名之类）。
// 分工：实时字幕、断句、判断是不是人声、认语种照旧用 SenseVoice（非自回归，几十毫秒）；SenseVoice 认定是人声的一句，再交给 Whisper 出定稿。
// Whisper 出错、出空、像是幻觉（「字幕由…提供」「谢谢观看」这类、比 SenseVoice 长出好几倍）时用 SenseVoice 的。
#pragma once

#include "Backend.h"

#include <mutex>
#include <string>

struct whisper_context;
struct whisper_state;

namespace voice {

class Whisper {
public:
    explicit Whisper(const Backend& backend);
    ~Whisper();
    Whisper(const Whisper&) = delete;
    Whisper& operator=(const Whisper&) = delete;

    /// ggml 格式的 Whisper 模型（ggml-large-v3-turbo-q5_0.bin）；在 backend 选的那张卡上跑
    bool load(const std::string& path, std::string* error);
    bool loaded() const { return m_ctx != nullptr; }
    const std::string& path() const { return m_path; }

    struct Result {
        bool ok = false;
        std::string text;
        double seconds = 0;
        std::string error;
    };
    /// pcm：16 kHz 单声道 [-1,1]。language：SenseVoice 认出的 zh / en / yue / ja / ko（别的按 auto）。
    /// prompt：偏向的词、上文（客户端给的热词）。线程安全（内部串行）
    Result transcribe(const float* pcm, size_t count, const std::string& language, const std::string& prompt);

    /// 定稿用哪个：Whisper 的看着像幻觉（视频网站的字幕套话、比 SenseVoice 长出好几倍）就用 SenseVoice 的
    static bool plausible(const std::string& whisper, const std::string& sensevoice);

    /// 两边合起来（SenseVoice 中文更准，Whisper 英文更准）：按字 / 英文词对齐，一样的地方中文和标点用 SenseVoice 的、英文词用 Whisper 的
    /// （大小写、拼写）；不一样的一段：Whisper 那边是英文词、SenseVoice 那边是汉字（把英文按读音写成了汉字，「C亏欠」↔「Secret Chat」）
    /// 或两边都是英文 → 用 Whisper 的，别的（中文对中文、繁简、数字写法）→ 用 SenseVoice 的。language 是 en 时整句用 Whisper 的
    static std::string merge(const std::string& sensevoice, const std::string& whisper, const std::string& language);

private:
    const Backend& m_backend;
    std::string m_path;
    whisper_context* m_ctx = nullptr;
    /// 两个状态，编码窗口各自固定：短的（kShortCtx，约 10 秒，绝大多数句子）、长的（默认 1500 = 30 秒）。
    /// 同一个状态前后两次窗口长度不一样时，后面的句子越来越不准（热词不起作用、英文全小写；whisper.cpp v1.9.5 实测），所以不按每句的长度改窗口
    whisper_state* m_short = nullptr;
    whisper_state* m_long = nullptr;
    static constexpr int kShortCtx = 512;
    std::mutex m_mutex;
};

} // namespace voice
