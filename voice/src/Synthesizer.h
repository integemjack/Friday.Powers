// 合成：CosyVoice3（cosyvoice.cpp）。一个模型、多个音色（voices/<名>.gguf，离线生成的 prompt_speech），
// 合成边出边回调（流式），同一时间只合成一段（GPU 上一个 worker），别的线程排队。
// v0.5 用户音色（克隆出来的、导入的）在 <models>/user-voices/<名>.gguf：启动时扫进来，运行中可以热加、热删（内置的不行）。
#pragma once

#include "Backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct cosyvoice_context;
struct cosyvoice_prompt_speech;
struct cosyvoice_prompt;
struct cosyvoice_tts_context;

namespace voice {

struct SpeakResult {
    bool ok = false;
    bool stopped = false;       // 被 stop() 或回调叫停
    std::string error;
    size_t samples = 0;         // 出了多少采样（模型采样率）
    double seconds = 0;         // 合成用时
    double firstAudioSeconds = 0;   // 开始到第一块音频
    double queuedSeconds = 0;       // 排队等别人合成完的时间
    std::vector<unsigned> chunks;   // 每块要的语音 token 数（自适应分块挑的，调参看）
};

/// v0.5.1 念的那头「到了就放」时手里还剩多少没放的声音（估）。边写边读按它挑下一块多大：剩得多就用大块（总速度快），
/// 剩得少第一块就小（第一声早）。新 Friday 会先攒够再放，实际剩得比这里估的多，估少了只是块挑小了些
class PlaybackClock {
public:
    /// 刚发出去 seconds 秒的声音；inSegment = 同一段里的后续块（这时放空了算一次断档，段与段之间等字不算）
    void sent(double seconds, bool inSegment)
    {
        const auto now = std::chrono::steady_clock::now();
        if (!m_started || now >= m_end) {
            if (m_started && inSegment) {
                m_gapSeconds += std::chrono::duration<double>(now - m_end).count();
                ++m_gaps;
            }
            m_started = true;
            m_end = now;
        }
        m_end += std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
    }
    double ahead() const
    {
        if (!m_started)
            return 0;
        return std::max(0.0, std::chrono::duration<double>(m_end - std::chrono::steady_clock::now()).count());
    }
    double gapSeconds() const { return m_gapSeconds; }
    int gaps() const { return m_gaps; }
    void resetGaps()
    {
        m_gapSeconds = 0;
        m_gaps = 0;
    }

private:
    bool m_started = false;
    std::chrono::steady_clock::time_point m_end;
    double m_gapSeconds = 0;
    int m_gaps = 0;
};

class Synthesizer {
public:
    using OnAudio = std::function<bool(const float* samples, size_t count)>;   // 返回 false 中止
    /// 念的那头现在手里还剩多少秒没放（PlaybackClock::ahead）；每块之间问一次，空 = 当作 0
    using Ahead = std::function<double()>;

    explicit Synthesizer(const Backend& backend);
    ~Synthesizer();
    Synthesizer(const Synthesizer&) = delete;
    Synthesizer& operator=(const Synthesizer&) = delete;

    /// 加载模型和 voicesDir 里所有音色（default 音色优先叫 default）；userVoicesDir 非空时一并扫进去，
    /// 这部分是用户克隆的音色（可删、可热加，内置的不行）
    bool load(const std::string& modelPath, const std::string& voicesDir, std::string* error, const std::string& userVoicesDir = {});
    bool loaded() const { return m_ctx != nullptr; }
    int sampleRate() const { return m_sampleRate; }
    const std::string& path() const { return m_path; }
    std::vector<std::string> voices() const;
    std::vector<std::string> userVoices() const;
    bool hasVoice(const std::string& voice) const;
    bool isUserVoice(const std::string& voice) const;
    /// 用户音色热加进来（同一个进程不重启就能用）：speech 是读好的 prompt_speech（不管成败都归这里释放），
    /// file 是它落盘的 gguf（removeVoice 时一起删）。同名的：replace 且是用户音色就换掉，否则失败（内置的永远不换）
    bool addUserVoice(const std::string& name, cosyvoice_prompt_speech* speech, const std::string& file, bool replace, std::string* error);
    /// 是不是 CosyVoice3 的音色文件（prompt_speech gguf：feat、embedding、tokens、text 四个张量）。读之前先用 ggml 的 gguf
    /// 解析器看一遍：cosyvoice.cpp 读到不是 gguf 的文件时不报错、直接崩（解析失败没拦住）
    static bool isVoiceFile(const std::string& path);
    /// 删掉一个用户音色和它的文件（正在合成则等它完）；内置音色、没有的返回 false
    bool removeVoice(const std::string& name, std::string* error);
    std::string defaultVoice() const;
    /// 0.5.2 flow 的扩散步数（load 之前设）：0 = 模型自带的 10 步；少了快、细节差一点（M1 上 5 步比 10 步快约四成）
    void setDiffusionSteps(int steps) { m_diffusionSteps = std::max(0, steps); }
    /// 0.5.2 DiT KV 缓存和音色提示前缀（load 之前设，默认开；关掉 = 0.5.1 的样子，对比用）
    void setDitCache(bool enabled) { m_ditCache = enabled; }
    /// 流式时每块固定多少个语音 token（越小第一声越早，太小拖慢总速度；调参、对比用）；0 = 自适应（v0.5.1 默认，见 speak）
    void setChunkTokens(unsigned tokens);
    /// 固定的块大小；自适应时是 0
    unsigned chunkTokens() const { return m_fixedChunk; }

    /// 合成一段（阻塞到合成完）。owner 用来让 stop(owner) 只叫停自己那段。
    /// 自适应分块（没设 setChunkTokens）：第一块挑「刚好盖住下一块合成时间」的最小块，之后用大块；念的那头手里剩得多（ahead）
    /// 就直接上大块。每块的耗时边念边学（按音色）。0.5.1 时每块固定开销大（flow 把音色提示和前面念过的重算，2060 上约 1.1 秒），
    /// 0.5.2 开了 DiT 缓存后几乎没有了，块大小主要影响第一声早晚
    SpeakResult speak(uint64_t owner, const std::string& text, const std::string& voice, float speed, const std::string& instruction,
                      const OnAudio& onAudio, const Ahead& ahead = {});
    /// 叫停 owner 正在合成的那段（没在合成就什么也不做）；会等它停下来
    void stop(uint64_t owner);

private:
    struct Voice {
        cosyvoice_prompt_speech* speech = nullptr;
        cosyvoice_prompt* prompt = nullptr;
        cosyvoice_tts_context* tts = nullptr;
        std::string file;      // 音色 gguf 的路径（删用户音色时把文件也删掉）
        bool user = false;     // 用户克隆的（可热删）；随包发的是内置的
    };

    bool scanVoices(const std::string& dir, bool user, std::string* error);
    /// speech → prompt → tts 上下文；失败时释放已建的、返回 false（speech 也释放）
    bool initVoice(Voice& v);
    static void freeVoice(Voice& v);

    const Backend& m_backend;
    cosyvoice_context* m_ctx = nullptr;
    std::map<std::string, Voice> m_voices;
    std::string m_defaultVoice;
    std::string m_path;
    int m_sampleRate = 24000;

    std::mutex m_mutex;                    // 同时只合成一段；增删音色也要拿它（不能删正在念的）
    mutable std::mutex m_voicesMutex;      // m_voices / m_defaultVoice（要两把时先 m_mutex 后它）
    std::mutex m_ownerMutex;
    uint64_t m_owner = 0;                  // 正在合成的是谁（0 = 没有）

    // 0.5.2 flow 的设置（load 时用）
    int m_diffusionSteps = 0;              // 0 = 模型自带
    bool m_ditCache = true;
    double m_ditCacheBytes = 0;            // 开了的 DiT 缓存多大（0 = 没开）

    // 分块（m_mutex 管）
    unsigned m_fixedChunk = 0;             // 固定块大小（0 = 自适应）
    unsigned m_minChunk = 28;              // 自适应的最小块：模型默认的块
    /// 每个音色的耗时系数：实测耗时 / 参考耗时（参考是 M1 + Metal 上开着 DiT 缓存量的，见 Synthesizer.cpp）
    std::map<std::string, double> m_costScale;
    unsigned pickFirstChunk(double scale, double secondsPerToken, double ahead) const;
    unsigned pickNextChunk(double scale, double secondsPerToken, double ahead) const;
};

} // namespace voice
