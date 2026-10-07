// 合成：CosyVoice3（cosyvoice.cpp）。一个模型、多个音色（voices/<名>.gguf，离线生成的 prompt_speech），
// 合成边出边回调（流式），同一时间只合成一段（GPU 上一个 worker），别的线程排队。
#pragma once

#include "Backend.h"

#include <atomic>
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
};

class Synthesizer {
public:
    using OnAudio = std::function<bool(const float* samples, size_t count)>;   // 返回 false 中止

    explicit Synthesizer(const Backend& backend);
    ~Synthesizer();
    Synthesizer(const Synthesizer&) = delete;
    Synthesizer& operator=(const Synthesizer&) = delete;

    /// 加载模型和 voicesDir 里所有音色；default 音色优先叫 default
    bool load(const std::string& modelPath, const std::string& voicesDir, std::string* error);
    bool loaded() const { return m_ctx != nullptr; }
    int sampleRate() const { return m_sampleRate; }
    const std::string& path() const { return m_path; }
    std::vector<std::string> voices() const;
    bool hasVoice(const std::string& voice) const { return m_voices.count(voice) > 0; }
    const std::string& defaultVoice() const { return m_defaultVoice; }
    /// 流式时每块多少个语音 token（越小第一声越早，太小拖慢总速度）；0 = 模型默认
    void setChunkTokens(unsigned tokens);
    unsigned chunkTokens() const;

    /// 合成一段（阻塞到合成完）。owner 用来让 stop(owner) 只叫停自己那段
    SpeakResult speak(uint64_t owner, const std::string& text, const std::string& voice, float speed, const std::string& instruction,
                      const OnAudio& onAudio);
    /// 叫停 owner 正在合成的那段（没在合成就什么也不做）；会等它停下来
    void stop(uint64_t owner);

private:
    struct Voice {
        cosyvoice_prompt_speech* speech = nullptr;
        cosyvoice_prompt* prompt = nullptr;
        cosyvoice_tts_context* tts = nullptr;
    };

    const Backend& m_backend;
    cosyvoice_context* m_ctx = nullptr;
    std::map<std::string, Voice> m_voices;
    std::string m_defaultVoice;
    std::string m_path;
    int m_sampleRate = 24000;

    std::mutex m_mutex;                    // 同时只合成一段
    std::mutex m_ownerMutex;
    uint64_t m_owner = 0;                  // 正在合成的是谁（0 = 没有）
};

} // namespace voice
