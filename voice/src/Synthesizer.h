// 合成：CosyVoice3（cosyvoice.cpp）。一个模型、多个音色（voices/<名>.gguf，离线生成的 prompt_speech），
// 合成边出边回调（流式），同一时间只合成一段（GPU 上一个 worker），别的线程排队。
// v0.5 用户音色（克隆出来的、导入的）在 <models>/user-voices/<名>.gguf：启动时扫进来，运行中可以热加、热删（内置的不行）。
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
};

} // namespace voice
