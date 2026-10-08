// 一路实时听写（一条 WebSocket）：喂音频 → FSMN-VAD 每 10 毫秒判一帧 → 断句状态机 →
//   说话中每 partialIntervalMs 对这句整段重识别一次出 partial（实时字幕）；说完出 final（带标点）。
// 协议见 docs/voice-protocol.md。feed 在连接线程调，识别在推理线程跑，结果通过 send 回调发出去（send 要线程安全）。
// 声纹（start 帧 speaker: true、加载了 campplus）：final 带这句的声纹（speaker，192 维单位向量）；说了 1 秒以上的 partial
// 每多 0.8 秒也带一次（客户端判断插话的是不是登记过的人）。
#pragma once

#include "AudioIO.h"
#include "FsmnVad.h"
#include "InferenceQueue.h"
#include "SenseVoice.h"
#include "SpeakerEncoder.h"

#include "json.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace voice {

struct StreamConfig {
    std::string language = "auto";
    int sampleRate = kSampleRate;
    int partialIntervalMs = 300;
    int endSilenceMs = 450;
    int maxSegmentMs = 20000;
    int prerollMs = 300;
    float threshold = 0.6f;   // 一帧算「在说话」的概率门限
    bool partials = true;
    bool itn = true;
    bool speaker = false;   // final / partial 带上声纹
};

/// 整段离线断句（一次性听写用，和实时同一套状态机）：每帧概率 → [开始帧, 结束帧) 列表
std::vector<std::pair<int, int>> segmentFrames(const std::vector<float>& probability, const StreamConfig& config);

/// 整段识别：VAD 断句后逐句识别，按语种接起来（中日韩不加空格）
std::string transcribeWhole(SenseVoice& recognizer, const FsmnVad& vad, const Fbank& fbank, const std::vector<float>& pcm16k,
                            const StreamConfig& config, std::string* language, std::string* error);

class Streamer : public std::enable_shared_from_this<Streamer> {
public:
    using Send = std::function<void(const std::string& json)>;

    /// voiceprint 可以是空（没装声纹模型）
    Streamer(SenseVoice& recognizer, const FsmnVad& vad, const Fbank& fbank, InferenceQueue& queue, Send send, int vadThreads,
             SpeakerEncoder* voiceprint = nullptr);
    ~Streamer();

    /// 收到 start 帧时调（只在第一段音频之前有效）
    void configure(const StreamConfig& config);
    const StreamConfig& config() const { return m_config; }

    /// 16 位小端 PCM（采样率按 config）
    void feedPcm16(const uint8_t* bytes, size_t size);
    /// 正在说的这句马上定稿
    void flush();
    /// 流结束：算完剩下的，正在说的定稿
    void finish();
    /// 连接关了：之后的结果都不发了
    void close();

private:
    void process(const std::vector<float>& samples);
    void onFrame(int frame, float probability);
    void beginSegment(int frame);
    void endSegment(int endFrame, const char* reason);
    void submitPartial();
    std::vector<float> slice(int fromFrame, int toFrame) const;
    void trimBuffer();
    void emit(const std::string& json);

    /// 这段的声纹（JSON 数组，保留 4 位小数）；不要 / 算不了时是 null
    nlohmann::json voiceprintOf(const std::vector<float>& samples) const;

    SenseVoice& m_recognizer;
    SpeakerEncoder* m_voiceprint;
    InferenceQueue& m_queue;
    Send m_send;
    StreamConfig m_config;
    bool m_started = false;
    std::unique_ptr<StreamResampler> m_resampler;
    std::unique_ptr<VadStream> m_vad;
    uint64_t m_key;

    // 16 kHz 音频：m_audio[0] 是全局采样号 m_audioBase
    std::vector<float> m_audio;
    int64_t m_audioBase = 0;
    int64_t m_audioTotal = 0;

    // 断句状态（帧 = 10 毫秒）
    std::vector<uint8_t> m_window;   // 最近 20 帧是否在说话
    int m_windowPos = 0, m_windowSum = 0;
    bool m_inSpeech = false;
    int m_segmentStart = 0;          // 这句开始的帧
    int m_silence = 0;               // 连着多少帧没说话
    int m_lastEnd = 0;               // 上一句结束的帧
    int m_lastFrame = -1;
    int m_lastPartialFrame = 0;
    int m_segment = 0;               // 当前 / 下一句的编号

    std::mutex m_mutex;              // 推理线程回来时要碰的字段
    int m_finalizedThrough = 0;      // 编号 ≤ 这个的句子已经定稿（晚到的 partial 丢掉）
    std::string m_lastPartial;
    int m_voiceprintSegment = -1;    // partial 上次带声纹的句子和那时的长度（采样）
    size_t m_voiceprintSamples = 0;
    std::atomic<bool> m_closed { false };
};

} // namespace voice
