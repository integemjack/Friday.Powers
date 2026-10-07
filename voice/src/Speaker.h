// 一路边写边读（一条 WebSocket）：大模型的字一段段进来 → TextSplitter 切句 → 自己的线程按顺序合成 →
// 16 位 PCM 二进制帧发回去。随时可以打断（cancel）：没念的丢掉，正在念的马上停。协议见 docs/voice-protocol.md。
#pragma once

#include "Synthesizer.h"
#include "TextSplitter.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace voice {

struct SpeakConfig {
    std::string voice;          // 空 = 默认音色
    std::string instruction;    // CosyVoice3 指令（如「用四川话说」），空 = 普通朗读
    float speed = 1.0f;
    int sampleRate = 24000;     // 发回去的采样率（模型是 24 kHz，别的就重采样）
};

class Speaker : public std::enable_shared_from_this<Speaker> {
public:
    using Send = std::function<void(const std::string& data)>;

    Speaker(Synthesizer& synthesizer, Send sendText, Send sendBinary);
    ~Speaker();

    /// 起合成线程（要在 shared_ptr 建好之后调）
    void start();
    void configure(const SpeakConfig& config);
    /// 追加一段字（大模型新吐的）
    void text(const std::string& delta);
    /// 这一轮说完：剩下的念完后发 done
    void flush();
    /// 打断：丢掉还没念的、停下正在念的，然后发 cancelled
    void cancel();
    /// 连接关了：停下、等合成线程退出
    void close();

private:
    struct Job {
        uint64_t turn = 0;
        int index = 0;
        std::string text;
        bool endOfTurn = false;
    };

    void enqueue(const std::vector<std::string>& segments, bool endOfTurn);
    void run();
    void speakOne(const Job& job, const SpeakConfig& config);
    bool current(uint64_t turn) const { return !m_closed && m_turn.load() == turn; }

    Synthesizer& m_synth;
    Send m_sendText, m_sendBinary;
    const uint64_t m_id;

    // 连接线程用
    TextSplitter m_splitter;
    int m_nextIndex = 0;

    std::mutex m_mutex;
    std::condition_variable m_wake;
    std::deque<Job> m_jobs;
    SpeakConfig m_config;
    std::atomic<uint64_t> m_turn { 1 };
    std::atomic<bool> m_closed { false };
    std::thread m_thread;
};

} // namespace voice
