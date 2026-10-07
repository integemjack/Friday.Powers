// 音频：文件 / 内存 → 16 kHz 单声道 float（miniaudio：wav / mp3 / flac，任意采样率、声道）；流式重采样。
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace voice {

constexpr int kSampleRate = 16000;

/// 解码一段内存里的音频文件；失败返回 false
bool decodeAudio(const void* data, size_t size, std::vector<float>& out, std::string* error);
bool decodeAudioFile(const std::string& path, std::vector<float>& out, std::string* error);

/// 16 位小端 PCM → float
void pcm16ToFloat(const uint8_t* bytes, size_t size, std::vector<float>& out);
/// float → 16 位小端 PCM（追加到 out 后面，超出 ±1 的截断）
void floatToPcm16(const float* samples, size_t count, std::string& out);

/// 单声道 float 流换采样率（默认换成 16 kHz；两边一样时原样通过）
class StreamResampler {
public:
    explicit StreamResampler(int inputRate, int outputRate = kSampleRate);
    ~StreamResampler();
    StreamResampler(const StreamResampler&) = delete;
    StreamResampler& operator=(const StreamResampler&) = delete;
    void process(const float* in, size_t count, std::vector<float>& out);
    int inputRate() const { return m_rate; }
    int outputRate() const { return m_outRate; }

private:
    int m_rate;
    int m_outRate;
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace voice
