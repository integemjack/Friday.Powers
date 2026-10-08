// FunASR 口径的 80 维 log-mel fbank（SenseVoice 和 FSMN-VAD 共用）+ LFR 拼帧。
// 声纹（CAM++，SpeakerEncoder）用 Kaldi 默认的 povey 窗（torchaudio.compliance.kaldi.fbank），其余同。
// 数值细节照 FunASR llama.cpp runtime（funasr-sensevoice.cpp / funasr_vad.h）：×32768、帧内去均值、0.97 预加重、
// 汉明窗、512 点 FFT、20–8000 Hz 三角滤波器、log 下限 1.19e-7。滤波器组和 FFT 旋转因子只算一次。
#pragma once

#include <cstddef>
#include <vector>

namespace voice {

class Fbank {
public:
    static constexpr int kSampleRate = 16000;
    static constexpr int kWindow = 400;   // 25 ms
    static constexpr int kShift = 160;    // 10 ms
    static constexpr int kFft = 512;
    static constexpr int kMel = 80;

    enum class Window { Hamming, Povey };

    explicit Fbank(Window window = Window::Hamming);

    /// 一帧：samples 指向 kWindow 个采样（[-1,1]），out 写 kMel 个值
    void frame(const float* samples, float* out) const;
    /// 整段：返回 frames×kMel
    std::vector<float> compute(const float* samples, size_t count, int* frames) const;

    static int frameCount(size_t samples) { return samples < size_t(kWindow) ? 0 : int((samples - kWindow) / kShift + 1); }

    /// LFR：每 n 帧取一次、把 m 帧拼成一帧；左边补 (m-1)/2 个首帧，右边不够补末帧。frames×kMel → outFrames×(m·kMel)
    static std::vector<float> lfr(const float* feat, int frames, int m, int n, int* outFrames);

private:
    struct Weight {
        int bin;
        float value;
    };
    std::vector<float> m_window;
    std::vector<std::vector<Weight>> m_mel;
    std::vector<int> m_reverse;
    std::vector<float> m_cos;
    std::vector<float> m_sin;
};

} // namespace voice
