#include "Fbank.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace voice {

namespace {
constexpr float kPreemph = 0.97f;
constexpr float kLowHz = 20.0f;
constexpr float kHighHz = 8000.0f;
constexpr float kFloor = 1.1920929e-07f;

inline float mel(float hz) { return 1127.0f * std::log(1.0f + hz / 700.0f); }
} // namespace

Fbank::Fbank()
{
    m_window.resize(kWindow);
    for (int i = 0; i < kWindow; ++i)
        m_window[i] = 0.54f - 0.46f * std::cos(2.0f * float(M_PI) * i / (kWindow - 1));

    const int bins = kFft / 2 + 1;
    const float binHz = float(kSampleRate) / kFft;
    const float low = mel(kLowHz), high = mel(kHighHz), step = (high - low) / (kMel + 1);
    m_mel.resize(kMel);
    for (int m = 0; m < kMel; ++m) {
        const float left = low + m * step, center = low + (m + 1) * step, right = low + (m + 2) * step;
        for (int k = 0; k < bins; ++k) {
            const float f = mel(binHz * k);
            if (f > left && f < right)
                m_mel[m].push_back({ k, f <= center ? (f - left) / (center - left) : (right - f) / (right - center) });
        }
    }

    m_reverse.resize(kFft);
    for (int i = 1, j = 0; i < kFft; ++i) {
        int bit = kFft >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        m_reverse[i] = j;
    }
    m_cos.resize(kFft / 2);
    m_sin.resize(kFft / 2);
    for (int i = 0; i < kFft / 2; ++i) {
        m_cos[i] = float(std::cos(-2.0 * M_PI * i / kFft));
        m_sin[i] = float(std::sin(-2.0 * M_PI * i / kFft));
    }
}

void Fbank::frame(const float* samples, float* out) const
{
    float frame[kWindow];
    double mean = 0;
    for (int i = 0; i < kWindow; ++i) {
        frame[i] = samples[i] * 32768.0f;
        mean += frame[i];
    }
    mean /= kWindow;
    for (int i = 0; i < kWindow; ++i)
        frame[i] -= float(mean);
    for (int i = kWindow - 1; i > 0; --i)
        frame[i] -= kPreemph * frame[i - 1];
    frame[0] -= kPreemph * frame[0];

    float re[kFft], im[kFft];
    for (int i = 0; i < kFft; ++i) {
        const int src = m_reverse[i];
        re[i] = src < kWindow ? frame[src] * m_window[src] : 0.0f;
        im[i] = 0.0f;
    }
    for (int len = 2; len <= kFft; len <<= 1) {
        const int half = len / 2, stride = kFft / len;
        for (int i = 0; i < kFft; i += len) {
            for (int k = 0; k < half; ++k) {
                const float wr = m_cos[k * stride], wi = m_sin[k * stride];
                const float xr = re[i + k + half], xi = im[i + k + half];
                const float vr = xr * wr - xi * wi, vi = xr * wi + xi * wr;
                re[i + k + half] = re[i + k] - vr;
                im[i + k + half] = im[i + k] - vi;
                re[i + k] += vr;
                im[i + k] += vi;
            }
        }
    }
    for (int m = 0; m < kMel; ++m) {
        float energy = 0;
        for (const Weight& w : m_mel[m])
            energy += w.value * (re[w.bin] * re[w.bin] + im[w.bin] * im[w.bin]);
        out[m] = std::log(energy > kFloor ? energy : kFloor);
    }
}

std::vector<float> Fbank::compute(const float* samples, size_t count, int* frames) const
{
    const int n = frameCount(count);
    std::vector<float> out(size_t(n) * kMel);
    for (int t = 0; t < n; ++t)
        frame(samples + size_t(t) * kShift, out.data() + size_t(t) * kMel);
    if (frames)
        *frames = n;
    return out;
}

std::vector<float> Fbank::lfr(const float* feat, int frames, int m, int n, int* outFrames)
{
    if (frames < 1) {
        if (outFrames)
            *outFrames = 0;
        return {};
    }
    const int pad = (m - 1) / 2;
    const int count = (frames + n - 1) / n;
    std::vector<float> out(size_t(count) * m * kMel);
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < m; ++j) {
            int src = i * n + j - pad;
            src = std::clamp(src, 0, frames - 1);
            std::memcpy(&out[(size_t(i) * m + j) * kMel], feat + size_t(src) * kMel, kMel * sizeof(float));
        }
    }
    if (outFrames)
        *outFrames = count;
    return out;
}

} // namespace voice
