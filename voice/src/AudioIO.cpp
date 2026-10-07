#include "AudioIO.h"

// 只要解码器和数据转换（重采样），不要播放 / 录音
#define MA_NO_DEVICE_IO
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_NO_THREADING
#define MA_NO_NODE_GRAPH
#define MA_NO_RESOURCE_MANAGER
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <cstring>

namespace voice {

namespace {

bool readAll(ma_decoder& decoder, std::vector<float>& out)
{
    out.clear();
    std::vector<float> buffer(16000);
    for (;;) {
        ma_uint64 got = 0;
        const ma_result result = ma_decoder_read_pcm_frames(&decoder, buffer.data(), buffer.size(), &got);
        out.insert(out.end(), buffer.begin(), buffer.begin() + std::ptrdiff_t(got));
        if (got < buffer.size() || (result != MA_SUCCESS && result != MA_AT_END))
            break;
    }
    return !out.empty();
}

} // namespace

bool decodeAudio(const void* data, size_t size, std::vector<float>& out, std::string* error)
{
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, kSampleRate);
    ma_decoder decoder;
    if (ma_decoder_init_memory(data, size, &config, &decoder) != MA_SUCCESS) {
        if (error)
            *error = "解不开这段音频（支持 wav / mp3 / flac）";
        return false;
    }
    const bool ok = readAll(decoder, out);
    ma_decoder_uninit(&decoder);
    if (!ok && error)
        *error = "音频是空的";
    return ok;
}

bool decodeAudioFile(const std::string& path, std::vector<float>& out, std::string* error)
{
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, kSampleRate);
    ma_decoder decoder;
    if (ma_decoder_init_file(path.c_str(), &config, &decoder) != MA_SUCCESS) {
        if (error)
            *error = "打不开 / 解不开：" + path + "（支持 wav / mp3 / flac）";
        return false;
    }
    const bool ok = readAll(decoder, out);
    ma_decoder_uninit(&decoder);
    if (!ok && error)
        *error = "音频是空的：" + path;
    return ok;
}

void pcm16ToFloat(const uint8_t* bytes, size_t size, std::vector<float>& out)
{
    const size_t count = size / 2;
    const size_t old = out.size();
    out.resize(old + count);
    for (size_t i = 0; i < count; ++i) {
        const int16_t v = int16_t(uint16_t(bytes[2 * i]) | (uint16_t(bytes[2 * i + 1]) << 8));
        out[old + i] = float(v) / 32768.0f;
    }
}

void floatToPcm16(const float* samples, size_t count, std::string& out)
{
    const size_t old = out.size();
    out.resize(old + count * 2);
    for (size_t i = 0; i < count; ++i) {
        const float v = samples[i] < -1.0f ? -1.0f : (samples[i] > 1.0f ? 1.0f : samples[i]);
        const int16_t s = int16_t(v * 32767.0f);
        out[old + 2 * i] = char(uint16_t(s) & 0xFF);
        out[old + 2 * i + 1] = char(uint16_t(s) >> 8);
    }
}

struct StreamResampler::Impl {
    ma_resampler resampler {};
    bool ready = false;
};

StreamResampler::StreamResampler(int inputRate, int outputRate)
    : m_rate(inputRate), m_outRate(outputRate), m_impl(std::make_unique<Impl>())
{
    if (m_rate != m_outRate && m_rate > 0 && m_outRate > 0) {
        ma_resampler_config config = ma_resampler_config_init(ma_format_f32, 1, ma_uint32(m_rate), ma_uint32(m_outRate), ma_resample_algorithm_linear);
        m_impl->ready = ma_resampler_init(&config, nullptr, &m_impl->resampler) == MA_SUCCESS;
    }
}

StreamResampler::~StreamResampler()
{
    if (m_impl->ready)
        ma_resampler_uninit(&m_impl->resampler, nullptr);
}

void StreamResampler::process(const float* in, size_t count, std::vector<float>& out)
{
    if (!m_impl->ready) {
        out.insert(out.end(), in, in + count);
        return;
    }
    ma_uint64 expected = 0;
    ma_resampler_get_expected_output_frame_count(&m_impl->resampler, count, &expected);
    const size_t old = out.size();
    out.resize(old + size_t(expected) + 16);
    ma_uint64 inFrames = count;
    ma_uint64 outFrames = expected + 16;
    ma_resampler_process_pcm_frames(&m_impl->resampler, in, &inFrames, out.data() + old, &outFrames);
    out.resize(old + size_t(outFrames));
}

} // namespace voice
