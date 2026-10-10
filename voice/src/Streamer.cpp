#include "Streamer.h"

#include "Log.h"

#include "json.hpp"

#include <cmath>

#include <algorithm>
#include <cctype>

namespace voice {

using json = nlohmann::json;

namespace {

constexpr int kFrameSamples = Fbank::kShift;   // 10 毫秒一帧
constexpr int kWindowFrames = 20;              // 看最近 200 毫秒
constexpr int kStartFrames = 12;               // 其中 ≥ 120 毫秒在说话 → 开口
constexpr int kTailFrames = 12;                // 定稿时句尾多留 120 毫秒
constexpr int kMinPartialFrames = 40;          // 一句至少 0.4 秒才出实时字幕

std::atomic<uint64_t> g_nextKey { 1 };

/// 一段 UTF-8 文字的末字 / 首字是不是中日韩（接句子时不加空格）
bool isCjkLead(const std::string& s, bool last)
{
    if (s.empty())
        return false;
    size_t i = last ? s.size() - 1 : 0;
    if (last)
        while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80)
            --i;
    const unsigned char c = static_cast<unsigned char>(s[i]);
    return c >= 0xE3 && c <= 0xEF;   // U+3000 以上的三字节字符：中日韩文字和全角标点
}

/// 实时字幕值不值得发：刚开口那零点几秒里 VAD 常把气声、杂音当成说话，模型会猜出「Yeah.」「Hい。」之类，
/// 这时情绪标签多是 EMO_UNKNOWN；还有只剩标点的。一句的第一条字幕要是只有一个拉丁词，也等下一条再说。（定稿不过这道）
bool worthShowing(const Transcript& t, bool firstInSegment)
{
    if (t.emotion == "EMO_UNKNOWN")
        return false;
    int letters = 0, cjk = 0, words = 0;
    bool inWord = false;
    for (size_t i = 0; i < t.text.size();) {
        const unsigned char c = static_cast<unsigned char>(t.text[i]);
        uint32_t cp = c;
        size_t len = 1;
        if (c >= 0xF0) {
            cp = c & 0x07, len = 4;
        } else if (c >= 0xE0) {
            cp = c & 0x0F, len = 3;
        } else if (c >= 0xC0) {
            cp = c & 0x1F, len = 2;
        }
        for (size_t k = 1; k < len && i + k < t.text.size(); ++k)
            cp = (cp << 6) | (static_cast<unsigned char>(t.text[i + k]) & 0x3F);
        i += len;
        const bool latin = (cp < 0x80 && std::isalnum(int(cp))) || (cp >= 0xC0 && cp < 0x2000);
        const bool wide = (cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x3400 && cp <= 0x9FFF) || (cp >= 0xAC00 && cp <= 0xD7AF)
            || (cp >= 0xF900 && cp <= 0xFAFF);
        letters += latin ? 1 : 0;
        cjk += wide ? 1 : 0;
        if (latin && !inWord)
            ++words;
        inWord = latin;
    }
    if (letters + cjk == 0)
        return false;
    if (firstInSegment && cjk == 0 && words < 2)
        return false;
    return true;
}

/// 定稿值不值得发：整句都是杂音（敲键盘、咳嗽、背景音乐）时模型也会猜出「Yeah。」「嗯。」——
/// 事件标签不是 Speech 的、情绪标签是 EMO_UNKNOWN 且只有一个词 / 两个字以内的，当成没说话（发 speech_end）
bool worthFinal(const Transcript& t)
{
    if (!t.event.empty() && t.event != "Speech")
        return false;
    if (t.emotion != "EMO_UNKNOWN")
        return true;
    int cjk = 0, words = 0;
    bool inWord = false;
    for (size_t i = 0; i < t.text.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(t.text[i]);
        if (c >= 0xE3 && c <= 0xEF) {
            ++cjk;   // 三字节字符（中日韩文字和全角标点，标点也算一个，宽一点）
            i += 2;
            inWord = false;
        } else if (std::isalnum(c)) {
            if (!inWord)
                ++words;
            inWord = true;
        } else {
            inWord = false;
        }
    }
    return cjk > 3 || words > 1;
}

struct Machine {
    const StreamConfig& config;
    std::vector<uint8_t> window = std::vector<uint8_t>(kWindowFrames, 0);
    int pos = 0, sum = 0, silence = 0, start = 0, lastEnd = 0;
    bool inSpeech = false;

    /// 返回：0 无事；1 开口（start 有效）；2 说完（end 写进 *end）
    int step(int frame, float probability, int* end)
    {
        const uint8_t speech = probability >= config.threshold ? 1 : 0;
        sum += int(speech) - int(window[size_t(pos)]);
        window[size_t(pos)] = speech;
        pos = (pos + 1) % kWindowFrames;
        if (!inSpeech) {
            if (sum >= kStartFrames) {
                inSpeech = true;
                silence = 0;
                start = std::max(lastEnd, frame - kWindowFrames - config.prerollMs / 10);
                start = std::max(start, 0);
                return 1;
            }
            return 0;
        }
        silence = speech ? 0 : silence + 1;
        if (silence >= std::max(1, config.endSilenceMs / 10)) {
            *end = frame - silence + kTailFrames;
            inSpeech = false;
            lastEnd = *end;
            return 2;
        }
        if (frame - start >= std::max(100, config.maxSegmentMs / 10)) {
            *end = frame;
            inSpeech = false;
            lastEnd = *end;
            return 2;
        }
        return 0;
    }
};

} // namespace

std::vector<std::pair<int, int>> segmentFrames(const std::vector<float>& probability, const StreamConfig& config)
{
    std::vector<std::pair<int, int>> segments;
    Machine machine { config };
    for (int frame = 0; frame < int(probability.size()); ++frame) {
        int end = 0;
        if (machine.step(frame, probability[size_t(frame)], &end) == 2)
            segments.push_back({ machine.start, end });
    }
    if (machine.inSpeech)
        segments.push_back({ machine.start, int(probability.size()) });
    return segments;
}

std::string finalText(const Transcript& sensevoice, Whisper* whisper, const StreamConfig& config, const float* pcm, size_t count,
                      std::string* model, double* whisperSeconds)
{
    if (model)
        *model = "sensevoice";
    if (!whisper || !config.whisper || sensevoice.text.empty())
        return sensevoice.text;
    const Whisper::Result w = whisper->transcribe(pcm, count, sensevoice.language, config.prompt);
    if (whisperSeconds)
        *whisperSeconds = w.seconds;
    if (!w.ok || !Whisper::plausible(w.text, sensevoice.text)) {
        VLOG_DEBUG("Whisper 的不用（%s）：%s；用 SenseVoice 的：%s", w.ok ? "不像样" : w.error.c_str(), w.text.c_str(), sensevoice.text.c_str());
        return sensevoice.text;
    }
    if (model)
        *model = "whisper";
    // 中文用 SenseVoice 的、英文词用 Whisper 的（见 Whisper::merge）
    return Whisper::merge(sensevoice.text, w.text, sensevoice.language);
}

std::string transcribeWhole(SenseVoice& recognizer, const FsmnVad& vad, const Fbank& fbank, const std::vector<float>& pcm,
                            const StreamConfig& config, std::string* language, std::string* error, Whisper* whisper)
{
    const auto started = std::chrono::steady_clock::now();
    VadStream stream(vad, fbank, 4);
    std::vector<float> probability = stream.feed(pcm.data(), pcm.size());
    const std::vector<float> tail = stream.finish();
    probability.insert(probability.end(), tail.begin(), tail.end());
    const auto segments = segmentFrames(probability, config);
    VLOG_DEBUG("断句：%d 句，用时 %.3f 秒", int(segments.size()),
               std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    std::string text;
    for (const auto& [from, to] : segments) {
        const size_t begin = std::min(pcm.size(), size_t(from) * kFrameSamples);
        const size_t end = std::min(pcm.size(), size_t(to) * kFrameSamples + Fbank::kWindow);
        if (end <= begin + Fbank::kWindow)
            continue;
        const Transcript t = recognizer.transcribe(pcm.data() + begin, end - begin, config.language, config.itn);
        VLOG_DEBUG("  %.2f–%.2f 秒：识别 %.3f 秒", begin / double(kSampleRate), end / double(kSampleRate), t.seconds);
        if (!t.ok) {
            if (error)
                *error = t.error;
            return {};
        }
        if (t.text.empty())
            continue;
        if (language && language->empty() && !t.language.empty() && t.language != "nospeech")
            *language = t.language;
        double whisperSeconds = 0;
        const std::string sentence = finalText(t, whisper, config, pcm.data() + begin, end - begin, nullptr, &whisperSeconds);
        if (whisperSeconds > 0)
            VLOG_DEBUG("    Whisper %.3f 秒：%s", whisperSeconds, sentence.c_str());
        if (!text.empty() && !isCjkLead(text, true) && !isCjkLead(sentence, false))
            text += ' ';
        text += sentence;
    }
    return text;
}

Streamer::Streamer(SenseVoice& recognizer, const FsmnVad& vad, const Fbank& fbank, InferenceQueue& queue, Send send, int vadThreads,
                   SpeakerEncoder* voiceprint, Whisper* whisper)
    : m_recognizer(recognizer)
    , m_voiceprint(voiceprint)
    , m_whisper(whisper)
    , m_queue(queue)
    , m_send(std::move(send))
    , m_vad(std::make_unique<VadStream>(vad, fbank, vadThreads))
    , m_key(g_nextKey++)
{
    m_window.assign(kWindowFrames, 0);
}

Streamer::~Streamer()
{
    m_queue.cancelPartial(m_key);
}

void Streamer::configure(const StreamConfig& config)
{
    if (m_started)
        return;
    m_config = config;
    if (!SenseVoice::supportsLanguage(m_config.language))
        m_config.language = "auto";
    m_config.partialIntervalMs = std::clamp(m_config.partialIntervalMs, 100, 5000);
    m_config.endSilenceMs = std::clamp(m_config.endSilenceMs, 150, 5000);
    m_config.maxSegmentMs = std::clamp(m_config.maxSegmentMs, 2000, 30000);
    m_config.prerollMs = std::clamp(m_config.prerollMs, 0, 1000);
    m_config.threshold = std::clamp(m_config.threshold, 0.05f, 0.99f);
    if (m_config.sampleRate < 8000 || m_config.sampleRate > 96000)
        m_config.sampleRate = kSampleRate;
}

void Streamer::feedPcm16(const uint8_t* bytes, size_t size)
{
    m_started = true;
    if (!m_resampler)
        m_resampler = std::make_unique<StreamResampler>(m_config.sampleRate);
    std::vector<float> raw;
    pcm16ToFloat(bytes, size, raw);
    std::vector<float> samples;
    samples.reserve(raw.size() * kSampleRate / size_t(std::max(1, m_config.sampleRate)) + 16);
    m_resampler->process(raw.data(), raw.size(), samples);
    process(samples);
}

void Streamer::process(const std::vector<float>& samples)
{
    if (!samples.empty()) {
        m_audio.insert(m_audio.end(), samples.begin(), samples.end());
        m_audioTotal += int64_t(samples.size());
    }
    const std::vector<float> probability = m_vad->feed(samples.data(), samples.size());
    const int first = m_vad->framesDone() - int(probability.size());
    for (size_t i = 0; i < probability.size(); ++i)
        onFrame(first + int(i), probability[i]);
    if (m_inSpeech && m_config.partials && m_lastFrame - m_lastPartialFrame >= m_config.partialIntervalMs / 10
        && m_lastFrame - m_segmentStart >= kMinPartialFrames)
        submitPartial();
    trimBuffer();
}

void Streamer::onFrame(int frame, float probability)
{
    m_lastFrame = frame;
    const uint8_t speech = probability >= m_config.threshold ? 1 : 0;
    m_windowSum += int(speech) - int(m_window[size_t(m_windowPos)]);
    m_window[size_t(m_windowPos)] = speech;
    m_windowPos = (m_windowPos + 1) % kWindowFrames;
    if (!m_inSpeech) {
        if (m_windowSum >= kStartFrames)
            beginSegment(frame);
        return;
    }
    m_silence = speech ? 0 : m_silence + 1;
    if (m_silence >= std::max(1, m_config.endSilenceMs / 10))
        endSegment(frame - m_silence + kTailFrames, "silence");
    else if (frame - m_segmentStart >= m_config.maxSegmentMs / 10)
        endSegment(frame, "max");
}

void Streamer::beginSegment(int frame)
{
    m_inSpeech = true;
    m_silence = 0;
    int start = std::max(m_lastEnd, frame - kWindowFrames - m_config.prerollMs / 10);
    start = std::max<int>(start, int(m_audioBase / kFrameSamples));
    m_segmentStart = std::max(start, 0);
    m_lastPartialFrame = frame;
    ++m_segment;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_lastPartial.clear();
    }
    emit(json { { "type", "speech_start" }, { "segment", m_segment }, { "start_ms", int64_t(m_segmentStart) * 10 } }.dump());
}

std::vector<float> Streamer::slice(int fromFrame, int toFrame) const
{
    const int64_t begin = std::max<int64_t>(m_audioBase, int64_t(fromFrame) * kFrameSamples);
    const int64_t end = std::min<int64_t>(m_audioTotal, int64_t(toFrame) * kFrameSamples + (Fbank::kWindow - Fbank::kShift));
    if (end <= begin)
        return {};
    return std::vector<float>(m_audio.begin() + std::ptrdiff_t(begin - m_audioBase), m_audio.begin() + std::ptrdiff_t(end - m_audioBase));
}

void Streamer::endSegment(int endFrame, const char* reason)
{
    endFrame = std::min(endFrame, m_lastFrame + 1);
    std::vector<float> samples = slice(m_segmentStart, endFrame);
    const int segment = m_segment;
    const int64_t startMs = int64_t(m_segmentStart) * 10, endMs = int64_t(endFrame) * 10;
    m_inSpeech = false;
    m_silence = 0;
    m_lastEnd = endFrame;
    m_queue.cancelPartial(m_key);
    VLOG_DEBUG("第 %d 句说完（%s）：%.2f–%.2f 秒", segment, reason, startMs / 1000.0, endMs / 1000.0);

    auto self = shared_from_this();
    const auto detected = std::chrono::steady_clock::now();
    m_queue.submitFinal([self, samples = std::move(samples), segment, startMs, endMs, detected] {
        const Transcript t = self->m_recognizer.transcribe(samples.data(), samples.size(), self->m_config.language, self->m_config.itn);
        {
            std::lock_guard<std::mutex> lock(self->m_mutex);
            self->m_finalizedThrough = std::max(self->m_finalizedThrough, segment);
        }
        if (!t.ok) {
            self->emit(json { { "type", "error" }, { "segment", segment }, { "message", t.error } }.dump());
        } else if (t.text.empty() || !worthFinal(t)) {
            VLOG_DEBUG("第 %d 句不算说话：%s [%s/%s]", segment, t.text.c_str(), t.emotion.c_str(), t.event.c_str());
            self->emit(json { { "type", "speech_end" }, { "segment", segment } }.dump());
        } else {
            // v0.3 是人声：定稿交给 Whisper（中英混说准），不像样就用 SenseVoice 的
            std::string model;
            double whisperSeconds = 0;
            const std::string text = finalText(t, self->m_whisper, self->m_config, samples.data(), samples.size(), &model, &whisperSeconds);
            const int64_t latency = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - detected).count();
            VLOG_DEBUG("第 %d 句定稿（%s，%lld 毫秒）：%s", segment, model.c_str(), static_cast<long long>(latency), text.c_str());
            json final { { "type", "final" }, { "segment", segment }, { "text", text }, { "start_ms", startMs }, { "end_ms", endMs },
                         { "language", t.language }, { "emotion", t.emotion }, { "event", t.event },
                         { "latency_ms", latency }, { "infer_ms", int64_t(t.seconds * 1000) }, { "model", model } };
            if (model == "whisper" || whisperSeconds > 0)
                final["whisper_ms"] = int64_t(whisperSeconds * 1000);
            if (json speaker = self->voiceprintOf(samples); !speaker.is_null())
                final["speaker"] = std::move(speaker);
            self->emit(final.dump());
        }
    });
}

void Streamer::submitPartial()
{
    m_lastPartialFrame = m_lastFrame;
    std::vector<float> samples = slice(m_segmentStart, m_lastFrame + 1);
    const int segment = m_segment;
    auto self = shared_from_this();
    m_queue.submitPartial(m_key, [self, samples = std::move(samples), segment] {
        {
            std::lock_guard<std::mutex> lock(self->m_mutex);
            if (segment <= self->m_finalizedThrough || self->m_closed)
                return;
        }
        const Transcript t = self->m_recognizer.transcribe(samples.data(), samples.size(), self->m_config.language, self->m_config.itn);
        VLOG_DEBUG("partial #%d（%.2f 秒）：%s [%s/%s/%s]", segment, samples.size() / double(kSampleRate), t.text.c_str(),
                   t.language.c_str(), t.emotion.c_str(), t.event.c_str());
        if (!t.ok || t.text.empty())
            return;
        bool withVoiceprint = false;
        {
            std::lock_guard<std::mutex> lock(self->m_mutex);
            // 定稿和实时字幕在同一条推理线程上排队，不会同时跑：这里看到还没定稿，就一定在定稿之前发出去
            if (segment <= self->m_finalizedThrough || t.text == self->m_lastPartial || !worthShowing(t, self->m_lastPartial.empty()))
                return;
            self->m_lastPartial = t.text;
            // 声纹：说到 1 秒以上、比上次带的时候又多了 0.8 秒
            if (self->m_config.speaker && samples.size() >= size_t(kSampleRate)
                && (self->m_voiceprintSegment != segment || samples.size() >= self->m_voiceprintSamples + size_t(kSampleRate * 8 / 10))) {
                self->m_voiceprintSegment = segment;
                self->m_voiceprintSamples = samples.size();
                withVoiceprint = true;
            }
        }
        json partial { { "type", "partial" }, { "segment", segment }, { "text", t.text } };
        if (withVoiceprint) {
            if (json speaker = self->voiceprintOf(samples); !speaker.is_null())
                partial["speaker"] = std::move(speaker);
        }
        self->emit(partial.dump());
    });
}

void Streamer::trimBuffer()
{
    // 只留下还会用到的：在说话时从这句开头；没说话时最近 window + preroll
    const int64_t keepFrame = m_inSpeech ? m_segmentStart : int64_t(m_lastFrame) - kWindowFrames - m_config.prerollMs / 10 - 10;
    const int64_t keepFrom = std::max<int64_t>(0, keepFrame) * kFrameSamples;
    if (keepFrom > m_audioBase + kSampleRate) {
        m_audio.erase(m_audio.begin(), m_audio.begin() + std::ptrdiff_t(keepFrom - m_audioBase));
        m_audioBase = keepFrom;
    }
}

void Streamer::flush()
{
    if (m_inSpeech)
        endSegment(m_lastFrame + 1, "flush");
}

void Streamer::finish()
{
    const std::vector<float> probability = m_vad->finish();
    const int first = m_vad->framesDone() - int(probability.size());
    for (size_t i = 0; i < probability.size(); ++i)
        onFrame(first + int(i), probability[i]);
    if (m_inSpeech)
        endSegment(m_lastFrame + 1, "end");
}

void Streamer::close()
{
    m_closed = true;
    m_queue.cancelPartial(m_key);
}

json Streamer::voiceprintOf(const std::vector<float>& samples) const
{
    if (!m_config.speaker || !m_voiceprint || !m_voiceprint->loaded())
        return nullptr;
    const SpeakerEmbedding e = m_voiceprint->embed(samples.data(), samples.size());
    if (!e.ok) {
        VLOG_DEBUG("声纹没算：%s", e.error.c_str());
        return nullptr;
    }
    json values = json::array();
    for (const float v : e.vector)
        values.push_back(std::round(double(v) * 1e4) / 1e4);
    return values;
}

void Streamer::emit(const std::string& message)
{
    if (!m_closed)
        m_send(message);
}

} // namespace voice
