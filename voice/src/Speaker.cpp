#include "Speaker.h"

#include "AudioIO.h"
#include "Log.h"

#include "json.hpp"

#include <algorithm>
#include <chrono>

namespace voice {

using json = nlohmann::json;

namespace {

std::atomic<uint64_t> g_nextSpeaker { 1 };

} // namespace

Speaker::Speaker(Synthesizer& synthesizer, Send sendText, Send sendBinary)
    : m_synth(synthesizer), m_sendText(std::move(sendText)), m_sendBinary(std::move(sendBinary)), m_id(g_nextSpeaker++)
{
    m_config.sampleRate = m_synth.sampleRate();
}

Speaker::~Speaker()
{
    close();
}

void Speaker::start()
{
    m_thread = std::thread([this] { run(); });
}

void Speaker::configure(const SpeakConfig& config)
{
    SpeakConfig c = config;
    if (!c.voice.empty() && !m_synth.hasVoice(c.voice)) {
        m_sendText(json { { "type", "error" }, { "message", "没有这个音色：" + c.voice + "，用默认的" } }.dump());
        c.voice.clear();
    }
    c.speed = std::clamp(c.speed, 0.5f, 2.0f);
    if (c.sampleRate < 8000 || c.sampleRate > 48000)
        c.sampleRate = m_synth.sampleRate();
    std::lock_guard<std::mutex> lock(m_mutex);
    m_config = c;
}

void Speaker::text(const std::string& delta)
{
    enqueue(m_splitter.push(delta), false);
}

void Speaker::flush()
{
    enqueue(m_splitter.flush(), true);
}

void Speaker::enqueue(const std::vector<std::string>& segments, bool endOfTurn)
{
    if (segments.empty() && !endOfTurn)
        return;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const uint64_t turn = m_turn.load();
        for (const std::string& s : segments)
            m_jobs.push_back({ turn, m_nextIndex++, s, false });
        if (endOfTurn)
            m_jobs.push_back({ turn, -1, {}, true });
    }
    m_wake.notify_one();
}

void Speaker::cancel()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_turn;
        m_jobs.clear();
    }
    m_splitter.reset();
    m_synth.stop(m_id);   // 等正在念的那段停下，之后不会再有旧音频
    m_sendText(json { { "type", "cancelled" } }.dump());
}

void Speaker::close()
{
    if (m_closed.exchange(true))
        return;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_turn;
        m_jobs.clear();
    }
    m_wake.notify_all();
    m_synth.stop(m_id);
    if (m_thread.joinable())
        m_thread.join();
}

void Speaker::run()
{
    while (true) {
        Job job;
        SpeakConfig config;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_wake.wait(lock, [this] { return m_closed || !m_jobs.empty(); });
            if (m_closed)
                return;
            job = std::move(m_jobs.front());
            m_jobs.pop_front();
            config = m_config;
        }
        if (!current(job.turn))
            continue;
        if (job.endOfTurn)
            m_sendText(json { { "type", "done" } }.dump());
        else
            speakOne(job, config);
    }
}

void Speaker::speakOne(const Job& job, const SpeakConfig& config)
{
    m_sendText(json { { "type", "segment_start" }, { "index", job.index }, { "text", job.text } }.dump());
    StreamResampler resampler(m_synth.sampleRate(), config.sampleRate);
    std::vector<float> resampled;
    std::string pcm;
    size_t sent = 0;
    const SpeakResult r = m_synth.speak(m_id, job.text, config.voice, config.speed, config.instruction,
                                        [&](const float* samples, size_t count) {
                                            if (!current(job.turn))
                                                return false;
                                            resampled.clear();
                                            resampler.process(samples, count, resampled);
                                            pcm.clear();
                                            floatToPcm16(resampled.data(), resampled.size(), pcm);
                                            sent += resampled.size();
                                            m_sendBinary(pcm);
                                            return true;
                                        });
    if (!current(job.turn))
        return;   // 被打断的那一轮：cancel 已经回了 cancelled，不再发它的事件
    if (!r.ok && !r.stopped) {
        m_sendText(json { { "type", "error" }, { "index", job.index }, { "message", r.error } }.dump());
        return;
    }
    const double audioSeconds = r.samples / double(std::max(1, m_synth.sampleRate()));
    VLOG_DEBUG("念第 %d 段：%.2f 秒音频，合成 %.3f 秒（首块 %.3f 秒，排队 %.3f 秒）%s：%s", job.index, audioSeconds, r.seconds,
               r.firstAudioSeconds, r.queuedSeconds, r.stopped ? "，被打断" : "", job.text.c_str());
    m_sendText(json { { "type", "segment_end" }, { "index", job.index }, { "audio_ms", int64_t(sent * 1000 / std::max(1, config.sampleRate)) },
                      { "first_audio_ms", int64_t(r.firstAudioSeconds * 1000) }, { "infer_ms", int64_t(r.seconds * 1000) },
                      { "queued_ms", int64_t(r.queuedSeconds * 1000) }, { "stopped", r.stopped } }
                   .dump());
}

} // namespace voice
