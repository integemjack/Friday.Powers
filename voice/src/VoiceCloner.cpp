#include "VoiceCloner.h"

#include "AudioIO.h"
#include "Log.h"

#include "cosyvoice.h"
#include "cosyvoice-frontend.h"
#include "cosyvoice-lowlevel.h"

#include <chrono>

namespace voice {

namespace {
double since(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
} // namespace

void VoiceCloner::configure(const std::string& speechTokenizer, const std::string& campplus)
{
    m_tokenizer = speechTokenizer;
    m_campplus = campplus;
}

bool VoiceCloner::ensureLoaded(std::string* error)
{
    if (m_ctx)
        return true;
    if (m_tokenizer.empty() || m_campplus.empty()) {
        if (error)
            *error = "没配克隆前端模型（speech tokenizer / campplus）";
        return false;
    }
    const auto started = std::chrono::steady_clock::now();
    VLOG_INFO("加载克隆前端：%s + %s", m_tokenizer.c_str(), m_campplus.c_str());
    m_ctx = cosyvoice_frontend_load_from_files(m_tokenizer.c_str(), m_campplus.c_str());
    if (!m_ctx) {
        if (error)
            *error = "克隆前端加载失败（speech tokenizer / campplus 的 ONNX 打不开？）";
        return false;
    }
    VLOG_INFO("克隆前端就绪：%.1f 秒", since(started));
    return true;
}

cosyvoice_prompt_speech* VoiceCloner::clone(const float* speech16k, size_t len16k, const float* speech24k, size_t len24k,
                                             const std::string& promptText, std::string* error)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!ensureLoaded(error))
        return nullptr;
    const auto started = std::chrono::steady_clock::now();
    cosyvoice_prompt_speech_t speech = cosyvoice_frontend_prompt_speech_direct(m_ctx, const_cast<float*>(speech16k), uint32_t(len16k),
                                                                               const_cast<float*>(speech24k), uint32_t(len24k), promptText.c_str());
    if (!speech) {
        if (error)
            *error = "提取音色失败（录音太短、没声音，或底稿和录音对不上？）";
        return nullptr;
    }
    VLOG_INFO("音色克隆：%.1f 秒录音，用时 %.1f 秒", len16k / double(kSampleRate), since(started));
    return speech;
}

VoiceCloner::~VoiceCloner()
{
    if (m_ctx)
        cosyvoice_frontend_free(m_ctx);
}

} // namespace voice
