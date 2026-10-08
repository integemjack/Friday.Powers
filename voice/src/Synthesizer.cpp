#include "Synthesizer.h"

#include "Log.h"

#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"   // 要在 cosyvoice.h 之后

#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

namespace voice {

namespace {

double since(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

} // namespace

Synthesizer::Synthesizer(const Backend& backend) : m_backend(backend)
{
}

Synthesizer::~Synthesizer()
{
    for (auto& [name, v] : m_voices) {
        if (v.tts)
            cosyvoice_tts_context_free(v.tts);
        if (v.prompt)
            cosyvoice_prompt_free(v.prompt);
        if (v.speech)
            cosyvoice_prompt_speech_free(v.speech);
    }
    if (m_ctx)
        cosyvoice_free(m_ctx);
}

bool Synthesizer::load(const std::string& modelPath, const std::string& voicesDir, std::string* error)
{
    const auto started = std::chrono::steady_clock::now();
    // cosyvoice 会接管传进去的 backend（用完自己释放），所以在选好的设备上另开一个
    ggml_backend_t backend = nullptr;
    if (m_backend.gpu())
        backend = ggml_backend_dev_init(ggml_backend_get_device(m_backend.gpu()), nullptr);
    if (!backend)
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    cosyvoice_context_params_v4_t params {};
    cosyvoice_init_default_context_params(&params.base_params.base_params.base_params);
    params.base_params.base_params.n_workers = 1;
    params.base_params.dit_kv_cache_type =
        COSYVOICE_MAKE_SEPARATE_KV_CACHE(COSYVOICE_KV_CACHE_TYPE_Q8_0, COSYVOICE_KV_CACHE_TYPE_Q4_0, COSYVOICE_KV_CACHE_TYPE_Q8_0);
    params.base_params.dit_allow_kv_cache_fallback = true;
    params.strict_seed_mode = false;   // 不要求同一种子逐位相同，省一次预填充
    // Metal 上 LLM 的 flash attention 算错（2026-10-08，M1 + ggml v0.26.0：念出来是乱语、一直念到长度上限，KV 缓存换 F16 也一样；
    // 关掉就对了）。flow 的 flash attention 没问题，留着（全关更慢）
    if (m_backend.isMetal())
        params.base_params.base_params.base_params.llm_use_flash_attn = false;
    m_ctx = cosyvoice_load_from_file_ext(modelPath.c_str(), &params.base_params.base_params.base_params, backend,
                                         uint32_t(m_backend.threads()), COSYVOICE_CONTEXT_PARAMS_V4_VERSION);
    if (!m_ctx) {
        if (error)
            *error = "打不开合成模型：" + modelPath;
        return false;
    }
    m_path = modelPath;
    m_sampleRate = int(cosyvoice_get_sample_rate(m_ctx));

    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(fs::u8path(voicesDir), ec)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".gguf")
            continue;
        const std::string name = entry.path().stem().u8string();
        Voice v;
        v.speech = cosyvoice_prompt_speech_load_from_file(entry.path().u8string().c_str());
        if (v.speech)
            v.prompt = cosyvoice_prompt_init_from_prompt_speech(m_ctx, v.speech);
        if (v.prompt)
            v.tts = cosyvoice_tts_context_new(m_ctx, v.prompt);
        if (!v.tts) {
            VLOG_WARN("音色 %s 加载失败，跳过", entry.path().u8string().c_str());
            if (v.prompt)
                cosyvoice_prompt_free(v.prompt);
            if (v.speech)
                cosyvoice_prompt_speech_free(v.speech);
            continue;
        }
        m_voices[name] = v;
    }
    if (m_voices.empty()) {
        if (error)
            *error = "没有可用的音色（" + voicesDir + " 里要有 <名字>.gguf）";
        return false;
    }
    m_defaultVoice = m_voices.count("default") ? "default" : m_voices.begin()->first;
    VLOG_INFO("CosyVoice3 已加载：%s（%d Hz，%d 个音色，每块 %u 个语音 token，%.1f 秒）", modelPath.c_str(), m_sampleRate,
              int(m_voices.size()), cosyvoice_get_chunk_tokens(m_ctx), since(started));
    return true;
}

void Synthesizer::setChunkTokens(unsigned tokens)
{
    if (m_ctx && tokens > 0)
        cosyvoice_set_chunk_tokens(m_ctx, tokens);
}

unsigned Synthesizer::chunkTokens() const
{
    return m_ctx ? cosyvoice_get_chunk_tokens(m_ctx) : 0;
}

std::vector<std::string> Synthesizer::voices() const
{
    std::vector<std::string> names;
    for (const auto& [name, v] : m_voices)
        names.push_back(name);
    return names;
}

SpeakResult Synthesizer::speak(uint64_t owner, const std::string& text, const std::string& voice, float speed,
                               const std::string& instruction, const OnAudio& onAudio)
{
    SpeakResult result;
    if (!m_ctx) {
        result.error = "合成模型没加载";
        return result;
    }
    const auto it = m_voices.find(voice.empty() ? m_defaultVoice : voice);
    if (it == m_voices.end()) {
        result.error = "没有这个音色：" + voice;
        return result;
    }
    const auto queued = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(m_mutex);
    result.queuedSeconds = since(queued);
    {
        std::lock_guard<std::mutex> ownerLock(m_ownerMutex);
        m_owner = owner;
    }

    struct State {
        const OnAudio* onAudio;
        std::chrono::steady_clock::time_point started;
        size_t samples = 0;
        double firstAudio = 0;
        bool aborted = false;
    } state { &onAudio, std::chrono::steady_clock::now() };
    const auto callback = [](const float* audio, uint32_t count, void* user) -> bool {
        auto* s = static_cast<State*>(user);
        if (s->samples == 0)
            s->firstAudio = since(s->started);
        s->samples += count;
        if (!(*s->onAudio)(audio, count)) {
            s->aborted = true;
            return false;
        }
        return true;
    };
    const float rate = speed > 0.25f && speed < 4.0f ? speed : 1.0f;
    const bool ok = instruction.empty()
        ? cosyvoice_tts_zero_shot_stream(it->second.tts, text.c_str(), rate, callback, &state)
        : cosyvoice_tts_instruct_stream(it->second.tts, text.c_str(), instruction.c_str(), rate, callback, &state);
    {
        std::lock_guard<std::mutex> ownerLock(m_ownerMutex);
        m_owner = 0;
    }
    result.seconds = since(state.started);
    result.firstAudioSeconds = state.firstAudio;
    result.samples = state.samples;
    result.stopped = state.aborted || (!ok && state.samples > 0);
    result.ok = ok || result.stopped;
    if (!result.ok)
        result.error = "合成失败";
    return result;
}

void Synthesizer::stop(uint64_t owner)
{
    std::lock_guard<std::mutex> ownerLock(m_ownerMutex);
    if (m_ctx && owner != 0 && m_owner == owner)
        cosyvoice_request_stop(m_ctx);
}

} // namespace voice
