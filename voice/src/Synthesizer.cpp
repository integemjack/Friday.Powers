#include "Synthesizer.h"

#include "Log.h"

#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"   // 要在 cosyvoice.h 之后
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <mutex>

namespace fs = std::filesystem;

namespace voice {

namespace {

double since(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// 流式分块（v0.5.1）。参考耗时：RTX 2060 + Vulkan、CosyVoice3 Q8、142 KB 的克隆音色（2026-11-10 用 speak --chunk-tokens 28/56/84/112 量的）：
// 一段的第一块约 0.77 + 0.0083·n 秒，之后每块约 1.10 + 0.0098·n 秒（n = 这块的语音 token 数）。固定开销大（每块 flow 都带着音色的
// 提示 mel 算一遍），块小了总速度跟不上：28 个一块时实时率 1.6，84 个 0.76，112 个 0.63。别的机器、音色乘各自学出来的系数
constexpr unsigned kChunkStep = 14;
constexpr unsigned kNextChunk = 84;     // 第一块之后的块
constexpr unsigned kMaxChunk = 140;     // 念的那头剩得多时最大到这么多
constexpr double kTokenSeconds = 0.04;  // CosyVoice3 的语音 token 是 25 Hz：1 倍速一个 0.04 秒

double firstChunkCost(unsigned n)
{
    return 0.77 + 0.0083 * n;
}

double nextChunkCost(unsigned n)
{
    return 1.10 + 0.0098 * n;
}

} // namespace

unsigned Synthesizer::pickFirstChunk(double scale, double secondsPerToken, double ahead) const
{
    // 念的那头剩的够等一个大块：第一声早晚无所谓，挑剩的盖得住的最大块
    if (ahead >= firstChunkCost(kNextChunk) * scale) {
        unsigned n = kNextChunk;
        while (n + kChunkStep <= kMaxChunk && firstChunkCost(n + kChunkStep) * scale <= ahead)
            n += kChunkStep;
        return n;
    }
    // 不然挑最小的、念完它（加上剩的）时下一个大块已经合成好的
    for (unsigned n = m_minChunk; n < kNextChunk; n += kChunkStep) {
        if (ahead + n * secondsPerToken >= nextChunkCost(kNextChunk) * scale)
            return n;
    }
    return kNextChunk;
}

unsigned Synthesizer::pickNextChunk(double scale, double secondsPerToken, double ahead) const
{
    // 大块也跟不上实时（慢机器、CPU 吃满）：放空免不了，用最大的块，总速度最快（客户端攒够再放时等得最少）
    if (nextChunkCost(kNextChunk) * scale > kNextChunk * secondsPerToken)
        return kMaxChunk;
    unsigned n = kNextChunk;
    while (n + kChunkStep <= kMaxChunk && nextChunkCost(n + kChunkStep) * scale <= ahead)
        n += kChunkStep;
    return n;
}

Synthesizer::Synthesizer(const Backend& backend) : m_backend(backend)
{
}

Synthesizer::~Synthesizer()
{
    for (auto& [name, v] : m_voices)
        freeVoice(v);
    if (m_ctx)
        cosyvoice_free(m_ctx);
}

bool Synthesizer::load(const std::string& modelPath, const std::string& voicesDir, std::string* error, const std::string& userVoicesDir)
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
    m_minChunk = std::max(kChunkStep, cosyvoice_get_chunk_tokens(m_ctx));

    std::error_code ec;
    if (!scanVoices(voicesDir, false, error))
        return false;
    if (!userVoicesDir.empty() && fs::exists(fs::u8path(userVoicesDir), ec)) {
        std::string userError;
        scanVoices(userVoicesDir, true, &userError);   // 用户音色坏了不拖累启动，跳过就行
        if (!userError.empty())
            VLOG_WARN("用户音色目录扫出了问题（忽略）：%s", userError.c_str());
    }
    if (m_voices.empty()) {
        if (error)
            *error = "没有可用的音色（" + voicesDir + " 里要有 <名字>.gguf）";
        return false;
    }
    m_defaultVoice = m_voices.count("default") ? "default" : m_voices.begin()->first;
    VLOG_INFO("CosyVoice3 已加载：%s（%d Hz，%d 个音色（其中 %d 个用户克隆），流式分块 %u–%u 个语音 token 自适应，%.1f 秒）", modelPath.c_str(),
              m_sampleRate, int(m_voices.size()),
              int(std::count_if(m_voices.begin(), m_voices.end(), [](const auto& v) { return v.second.user; })), m_minChunk, kMaxChunk,
              since(started));
    return true;
}

bool Synthesizer::scanVoices(const std::string& dir, bool user, std::string* error)
{
    if (dir.empty())
        return true;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(fs::u8path(dir), ec)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".gguf")
            continue;
        const std::string name = entry.path().stem().u8string();
        if (name.empty() || name[0] == '.' || m_voices.count(name))
            continue;   // 点开头的是正在写的临时文件；用户音色和内置同名：内置优先
        Voice v;
        v.user = user;
        v.file = entry.path().u8string();
        if (isVoiceFile(v.file))
            v.speech = cosyvoice_prompt_speech_load_from_file(v.file.c_str());
        if (!initVoice(v)) {
            VLOG_WARN("音色 %s 加载失败，跳过", entry.path().u8string().c_str());
            continue;
        }
        m_voices[name] = v;
    }
    if (ec && error)
        *error = "打不开音色目录：" + dir;
    return !ec;
}

bool Synthesizer::isVoiceFile(const std::string& path)
{
    ggml_context* meta = nullptr;
    gguf_init_params params { true, &meta };
    gguf_context* gguf = gguf_init_from_file(path.c_str(), params);   // ggml_fopen：Windows 上 UTF-8 路径也认
    if (!gguf)
        return false;
    const auto has = [&](const char* name, ggml_type type, int dims) {
        const ggml_tensor* t = meta ? ggml_get_tensor(meta, name) : nullptr;
        return t && t->type == type && ggml_n_dims(t) <= dims && ggml_nelements(t) > 0;
    };
    const bool ok = has("feat", GGML_TYPE_F32, 2) && has("embedding", GGML_TYPE_F32, 2) && has("tokens", GGML_TYPE_I32, 1)
                    && has("text", GGML_TYPE_I8, 1);
    gguf_free(gguf);
    if (meta)
        ggml_free(meta);
    return ok;
}

bool Synthesizer::initVoice(Voice& v)
{
    if (v.speech)
        v.prompt = cosyvoice_prompt_init_from_prompt_speech(m_ctx, v.speech);
    if (v.prompt)
        v.tts = cosyvoice_tts_context_new(m_ctx, v.prompt);
    if (v.tts)
        return true;
    freeVoice(v);
    return false;
}

void Synthesizer::freeVoice(Voice& v)
{
    if (v.tts)
        cosyvoice_tts_context_free(v.tts);
    if (v.prompt)
        cosyvoice_prompt_free(v.prompt);
    if (v.speech)
        cosyvoice_prompt_speech_free(v.speech);
    v.tts = nullptr;
    v.prompt = nullptr;
    v.speech = nullptr;
}

void Synthesizer::setChunkTokens(unsigned tokens)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_fixedChunk = tokens;
}

std::vector<std::string> Synthesizer::voices() const
{
    std::lock_guard<std::mutex> lock(m_voicesMutex);
    std::vector<std::string> names;
    for (const auto& [name, v] : m_voices)
        names.push_back(name);
    return names;
}

std::vector<std::string> Synthesizer::userVoices() const
{
    std::lock_guard<std::mutex> lock(m_voicesMutex);
    std::vector<std::string> names;
    for (const auto& [name, v] : m_voices) {
        if (v.user)
            names.push_back(name);
    }
    return names;
}

bool Synthesizer::hasVoice(const std::string& voice) const
{
    std::lock_guard<std::mutex> lock(m_voicesMutex);
    return m_voices.count(voice) > 0;
}

bool Synthesizer::isUserVoice(const std::string& voice) const
{
    std::lock_guard<std::mutex> lock(m_voicesMutex);
    const auto it = m_voices.find(voice);
    return it != m_voices.end() && it->second.user;
}

std::string Synthesizer::defaultVoice() const
{
    std::lock_guard<std::mutex> lock(m_voicesMutex);
    return m_defaultVoice;
}

bool Synthesizer::addUserVoice(const std::string& name, cosyvoice_prompt_speech* speech, const std::string& file, bool replace,
                               std::string* error)
{
    Voice v;
    v.user = true;
    v.file = file;
    v.speech = speech;
    if (!m_ctx) {
        freeVoice(v);
        if (error)
            *error = "合成模型没加载";
        return false;
    }
    std::lock_guard<std::mutex> lock(m_mutex);   // 和正在念的互斥：等它念完再加（建 prompt 要用模型的分词）
    {
        std::lock_guard<std::mutex> voicesLock(m_voicesMutex);
        const auto it = m_voices.find(name);
        if (it != m_voices.end() && (!replace || !it->second.user)) {
            freeVoice(v);
            if (error)
                *error = it->second.user ? "已有同名音色：" + name : "和内置音色重名：" + name;
            return false;
        }
    }
    if (!initVoice(v)) {
        if (error)
            *error = "音色用不了（不是 CosyVoice3 的音色文件，或者坏了）：" + name;
        return false;
    }
    Voice old;
    {
        std::lock_guard<std::mutex> voicesLock(m_voicesMutex);
        const auto it = m_voices.find(name);
        if (it != m_voices.end())
            old = it->second;
        m_voices[name] = v;
    }
    freeVoice(old);   // 换掉的旧音色（文件已经被新的盖掉了，不删）
    VLOG_INFO("音色已注册：%s（%s）", name.c_str(), file.c_str());
    return true;
}

bool Synthesizer::removeVoice(const std::string& name, std::string* error)
{
    std::lock_guard<std::mutex> lock(m_mutex);   // 正在念就等它念完
    Voice v;
    {
        std::lock_guard<std::mutex> voicesLock(m_voicesMutex);
        const auto it = m_voices.find(name);
        if (it == m_voices.end()) {
            if (error)
                *error = "没有这个音色：" + name;
            return false;
        }
        if (!it->second.user) {
            if (error)
                *error = "内置音色删不掉：" + name;
            return false;
        }
        v = it->second;
        m_voices.erase(it);
        if (m_defaultVoice == name)
            m_defaultVoice = m_voices.count("default") ? "default" : (m_voices.empty() ? std::string() : m_voices.begin()->first);
    }
    freeVoice(v);
    std::error_code ec;
    if (!v.file.empty())
        fs::remove(fs::u8path(v.file), ec);   // gguf 文件一并删掉（下次启动不再加载）
    VLOG_INFO("音色已删：%s（%s）", name.c_str(), ec ? "文件没删掉" : "连文件一起删了");
    return true;
}

SpeakResult Synthesizer::speak(uint64_t owner, const std::string& text, const std::string& voice, float speed,
                               const std::string& instruction, const OnAudio& onAudio, const Ahead& ahead)
{
    SpeakResult result;
    if (!m_ctx) {
        result.error = "合成模型没加载";
        return result;
    }
    const auto queued = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(m_mutex);   // 拿到它之后音色不会被删（v0.5 增删音色也要它）
    result.queuedSeconds = since(queued);
    cosyvoice_tts_context* tts = nullptr;
    std::string voiceName;
    {
        std::lock_guard<std::mutex> voicesLock(m_voicesMutex);
        voiceName = voice.empty() ? m_defaultVoice : voice;
        const auto it = m_voices.find(voiceName);
        if (it != m_voices.end())
            tts = it->second.tts;
    }
    if (!tts) {
        result.error = "没有这个音色：" + voice;
        return result;
    }
    {
        std::lock_guard<std::mutex> ownerLock(m_ownerMutex);
        m_owner = owner;
    }

    const float rate = speed > 0.25f && speed < 4.0f ? speed : 1.0f;
    // 分块：cosyvoice_tts_stream 每块开头都重新读一次块大小，所以在回调里（块与块之间）改下一块的
    const bool adaptive = m_fixedChunk == 0;
    double& scale = m_costScale.try_emplace(voiceName, 1.0).first->second;
    const double secondsPerToken = kTokenSeconds / rate;
    const unsigned firstChunk = adaptive ? pickFirstChunk(scale, secondsPerToken, ahead ? ahead() : 0.0) : m_fixedChunk;
    cosyvoice_set_chunk_tokens(m_ctx, firstChunk);

    struct State {
        Synthesizer* self;
        const OnAudio* onAudio;
        const Ahead* ahead;
        double* scale;
        double secondsPerToken;
        bool adaptive;
        std::chrono::steady_clock::time_point started;
        std::chrono::steady_clock::time_point last;   // 上一块交出去（回调返回）的时候：到这一块来之间是合成这块用的
        std::vector<unsigned> chunks;                 // 每块要的 token 数（最后多一个没用上的，收尾时去掉）
        size_t samples = 0;
        int callbacks = 0;
        double firstAudio = 0;
        bool aborted = false;
    } state { this, &onAudio, &ahead, &scale, secondsPerToken, adaptive, std::chrono::steady_clock::now(), {}, { firstChunk } };
    state.last = state.started;
    const auto callback = [](const float* audio, uint32_t count, void* user) -> bool {
        auto* s = static_cast<State*>(user);
        const double spent = since(s->last);
        const bool first = s->samples == 0;
        if (first)
            s->firstAudio = since(s->started);
        s->samples += count;
        ++s->callbacks;
        if (!(*s->onAudio)(audio, count)) {
            s->aborted = true;
            return false;
        }
        if (s->adaptive) {
            // 学这台、这个音色的耗时系数：只看满的块（最后不满的一块的固定开销摊不开，会把系数拉高）
            const unsigned requested = s->chunks.back();
            const double tokens = count / (s->self->m_sampleRate * s->secondsPerToken);
            if (tokens >= requested * 0.7) {
                const double reference = first ? firstChunkCost(requested) : nextChunkCost(requested);
                const double ratio = std::clamp(spent / reference, 0.05, 20.0);
                const double weight = first ? 0.15 : 0.3;
                *s->scale = *s->scale * (1 - weight) + ratio * weight;
            }
            const unsigned next = s->self->pickNextChunk(*s->scale, s->secondsPerToken, *s->ahead ? (*s->ahead)() : 0.0);
            cosyvoice_set_chunk_tokens(s->self->m_ctx, next);
            s->chunks.push_back(next);
        }
        s->last = std::chrono::steady_clock::now();
        return true;
    };
    const bool ok = instruction.empty()
        ? cosyvoice_tts_zero_shot_stream(tts, text.c_str(), rate, callback, &state)
        : cosyvoice_tts_instruct_stream(tts, text.c_str(), instruction.c_str(), rate, callback, &state);
    {
        std::lock_guard<std::mutex> ownerLock(m_ownerMutex);
        m_owner = 0;
    }
    if (state.chunks.size() > size_t(std::max(1, state.callbacks)))
        state.chunks.resize(size_t(std::max(1, state.callbacks)));
    result.chunks = std::move(state.chunks);
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
