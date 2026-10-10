// 生图 / 生视频的任务队列（见 Generator.h）
#include "Generator.h"

#include "Log.h"

#include "media_io.h"
#include "stable-diffusion.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <random>

namespace diffusion {
namespace {

int64_t nowMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string randomId()
{
    static const char hex[] = "0123456789abcdef";
    thread_local std::mt19937_64 generator(std::random_device {}());
    std::string id = "job-";
    for (int i = 0; i < 16; ++i)
        id += hex[generator() % 16];
    return id;
}

/// 「data:image/png;base64,…」或纯 base64 → 字节
bool decodeBase64(const std::string& input, std::vector<uint8_t>& out)
{
    std::string text = input;
    const auto comma = text.find(',');
    if (text.rfind("data:", 0) == 0 && comma != std::string::npos)
        text = text.substr(comma + 1);
    static int table[256];
    static bool ready = false;
    if (!ready) {
        std::fill(std::begin(table), std::end(table), -1);
        const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i)
            table[(unsigned char) alphabet[i]] = i;
        table[(unsigned char) '-'] = 62;
        table[(unsigned char) '_'] = 63;
        ready = true;
    }
    out.clear();
    out.reserve(text.size() * 3 / 4);
    int value = 0;
    int bits = -8;
    for (unsigned char c : text) {
        if (c == '=' || c == '\r' || c == '\n' || c == ' ')
            continue;
        if (table[c] < 0)
            return false;
        value = (value << 6) + table[c];
        bits += 6;
        if (bits >= 0) {
            out.push_back(uint8_t((value >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return !out.empty();
}

/// 一张输入图（RGB）：解码失败为空
struct InputImage {
    uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    ~InputImage() { std::free(data); }
    sd_image_t image() const { return sd_image_t { uint32_t(width), uint32_t(height), 3, data }; }
};

bool loadImage(const json& value, InputImage& out, int width = 0, int height = 0)
{
    if (!value.is_string())
        return false;
    std::vector<uint8_t> bytes;
    if (!decodeBase64(value.get<std::string>(), bytes))
        return false;
    int channels = 0;
    out.data = load_image_from_memory(reinterpret_cast<const char*>(bytes.data()), int(bytes.size()), out.width, out.height, channels, width,
                                      height, 3);
    return out.data != nullptr;
}

template <class T>
T param(const json& params, const char* key, T fallback)
{
    const auto it = params.find(key);
    if (it == params.end() || it->is_null())
        return fallback;
    try {
        return it->template get<T>();
    } catch (const std::exception&) {
        return fallback;
    }
}

/// "1024x768" → 宽、高
bool parseSize(const std::string& size, int& width, int& height)
{
    const auto x = size.find_first_of("xX*");
    if (x == std::string::npos)
        return false;
    width = std::atoi(size.substr(0, x).c_str());
    height = std::atoi(size.substr(x + 1).c_str());
    return width > 0 && height > 0;
}

void fillSampleParams(sd_sample_params_t& sample, const json& params, sd_ctx_t* ctx)
{
    sd_sample_params_init(&sample);
    sample.sample_steps = param<int>(params, "steps", sample.sample_steps);
    sample.guidance.txt_cfg = param<float>(params, "cfg_scale", sample.guidance.txt_cfg);
    sample.guidance.distilled_guidance = param<float>(params, "guidance", sample.guidance.distilled_guidance);
    sample.flow_shift = param<float>(params, "flow_shift", sample.flow_shift);
    const std::string sampler = param<std::string>(params, "sampler", std::string());
    if (!sampler.empty()) {
        const sample_method_t method = str_to_sample_method(sampler.c_str());
        if (method != SAMPLE_METHOD_COUNT)
            sample.sample_method = method;
    }
    if (sample.sample_method == SAMPLE_METHOD_COUNT && ctx)
        sample.sample_method = sd_get_default_sample_method(ctx);
    const std::string scheduler = param<std::string>(params, "scheduler", std::string());
    if (!scheduler.empty()) {
        const scheduler_t value = str_to_scheduler(scheduler.c_str());
        if (value != SCHEDULER_COUNT)
            sample.scheduler = value;
    }
    if (sample.scheduler == SCHEDULER_COUNT && ctx)
        sample.scheduler = sd_get_default_scheduler(ctx, sample.sample_method);
}

void progressCallback(int step, int steps, float, void* data)
{
    auto* job = static_cast<Job*>(data);
    if (!job)
        return;
    job->step = step;
    job->steps = steps;
    job->progress = steps > 0 ? double(step) / double(steps) : 0.0;
}

void sdLog(enum sd_log_level_t level, const char* text, void*)
{
    if (!text)
        return;
    std::string line = text;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
        line.pop_back();
    if (line.empty())
        return;
    switch (level) {
    case SD_LOG_ERROR: DLOG_ERROR("sd: %s", line.c_str()); break;
    case SD_LOG_WARN: DLOG_WARN("sd: %s", line.c_str()); break;
    // stable-diffusion.cpp 的 info 很碎（每一步加载、建图都打一行，宿主会把它们都记进日志）：--verbose 才打；
    // 加载用时、任务完成这些我们自己记
    default: DLOG_DEBUG("sd: %s", line.c_str()); break;
    }
}

const char* nullable(const std::string& text)
{
    return text.empty() ? nullptr : text.c_str();
}

} // namespace

// MARK: - ModelFiles

std::string ModelFiles::key() const
{
    return model + "|" + diffusionModel + "|" + highNoiseDiffusionModel + "|" + vae + "|" + llm + "|" + llmVision + "|" + t5xxl + "|" + clipL
        + "|" + clipG + "|" + clipVision + "|" + embeddingsConnectors + "|" + audioVae;
}

ModelFiles ModelFiles::fromJson(const json& value)
{
    ModelFiles files;
    if (!value.is_object())
        return files;
    const auto get = [&value](const char* key) { return value.contains(key) && value[key].is_string() ? value[key].get<std::string>() : std::string(); };
    files.model = get("model");
    files.diffusionModel = get("diffusion_model");
    files.highNoiseDiffusionModel = get("high_noise_diffusion_model");
    files.vae = get("vae");
    files.llm = get("llm");
    files.llmVision = get("llm_vision");
    files.t5xxl = get("t5xxl");
    files.clipL = get("clip_l");
    files.clipG = get("clip_g");
    files.clipVision = get("clip_vision");
    files.embeddingsConnectors = get("embeddings_connectors");
    files.audioVae = get("audio_vae");
    return files;
}

json ModelFiles::toJson() const
{
    json value = json::object();
    const auto put = [&value](const char* key, const std::string& path) {
        if (!path.empty())
            value[key] = path;
    };
    put("model", model);
    put("diffusion_model", diffusionModel);
    put("high_noise_diffusion_model", highNoiseDiffusionModel);
    put("vae", vae);
    put("llm", llm);
    put("llm_vision", llmVision);
    put("t5xxl", t5xxl);
    put("clip_l", clipL);
    put("clip_g", clipG);
    put("clip_vision", clipVision);
    put("embeddings_connectors", embeddingsConnectors);
    put("audio_vae", audioVae);
    return value;
}

// MARK: - Generator

Generator::Generator(json families, int idleSeconds)
    : m_families(std::move(families))
    , m_idleSeconds(idleSeconds)
{
    sd_set_log_callback(&sdLog, nullptr);
    m_thread = std::thread([this] { run(); });
}

Generator::~Generator()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
        for (const auto& job : m_queue)
            job->cancelled = true;
        if (m_current)
            m_current->cancelled = true;
    }
    m_wake.notify_all();
    if (m_thread.joinable())
        m_thread.join();
}

json Generator::defaultsFor(const std::string& family) const
{
    if (!m_families.is_array())
        return json::object();
    for (const json& entry : m_families) {
        if (entry.value("id", std::string()) == family && entry.contains("defaults") && entry["defaults"].is_object())
            return entry["defaults"];
    }
    return json::object();
}

std::shared_ptr<Job> Generator::submit(const json& body, std::string* error)
{
    const auto fail = [error](const std::string& message) -> std::shared_ptr<Job> {
        if (error)
            *error = message;
        return nullptr;
    };
    if (!body.is_object())
        return fail("请求体不是 JSON 对象");
    auto job = std::make_shared<Job>();
    job->id = randomId();
    job->kind = body.value("kind", std::string("image"));
    if (job->kind != "image" && job->kind != "video")
        return fail("kind 只能是 image 或 video");
    const json extra = body.contains("x_friday") && body["x_friday"].is_object() ? body["x_friday"] : json::object();
    job->family = extra.value("family", std::string());
    job->files = ModelFiles::fromJson(extra.contains("files") ? extra["files"] : json());
    if (job->files.empty())
        return fail("没有给模型文件（x_friday.files.diffusion_model 或 model）");
    if (body.value("prompt", std::string()).empty())
        return fail("没有 prompt");

    // 参数：家族缺省 → 请求
    json params = json::object();
    const json defaults = defaultsFor(job->family);
    if (defaults.contains(job->kind) && defaults[job->kind].is_object())
        params = defaults[job->kind];
    for (const auto& item : body.items()) {
        if (item.key() != "x_friday" && item.key() != "kind")
            params[item.key()] = item.value();
    }
    if (params.contains("size") && params["size"].is_string()) {
        int width = 0;
        int height = 0;
        if (parseSize(params["size"].get<std::string>(), width, height)) {
            params["width"] = width;
            params["height"] = height;
        }
    }
    job->params = std::move(params);
    job->created = nowMs();
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_stop)
            return fail("正在退出");
        m_queue.push_back(job);
        m_history.push_back(job);
        while (m_history.size() > 32)
            m_history.pop_front();
    }
    m_wake.notify_all();
    DLOG_INFO("任务 %s：%s，%s", job->id.c_str(), job->kind.c_str(), job->family.empty() ? "-" : job->family.c_str());
    return job;
}

std::shared_ptr<Job> Generator::find(const std::string& id) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& job : m_history) {
        if (job->id == id)
            return job;
    }
    return nullptr;
}

bool Generator::cancel(const std::string& id)
{
    std::shared_ptr<Job> job = find(id);
    if (!job)
        return false;
    job->cancelled = true;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_current == job && m_ctx)
            sd_cancel_generation(m_ctx, SD_CANCEL_ALL);
        const auto it = std::find(m_queue.begin(), m_queue.end(), job);
        if (it != m_queue.end()) {
            m_queue.erase(it);
            job->status = "cancelled";
            job->finished = nowMs();
        }
    }
    m_done.notify_all();
    return true;
}

bool Generator::wait(const std::shared_ptr<Job>& job, int timeoutMs)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    const auto finished = [&job] { return job->status == "done" || job->status == "failed" || job->status == "cancelled"; };
    if (timeoutMs < 0) {
        m_done.wait(lock, finished);
        return true;
    }
    return m_done.wait_for(lock, std::chrono::milliseconds(timeoutMs), finished);
}

bool Generator::release()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_releaseRequested = true;
    }
    m_wake.notify_all();
    // 等工作线程放掉（正在跑的任务做完才放）
    std::unique_lock<std::mutex> lock(m_mutex);
    const bool wasLoaded = m_loaded.load();
    m_done.wait_for(lock, std::chrono::minutes(30), [this] { return !m_releaseRequested; });
    return wasLoaded;
}

json Generator::describe(const Job& job) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    json outputs = json::array();
    for (size_t i = 0; i < job.outputs.size(); ++i) {
        outputs.push_back(json {
            { "index", i },
            { "mime", job.outputs[i].mime },
            { "bytes", job.outputs[i].data.size() },
            { "width", job.outputs[i].width },
            { "height", job.outputs[i].height },
            { "url", "/v1/jobs/" + job.id + "/outputs/" + std::to_string(i) },
        });
    }
    json value {
        { "id", job.id },
        { "object", "job" },
        { "kind", job.kind },
        { "family", job.family },
        { "status", job.status },
        { "progress", job.progress.load() },
        { "step", job.step.load() },
        { "steps", job.steps.load() },
        { "created", job.created / 1000 },
        { "outputs", outputs },
    };
    if (!job.error.empty())
        value["error"] = json { { "message", job.error } };
    if (job.started > 0)
        value["elapsed_ms"] = (job.finished > 0 ? job.finished : nowMs()) - job.started;
    return value;
}

bool Generator::output(const Job& job, size_t index, Output& out) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (job.status != "done" || index >= job.outputs.size())
        return false;
    out = job.outputs[index];
    return true;
}

json Generator::info() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return json {
        { "loaded", m_loaded.load() },
        { "family", m_ctxFamily },
        { "queued", m_queue.size() },
        { "running", m_current ? json(m_current->id) : json(nullptr) },
        { "idle_unload_seconds", m_idleSeconds },
    };
}

void Generator::run()
{
    while (true) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_wake.wait_for(lock, std::chrono::seconds(5), [this] { return m_stop || !m_queue.empty() || m_releaseRequested; });
            if (m_stop)
                break;
            if (m_releaseRequested && m_queue.empty()) {
                lock.unlock();
                freeContext();
                lock.lock();
                m_releaseRequested = false;
                m_done.notify_all();
                continue;
            }
            if (m_queue.empty()) {
                // 空闲久了放掉模型，把显存让给大模型
                if (m_ctx && m_idleSeconds > 0 && nowMs() - m_lastUsed > int64_t(m_idleSeconds) * 1000) {
                    lock.unlock();
                    DLOG_INFO("空闲 %d 秒，释放模型", m_idleSeconds);
                    freeContext();
                }
                continue;
            }
            job = m_queue.front();
            m_queue.pop_front();
            m_current = job;
        }
        if (!job->cancelled)
            execute(*job);
        m_lastUsed = nowMs();
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (job->cancelled && job->status != "done" && job->status != "failed")
                job->status = "cancelled";
            if (job->finished == 0)
                job->finished = nowMs();
            m_current.reset();
        }
        m_done.notify_all();
    }
    freeContext();
}

void Generator::freeContext()
{
    if (m_ctx) {
        free_sd_ctx(m_ctx);
        m_ctx = nullptr;
        DLOG_INFO("模型已释放");
    }
    m_ctxKey.clear();
    m_ctxFamily.clear();
    m_loaded = false;
}

bool Generator::ensureContext(Job& job)
{
    const std::string key = job.files.key();
    if (m_ctx && key == m_ctxKey)
        return true;
    freeContext();
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        job.status = "loading";
    }
    const auto started = std::chrono::steady_clock::now();
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = nullable(job.files.model);
    params.diffusion_model_path = nullable(job.files.diffusionModel);
    params.high_noise_diffusion_model_path = nullable(job.files.highNoiseDiffusionModel);
    params.vae_path = nullable(job.files.vae);
    params.llm_path = nullable(job.files.llm);
    params.llm_vision_path = nullable(job.files.llmVision);
    params.t5xxl_path = nullable(job.files.t5xxl);
    params.clip_l_path = nullable(job.files.clipL);
    params.clip_g_path = nullable(job.files.clipG);
    params.clip_vision_path = nullable(job.files.clipVision);
    params.embeddings_connectors_path = nullable(job.files.embeddingsConnectors);
    params.audio_vae_path = nullable(job.files.audioVae);
    params.diffusion_flash_attn = true;
    // 显存不够时 sd.cpp 自己把权重分到内存里（auto_fit，缺省开）
    params.auto_fit = true;
    m_ctx = new_sd_ctx(&params);
    if (!m_ctx) {
        job.error = "模型加载失败（文件不对、显存 / 内存不够，或者这个模型 stable-diffusion.cpp 还不支持；详见日志）";
        return false;
    }
    m_ctxKey = key;
    m_ctxFamily = job.family;
    m_loaded = true;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    DLOG_INFO("模型已加载（%s，%.1f 秒）：%s", sd_get_model_version_name(m_ctx), seconds,
              job.files.diffusionModel.empty() ? job.files.model.c_str() : job.files.diffusionModel.c_str());
    return true;
}

void Generator::execute(Job& job)
{
    job.started = nowMs();
    if (!ensureContext(job)) {
        std::lock_guard<std::mutex> lock(m_mutex);
        job.status = "failed";
        return;
    }
    const bool video = job.kind == "video";
    if (video ? !sd_ctx_supports_video_generation(m_ctx) : !sd_ctx_supports_image_generation(m_ctx)) {
        std::lock_guard<std::mutex> lock(m_mutex);
        job.status = "failed";
        job.error = video ? "这个模型不能生视频" : "这个模型不能生图（视频模型请用 kind = video）";
        return;
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        job.status = "running";
    }
    sd_cancel_generation(m_ctx, SD_CANCEL_RESET);
    sd_set_progress_callback(&progressCallback, &job);
    const json& p = job.params;
    const std::string prompt = param<std::string>(p, "prompt", std::string());
    const std::string negative = param<std::string>(p, "negative_prompt", std::string());
    const int width = param<int>(p, "width", 1024);
    const int height = param<int>(p, "height", 1024);
    const int64_t seed = param<int64_t>(p, "seed", -1);
    const auto started = std::chrono::steady_clock::now();

    InputImage init;
    InputImage end;
    std::vector<std::unique_ptr<InputImage>> references;
    std::vector<sd_image_t> referenceImages;
    if (p.contains("image") && !loadImage(p["image"], init, width, height))
        DLOG_WARN("任务 %s：起始图解不开，忽略", job.id.c_str());
    if (p.contains("end_image") && !loadImage(p["end_image"], end, width, height))
        DLOG_WARN("任务 %s：结束帧解不开，忽略", job.id.c_str());
    if (p.contains("reference_images") && p["reference_images"].is_array()) {
        for (const json& value : p["reference_images"]) {
            auto image = std::make_unique<InputImage>();
            if (loadImage(value, *image)) {
                referenceImages.push_back(image->image());
                references.push_back(std::move(image));
            }
        }
    }

    std::string failure;
    std::vector<Output> outputs;
    if (!video) {
        sd_img_gen_params_t params;
        sd_img_gen_params_init(&params);
        params.prompt = prompt.c_str();
        params.negative_prompt = negative.c_str();
        params.width = width;
        params.height = height;
        params.seed = seed;
        params.batch_count = std::max(1, std::min(8, param<int>(p, "n", 1)));
        fillSampleParams(params.sample_params, p, m_ctx);
        if (init.data) {
            params.init_image = init.image();
            params.strength = param<float>(p, "strength", 0.75f);
        }
        if (!referenceImages.empty()) {
            params.ref_images = referenceImages.data();
            params.ref_images_count = int(referenceImages.size());
        }
        sd_image_t* images = nullptr;
        int count = 0;
        const bool ok = generate_image(m_ctx, &params, &images, &count);
        if (ok && images) {
            std::lock_guard<std::mutex> lock(m_mutex);
            job.status = "encoding";
        }
        for (int i = 0; ok && images && i < count; ++i) {
            if (!images[i].data)
                continue;
            Output output;
            output.mime = "image/png";
            output.width = int(images[i].width);
            output.height = int(images[i].height);
            output.data = encode_image_to_vector(EncodedImageFormat::PNG, images[i].data, output.width, output.height, int(images[i].channel));
            outputs.push_back(std::move(output));
        }
        if (images) {
            for (int i = 0; i < count; ++i)
                std::free(images[i].data);
            std::free(images);
        }
        if (outputs.empty())
            failure = job.cancelled ? "已取消" : "生成失败（详见日志）";
    } else {
        sd_vid_gen_params_t params;
        sd_vid_gen_params_init(&params);
        params.prompt = prompt.c_str();
        params.negative_prompt = negative.c_str();
        params.width = width;
        params.height = height;
        params.seed = seed;
        params.video_frames = std::max(1, param<int>(p, "frames", 33));
        params.fps = std::max(1, param<int>(p, "fps", 16));
        fillSampleParams(params.sample_params, p, m_ctx);
        fillSampleParams(params.high_noise_sample_params, p.contains("high_noise") ? p["high_noise"] : p, m_ctx);
        if (init.data) {
            params.init_image = init.image();
            params.strength = param<float>(p, "strength", 1.0f);
        }
        if (end.data)
            params.end_image = end.image();
        sd_image_t* frames = nullptr;
        int count = 0;
        sd_audio_t* audio = nullptr;
        int fps = params.fps;
        const bool ok = generate_video(m_ctx, &params, &frames, &count, &audio, &fps);
        if (ok && frames && count > 0) {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                job.status = "encoding";
            }
            Output output;
            output.mime = "video/webm";
            output.width = int(frames[0].width);
            output.height = int(frames[0].height);
            output.data = create_video_from_sd_images_to_vector("webm", frames, count, fps > 0 ? fps : params.fps, 90, audio, std::string());
            if (!output.data.empty())
                outputs.push_back(std::move(output));
        }
        if (frames) {
            for (int i = 0; i < count; ++i)
                std::free(frames[i].data);
            std::free(frames);
        }
        if (audio)
            free_sd_audio(audio);
        if (outputs.empty())
            failure = job.cancelled ? "已取消" : "生成失败（详见日志）";
    }
    sd_set_progress_callback(nullptr, nullptr);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::lock_guard<std::mutex> lock(m_mutex);
    job.finished = nowMs();
    job.outputs = std::move(outputs);
    if (!failure.empty()) {
        job.status = job.cancelled ? "cancelled" : "failed";
        job.error = failure;
        DLOG_WARN("任务 %s：%s（%.1f 秒）", job.id.c_str(), failure.c_str(), seconds);
    } else {
        job.status = "done";
        job.progress = 1.0;
        DLOG_INFO("任务 %s 完成：%zu 个结果，%.1f 秒", job.id.c_str(), job.outputs.size(), seconds);
    }
}

} // namespace diffusion
