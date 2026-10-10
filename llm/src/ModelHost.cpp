// 模型的宿主（见 ModelHost.h）
#include "ModelHost.h"

#include "RuntimeLog.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <regex>

namespace llm {
namespace fs = std::filesystem;

namespace {

std::string lower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return text;
}

std::string utf8(const fs::path& path)
{
    const auto text = path.u8string();
    return std::string(text.begin(), text.end());
}

fs::path pathOf(const std::string& text)
{
    return fs::u8path(text);
}

bool hasSuffix(const std::string& text, const std::string& suffix)
{
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// 文件名里的量化档（Q4_K_M、IQ3_XXS、BF16……）；没有为空
std::string quantOf(const std::string& fileName)
{
    static const std::regex pattern(R"((?:^|[-_.])((?:I?Q\d(?:_[0-9A-Z]+)*)|BF16|F16|F32|MXFP4)(?=[-_.]|$))", std::regex::icase);
    std::smatch match;
    std::string stem = fileName;
    if (hasSuffix(lower(stem), ".gguf"))
        stem = stem.substr(0, stem.size() - 5);
    std::string found;
    for (auto it = std::sregex_iterator(stem.begin(), stem.end(), pattern); it != std::sregex_iterator(); ++it)
        found = (*it)[1].str();
    std::transform(found.begin(), found.end(), found.begin(), [](unsigned char c) { return char(std::toupper(c)); });
    return found;
}

/// 一个目录（递归不深）里的主模型文件
std::vector<fs::path> mainModels(const fs::path& directory, int depth = 3)
{
    std::vector<fs::path> out;
    std::error_code error;
    if (!fs::is_directory(directory, error))
        return out;
    fs::recursive_directory_iterator it(directory, fs::directory_options::skip_permission_denied, error);
    for (; !error && it != fs::recursive_directory_iterator(); it.increment(error)) {
        if (it.depth() >= depth)
            it.disable_recursion_pending();
        const fs::path& path = it->path();
        if (!it->is_regular_file(error))
            continue;
        const std::string name = utf8(path.filename());
        if (!hasSuffix(lower(name), ".gguf") || isProjector(name) || isLaterShard(name))
            continue;
        // 还没下完的（Friday 的下载器先写 .part 再改名；分片要齐）
        out.push_back(path);
    }
    std::sort(out.begin(), out.end());
    return out;
}

/// 模型 id「组织/仓库」在各个模型目录里对应的目录（魔搭布局、HF 布局）
std::vector<fs::path> repoDirectories(const std::vector<std::string>& roots, const std::string& repo)
{
    std::vector<fs::path> out;
    const auto slash = repo.find('/');
    if (slash == std::string::npos)
        return out;
    const std::string org = repo.substr(0, slash);
    const std::string name = repo.substr(slash + 1);
    std::error_code error;
    for (const std::string& root : roots) {
        const fs::path base = pathOf(root);
        const fs::path modelScope = base / pathOf(org) / pathOf(name);
        if (fs::is_directory(modelScope, error))
            out.push_back(modelScope);
        const fs::path hub = base / pathOf("models--" + org + "--" + name) / "snapshots";
        if (fs::is_directory(hub, error)) {
            for (const auto& entry : fs::directory_iterator(hub, error)) {
                if (entry.is_directory(error))
                    out.push_back(entry.path());
            }
        }
    }
    return out;
}

/// 几个文件里挑默认的量化档：常用的几档优先，否则排第一个的
fs::path pickDefault(const std::vector<fs::path>& files)
{
    static const char* preferred[] = { "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0", "Q6_K", "Q8_0", "IQ4_XS", "MXFP4", "BF16", "F16" };
    for (const char* quant : preferred) {
        for (const fs::path& file : files) {
            if (quantOf(utf8(file.filename())) == quant)
                return file;
        }
    }
    return files.empty() ? fs::path() : files.front();
}

/// 模型旁边的视觉投影（同一目录及子目录；f16 / bf16 的优先）
std::string projectorNear(const fs::path& model)
{
    std::error_code error;
    std::vector<fs::path> found;
    const fs::path directory = model.parent_path();
    fs::recursive_directory_iterator it(directory, fs::directory_options::skip_permission_denied, error);
    for (; !error && it != fs::recursive_directory_iterator(); it.increment(error)) {
        if (it.depth() >= 2)
            it.disable_recursion_pending();
        const std::string name = utf8(it->path().filename());
        if (it->is_regular_file(error) && hasSuffix(lower(name), ".gguf") && isProjector(name))
            found.push_back(it->path());
    }
    if (found.empty())
        return {};
    std::sort(found.begin(), found.end());
    for (const fs::path& path : found) {
        const std::string name = lower(utf8(path.filename()));
        if (name.find("f16") != std::string::npos)
            return utf8(path);
    }
    return utf8(found.front());
}

/// 最小的那张显卡的显存（MiB；没有显卡为 0）
int smallestGpuMiB()
{
    size_t smallest = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU)
            continue;
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        if (total > 0 && (smallest == 0 || total < smallest))
            smallest = total;
    }
    return int(smallest / 1048576);
}

/// llama.cpp 的 fit 缺省就给每张卡留 1 GiB；预留比这个小时不用另给
constexpr int kDefaultFitTargetMiB = 1024;
/// 预留到期 / 调小多少以上才值得重新加载
constexpr int kRegrowMiB = 512;

} // namespace

bool isProjector(const std::string& fileName)
{
    return lower(fileName).find("mmproj") != std::string::npos;
}

bool isLaterShard(const std::string& fileName)
{
    static const std::regex shard(R"(-(\d{5})-of-(\d{5})\.gguf$)", std::regex::icase);
    std::smatch match;
    return std::regex_search(fileName, match, shard) && match[1].str() != "00001";
}

ModelHost::ModelHost(HostOptions options)
    : m_options(std::move(options))
{
}

ModelHost::~ModelHost()
{
    std::shared_ptr<flr::Runtime> current;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        current = std::move(m_current);
    }
    if (current)
        current->shutdown();
}

bool ModelHost::resolve(const json& body, ModelChoice& out, std::string* error) const
{
    const auto fail = [error](const std::string& message) {
        if (error)
            *error = message;
        return false;
    };
    json extra = json::object();
    if (body.contains("x_friday") && body["x_friday"].is_object())
        extra = body["x_friday"];
    const std::string model = body.contains("model") && body["model"].is_string() ? body["model"].get<std::string>() : std::string();
    std::string path = extra.contains("model_path") && extra["model_path"].is_string() ? extra["model_path"].get<std::string>() : std::string();
    std::error_code fsError;

    if (path.empty() && !model.empty() && fs::is_regular_file(pathOf(model), fsError))
        path = model;
    if (path.empty() && !model.empty()) {
        // 「组织/仓库:量化档」或「组织/仓库」
        std::string repo = model;
        std::string quant;
        const auto colon = model.rfind(':');
        if (colon != std::string::npos && colon > 0 && model.find('/') != std::string::npos && colon > model.find('/')) {
            repo = model.substr(0, colon);
            quant = model.substr(colon + 1);
        }
        std::vector<fs::path> files;
        for (const fs::path& directory : repoDirectories(m_options.modelRoots, repo)) {
            for (const fs::path& file : mainModels(directory))
                files.push_back(file);
        }
        if (!quant.empty()) {
            const std::string wanted = lower(quant);
            std::vector<fs::path> matching;
            for (const fs::path& file : files) {
                if (lower(quantOf(utf8(file.filename()))) == wanted || lower(utf8(file.filename())).find(wanted) != std::string::npos)
                    matching.push_back(file);
            }
            files.swap(matching);
        }
        const fs::path chosen = pickDefault(files);
        if (!chosen.empty())
            path = utf8(chosen);
    }
    if (path.empty())
        return fail(model.empty() ? std::string("请求里没有 model") : "本机没有这个模型：" + model + "（先在 Friday 的模型库里下载）");
    if (!fs::is_regular_file(pathOf(path), fsError))
        return fail("找不到模型文件：" + path);

    out.path = path;
    out.id = model.empty() ? utf8(pathOf(path).filename()) : model;
    if (extra.contains("mmproj_path") && extra["mmproj_path"].is_string()) {
        // 显式给了（空 = 不要看图）
        out.mmproj = extra["mmproj_path"].get<std::string>();
        if (!out.mmproj.empty() && !fs::is_regular_file(pathOf(out.mmproj), fsError))
            return fail("找不到视觉投影文件：" + out.mmproj);
    } else {
        out.mmproj = projectorNear(pathOf(path));
    }
    return true;
}

bool ModelHost::same(const flr::Runtime& runtime, const ModelChoice& choice) const
{
    std::error_code error;
    const bool samePath = runtime.modelPath() == choice.path || fs::equivalent(pathOf(runtime.modelPath()), pathOf(choice.path), error);
    const bool sameProjector = runtime.mmprojPath() == choice.mmproj;
    return samePath && sameProjector;
}

int ModelHost::activeReserve() const
{
    if (m_reserveMiB <= 0 || std::chrono::steady_clock::now() >= m_reserveUntil)
        return 0;
    return m_reserveMiB;
}

std::shared_ptr<flr::Runtime> ModelHost::acquire(const ModelChoice& choice, int* code, std::string* error)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while (true) {
        // 加载时为别的程序留了显存、现在预留到期或调小了：没有别的请求在算时重新加载，把显存要回来
        const int reserve = activeReserve();
        const bool regrow = m_current && m_loadedReserve > std::max(reserve, kDefaultFitTargetMiB) + kRegrowMiB && !m_current->busy();
        if (m_current && !m_loading && same(*m_current, choice) && !regrow)
            return m_current;
        if (m_loading) {
            m_changed.wait(lock);
            continue;
        }
        // 换模型：等加载着的那个的请求都结束（别的请求同时在用它）
        if (m_current && m_current->busy()) {
            m_changed.wait_for(lock, std::chrono::milliseconds(100));
            continue;
        }
        m_loading = true;
        m_loadingId = choice.id;
        const int previousReserve = m_loadedReserve;
        std::shared_ptr<flr::Runtime> old = std::move(m_current);
        lock.unlock();
        if (old) {
            if (regrow && same(*old, choice))
                flr::logf(FLR_LOG_NOTICE, "显存预留已解除（%d → %d MiB）：重新加载 %s", previousReserve, reserve, choice.id.c_str());
            else
                flr::logf(FLR_LOG_NOTICE, "换模型：%s → %s", old->modelId().c_str(), choice.id.c_str());
            old->shutdown();
            old.reset();
        }
        flr::LoadOptions options;
        options.modelPath = choice.path;
        options.mmprojPath = choice.mmproj;
        options.modelId = choice.id;
        options.parallel = m_options.parallel;
        options.sequences = m_options.sequences;
        options.contextTokens = m_options.contextTokens;
        options.perSequence = m_options.perSequence;
        options.extraArgs = m_options.extraArgs;
        // 显存预留：放在追加参数后面，覆盖 FRIDAY_LLAMA_ARGS 里的 --fit-target
        if (reserve > kDefaultFitTargetMiB) {
            options.extraArgs.push_back("--fit-target");
            options.extraArgs.push_back(std::to_string(reserve));
            flr::logf(FLR_LOG_NOTICE, "给别的程序留 %d MiB 显存（生图 / 生视频）", reserve);
        }
        flr::LoadError loadError;
        std::unique_ptr<flr::Runtime> loaded = flr::Runtime::load(options, nullptr, nullptr, loadError);
        lock.lock();
        m_loading = false;
        m_loadingId.clear();
        m_loadedReserve = loaded ? reserve : 0;
        m_current = std::shared_ptr<flr::Runtime>(std::move(loaded));
        m_lastError = m_current ? std::string() : loadError.message;
        m_changed.notify_all();
        if (!m_current) {
            if (code)
                *code = loadError.code;
            if (error)
                *error = loadError.message;
            return nullptr;
        }
    }
}

std::string ModelHost::unload()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_changed.wait(lock, [this] { return !m_loading; });
    while (m_current && m_current->busy())
        m_changed.wait_for(lock, std::chrono::milliseconds(100));
    std::shared_ptr<flr::Runtime> old = std::move(m_current);
    lock.unlock();
    if (!old)
        return {};
    const std::string id = old->modelId();
    old->shutdown();
    old.reset();
    flr::logf(FLR_LOG_NOTICE, "已卸载模型 %s", id.c_str());
    return id;
}

json ModelHost::reserve(int mib, int seconds)
{
    mib = std::max(0, mib);
    seconds = std::clamp(seconds, 1, 24 * 3600);
    // 至少给大模型自己留一成：留得比整张卡还多时 fit 会把整个模型挪到内存里
    if (const int total = smallestGpuMiB(); total > 0)
        mib = std::min(mib, total * 9 / 10);
    bool unloadNow = false;
    int before = 0;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        before = activeReserve();
        m_reserveMiB = mib;
        m_reserveUntil = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
        // 现在的模型留得不够：卸掉，下个请求按新的预留加载
        unloadNow = m_current && mib > std::max(m_loadedReserve, kDefaultFitTargetMiB) + kRegrowMiB / 2;
    }
    // 生视频时宿主每次轮询都续一下：只在变了时记
    if (mib != before)
        flr::logf(FLR_LOG_NOTICE, mib > 0 ? "显存预留 %d MiB，%d 秒" : "显存预留取消", mib, seconds);
    std::string unloaded;
    if (unloadNow)
        unloaded = unload();
    return json {
        { "reserved_mib", mib },
        { "seconds", mib > 0 ? seconds : 0 },
        { "unloaded", unloaded.empty() ? json(nullptr) : json(unloaded) },
    };
}

json ModelHost::info() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    json result {
        { "loaded", m_current ? m_current->info() : json(nullptr) },
        { "loading", m_loading ? json(m_loadingId) : json(nullptr) },
        { "reserved_mib", activeReserve() },
        { "loaded_reserve_mib", m_current ? m_loadedReserve : 0 },
    };
    if (!m_lastError.empty())
        result["last_error"] = m_lastError;
    return result;
}

bool ModelHost::loading() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_loading;
}

json ModelHost::listModels() const
{
    json data = json::array();
    std::string loaded;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_current)
            loaded = m_current->modelPath();
    }
    std::error_code error;
    for (const std::string& root : m_options.modelRoots) {
        const fs::path base = pathOf(root);
        for (const fs::path& file : mainModels(base, 6)) {
            const fs::path relative = file.lexically_relative(base);
            std::vector<std::string> parts;
            for (const fs::path& part : relative)
                parts.push_back(utf8(part));
            std::string repo;
            if (parts.size() >= 2 && parts[0].rfind("models--", 0) == 0) {
                // HF：models--组织--仓库/snapshots/<提交>/文件
                std::string name = parts[0].substr(8);
                const auto dash = name.find("--");
                repo = dash == std::string::npos ? name : name.substr(0, dash) + "/" + name.substr(dash + 2);
            } else if (parts.size() >= 3) {
                repo = parts[0] + "/" + parts[1];
            }
            const std::string fileName = utf8(file.filename());
            const std::string quant = quantOf(fileName);
            const std::string id = repo.empty() ? fileName : repo + ":" + (quant.empty() ? fileName : quant);
            data.push_back(json {
                { "id", id },
                { "object", "model" },
                { "owned_by", "local" },
                { "path", utf8(file) },
                { "size", uint64_t(fs::file_size(file, error)) },
                { "vision", !projectorNear(file).empty() },
                { "loaded", !loaded.empty() && utf8(file) == loaded },
            });
        }
    }
    return json { { "object", "list" }, { "data", data } };
}

} // namespace llm
