// 模型的宿主：按请求加载、换模型、卸载（docs/llm-protocol.md §2）。
// 同一时间只加载一个大模型（显存只够一个）。请求要的模型和加载着的不一样时，等加载着的那个的请求都结束再换；
// 换的时候来的请求等换完。模型文件由 Friday 的模型库下载：请求里 x_friday.model_path 直接给路径，或者 model 写模型 id
// （「组织/仓库:量化档」），在 --models 给的目录里找（魔搭的缓存布局 <目录>/<组织>/<仓库>/*.gguf，HF 的 models--组织--仓库/snapshots/*/）。
#pragma once

#include "Runtime.h"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace llm {

using json = flr::json;

struct HostOptions {
    /// 模型库的目录（可以有几个）
    std::vector<std::string> modelRoots;
    /// 同时在算的请求最多几个
    int parallel = 4;
    /// 缓存的序列数（0 = 按模型自动）
    int sequences = 0;
    /// KV 池（0 = 按显存自动）
    int contextTokens = 0;
    /// 单序列上限（0 = 不另限）
    int perSequence = 0;
    /// 追加给 llama 的参数（llama-server 的写法）
    std::vector<std::string> extraArgs;
};

/// 一个请求要的模型
struct ModelChoice {
    /// chunk 里的 model 字段（Friday 的模型 id；没有就用文件名）
    std::string id;
    std::string path;
    /// 视觉投影；空 = 纯文本（auto 时在模型旁边找）
    std::string mmproj;
};

class ModelHost {
public:
    explicit ModelHost(HostOptions options);
    ~ModelHost();

    /// 请求体 → 模型文件。找不到时 false + error（给人看的）
    bool resolve(const json& body, ModelChoice& out, std::string* error) const;
    /// 拿到这个模型的运行时（没加载就加载，加载着别的就等它空闲后换掉）。失败返回空，code 是 FLR_*
    std::shared_ptr<flr::Runtime> acquire(const ModelChoice& choice, int* code, std::string* error);
    /// 卸载（腾显存）：等在途请求结束。返回卸载前加载的模型 id（没有为空）
    std::string unload();
    /// 给别的程序（friday-diffusion 生图 / 生视频）留显存：之后加载模型时每张卡至少空出 mib MiB（llama.cpp 的 --fit-target，
    /// 放不下就少放几层到显卡、上下文小一些），seconds 秒后失效（宿主不续就自动恢复）。现在加载着的模型留得不够就卸掉（等在途请求结束），
    /// 下个请求按新的预留加载；预留到期或调小后，下个请求来时如果没有别的请求在算，按新的预留重新加载、把显存要回来。
    /// mib = 0 取消。返回 {reserved_mib, seconds, unloaded}
    json reserve(int mib, int seconds);

    /// /v1/info 的 model 部分（没加载为 null）
    json info() const;
    /// /v1/models：模型库目录里的 GGUF（主模型文件，分片只列第一片）
    json listModels() const;
    bool loading() const;

private:
    bool same(const flr::Runtime& runtime, const ModelChoice& choice) const;
    /// 现在有效的预留（MiB，到期为 0）。持锁调
    int activeReserve() const;

    HostOptions m_options;
    mutable std::mutex m_mutex;
    std::condition_variable m_changed;
    std::shared_ptr<flr::Runtime> m_current;
    bool m_loading = false;
    std::string m_loadingId;
    std::string m_lastError;
    /// 显存预留（reserve）与它的到期时间；m_loadedReserve 是现在这个模型加载时按的预留
    int m_reserveMiB = 0;
    std::chrono::steady_clock::time_point m_reserveUntil;
    int m_loadedReserve = 0;
};

/// 一个 GGUF 文件名是不是视觉投影（mmproj）
bool isProjector(const std::string& fileName);
/// 分片模型的第一片之外的那些（-00002-of-00003.gguf …）
bool isLaterShard(const std::string& fileName);

} // namespace llm
