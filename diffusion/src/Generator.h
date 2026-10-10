// 生图 / 生视频的任务队列（docs/diffusion-protocol.md）：
//   - 一个工作线程，任务按先后一个个做（显卡一次只跑一个）；
//   - 模型上下文（sd_ctx）按「用到的那几个文件」缓存：同一组文件的任务接着用，换一组就释放旧的、加载新的；
//     空闲一段时间（缺省 10 分钟）自动释放，腾显存给大模型；/v1/memory/release 马上释放；
//   - 参数：请求里给的优先，其次模型家族的缺省（power.json 的 families），再其次 sd.cpp 的缺省；
//   - 结果在内存里留着（最近 32 个任务），图片 PNG，视频 WebM（VP8）。
#pragma once

#include "json.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct sd_ctx_t;

namespace diffusion {

using json = nlohmann::ordered_json;

/// 一个模型要的文件（都是本机路径；没有的为空）
struct ModelFiles {
    /// 整个模型一个文件（SD 1.x / SDXL 的 checkpoint）
    std::string model;
    /// 只有扩散部分的文件（Z-Image、FLUX、Wan 的 GGUF）
    std::string diffusionModel;
    /// Wan2.2 A14B 的高噪声那一半
    std::string highNoiseDiffusionModel;
    std::string vae;
    /// 大模型做文本编码器（Z-Image、FLUX.2、Qwen-Image）
    std::string llm;
    std::string llmVision;
    std::string t5xxl;
    std::string clipL;
    std::string clipG;
    std::string clipVision;
    std::string embeddingsConnectors;
    std::string audioVae;

    std::string key() const;
    bool empty() const { return model.empty() && diffusionModel.empty(); }
    static ModelFiles fromJson(const json& value);
    json toJson() const;
};

struct Output {
    std::string mime;
    std::vector<uint8_t> data;
    int width = 0;
    int height = 0;
};

/// 一个任务（状态由工作线程改，读的时候加锁）
struct Job {
    std::string id;
    /// image | video
    std::string kind;
    std::string family;
    ModelFiles files;
    json params;
    /// queued | loading | running | encoding | done | failed | cancelled
    std::string status = "queued";
    std::string error;
    /// 进度（采样回调里写，不加锁）
    std::atomic<int> step { 0 };
    std::atomic<int> steps { 0 };
    std::atomic<double> progress { 0 };
    int64_t created = 0;
    int64_t started = 0;
    int64_t finished = 0;
    std::vector<Output> outputs;
    std::atomic<bool> cancelled { false };
};

class Generator {
public:
    /// families：power.json 里的 families（id → defaults）
    Generator(json families, int idleSeconds);
    ~Generator();

    /// 提交一个任务。body：{kind, prompt, negative_prompt, size / width / height, n, steps, cfg_scale, seed, sampler, scheduler,
    /// flow_shift, frames, fps, image（data URL，图生图 / 图生视频的起始帧）, end_image, reference_images[], strength,
    /// x_friday{family, files{…}}}。参数不对时返回空、error 写原因
    std::shared_ptr<Job> submit(const json& body, std::string* error);
    std::shared_ptr<Job> find(const std::string& id) const;
    bool cancel(const std::string& id);
    /// 等任务结束（同步接口用；timeoutMs < 0 一直等）
    bool wait(const std::shared_ptr<Job>& job, int timeoutMs);
    /// 释放模型（腾显存）；正在跑的任务结束后才放。返回放之前有没有加载着
    bool release();

    /// 任务的状态（加锁读）
    json describe(const Job& job) const;
    /// 结果的字节（任务完成后才有；没有返回 false）
    bool output(const Job& job, size_t index, Output& out) const;
    json info() const;

private:
    void run();
    void execute(Job& job);
    bool ensureContext(Job& job);
    void freeContext();
    json defaultsFor(const std::string& family) const;

    json m_families;
    int m_idleSeconds;
    mutable std::mutex m_mutex;
    std::condition_variable m_wake;
    std::condition_variable m_done;
    std::deque<std::shared_ptr<Job>> m_queue;
    std::deque<std::shared_ptr<Job>> m_history;
    std::shared_ptr<Job> m_current;
    std::thread m_thread;
    bool m_stop = false;
    bool m_releaseRequested = false;

    // —— 工作线程 ——
    sd_ctx_t* m_ctx = nullptr;
    std::string m_ctxKey;
    std::string m_ctxFamily;
    int64_t m_lastUsed = 0;
    std::atomic<bool> m_loaded { false };
};

} // namespace diffusion
