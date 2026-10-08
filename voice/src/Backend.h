// 选计算设备：ggml 所有后端都静态编在这个文件里，运行时挑一个 GPU（CUDA → Vulkan → Metal → OpenCL），没有就用 CPU。
// GPU 不支持的算子由 ggml_backend_sched 自动落到 CPU。
#pragma once

#include "ggml-backend.h"

#include <string>
#include <vector>

namespace voice {

struct DeviceInfo {
    std::string backend;       // CUDA / Vulkan / Metal / OpenCL / CPU
    std::string name;          // 如 CUDA0、Vulkan0
    std::string description;   // 如 NVIDIA GeForce RTX 5080
    std::string type;          // gpu / igpu / cpu / accel
    size_t memoryFree = 0;
    size_t memoryTotal = 0;
};

/// 本机能用的所有设备（各后端各报一遍，同一张卡可能出现两次：CUDA0 和 Vulkan0）
std::vector<DeviceInfo> listDevices();

class Backend {
public:
    Backend() = default;
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    /// want：auto | cpu | cuda | vulkan | metal | opencl，可带序号（cuda:1、vulkan:0）
    bool init(const std::string& want, int threads, std::string* error);

    ggml_backend_t gpu() const { return m_gpu; }
    ggml_backend_t cpu() const { return m_cpu; }
    /// 放权重的缓冲类型：有 GPU 就是 GPU 的
    ggml_backend_buffer_type_t weightsBufferType() const;
    /// 新建一个调度器（GPU 优先、CPU 兜底）；用完 ggml_backend_sched_free
    ggml_backend_sched_t newScheduler(size_t graphSize) const;

    const DeviceInfo& device() const { return m_device; }
    int threads() const { return m_threads; }
    bool usesGpu() const { return m_gpu != nullptr; }
    /// 用的是 Metal（Apple）
    bool isMetal() const;

private:
    ggml_backend_t m_gpu = nullptr;
    ggml_backend_t m_cpu = nullptr;
    DeviceInfo m_device;
    int m_threads = 4;
};

} // namespace voice
