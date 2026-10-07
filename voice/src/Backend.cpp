#include "Backend.h"

#include "Log.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cctype>

namespace voice {

namespace {

std::string lower(std::string text)
{
    for (char& c : text)
        c = char(std::tolower(static_cast<unsigned char>(c)));
    return text;
}

const char* typeName(enum ggml_backend_dev_type type)
{
    switch (type) {
    case GGML_BACKEND_DEVICE_TYPE_CPU: return "cpu";
    case GGML_BACKEND_DEVICE_TYPE_GPU: return "gpu";
    case GGML_BACKEND_DEVICE_TYPE_IGPU: return "igpu";
    case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accel";
    default: return "other";
    }
}

DeviceInfo describe(ggml_backend_dev_t dev)
{
    DeviceInfo info;
    info.backend = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
    info.name = ggml_backend_dev_name(dev);
    info.description = ggml_backend_dev_description(dev);
    info.type = typeName(ggml_backend_dev_type(dev));
    ggml_backend_dev_memory(dev, &info.memoryFree, &info.memoryTotal);
    return info;
}

bool isGpu(ggml_backend_dev_t dev)
{
    const auto type = ggml_backend_dev_type(dev);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

/// auto 的优先顺序：CUDA 最快；其次 Vulkan（独显优先于核显）；Metal；OpenCL
int score(ggml_backend_dev_t dev)
{
    const std::string backend = lower(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)));
    const bool discrete = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
    int base = 0;
    if (backend.find("cuda") != std::string::npos)
        base = 400;
    else if (backend.find("metal") != std::string::npos)
        base = 350;
    else if (backend.find("vulkan") != std::string::npos)
        base = 300;
    else if (backend.find("opencl") != std::string::npos)
        base = 200;
    else
        base = 100;
    return base + (discrete ? 10 : 0);
}

} // namespace

std::vector<DeviceInfo> listDevices()
{
    std::vector<DeviceInfo> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        devices.push_back(describe(ggml_backend_dev_get(i)));
    return devices;
}

Backend::~Backend()
{
    if (m_gpu)
        ggml_backend_free(m_gpu);
    if (m_cpu)
        ggml_backend_free(m_cpu);
}

bool Backend::init(const std::string& wantRaw, int threads, std::string* error)
{
    m_threads = std::max(1, threads);
    std::string want = lower(wantRaw.empty() ? std::string("auto") : wantRaw);
    int index = -1;
    if (const auto colon = want.find(':'); colon != std::string::npos) {
        index = std::atoi(want.c_str() + colon + 1);
        want = want.substr(0, colon);
    }

    m_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!m_cpu) {
        if (error)
            *error = "CPU 后端初始化失败";
        return false;
    }
    ggml_backend_cpu_set_n_threads(m_cpu, m_threads);

    ggml_backend_dev_t chosen = nullptr;
    if (want != "cpu") {
        std::vector<ggml_backend_dev_t> candidates;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!isGpu(dev))
                continue;
            const std::string backend = lower(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)));
            if (want != "auto" && backend.find(want) == std::string::npos)
                continue;
            candidates.push_back(dev);
        }
        if (want == "auto") {
            std::stable_sort(candidates.begin(), candidates.end(),
                             [](ggml_backend_dev_t a, ggml_backend_dev_t b) { return score(a) > score(b); });
            if (!candidates.empty())
                chosen = candidates.front();
        } else if (!candidates.empty()) {
            const int pick = index < 0 ? 0 : index;
            if (pick >= int(candidates.size())) {
                if (error)
                    *error = "没有第 " + std::to_string(pick) + " 个 " + want + " 设备";
                return false;
            }
            chosen = candidates[size_t(pick)];
        } else {
            if (error)
                *error = "这个版本里没有可用的 " + want + " 设备（没编进来，或驱动 / 显卡不支持）";
            return false;
        }
    }

    if (chosen) {
        m_gpu = ggml_backend_dev_init(chosen, nullptr);
        if (!m_gpu) {
            VLOG_WARN("初始化 %s 失败，改用 CPU", ggml_backend_dev_description(chosen));
            chosen = nullptr;
        }
    }
    if (chosen) {
        m_device = describe(chosen);
    } else {
        m_device = describe(ggml_backend_get_device(m_cpu));
        m_device.backend = "CPU";
    }
    VLOG_INFO("计算设备：%s %s（%s）", m_device.backend.c_str(), m_device.name.c_str(), m_device.description.c_str());
    return true;
}

ggml_backend_buffer_type_t Backend::weightsBufferType() const
{
    return ggml_backend_get_default_buffer_type(m_gpu ? m_gpu : m_cpu);
}

ggml_backend_sched_t Backend::newScheduler(size_t graphSize) const
{
    ggml_backend_t backends[2];
    int n = 0;
    if (m_gpu)
        backends[n++] = m_gpu;
    backends[n++] = m_cpu;
    return ggml_backend_sched_new(backends, nullptr, n, graphSize, false, true);
}

} // namespace voice
