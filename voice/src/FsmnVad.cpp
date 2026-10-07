#include "FsmnVad.h"

#include "Log.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cstring>

namespace voice {

FsmnVad::~FsmnVad()
{
    if (m_ctx)
        ggml_free(m_ctx);
}

ggml_tensor* FsmnVad::tensor(const std::string& name) const
{
    const auto it = m_tensors.find(name);
    return it == m_tensors.end() ? nullptr : it->second;
}

bool FsmnVad::load(const std::string& path, std::string* error)
{
    gguf_init_params params { false, &m_ctx };   // 权重直接读进 CPU 内存
    gguf_context* gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf) {
        if (error)
            *error = "打不开 VAD 模型：" + path;
        return false;
    }
    const auto u32 = [&](const char* key, int fallback) {
        const int64_t i = gguf_find_key(gguf, key);
        return i < 0 ? fallback : int(gguf_get_val_u32(gguf, i));
    };
    m_inputDim = u32("vad.input_dim", 400);
    m_projDim = u32("vad.proj_dim", 128);
    m_layers = u32("vad.fsmn_layers", 4);
    m_lorder = u32("vad.lorder", 20);
    m_outputDim = u32("vad.output_dim", 248);
    m_lfrM = u32("vad.lfr_m", 5);
    m_lfrN = u32("vad.lfr_n", 1);
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        const char* name = gguf_get_tensor_name(gguf, i);
        m_tensors[name] = ggml_get_tensor(m_ctx, name);
    }
    gguf_free(gguf);

    std::vector<std::string> needed = { "cmvn.shift", "cmvn.scale", "encoder.in_linear1.linear.weight", "encoder.in_linear2.linear.weight",
                                        "encoder.out_linear1.linear.weight", "encoder.out_linear2.linear.weight" };
    for (int i = 0; i < m_layers; ++i) {
        const std::string p = "encoder.fsmn." + std::to_string(i) + ".";
        needed.push_back(p + "linear.linear.weight");
        needed.push_back(p + "fsmn_block.conv_left.weight");
        needed.push_back(p + "affine.linear.weight");
    }
    for (const std::string& name : needed) {
        if (!tensor(name)) {
            if (error)
                *error = "VAD 模型缺 " + name;
            return false;
        }
    }
    if (m_lfrN != 1) {
        if (error)
            *error = "VAD 模型的 LFR 不是 m?n1，流式算法不支持";
        return false;
    }
    m_shift = static_cast<const float*>(tensor("cmvn.shift")->data);
    m_scale = static_cast<const float*>(tensor("cmvn.scale")->data);
    VLOG_INFO("FSMN-VAD 已加载：%s", path.c_str());
    return true;
}

std::vector<float> FsmnVad::speechProbability(const float* features, int frames, ggml_backend_t cpu) const
{
    if (frames < 1)
        return {};
    const auto lin = [&](ggml_context* c, const std::string& w, ggml_tensor* x) {
        ggml_tensor* y = ggml_mul_mat(c, tensor(w + ".weight"), x);
        ggml_tensor* b = tensor(w + ".bias");
        return b ? ggml_add(c, y, b) : y;
    };
    ggml_init_params params { size_t(16) * 1024 * 1024, nullptr, true };
    ggml_context* c = ggml_init(params);
    ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, m_inputDim, frames);
    ggml_set_input(x);
    ggml_tensor* h = lin(c, "encoder.in_linear1.linear", x);
    h = ggml_relu(c, lin(c, "encoder.in_linear2.linear", h));
    for (int i = 0; i < m_layers; ++i) {
        const std::string p = "encoder.fsmn." + std::to_string(i) + ".";
        ggml_tensor* z = ggml_mul_mat(c, tensor(p + "linear.linear.weight"), h);
        ggml_tensor* kernel = tensor(p + "fsmn_block.conv_left.weight");
        ggml_tensor* padded = ggml_pad_ext(c, z, 0, 0, m_lorder - 1, 0, 0, 0, 0, 0);
        ggml_tensor* acc = z;
        for (int j = 0; j < m_lorder; ++j) {
            ggml_tensor* slice = ggml_view_2d(c, padded, m_projDim, frames, padded->nb[1], size_t(j) * padded->nb[1]);
            ggml_tensor* w = ggml_view_1d(c, kernel, m_projDim, size_t(j) * kernel->nb[1]);
            acc = ggml_add(c, acc, ggml_mul(c, slice, w));
        }
        h = ggml_relu(c, lin(c, p + "affine.linear", acc));
    }
    h = lin(c, "encoder.out_linear1.linear", h);
    h = lin(c, "encoder.out_linear2.linear", h);
    h = ggml_soft_max(c, h);
    ggml_set_output(h);
    ggml_cgraph* graph = ggml_new_graph(c);
    ggml_build_forward_expand(graph, h);
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    std::vector<float> probability;
    if (ggml_gallocr_alloc_graph(allocator, graph)) {
        ggml_backend_tensor_set(x, features, 0, ggml_nbytes(x));
        if (ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS) {
            std::vector<float> scores(size_t(m_outputDim) * frames);
            ggml_backend_tensor_get(h, scores.data(), 0, ggml_nbytes(h));
            probability.resize(size_t(frames));
            for (int t = 0; t < frames; ++t)
                probability[size_t(t)] = 1.0f - scores[size_t(t) * m_outputDim];   // 第 0 类是静音
        }
    }
    ggml_gallocr_free(allocator);
    ggml_free(c);
    return probability;
}

VadStream::VadStream(const FsmnVad& model, const Fbank& fbank, int threads) : m_model(model), m_fbank(fbank)
{
    m_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_cpu_set_n_threads(m_cpu, std::max(1, threads));
}

VadStream::~VadStream()
{
    if (m_cpu)
        ggml_backend_free(m_cpu);
}

std::vector<float> VadStream::feed(const float* pcm, size_t count)
{
    m_pending.insert(m_pending.end(), pcm, pcm + count);
    const int ready = Fbank::frameCount(m_pending.size());
    if (ready > 0) {
        const size_t old = m_frames.size();
        m_frames.resize(old + size_t(ready) * Fbank::kMel);
        for (int t = 0; t < ready; ++t)
            m_fbank.frame(m_pending.data() + size_t(t) * Fbank::kShift, m_frames.data() + old + size_t(t) * Fbank::kMel);
        m_pending.erase(m_pending.begin(), m_pending.begin() + std::ptrdiff_t(size_t(ready) * Fbank::kShift));
        m_total += ready;
    }
    return compute(false);
}

std::vector<float> VadStream::finish()
{
    return compute(true);
}

std::vector<float> VadStream::compute(bool final)
{
    // LFR m 帧：帧 i 用 fbank 的 i−pad … i+pad；没到流尾时要等后 pad 帧到了才能算
    const int pad = (m_model.lfrM() - 1) / 2;
    const int available = final ? m_total : m_total - pad;
    if (available <= m_done || m_total <= 0)
        return {};
    const int context = m_model.contextFrames();
    const int start = std::max(0, m_done - context);
    const int count = available - start;
    const int dim = m_model.inputDim();
    const int mel = Fbank::kMel;

    std::vector<float> features(size_t(count) * dim);
    const float* shift = m_model.cmvnShift();
    const float* scale = m_model.cmvnScale();
    for (int i = 0; i < count; ++i) {
        const int frame = start + i;
        for (int j = 0; j < m_model.lfrM(); ++j) {
            const int src = std::clamp(frame + j - pad, 0, m_total - 1);
            const float* row = m_frames.data() + size_t(src - m_base) * mel;
            float* dst = features.data() + size_t(i) * dim + size_t(j) * mel;
            for (int k = 0; k < mel; ++k)
                dst[k] = (row[k] + shift[j * mel + k]) * scale[j * mel + k];
        }
    }
    std::vector<float> probability = m_model.speechProbability(features.data(), count, m_cpu);
    std::vector<float> fresh;
    if (int(probability.size()) == count)
        fresh.assign(probability.begin() + (m_done - start), probability.end());
    m_done = available;

    // 只留下下次要用的 fbank 帧：左侧上下文 + LFR 的 pad
    const int keepFrom = std::max(0, m_done - context - pad - 1);
    if (keepFrom > m_base) {
        m_frames.erase(m_frames.begin(), m_frames.begin() + std::ptrdiff_t(size_t(keepFrom - m_base) * mel));
        m_base = keepFrom;
    }
    return fresh;
}

} // namespace voice
