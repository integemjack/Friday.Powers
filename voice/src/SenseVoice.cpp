#include "SenseVoice.h"

#include "Log.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>

namespace voice {

namespace {

constexpr float kLayerNormEps = 1e-5f;
constexpr int kInputDim = 560;   // LFR m7 × 80
constexpr int kLfrM = 7, kLfrN = 6;
constexpr size_t kGraphSize = 16384;

/// SenseVoice 的查询 token：语言 / 事件情绪（固定 1、2）/ 是否带标点（FunASR SenseVoiceSmall 的 lid_dict、textnorm_dict）
int languageId(const std::string& language)
{
    if (language == "zh") return 3;
    if (language == "en") return 4;
    if (language == "yue") return 7;
    if (language == "ja") return 11;
    if (language == "ko") return 12;
    return 0;   // auto
}

/// 正弦位置编码（深度 = 输入维度 560），pos 从 1 开始
void addPositionalEncoding(std::vector<float>& x, int frames, int depth)
{
    const double increment = std::log(10000.0) / (depth / 2.0 - 1.0);
    for (int t = 0; t < frames; ++t) {
        const double pos = t + 1;
        for (int i = 0; i < depth / 2; ++i) {
            const double v = pos * std::exp(i * -increment);
            x[size_t(t) * depth + i] += float(std::sin(v));
            x[size_t(t) * depth + depth / 2 + i] += float(std::cos(v));
        }
    }
}

std::string trimSpaces(const std::string& s)
{
    const size_t a = s.find_first_not_of(' ');
    if (a == std::string::npos)
        return {};
    return s.substr(a, s.find_last_not_of(' ') - a + 1);
}

} // namespace

SenseVoice::SenseVoice(Backend& backend) : m_backend(backend) {}

SenseVoice::~SenseVoice()
{
    if (m_sched)
        ggml_backend_sched_free(m_sched);
    if (m_weights)
        ggml_backend_buffer_free(m_weights);
    if (m_weightsCtx)
        ggml_free(m_weightsCtx);
}

bool SenseVoice::supportsLanguage(const std::string& language)
{
    return language.empty() || language == "auto" || language == "zh" || language == "en" || language == "yue"
        || language == "ja" || language == "ko";
}

ggml_tensor* SenseVoice::tensor(const std::string& name)
{
    const auto it = m_tensors.find(name);
    if (it == m_tensors.end()) {
        VLOG_ERROR("SenseVoice 模型里缺 %s", name.c_str());
        return nullptr;
    }
    return it->second;
}

bool SenseVoice::load(const std::string& path, std::string* error)
{
    const auto fail = [&](const std::string& message) {
        if (error)
            *error = message;
        return false;
    };
    ggml_context* meta = nullptr;
    gguf_init_params params { true, &meta };
    gguf_context* gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf)
        return fail("打不开模型：" + path);

    const auto u32 = [&](const char* key, int fallback) {
        const int64_t i = gguf_find_key(gguf, key);
        return i < 0 ? fallback : int(gguf_get_val_u32(gguf, i));
    };
    m_config.dModel = u32("sv.output_size", 512);
    m_config.heads = u32("sv.attention_heads", 4);
    m_config.blocks = u32("sv.num_blocks", 50);
    m_config.tpBlocks = u32("sv.tp_blocks", 20);
    m_config.kernel = u32("sv.kernel_size", 11);
    m_config.vocab = u32("sv.vocab_size", 25055);
    m_config.blank = u32("sv.blank_id", 0);
    if (const int64_t key = gguf_find_key(gguf, "sv.vocab"); key >= 0) {
        const size_t n = gguf_get_arr_n(gguf, key);
        m_vocab.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const char* s = gguf_get_arr_str(gguf, key, i);
            m_vocab[i] = s ? s : "";
        }
    }
    if (m_vocab.empty()) {
        gguf_free(gguf);
        ggml_free(meta);
        return fail("模型里没有词表（sv.vocab），不是 FunASR 导出的 SenseVoice GGUF？");
    }

    const int64_t count = gguf_get_n_tensors(gguf);
    ggml_init_params wp { size_t(count + 1) * ggml_tensor_overhead(), nullptr, true };
    m_weightsCtx = ggml_init(wp);
    for (int64_t i = 0; i < count; ++i) {
        const char* name = gguf_get_tensor_name(gguf, i);
        ggml_tensor* weight = ggml_dup_tensor(m_weightsCtx, ggml_get_tensor(meta, name));
        ggml_set_name(weight, name);
        m_tensors[name] = weight;
    }
    m_weights = ggml_backend_alloc_ctx_tensors_from_buft(m_weightsCtx, m_backend.weightsBufferType());
    if (!m_weights) {
        gguf_free(gguf);
        ggml_free(meta);
        return fail("给模型权重分配显存 / 内存失败");
    }
    ggml_backend_buffer_set_usage(m_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    m_weightsBytes = ggml_backend_buffer_get_size(m_weights);

    std::ifstream in(path, std::ios::binary);
    std::vector<char> buffer;
    const size_t base = gguf_get_data_offset(gguf);
    for (int64_t i = 0; i < count; ++i) {
        ggml_tensor* weight = m_tensors[gguf_get_tensor_name(gguf, i)];
        const size_t bytes = ggml_nbytes(weight);
        buffer.resize(bytes);
        in.seekg(std::streamoff(base + gguf_get_tensor_offset(gguf, i)));
        in.read(buffer.data(), std::streamsize(bytes));
        if (!in) {
            gguf_free(gguf);
            ggml_free(meta);
            return fail(std::string("读模型权重失败：") + ggml_get_name(weight));
        }
        ggml_backend_tensor_set(weight, buffer.data(), 0, bytes);
    }
    gguf_free(gguf);
    ggml_free(meta);

    ggml_tensor* embed = tensor("embed.weight");
    if (!embed || embed->ne[0] != kInputDim)
        return fail("模型的 embed.weight 形状不对");
    m_embed.resize(size_t(ggml_nelements(embed)));
    if (embed->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(embed, m_embed.data(), 0, ggml_nbytes(embed));
    } else if (embed->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(m_embed.size());
        ggml_backend_tensor_get(embed, half.data(), 0, ggml_nbytes(embed));
        for (size_t i = 0; i < half.size(); ++i)
            m_embed[i] = ggml_fp16_to_fp32(half[i]);
    } else {
        return fail(std::string("不支持的 embed 类型：") + ggml_type_name(embed->type));
    }
    m_sched = m_backend.newScheduler(kGraphSize);
    m_path = path;
    VLOG_INFO("SenseVoice 已加载：%s（%.0f MB，词表 %zu）", path.c_str(), m_weightsBytes / 1048576.0, m_vocab.size());
    return true;
}

ggml_tensor* SenseVoice::linear(ggml_context* ctx, const std::string& weight, ggml_tensor* x)
{
    ggml_tensor* y = ggml_mul_mat(ctx, tensor(weight + ".weight"), x);
    const auto bias = m_tensors.find(weight + ".bias");
    return bias == m_tensors.end() ? y : ggml_add(ctx, y, bias->second);
}

ggml_tensor* SenseVoice::layerNorm(ggml_context* ctx, const std::string& prefix, ggml_tensor* x)
{
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, kLayerNormEps), tensor(prefix + ".weight")), tensor(prefix + ".bias"));
}

ggml_tensor* SenseVoice::attention(ggml_context* ctx, const std::string& p, ggml_tensor* x, int frames)
{
    const int d = m_config.dModel, heads = m_config.heads, dk = d / heads, kernel = m_config.kernel;
    ggml_tensor* qkv = linear(ctx, p + "linear_q_k_v", x);
    const size_t row = qkv->nb[1];
    ggml_tensor* q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, d, frames, row, 0));
    ggml_tensor* k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, d, frames, row, size_t(d) * sizeof(float)));
    ggml_tensor* v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, d, frames, row, size_t(2) * d * sizeof(float)));

    // FSMN 记忆块：对 v 沿时间做深度卷积（核 11、两边各补 5）
    const int pad = (kernel - 1) / 2;
    ggml_tensor* weights = tensor(p + "fsmn_block.weight");
    ggml_tensor* padded = ggml_pad_ext(ctx, v, 0, 0, pad, pad, 0, 0, 0, 0);
    ggml_tensor* memory = v;
    for (int j = 0; j < kernel; ++j) {
        ggml_tensor* slice = ggml_view_2d(ctx, padded, d, frames, padded->nb[1], size_t(j) * padded->nb[1]);
        ggml_tensor* w = ggml_view_1d(ctx, weights, d, size_t(j) * weights->nb[1]);
        memory = ggml_add(ctx, memory, ggml_mul(ctx, ggml_cont(ctx, slice), w));
    }

    q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, dk, heads, frames), 0, 2, 1, 3);
    k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, dk, heads, frames), 0, 2, 1, 3);
    ggml_tensor* vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, dk, heads, frames), 1, 2, 0, 3));
    ggml_tensor* scores = ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1.0f / std::sqrt(float(dk))));
    ggml_tensor* out = ggml_cont_2d(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, vh, scores), 0, 2, 1, 3), d, frames);
    return ggml_add(ctx, linear(ctx, p + "linear_out", out), memory);
}

ggml_tensor* SenseVoice::layer(ggml_context* ctx, const std::string& p, ggml_tensor* x, int frames, bool residual)
{
    ggml_tensor* h = layerNorm(ctx, p + "norm1", x);
    ggml_tensor* attended = attention(ctx, p + "self_attn.", h, frames);
    x = residual ? ggml_add(ctx, x, attended) : attended;
    h = layerNorm(ctx, p + "norm2", x);
    h = ggml_relu(ctx, linear(ctx, p + "feed_forward.w_1", h));
    h = linear(ctx, p + "feed_forward.w_2", h);
    return ggml_add(ctx, x, h);
}

std::string SenseVoice::detokenize(const std::vector<int>& ids, Transcript* out) const
{
    std::string text;
    for (int id : ids) {
        if (id < 0 || id >= int(m_vocab.size()))
            continue;
        const std::string& piece = m_vocab[size_t(id)];
        if (piece.size() >= 4 && piece.compare(0, 2, "<|") == 0) {
            // <|zh|><|NEUTRAL|><|Speech|><|withitn|>
            const std::string tag = piece.substr(2, piece.size() - 4);
            if (tag == "zh" || tag == "en" || tag == "yue" || tag == "ja" || tag == "ko" || tag == "nospeech")
                out->language = tag;
            else if (tag == "NEUTRAL" || tag == "HAPPY" || tag == "SAD" || tag == "ANGRY" || tag == "FEARFUL" || tag == "DISGUSTED"
                     || tag == "SURPRISED" || tag == "EMO_UNKNOWN")
                out->emotion = tag;
            else if (tag != "withitn" && tag != "woitn")
                out->event = tag;
            continue;
        }
        text += piece;
    }
    static const std::string kSpace = "\xe2\x96\x81";   // ▁
    for (size_t at; (at = text.find(kSpace)) != std::string::npos;)
        text.replace(at, kSpace.size(), " ");
    return trimSpaces(text);
}

Transcript SenseVoice::transcribe(const float* pcm, size_t count, const std::string& language, bool itn)
{
    Transcript result;
    const auto started = std::chrono::steady_clock::now();
    if (!m_weights) {
        result.error = "模型没有加载";
        return result;
    }
    if (count < size_t(Fbank::kWindow)) {
        result.ok = true;
        result.language = "nospeech";
        return result;
    }

    // 特征：80 维 fbank → LFR(7,6) → 前面加 4 个查询 token → ×√d → 位置编码（都在 CPU 上，便宜）
    int frames = 0;
    const std::vector<float> fbank = m_fbank.compute(pcm, count, &frames);
    int lfrFrames = 0;
    const std::vector<float> features = Fbank::lfr(fbank.data(), frames, kLfrM, kLfrN, &lfrFrames);
    const int queries[4] = { languageId(language), 1, 2, itn ? 14 : 15 };
    const int n = 4 + lfrFrames;
    std::vector<float> input(size_t(n) * kInputDim);
    for (int i = 0; i < 4; ++i)
        std::memcpy(&input[size_t(i) * kInputDim], &m_embed[size_t(queries[i]) * kInputDim], kInputDim * sizeof(float));
    std::memcpy(&input[size_t(4) * kInputDim], features.data(), features.size() * sizeof(float));
    const float scale = std::sqrt(float(m_config.dModel));
    for (float& v : input)
        v *= scale;
    addPositionalEncoding(input, n, kInputDim);

    std::lock_guard<std::mutex> lock(m_mutex);
    ggml_init_params params { ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false), nullptr, true };
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kInputDim, n);
    ggml_set_name(x, "input");
    ggml_set_input(x);
    ggml_tensor* h = layer(ctx, "encoder.encoders0.0.", x, n, false);
    for (int i = 0; i < m_config.blocks - 1; ++i)
        h = layer(ctx, "encoder.encoders." + std::to_string(i) + ".", h, n, true);
    h = layerNorm(ctx, "encoder.after_norm", h);
    for (int i = 0; i < m_config.tpBlocks; ++i)
        h = layer(ctx, "encoder.tp_encoders." + std::to_string(i) + ".", h, n, true);
    h = layerNorm(ctx, "encoder.tp_norm", h);
    ggml_tensor* logits = linear(ctx, "ctc.ctc_lo", h);   // [vocab, n]
    ggml_tensor* best = ggml_argmax(ctx, logits);          // [n]（CTC 贪心：每帧取最大）
    ggml_set_name(best, "best");
    ggml_set_output(best);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, kGraphSize, false);
    ggml_build_forward_expand(graph, best);

    ggml_backend_sched_reset(m_sched);
    if (!ggml_backend_sched_alloc_graph(m_sched, graph)) {
        ggml_free(ctx);
        result.error = "分配计算图失败（显存不够？）";
        return result;
    }
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    if (ggml_backend_sched_graph_compute(m_sched, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        result.error = "推理失败";
        return result;
    }
    std::vector<int32_t> bestIds(static_cast<size_t>(n));
    ggml_backend_tensor_get(best, bestIds.data(), 0, bestIds.size() * sizeof(int32_t));
    ggml_free(ctx);

    std::vector<int> ids;
    int previous = -1;
    for (int32_t id : bestIds) {
        if (id != previous && id != m_config.blank)
            ids.push_back(id);
        previous = id;
    }
    result.text = detokenize(ids, &result);
    result.ok = true;
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

} // namespace voice
