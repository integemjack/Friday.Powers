#include "SpeakerEncoder.h"

#include "Log.h"
#include "OnnxWeights.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace voice {

namespace {

constexpr int kMel = Fbank::kMel;
constexpr int kGraphSize = 8192;
constexpr int kSegment = 100;   // CAM 层分段平均的段长（帧）
// 网络结构（3D-Speaker CAMPPlus）：FCM 头 12 个 2D 卷积、TDNN 1 个、三个稠密块各层 4 个卷积、3 个过渡层、全连接 1 个
constexpr int kFcmConvs = 12;
constexpr int kBlockLayers[3] = { 12, 24, 16 };

std::vector<int> intsOr(const OnnxNode& node, const char* name, std::vector<int> fallback)
{
    const auto it = node.ints.find(name);
    if (it == node.ints.end())
        return fallback;
    std::vector<int> values;
    for (const int64_t v : it->second)
        values.push_back(int(v));
    return values;
}

} // namespace

SpeakerEncoder::SpeakerEncoder(Backend& backend)
    : m_backend(backend)
{
}

SpeakerEncoder::~SpeakerEncoder()
{
    if (m_sched)
        ggml_backend_sched_free(m_sched);
    if (m_weights)
        ggml_backend_buffer_free(m_weights);
    if (m_weightsCtx)
        ggml_free(m_weightsCtx);
}

bool SpeakerEncoder::load(const std::string& onnxPath, std::string* error)
{
    const auto fail = [&](const std::string& message) {
        if (error)
            *error = message;
        return false;
    };
    OnnxWeights onnx;
    std::string reason;
    if (!onnx.load(onnxPath, &reason))
        return fail(reason);

    // 先把 Conv / BatchNormalization 按先后读出来（形状、属性、数据），再一次建 ggml 张量
    struct ConvData {
        Conv conv;
        std::vector<int64_t> dims;
        const OnnxTensor* weight;
        const OnnxTensor* bias;
    };
    struct NormData {
        std::vector<float> scale, shift;
    };
    std::vector<ConvData> convs;
    std::vector<NormData> norms;
    for (const OnnxNode& node : onnx.nodes()) {
        if (node.op == "Conv") {
            if (node.inputs.size() < 2)
                return fail("Conv 节点没有权重");
            const OnnxTensor* weight = onnx.tensor(node.inputs[1]);
            const OnnxTensor* bias = node.inputs.size() > 2 && !node.inputs[2].empty() ? onnx.tensor(node.inputs[2]) : nullptr;
            if (!weight)
                return fail("找不到卷积权重 " + node.inputs[1]);
            if (intsOr(node, "group", { 1 }).front() != 1)
                return fail("不支持分组卷积");
            ConvData data { {}, weight->dims, weight, bias };
            Conv& conv = data.conv;
            conv.cout = int(weight->dims[0]);
            conv.cin = int(weight->dims[1]);
            if (weight->dims.size() == 4) {
                conv.twoD = true;
                conv.kh = int(weight->dims[2]);
                conv.kw = int(weight->dims[3]);
                const auto strides = intsOr(node, "strides", { 1, 1 });
                const auto pads = intsOr(node, "pads", { 0, 0, 0, 0 });
                conv.sh = strides[0];
                conv.sw = strides[1];
                conv.ph = pads[0];
                conv.pw = pads[1];
            } else if (weight->dims.size() == 3) {
                conv.kernel = int(weight->dims[2]);
                conv.stride = intsOr(node, "strides", { 1 }).front();
                conv.pad = intsOr(node, "pads", { 0, 0 }).front();
                conv.dilation = intsOr(node, "dilations", { 1 }).front();
            } else {
                return fail("卷积权重的维数不对：" + node.inputs[1]);
            }
            if (bias && int(bias->data.size()) != conv.cout)
                return fail("卷积偏置长度不对：" + node.inputs[2]);
            convs.push_back(std::move(data));
        } else if (node.op == "BatchNormalization") {
            if (node.inputs.size() < 5)
                return fail("BatchNormalization 节点输入不全");
            const OnnxTensor* mean = onnx.tensor(node.inputs[3]);
            const OnnxTensor* var = onnx.tensor(node.inputs[4]);
            if (!mean || !var || mean->data.size() != var->data.size())
                return fail("BatchNormalization 缺均值 / 方差");
            const size_t channels = mean->data.size();
            // 没有仿射参数的（全连接后面那个）：scale、bias 是常量 1、0
            const OnnxTensor* gamma = onnx.tensor(node.inputs[1]);
            const OnnxTensor* beta = onnx.tensor(node.inputs[2]);
            const auto eps = node.floats.find("epsilon");
            const float epsilon = eps == node.floats.end() ? 1e-5f : eps->second;
            NormData data;
            data.scale.resize(channels);
            data.shift.resize(channels);
            for (size_t c = 0; c < channels; ++c) {
                const float g = gamma && gamma->data.size() == channels ? gamma->data[c] : 1.0f;
                const float b = beta && beta->data.size() == channels ? beta->data[c] : 0.0f;
                const float s = g / std::sqrt(var->data[c] + epsilon);
                data.scale[c] = s;
                data.shift[c] = b - mean->data[c] * s;
            }
            norms.push_back(std::move(data));
        }
    }
    int layers = 0;
    for (int n : kBlockLayers)
        layers += n;
    const size_t expectedConvs = size_t(kFcmConvs + 1 + layers * 4 + 3 + 1);
    const size_t expectedNorms = size_t(layers + 3 + 1);
    if (convs.size() != expectedConvs || norms.size() != expectedNorms)
        return fail("不是 CAM++（campplus）模型：卷积 " + std::to_string(convs.size()) + " 个（应为 " + std::to_string(expectedConvs)
                    + "）、BN " + std::to_string(norms.size()) + " 个（应为 " + std::to_string(expectedNorms) + "）");
    for (int i = 0; i < kFcmConvs; ++i) {
        if (!convs[size_t(i)].conv.twoD)
            return fail("CAM++ 的前 12 个卷积应是 2D 的");
    }
    if (convs[0].conv.cin != 1 || convs[kFcmConvs].conv.twoD)
        return fail("CAM++ 的卷积头形状不对");

    // 建权重张量
    const size_t tensorCount = convs.size() * 2 + norms.size() * 2 + 1;
    ggml_init_params params { tensorCount * ggml_tensor_overhead(), nullptr, true };
    m_weightsCtx = ggml_init(params);
    for (ConvData& data : convs) {
        Conv& conv = data.conv;
        if (conv.twoD) {
            conv.weight = ggml_new_tensor_4d(m_weightsCtx, GGML_TYPE_F32, conv.kw, conv.kh, conv.cin, conv.cout);
            if (data.bias)
                conv.bias = ggml_new_tensor_4d(m_weightsCtx, GGML_TYPE_F32, 1, 1, conv.cout, 1);
        } else {
            conv.weight = ggml_new_tensor_2d(m_weightsCtx, GGML_TYPE_F32, int64_t(conv.cin) * conv.kernel, conv.cout);
            if (data.bias)
                conv.bias = ggml_new_tensor_2d(m_weightsCtx, GGML_TYPE_F32, conv.cout, 1);
        }
    }
    m_norms.resize(norms.size());
    for (size_t i = 0; i < norms.size(); ++i) {
        const int64_t channels = int64_t(norms[i].scale.size());
        m_norms[i].scale = ggml_new_tensor_2d(m_weightsCtx, GGML_TYPE_F32, channels, 1);
        m_norms[i].shift = ggml_new_tensor_2d(m_weightsCtx, GGML_TYPE_F32, channels, 1);
    }
    m_weights = ggml_backend_alloc_ctx_tensors_from_buft(m_weightsCtx, m_backend.weightsBufferType());
    if (!m_weights)
        return fail("给声纹模型的权重分配显存 / 内存失败");
    ggml_backend_buffer_set_usage(m_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (ConvData& data : convs) {
        if (size_t(ggml_nelements(data.conv.weight)) != data.weight->data.size())
            return fail("卷积权重的大小不对");
        ggml_backend_tensor_set(data.conv.weight, data.weight->data.data(), 0, ggml_nbytes(data.conv.weight));
        if (data.conv.bias)
            ggml_backend_tensor_set(data.conv.bias, data.bias->data.data(), 0, ggml_nbytes(data.conv.bias));
        m_convs.push_back(data.conv);
    }
    for (size_t i = 0; i < norms.size(); ++i) {
        ggml_backend_tensor_set(m_norms[i].scale, norms[i].scale.data(), 0, ggml_nbytes(m_norms[i].scale));
        ggml_backend_tensor_set(m_norms[i].shift, norms[i].shift.data(), 0, ggml_nbytes(m_norms[i].shift));
    }
    m_sched = m_backend.newScheduler(kGraphSize);
    m_path = onnxPath;
    VLOG_INFO("声纹模型 %s：%zu 个参数，%zu 个卷积", onnxPath.c_str(), onnx.parameterCount(), m_convs.size());
    return true;
}

ggml_tensor* SpeakerEncoder::conv2d(ggml_context* ctx, const Conv& conv, ggml_tensor* x) const
{
    // x：[W=T, H=F, C, 1]；ONNX 的 strides / pads 是 (H, W)
    ggml_tensor* y = ggml_conv_2d(ctx, conv.weight, x, conv.sw, conv.sh, conv.pw, conv.ph, 1, 1);
    return conv.bias ? ggml_add(ctx, y, conv.bias) : y;
}

ggml_tensor* SpeakerEncoder::conv1dTime(ggml_context* ctx, const Conv& conv, ggml_tensor* x) const
{
    // 自己 im2col（F32）：ggml_conv_1d 会把激活展开成 F16，精度不够
    ggml_tensor* kernel = ggml_reshape_3d(ctx, conv.weight, conv.kernel, conv.cin, conv.cout);
    ggml_tensor* columns = ggml_im2col(ctx, kernel, x, conv.stride, 0, conv.pad, 0, conv.dilation, 0, false, GGML_TYPE_F32);
    columns = ggml_reshape_2d(ctx, columns, columns->ne[0], columns->ne[1]);
    ggml_tensor* y = ggml_mul_mat(ctx, conv.weight, columns);   // [Cout, Tout]
    return conv.bias ? ggml_add(ctx, y, conv.bias) : y;
}

ggml_tensor* SpeakerEncoder::conv1x1(ggml_context* ctx, const Conv& conv, ggml_tensor* x) const
{
    ggml_tensor* y = ggml_mul_mat(ctx, conv.weight, x);   // [Cin, Cout] × [Cin, T] → [Cout, T]
    return conv.bias ? ggml_add(ctx, y, conv.bias) : y;
}

ggml_tensor* SpeakerEncoder::norm(ggml_context* ctx, const Norm& n, ggml_tensor* x) const
{
    return ggml_add(ctx, ggml_mul(ctx, x, n.scale), n.shift);
}

std::vector<float> SpeakerEncoder::features(const float* pcm, size_t count, int* frames) const
{
    // Fbank 里按 FunASR 的口径先乘 32768；CosyVoice 给 CAM++ 的 fbank 是 [-1,1] 的录音直接算（静音帧碰到 log 下限的不一样，
    // 声纹差到 0.99）：先除回去
    std::vector<float> scaled(pcm, pcm + count);
    for (float& v : scaled)
        v *= 1.0f / 32768.0f;
    int n = 0;
    std::vector<float> feat = m_fbank.compute(scaled.data(), scaled.size(), &n);
    if (n > 0) {
        // 整句减去均值（CosyVoice：feat - feat.mean(dim=0)）
        std::vector<double> mean(kMel, 0.0);
        for (int t = 0; t < n; ++t)
            for (int m = 0; m < kMel; ++m)
                mean[size_t(m)] += feat[size_t(t) * kMel + m];
        for (int m = 0; m < kMel; ++m)
            mean[size_t(m)] /= n;
        for (int t = 0; t < n; ++t)
            for (int m = 0; m < kMel; ++m)
                feat[size_t(t) * kMel + m] -= float(mean[size_t(m)]);
    }
    if (frames)
        *frames = n;
    return feat;
}

SpeakerEmbedding SpeakerEncoder::embed(const float* pcm, size_t count)
{
    SpeakerEmbedding result;
    if (!loaded()) {
        result.error = "声纹模型没加载";
        return result;
    }
    const size_t minSamples = size_t(kMinSeconds * Fbank::kSampleRate);
    const size_t maxSamples = size_t(kMaxSeconds * Fbank::kSampleRate);
    if (count < minSamples) {
        result.error = "录音太短（不到 0.5 秒）";
        return result;
    }
    // 太长：用中间的 20 秒
    if (count > maxSamples) {
        pcm += (count - maxSamples) / 2;
        count = maxSamples;
    }
    int frames = 0;
    const std::vector<float> feat = features(pcm, count, &frames);
    result = embedFeatures(feat, frames);
    result.audioSeconds = double(count) / Fbank::kSampleRate;
    return result;
}

SpeakerEmbedding SpeakerEncoder::embedFeatures(const std::vector<float>& feat, int frames)
{
    SpeakerEmbedding result;
    const auto started = std::chrono::steady_clock::now();
    if (!loaded()) {
        result.error = "声纹模型没加载";
        return result;
    }
    const Conv& tdnn = m_convs[kFcmConvs];
    const int t2 = frames < 1 ? 0 : (frames + 2 * tdnn.pad - tdnn.dilation * (tdnn.kernel - 1) - 1) / tdnn.stride + 1;
    if (frames < 8 || t2 < 2 || feat.size() != size_t(frames) * kMel) {
        result.error = "特征太短";
        return result;
    }
    // 输入：[T, 80]（时间在前；2D 卷积看成 W=T、H=频率）
    std::vector<float> input(size_t(frames) * kMel);
    for (int t = 0; t < frames; ++t)
        for (int m = 0; m < kMel; ++m)
            input[size_t(m) * frames + t] = feat[size_t(t) * kMel + m];
    // CAM 层的上下文 = 整句平均 + 所在 100 帧段的平均：一个 [T2, T2] 的矩阵乘出来
    std::vector<float> context(size_t(t2) * t2);
    for (int a = 0; a < t2; ++a)
        for (int b = 0; b < t2; ++b)
            context[size_t(a) * t2 + b] = 1.0f / t2 + (a / kSegment == b / kSegment ? 1.0f / kSegment : 0.0f);

    std::lock_guard<std::mutex> lock(m_mutex);
    ggml_init_params params { ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false), nullptr, true };
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, frames, kMel, 1, 1);
    ggml_set_name(x, "input");
    ggml_set_input(x);
    ggml_tensor* contextMatrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t2, t2);
    ggml_set_name(contextMatrix, "context");
    ggml_set_input(contextMatrix);

    size_t c = 0, b = 0;
    // FCM：conv1 → layer1、layer2（各两块：第一块频率步长 2、带捷径卷积）→ conv2
    ggml_tensor* h = ggml_relu(ctx, conv2d(ctx, m_convs[c++], x));
    for (int layer = 0; layer < 2; ++layer) {
        const Conv& a1 = m_convs[c++];
        const Conv& a2 = m_convs[c++];
        const Conv& shortcut = m_convs[c++];
        ggml_tensor* out = conv2d(ctx, a2, ggml_relu(ctx, conv2d(ctx, a1, h)));
        h = ggml_relu(ctx, ggml_add(ctx, out, conv2d(ctx, shortcut, h)));
        const Conv& b1 = m_convs[c++];
        const Conv& b2 = m_convs[c++];
        out = conv2d(ctx, b2, ggml_relu(ctx, conv2d(ctx, b1, h)));
        h = ggml_relu(ctx, ggml_add(ctx, out, h));
    }
    h = ggml_relu(ctx, conv2d(ctx, m_convs[c++], h));
    // [T, F', C] → [T, C·F']（通道号 = c·F' + f）
    h = ggml_reshape_2d(ctx, h, h->ne[0], h->ne[1] * h->ne[2]);
    // TDNN（时间步长 2）→ [128, T2]
    h = ggml_relu(ctx, conv1dTime(ctx, m_convs[c++], h));
    for (int block = 0; block < 3; ++block) {
        for (int layer = 0; layer < kBlockLayers[block]; ++layer) {
            ggml_tensor* y = ggml_relu(ctx, norm(ctx, m_norms[b++], h));
            y = ggml_relu(ctx, conv1x1(ctx, m_convs[c++], y));            // 瓶颈 1×1（第二个 BN 并在里面）
            ggml_tensor* yt = ggml_cont(ctx, ggml_transpose(ctx, y));    // [T2, 128]
            ggml_tensor* local = conv1dTime(ctx, m_convs[c++], yt);       // [32, T2]
            ggml_tensor* ctxv = ggml_mul_mat(ctx, yt, contextMatrix);     // [128, T2]
            ctxv = ggml_relu(ctx, conv1x1(ctx, m_convs[c++], ctxv));
            ggml_tensor* mask = ggml_sigmoid(ctx, conv1x1(ctx, m_convs[c++], ctxv));
            h = ggml_concat(ctx, h, ggml_mul(ctx, local, mask), 0);
        }
        // 过渡层：BN → ReLU → 1×1（最后一个的 1×1 并了 out_nonlinear 的 BN，后面再 ReLU）
        h = ggml_relu(ctx, norm(ctx, m_norms[b++], h));
        h = conv1x1(ctx, m_convs[c++], h);
        if (block == 2)
            h = ggml_relu(ctx, h);
    }
    // 统计池化：每个通道的均值和标准差（无偏）
    ggml_tensor* ht = ggml_cont(ctx, ggml_transpose(ctx, h));   // [T2, C]
    ggml_tensor* mean = ggml_mean(ctx, ht);                      // [1, C]
    ggml_tensor* diff = ggml_sub(ctx, ht, mean);
    ggml_tensor* stdev = ggml_sqrt(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, diff)), 1.0f / float(t2 - 1)));
    ggml_tensor* stats = ggml_concat(ctx, mean, stdev, 1);      // [1, 2C]
    stats = ggml_reshape_2d(ctx, stats, stats->ne[1], 1);
    ggml_tensor* embedding = norm(ctx, m_norms[b++], conv1x1(ctx, m_convs[c++], stats));   // [192, 1]
    ggml_set_name(embedding, "embedding");
    ggml_set_output(embedding);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, kGraphSize, false);
    ggml_build_forward_expand(graph, embedding);

    ggml_backend_sched_reset(m_sched);
    if (!ggml_backend_sched_alloc_graph(m_sched, graph)) {
        ggml_free(ctx);
        result.error = "分配声纹计算图失败（显存不够？）";
        return result;
    }
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(contextMatrix, context.data(), 0, ggml_nbytes(contextMatrix));
    if (ggml_backend_sched_graph_compute(m_sched, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        result.error = "声纹推理失败";
        return result;
    }
    result.vector.resize(size_t(ggml_nelements(embedding)));
    ggml_backend_tensor_get(embedding, result.vector.data(), 0, ggml_nbytes(embedding));
    ggml_free(ctx);

    double norm2 = 0;
    for (float v : result.vector)
        norm2 += double(v) * v;
    const float inv = norm2 > 0 ? float(1.0 / std::sqrt(norm2)) : 0.0f;
    for (float& v : result.vector)
        v *= inv;
    result.ok = result.vector.size() == size_t(kDim);
    if (!result.ok)
        result.error = "声纹维数不对";
    result.audioSeconds = double(frames) / 100.0;
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

float SpeakerEncoder::cosine(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size() || a.empty())
        return 0.0f;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += double(a[i]) * b[i];
        na += double(a[i]) * a[i];
        nb += double(b[i]) * b[i];
    }
    return na > 0 && nb > 0 ? float(dot / std::sqrt(na * nb)) : 0.0f;
}

} // namespace voice
