// 声纹（说话人特征）：CAM++（3D-Speaker，CosyVoice 自带的 campplus.onnx）在 ggml 上，一段 16 kHz 录音 → 192 维向量。
// 两段声纹的余弦相似度高 = 同一个人。语音模式靠它只把登记过的人说的话交给助手（旁边的人说话不算、不打断）。
//
// 权重直接从 campplus.onnx 读（OnnxWeights），按图里 Conv / BatchNormalization 节点的先后对上网络结构
// （3D-Speaker 的 CAMPPlus：FCM 2D 卷积头 → TDNN → 三个 CAM 稠密块 12/24/16 层 + 过渡层 → 统计池化 → 全连接）。
// 数值和 onnxruntime 跑这个文件一致（ONNX 的分段平均最后一段不足 100 帧也除以 100，照它）。
// 特征：80 维 Kaldi fbank（povey 窗），整句减去均值（CosyVoice 的 frontend 同样做）。
#pragma once

#include "Backend.h"
#include "Fbank.h"

#include "ggml.h"

#include <mutex>
#include <string>
#include <vector>

namespace voice {

struct SpeakerEmbedding {
    bool ok = false;
    std::vector<float> vector;   // 192 维，已归一化成单位长度（两段的点积就是余弦相似度）
    double seconds = 0;          // 这次推理用时
    double audioSeconds = 0;     // 用了多长的录音
    std::string error;
};

class SpeakerEncoder {
public:
    explicit SpeakerEncoder(Backend& backend);
    ~SpeakerEncoder();
    SpeakerEncoder(const SpeakerEncoder&) = delete;
    SpeakerEncoder& operator=(const SpeakerEncoder&) = delete;

    bool load(const std::string& onnxPath, std::string* error);
    bool loaded() const { return m_weights != nullptr; }
    const std::string& path() const { return m_path; }

    /// pcm：16 kHz 单声道 [-1,1]。太短（不到 0.5 秒）不算；太长只用中间 20 秒。线程安全（内部串行）
    SpeakerEmbedding embed(const float* pcm, size_t count);
    /// 已经算好的特征（frames×80，已减均值）：测试用
    SpeakerEmbedding embedFeatures(const std::vector<float>& features, int frames);
    /// 这段录音的特征（frames×80，已减均值）：和 embed 用的一样
    std::vector<float> features(const float* pcm, size_t count, int* frames) const;

    static float cosine(const std::vector<float>& a, const std::vector<float>& b);

    static constexpr int kDim = 192;
    static constexpr double kMinSeconds = 0.5;
    static constexpr double kMaxSeconds = 20.0;

private:
    struct Conv {
        ggml_tensor* weight = nullptr;   // 1D：[Cin*K, Cout]；2D：[KW, KH, Cin, Cout]
        ggml_tensor* bias = nullptr;     // 1D：[Cout, 1]；2D：[1, 1, Cout, 1]；没有就是空
        int cin = 0, cout = 0, kernel = 1, stride = 1, pad = 0, dilation = 1;   // 1D
        int kh = 1, kw = 1, sh = 1, sw = 1, ph = 0, pw = 0;                    // 2D
        bool twoD = false;
    };
    struct Norm {
        ggml_tensor* scale = nullptr;   // [C, 1]
        ggml_tensor* shift = nullptr;
    };

    ggml_tensor* conv2d(ggml_context* ctx, const Conv& conv, ggml_tensor* x) const;
    /// x 是 [T, Cin]（时间在前）：展开成 [Cin*K, Tout] 再乘 → [Cout, Tout]
    ggml_tensor* conv1dTime(ggml_context* ctx, const Conv& conv, ggml_tensor* x) const;
    /// x 是 [Cin, T]：1×1 卷积 → [Cout, T]
    ggml_tensor* conv1x1(ggml_context* ctx, const Conv& conv, ggml_tensor* x) const;
    ggml_tensor* norm(ggml_context* ctx, const Norm& n, ggml_tensor* x) const;

    Backend& m_backend;
    Fbank m_fbank { Fbank::Window::Povey };
    std::string m_path;
    ggml_context* m_weightsCtx = nullptr;
    ggml_backend_buffer_t m_weights = nullptr;
    ggml_backend_sched_t m_sched = nullptr;
    std::vector<Conv> m_convs;   // 按图里的先后
    std::vector<Norm> m_norms;
    std::mutex m_mutex;
};

} // namespace voice
