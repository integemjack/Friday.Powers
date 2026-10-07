// FSMN-VAD（FunASR）：每 10 毫秒一帧「是不是在说话」。
// 网络照 FunASR llama.cpp runtime 的 funasr_vad.h（MIT），在 CPU 上跑（模型 1.7 MB，算一次不到 1 毫秒）。
// 原版只能整段算；这里改成流式：每来一段音频只算新帧——FSMN 的记忆块只看左边（4 层 × 19 帧），
// 所以拿「新帧 + 左边 80 帧」算一遍、只取新帧的结果，和整段算出来的一样。
#pragma once

#include "Fbank.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <map>
#include <string>
#include <vector>

namespace voice {

class FsmnVad {
public:
    FsmnVad() = default;
    ~FsmnVad();
    FsmnVad(const FsmnVad&) = delete;
    FsmnVad& operator=(const FsmnVad&) = delete;

    bool load(const std::string& path, std::string* error);
    bool loaded() const { return m_ctx != nullptr; }

    /// LFR 特征（frames × inputDim，已做 CMVN）→ 每帧说话的概率（1 − 静音分）
    std::vector<float> speechProbability(const float* features, int frames, ggml_backend_t cpu) const;

    int inputDim() const { return m_inputDim; }
    int lfrM() const { return m_lfrM; }
    int lfrN() const { return m_lfrN; }
    /// 算一帧需要的左侧上下文（帧）
    int contextFrames() const { return m_layers * (m_lorder - 1) + 4; }
    const float* cmvnShift() const { return m_shift; }
    const float* cmvnScale() const { return m_scale; }

private:
    ggml_tensor* tensor(const std::string& name) const;
    ggml_context* m_ctx = nullptr;
    std::map<std::string, ggml_tensor*> m_tensors;
    int m_inputDim = 400, m_projDim = 128, m_layers = 4, m_lorder = 20, m_outputDim = 248, m_lfrM = 5, m_lfrN = 1;
    const float* m_shift = nullptr;
    const float* m_scale = nullptr;
};

/// 一路音频的流式 VAD：喂 16 kHz 采样，吐出新算出来的每帧概率（帧 i 覆盖采样 [i·160, i·160+400)）
class VadStream {
public:
    VadStream(const FsmnVad& model, const Fbank& fbank, int threads);
    ~VadStream();
    VadStream(const VadStream&) = delete;
    VadStream& operator=(const VadStream&) = delete;

    std::vector<float> feed(const float* pcm, size_t count);
    /// 流结束：没有后面的帧了，把等着看后两帧的尾巴也算完
    std::vector<float> finish();
    /// 已经出过概率的帧数
    int framesDone() const { return m_done; }

private:
    std::vector<float> compute(bool final);

    const FsmnVad& m_model;
    const Fbank& m_fbank;
    ggml_backend_t m_cpu = nullptr;
    std::vector<float> m_pending;    // 还没算成帧的采样（含帧间重叠）
    std::vector<float> m_frames;     // fbank 帧（80 维），从全局帧号 m_base 开始
    int m_base = 0;
    int m_total = 0;                 // 已算出的 fbank 帧总数
    int m_done = 0;                  // 已出过概率的 LFR 帧数（m5n1：一帧对一帧）
};

} // namespace voice
