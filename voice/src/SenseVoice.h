// SenseVoiceSmall（SAN-M 编码器 + CTC）在 ggml 上：一段 16 kHz 录音 → 文字。
// 网络结构、权重名、特征照 FunASR llama.cpp runtime 的 funasr-sensevoice.cpp（MIT，见 third_party/funasr），
// 改成库：权重常驻 GPU、用调度器（GPU 不支持的算子落 CPU）、CTC 取最大值在 GPU 上做、每次请求可选语言和是否带标点。
// 非自回归：一次前向几十毫秒，所以实时字幕就是对正在说的这句整段重算。
#pragma once

#include "Backend.h"
#include "Fbank.h"

#include "ggml.h"
#include "gguf.h"

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace voice {

struct Transcript {
    bool ok = false;
    std::string text;       // 识别出的文字（去掉 <|…|> 标签）
    std::string language;   // zh / en / yue / ja / ko / nospeech
    std::string emotion;    // NEUTRAL / HAPPY / SAD / ANGRY …
    std::string event;      // Speech / BGM / Laughter / Applause …
    double seconds = 0;     // 这次推理用时
    std::string error;
};

class SenseVoice {
public:
    explicit SenseVoice(Backend& backend);
    ~SenseVoice();
    SenseVoice(const SenseVoice&) = delete;
    SenseVoice& operator=(const SenseVoice&) = delete;

    bool load(const std::string& path, std::string* error);

    /// pcm：16 kHz 单声道 [-1,1]。language：auto|zh|en|yue|ja|ko。itn：带标点、数字规整。
    /// 线程安全（内部串行：同一时间只有一个前向在 GPU 上）
    Transcript transcribe(const float* pcm, size_t count, const std::string& language, bool itn);

    bool loaded() const { return m_weights != nullptr; }
    const std::string& path() const { return m_path; }
    size_t weightsBytes() const { return m_weightsBytes; }

    static bool supportsLanguage(const std::string& language);

private:
    struct Config {
        int dModel = 512, heads = 4, blocks = 50, tpBlocks = 20, kernel = 11, vocab = 25055, blank = 0;
    };
    ggml_tensor* tensor(const std::string& name);
    ggml_tensor* linear(ggml_context* ctx, const std::string& weight, ggml_tensor* x);
    ggml_tensor* layerNorm(ggml_context* ctx, const std::string& prefix, ggml_tensor* x);
    ggml_tensor* attention(ggml_context* ctx, const std::string& prefix, ggml_tensor* x, int frames);
    ggml_tensor* layer(ggml_context* ctx, const std::string& prefix, ggml_tensor* x, int frames, bool residual);
    std::string detokenize(const std::vector<int>& ids, Transcript* out) const;

    Backend& m_backend;
    Fbank m_fbank;
    Config m_config;
    std::string m_path;
    ggml_context* m_weightsCtx = nullptr;
    ggml_backend_buffer_t m_weights = nullptr;
    size_t m_weightsBytes = 0;
    std::map<std::string, ggml_tensor*> m_tensors;
    std::vector<float> m_embed;               // embed.weight（16 × 560）放 CPU：查询 token 的向量
    std::vector<std::string> m_vocab;
    ggml_backend_sched_t m_sched = nullptr;
    std::mutex m_mutex;
};

} // namespace voice
