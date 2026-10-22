// HTTP + WebSocket 服务（civetweb）：
//   GET  /health                    就绪
//   GET  /v1/info                   Power 信息、设备、模型
//   GET  /v1/models                 OpenAI 兼容的模型列表
//   POST /v1/audio/transcriptions   OpenAI 兼容的一次性听录音（multipart：file、language、response_format）
//   GET  /v1/realtime/transcribe    WebSocket 实时听写（docs/voice-protocol.md；v0.3 有 Whisper 时定稿用它，start 帧可带 prompt 热词）
//   POST /v1/audio/speech           OpenAI 兼容的合成（JSON：input、voice、speed、response_format wav|pcm、instructions），边合成边流式返回
//   GET  /v1/realtime/speak         WebSocket 边写边读（docs/voice-protocol.md）
//   POST /v1/audio/speaker          一段录音的声纹（multipart：file）→ {embedding: 192 维单位向量, duration}；登记声纹用
//   v0.5 用户音色（存 <models>/user-voices/<名字>.gguf，热生效；docs/voice-protocol.md「用户音色」）：
//   GET  /v1/voices                 {voices, user_voices, default, clone: 能不能克隆}
//   POST /v1/voice/clone            克隆（multipart：file 录音、name、可选 text 底稿（空 = 听写）、language、replace）→ {voice, transcript, seconds, took}
//   POST /v1/voice/import           导入别处克隆好的音色（multipart：file .gguf、name、replace）→ {voice, took}
//   DELETE /v1/voice/<名字>          删掉一个用户音色（内置的 403）
#pragma once

#include "Backend.h"
#include "Fbank.h"
#include "FsmnVad.h"
#include "InferenceQueue.h"
#include "SenseVoice.h"

#include <memory>
#include <string>

struct mg_context;
struct mg_connection;

namespace voice {

class Synthesizer;
class SpeakerEncoder;
class Whisper;
class VoiceCloner;

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 0;          // 0 = 自己找一个空闲端口
    std::string token;     // 非空时所有请求要带 Authorization: Bearer <token>（WebSocket 也可以 ?token=）
    int threads = 16;
    int vadThreads = 2;
};

class Server {
public:
    /// synthesizer、voiceprint、whisper、cloner 可以是空（没装合成 / 声纹 / Whisper / 克隆前端模型时没有这些功能）
    Server(Backend& backend, SenseVoice& recognizer, FsmnVad& vad, Synthesizer* synthesizer, const ServerOptions& options,
           SpeakerEncoder* voiceprint = nullptr, Whisper* whisper = nullptr, VoiceCloner* cloner = nullptr,
           const std::string& userVoicesDir = {});
    ~Server();

    bool start(std::string* error);
    void stop();
    int port() const { return m_port; }
    std::string url() const;

    // civetweb 回调（public 以便静态函数转发）
    int handleHealth(mg_connection* conn);
    int handleInfo(mg_connection* conn);
    int handleModels(mg_connection* conn);
    int handleTranscriptions(mg_connection* conn);
    int handleSpeech(mg_connection* conn);
    int handleSpeaker(mg_connection* conn);
    int handleVoices(mg_connection* conn);
    /// rest：/v1/voice/ 后面那段（clone、import，或 DELETE 的音色名，没解码）
    int handleVoice(mg_connection* conn, const std::string& rest);
    bool cloneAvailable() const;
    bool authorized(const mg_connection* conn) const;
    void onSocketReady(mg_connection* conn);
    void onSpeakSocketReady(mg_connection* conn);
    int onSocketData(mg_connection* conn, int bits, char* data, size_t size);
    void onSocketClose(const mg_connection* conn);

private:
    Backend& m_backend;
    SenseVoice& m_recognizer;
    FsmnVad& m_vad;
    Synthesizer* m_synth;
    SpeakerEncoder* m_voiceprint;
    Whisper* m_whisper;
    VoiceCloner* m_cloner;
    std::string m_userVoicesDir;   // 用户音色的目录（空 = 没有用户音色：不能导入、克隆）
    Fbank m_fbank;
    InferenceQueue m_queue;
    ServerOptions m_options;
    mg_context* m_ctx = nullptr;
    int m_port = 0;
};

} // namespace voice
