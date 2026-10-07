// HTTP + WebSocket 服务（civetweb）：
//   GET  /health                    就绪
//   GET  /v1/info                   Power 信息、设备、模型
//   GET  /v1/models                 OpenAI 兼容的模型列表
//   POST /v1/audio/transcriptions   OpenAI 兼容的一次性听录音（multipart：file、language、response_format）
//   GET  /v1/realtime/transcribe    WebSocket 实时听写（docs/voice-protocol.md）
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

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 0;          // 0 = 自己找一个空闲端口
    std::string token;     // 非空时所有请求要带 Authorization: Bearer <token>（WebSocket 也可以 ?token=）
    int threads = 16;
    int vadThreads = 2;
};

class Server {
public:
    Server(Backend& backend, SenseVoice& recognizer, FsmnVad& vad, const ServerOptions& options);
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
    bool authorized(const mg_connection* conn) const;
    void onSocketReady(mg_connection* conn);
    int onSocketData(mg_connection* conn, int bits, char* data, size_t size);
    void onSocketClose(const mg_connection* conn);

private:
    Backend& m_backend;
    SenseVoice& m_recognizer;
    FsmnVad& m_vad;
    Fbank m_fbank;
    InferenceQueue m_queue;
    ServerOptions m_options;
    mg_context* m_ctx = nullptr;
    int m_port = 0;
};

} // namespace voice
