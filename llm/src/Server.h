// HTTP 服务（civetweb，docs/llm-protocol.md）：
//   GET  /health                  就绪
//   GET  /v1/info                 Power 信息、设备、加载着的模型、调度与基数树的统计
//   GET  /v1/models               OpenAI 兼容的模型列表（模型库目录里有的 GGUF）
//   POST /v1/chat/completions     OpenAI 兼容的对话补全（stream 时 SSE），x_friday 见协议文档
//   POST /v1/models/load          先把模型加载好（用户在 Friday 里选了它时预热）：{model, x_friday}
//   POST /v1/memory/release       卸载模型、腾显存（Friday 生视频前调）
#pragma once

#include "ModelHost.h"

#include <atomic>
#include <string>

struct mg_context;
struct mg_connection;

namespace llm {

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 0;          // 0 = 自己找一个空闲端口
    std::string token;     // 非空时所有请求要带 Authorization: Bearer <token>
    int threads = 48;
    /// 排队的请求超过这么多就回 429
    int maxQueued = 64;
};

class Server {
public:
    Server(ModelHost& host, const ServerOptions& options);
    ~Server();

    bool start(std::string* error);
    void stop();
    int port() const { return m_port; }
    std::string url() const;

    // civetweb 回调（public 以便静态函数转发）
    int handleHealth(mg_connection* conn);
    int handleInfo(mg_connection* conn);
    int handleModels(mg_connection* conn);
    int handleChat(mg_connection* conn);
    int handleLoad(mg_connection* conn);
    int handleRelease(mg_connection* conn);
    int handleReserve(mg_connection* conn);
    bool authorized(const mg_connection* conn) const;

private:
    ModelHost& m_host;
    ServerOptions m_options;
    mg_context* m_ctx = nullptr;
    int m_port = 0;
    std::atomic<int> m_inFlight { 0 };
};

/// 设备信息（/v1/info 与 info 子命令）
json devicesJson();

} // namespace llm
