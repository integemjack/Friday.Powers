// HTTP 服务（civetweb，docs/diffusion-protocol.md）：
//   GET  /health                         就绪
//   GET  /v1/info                        Power 信息、设备、加载着的模型、队列
//   POST /v1/jobs                        提交生图 / 生视频任务（异步）→ 202 {id, status…}
//   GET  /v1/jobs/<id>                   任务状态、进度、结果列表
//   GET  /v1/jobs/<id>/outputs/<n>       第 n 个结果（PNG / WebM）
//   POST /v1/jobs/<id>/cancel            取消
//   POST /v1/images/generations          OpenAI 兼容的生图（同步，b64_json）
//   POST /v1/memory/release              释放模型、腾显存
#pragma once

#include "Generator.h"

#include <string>

struct mg_context;
struct mg_connection;

namespace diffusion {

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 0;
    std::string token;
    int threads = 16;
};

class Server {
public:
    Server(Generator& generator, const ServerOptions& options);
    ~Server();

    bool start(std::string* error);
    void stop();
    std::string url() const;

    int handleHealth(mg_connection* conn);
    int handleInfo(mg_connection* conn);
    int handleJobs(mg_connection* conn);
    int handleImages(mg_connection* conn);
    int handleRelease(mg_connection* conn);
    bool authorized(const mg_connection* conn) const;

private:
    Generator& m_generator;
    ServerOptions m_options;
    mg_context* m_ctx = nullptr;
    int m_port = 0;
};

} // namespace diffusion
