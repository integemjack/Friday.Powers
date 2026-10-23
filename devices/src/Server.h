// HTTP + WebSocket 服务（civetweb，docs/devices-protocol.md）：
//   GET  /health        就绪
//   GET  /v1/info       Power 信息、桥连没连、客户端、开着的端口
//   GET  /v1/bridge     WebSocket：Friday 接进来（只认一条，见 Bridge.h）
//   POST /mcp           MCP（Streamable HTTP，只回 application/json）：initialize、tools/list、tools/call、ping
#pragma once

#include "Bridge.h"
#include "Ports.h"
#include "Tools.h"

#include <string>

struct mg_context;
struct mg_connection;

namespace devices {

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 0;
    std::string token;
    int threads = 24;
};

class Server {
public:
    explicit Server(const ServerOptions& options);
    ~Server();

    bool start(std::string* error);
    void stop();
    std::string url() const;

    int handleHealth(mg_connection* conn);
    int handleInfo(mg_connection* conn);
    int handleMcp(mg_connection* conn);
    bool authorized(const mg_connection* conn) const;

    Bridge& bridge() { return m_bridge; }

private:
    ServerOptions m_options;
    Bridge m_bridge;
    Ports m_ports;
    Tools m_tools;
    mg_context* m_ctx = nullptr;
    int m_port = 0;
};

} // namespace devices
