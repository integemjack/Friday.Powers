// 见 Server.h
#include "Server.h"

#include "Log.h"
#include "Socket.h"

#include "civetweb.h"

#include <cstring>

#ifndef DEVICES_VERSION
#define DEVICES_VERSION "0.0.0"
#endif

namespace devices {
namespace {

constexpr size_t kMaxBody = size_t(64) << 20;
/// 我们说的 MCP 版本（客户端要别的就照它的回：只用到 tools，几版都一样）
constexpr char kMcpVersion[] = "2025-06-18";

void reply(mg_connection* conn, int status, const std::string& body, const char* mime = "application/json")
{
    mg_printf(conn, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", status,
              mg_get_response_code_text(conn, status), mime, body.size());
    mg_write(conn, body.data(), body.size());
}

void replyJson(mg_connection* conn, int status, const json& value)
{
    reply(conn, status, value.dump(-1, ' ', false, json::error_handler_t::replace));
}

bool readBody(mg_connection* conn, std::string& body)
{
    const mg_request_info* info = mg_get_request_info(conn);
    if (info->content_length > (long long)kMaxBody)
        return false;
    char buffer[65536];
    while (true) {
        const int n = mg_read(conn, buffer, sizeof(buffer));
        if (n <= 0)
            break;
        body.append(buffer, size_t(n));
        if (body.size() > kMaxBody)
            return false;
    }
    return true;
}

int findFreePort()
{
    std::string error;
    int port = 0;
    const net::Handle probe = net::listenLocal(0, &port, &error);
    if (probe == net::kInvalid)
        return 0;
    net::closeHandle(probe);
    return port;
}

int healthHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleHealth(conn); }
int infoHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleInfo(conn); }
int mcpHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleMcp(conn); }

int bridgeConnect(const mg_connection* conn, void* data) { return static_cast<Server*>(data)->authorized(conn) ? 0 : 1; }
void bridgeReady(mg_connection* conn, void* data) { static_cast<Server*>(data)->bridge().attach(conn); }
int bridgeData(mg_connection* conn, int bits, char* payload, size_t size, void* data)
{
    static_cast<Server*>(data)->bridge().onFrame(conn, bits, payload, size);
    return 1;
}
void bridgeClose(const mg_connection* conn, void* data) { static_cast<Server*>(data)->bridge().detach(conn); }

int logMessage(const mg_connection*, const char* message)
{
    DLOG_DEBUG("civetweb: %s", message);
    return 1;
}

json rpcResult(const json& id, json result)
{
    return json { { "jsonrpc", "2.0" }, { "id", id }, { "result", std::move(result) } };
}

json rpcError(const json& id, int code, const std::string& message)
{
    return json { { "jsonrpc", "2.0" }, { "id", id }, { "error", { { "code", code }, { "message", message } } } };
}

} // namespace

Server::Server(const ServerOptions& options)
    : m_options(options)
    , m_ports(m_bridge)
    , m_tools(m_bridge, m_ports)
{
}

Server::~Server()
{
    stop();
}

std::string Server::url() const
{
    return "http://127.0.0.1:" + std::to_string(m_port);
}

bool Server::start(std::string* error)
{
    mg_init_library(0);
    net::initialize();
    for (int attempt = 0; attempt < 5 && !m_ctx; ++attempt) {
        m_port = m_options.port > 0 ? m_options.port : findFreePort();
        if (m_port <= 0)
            continue;
        const std::string ports = m_options.host + ":" + std::to_string(m_port);
        const std::string threads = std::to_string(m_options.threads);
        const char* options[] = { "listening_ports", ports.c_str(), "num_threads", threads.c_str(),
                                  "request_timeout_ms", "600000", "enable_websocket_ping_pong", "yes",
                                  "websocket_timeout_ms", "86400000", nullptr };
        mg_callbacks callbacks {};
        callbacks.log_message = logMessage;
        m_ctx = mg_start(&callbacks, this, options);
        if (m_options.port > 0)
            break;
    }
    if (!m_ctx) {
        if (error)
            *error = "监听 " + m_options.host + ":" + std::to_string(m_options.port) + " 失败（端口被占用？）";
        return false;
    }
    mg_set_request_handler(m_ctx, "/health", healthHandler, this);
    mg_set_request_handler(m_ctx, "/v1/info", infoHandler, this);
    mg_set_request_handler(m_ctx, "/mcp", mcpHandler, this);
    mg_set_websocket_handler(m_ctx, "/v1/bridge", bridgeConnect, bridgeReady, bridgeData, bridgeClose, this);
    return true;
}

void Server::stop()
{
    if (m_ctx) {
        mg_stop(m_ctx);
        m_ctx = nullptr;
    }
}

bool Server::authorized(const mg_connection* conn) const
{
    if (m_options.token.empty())
        return true;
    if (const char* header = mg_get_header(conn, "Authorization")) {
        if (std::string(header) == "Bearer " + m_options.token)
            return true;
    }
    const mg_request_info* info = mg_get_request_info(conn);
    if (info && info->query_string) {
        char value[256] = {};
        if (mg_get_var(info->query_string, std::strlen(info->query_string), "token", value, sizeof(value)) > 0 && m_options.token == value)
            return true;
    }
    return false;
}

int Server::handleHealth(mg_connection* conn)
{
    replyJson(conn, 200, json { { "status", "ok" } });
    return 200;
}

int Server::handleInfo(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyJson(conn, 401, json { { "error", { { "message", "unauthorized" } } } });
        return 401;
    }
    json clients = json::array();
    for (const ClientInfo& client : m_bridge.clients())
        clients.push_back(json { { "id", client.id }, { "name", client.name }, { "kinds", client.kinds } });
    replyJson(conn, 200,
              json { { "id", "devices" },
                     { "version", DEVICES_VERSION },
                     { "capabilities", json::array({ "clientDevices" }) },
                     { "bridge", m_bridge.connected() },
                     { "clients", clients },
                     { "ports", m_ports.list() } });
    return 200;
}

int Server::handleMcp(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyJson(conn, 401, json { { "error", { { "message", "unauthorized" } } } });
        return 401;
    }
    const mg_request_info* info = mg_get_request_info(conn);
    if (std::strcmp(info->request_method, "POST") != 0) {
        // 不开服务器推送的 SSE 流（GET）：规范允许回 405
        reply(conn, 405, "", "text/plain");
        return 405;
    }
    std::string body;
    if (!readBody(conn, body)) {
        replyJson(conn, 413, rpcError(nullptr, -32600, "请求太大"));
        return 413;
    }
    const json message = json::parse(body, nullptr, false);
    if (!message.is_object()) {
        replyJson(conn, 400, rpcError(nullptr, -32700, "不是 JSON 对象"));
        return 400;
    }
    const std::string method = message.value("method", "");
    // 通知（没有 id）：收下就行
    if (!message.contains("id")) {
        reply(conn, 202, "", "text/plain");
        return 202;
    }
    const json id = message["id"];
    const json params = message.value("params", json::object());
    if (method == "initialize") {
        const std::string version = params.value("protocolVersion", std::string(kMcpVersion));
        replyJson(conn, 200,
                  rpcResult(id, json { { "protocolVersion", version.empty() ? kMcpVersion : version },
                                       { "capabilities", { { "tools", json::object() } } },
                                       { "serverInfo", { { "name", "friday-devices" }, { "version", DEVICES_VERSION } } },
                                       { "instructions",
                                         "Friday Work 桌面客户端那台电脑上的硬件：串口开发板（ESP / Arduino / STM32）、摄像头、安卓 / iOS 手机、"
                                         "局域网设备。先用 devices 看有什么。动硬件时客户端那头的人会被问一次。" } }));
        return 200;
    }
    if (method == "ping") {
        replyJson(conn, 200, rpcResult(id, json::object()));
        return 200;
    }
    if (method == "tools/list") {
        replyJson(conn, 200, rpcResult(id, json { { "tools", m_tools.list() } }));
        return 200;
    }
    if (method == "tools/call") {
        replyJson(conn, 200, rpcResult(id, m_tools.call(params.value("name", ""), params.value("arguments", json::object()))));
        return 200;
    }
    replyJson(conn, 200, rpcError(id, -32601, "Method not found: " + method));
    return 200;
}

} // namespace devices
