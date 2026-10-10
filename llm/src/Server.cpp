// HTTP 服务（见 Server.h）
#include "Server.h"

#include "RuntimeLog.h"

#include "civetweb.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#ifndef LLM_VERSION
#define LLM_VERSION "0.0.0"
#endif

namespace llm {
namespace {

constexpr size_t kMaxBody = size_t(512) << 20;   // 带图的请求（base64）可能很大

int64_t nowMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

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

void replyError(mg_connection* conn, int status, const std::string& message, const char* type = "invalid_request_error")
{
    replyJson(conn, status, json { { "error", { { "message", message }, { "type", type }, { "code", status } } } });
}

/// 返回码 → HTTP 状态
int httpStatus(int code)
{
    switch (code) {
    case FLR_OK: return 200;
    case FLR_CANCELLED: return 499;
    case FLR_CONTEXT_OVERFLOW:
    case FLR_INVALID_REQUEST:
    case FLR_UNSUPPORTED_MEDIA: return 400;
    case FLR_KV_FULL: return 429;   // 同时的请求太多了：Friday 的请求闸门见到 429 会减并发、等一会儿重试
    case FLR_NOT_READY: return 503;
    default: return 500;
    }
}

const char* errorType(int code)
{
    switch (code) {
    case FLR_CONTEXT_OVERFLOW: return "exceed_context_size_error";
    case FLR_KV_FULL: return "rate_limit_error";
    case FLR_NOT_READY: return "unavailable_error";
    case FLR_INVALID_REQUEST:
    case FLR_UNSUPPORTED_MEDIA: return "invalid_request_error";
    default: return "server_error";
    }
}

/// 结果里的 error（runtime 给的 {"error":{"message",…}}）补上 type / code
json errorBody(const json& result, int code)
{
    json error = result.contains("error") && result["error"].is_object() ? result["error"] : json { { "message", "出错了" } };
    if (!error.contains("type"))
        error["type"] = errorType(code);
    error["code"] = httpStatus(code);
    return json { { "error", error } };
}

bool readBody(mg_connection* conn, std::string& body, std::string* error)
{
    const mg_request_info* info = mg_get_request_info(conn);
    if (info->content_length > (long long) kMaxBody) {
        if (error)
            *error = "请求体太大";
        return false;
    }
    if (info->content_length > 0)
        body.reserve(size_t(info->content_length));
    char buffer[65536];
    while (true) {
        const int n = mg_read(conn, buffer, sizeof(buffer));
        if (n <= 0)
            break;
        body.append(buffer, size_t(n));
        if (body.size() > kMaxBody) {
            if (error)
                *error = "请求体太大";
            return false;
        }
    }
    return true;
}

int findFreePort(const std::string& host)
{
    int port = 0;
    const auto fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_port = 0;
    inet_pton(AF_INET, host == "0.0.0.0" ? "0.0.0.0" : host.c_str(), &address.sin_addr);
    if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) {
        socklen_t length = sizeof(address);
        if (getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) == 0)
            port = ntohs(address.sin_port);
    }
#ifdef _WIN32
    closesocket(fd);
#else
    ::close(fd);
#endif
    return port;
}

/// 一次流式回复：头在第一块（或等了 3 秒还没有块）时才发，之前出的错照样能回 HTTP 状态
struct StreamWriter {
    mg_connection* conn = nullptr;
    bool streaming = false;
    bool headersSent = false;
    bool broken = false;
    int64_t lastWrite = 0;
    int64_t started = 0;

    bool sendHeaders()
    {
        if (headersSent)
            return !broken;
        headersSent = true;
        mg_printf(conn,
                  "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n"
                  "Connection: close\r\nX-Accel-Buffering: no\r\n\r\n");
        lastWrite = nowMs();
        return true;
    }
    bool write(const std::string& text)
    {
        if (broken)
            return false;
        if (!sendHeaders())
            return false;
        if (mg_send_chunk(conn, text.data(), unsigned(text.size())) < 0) {
            broken = true;
            return false;
        }
        lastWrite = nowMs();
        return true;
    }

    static bool onChunk(const char* chunk, void* user)
    {
        auto* self = static_cast<StreamWriter*>(user);
        if (!self->streaming)
            return true;
        return self->write(std::string("data: ") + chunk + "\n\n");
    }
    /// 等待中（预填长提示、排队）：头还没发、等了 3 秒就先发头；之后每 3 秒发一行 SSE 注释保活——写不出去说明连接断了，停下这个请求
    static bool shouldStop(void* user)
    {
        auto* self = static_cast<StreamWriter*>(user);
        if (!self->streaming || self->broken)
            return self->broken;
        const int64_t now = nowMs();
        if (!self->headersSent) {
            if (now - self->started < 3000)
                return false;
            self->sendHeaders();
            return false;
        }
        if (now - self->lastWrite >= 3000)
            return !self->write(": keep-alive\n\n");
        return false;
    }
};

// ── civetweb 回调转发 ──
int healthHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleHealth(conn); }
int infoHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleInfo(conn); }
int modelsHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleModels(conn); }
int chatHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleChat(conn); }
int loadHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleLoad(conn); }
int releaseHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleRelease(conn); }

int logMessage(const mg_connection*, const char* message)
{
    flr::logf(FLR_LOG_DEBUG, "civetweb: %s", message);
    return 1;
}

} // namespace

json devicesJson()
{
    json devices = json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        const char* type = "cpu";
        switch (ggml_backend_dev_type(device)) {
        case GGML_BACKEND_DEVICE_TYPE_GPU: type = "gpu"; break;
        case GGML_BACKEND_DEVICE_TYPE_IGPU: type = "igpu"; break;
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: type = "accel"; break;
        default: break;
        }
        devices.push_back(json {
            { "name", ggml_backend_dev_name(device) },
            { "description", ggml_backend_dev_description(device) },
            { "type", type },
            { "total", uint64_t(total) },
            { "free", uint64_t(free) },
        });
    }
    return devices;
}

Server::Server(ModelHost& host, const ServerOptions& options)
    : m_host(host)
    , m_options(options)
{
}

Server::~Server()
{
    stop();
}

std::string Server::url() const
{
    const std::string host = m_options.host == "0.0.0.0" ? "127.0.0.1" : m_options.host;
    return "http://" + host + ":" + std::to_string(m_port);
}

bool Server::start(std::string* error)
{
    mg_init_library(0);
    for (int attempt = 0; attempt < 5 && !m_ctx; ++attempt) {
        m_port = m_options.port > 0 ? m_options.port : findFreePort(m_options.host);
        if (m_port <= 0)
            continue;
        const std::string ports = m_options.host + ":" + std::to_string(m_port);
        const std::string threads = std::to_string(m_options.threads);
        // 生成可能很久（长回答、排队）：请求超时设成一小时
        const char* options[] = { "listening_ports", ports.c_str(), "num_threads", threads.c_str(), "request_timeout_ms", "3600000",
                                  "enable_keep_alive", "yes", nullptr };
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
    mg_set_request_handler(m_ctx, "/v1/models$", modelsHandler, this);
    mg_set_request_handler(m_ctx, "/v1/models/load", loadHandler, this);
    mg_set_request_handler(m_ctx, "/v1/chat/completions", chatHandler, this);
    mg_set_request_handler(m_ctx, "/v1/memory/release", releaseHandler, this);
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
    const char* header = mg_get_header(conn, "Authorization");
    return header && std::string(header) == "Bearer " + m_options.token;
}

int Server::handleHealth(mg_connection* conn)
{
    replyJson(conn, 200, json { { "status", "ok" } });
    return 200;
}

int Server::handleInfo(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
    replyJson(conn, 200,
              json {
                  { "id", "llm" },
                  { "version", LLM_VERSION },
                  { "llama.cpp", LLM_LLAMA_TAG },
                  { "capabilities", json::array({ "chat" }) },
                  { "devices", devicesJson() },
                  { "model", m_host.info() },
                  { "in_flight", m_inFlight.load() },
                  { "system_info", llama_print_system_info() },
              });
    return 200;
}

int Server::handleModels(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
    replyJson(conn, 200, m_host.listModels());
    return 200;
}

int Server::handleLoad(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
    std::string text;
    std::string error;
    if (!readBody(conn, text, &error)) {
        replyError(conn, 413, error);
        return 413;
    }
    json body;
    try {
        body = json::parse(text);
    } catch (const std::exception& exception) {
        replyError(conn, 400, std::string("请求不是合法的 JSON：") + exception.what());
        return 400;
    }
    ModelChoice choice;
    if (!m_host.resolve(body, choice, &error)) {
        replyError(conn, 404, error, "model_not_found");
        return 404;
    }
    int code = FLR_OK;
    std::shared_ptr<flr::Runtime> runtime = m_host.acquire(choice, &code, &error);
    if (!runtime) {
        replyError(conn, httpStatus(code), error, errorType(code));
        return httpStatus(code);
    }
    replyJson(conn, 200, json { { "model", runtime->info() } });
    return 200;
}

int Server::handleRelease(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
    std::string ignored;
    readBody(conn, ignored, nullptr);
    const std::string id = m_host.unload();
    replyJson(conn, 200, json { { "unloaded", id.empty() ? json(nullptr) : json(id) } });
    return 200;
}

int Server::handleChat(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
    if (std::strcmp(mg_get_request_info(conn)->request_method, "POST") != 0) {
        replyError(conn, 405, "只支持 POST");
        return 405;
    }
    std::string text;
    std::string error;
    if (!readBody(conn, text, &error)) {
        replyError(conn, 413, error);
        return 413;
    }
    json body;
    try {
        body = json::parse(text);
    } catch (const std::exception& exception) {
        replyError(conn, 400, std::string("请求不是合法的 JSON：") + exception.what());
        return 400;
    }
    if (!body.is_object()) {
        replyError(conn, 400, "请求体不是 JSON 对象");
        return 400;
    }
    struct InFlight {
        std::atomic<int>& count;
        explicit InFlight(std::atomic<int>& value) : count(value) { ++count; }
        ~InFlight() { --count; }
    } inFlight(m_inFlight);
    if (m_inFlight.load() > m_options.maxQueued) {
        replyError(conn, 429, "本机模型这会儿排队的请求太多了", "rate_limit_error");
        return 429;
    }

    ModelChoice choice;
    if (!m_host.resolve(body, choice, &error)) {
        replyError(conn, 404, error, "model_not_found");
        return 404;
    }
    int code = FLR_OK;
    std::shared_ptr<flr::Runtime> runtime = m_host.acquire(choice, &code, &error);
    if (!runtime) {
        replyError(conn, httpStatus(code), error, errorType(code));
        return httpStatus(code);
    }
    flr::CallGuard guard(*runtime);
    if (!guard) {
        replyError(conn, 503, "正在换模型 / 卸载，稍后再试", "unavailable_error");
        return 503;
    }

    StreamWriter writer;
    writer.conn = conn;
    writer.streaming = body.contains("stream") && body["stream"].is_boolean() && body["stream"].get<bool>();
    writer.started = nowMs();
    json result;
    const int status = runtime->chat(std::move(body), &StreamWriter::onChunk, &StreamWriter::shouldStop, &writer, result);
    if (!writer.streaming) {
        if (status == FLR_OK)
            replyJson(conn, 200, result);
        else if (status != FLR_CANCELLED)
            replyJson(conn, httpStatus(status), errorBody(result, status));
        return status == FLR_OK ? 200 : httpStatus(status);
    }
    if (status == FLR_OK) {
        // 最后一块之后：调度信息（x_friday.cache）单独一块，再 [DONE]
        if (result.contains("x_friday"))
            writer.write("data: " + json { { "object", "chat.completion.chunk" }, { "choices", json::array() }, { "x_friday", result["x_friday"] } }.dump() + "\n\n");
        writer.write("data: [DONE]\n\n");
    } else if (!writer.headersSent) {
        if (status != FLR_CANCELLED)
            replyJson(conn, httpStatus(status), errorBody(result, status));
        return httpStatus(status);
    } else if (status != FLR_CANCELLED) {
        writer.write("data: " + errorBody(result, status).dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n");
        writer.write("data: [DONE]\n\n");
    }
    if (!writer.broken)
        mg_send_chunk(conn, "", 0);
    return 200;
}

} // namespace llm
