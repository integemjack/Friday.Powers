// HTTP 服务（见 Server.h）
#include "Server.h"

#include "Log.h"

#include "civetweb.h"
#include "stable-diffusion.h"

#include <cstring>
#include <ctime>
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

#ifndef DIFFUSION_VERSION
#define DIFFUSION_VERSION "0.0.0"
#endif

namespace diffusion {
namespace {

constexpr size_t kMaxBody = size_t(256) << 20;

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

bool readBody(mg_connection* conn, std::string& body)
{
    const mg_request_info* info = mg_get_request_info(conn);
    if (info->content_length > (long long) kMaxBody)
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

std::string base64(const std::vector<uint8_t>& data)
{
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < data.size(); i += 3) {
        const uint32_t v = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | data[i + 2];
        out += alphabet[(v >> 18) & 63];
        out += alphabet[(v >> 12) & 63];
        out += alphabet[(v >> 6) & 63];
        out += alphabet[v & 63];
    }
    if (i < data.size()) {
        uint32_t v = uint32_t(data[i]) << 16;
        if (i + 1 < data.size())
            v |= uint32_t(data[i + 1]) << 8;
        out += alphabet[(v >> 18) & 63];
        out += alphabet[(v >> 12) & 63];
        out += i + 1 < data.size() ? alphabet[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
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

std::vector<std::string> splitPath(const std::string& uri)
{
    std::vector<std::string> parts;
    size_t start = 0;
    while (start < uri.size()) {
        const size_t slash = uri.find('/', start);
        const std::string part = uri.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        if (!part.empty())
            parts.push_back(part);
        if (slash == std::string::npos)
            break;
        start = slash + 1;
    }
    return parts;
}

int healthHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleHealth(conn); }
int infoHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleInfo(conn); }
int jobsHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleJobs(conn); }
int imagesHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleImages(conn); }
int releaseHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleRelease(conn); }

int logMessage(const mg_connection*, const char* message)
{
    DLOG_DEBUG("civetweb: %s", message);
    return 1;
}

} // namespace

Server::Server(Generator& generator, const ServerOptions& options)
    : m_generator(generator)
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
        const char* options[] = { "listening_ports", ports.c_str(), "num_threads", threads.c_str(), "request_timeout_ms", "3600000", nullptr };
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
    mg_set_request_handler(m_ctx, "/v1/jobs", jobsHandler, this);
    mg_set_request_handler(m_ctx, "/v1/images/generations", imagesHandler, this);
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
                  { "id", "diffusion" },
                  { "version", DIFFUSION_VERSION },
                  { "stable-diffusion.cpp", DIFFUSION_SD_TAG },
                  { "capabilities", json::array({ "imageGeneration", "videoGeneration" }) },
                  { "generator", m_generator.info() },
                  { "system_info", sd_get_system_info() },
              });
    return 200;
}

int Server::handleJobs(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
    const mg_request_info* info = mg_get_request_info(conn);
    const std::vector<std::string> parts = splitPath(info->local_uri ? info->local_uri : "");
    const bool post = std::strcmp(info->request_method, "POST") == 0;
    // /v1/jobs
    if (parts.size() == 2) {
        if (!post) {
            replyError(conn, 405, "只支持 POST");
            return 405;
        }
        std::string text;
        if (!readBody(conn, text)) {
            replyError(conn, 413, "请求体太大");
            return 413;
        }
        json body;
        try {
            body = json::parse(text);
        } catch (const std::exception& exception) {
            replyError(conn, 400, std::string("请求不是合法的 JSON：") + exception.what());
            return 400;
        }
        std::string error;
        std::shared_ptr<Job> job = m_generator.submit(body, &error);
        if (!job) {
            replyError(conn, 400, error);
            return 400;
        }
        replyJson(conn, 202, m_generator.describe(*job));
        return 202;
    }
    if (parts.size() < 3) {
        replyError(conn, 404, "没有这个接口", "not_found");
        return 404;
    }
    std::shared_ptr<Job> job = m_generator.find(parts[2]);
    if (!job) {
        replyError(conn, 404, "没有这个任务（只留最近 32 个）", "not_found");
        return 404;
    }
    // /v1/jobs/<id>
    if (parts.size() == 3) {
        replyJson(conn, 200, m_generator.describe(*job));
        return 200;
    }
    // /v1/jobs/<id>/cancel
    if (parts.size() == 4 && parts[3] == "cancel") {
        m_generator.cancel(job->id);
        replyJson(conn, 200, m_generator.describe(*job));
        return 200;
    }
    // /v1/jobs/<id>/outputs/<n>
    if (parts.size() == 5 && parts[3] == "outputs") {
        const size_t index = size_t(std::atoi(parts[4].c_str()));
        Output output;
        if (!m_generator.output(*job, index, output)) {
            replyError(conn, 404, "没有这个结果", "not_found");
            return 404;
        }
        mg_printf(conn, "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", output.mime.c_str(),
                  output.data.size());
        mg_write(conn, output.data.data(), output.data.size());
        return 200;
    }
    replyError(conn, 404, "没有这个接口", "not_found");
    return 404;
}

int Server::handleImages(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
    std::string text;
    if (!readBody(conn, text)) {
        replyError(conn, 413, "请求体太大");
        return 413;
    }
    json body;
    try {
        body = json::parse(text);
    } catch (const std::exception& exception) {
        replyError(conn, 400, std::string("请求不是合法的 JSON：") + exception.what());
        return 400;
    }
    body["kind"] = "image";
    std::string error;
    std::shared_ptr<Job> job = m_generator.submit(body, &error);
    if (!job) {
        replyError(conn, 400, error);
        return 400;
    }
    m_generator.wait(job, -1);
    if (job->status != "done") {
        replyError(conn, job->status == "cancelled" ? 499 : 500, job->error.empty() ? "生成失败" : job->error, "server_error");
        return 500;
    }
    json data = json::array();
    Output output;
    for (size_t i = 0; m_generator.output(*job, i, output); ++i)
        data.push_back(json { { "b64_json", base64(output.data) } });
    replyJson(conn, 200, json { { "created", std::time(nullptr) }, { "data", data }, { "x_friday", m_generator.describe(*job) } });
    return 200;
}

int Server::handleRelease(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
    std::string ignored;
    readBody(conn, ignored);
    const bool released = m_generator.release();
    replyJson(conn, 200, json { { "released", released } });
    return 200;
}

} // namespace diffusion
