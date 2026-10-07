#include "Server.h"

#include "Log.h"
#include "Streamer.h"

#include "civetweb.h"
#include "json.hpp"

#include <chrono>
#include <cstring>
#include <map>
#include <mutex>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#ifndef VOICE_VERSION
#define VOICE_VERSION "0.0.0"
#endif

namespace voice {

using json = nlohmann::json;

namespace {

/// 一条 WebSocket：写要加锁（推理线程和连接线程都会写），关了之后不再写
struct SocketIO {
    std::mutex mutex;
    mg_connection* conn = nullptr;
    bool closed = false;

    void send(const std::string& text)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (closed || !conn)
            return;
        mg_lock_connection(conn);
        mg_websocket_write(conn, MG_WEBSOCKET_OPCODE_TEXT, text.data(), text.size());
        mg_unlock_connection(conn);
    }
};

struct Session {
    std::shared_ptr<SocketIO> io;
    std::shared_ptr<Streamer> streamer;
};

void reply(mg_connection* conn, int status, const std::string& body, const char* mime = "application/json")
{
    if (status == 200) {
        mg_send_http_ok(conn, mime, (long long)body.size());
    } else {
        mg_printf(conn,
                  "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
                  status, mg_get_response_code_text(conn, status), mime, body.size());
    }
    mg_write(conn, body.data(), body.size());
}

void replyError(mg_connection* conn, int status, const std::string& message)
{
    reply(conn, status, json { { "error", { { "message", message }, { "type", "invalid_request_error" } } } }.dump());
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

// ── civetweb 回调转发 ──
int healthHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleHealth(conn); }
int infoHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleInfo(conn); }
int modelsHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleModels(conn); }
int transcriptionsHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleTranscriptions(conn); }
int socketConnect(const mg_connection* conn, void* data) { return static_cast<Server*>(data)->authorized(conn) ? 0 : 1; }
void socketReady(mg_connection* conn, void* data) { static_cast<Server*>(data)->onSocketReady(conn); }
int socketData(mg_connection* conn, int bits, char* payload, size_t size, void* data)
{
    return static_cast<Server*>(data)->onSocketData(conn, bits, payload, size);
}
void socketClose(const mg_connection* conn, void* data) { static_cast<Server*>(data)->onSocketClose(conn); }

int logMessage(const mg_connection*, const char* message)
{
    VLOG_DEBUG("civetweb: %s", message);
    return 1;
}

} // namespace

Server::Server(Backend& backend, SenseVoice& recognizer, FsmnVad& vad, const ServerOptions& options)
    : m_backend(backend), m_recognizer(recognizer), m_vad(vad), m_options(options)
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
        const char* options[] = { "listening_ports", ports.c_str(), "num_threads", threads.c_str(),
                                  "request_timeout_ms", "120000", "enable_websocket_ping_pong", "yes",
                                  "websocket_timeout_ms", "3600000", nullptr };
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
    mg_set_request_handler(m_ctx, "/v1/models", modelsHandler, this);
    mg_set_request_handler(m_ctx, "/v1/audio/transcriptions", transcriptionsHandler, this);
    mg_set_websocket_handler(m_ctx, "/v1/realtime/transcribe", socketConnect, socketReady, socketData, socketClose, this);
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
        if (mg_get_var(info->query_string, std::strlen(info->query_string), "token", value, sizeof(value)) > 0
            && m_options.token == value)
            return true;
    }
    return false;
}

int Server::handleHealth(mg_connection* conn)
{
    reply(conn, 200, json { { "status", "ok" } }.dump());
    return 200;
}

int Server::handleInfo(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
    json devices = json::array();
    for (const DeviceInfo& d : listDevices())
        devices.push_back({ { "backend", d.backend }, { "name", d.name }, { "description", d.description }, { "type", d.type },
                            { "memory_free", d.memoryFree }, { "memory_total", d.memoryTotal } });
    const DeviceInfo& d = m_backend.device();
    const json info {
        { "id", "voice" },
        { "version", VOICE_VERSION },
        { "capabilities", { "transcription", "realtimeTranscription" } },
        { "device", { { "backend", d.backend }, { "name", d.name }, { "description", d.description } } },
        { "devices", devices },
        { "models", { { "asr", m_recognizer.path() }, { "asr_bytes", m_recognizer.weightsBytes() } } },
        { "queue", m_queue.pending() },
    };
    reply(conn, 200, info.dump());
    return 200;
}

int Server::handleModels(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
    reply(conn, 200,
          json { { "object", "list" }, { "data", { { { "id", "sensevoice-small" }, { "object", "model" }, { "owned_by", "friday-powers" } } } } }.dump());
    return 200;
}

int Server::handleTranscriptions(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
    const mg_request_info* info = mg_get_request_info(conn);
    if (!info || std::strcmp(info->request_method, "POST") != 0) {
        replyError(conn, 405, "use POST multipart/form-data");
        return 405;
    }
    // civetweb 分块回调字段内容：同一字段的后续块 key 是空串，接到 field_found 时记下的字段上
    struct Form {
        std::map<std::string, std::string> fields;
        std::string current;
    } form;
    auto& fields = form.fields;
    mg_form_data_handler handler {};
    handler.field_found = [](const char* key, const char*, char*, size_t, void* user) -> int {
        static_cast<Form*>(user)->current = key ? key : "";
        return MG_FORM_FIELD_STORAGE_GET;
    };
    handler.field_get = [](const char* key, const char* value, size_t size, void* user) -> int {
        auto& f = *static_cast<Form*>(user);
        const std::string name = key && *key ? std::string(key) : f.current;
        if (value && !name.empty())
            f.fields[name].append(value, size);
        return MG_FORM_FIELD_HANDLE_GET;
    };
    handler.user_data = &form;
    if (mg_handle_form_request(conn, &handler) < 0 || fields["file"].empty()) {
        replyError(conn, 400, "missing multipart field 'file'");
        return 400;
    }

    const auto started = std::chrono::steady_clock::now();
    std::vector<float> pcm;
    std::string error;
    if (!decodeAudio(fields["file"].data(), fields["file"].size(), pcm, &error)) {
        replyError(conn, 400, error);
        return 400;
    }
    StreamConfig config;
    config.language = fields.count("language") ? fields["language"] : "auto";
    if (!SenseVoice::supportsLanguage(config.language))
        config.language = "auto";
    config.endSilenceMs = 600;
    std::string language;
    const std::string text = transcribeWhole(m_recognizer, m_vad, m_fbank, pcm, config, &language, &error);
    if (!error.empty()) {
        replyError(conn, 500, error);
        return 500;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    VLOG_INFO("听录音：%.1f 秒音频，用时 %.3f 秒", pcm.size() / double(kSampleRate), seconds);
    const std::string format = fields.count("response_format") ? fields["response_format"] : "json";
    if (format == "text") {
        reply(conn, 200, text, "text/plain; charset=utf-8");
    } else if (format == "verbose_json") {
        reply(conn, 200, json { { "text", text }, { "language", language }, { "duration", pcm.size() / double(kSampleRate) } }.dump());
    } else {
        reply(conn, 200, json { { "text", text } }.dump());
    }
    return 200;
}

void Server::onSocketReady(mg_connection* conn)
{
    auto io = std::make_shared<SocketIO>();
    io->conn = conn;
    auto* session = new Session;
    session->io = io;
    session->streamer = std::make_shared<Streamer>(m_recognizer, m_vad, m_fbank, m_queue,
                                                   [io](const std::string& text) { io->send(text); }, m_options.vadThreads);
    mg_set_user_connection_data(conn, session);
    const DeviceInfo& d = m_backend.device();
    io->send(json { { "type", "ready" }, { "model", "sensevoice-small" }, { "device", d.backend + " " + d.description },
                    { "sample_rate", kSampleRate } }
                 .dump());
    VLOG_INFO("实时听写连接");
}

int Server::onSocketData(mg_connection* conn, int bits, char* data, size_t size)
{
    auto* session = static_cast<Session*>(mg_get_user_connection_data(conn));
    if (!session)
        return 0;
    const int opcode = bits & 0x0f;
    if (opcode == MG_WEBSOCKET_OPCODE_CONNECTION_CLOSE)
        return 0;
    if (opcode == MG_WEBSOCKET_OPCODE_BINARY || opcode == MG_WEBSOCKET_OPCODE_CONTINUATION) {
        session->streamer->feedPcm16(reinterpret_cast<const uint8_t*>(data), size);
        return 1;
    }
    if (opcode != MG_WEBSOCKET_OPCODE_TEXT)
        return 1;
    const json message = json::parse(std::string(data, size), nullptr, false);
    if (!message.is_object())
        return 1;
    const std::string type = message.value("type", "");
    if (type == "start") {
        StreamConfig config;
        config.language = message.value("language", config.language);
        config.sampleRate = message.value("sample_rate", config.sampleRate);
        config.partialIntervalMs = message.value("partial_interval_ms", config.partialIntervalMs);
        config.endSilenceMs = message.value("end_silence_ms", config.endSilenceMs);
        config.maxSegmentMs = message.value("max_segment_ms", config.maxSegmentMs);
        config.threshold = message.value("vad_threshold", config.threshold);
        config.partials = message.value("partials", config.partials);
        config.itn = message.value("itn", config.itn);
        session->streamer->configure(config);
    } else if (type == "flush") {
        session->streamer->flush();
    } else if (type == "stop") {
        session->streamer->finish();
    }
    return 1;
}

void Server::onSocketClose(const mg_connection* conn)
{
    auto* session = static_cast<Session*>(mg_get_user_connection_data(conn));
    if (!session)
        return;
    {
        std::lock_guard<std::mutex> lock(session->io->mutex);
        session->io->closed = true;
    }
    session->streamer->close();
    delete session;
    VLOG_INFO("实时听写断开");
}

} // namespace voice
