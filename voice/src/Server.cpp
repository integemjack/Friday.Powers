#include "Server.h"

#include "SpeakerEncoder.h"

#include "Log.h"
#include "Streamer.h"
#include "WhisperRecognizer.h"
#ifdef VOICE_HAS_TTS
#include "Speaker.h"
#include "Synthesizer.h"
#include "TextSplitter.h"
#include "cosyvoice.h"
#endif
#ifdef VOICE_HAS_CLONE
#include "VoiceCloner.h"
#endif

#include "civetweb.h"
#include "json.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
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

    void send(const std::string& text) { write(MG_WEBSOCKET_OPCODE_TEXT, text); }
    void sendBinary(const std::string& data) { write(MG_WEBSOCKET_OPCODE_BINARY, data); }

    void write(int opcode, const std::string& data)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (closed || !conn)
            return;
        mg_lock_connection(conn);
        mg_websocket_write(conn, opcode, data.data(), data.size());
        mg_unlock_connection(conn);
    }
};

/// 一条 WebSocket 连接：实时听写（streamer）或边写边读（speaker），二选一
struct Session {
    std::shared_ptr<SocketIO> io;
    std::shared_ptr<Streamer> streamer;
#ifdef VOICE_HAS_TTS
    std::shared_ptr<Speaker> speaker;
#endif
};

std::atomic<uint64_t> g_nextSpeechOwner { uint64_t(1) << 48 };   // HTTP 合成请求的 owner，和 Speaker 的分开

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

/// multipart/form-data 的各个字段（文件字段也整个读进内存：录音、音色都只有几百 KB）
bool readForm(mg_connection* conn, std::map<std::string, std::string>& fields)
{
    struct Form {
        std::map<std::string, std::string>* fields;
        std::string current;
    } form { &fields, {} };
    mg_form_data_handler handler {};
    handler.field_found = [](const char* key, const char*, char*, size_t, void* user) -> int {
        static_cast<Form*>(user)->current = key ? key : "";
        return MG_FORM_FIELD_STORAGE_GET;
    };
    handler.field_get = [](const char* key, const char* value, size_t size, void* user) -> int {
        auto& f = *static_cast<Form*>(user);
        const std::string name = key && *key ? std::string(key) : f.current;
        if (value && !name.empty())
            (*f.fields)[name].append(value, size);
        return MG_FORM_FIELD_HANDLE_GET;
    };
    handler.user_data = &form;
    return mg_handle_form_request(conn, &handler) >= 0;
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
int speechHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleSpeech(conn); }
int speakerHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleSpeaker(conn); }
int voicesHandler(mg_connection* conn, void* data) { return static_cast<Server*>(data)->handleVoices(conn); }
/// /v1/voice/clone、/v1/voice/import（POST），/v1/voice/<名字>（DELETE）
int voiceHandler(mg_connection* conn, void* data)
{
    const mg_request_info* info = mg_get_request_info(conn);
    std::string rest = info && info->local_uri_raw ? info->local_uri_raw : "";   // 没解码的：名字里的 %xx 在 handleVoice 里解一次
    rest = rest.size() > 10 && rest.compare(0, 10, "/v1/voice/") == 0 ? rest.substr(10) : std::string();
    return static_cast<Server*>(data)->handleVoice(conn, rest);
}
int socketConnect(const mg_connection* conn, void* data) { return static_cast<Server*>(data)->authorized(conn) ? 0 : 1; }
void socketReady(mg_connection* conn, void* data) { static_cast<Server*>(data)->onSocketReady(conn); }
void speakSocketReady(mg_connection* conn, void* data) { static_cast<Server*>(data)->onSpeakSocketReady(conn); }
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

Server::Server(Backend& backend, SenseVoice& recognizer, FsmnVad& vad, Synthesizer* synthesizer, const ServerOptions& options,
               SpeakerEncoder* voiceprint, Whisper* whisper, VoiceCloner* cloner, const std::string& userVoicesDir)
    : m_backend(backend), m_recognizer(recognizer), m_vad(vad), m_synth(synthesizer), m_voiceprint(voiceprint), m_whisper(whisper),
      m_cloner(cloner), m_userVoicesDir(userVoicesDir), m_options(options)
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
    if (m_voiceprint)
        mg_set_request_handler(m_ctx, "/v1/audio/speaker", speakerHandler, this);
    if (m_synth) {
        mg_set_request_handler(m_ctx, "/v1/audio/speech", speechHandler, this);
        mg_set_websocket_handler(m_ctx, "/v1/realtime/speak", socketConnect, speakSocketReady, socketData, socketClose, this);
        // v0.5 用户音色：导入、删除要合成 + 用户音色目录；克隆还要克隆前端（handleVoice 里看）
        if (!m_userVoicesDir.empty()) {
            mg_set_request_handler(m_ctx, "/v1/voices", voicesHandler, this);
            mg_set_request_handler(m_ctx, "/v1/voice", voiceHandler, this);
        }
    }
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
    json capabilities = { "transcription", "realtimeTranscription" };
    json models = { { "asr", m_recognizer.path() }, { "asr_bytes", m_recognizer.weightsBytes() } };
    if (m_voiceprint) {
        capabilities.push_back("speakerEmbedding");
        models["speaker"] = m_voiceprint->path();
    }
    // v0.3 定稿用的 Whisper（中英混说）
    if (m_whisper)
        models["whisper"] = m_whisper->path();
#ifdef VOICE_HAS_TTS
    if (m_synth) {
        capabilities.push_back("speech");
        capabilities.push_back("realtimeSpeech");
        models["tts"] = m_synth->path();
        models["tts_sample_rate"] = m_synth->sampleRate();
        models["voices"] = m_synth->voices();
        // v0.5 用户音色（克隆的、导入的）
        if (!m_userVoicesDir.empty()) {
            capabilities.push_back("voiceImport");
            models["user_voices"] = m_synth->userVoices();
        }
        if (cloneAvailable())
            capabilities.push_back("voiceCloning");
    }
#endif
    const json info {
        { "id", "voice" },
        { "version", VOICE_VERSION },
        { "capabilities", capabilities },
        { "device", { { "backend", d.backend }, { "name", d.name }, { "description", d.description } } },
        { "devices", devices },
        { "models", models },
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
    json data = json::array();
    data.push_back({ { "id", "sensevoice-small" }, { "object", "model" }, { "owned_by", "friday-powers" } });
    if (m_synth)
        data.push_back({ { "id", "cosyvoice3" }, { "object", "model" }, { "owned_by", "friday-powers" } });
    if (m_voiceprint)
        data.push_back({ { "id", "campplus" }, { "object", "model" }, { "owned_by", "friday-powers" } });
    reply(conn, 200, json { { "object", "list" }, { "data", data } }.dump());
    return 200;
}

int Server::handleSpeaker(mg_connection* conn)
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
    std::map<std::string, std::string> fields;
    if (!readForm(conn, fields) || fields["file"].empty()) {
        replyError(conn, 400, "missing multipart field 'file'");
        return 400;
    }
    std::vector<float> pcm;
    std::string error;
    if (!decodeAudio(fields["file"].data(), fields["file"].size(), pcm, &error)) {
        replyError(conn, 400, error);
        return 400;
    }
    const SpeakerEmbedding e = m_voiceprint->embed(pcm.data(), pcm.size());
    if (!e.ok) {
        replyError(conn, 400, e.error);
        return 400;
    }
    VLOG_INFO("声纹：%.1f 秒录音，用时 %.3f 秒", e.audioSeconds, e.seconds);
    reply(conn, 200, json { { "embedding", e.vector }, { "dim", e.vector.size() }, { "duration", pcm.size() / double(kSampleRate) } }.dump());
    return 200;
}

// ── v0.5 用户音色：克隆（参考录音 + 底稿 → prompt_speech）、导入（别处克隆好的 .gguf）、删除；都热生效，存在 <models>/user-voices ──

bool Server::cloneAvailable() const
{
#ifdef VOICE_HAS_CLONE
    return m_synth && m_cloner && m_cloner->available() && !m_userVoicesDir.empty();
#else
    return false;
#endif
}

namespace {

/// 音色名：1–40 个字，不要路径分隔符、控制字符和文件名里不能有的，不以点、空白开头结尾（文件名就是它）
bool validVoiceName(const std::string& name)
{
    if (name.empty() || name.size() > 120 || name[0] == '.' || name[0] == ' ' || name.back() == ' ' || name.back() == '.')
        return false;
    size_t chars = 0;
    for (const unsigned char c : name) {
        if ((c & 0xC0) != 0x80)
            ++chars;
        if (c < 0x20 || c == 0x7F || std::strchr("\\/:*?\"<>|", c))
            return false;
    }
    return chars <= 40;
}

} // namespace

int Server::handleVoices(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
#ifdef VOICE_HAS_TTS
    if (m_synth) {
        reply(conn, 200, json { { "voices", m_synth->voices() }, { "user_voices", m_synth->userVoices() }, { "default", m_synth->defaultVoice() },
                                { "clone", cloneAvailable() } }.dump());
        return 200;
    }
#endif
    replyError(conn, 404, "没有合成模型");
    return 404;
}

int Server::handleVoice(mg_connection* conn, const std::string& rest)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
#ifndef VOICE_HAS_TTS
    (void)rest;
    replyError(conn, 404, "这个版本没编合成");
    return 404;
#else
    if (!m_synth) {
        replyError(conn, 404, "没有合成模型");
        return 404;
    }
    const mg_request_info* info = mg_get_request_info(conn);
    const std::string method = info && info->request_method ? info->request_method : "";
    if (method == "DELETE") {
        std::string name(rest.size(), '\0');
        name.resize(size_t(std::max(0, mg_url_decode(rest.c_str(), int(rest.size()), name.data(), int(name.size()) + 1, 0))));
        std::string error;
        if (!m_synth->removeVoice(name, &error)) {
            const int status = m_synth->hasVoice(name) ? 403 : 404;
            replyError(conn, status, error);
            return status;
        }
        reply(conn, 200, json { { "deleted", name } }.dump());
        return 200;
    }
    if (method != "POST" || (rest != "clone" && rest != "import")) {
        replyError(conn, 405, "POST /v1/voice/clone、POST /v1/voice/import、DELETE /v1/voice/<名字>");
        return 405;
    }
    if (rest == "clone" && !cloneAvailable()) {
        replyError(conn, 501, "这台的本地语音不能克隆（克隆模型没下载，或者这个包没带克隆）");
        return 501;
    }
    std::map<std::string, std::string> fields;
    if (!readForm(conn, fields) || fields["file"].empty()) {
        replyError(conn, 400, "missing multipart field 'file'");
        return 400;
    }
    const std::string name = fields["name"];
    const bool replace = fields["replace"] == "true" || fields["replace"] == "1";
    if (!validVoiceName(name)) {
        replyError(conn, 400, "音色名要 1–40 个字，不能有 \\ / : * ? \" < > | 这些符号");
        return 400;
    }
    if (m_synth->hasVoice(name) && (!replace || !m_synth->isUserVoice(name))) {
        replyError(conn, 409, m_synth->isUserVoice(name) ? "已经有叫「" + name + "」的音色了" : "「" + name + "」是内置音色的名字，换一个");
        return 409;
    }
    const auto started = std::chrono::steady_clock::now();
    std::string error;
    json result { { "voice", name } };
    // 先写成临时文件（点开头，启动时扫描跳过），注册成功再改名成 <名字>.gguf
    namespace fs = std::filesystem;
    static std::atomic<unsigned> sequence { 0 };
    const fs::path dir = fs::u8path(m_userVoicesDir);
    const fs::path temp = dir / (".pending-" + std::to_string(++sequence) + ".gguf");
    const fs::path target = dir / fs::u8path(name + ".gguf");
    std::error_code ec;
    fs::create_directories(dir, ec);
    const auto fail = [&](int status, const std::string& message) {
        std::error_code ignored;
        fs::remove(temp, ignored);
        replyError(conn, status, message);
        return status;
    };
    if (rest == "import") {
        // 别处克隆好的音色（CosyVoice3 的 prompt_speech gguf，比如「克隆声音」工具的音色库）：原样存下，读得出来才收
        const std::string& data = fields["file"];
        {
            std::ofstream out(temp, std::ios::binary);
            out.write(data.data(), std::streamsize(data.size()));
            if (!out.good())
                return fail(500, "写不了音色文件：" + temp.u8string());
        }
        if (!Synthesizer::isVoiceFile(temp.u8string()))
            return fail(400, "不是 CosyVoice3 的音色文件（.gguf），或者文件坏了");
    } else {
#ifdef VOICE_HAS_CLONE
        // 同一段录音解两份：16 kHz 认语音 token、底稿听写；24 kHz 给说话人向量和 mel（别从 16 kHz 升，声音会发闷）
        const std::string& file = fields["file"];
        std::vector<float> pcm16, pcm24;
        if (!decodeAudio(file.data(), file.size(), pcm16, &error) || !decodeAudio(file.data(), file.size(), pcm24, &error, 24000))
            return fail(400, error);
        double seconds = pcm16.size() / double(kSampleRate);
        if (seconds < 3.0)
            return fail(400, "录音太短（至少 3 秒，最好 8～15 秒）");
        if (seconds > 30.0) {   // 太长前端慢、像不像也不再涨：只用前 30 秒
            pcm16.resize(size_t(30.0 * kSampleRate));
            pcm24.resize(size_t(30.0 * 24000));
            seconds = 30.0;
        }
        // 底稿：没给就听写（SenseVoice + Whisper 定稿，同 /v1/audio/transcriptions）
        std::string text = fields["text"];
        if (text.empty()) {
            StreamConfig config;
            config.language = fields["language"].empty() ? "auto" : fields["language"];
            config.endSilenceMs = 600;
            std::string language;
            text = transcribeWhole(m_recognizer, m_vad, m_fbank, pcm16, config, &language, &error, m_whisper);
            if (!error.empty())
                return fail(500, "听写录音失败：" + error);
            VLOG_INFO("克隆底稿（%s，自动听写）：%s", language.c_str(), text.c_str());
            if (text.empty())
                return fail(400, "录音里没听出说的话");
        }
        cosyvoice_prompt_speech_t cloned = m_cloner->clone(pcm16.data(), pcm16.size(), pcm24.data(), pcm24.size(), text, &error);
        if (!cloned)
            return fail(400, error);
        const bool saved = cosyvoice_prompt_speech_save_to_file(cloned, temp.u8string().c_str());
        cosyvoice_prompt_speech_free(cloned);
        if (!saved)
            return fail(500, "写不了音色文件：" + temp.u8string());
        result["transcript"] = text;
        result["seconds"] = seconds;
#endif
    }
    cosyvoice_prompt_speech_t speech = cosyvoice_prompt_speech_load_from_file(temp.u8string().c_str());
    if (!speech)
        return fail(400, "音色文件读不出来（CRC 不对，或者文件坏了）");
    if (!m_synth->addUserVoice(name, speech, target.u8string(), replace, &error))   // speech 归它了
        return fail(400, error);
    fs::rename(temp, target, ec);
    if (ec) {
        std::string ignored;
        m_synth->removeVoice(name, &ignored);
        return fail(500, "写不了音色文件：" + target.u8string());
    }
    const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    result["took"] = took;
    VLOG_INFO("用户音色%s：%s，用时 %.1f 秒", rest == "import" ? "导入" : "克隆", name.c_str(), took);
    reply(conn, 200, result.dump());
    return 200;
#endif
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
    // v0.3 OpenAI 兼容的 prompt（热词、上文）交给 Whisper；带 whisper=false 时只用 SenseVoice
    config.prompt = fields.count("prompt") ? fields["prompt"] : std::string();
    config.whisper = !(fields.count("whisper") && fields["whisper"] == "false");
    if (!config.prompt.empty())
        VLOG_DEBUG("听录音的热词：%s", config.prompt.c_str());
    std::string language;
    const std::string text = transcribeWhole(m_recognizer, m_vad, m_fbank, pcm, config, &language, &error, m_whisper);
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

int Server::handleSpeech(mg_connection* conn)
{
    if (!authorized(conn)) {
        replyError(conn, 401, "unauthorized");
        return 401;
    }
    const mg_request_info* info = mg_get_request_info(conn);
    if (!info || std::strcmp(info->request_method, "POST") != 0) {
        replyError(conn, 405, "use POST application/json");
        return 405;
    }
#ifndef VOICE_HAS_TTS
    replyError(conn, 501, "built without speech synthesis");
    return 501;
#else
    std::string body;
    char buffer[8192];
    for (int n; (n = mg_read(conn, buffer, sizeof(buffer))) > 0;)
        body.append(buffer, size_t(n));
    const json request = json::parse(body, nullptr, false);
    if (!request.is_object() || !request.contains("input") || !request["input"].is_string()) {
        replyError(conn, 400, "missing string field 'input'");
        return 400;
    }
    // OpenAI 的音色名（alloy 等）这里没有，认不出的一律用默认音色
    std::string voice = request.value("voice", std::string());
    if (!m_synth->hasVoice(voice))
        voice.clear();
    const float speed = std::clamp(request.value("speed", 1.0f), 0.5f, 2.0f);
    const std::string instruction = request.value("instructions", std::string());
    // 只出 wav / pcm（不带编码器）；要 mp3 等也给 wav
    const std::string format = request.value("response_format", std::string("wav")) == "pcm" ? "pcm" : "wav";
    const int sampleRate = std::clamp(request.value("sample_rate", m_synth->sampleRate()), 8000, 48000);

    TextSplitter splitter;
    std::vector<std::string> segments = splitter.push(request["input"].get<std::string>());
    for (std::string& s : splitter.flush())
        segments.push_back(std::move(s));
    if (segments.empty()) {
        replyError(conn, 400, "nothing to say in 'input'");
        return 400;
    }

    // 第一块音频出来才发响应头（之前失败还能回错误）；之后分块边合成边发
    bool headersSent = false, broken = false;
    const auto sendHeaders = [&] {
        if (headersSent)
            return;
        headersSent = true;
        mg_printf(conn,
                  "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\n"
                  "Connection: close\r\n\r\n",
                  format == "pcm" ? "audio/pcm" : "audio/wav");
        if (format == "wav") {
            // 流式 WAV：长度先写成最大值，播放器读到流结束为止
            const uint32_t rate = uint32_t(sampleRate), byteRate = rate * 2, unknown = 0xFFFFFFFFu;
            std::string header = "RIFF";
            const auto u32 = [&](uint32_t v) { header.append(reinterpret_cast<const char*>(&v), 4); };
            const auto u16 = [&](uint16_t v) { header.append(reinterpret_cast<const char*>(&v), 2); };
            u32(unknown);
            header += "WAVEfmt ";
            u32(16);
            u16(1);
            u16(1);
            u32(rate);
            u32(byteRate);
            u16(2);
            u16(16);
            header += "data";
            u32(unknown);
            mg_send_chunk(conn, header.data(), unsigned(header.size()));
        }
    };
    const uint64_t owner = g_nextSpeechOwner++;
    const auto started = std::chrono::steady_clock::now();
    StreamResampler resampler(m_synth->sampleRate(), sampleRate);
    std::vector<float> resampled;
    std::string pcm;
    size_t total = 0;
    double firstAudio = -1;
    for (const std::string& segment : segments) {
        const SpeakResult r = m_synth->speak(owner, segment, voice, speed, instruction, [&](const float* samples, size_t count) {
            if (firstAudio < 0)
                firstAudio = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            resampled.clear();
            resampler.process(samples, count, resampled);
            pcm.clear();
            floatToPcm16(resampled.data(), resampled.size(), pcm);
            total += resampled.size();
            sendHeaders();
            if (mg_send_chunk(conn, pcm.data(), unsigned(pcm.size())) < 0) {
                broken = true;   // 对方断开了
                return false;
            }
            return true;
        });
        if (broken)
            break;
        if (!r.ok) {
            if (!headersSent) {
                replyError(conn, 500, r.error);
                return 500;
            }
            break;
        }
    }
    if (!headersSent) {
        replyError(conn, 500, "合成没有出声音");
        return 500;
    }
    if (!broken)
        mg_send_chunk(conn, "", 0);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    VLOG_INFO("合成：%d 段，%.2f 秒音频，用时 %.3f 秒（首块 %.3f 秒）", int(segments.size()), total / double(sampleRate), seconds, firstAudio);
    return 200;
#endif
}

void Server::onSpeakSocketReady(mg_connection* conn)
{
#ifdef VOICE_HAS_TTS
    auto io = std::make_shared<SocketIO>();
    io->conn = conn;
    auto* session = new Session;
    session->io = io;
    session->speaker = std::make_shared<Speaker>(*m_synth, [io](const std::string& text) { io->send(text); },
                                                 [io](const std::string& data) { io->sendBinary(data); });
    session->speaker->start();
    mg_set_user_connection_data(conn, session);
    const DeviceInfo& d = m_backend.device();
    io->send(json { { "type", "ready" }, { "model", "cosyvoice3" }, { "device", d.backend + " " + d.description },
                    { "sample_rate", m_synth->sampleRate() }, { "voices", m_synth->voices() }, { "voice", m_synth->defaultVoice() } }
                 .dump());
    VLOG_INFO("边写边读连接");
#else
    (void)conn;
#endif
}

void Server::onSocketReady(mg_connection* conn)
{
    auto io = std::make_shared<SocketIO>();
    io->conn = conn;
    auto* session = new Session;
    session->io = io;
    session->streamer = std::make_shared<Streamer>(m_recognizer, m_vad, m_fbank, m_queue,
                                                   [io](const std::string& text) { io->send(text); }, m_options.vadThreads, m_voiceprint,
                                                   m_whisper);
    mg_set_user_connection_data(conn, session);
    const DeviceInfo& d = m_backend.device();
    io->send(json { { "type", "ready" }, { "model", "sensevoice-small" }, { "device", d.backend + " " + d.description },
                    { "sample_rate", kSampleRate }, { "speaker", m_voiceprint != nullptr }, { "whisper", m_whisper != nullptr } }
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
#ifdef VOICE_HAS_TTS
    if (session->speaker) {
        if (opcode != MG_WEBSOCKET_OPCODE_TEXT)
            return 1;
        const json message = json::parse(std::string(data, size), nullptr, false);
        if (!message.is_object())
            return 1;
        const std::string type = message.value("type", "");
        if (type == "start") {
            SpeakConfig config;
            config.voice = message.value("voice", config.voice);
            config.instruction = message.value("instruction", config.instruction);
            config.speed = message.value("speed", config.speed);
            config.sampleRate = message.value("sample_rate", m_synth->sampleRate());
            session->speaker->configure(config);
        } else if (type == "text") {
            session->speaker->text(message.value("text", std::string()));
        } else if (type == "flush") {
            session->speaker->flush();
        } else if (type == "say") {
            session->speaker->text(message.value("text", std::string()));
            session->speaker->flush();
        } else if (type == "cancel") {
            session->speaker->cancel();
        }
        return 1;
    }
#endif
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
        config.speaker = message.value("speaker", config.speaker) && m_voiceprint;
        config.whisper = message.value("whisper", config.whisper);
        config.prompt = message.value("prompt", config.prompt);
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
#ifdef VOICE_HAS_TTS
    if (session->speaker) {
        session->speaker->close();
        delete session;
        VLOG_INFO("边写边读断开");
        return;
    }
#endif
    session->streamer->close();
    delete session;
    VLOG_INFO("实时听写断开");
}

} // namespace voice
