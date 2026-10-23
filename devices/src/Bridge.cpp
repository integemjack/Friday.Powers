// 见 Bridge.h
#include "Bridge.h"

#include "Log.h"

#include "civetweb.h"

#include <chrono>

namespace devices {

namespace {

/// 没认领的通道最多攒这么多（多了丢掉、关通道：没人要的会话）
constexpr size_t kMaxBuffered = 4 * 1024 * 1024;

} // namespace

Bridge::Bridge() = default;

Bridge::~Bridge()
{
    failAll("能力在退出");
}

std::string Bridge::key(const std::string& client, const std::string& label)
{
    return client + "\n" + label;
}

bool Bridge::connected() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_conn != nullptr;
}

std::vector<ClientInfo> Bridge::clients() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<ClientInfo> list;
    for (const auto& [id, client] : m_clients)
        list.push_back(client);
    return list;
}

ClientInfo Bridge::pickClient(const std::string& hint) const
{
    const std::vector<ClientInfo> list = clients();
    if (list.empty())
        throw DeviceError { 21, connected() ? "没有连着的客户端（要 Friday Work 桌面客户端 0.5.84 以上连着这台）"
                                            : "Friday 还没接上这个能力" };
    const ClientInfo* best = nullptr;
    for (const ClientInfo& client : list) {
        if (!hint.empty() && client.id != hint && client.name.find(hint) == std::string::npos)
            continue;
        if (!best || client.order > best->order)
            best = &client;
    }
    if (!best)
        throw DeviceError { 21, "没有叫「" + hint + "」的客户端" };
    return *best;
}

// MARK: - WebSocket

void Bridge::attach(mg_connection* conn)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_conn && m_conn != conn)
        DLOG_WARN("又一条桥连进来：顶掉之前那条");
    m_conn = conn;
    m_fragments.clear();
    m_fragmentOpcode = 0;
    DLOG_INFO("Friday 接上了");
}

void Bridge::detach(const mg_connection* conn)
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_conn != conn)
            return;
        m_conn = nullptr;
        m_clients.clear();
    }
    DLOG_INFO("Friday 断开了");
    failAll("和 Friday 的连接断了");
}

void Bridge::failAll(const std::string& reason)
{
    std::map<std::string, ChannelState> channels;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto& [id, pending] : m_pending) {
            if (!pending->done) {
                pending->done = true;
                pending->error = DeviceError { 21, reason };
            }
        }
        m_pending.clear();
        channels.swap(m_channels);
    }
    m_wake.notify_all();
    for (auto& [key, channel] : channels) {
        if (channel.sink)
            channel.sink->onClosed();
    }
}

void Bridge::onFrame(const mg_connection* conn, int bits, const char* data, size_t size)
{
    const int opcode = bits & 0x0f;
    const bool fin = (bits & 0x80) != 0;
    std::string message;
    int messageOpcode = 0;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (conn != m_conn)
            return;
        if (opcode == MG_WEBSOCKET_OPCODE_TEXT || opcode == MG_WEBSOCKET_OPCODE_BINARY) {
            m_fragmentOpcode = opcode;
            m_fragments.assign(data, size);
        } else if (opcode == MG_WEBSOCKET_OPCODE_CONTINUATION) {
            m_fragments.append(data, size);
        } else {
            return;   // ping / pong / close：civetweb 自己处理
        }
        if (!fin)
            return;
        message.swap(m_fragments);
        messageOpcode = m_fragmentOpcode;
    }
    if (messageOpcode == MG_WEBSOCKET_OPCODE_BINARY) {
        onBinary(message);
        return;
    }
    const json parsed = json::parse(message, nullptr, false);
    if (parsed.is_object())
        onText(parsed);
}

void Bridge::writeText(const json& message)
{
    const std::string text = message.dump(-1, ' ', false, json::error_handler_t::replace);
    std::lock_guard<std::mutex> writeLock(m_writeMutex);
    mg_connection* conn = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        conn = m_conn;
    }
    if (!conn)
        return;
    mg_lock_connection(conn);
    mg_websocket_write(conn, MG_WEBSOCKET_OPCODE_TEXT, text.data(), text.size());
    mg_unlock_connection(conn);
}

void Bridge::onText(const json& message)
{
    const std::string type = message.value("type", "");
    if (type == "hello") {
        return;
    }
    if (type == "clients") {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::map<std::string, ClientInfo> next;
        for (const json& entry : message.value("clients", json::array())) {
            ClientInfo client;
            client.id = entry.value("id", "");
            client.name = entry.value("name", "");
            for (const json& kind : entry.value("kinds", json::array()))
                client.kinds.push_back(kind.get<std::string>());
            const auto previous = m_clients.find(client.id);
            client.order = previous != m_clients.end() ? previous->second.order : ++m_nextOrder;
            if (!client.id.empty())
                next[client.id] = client;
        }
        for (const auto& [id, client] : next) {
            if (!m_clients.count(id))
                DLOG_INFO("客户端连上：%s（%s）", client.name.c_str(), id.c_str());
        }
        m_clients.swap(next);
        return;
    }
    if (type == "response") {
        const uint64_t id = message.value("id", uint64_t(0));
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            const auto it = m_pending.find(id);
            if (it == m_pending.end())
                return;
            Pending& pending = *it->second;
            pending.done = true;
            if (message.contains("error")) {
                const json error = message["error"];
                pending.error = DeviceError { error.value("code", 23), error.value("message", std::string("客户端出错")) };
            } else {
                pending.ok = true;
                pending.result = message.value("result", json::object());
            }
            m_pending.erase(it);
        }
        m_wake.notify_all();
        return;
    }
    if (type == "notify") {
        std::function<void(const std::string&, const std::string&, const json&)> handler;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            handler = m_notify;
        }
        if (handler)
            handler(message.value("client", ""), message.value("method", ""), message.value("params", json::object()));
        return;
    }
    const std::string channelKey = key(message.value("client", ""), message.value("label", ""));
    if (type == "channel.open") {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_channels.emplace(channelKey, ChannelState {});
        return;
    }
    if (type == "channel.closed") {
        std::shared_ptr<ChannelSink> sink;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            const auto it = m_channels.find(channelKey);
            if (it == m_channels.end())
                return;
            sink = it->second.sink;
            if (sink)
                m_channels.erase(it);
            else
                it->second.closed = true;   // 还没认领：认领时马上告诉它
        }
        if (sink)
            sink->onClosed();
        return;
    }
    if (type == "channel.pause" || type == "channel.resume") {
        const bool paused = type == "channel.pause";
        std::shared_ptr<ChannelSink> sink;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            const auto it = m_channels.find(channelKey);
            if (it == m_channels.end())
                return;
            it->second.paused = paused;
            sink = it->second.sink;
        }
        if (sink)
            sink->onPause(paused);
    }
}

void Bridge::onBinary(const std::string& frame)
{
    if (frame.empty())
        return;
    const size_t headerSize = uint8_t(frame[0]);
    if (frame.size() < 1 + headerSize)
        return;
    const std::string channelKey = frame.substr(1, headerSize);
    std::shared_ptr<ChannelSink> sink;
    bool overflow = false;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_channels.find(channelKey);
        if (it == m_channels.end())
            return;
        sink = it->second.sink;
        if (!sink) {
            it->second.buffered.append(frame, 1 + headerSize, std::string::npos);
            overflow = it->second.buffered.size() > kMaxBuffered;
        }
    }
    if (sink) {
        sink->onData(frame.substr(1 + headerSize));
        return;
    }
    if (overflow) {
        const size_t newline = channelKey.find('\n');
        DLOG_WARN("通道 %s 一直没人认领、攒得太多：关掉", channelKey.c_str());
        closeChannel(channelKey.substr(0, newline), channelKey.substr(newline + 1));
    }
}

// MARK: - 请求

json Bridge::request(const std::string& client, const std::string& method, const json& params, int timeoutMs)
{
    auto pending = std::make_shared<Pending>();
    uint64_t id = 0;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_conn)
            throw DeviceError { 21, "Friday 还没接上这个能力" };
        if (!m_clients.count(client))
            throw DeviceError { 21, "这个客户端已经断开了" };
        id = ++m_nextID;
        m_pending[id] = pending;
    }
    writeText(json { { "type", "request" }, { "id", id }, { "client", client }, { "method", method }, { "params", params },
                     { "timeoutMs", timeoutMs } });
    std::unique_lock<std::mutex> lock(m_mutex);
    // Friday 那头到点会回超时；这里多等 5 秒兜底（桥半死不活时）
    const bool finished = m_wake.wait_for(lock, std::chrono::milliseconds(timeoutMs + 5000), [&] { return pending->done; });
    if (!finished) {
        m_pending.erase(id);
        throw DeviceError { 23, "客户端处理超时" };
    }
    if (!pending->ok)
        throw pending->error;
    return pending->result;
}

void Bridge::setNotifyHandler(std::function<void(const std::string&, const std::string&, const json&)> handler)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_notify = std::move(handler);
}

// MARK: - 通道

void Bridge::bindChannel(const std::string& client, const std::string& label, std::shared_ptr<ChannelSink> sink)
{
    std::string buffered;
    bool closed = false;
    bool paused = false;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        ChannelState& state = m_channels[key(client, label)];
        state.sink = sink;
        buffered.swap(state.buffered);
        closed = state.closed;
        paused = state.paused;
        if (closed)
            m_channels.erase(key(client, label));
    }
    if (!buffered.empty())
        sink->onData(buffered);
    if (paused)
        sink->onPause(true);
    if (closed)
        sink->onClosed();
}

void Bridge::unbindChannel(const std::string& client, const std::string& label)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_channels.erase(key(client, label));
}

void Bridge::sendChannel(const std::string& client, const std::string& label, const char* data, size_t size)
{
    const std::string header = key(client, label);
    if (header.size() > 255)
        return;
    std::string frame;
    frame.reserve(1 + header.size() + size);
    frame.push_back(char(uint8_t(header.size())));
    frame += header;
    frame.append(data, size);
    std::lock_guard<std::mutex> writeLock(m_writeMutex);
    mg_connection* conn = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        conn = m_conn;
    }
    if (!conn)
        return;
    mg_lock_connection(conn);
    mg_websocket_write(conn, MG_WEBSOCKET_OPCODE_BINARY, frame.data(), frame.size());
    mg_unlock_connection(conn);
}

void Bridge::closeChannel(const std::string& client, const std::string& label)
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_channels.erase(key(client, label));
    }
    writeText(json { { "type", "channel.close" }, { "client", client }, { "label", label } });
}

} // namespace devices
