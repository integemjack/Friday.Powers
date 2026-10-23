// 见 Ports.h
#include "Ports.h"

#include "Log.h"

#include <algorithm>
#include <condition_variable>

namespace devices {

namespace {

/// 开会话要等客户端那头的人点授权（最多 60 秒）加上打开设备
constexpr int kOpenTimeoutMs = 120000;

std::string urlFor(const std::string& kind, int port, const std::string& mode)
{
    const std::string address = "127.0.0.1:" + std::to_string(port);
    if (kind == "serial")
        return (mode == "raw" ? "socket://" : "rfc2217://") + address;
    if (kind == "adb")
        return "tcp:" + address;
    return address;
}

} // namespace

/// 一个 TCP 连接 ⇆ 一条专用通道
struct Ports::Connection : ChannelSink {
    net::Handle socket = net::kInvalid;
    std::string client;
    std::string label;
    std::mutex mutex;
    std::condition_variable wake;
    bool paused = false;
    std::atomic<bool> closed { false };

    void onData(const std::string& data) override
    {
        if (!closed && !net::sendAll(socket, data.data(), data.size()))
            shut();
    }
    void onClosed() override { shut(); }
    void onPause(bool value) override
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            paused = value;
        }
        wake.notify_all();
    }
    void shut()
    {
        if (closed.exchange(true))
            return;
        net::shutdownBoth(socket);
        wake.notify_all();
    }
    void waitWhilePaused()
    {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock, [this] { return !paused || closed.load(); });
    }
};

struct Ports::Port {
    int id = 0;
    PortSpec spec;
    int port = 0;
    std::string url;
    net::Handle listener = net::kInvalid;
    std::atomic<bool> stopping { false };
    std::thread acceptThread;
    std::mutex mutex;
    std::vector<std::shared_ptr<Connection>> connections;
    std::string lastError;
    int served = 0;
};

Ports::Ports(Bridge& bridge)
    : m_bridge(bridge)
{
}

Ports::~Ports()
{
    std::vector<int> ids;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& [id, port] : m_ports)
            ids.push_back(id);
    }
    for (int id : ids)
        close(id);
}

json Ports::open(const PortSpec& spec)
{
    std::string error;
    int actual = 0;
    const net::Handle listener = net::listenLocal(spec.port, &actual, &error);
    if (listener == net::kInvalid)
        throw DeviceError { 23, error };
    auto port = std::make_shared<Port>();
    port->spec = spec;
    port->port = actual;
    port->listener = listener;
    port->url = urlFor(spec.kind, actual, spec.openParams.value("mode", std::string("rfc2217")));
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        port->id = ++m_nextID;
        m_ports[port->id] = port;
    }
    port->acceptThread = std::thread([this, port] { acceptLoop(port); });
    DLOG_INFO("端口 %d：%s → %s（%s）", actual, spec.kind.c_str(), spec.target.c_str(), spec.clientName.c_str());
    return describe(*port);
}

bool Ports::close(int id)
{
    std::shared_ptr<Port> port;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_ports.find(id);
        if (it == m_ports.end())
            return false;
        port = it->second;
        m_ports.erase(it);
    }
    port->stopping = true;
    net::shutdownBoth(port->listener);
    net::closeHandle(port->listener);
    std::vector<std::shared_ptr<Connection>> connections;
    {
        std::lock_guard<std::mutex> lock(port->mutex);
        connections = port->connections;
    }
    for (const auto& connection : connections) {
        connection->shut();
        if (!connection->label.empty())
            m_bridge.closeChannel(connection->client, connection->label);
    }
    if (port->acceptThread.joinable())
        port->acceptThread.join();
    DLOG_INFO("端口 %d 关了", port->port);
    return true;
}

void Ports::closeClient(const std::string& client)
{
    std::vector<int> ids;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& [id, port] : m_ports) {
            if (port->spec.client == client)
                ids.push_back(id);
        }
    }
    for (int id : ids)
        close(id);
}

json Ports::describe(const Port& port)
{
    return json { { "id", port.id },
                  { "kind", port.spec.kind },
                  { "port", port.port },
                  { "url", port.url },
                  { "target", port.spec.target },
                  { "client", port.spec.clientName },
                  { "clientId", port.spec.client } };
}

json Ports::list() const
{
    std::vector<std::shared_ptr<Port>> ports;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& [id, port] : m_ports)
            ports.push_back(port);
    }
    json list = json::array();
    for (const auto& port : ports) {
        json entry = describe(*port);
        std::lock_guard<std::mutex> lock(port->mutex);
        entry["connections"] = port->connections.size();
        entry["served"] = port->served;
        if (!port->lastError.empty())
            entry["lastError"] = port->lastError;
        list.push_back(entry);
    }
    return list;
}

void Ports::acceptLoop(std::shared_ptr<Port> port)
{
    while (!port->stopping) {
        const net::Handle socket = net::acceptOne(port->listener);
        if (socket == net::kInvalid) {
            if (port->stopping)
                break;
            continue;
        }
        std::thread([this, port, socket] { serve(port, socket); }).detach();
    }
}

void Ports::serve(std::shared_ptr<Port> port, net::Handle socket)
{
    auto connection = std::make_shared<Connection>();
    connection->socket = socket;
    connection->client = port->spec.client;
    {
        std::lock_guard<std::mutex> lock(port->mutex);
        port->connections.push_back(connection);
        ++port->served;
    }
    const auto forget = [&] {
        std::lock_guard<std::mutex> lock(port->mutex);
        auto& list = port->connections;
        list.erase(std::remove(list.begin(), list.end(), connection), list.end());
    };
    try {
        const json opened = m_bridge.request(port->spec.client, port->spec.openMethod, port->spec.openParams, kOpenTimeoutMs);
        connection->label = opened.value("channel", "");
        if (connection->label.empty())
            throw DeviceError { 23, "客户端没给通道" };
        {
            std::lock_guard<std::mutex> lock(port->mutex);
            port->lastError.clear();
        }
    } catch (const DeviceError& error) {
        DLOG_WARN("端口 %d 开会话失败：%s", port->port, error.message.c_str());
        {
            std::lock_guard<std::mutex> lock(port->mutex);
            port->lastError = error.message;
        }
        net::shutdownBoth(socket);
        net::closeHandle(socket);
        forget();
        return;
    }
    if (port->stopping) {
        m_bridge.closeChannel(connection->client, connection->label);
        net::closeHandle(socket);
        forget();
        return;
    }
    m_bridge.bindChannel(connection->client, connection->label, connection);
    // socket → 通道（这个线程）；通道 → socket 在桥的线程上（Connection::onData）
    char buffer[64 * 1024];
    while (!connection->closed) {
        const long n = net::receiveSome(socket, buffer, sizeof(buffer));
        if (n <= 0)
            break;
        connection->waitWhilePaused();
        if (connection->closed)
            break;
        m_bridge.sendChannel(connection->client, connection->label, buffer, size_t(n));
    }
    const bool closedByPeer = connection->closed.exchange(true);
    m_bridge.unbindChannel(connection->client, connection->label);
    if (!closedByPeer)
        m_bridge.closeChannel(connection->client, connection->label);
    net::shutdownBoth(socket);
    net::closeHandle(socket);
    forget();
}

} // namespace devices
