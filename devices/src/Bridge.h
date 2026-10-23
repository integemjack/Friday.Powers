// 和 Friday 之间的桥（docs/devices-protocol.md §2）：Friday 连 /v1/bridge（WebSocket），一次只有一条（新的顶掉旧的）。
//
// 文本帧（JSON，type 区分）：
//   Friday → 这里：hello、clients（声明了 clientDevices 的客户端全表）、response（request 的结果）、notify（客户端的 device.* 通知）、
//                  channel.open / channel.closed（客户端开 / 关了一条专用通道）、channel.pause / channel.resume（那条通道积压了 / 发完了）
//   这里 → Friday：request（{id, client, method, params, timeoutMs}：转给那个客户端的 device.* 请求）、channel.close（关一条通道）
// 二进制帧：[1 字节头长 H][H 字节 "<client>\n<label>"][数据]——专用通道上的字节，两个方向一样。
//
// 通道和本机 socket 的配对：开会话的请求（device.serial.open / device.tcp.open）回来时才知道 label，
// 而客户端开通道、发字节可能比回复先到——没认领的通道先把字节攒着（最多 4 MB），bindChannel 时一起交出去。
#pragma once

#include "json.hpp"

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct mg_connection;

namespace devices {

using json = nlohmann::json;

/// 客户端回的错误（JSON-RPC 的 code / message），或者桥这头的（没连着、超时……）
struct DeviceError {
    int code = 23;
    std::string message;
};

struct ClientInfo {
    std::string id;
    std::string name;
    std::vector<std::string> kinds;
    /// 第几个连上的（挑缺省客户端用：最近连上的）
    uint64_t order = 0;
};

/// 一条专用通道在本机这头的接收者（Ports 里的一个 TCP 连接）
struct ChannelSink {
    virtual ~ChannelSink() = default;
    /// 通道来的字节（桥的线程上调，要快）
    virtual void onData(const std::string& data) = 0;
    /// 通道关了（客户端关的 / 客户端断了 / 桥断了）
    virtual void onClosed() = 0;
    /// Friday 那头的 DataChannel 积压了：先别往里写；resume 后接着
    virtual void onPause(bool paused) = 0;
};

class Bridge {
public:
    Bridge();
    ~Bridge();

    // MARK: civetweb 的 WebSocket 回调（Server 调）
    void attach(mg_connection* conn);
    void detach(const mg_connection* conn);
    /// 一帧（可能是分片）：FIN 到了才处理整条
    void onFrame(const mg_connection* conn, int bits, const char* data, size_t size);

    bool connected() const;
    std::vector<ClientInfo> clients() const;
    /// 按 id 或名字里的一段找客户端；空 = 缺省（最近连上的）。找不到抛 DeviceError
    ClientInfo pickClient(const std::string& hint) const;

    /// 向客户端发 device.* 请求，阻塞等结果（工作线程调）；失败抛 DeviceError
    json request(const std::string& client, const std::string& method, const json& params, int timeoutMs);

    /// 认领一条通道：攒着的字节马上交给 sink；通道已经关了就马上 onClosed
    void bindChannel(const std::string& client, const std::string& label, std::shared_ptr<ChannelSink> sink);
    void unbindChannel(const std::string& client, const std::string& label);
    /// 往通道里写（经 Friday 发给客户端）
    void sendChannel(const std::string& client, const std::string& label, const char* data, size_t size);
    /// 关通道（Friday 关掉那条 DataChannel）
    void closeChannel(const std::string& client, const std::string& label);

    /// 客户端的通知（device.esp.progress 等）：Tools 订
    void setNotifyHandler(std::function<void(const std::string& client, const std::string& method, const json& params)> handler);

private:
    struct Pending {
        bool done = false;
        bool ok = false;
        json result;
        DeviceError error;
    };
    struct ChannelState {
        std::shared_ptr<ChannelSink> sink;
        std::string buffered;
        bool closed = false;
        bool paused = false;
    };

    static std::string key(const std::string& client, const std::string& label);
    void writeText(const json& message);
    void onText(const json& message);
    void onBinary(const std::string& frame);
    void failAll(const std::string& reason);

    mutable std::mutex m_mutex;
    std::condition_variable m_wake;
    /// 写要另加一把锁（civetweb 的连接锁之外）：多个工作线程会同时写
    std::mutex m_writeMutex;
    mg_connection* m_conn = nullptr;
    std::string m_fragments;
    int m_fragmentOpcode = 0;
    std::map<std::string, ClientInfo> m_clients;
    uint64_t m_nextOrder = 0;
    uint64_t m_nextID = 0;
    std::map<uint64_t, std::shared_ptr<Pending>> m_pending;
    std::map<std::string, ChannelState> m_channels;
    std::function<void(const std::string&, const std::string&, const json&)> m_notify;
};

} // namespace devices
