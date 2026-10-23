// 本机端口（docs/devices-protocol.md §3）：把客户端的一样东西开成 127.0.0.1 上的一个端口，服务端的程序当本机的用。
//
//   serial  ：rfc2217://127.0.0.1:<port>（或 raw 字节）→ 每来一个连接调 device.serial.open
//   tcp     ：客户端那台能连到的 host:port（局域网里的树莓派、客户端本机的 openocd……）→ device.tcp.open {host, port}
//   adb     ：客户端的 adb server，只放行授权过的那台手机 → device.tcp.open {service: adb, deviceId}
//   usbmux  ：客户端的 usbmuxd（iOS）→ device.tcp.open {service: usbmux}
// 每个连接：开会话的请求回来拿到通道 label → Bridge::bindChannel；socket 读到的写进通道，通道来的写进 socket；
// 任一边关了另一边跟着关。开会话失败（客户端拒绝、设备拔了）就直接关这个连接，错误记在端口上（ports 工具里看得到）。
#pragma once

#include "Bridge.h"
#include "Socket.h"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace devices {

struct PortSpec {
    /// serial / tcp / adb / usbmux
    std::string kind;
    std::string client;
    std::string clientName;
    /// 每来一个连接发的请求（device.serial.open / device.tcp.open）和参数
    std::string openMethod;
    json openParams;
    /// 给人看的：COM5 · CP210x、192.168.1.50:22、Pixel 8……
    std::string target;
    /// 想要的本机端口（0 = 系统分配）
    int port = 0;
};

class Ports {
public:
    explicit Ports(Bridge& bridge);
    ~Ports();

    /// 开一个端口（立即返回）；端口用不了抛 DeviceError
    json open(const PortSpec& spec);
    /// 关一个端口和它上面的连接；没有这个 id 返回 false
    bool close(int id);
    /// 关掉某个客户端的所有端口（它断开很久了）
    void closeClient(const std::string& client);
    /// [{id, kind, port, url, target, client, connections, lastError}]
    json list() const;

private:
    struct Connection;
    struct Port;

    void acceptLoop(std::shared_ptr<Port> port);
    void serve(std::shared_ptr<Port> port, net::Handle socket);
    static json describe(const Port& port);

    Bridge& m_bridge;
    mutable std::mutex m_mutex;
    std::map<int, std::shared_ptr<Port>> m_ports;
    int m_nextID = 0;
};

} // namespace devices
