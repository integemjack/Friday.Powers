// 串口监视（docs/devices-protocol.md §4「串口监视」）：智能体不用在服务端写脚本，直接读串口输出、发数据、复位板子。
//
// 一个监视 = 向客户端开一个 rfc2217 串口会话（device.serial.open），通道认领到这里：
//   - 这里是 RFC 2217 的客户端那头：开场协商、设波特率、用 SET-CONTROL 拉 DTR / RTS 复位；
//   - 收到的串口数据去掉 Telnet 转义后放进缓冲（只留最近 512 KB，按总字节数编游标），serial_read 从游标往后取；
//   - 客户端那边烧录同一个口时会话被挂起、烧完恢复，监视不断——正好接着看新固件的启动日志。
#pragma once

#include "Bridge.h"

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>

namespace devices {

class Monitor : public ChannelSink, public std::enable_shared_from_this<Monitor> {
public:
    Monitor(Bridge& bridge, int id, std::string client, std::string clientName, std::string deviceId, std::string target);

    /// 开会话、认领通道、协商、设波特率（工作线程调；失败抛 DeviceError）
    void open(int baudRate);
    void close();

    /// 读 cursor 之后的输出：最多等 waitMs，until 非空时等到输出里出现它（或超时 / 会话关了）。
    /// cursor < 0 = 从上次读到的地方接着读
    json read(long long cursor, int waitMs, const std::string& until, size_t maxBytes, bool hex);
    void write(const std::string& data);
    /// 复位：rts = ESP 的 EN（RTS 拉低 100 ms 再放开，DTR 放开）；dtr = Arduino（DTR 拉一下，经电容复位）
    void reset(const std::string& mode);
    void setBaudRate(int baudRate);
    json describe() const;

    int id() const { return m_id; }
    const std::string& client() const { return m_client; }
    bool closed() const;

    // ChannelSink
    void onData(const std::string& data) override;
    void onClosed() override;
    void onPause(bool) override {}

private:
    void send(const std::string& raw);
    void control(uint8_t value);
    void negotiate(uint8_t command, uint8_t option);

    Bridge& m_bridge;
    const int m_id;
    const std::string m_client;
    const std::string m_clientName;
    const std::string m_deviceId;
    const std::string m_target;
    std::string m_label;
    int m_baudRate = 115200;

    mutable std::mutex m_mutex;
    std::condition_variable m_wake;
    /// 缓冲：m_buffer[0] 是第 m_base 个字节（从会话开始算）
    std::string m_buffer;
    unsigned long long m_base = 0;
    unsigned long long m_readCursor = 0;
    bool m_closed = false;
    std::string m_closeReason;

    // Telnet 解析状态
    enum class State { Data, Iac, Option, Sub, SubIac } m_state = State::Data;
    uint8_t m_optionCommand = 0;
    /// 已经同意 / 请求过的选项（不重复回，免得和对端来回打转）
    std::set<uint8_t> m_local;
    std::set<uint8_t> m_remote;
};

} // namespace devices
