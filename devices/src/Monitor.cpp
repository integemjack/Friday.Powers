// 见 Monitor.h
#include "Monitor.h"

#include "Log.h"

#include <chrono>
#include <thread>

namespace devices {

namespace {

constexpr uint8_t IAC = 255, DONT = 254, DO = 253, WONT = 252, WILL = 251, SB = 250, SE = 240;
constexpr uint8_t OPT_BINARY = 0, OPT_SGA = 3, OPT_COM_PORT = 44;
constexpr uint8_t SET_BAUDRATE = 1, SET_CONTROL = 5;
constexpr uint8_t DTR_ON = 8, DTR_OFF = 9, RTS_ON = 11, RTS_OFF = 12;
constexpr size_t kKeepBytes = 512 * 1024;
constexpr int kOpenTimeoutMs = 120000;

bool supported(uint8_t option)
{
    return option == OPT_BINARY || option == OPT_SGA || option == OPT_COM_PORT;
}

std::string subnegotiation(uint8_t code, const std::string& value)
{
    std::string out { char(IAC), char(SB), char(OPT_COM_PORT), char(code) };
    for (char c : value) {
        out.push_back(c);
        if (uint8_t(c) == IAC)
            out.push_back(c);
    }
    out.push_back(char(IAC));
    out.push_back(char(SE));
    return out;
}

std::string hexOf(const std::string& data)
{
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(data.size() * 3);
    for (char c : data) {
        if (!out.empty())
            out.push_back(' ');
        out.push_back(digits[uint8_t(c) >> 4]);
        out.push_back(digits[uint8_t(c) & 15]);
    }
    return out;
}

} // namespace

Monitor::Monitor(Bridge& bridge, int id, std::string client, std::string clientName, std::string deviceId, std::string target)
    : m_bridge(bridge)
    , m_id(id)
    , m_client(std::move(client))
    , m_clientName(std::move(clientName))
    , m_deviceId(std::move(deviceId))
    , m_target(std::move(target))
{
}

void Monitor::open(int baudRate)
{
    m_baudRate = baudRate;
    const json opened = m_bridge.request(m_client, "device.serial.open",
                                         json { { "deviceId", m_deviceId }, { "mode", "rfc2217" }, { "baudRate", baudRate } }, kOpenTimeoutMs);
    m_label = opened.value("channel", "");
    if (m_label.empty())
        throw DeviceError { 23, "客户端没给通道" };
    m_bridge.bindChannel(m_client, m_label, shared_from_this());
    // 我们这头：要 BINARY、SGA、COM-PORT-OPTION（两个方向），记下来免得对端回同意时再回
    std::string hello;
    for (uint8_t option : { OPT_BINARY, OPT_SGA, OPT_COM_PORT }) {
        hello += std::string { char(IAC), char(WILL), char(option), char(IAC), char(DO), char(option) };
        m_local.insert(option);
        m_remote.insert(option);
    }
    send(hello);
    setBaudRate(baudRate);
    DLOG_INFO("串口监视 %d：%s（%s）", m_id, m_target.c_str(), m_clientName.c_str());
}

void Monitor::close()
{
    if (closed())
        return;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_closed = true;
        m_closeReason = "关了";
    }
    m_wake.notify_all();
    if (!m_label.empty()) {
        m_bridge.unbindChannel(m_client, m_label);
        m_bridge.closeChannel(m_client, m_label);
    }
}

bool Monitor::closed() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_closed;
}

json Monitor::describe() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    json out { { "id", m_id },           { "kind", "monitor" },         { "target", m_target },
               { "deviceId", m_deviceId }, { "client", m_clientName },   { "baudRate", m_baudRate },
               { "bytes", m_base + m_buffer.size() }, { "cursor", m_readCursor }, { "closed", m_closed } };
    if (m_closed)
        out["closeReason"] = m_closeReason;
    return out;
}

// MARK: - 收

void Monitor::onData(const std::string& data)
{
    std::string replies;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (char c : data) {
            const uint8_t byte = uint8_t(c);
            switch (m_state) {
            case State::Data:
                if (byte == IAC)
                    m_state = State::Iac;
                else
                    m_buffer.push_back(c);
                break;
            case State::Iac:
                if (byte == IAC) {
                    m_buffer.push_back(c);
                    m_state = State::Data;
                } else if (byte == WILL || byte == WONT || byte == DO || byte == DONT) {
                    m_optionCommand = byte;
                    m_state = State::Option;
                } else if (byte == SB) {
                    m_state = State::Sub;
                } else {
                    m_state = State::Data;
                }
                break;
            case State::Option: {
                m_state = State::Data;
                // 对端要我们开 / 它要开：支持的、还没同意过的回一次，不支持的拒绝
                const bool remoteSide = m_optionCommand == WILL || m_optionCommand == WONT;
                std::set<uint8_t>& agreed = remoteSide ? m_remote : m_local;
                if (m_optionCommand == WILL || m_optionCommand == DO) {
                    if (!supported(byte))
                        replies += std::string { char(IAC), char(remoteSide ? DONT : WONT), char(byte) };
                    else if (agreed.insert(byte).second)
                        replies += std::string { char(IAC), char(remoteSide ? DO : WILL), char(byte) };
                } else {
                    agreed.erase(byte);
                }
                break;
            }
            case State::Sub:
                // 对端的确认（波特率、控制线）：不用
                if (byte == IAC)
                    m_state = State::SubIac;
                break;
            case State::SubIac:
                m_state = byte == SE ? State::Data : State::Sub;
                break;
            }
        }
        if (m_buffer.size() > kKeepBytes) {
            const size_t drop = m_buffer.size() - kKeepBytes;
            m_buffer.erase(0, drop);
            m_base += drop;
        }
    }
    m_wake.notify_all();
    if (!replies.empty())
        send(replies);
}

void Monitor::onClosed()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_closed)
            return;
        m_closed = true;
        m_closeReason = "客户端那边关了（设备拔掉、用户点了断开，或客户端断开）";
    }
    m_wake.notify_all();
}

json Monitor::read(long long cursor, int waitMs, const std::string& until, size_t maxBytes, bool hex)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    unsigned long long from = cursor < 0 ? m_readCursor : (unsigned long long)cursor;
    const auto ready = [&] {
        const unsigned long long end = m_base + m_buffer.size();
        if (m_closed)
            return true;
        if (until.empty())
            return end > from;
        const unsigned long long start = from < m_base ? m_base : from;
        return m_buffer.find(until, size_t(start - m_base)) != std::string::npos;
    };
    m_wake.wait_for(lock, std::chrono::milliseconds(waitMs), ready);
    bool dropped = false;
    if (from < m_base) {
        dropped = true;
        from = m_base;
    }
    const unsigned long long end = m_base + m_buffer.size();
    if (from > end)
        from = end;
    size_t length = size_t(end - from);
    bool more = false;
    if (length > maxBytes) {
        length = maxBytes;
        more = true;
    }
    const std::string chunk = m_buffer.substr(size_t(from - m_base), length);
    const unsigned long long next = from + length;
    m_readCursor = next;
    json out { { "cursor", next }, { "bytes", length }, { "more", more }, { "closed", m_closed } };
    if (hex)
        out["hex"] = hexOf(chunk);
    else
        out["text"] = chunk;
    if (dropped)
        out["dropped"] = "游标之前的输出已经被挤掉（只留最近 512 KB）";
    if (!until.empty())
        out["found"] = chunk.find(until) != std::string::npos;
    if (m_closed)
        out["closeReason"] = m_closeReason;
    return out;
}

// MARK: - 发

void Monitor::send(const std::string& raw)
{
    if (!closed() && !m_label.empty())
        m_bridge.sendChannel(m_client, m_label, raw.data(), raw.size());
}

void Monitor::write(const std::string& data)
{
    if (closed())
        throw DeviceError { 21, "这个串口监视已经关了" };
    std::string escaped;
    escaped.reserve(data.size());
    for (char c : data) {
        escaped.push_back(c);
        if (uint8_t(c) == IAC)
            escaped.push_back(c);
    }
    send(escaped);
}

void Monitor::control(uint8_t value)
{
    send(subnegotiation(SET_CONTROL, std::string(1, char(value))));
}

void Monitor::setBaudRate(int baudRate)
{
    m_baudRate = baudRate;
    const uint32_t v = uint32_t(baudRate);
    send(subnegotiation(SET_BAUDRATE, std::string { char(v >> 24), char((v >> 16) & 0xff), char((v >> 8) & 0xff), char(v & 0xff) }));
}

void Monitor::reset(const std::string& mode)
{
    if (closed())
        throw DeviceError { 21, "这个串口监视已经关了" };
    using namespace std::chrono_literals;
    if (mode == "dtr") {
        // Arduino：DTR 置位的那一下经板上电容拉低 RESET
        control(DTR_OFF);
        control(RTS_OFF);
        std::this_thread::sleep_for(50ms);
        control(DTR_ON);
        std::this_thread::sleep_for(100ms);
        control(DTR_OFF);
        return;
    }
    // ESP（esptool 的 hard_reset）：DTR 放开（IO0 高，正常启动），RTS 拉 EN 低 100 ms 再放开
    control(DTR_OFF);
    control(RTS_ON);
    std::this_thread::sleep_for(100ms);
    control(RTS_OFF);
}

} // namespace devices
