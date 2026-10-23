// 见 Tools.h
#include "Tools.h"

#include "Log.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

namespace devices {

namespace fs = std::filesystem;

namespace {

/// 单个工具调用里最多等多久（Friday 那头 120 秒就算超时）
constexpr int kInlineWaitSeconds = 100;

json textResult(const std::string& text)
{
    return json { { "content", json::array({ json { { "type", "text" }, { "text", text } } }) } };
}

json errorResult(const std::string& text)
{
    json result = textResult(text);
    result["isError"] = true;
    return result;
}

std::string pretty(const json& value)
{
    return value.dump(2, ' ', false, json::error_handler_t::replace);
}

std::string base64(const std::string& data)
{
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < data.size(); i += 3) {
        const uint32_t v = (uint32_t(uint8_t(data[i])) << 16) | (uint32_t(uint8_t(data[i + 1])) << 8) | uint8_t(data[i + 2]);
        out += alphabet[(v >> 18) & 63];
        out += alphabet[(v >> 12) & 63];
        out += alphabet[(v >> 6) & 63];
        out += alphabet[v & 63];
    }
    if (i < data.size()) {
        uint32_t v = uint32_t(uint8_t(data[i])) << 16;
        if (i + 1 < data.size())
            v |= uint32_t(uint8_t(data[i + 1])) << 8;
        out += alphabet[(v >> 18) & 63];
        out += alphabet[(v >> 12) & 63];
        out += i + 1 < data.size() ? alphabet[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

/// 读服务端本机的文件（固件、要交给客户端工具的文件）；UTF-8 路径
std::string readFile(const std::string& path)
{
    const fs::path file = fs::u8path(path);
    std::ifstream in(file, std::ios::binary);
    if (!in)
        throw DeviceError { -32602, "读不了文件：" + path };
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (data.size() > size_t(32) << 20)
        throw DeviceError { -32602, "文件太大（超过 32 MB）：" + path };
    return data;
}

std::string lowerExtension(const std::string& path)
{
    std::string ext = fs::u8path(path).extension().u8string();
    for (char& c : ext)
        c = char(std::tolower(uint8_t(c)));
    return ext;
}

std::string requireString(const json& args, const char* name)
{
    if (!args.contains(name) || !args[name].is_string() || args[name].get<std::string>().empty())
        throw DeviceError { -32602, std::string("缺少参数 ") + name };
    return args[name].get<std::string>();
}

std::string optionalString(const json& args, const char* name)
{
    return args.contains(name) && args[name].is_string() ? args[name].get<std::string>() : std::string();
}

int optionalInt(const json& args, const char* name, int fallback)
{
    return args.contains(name) && args[name].is_number_integer() ? args[name].get<int>() : fallback;
}

json schema(json properties, json required = json::array())
{
    return json { { "type", "object" }, { "properties", std::move(properties) }, { "required", std::move(required) } };
}

const json kClientProperty = { { "type", "string" }, { "description", "哪台客户端（id 或名字里的一段）；不填用最近连上的那台" } };
const json kLocalPort = { { "type", "integer" }, { "description", "本机端口（不填系统分配）" } };

} // namespace

struct Tools::Job {
    std::string id;
    std::string title;
    bool done = false;
    bool ok = false;
    json result;
    std::string error;
    std::string progress;
};

Tools::Tools(Bridge& bridge, Ports& ports)
    : m_bridge(bridge)
    , m_ports(ports)
{
    m_bridge.setNotifyHandler([this](const std::string& client, const std::string& method, const json& params) {
        onNotify(client, method, params);
    });
}

Tools::~Tools()
{
    m_bridge.setNotifyHandler(nullptr);
}

json Tools::list() const
{
    json tools = json::array();
    const auto add = [&](const char* name, const char* description, json inputSchema, bool readOnly = false) {
        json tool { { "name", name }, { "description", description }, { "inputSchema", std::move(inputSchema) } };
        if (readOnly)
            tool["annotations"] = json { { "readOnlyHint", true } };
        tools.push_back(std::move(tool));
    };
    add("devices",
        "列出连着这台 Friday 的客户端（Friday Work 桌面客户端）和它们那台电脑上的设备：串口（ESP / Arduino / STM32 等开发板）、"
        "摄像头、安卓手机（adb）、iOS 设备、UF2 烧录盘。用别的工具前先看这个拿 device_id。",
        schema({ { "client", kClientProperty } }), true);
    add("serial_open",
        "把客户端的一个串口开成本机端口：rfc2217://127.0.0.1:端口（带波特率、DTR/RTS，esptool / pyserial / PlatformIO / idf.py 直接用），"
        "mode=raw 时是 socket://（只有字节）。每次有程序连上来，客户端那头会话一个会话（第一次会问那边的人）。",
        schema({ { "device_id", { { "type", "string" }, { "description", "devices 给的 serial:… 标识" } } },
                 { "mode", { { "type", "string" }, { "enum", { "rfc2217", "raw" } } } },
                 { "baud_rate", { { "type", "integer" }, { "description", "初始波特率，默认 115200" } } },
                 { "port", kLocalPort },
                 { "client", kClientProperty } },
               { "device_id" }));
    add("tunnel_open",
        "TCP 隧道：本机端口 → 客户端那台电脑能连到的 host:port（局域网里的树莓派 SSH、网络摄像头 RTSP、Modbus TCP，"
        "或客户端本机的 openocd 3333 / J-Link GDB Server）。默认只放行私有网段和客户端本机。",
        schema({ { "host", { { "type", "string" }, { "description", "客户端那边看到的地址，如 192.168.1.50、127.0.0.1" } } },
                 { "port", { { "type", "integer" }, { "description", "那边的端口" } } },
                 { "local_port", kLocalPort },
                 { "client", kClientProperty } },
               { "host", "port" }));
    add("android_attach",
        "把客户端插着的一台安卓手机交给本机的 adb：开一个端口代理客户端的 adb server（只放行授权过的这台手机），"
        "之后设 ADB_SERVER_SOCKET=tcp:127.0.0.1:端口 再用 adb（shell / install / logcat / screencap / input）。"
        "两边 adb 版本要一致（不然本机的 adb 会想重启对方的 server，会被拒绝）。",
        schema({ { "device_id", { { "type", "string" }, { "description", "devices 给的 android:… 标识；只有一台时可不填" } } },
                 { "port", kLocalPort },
                 { "client", kClientProperty } }));
    add("ios_attach",
        "把客户端的 usbmuxd（iOS 设备服务，Windows 上要装 Apple 设备 / iTunes）开成本机端口，"
        "之后设 USBMUXD_SOCKET_ADDRESS=127.0.0.1:端口 用 libimobiledevice / pymobiledevice3。iOS 17 起的开发者服务要另起隧道，编译签名要 Mac。",
        schema({ { "port", kLocalPort }, { "client", kClientProperty } }));
    add("serial_monitor",
        "开一个串口监视：客户端那个串口的输出一直缓存在这里（最近 512 KB），之后用 serial_read 取新输出、serial_write 发数据、"
        "serial_reset 复位板子看启动日志，用完 serial_close。不用在服务端写脚本、装 pyserial。同一个口同时只能开一个（和 serial_open 的端口互斥）；"
        "这个口烧录时监视不断，烧完接着收新固件的输出。",
        schema({ { "device_id", { { "type", "string" }, { "description", "devices 给的 serial:… 标识" } } },
                 { "baud_rate", { { "type", "integer" }, { "description", "默认 115200" } } },
                 { "client", kClientProperty } },
               { "device_id" }));
    add("serial_read",
        "取串口监视的新输出（从上次读到的地方接着读，或从给的 cursor 读）。wait_s 秒内没有新输出就回空；给 until 时等到输出里出现这段文字为止"
        "（如 \"ready\"、\"WiFi connected\"）。回 text（或 hex）和下一次的 cursor。",
        schema({ { "monitor_id", { { "type", "integer" } } },
                 { "wait_s", { { "type", "number" }, { "description", "最多等多少秒（默认 2，最多 90）" } } },
                 { "until", { { "type", "string" }, { "description", "等到输出里出现这段文字" } } },
                 { "cursor", { { "type", "integer" }, { "description", "从这个位置读（serial_read 回的 cursor；0 = 从开头留着的读）" } } },
                 { "max_bytes", { { "type", "integer" }, { "description", "最多回多少字节（默认 16384）" } } },
                 { "hex", { { "type", "boolean" }, { "description", "按十六进制回（二进制协议用）" } } } },
               { "monitor_id" }));
    add("serial_write",
        "往串口发数据：text 原样发，line 发完补换行（newline 默认 \\r\\n），hex 发十六进制字节（如 \"7e 01 00\"）。",
        schema({ { "monitor_id", { { "type", "integer" } } },
                 { "text", { { "type", "string" } } },
                 { "line", { { "type", "string" } } },
                 { "newline", { { "type", "string" } } },
                 { "hex", { { "type", "string" } } } },
               { "monitor_id" }));
    add("serial_reset",
        "复位板子（之后用 serial_read 看启动日志）：mode=rts 是 ESP 的复位（RTS 拉 EN 低 100 ms，DTR 放开 = 正常启动，默认），"
        "mode=dtr 是 Arduino 的复位（DTR 拉一下）。",
        schema({ { "monitor_id", { { "type", "integer" } } }, { "mode", { { "type", "string" }, { "enum", { "rts", "dtr" } } } } },
               { "monitor_id" }));
    add("serial_close", "关掉一个串口监视（串口还给客户端）。", schema({ { "monitor_id", { { "type", "integer" } } } }, { "monitor_id" }));
    add("ports", "列出开着的本机端口（串口、隧道、adb、usbmux）、串口监视、上面的连接数和最近的错误，以及后台跑着的客户端工具。",
        schema(json::object()), true);
    add("port_close", "关一个本机端口（ports 给的 id）和它上面的连接。",
        schema({ { "id", { { "type", "integer" } } } }, { "id" }));
    add("flash",
        "在客户端本机烧录开发板（固件是这台机器上的文件）：method=esp（ESP32 / ESP8266，esptool-js，可多个文件带地址）、"
        "uf2（树莓派 Pico 等：按住 BOOTSEL 插上出现的盘，device_id 是 uf2:…）、stk500（Arduino Uno / Nano，.hex 或 .bin）、"
        "stm32（STM32 串口引导，BOOT0 要拉高或用 reset=dtr-rts）。不填 method：uf2:… 用 uf2，其余用 esp。"
        "要很久的会先回 job_id，用 job_wait 等。",
        schema({ { "device_id", { { "type", "string" } } },
                 { "method", { { "type", "string" }, { "enum", { "esp", "uf2", "stk500", "stm32" } } } },
                 { "files",
                   { { "type", "array" },
                     { "items", schema({ { "path", { { "type", "string" }, { "description", "这台机器上的文件路径" } } },
                                         { "address", { { "type", { "string", "integer" } }, { "description", "ESP / STM32 的烧录地址，如 0x10000" } } } },
                                       { "path" }) } } },
                 { "baud_rate", { { "type", "integer" } } },
                 { "erase_all", { { "type", "boolean" } } },
                 { "flash_mode", { { "type", "string" } } },
                 { "flash_freq", { { "type", "string" } } },
                 { "flash_size", { { "type", "string" } } },
                 { "reset", { { "type", "string" }, { "enum", { "none", "dtr-rts" } }, { "description", "stm32：用 DTR/RTS 拉 BOOT0 / 复位" } } },
                 { "client", kClientProperty } },
               { "device_id", "files" }));
    add("esp_info", "复位 ESP 进下载模式读芯片型号、MAC、flash 大小（读完复位回应用）。",
        schema({ { "device_id", { { "type", "string" } } }, { "client", kClientProperty } }, { "device_id" }));
    add("camera_snapshot", "用客户端的摄像头拍一张照片（JPEG），直接看到图。",
        schema({ { "device_id", { { "type", "string" }, { "description", "camera:…；不填用默认摄像头" } } },
                 { "width", { { "type", "integer" } } },
                 { "height", { { "type", "integer" } } },
                 { "client", kClientProperty } }));
    add("run_tool",
        "在客户端本机跑一个烧录 / 调试工具（白名单：avrdude、openocd、st-flash、st-info、stm32flash、STM32_Programmer_CLI、picotool、"
        "dfu-util、nrfjprog、JLinkExe、esptool），每次都会问那边的人。files 里的文件先传过去，args 里写 {file:名字} 换成那边的路径。"
        "background=true 时不等它结束（如 openocd 当 GDB 服务，再用 tunnel_open 连 127.0.0.1:3333），用 stop_tool 停。",
        schema({ { "tool", { { "type", "string" } } },
                 { "args", { { "type", "array" }, { "items", { { "type", "string" } } } } },
                 { "files",
                   { { "type", "array" },
                     { "items", schema({ { "path", { { "type", "string" } } }, { "name", { { "type", "string" } } } }, { "path" }) } } },
                 { "background", { { "type", "boolean" } } },
                 { "timeout_s", { { "type", "integer" }, { "description", "前台运行的上限，默认 300 秒" } } },
                 { "client", kClientProperty } },
               { "tool" }));
    add("stop_tool", "停掉 run_tool 后台跑着的工具。",
        schema({ { "run_id", { { "type", "string" } } }, { "client", kClientProperty } }, { "run_id" }));
    add("job_wait", "等一个后台任务（flash / run_tool 回的 job_id），最多等 timeout_s 秒（默认 90，最多 100）。",
        schema({ { "job_id", { { "type", "string" } } }, { "timeout_s", { { "type", "integer" } } } }, { "job_id" }));
    return tools;
}

json Tools::call(const std::string& name, const json& arguments)
{
    const json args = arguments.is_object() ? arguments : json::object();
    try {
        if (name == "devices")
            return devicesTool(args);
        if (name == "serial_open")
            return serialOpen(args);
        if (name == "tunnel_open")
            return tunnelOpen(args);
        if (name == "android_attach")
            return androidAttach(args);
        if (name == "ios_attach")
            return iosAttach(args);
        if (name == "ports")
            return portsTool(args);
        if (name == "port_close")
            return portClose(args);
        if (name == "flash")
            return flash(args);
        if (name == "esp_info")
            return espInfo(args);
        if (name == "camera_snapshot")
            return cameraSnapshot(args);
        if (name == "run_tool")
            return runTool(args);
        if (name == "stop_tool")
            return stopTool(args);
        if (name == "job_wait")
            return jobWait(args);
        if (name == "serial_monitor")
            return serialMonitor(args);
        if (name == "serial_read")
            return serialRead(args);
        if (name == "serial_write")
            return serialWrite(args);
        if (name == "serial_reset")
            return serialReset(args);
        if (name == "serial_close")
            return serialClose(args);
        return errorResult("没有这个工具：" + name);
    } catch (const DeviceError& error) {
        return errorResult(error.message + "（错误码 " + std::to_string(error.code) + "）");
    } catch (const std::exception& error) {
        return errorResult(std::string("出错了：") + error.what());
    }
}

// MARK: - 设备

json Tools::devicesTool(const json& args)
{
    const std::string hint = optionalString(args, "client");
    std::vector<ClientInfo> clients;
    if (hint.empty())
        clients = m_bridge.clients();
    else
        clients.push_back(m_bridge.pickClient(hint));
    if (clients.empty())
        m_bridge.pickClient("");   // 抛「没有连着的客户端」
    json out = json::array();
    for (const ClientInfo& client : clients) {
        json entry { { "client", client.name }, { "clientId", client.id }, { "kinds", client.kinds } };
        try {
            entry["devices"] = m_bridge.request(client.id, "device.list", json::object(), 20000).value("devices", json::array());
        } catch (const DeviceError& error) {
            entry["error"] = error.message;
        }
        out.push_back(entry);
    }
    return textResult(pretty(out));
}

std::string Tools::deviceLabel(const ClientInfo& client, const std::string& deviceId)
{
    try {
        for (const json& device : m_bridge.request(client.id, "device.list", json::object(), 15000).value("devices", json::array())) {
            if (device.value("id", "") == deviceId)
                return device.value("label", deviceId);
        }
    } catch (const DeviceError&) {
    }
    return deviceId;
}

json Tools::serialOpen(const json& args)
{
    const ClientInfo client = m_bridge.pickClient(optionalString(args, "client"));
    const std::string deviceId = requireString(args, "device_id");
    const std::string mode = optionalString(args, "mode").empty() ? "rfc2217" : optionalString(args, "mode");
    PortSpec spec;
    spec.kind = "serial";
    spec.client = client.id;
    spec.clientName = client.name;
    spec.openMethod = "device.serial.open";
    spec.openParams = json { { "deviceId", deviceId }, { "mode", mode }, { "baudRate", optionalInt(args, "baud_rate", 115200) } };
    spec.target = deviceLabel(client, deviceId);
    spec.port = optionalInt(args, "port", 0);
    const json port = m_ports.open(spec);
    const std::string url = port.value("url", "");
    std::string text = "已开：" + url + "（" + spec.target + "，在 " + client.name + " 上）\n";
    if (mode == "rfc2217")
        text += "用法：esptool --port " + url + " chip_id；pio run -t upload --upload-port " + url + "；python -m serial.tools.miniterm " + url +
                " 115200\n";
    text += "第一次连上时客户端那头会问拿着设备的人。用完 port_close " + std::to_string(port.value("id", 0)) + "。\n" + pretty(port);
    return textResult(text);
}

json Tools::tunnelOpen(const json& args)
{
    const ClientInfo client = m_bridge.pickClient(optionalString(args, "client"));
    const std::string host = requireString(args, "host");
    const int remotePort = optionalInt(args, "port", 0);
    if (remotePort <= 0 || remotePort > 65535)
        throw DeviceError { -32602, "port 要是 1..65535" };
    PortSpec spec;
    spec.kind = "tcp";
    spec.client = client.id;
    spec.clientName = client.name;
    spec.openMethod = "device.tcp.open";
    spec.openParams = json { { "host", host }, { "port", remotePort } };
    spec.target = host + ":" + std::to_string(remotePort);
    spec.port = optionalInt(args, "local_port", 0);
    const json port = m_ports.open(spec);
    return textResult("已开：127.0.0.1:" + std::to_string(port.value("port", 0)) + " → " + spec.target + "（经 " + client.name +
                      "）。例：ssh -p " + std::to_string(port.value("port", 0)) + " pi@127.0.0.1\n" + pretty(port));
}

json Tools::androidAttach(const json& args)
{
    const ClientInfo client = m_bridge.pickClient(optionalString(args, "client"));
    std::string deviceId = optionalString(args, "device_id");
    std::string label = deviceId;
    if (deviceId.empty()) {
        std::vector<json> phones;
        for (const json& device : m_bridge.request(client.id, "device.list", json::object(), 20000).value("devices", json::array())) {
            if (device.value("kind", "") == "android")
                phones.push_back(device);
        }
        if (phones.empty())
            throw DeviceError { 21, client.name + " 上没有连着的安卓手机（要打开 USB 调试、在电脑上点允许，客户端那台要装 platform-tools）" };
        if (phones.size() > 1)
            throw DeviceError { -32602, "有好几台安卓手机，用 device_id 指定：" + pretty(phones) };
        deviceId = phones.front().value("id", "");
        label = phones.front().value("label", deviceId);
    } else {
        label = deviceLabel(client, deviceId);
    }
    PortSpec spec;
    spec.kind = "adb";
    spec.client = client.id;
    spec.clientName = client.name;
    spec.openMethod = "device.tcp.open";
    spec.openParams = json { { "service", "adb" }, { "deviceId", deviceId } };
    spec.target = label;
    spec.port = optionalInt(args, "port", 0);
    const json port = m_ports.open(spec);
    const std::string socket = "tcp:127.0.0.1:" + std::to_string(port.value("port", 0));
    return textResult("已开：" + label + "（" + client.name + "）。\n用法：ADB_SERVER_SOCKET=" + socket +
                      " adb devices（Windows PowerShell：$env:ADB_SERVER_SOCKET='" + socket + "'）\n" + pretty(port));
}

json Tools::iosAttach(const json& args)
{
    const ClientInfo client = m_bridge.pickClient(optionalString(args, "client"));
    PortSpec spec;
    spec.kind = "usbmux";
    spec.client = client.id;
    spec.clientName = client.name;
    spec.openMethod = "device.tcp.open";
    spec.openParams = json { { "service", "usbmux" } };
    spec.target = "usbmuxd（" + client.name + " 上所有的 iOS 设备）";
    spec.port = optionalInt(args, "port", 0);
    const json port = m_ports.open(spec);
    const std::string address = "127.0.0.1:" + std::to_string(port.value("port", 0));
    return textResult("已开：" + address + " → " + client.name + " 的 usbmuxd。\n用法：USBMUXD_SOCKET_ADDRESS=" + address +
                      " idevice_id -l；ideviceinfo；ideviceinstaller -i x.ipa\n" + pretty(port));
}

json Tools::portsTool(const json&)
{
    json out { { "ports", m_ports.list() } };
    std::lock_guard<std::mutex> lock(m_mutex);
    json monitors = json::array();
    for (const auto& [id, monitor] : m_monitors)
        monitors.push_back(monitor->describe());
    out["monitors"] = monitors;
    json runs = json::array();
    for (const auto& [key, run] : m_toolRuns)
        runs.push_back(run);
    out["toolRuns"] = runs;
    return textResult(pretty(out));
}

json Tools::portClose(const json& args)
{
    const int id = optionalInt(args, "id", 0);
    if (!m_ports.close(id))
        throw DeviceError { -32602, "没有这个端口：" + std::to_string(id) };
    return textResult("端口 " + std::to_string(id) + " 已关");
}

// MARK: - 串口监视

std::shared_ptr<Monitor> Tools::monitor(const json& args)
{
    const int id = optionalInt(args, "monitor_id", 0);
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto it = m_monitors.find(id);
    if (it == m_monitors.end())
        throw DeviceError { -32602, "没有这个串口监视：" + std::to_string(id) + "（用 serial_monitor 开，ports 里看开着的）" };
    return it->second;
}

json Tools::serialMonitor(const json& args)
{
    const ClientInfo client = m_bridge.pickClient(optionalString(args, "client"));
    const std::string deviceId = requireString(args, "device_id");
    const int baudRate = std::clamp(optionalInt(args, "baud_rate", 115200), 50, 16000000);
    int id = 0;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        id = ++m_nextMonitor;
    }
    auto session = std::make_shared<Monitor>(m_bridge, id, client.id, client.name, deviceId, deviceLabel(client, deviceId));
    session->open(baudRate);
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_monitors[id] = session;
    }
    return textResult("已开串口监视 monitor_id=" + std::to_string(id) + "（" + session->describe().value("target", deviceId) + "，" +
                      std::to_string(baudRate) + " 波特）。用 serial_read 取输出、serial_reset 复位看启动日志、serial_write 发数据。");
}

json Tools::serialRead(const json& args)
{
    const auto session = monitor(args);
    const double waitSeconds = args.contains("wait_s") && args["wait_s"].is_number() ? args["wait_s"].get<double>() : 2.0;
    const int waitMs = int(std::clamp(waitSeconds, 0.0, 90.0) * 1000);
    const long long cursor = args.contains("cursor") && args["cursor"].is_number_integer() ? args["cursor"].get<long long>() : -1;
    const size_t maxBytes = size_t(std::clamp(optionalInt(args, "max_bytes", 16384), 64, 262144));
    const json out = session->read(cursor, waitMs, optionalString(args, "until"), maxBytes, args.value("hex", false));
    std::string head = "cursor=" + std::to_string(out.value("cursor", 0ULL)) + "，" + std::to_string(out.value("bytes", 0ULL)) + " 字节";
    if (out.value("more", false))
        head += "，后面还有（再读一次）";
    if (out.contains("found"))
        head += out.value("found", false) ? "，等到了" : "，没等到";
    if (out.value("closed", false))
        head += "，会话已关：" + out.value("closeReason", std::string());
    if (out.contains("dropped"))
        head += "，" + out.value("dropped", std::string());
    return textResult(head + "\n" + (out.contains("hex") ? out.value("hex", std::string()) : out.value("text", std::string())));
}

json Tools::serialWrite(const json& args)
{
    const auto session = monitor(args);
    std::string data;
    if (args.contains("hex") && args["hex"].is_string()) {
        std::string digits;
        for (char c : args["hex"].get<std::string>()) {
            if (std::isxdigit(uint8_t(c)))
                digits.push_back(c);
            else if (c != ' ' && c != ',' && c != ':')
                throw DeviceError { -32602, "hex 只能是十六进制数字（可以用空格隔开）" };
        }
        if (digits.size() % 2 != 0)
            throw DeviceError { -32602, "hex 的位数要是偶数" };
        for (size_t i = 0; i < digits.size(); i += 2)
            data.push_back(char(std::stoi(digits.substr(i, 2), nullptr, 16)));
    } else if (args.contains("line") && args["line"].is_string()) {
        data = args["line"].get<std::string>() + (args.contains("newline") && args["newline"].is_string() ? args["newline"].get<std::string>() : "\r\n");
    } else if (args.contains("text") && args["text"].is_string()) {
        data = args["text"].get<std::string>();
    } else {
        throw DeviceError { -32602, "要给 text、line 或 hex" };
    }
    session->write(data);
    return textResult("已发 " + std::to_string(data.size()) + " 字节（用 serial_read 看回应）");
}

json Tools::serialReset(const json& args)
{
    const auto session = monitor(args);
    const std::string mode = optionalString(args, "mode").empty() ? "rts" : optionalString(args, "mode");
    if (mode != "rts" && mode != "dtr")
        throw DeviceError { -32602, "mode 只能是 rts（ESP）/ dtr（Arduino）" };
    session->reset(mode);
    return textResult("已复位（" + mode + "）。用 serial_read（可带 until）看启动日志。");
}

json Tools::serialClose(const json& args)
{
    const auto session = monitor(args);
    session->close();
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_monitors.erase(session->id());
    }
    return textResult("串口监视 " + std::to_string(session->id()) + " 已关");
}

// MARK: - 烧录、芯片、摄像头

json Tools::flash(const json& args)
{
    const ClientInfo client = m_bridge.pickClient(optionalString(args, "client"));
    const std::string deviceId = requireString(args, "device_id");
    std::string method = optionalString(args, "method");
    if (method.empty())
        method = deviceId.rfind("uf2:", 0) == 0 ? "uf2" : "esp";
    if (!args.contains("files") || !args["files"].is_array() || args["files"].empty())
        throw DeviceError { -32602, "files 至少要一个 {path}" };
    json files = json::array();
    size_t total = 0;
    for (const json& file : args["files"]) {
        const std::string path = requireString(file, "path");
        const std::string data = readFile(path);
        total += data.size();
        json entry { { "data", base64(data) }, { "path", path } };
        if (file.contains("address"))
            entry["address"] = file["address"];
        else if (method == "esp")
            throw DeviceError { -32602, "esp 的每个文件要带 address（如 bootloader 0x0 / 0x1000、partitions 0x8000、应用 0x10000）" };
        files.push_back(entry);
    }
    const std::string title = method + " 烧录 " + std::to_string(files.size()) + " 个文件（" + std::to_string(total / 1024) + " KB）";
    return startJob(title, kInlineWaitSeconds, [this, client, deviceId, method, files, args](Job& job) {
        json params { { "deviceId", deviceId }, { "progressToken", job.id } };
        std::string requestMethod = "device.esp.flash";
        if (method == "esp") {
            json list = json::array();
            for (const json& file : files)
                list.push_back(json { { "address", file["address"] }, { "data", file["data"] } });
            params["files"] = list;
            for (const auto& [key, name] : { std::pair { "baud_rate", "baudRate" }, std::pair { "erase_all", "eraseAll" },
                                             std::pair { "flash_mode", "flashMode" }, std::pair { "flash_freq", "flashFreq" },
                                             std::pair { "flash_size", "flashSize" } }) {
                if (args.contains(key))
                    params[name] = args[key];
            }
        } else {
            if (files.size() != 1)
                throw DeviceError { -32602, method + " 一次烧一个文件" };
            requestMethod = "device.flash";
            params["method"] = method;
            params["data"] = files[0]["data"];
            params["format"] = lowerExtension(files[0].value("path", "")) == ".hex" ? "hex" : "bin";
            if (files[0].contains("address"))
                params["address"] = files[0]["address"];
            if (args.contains("baud_rate"))
                params["baudRate"] = args["baud_rate"];
            if (args.contains("reset"))
                params["reset"] = args["reset"];
        }
        return m_bridge.request(client.id, requestMethod, params, 15 * 60 * 1000);
    });
}

json Tools::espInfo(const json& args)
{
    const ClientInfo client = m_bridge.pickClient(optionalString(args, "client"));
    const json result = m_bridge.request(client.id, "device.esp.info", json { { "deviceId", requireString(args, "device_id") } }, 180000);
    return textResult(pretty(result));
}

json Tools::cameraSnapshot(const json& args)
{
    const ClientInfo client = m_bridge.pickClient(optionalString(args, "client"));
    json params = json::object();
    if (!optionalString(args, "device_id").empty())
        params["deviceId"] = optionalString(args, "device_id");
    for (const char* key : { "width", "height" }) {
        if (args.contains(key))
            params[key] = args[key];
    }
    const json shot = m_bridge.request(client.id, "device.camera.snapshot", params, 120000);
    json content = json::array();
    content.push_back(json { { "type", "image" }, { "data", shot.value("data", "") }, { "mimeType", shot.value("mime", "image/jpeg") } });
    content.push_back(json { { "type", "text" },
                             { "text", shot.value("label", std::string("摄像头")) + "，" + std::to_string(shot.value("width", 0)) + "×" +
                                           std::to_string(shot.value("height", 0)) + "，" + shot.value("takenAt", "") } });
    return json { { "content", content } };
}

// MARK: - 客户端工具

json Tools::runTool(const json& args)
{
    const ClientInfo client = m_bridge.pickClient(optionalString(args, "client"));
    json params { { "tool", requireString(args, "tool") }, { "args", args.value("args", json::array()) } };
    json files = json::array();
    for (const json& file : args.value("files", json::array())) {
        const std::string path = requireString(file, "path");
        const std::string name = optionalString(file, "name").empty() ? fs::u8path(path).filename().u8string() : optionalString(file, "name");
        files.push_back(json { { "name", name }, { "data", base64(readFile(path)) } });
    }
    params["files"] = files;
    const bool background = args.value("background", false);
    params["background"] = background;
    const int timeoutSeconds = std::clamp(optionalInt(args, "timeout_s", 300), 5, 3600);
    params["timeoutMs"] = timeoutSeconds * 1000;
    if (background) {
        const json result = m_bridge.request(client.id, "device.tool.run", params, 120000);
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_toolRuns[client.id + "/" + result.value("runId", "")] =
                json { { "client", client.name }, { "runId", result.value("runId", "") }, { "tool", params["tool"] }, { "running", true } };
        }
        return textResult("已在 " + client.name + " 上后台运行，run_id=" + result.value("runId", "") + "\n" + pretty(result));
    }
    const std::string title = "在 " + client.name + " 上运行 " + params["tool"].get<std::string>();
    return startJob(title, kInlineWaitSeconds, [this, client, params, timeoutSeconds](Job&) {
        // 等人点授权（60 秒）+ 工具本身
        return m_bridge.request(client.id, "device.tool.run", params, (timeoutSeconds + 90) * 1000);
    });
}

json Tools::stopTool(const json& args)
{
    const ClientInfo client = m_bridge.pickClient(optionalString(args, "client"));
    const std::string runId = requireString(args, "run_id");
    const json result = m_bridge.request(client.id, "device.tool.stop", json { { "runId", runId } }, 20000);
    return textResult("已停：" + runId + "\n" + pretty(result));
}

// MARK: - 后台任务

json Tools::startJob(const std::string& title, int waitSeconds, std::function<json(Job&)> work)
{
    auto job = std::make_shared<Job>();
    job->title = title;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        job->id = "job-" + std::to_string(++m_nextJob);
        m_jobs[job->id] = job;
    }
    std::thread([this, job, work] {
        json result;
        std::string error;
        bool ok = false;
        try {
            result = work(*job);
            ok = true;
        } catch (const DeviceError& e) {
            error = e.message + "（错误码 " + std::to_string(e.code) + "）";
        } catch (const std::exception& e) {
            error = e.what();
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            job->done = true;
            job->ok = ok;
            job->result = result;
            job->error = error;
        }
        m_wake.notify_all();
    }).detach();
    return waitJob(job, waitSeconds);
}

json Tools::waitJob(const std::shared_ptr<Job>& job, int waitSeconds)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_wake.wait_for(lock, std::chrono::seconds(waitSeconds), [&] { return job->done; });
    if (!job->done) {
        return textResult(job->title + "：还在进行" + (job->progress.empty() ? std::string() : "（" + job->progress + "）") +
                          "。用 job_wait job_id=" + job->id + " 接着等。");
    }
    if (!job->ok)
        return errorResult(job->title + " 失败：" + job->error);
    return textResult(job->title + " 完成\n" + pretty(job->result));
}

json Tools::jobWait(const json& args)
{
    const std::string id = requireString(args, "job_id");
    std::shared_ptr<Job> job;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_jobs.find(id);
        if (it == m_jobs.end())
            throw DeviceError { -32602, "没有这个任务：" + id };
        job = it->second;
    }
    return waitJob(job, std::clamp(optionalInt(args, "timeout_s", 90), 1, kInlineWaitSeconds));
}

void Tools::onNotify(const std::string& client, const std::string& method, const json& params)
{
    if (method == "device.esp.progress" || method == "device.flash.progress") {
        const std::string token = params.value("token", "");
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_jobs.find(token);
        if (it == m_jobs.end())
            return;
        const long long written = params.value("written", 0LL);
        const long long total = params.value("total", 0LL);
        it->second->progress = "第 " + std::to_string(params.value("fileIndex", 0) + 1) + " 个文件 " +
                               (total > 0 ? std::to_string(written * 100 / total) + "%" : std::to_string(written) + " 字节");
        return;
    }
    if (method == "device.tool.exited") {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_toolRuns.find(client + "/" + params.value("runId", ""));
        if (it == m_toolRuns.end())
            return;
        it->second["running"] = false;
        it->second["exitCode"] = params.value("exitCode", -1);
        it->second["output"] = params.value("output", "");
    }
}

} // namespace devices
