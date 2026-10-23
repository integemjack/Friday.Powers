// friday-devices：Friday 的客户端设备 Power（docs/devices-protocol.md）。
//
// 远程客户端（Friday Work 桌面客户端）那台电脑上的硬件——开发板串口、摄像头、安卓 / iOS 手机、局域网设备——
// 经 Friday 的工作通道开成这台机器上 127.0.0.1 的端口，再给智能体一组 MCP 工具（烧录、拍照、adb、隧道……）。
// 不用显卡、不要模型：一个小的可执行文件。
//
//   friday-devices serve [--host 127.0.0.1] [--port 0] [--token T] [--parent-stdin] [--verbose]
//   friday-devices info
//   friday-devices --version
//
// serve 就绪后在 stdout 打一行 {"event":"listening","url":"http://127.0.0.1:端口"}；--parent-stdin 时 stdin 关了就退出。日志都在 stderr。
#include "Log.h"
#include "Server.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

#ifndef DEVICES_VERSION
#define DEVICES_VERSION "0.0.0"
#endif

using devices::json;

namespace {

struct Options {
    std::string command;
    std::string host = "127.0.0.1", token;
    int port = 0;
    bool parentStdin = false, verbose = false;
};

void usage()
{
    std::fprintf(stderr,
                 "friday-devices %s —— Friday 的客户端设备（远程客户端那台电脑上的串口、摄像头、手机、局域网设备）\n\n"
                 "  friday-devices serve [--host 127.0.0.1] [--port 0] [--token T] [--parent-stdin] [--verbose]\n"
                 "  friday-devices info\n"
                 "  friday-devices --version\n",
                 DEVICES_VERSION);
}

bool parse(int argc, char** argv, Options& o)
{
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&](std::string& v) {
            if (i + 1 >= argc)
                return false;
            v = argv[++i];
            return true;
        };
        std::string v;
        if (a == "--version") {
            o.command = "version";
        } else if (a == "--host") {
            if (!next(o.host))
                return false;
        } else if (a == "--token") {
            if (!next(o.token))
                return false;
        } else if (a == "--port") {
            if (!next(v))
                return false;
            o.port = std::atoi(v.c_str());
        } else if (a == "--parent-stdin") {
            o.parentStdin = true;
        } else if (a == "--verbose" || a == "-v") {
            o.verbose = true;
        } else if (a == "--data-dir" || a == "--device") {
            // 宿主协议里的通用参数：这个 Power 用不上
            if (!next(v))
                return false;
        } else if (a == "-h" || a == "--help") {
            return false;
        } else if (o.command.empty() && a.rfind("--", 0) != 0) {
            o.command = a;
        } else {
            std::fprintf(stderr, "不认识的参数：%s\n", a.c_str());
            return false;
        }
    }
    return !o.command.empty();
}

std::atomic<bool> g_stop { false };
std::mutex g_stopMutex;
std::condition_variable g_stopWake;

void requestStop()
{
    g_stop = true;
    g_stopWake.notify_all();
}

} // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    std::vector<std::string> utf8Args;
    std::vector<char*> utf8Argv;
    if (int count = 0; LPWSTR* wide = CommandLineToArgvW(GetCommandLineW(), &count)) {
        for (int i = 0; i < count; ++i) {
            const int size = WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, nullptr, 0, nullptr, nullptr);
            std::string s(size_t(std::max(size, 1)) - 1, '\0');
            WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, s.data(), size, nullptr, nullptr);
            utf8Args.push_back(std::move(s));
        }
        LocalFree(wide);
        for (std::string& s : utf8Args)
            utf8Argv.push_back(s.data());
        argc = int(utf8Argv.size());
        argv = utf8Argv.data();
    }
    SetConsoleOutputCP(CP_UTF8);
#else
    std::signal(SIGPIPE, SIG_IGN);
#endif
    Options o;
    if (!parse(argc, argv, o)) {
        usage();
        return 2;
    }
    if (o.verbose)
        devices::logLevel() = devices::LogLevel::Debug;
    if (o.command == "version") {
        std::printf("friday-devices %s\n", DEVICES_VERSION);
        return 0;
    }
    if (o.command == "info") {
        std::printf("%s\n", json { { "id", "devices" }, { "version", DEVICES_VERSION }, { "capabilities", json::array({ "clientDevices" }) } }
                                .dump(2)
                                .c_str());
        return 0;
    }
    if (o.command != "serve") {
        usage();
        return 2;
    }

    devices::ServerOptions options;
    options.host = o.host;
    options.port = o.port;
    options.token = o.token;
    devices::Server server(options);
    std::string error;
    if (!server.start(&error)) {
        DLOG_ERROR("%s", error.c_str());
        return 6;
    }
    DLOG_INFO("friday-devices %s 在 %s 上听", DEVICES_VERSION, server.url().c_str());
    std::printf("%s\n", json { { "event", "listening" }, { "url", server.url() }, { "version", DEVICES_VERSION } }.dump().c_str());
    std::fflush(stdout);

    std::signal(SIGINT, [](int) { g_stop = true; });
    std::signal(SIGTERM, [](int) { g_stop = true; });
    if (o.parentStdin) {
        std::thread([] {
            char buffer[256];
            while (std::fread(buffer, 1, sizeof(buffer), stdin) > 0) {
            }
            DLOG_INFO("宿主关了 stdin，退出");
            requestStop();
        }).detach();
    }
    {
        std::unique_lock<std::mutex> lock(g_stopMutex);
        while (!g_stop.load())
            g_stopWake.wait_for(lock, std::chrono::milliseconds(200));
    }
    server.stop();
    DLOG_INFO("已退出");
    return 0;
}
