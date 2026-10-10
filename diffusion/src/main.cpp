// friday-diffusion：Friday 的本地生图 / 生视频 Power（stable-diffusion.cpp）。
//
//   friday-diffusion serve [--host 127.0.0.1] [--port 0] [--token T] [--parent-stdin] [--idle 600] [--manifest power.json] [--verbose]
//   friday-diffusion info
//   friday-diffusion --version
//
// serve 就绪后在 stdout 打一行 {"event":"listening","url":"http://127.0.0.1:端口"}；--parent-stdin 时 stdin 关了就退出。
// 日志都在 stderr。模型家族的缺省参数从可执行文件旁边的 power.json（families）读（docs/diffusion-protocol.md）。
#include "Generator.h"
#include "Log.h"
#include "Server.h"

#include "stable-diffusion.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#ifndef DIFFUSION_VERSION
#define DIFFUSION_VERSION "0.0.0"
#endif

namespace fs = std::filesystem;
using diffusion::json;

namespace {

struct Options {
    std::string command;
    std::string host = "127.0.0.1", token, manifest;
    int port = 0, idle = 600;
    bool parentStdin = false, verbose = false;
};

void usage()
{
    std::fprintf(stderr,
                 "friday-diffusion %s —— Friday 的本地生图 / 生视频（stable-diffusion.cpp %s）\n\n"
                 "  friday-diffusion serve [--host 127.0.0.1] [--port 0] [--token T] [--parent-stdin]\n"
                 "                         [--idle 600]（空闲多少秒释放模型，0 = 不释放）[--manifest power.json] [--verbose]\n"
                 "  friday-diffusion info\n"
                 "  friday-diffusion --version\n",
                 DIFFUSION_VERSION, DIFFUSION_SD_TAG);
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
            if (!next(o.host)) return false;
        } else if (a == "--token") {
            if (!next(o.token)) return false;
        } else if (a == "--manifest") {
            if (!next(o.manifest)) return false;
        } else if (a == "--port") {
            if (!next(v)) return false;
            o.port = std::atoi(v.c_str());
        } else if (a == "--idle") {
            if (!next(v)) return false;
            o.idle = std::atoi(v.c_str());
        } else if (a == "--parent-stdin") {
            o.parentStdin = true;
        } else if (a == "--verbose" || a == "-v") {
            o.verbose = true;
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

/// 可执行文件所在的目录
fs::path executableDirectory()
{
#ifdef _WIN32
    std::wstring path(MAX_PATH, L'\0');
    while (true) {
        const DWORD length = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
        if (length == 0)
            return {};
        if (length < path.size()) {
            path.resize(length);
            break;
        }
        path.resize(path.size() * 2);
    }
    return fs::path(path).parent_path();
#elif defined(__APPLE__)
    char buffer[4096];
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) != 0)
        return {};
    return fs::weakly_canonical(fs::path(buffer)).parent_path();
#else
    std::error_code error;
    return fs::read_symlink("/proc/self/exe", error).parent_path();
#endif
}

json loadFamilies(const std::string& manifest)
{
    fs::path path = manifest.empty() ? executableDirectory() / "power.json" : fs::u8path(manifest);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        DLOG_WARN("没有找到 %s：模型家族的缺省参数用不上（请求里得给全）", path.u8string().c_str());
        return json::array();
    }
    try {
        const json value = json::parse(in);
        return value.contains("families") ? value["families"] : json::array();
    } catch (const std::exception& exception) {
        DLOG_WARN("power.json 解析失败：%s", exception.what());
        return json::array();
    }
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
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    Options o;
    if (!parse(argc, argv, o)) {
        usage();
        return 2;
    }
    if (o.verbose)
        diffusion::logLevel() = diffusion::LogLevel::Debug;
    if (o.command == "version") {
        std::printf("friday-diffusion %s (stable-diffusion.cpp %s)\n", DIFFUSION_VERSION, DIFFUSION_SD_TAG);
        return 0;
    }
    if (o.command == "info") {
        std::printf("%s\n", json { { "id", "diffusion" }, { "version", DIFFUSION_VERSION }, { "stable-diffusion.cpp", DIFFUSION_SD_TAG },
                                   { "system_info", sd_get_system_info() } }
                                .dump(2)
                                .c_str());
        return 0;
    }
    if (o.command != "serve") {
        usage();
        return 2;
    }

    diffusion::Generator generator(loadFamilies(o.manifest), o.idle);
    diffusion::ServerOptions options;
    options.host = o.host;
    options.port = o.port;
    options.token = o.token;
    diffusion::Server server(generator, options);
    std::string error;
    if (!server.start(&error)) {
        DLOG_ERROR("%s", error.c_str());
        return 6;
    }
    DLOG_INFO("friday-diffusion %s 在 %s 上听", DIFFUSION_VERSION, server.url().c_str());
    std::printf("%s\n", json { { "event", "listening" }, { "url", server.url() }, { "version", DIFFUSION_VERSION } }.dump().c_str());
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
