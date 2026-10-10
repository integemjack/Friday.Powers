// friday-llm：Friday 的本地大模型 Power（llama.cpp + 自己的调度：连续批处理、多序列并发、基数树前缀共享）。
//
//   friday-llm serve  [--models DIR]… [--host 127.0.0.1] [--port 0] [--token T] [--parent-stdin]
//                     [--parallel 4] [--sequences 0] [--ctx 0] [--per-sequence 0] [--llama-arg ARG]… [--preload MODEL] [--verbose]
//   friday-llm info
//   friday-llm --version
//
// serve 就绪后在 stdout 打一行 {"event":"listening","url":"http://127.0.0.1:端口"}；--parent-stdin 时 stdin 关了就退出
// （Friday 拉起时用，Friday 退出 / 崩溃 Power 跟着走）。日志都在 stderr。模型按请求加载（docs/llm-protocol.md）。
// 环境变量 FRIDAY_LLAMA_ARGS（空格分开）追加给 llama 的参数，调试用。
#include "ModelHost.h"
#include "RuntimeLog.h"
#include "Server.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

#ifndef LLM_VERSION
#define LLM_VERSION "0.0.0"
#endif

namespace {

using json = flr::json;

struct Options {
    std::string command;
    std::vector<std::string> models;
    std::string host = "127.0.0.1", token, preload;
    int port = 0, parallel = 4, sequences = 0, ctx = 0, perSequence = 0;
    bool parentStdin = false, verbose = false;
    std::vector<std::string> llamaArgs;
};

void usage()
{
    std::fprintf(stderr,
                 "friday-llm %s —— Friday 的本地大模型（llama.cpp %s）\n\n"
                 "  friday-llm serve  [--models DIR]… [--host 127.0.0.1] [--port 0] [--token T] [--parent-stdin]\n"
                 "                    [--parallel 4]（同时在算的请求）[--sequences 0]（缓存的序列，0 = 按模型自动）\n"
                 "                    [--ctx 0]（KV 池，0 = 按显存自动）[--per-sequence 0]（单序列上限）\n"
                 "                    [--llama-arg ARG]…（追加给 llama 的参数）[--preload 模型]（启动就加载）[--verbose]\n"
                 "  friday-llm info\n"
                 "  friday-llm --version\n",
                 LLM_VERSION, LLM_LLAMA_TAG);
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
        const auto number = [&](int& v) {
            std::string text;
            if (!next(text))
                return false;
            v = std::atoi(text.c_str());
            return true;
        };
        std::string v;
        if (a == "--version") {
            o.command = "version";
        } else if (a == "--models") {
            if (!next(v)) return false;
            o.models.push_back(v);
        } else if (a == "--host") {
            if (!next(o.host)) return false;
        } else if (a == "--token") {
            if (!next(o.token)) return false;
        } else if (a == "--port") {
            if (!number(o.port)) return false;
        } else if (a == "--parallel") {
            if (!number(o.parallel)) return false;
        } else if (a == "--sequences") {
            if (!number(o.sequences)) return false;
        } else if (a == "--ctx") {
            if (!number(o.ctx)) return false;
        } else if (a == "--per-sequence") {
            if (!number(o.perSequence)) return false;
        } else if (a == "--llama-arg") {
            if (!next(v)) return false;
            o.llamaArgs.push_back(v);
        } else if (a == "--preload") {
            if (!next(o.preload)) return false;
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

std::atomic<bool> g_stop { false };
std::mutex g_stopMutex;
std::condition_variable g_stopWake;

void requestStop()
{
    g_stop = true;
    g_stopWake.notify_all();
}

/// 内核的日志写 stderr（NOTICE 及以上；--verbose 时全部）
void logToStderr(int level, const char* text, void*)
{
    static const char* names[] = { "debug", "info", "notice", "warn", "error" };
    const auto now = std::chrono::system_clock::now();
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    const int millis = int(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
    std::tm tm {};
#ifdef _WIN32
    localtime_s(&tm, &seconds);
#else
    localtime_r(&seconds, &tm);
#endif
    std::string line = text ? text : "";
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
        line.pop_back();
    if (line.empty())
        return;
    std::fprintf(stderr, "%02d:%02d:%02d.%03d [%s] %s\n", tm.tm_hour, tm.tm_min, tm.tm_sec, millis, names[std::max(0, std::min(level, 4))],
                 line.c_str());
    std::fflush(stderr);
}

std::vector<std::string> splitArgs(const char* text)
{
    std::vector<std::string> out;
    if (!text)
        return out;
    std::istringstream stream(text);
    std::string word;
    while (stream >> word)
        out.push_back(word);
    return out;
}

} // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    // Windows 的 argv 是系统代码页（中文系统是 GBK），中文路径会乱：改从宽字符命令行转 UTF-8
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
    // GGML_ABORT 时别弹 CRT 的对话框（Power 是后台进程）
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    Options o;
    if (!parse(argc, argv, o)) {
        usage();
        return 2;
    }
    if (o.command == "version") {
        std::printf("friday-llm %s (llama.cpp %s)\n", LLM_VERSION, LLM_LLAMA_TAG);
        return 0;
    }

    flr::setLogSink(&logToStderr, nullptr, o.verbose ? FLR_LOG_DEBUG : FLR_LOG_NOTICE);
    flr::installLibraryLogging();
    llama_backend_init();

    if (o.command == "info") {
        std::printf("%s\n", json { { "id", "llm" }, { "version", LLM_VERSION }, { "llama.cpp", LLM_LLAMA_TAG }, { "devices", llm::devicesJson() },
                                   { "system_info", llama_print_system_info() } }
                                .dump(2)
                                .c_str());
        return 0;
    }
    if (o.command != "serve") {
        usage();
        return 2;
    }

    llm::HostOptions hostOptions;
    hostOptions.modelRoots = o.models;
    hostOptions.parallel = std::max(1, o.parallel);
    hostOptions.sequences = o.sequences;
    hostOptions.contextTokens = o.ctx;
    hostOptions.perSequence = o.perSequence;
    hostOptions.extraArgs = o.llamaArgs;
    for (const std::string& arg : splitArgs(std::getenv("FRIDAY_LLAMA_ARGS")))
        hostOptions.extraArgs.push_back(arg);
    llm::ModelHost host(hostOptions);

    llm::ServerOptions serverOptions;
    serverOptions.host = o.host;
    serverOptions.port = o.port;
    serverOptions.token = o.token;
    llm::Server server(host, serverOptions);
    std::string error;
    if (!server.start(&error)) {
        flr::logf(FLR_LOG_ERROR, "%s", error.c_str());
        return 6;
    }
    if (!o.preload.empty()) {
        llm::ModelChoice choice;
        if (!host.resolve(json { { "model", o.preload } }, choice, &error)) {
            flr::logf(FLR_LOG_ERROR, "预加载：%s", error.c_str());
        } else {
            int code = FLR_OK;
            if (!host.acquire(choice, &code, &error))
                flr::logf(FLR_LOG_ERROR, "预加载失败：%s", error.c_str());
        }
    }
    flr::logf(FLR_LOG_NOTICE, "friday-llm %s 在 %s 上听（同时 %d 个请求）", LLM_VERSION, server.url().c_str(), hostOptions.parallel);
    std::printf("%s\n", json { { "event", "listening" }, { "url", server.url() }, { "version", LLM_VERSION } }.dump().c_str());
    std::fflush(stdout);

    // 信号处理函数里只置标志（别的都不是异步信号安全的），主线程轮询
    std::signal(SIGINT, [](int) { g_stop = true; });
    std::signal(SIGTERM, [](int) { g_stop = true; });
    if (o.parentStdin) {
        std::thread([] {
            char buffer[256];
            while (std::fread(buffer, 1, sizeof(buffer), stdin) > 0) {
            }
            flr::logf(FLR_LOG_NOTICE, "宿主关了 stdin，退出");
            requestStop();
        }).detach();
    }
    {
        std::unique_lock<std::mutex> lock(g_stopMutex);
        while (!g_stop.load())
            g_stopWake.wait_for(lock, std::chrono::milliseconds(200));
    }
    server.stop();
    host.unload();
    flr::logf(FLR_LOG_NOTICE, "已退出");
    return 0;
}
