// friday-voice：Friday 的本地 GPU 实时语音 Power。
//
//   friday-voice serve      --models <目录> | --model <sensevoice.gguf> --vad <fsmn-vad.gguf>
//                           [--host 127.0.0.1] [--port 0] [--device auto|cpu|cuda[:N]|vulkan[:N]|metal|opencl]
//                           [--token T] [--threads N] [--parent-stdin]
//   friday-voice transcribe （同上的模型参数） [--language auto|zh|en|yue|ja|ko] [--no-itn] [--repeat N] <音频文件>
//   friday-voice info       [--device …]
//   friday-voice --version
//
// serve 就绪后在 stdout 打一行 {"event":"listening","url":"http://127.0.0.1:端口"}；--parent-stdin 时 stdin 关了就退出
// （Friday 拉起时用，Friday 退出 / 崩溃 Power 跟着走）。日志都在 stderr。
#include "AudioIO.h"
#include "Backend.h"
#include "FsmnVad.h"
#include "Log.h"
#include "SenseVoice.h"
#include "Server.h"
#include "Streamer.h"

#include "json.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef VOICE_VERSION
#define VOICE_VERSION "0.0.0"
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Options {
    std::string command;
    std::string models, model, vad, device = "auto", host = "127.0.0.1", token, language = "auto", input;
    int port = 0, threads = 0, repeat = 1;
    bool parentStdin = false, itn = true, verbose = false;
};

void usage()
{
    std::fprintf(stderr,
                 "friday-voice %s —— Friday 的本地 GPU 实时语音\n\n"
                 "  friday-voice serve      (--models DIR | --model SENSEVOICE.gguf --vad FSMN-VAD.gguf)\n"
                 "                          [--host 127.0.0.1] [--port 0] [--device auto|cpu|cuda[:N]|vulkan[:N]|metal|opencl]\n"
                 "                          [--token T] [--threads N] [--parent-stdin] [--verbose]\n"
                 "  friday-voice transcribe (模型参数同上) [--language auto|zh|en|yue|ja|ko] [--no-itn] [--repeat N] AUDIO\n"
                 "  friday-voice info       [--device …]\n"
                 "  friday-voice --version\n",
                 VOICE_VERSION);
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
        } else if (a == "--models") {
            if (!next(o.models)) return false;
        } else if (a == "--model") {
            if (!next(o.model)) return false;
        } else if (a == "--vad") {
            if (!next(o.vad)) return false;
        } else if (a == "--device") {
            if (!next(o.device)) return false;
        } else if (a == "--host") {
            if (!next(o.host)) return false;
        } else if (a == "--token") {
            if (!next(o.token)) return false;
        } else if (a == "--language") {
            if (!next(o.language)) return false;
        } else if (a == "--port") {
            if (!next(v)) return false;
            o.port = std::atoi(v.c_str());
        } else if (a == "--threads") {
            if (!next(v)) return false;
            o.threads = std::atoi(v.c_str());
        } else if (a == "--repeat") {
            if (!next(v)) return false;
            o.repeat = std::max(1, std::atoi(v.c_str()));
        } else if (a == "--parent-stdin") {
            o.parentStdin = true;
        } else if (a == "--no-itn") {
            o.itn = false;
        } else if (a == "--verbose") {
            o.verbose = true;
        } else if (a == "-h" || a == "--help") {
            return false;
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "不认识的参数：%s\n", a.c_str());
            return false;
        } else if (o.command.empty()) {
            o.command = a;
        } else {
            o.input = a;
        }
    }
    return !o.command.empty();
}

/// --models 目录里按惯用文件名找模型
bool resolveModels(Options& o, std::string* error)
{
    if (!o.models.empty()) {
        const fs::path dir = fs::u8path(o.models);
        if (o.model.empty()) {
            for (const char* name : { "sensevoice-small-q8.gguf", "sensevoice-small-f16.gguf", "sensevoice-small-q4.gguf" }) {
                if (fs::exists(dir / name)) {
                    o.model = (dir / name).u8string();
                    break;
                }
            }
        }
        if (o.vad.empty() && fs::exists(dir / "fsmn-vad.gguf"))
            o.vad = (dir / "fsmn-vad.gguf").u8string();
    }
    if (o.model.empty() || o.vad.empty()) {
        if (error)
            *error = "要给识别模型和 VAD 模型：--models <目录> 或 --model … --vad …";
        return false;
    }
    return true;
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
    Options o;
    if (!parse(argc, argv, o)) {
        usage();
        return 2;
    }
    if (o.verbose)
        voice::logLevel() = voice::LogLevel::Debug;
    if (o.command == "version") {
        std::printf("friday-voice %s\n", VOICE_VERSION);
        return 0;
    }
    const int threads = o.threads > 0 ? o.threads : std::max(2, int(std::thread::hardware_concurrency()) / 2);

    if (o.command == "info") {
        voice::Backend backend;
        std::string error;
        const bool ok = backend.init(o.device, threads, &error);
        json devices = json::array();
        for (const voice::DeviceInfo& d : voice::listDevices())
            devices.push_back({ { "backend", d.backend }, { "name", d.name }, { "description", d.description }, { "type", d.type },
                                { "memory_free", d.memoryFree }, { "memory_total", d.memoryTotal } });
        json out { { "id", "voice" }, { "version", VOICE_VERSION }, { "devices", devices } };
        if (ok)
            out["selected"] = { { "backend", backend.device().backend }, { "name", backend.device().name },
                                { "description", backend.device().description } };
        else
            out["error"] = error;
        std::printf("%s\n", out.dump(2).c_str());
        return ok ? 0 : 1;
    }

    if (o.command != "serve" && o.command != "transcribe") {
        usage();
        return 2;
    }
    std::string error;
    if (!resolveModels(o, &error)) {
        VLOG_ERROR("%s", error.c_str());
        return 2;
    }
    voice::Backend backend;
    if (!backend.init(o.device, threads, &error)) {
        VLOG_ERROR("%s", error.c_str());
        return 3;
    }
    voice::SenseVoice recognizer(backend);
    voice::FsmnVad vad;
    if (!recognizer.load(o.model, &error) || !vad.load(o.vad, &error)) {
        VLOG_ERROR("%s", error.c_str());
        return 3;
    }

    if (o.command == "transcribe") {
        if (o.input.empty()) {
            usage();
            return 2;
        }
        std::vector<float> pcm;
        if (!voice::decodeAudioFile(o.input, pcm, &error)) {
            VLOG_ERROR("%s", error.c_str());
            return 4;
        }
        voice::Fbank fbank;
        voice::StreamConfig config;
        config.language = o.language;
        config.itn = o.itn;
        config.endSilenceMs = 600;
        std::string text, language;
        for (int i = 0; i < o.repeat; ++i) {
            const auto started = std::chrono::steady_clock::now();
            language.clear();
            text = voice::transcribeWhole(recognizer, vad, fbank, pcm, config, &language, &error);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            VLOG_INFO("第 %d 次：%.2f 秒音频，用时 %.3f 秒（实时率 %.4f）", i + 1, pcm.size() / double(voice::kSampleRate), seconds,
                      seconds / (pcm.size() / double(voice::kSampleRate)));
            if (!error.empty()) {
                VLOG_ERROR("%s", error.c_str());
                return 5;
            }
        }
        std::printf("%s\n", text.c_str());
        std::fflush(stdout);
        return 0;
    }

    // serve
    voice::ServerOptions options;
    options.host = o.host;
    options.port = o.port;
    options.token = o.token;
    voice::Server server(backend, recognizer, vad, options);
    if (!server.start(&error)) {
        VLOG_ERROR("%s", error.c_str());
        return 6;
    }
    // 预热一次（第一次推理要建 CUDA 图 / 编 Vulkan 管线，别让第一句话等）
    {
        std::vector<float> silence(voice::kSampleRate, 0.0f);
        const voice::Transcript warm = recognizer.transcribe(silence.data(), silence.size(), "auto", true);
        VLOG_INFO("预热完成：%.3f 秒", warm.seconds);
    }
    VLOG_INFO("在 %s 上听", server.url().c_str());
    std::printf("%s\n", json { { "event", "listening" }, { "url", server.url() }, { "version", VOICE_VERSION } }.dump().c_str());
    std::fflush(stdout);

    // 信号处理函数里只置标志（别的都不是异步信号安全的），主线程轮询
    std::signal(SIGINT, [](int) { g_stop = true; });
    std::signal(SIGTERM, [](int) { g_stop = true; });
    if (o.parentStdin) {
        std::thread([] {
            char buffer[256];
            while (std::fread(buffer, 1, sizeof(buffer), stdin) > 0) {
            }
            VLOG_INFO("宿主关了 stdin，退出");
            requestStop();
        }).detach();
    }
    {
        std::unique_lock<std::mutex> lock(g_stopMutex);
        while (!g_stop.load())
            g_stopWake.wait_for(lock, std::chrono::milliseconds(200));
    }
    server.stop();
    VLOG_INFO("已退出");
    return 0;
}
