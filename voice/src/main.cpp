// friday-voice：Friday 的本地 GPU 实时语音 Power。
//
//   friday-voice serve      --models <目录> | --model <sensevoice.gguf> --vad <fsmn-vad.gguf>
//                           [--host 127.0.0.1] [--port 0] [--device auto|cpu|cuda[:N]|vulkan[:N]|metal|opencl]
//                           [--token T] [--threads N] [--parent-stdin]
//   friday-voice transcribe （同上的模型参数） [--language auto|zh|en|yue|ja|ko] [--no-itn] [--repeat N] <音频文件>
//   friday-voice speak      --models <目录> | --tts-model <CosyVoice3.gguf> --voices <目录>
//                           --text <文字> --output <out.wav> [--voice 名] [--speed 1.0] [--instruction 指令] [--repeat N]
//   friday-voice info       [--device …]
//
// 合成模型可选：--models 目录下有 cosyvoice3/CosyVoice3*.gguf 和 voices/*.gguf 时 serve 顺带开合成（--no-tts 关掉）。
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
#ifdef VOICE_HAS_TTS
#include "Synthesizer.h"
#include "TextSplitter.h"
#endif

#include "json.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

#ifndef VOICE_VERSION
#define VOICE_VERSION "0.0.0"
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Options {
    std::string command;
    std::string models, model, vad, device = "auto", host = "127.0.0.1", token, language = "auto", input;
    std::string ttsModel, voices, text, output, voice, instruction;
    float speed = 1.0f;
    int port = 0, threads = 0, repeat = 1, chunkTokens = 0;
    bool parentStdin = false, itn = true, verbose = false, tts = true;
};

void usage()
{
    std::fprintf(stderr,
                 "friday-voice %s —— Friday 的本地 GPU 实时语音\n\n"
                 "  friday-voice serve      (--models DIR | --model SENSEVOICE.gguf --vad FSMN-VAD.gguf)\n"
                 "                          [--host 127.0.0.1] [--port 0] [--device auto|cpu|cuda[:N]|vulkan[:N]|metal|opencl]\n"
                 "                          [--token T] [--threads N] [--parent-stdin] [--verbose]\n"
                 "                          [--tts-model COSYVOICE3.gguf --voices DIR | --no-tts]\n"
                 "  friday-voice transcribe (模型参数同上) [--language auto|zh|en|yue|ja|ko] [--no-itn] [--repeat N] AUDIO\n"
                 "  friday-voice speak      (--models DIR | --tts-model COSYVOICE3.gguf --voices DIR) --text TEXT --output OUT.wav\n"
                 "                          [--voice NAME] [--speed 1.0] [--instruction TEXT] [--repeat N]\n"
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
        } else if (a == "--tts-model") {
            if (!next(o.ttsModel)) return false;
        } else if (a == "--voices") {
            if (!next(o.voices)) return false;
        } else if (a == "--text") {
            if (!next(o.text)) return false;
        } else if (a == "--output") {
            if (!next(o.output)) return false;
        } else if (a == "--voice") {
            if (!next(o.voice)) return false;
        } else if (a == "--instruction") {
            if (!next(o.instruction)) return false;
        } else if (a == "--speed") {
            if (!next(v)) return false;
            o.speed = float(std::atof(v.c_str()));
        } else if (a == "--chunk-tokens") {
            if (!next(v)) return false;
            o.chunkTokens = std::atoi(v.c_str());
        } else if (a == "--no-tts") {
            o.tts = false;
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
        if (o.ttsModel.empty()) {
            // 优先 Q8_0，其次随便哪个量化
            for (const fs::path& sub : { dir / "cosyvoice3", dir }) {
                if (fs::exists(sub / "CosyVoice3-2512_Q8_0.gguf")) {
                    o.ttsModel = (sub / "CosyVoice3-2512_Q8_0.gguf").u8string();
                    break;
                }
                std::error_code ec;
                for (const auto& entry : fs::directory_iterator(sub, ec)) {
                    const std::string name = entry.path().filename().u8string();
                    if (name.rfind("CosyVoice3", 0) == 0 && entry.path().extension() == ".gguf") {
                        o.ttsModel = entry.path().u8string();
                        break;
                    }
                }
                if (!o.ttsModel.empty())
                    break;
            }
        }
        if (o.voices.empty() && fs::exists(dir / "voices"))
            o.voices = (dir / "voices").u8string();
    }
    if (o.command == "speak")
        return true;   // 只要合成模型
    if (o.model.empty() || o.vad.empty()) {
        if (error)
            *error = "要给识别模型和 VAD 模型：--models <目录> 或 --model … --vad …";
        return false;
    }
    return true;
}

#ifdef VOICE_HAS_TTS
bool writeWav(const std::string& path, const std::vector<float>& samples, int rate)
{
    std::string pcm;
    voice::floatToPcm16(samples.data(), samples.size(), pcm);
    FILE* f = nullptr;
#ifdef _WIN32
    f = _wfopen(fs::u8path(path).wstring().c_str(), L"wb");
#else
    f = std::fopen(path.c_str(), "wb");
#endif
    if (!f)
        return false;
    std::string header = "RIFF";
    const auto u32 = [&](uint32_t v) { header.append(reinterpret_cast<const char*>(&v), 4); };
    const auto u16 = [&](uint16_t v) { header.append(reinterpret_cast<const char*>(&v), 2); };
    u32(uint32_t(36 + pcm.size()));
    header += "WAVEfmt ";
    u32(16);
    u16(1);
    u16(1);
    u32(uint32_t(rate));
    u32(uint32_t(rate * 2));
    u16(2);
    u16(16);
    header += "data";
    u32(uint32_t(pcm.size()));
    const bool ok = std::fwrite(header.data(), 1, header.size(), f) == header.size() && std::fwrite(pcm.data(), 1, pcm.size(), f) == pcm.size();
    std::fclose(f);
    return ok;
}

/// friday-voice speak：合成一段字存成 wav，打印每段首块延迟和实时率（调参、测速用）
int runSpeak(const Options& o, const voice::Backend& backend)
{
    if (o.ttsModel.empty() || o.voices.empty() || o.text.empty() || o.output.empty()) {
        VLOG_ERROR("speak 要 --tts-model / --voices（或 --models）、--text、--output");
        return 2;
    }
    voice::Synthesizer synth(backend);
    std::string error;
    if (!synth.load(o.ttsModel, o.voices, &error)) {
        VLOG_ERROR("%s", error.c_str());
        return 3;
    }
    synth.setChunkTokens(unsigned(std::max(0, o.chunkTokens)));
    voice::TextSplitter splitter;
    std::vector<std::string> segments = splitter.push(o.text);
    for (std::string& s : splitter.flush())
        segments.push_back(std::move(s));
    std::vector<float> audio;
    for (int round = 0; round < o.repeat; ++round) {
        audio.clear();
        const auto started = std::chrono::steady_clock::now();
        double firstAudio = -1;
        for (const std::string& segment : segments) {
            const voice::SpeakResult r = synth.speak(1, segment, o.voice, o.speed, o.instruction, [&](const float* s, size_t n) {
                if (firstAudio < 0)
                    firstAudio = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                audio.insert(audio.end(), s, s + n);
                return true;
            });
            if (!r.ok) {
                VLOG_ERROR("%s", r.error.c_str());
                return 5;
            }
            VLOG_INFO("  「%s」%.2f 秒音频，合成 %.3f 秒，首块 %.3f 秒", segment.c_str(), r.samples / double(synth.sampleRate()), r.seconds,
                      r.firstAudioSeconds);
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const double audioSeconds = audio.size() / double(synth.sampleRate());
        VLOG_INFO("第 %d 次：%d 段，%.2f 秒音频，用时 %.3f 秒（实时率 %.3f，首块 %.3f 秒）", round + 1, int(segments.size()), audioSeconds,
                  seconds, seconds / std::max(0.001, audioSeconds), firstAudio);
    }
    if (!writeWav(o.output, audio, synth.sampleRate())) {
        VLOG_ERROR("写不了 %s", o.output.c_str());
        return 4;
    }
    return 0;
}
#endif

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
    // Windows 的 argv 是系统代码页（中文系统是 GBK），中文文字、路径会乱：改从宽字符命令行转 UTF-8
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
#endif
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

    if (o.command != "serve" && o.command != "transcribe" && o.command != "speak") {
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
    if (o.command == "speak") {
#ifdef VOICE_HAS_TTS
        return runSpeak(o, backend);
#else
        VLOG_ERROR("这个版本没编合成");
        return 2;
#endif
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
    voice::Synthesizer* synthesizer = nullptr;
#ifdef VOICE_HAS_TTS
    std::unique_ptr<voice::Synthesizer> synth;
    if (o.tts && !o.ttsModel.empty() && !o.voices.empty()) {
        synth = std::make_unique<voice::Synthesizer>(backend);
        if (synth->load(o.ttsModel, o.voices, &error)) {
            synth->setChunkTokens(unsigned(std::max(0, o.chunkTokens)));
            synthesizer = synth.get();
        } else {
            VLOG_WARN("合成用不了，只开听写：%s", error.c_str());
            synth.reset();
        }
    } else if (o.tts) {
        VLOG_INFO("没找到合成模型，只开听写");
    }
#endif
    voice::ServerOptions options;
    options.host = o.host;
    options.port = o.port;
    options.token = o.token;
    voice::Server server(backend, recognizer, vad, synthesizer, options);
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
#ifdef VOICE_HAS_TTS
    if (synthesizer) {
        const voice::SpeakResult warm = synthesizer->speak(0, "你好。", "", 1.0f, "", [](const float*, size_t) { return true; });
        VLOG_INFO("合成预热完成：%.3f 秒", warm.seconds);
    }
#endif
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
