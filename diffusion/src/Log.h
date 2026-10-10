// 日志：一律写 stderr（stdout 留给给宿主的 JSON 事件）
#pragma once

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <mutex>

namespace diffusion {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

inline LogLevel& logLevel()
{
    static LogLevel level = LogLevel::Info;
    return level;
}

inline void logf(LogLevel level, const char* format, ...)
{
    if (level < logLevel())
        return;
    static std::mutex mutex;
    static const char* names[] = { "debug", "info", "warn", "error" };
    const auto now = std::chrono::system_clock::now();
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    const int millis = int(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
    std::tm tm {};
#ifdef _WIN32
    localtime_s(&tm, &seconds);
#else
    localtime_r(&seconds, &tm);
#endif
    std::lock_guard<std::mutex> lock(mutex);
    std::fprintf(stderr, "%02d:%02d:%02d.%03d [%s] ", tm.tm_hour, tm.tm_min, tm.tm_sec, millis, names[int(level)]);
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

} // namespace diffusion

#define DLOG_DEBUG(...) ::diffusion::logf(::diffusion::LogLevel::Debug, __VA_ARGS__)
#define DLOG_INFO(...) ::diffusion::logf(::diffusion::LogLevel::Info, __VA_ARGS__)
#define DLOG_WARN(...) ::diffusion::logf(::diffusion::LogLevel::Warn, __VA_ARGS__)
#define DLOG_ERROR(...) ::diffusion::logf(::diffusion::LogLevel::Error, __VA_ARGS__)
