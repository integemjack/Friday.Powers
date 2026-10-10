// friday-llama 的日志接管（WP2，见 RuntimeLog.h）
#include "RuntimeLog.h"

#include "internal.h"

#include "log.h"
#include "mtmd-helper.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <mutex>

#if defined(_WIN32)
#include <stdlib.h>
#endif

namespace flr {
namespace {

std::mutex g_mutex;
flr_log_fn g_sink = nullptr;
void* g_user = nullptr;
int g_minLevel = FLR_LOG_INFO;
/// ggml 的 CONT（同一行的续写）沿用上一条的级别
int g_lastLibraryLevel = FLR_LOG_INFO;

/// 每条日志 +1（logMark / lastErrorSince）
uint64_t g_sequence = 0;
/// 最后一条 ERROR（CONT 续写接在后面）
std::string g_lastError;
uint64_t g_lastErrorSequence = 0;
bool g_lastWasError = false;

/// 最近的日志（INFO 及以上；一条没以换行结尾时下一段接在它后面）
std::deque<std::string> g_recent;
bool g_recentOpen = false;
constexpr size_t kRecentLines = 200;

/// 崩溃记录的文件（setCrashLog）
std::filesystem::path g_crashPath;

/// setLibraryErrorsQuiet
std::atomic<bool> g_quietLibraryErrors { false };

/// 调用方持 g_mutex
void rememberLocked(int level, const char* text, bool continuation)
{
    ++g_sequence;
    if (level == FLR_LOG_ERROR) {
        if (continuation && g_lastWasError) {
            g_lastError += text;
        } else {
            g_lastError = text;
        }
        g_lastErrorSequence = g_sequence;
        g_lastWasError = true;
    } else if (!continuation) {
        g_lastWasError = false;
    }
    if (level < FLR_LOG_INFO)
        return;
    if (!g_recentOpen || g_recent.empty())
        g_recent.emplace_back();
    g_recent.back() += text;
    const size_t length = std::char_traits<char>::length(text);
    g_recentOpen = length == 0 || text[length - 1] != '\n';
    while (g_recent.size() > kRecentLines)
        g_recent.pop_front();
}

/// 回调在锁里调：多个线程的日志不会交错
void deliverLocked(int level, const char* text, bool continuation = false)
{
    if (!text || !*text)
        return;
    rememberLocked(level, text, continuation);
    if (!g_sink || level < g_minLevel)
        return;
    g_sink(level, text, g_user);
}

void deliver(int level, const char* text)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    deliverLocked(level, text);
}

/// 库里报成 WARN、其实是正常现象的几条（降成 INFO：照样进 llama.log，不进应用的日志面板——面板只看 NOTICE 以上）
bool benignWarning(const char* text)
{
    static const char* const patterns[] = {
        // M-RoPE 的图片块（Qwen-VL 系列）：块里单元的位置（t 维）都等于起始位置，混合 / 循环模型每块图片报几行（llama-memory-recurrent.cpp）
        "non-consecutive token position",
        // 每次加载 Qwen-VL 的 mmproj 都报（clip.cpp 的 load_hparams，llama-server 同样打印）：说的是定位任务，看图描述用不着
        "Qwen-VL models require at minimum",
        "try adding --image-min-tokens",
        "github.com/ggml-org/llama.cpp/issues/16842",
    };
    for (const char* pattern : patterns) {
        if (std::strstr(text, pattern))
            return true;
    }
    // 只有换行的 WARN（上面那段提示最后的空行）
    return text[std::strspn(text, " \t\r\n")] == '\0';
}

/// llama / ggml / mtmd 的日志回调
void libraryLog(enum ggml_log_level level, const char* text, void*)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    int mapped = FLR_LOG_INFO;
    switch (level) {
    case GGML_LOG_LEVEL_DEBUG: mapped = FLR_LOG_DEBUG; break;
    case GGML_LOG_LEVEL_INFO: mapped = FLR_LOG_INFO; break;
    case GGML_LOG_LEVEL_WARN: mapped = FLR_LOG_WARN; break;
    case GGML_LOG_LEVEL_ERROR: mapped = FLR_LOG_ERROR; break;
    case GGML_LOG_LEVEL_CONT: mapped = g_lastLibraryLevel; break;
    default: mapped = FLR_LOG_INFO; break;
    }
    if (level != GGML_LOG_LEVEL_CONT) {
        if (mapped >= FLR_LOG_WARN && g_quietLibraryErrors.load(std::memory_order_relaxed))
            mapped = FLR_LOG_DEBUG;
        if (mapped == FLR_LOG_WARN && text && benignWarning(text))
            mapped = FLR_LOG_INFO;
        g_lastLibraryLevel = mapped;
    }
    deliverLocked(mapped, text, level == GGML_LOG_LEVEL_CONT);
}

std::string trimmed(const std::string& text)
{
    const size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
        return {};
    const size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

/// 调用方持 g_mutex
std::string recentLocked(size_t maxLines)
{
    std::string text;
    const size_t start = g_recent.size() > maxLines ? g_recent.size() - maxLines : 0;
    for (size_t i = start; i < g_recent.size(); ++i) {
        text += g_recent[i];
        if (text.empty() || text.back() != '\n')
            text += '\n';
    }
    return text;
}

/// ggml_abort 之前调用（之后 ggml 调 abort()，进程结束）。可能在任意线程，也可能是日志锁正被占着的时候（日志回调里断言失败）：
/// 只 try_lock，拿不到就不带最近日志
void onAbort(const char* message)
{
    std::string recent;
    std::filesystem::path path;
    {
        std::unique_lock<std::mutex> lock(g_mutex, std::try_to_lock);
        if (lock.owns_lock()) {
            recent = recentLocked(kRecentLines);
            path = g_crashPath;
            if (g_sink)
                g_sink(FLR_LOG_ERROR, message ? message : "", g_user);
        } else {
            recent = "（日志正被占用，没有取到）\n";
            path = g_crashPath;
        }
    }
    if (path.empty())
        return;
    char when[64] = {};
    const std::time_t now = std::time(nullptr);
    std::tm local {};
#if defined(_WIN32)
    localtime_s(&local, &now);
    FILE* file = _wfopen(path.wstring().c_str(), L"wb");
#else
    localtime_r(&now, &local);
    FILE* file = std::fopen(path.string().c_str(), "wb");
#endif
    std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", &local);
    if (!file)
        return;
    std::fprintf(file, "本地推理异常退出：%s\n时间：%s\n\n最近的推理日志：\n%s", message ? message : "", when, recent.c_str());
    std::fclose(file);
}

} // namespace

void setLogSink(flr_log_fn fn, void* user, int minLevel)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_sink = fn;
    g_user = user;
    g_minLevel = minLevel;
}

void installLibraryLogging()
{
    llama_log_set(libraryLog, nullptr);
    mtmd_helper_log_set(libraryLog, nullptr);
    silenceCommonLog();
}

void silenceCommonLog()
{
    // common_log 由自己的工作线程写 stdout / stderr，没有回调接口：暂停它（之后的消息直接丢掉，不排队、不阻塞），
    // 阈值也压到最低（LOG_* 宏连格式化都省了）。都是幂等的
    common_log_pause(common_log_main());
    common_log_set_verbosity_thold(-1);
}

void setCrashLog(const std::string& path)
{
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_crashPath = path.empty() ? std::filesystem::path() : std::filesystem::u8path(path);
    }
    if (path.empty())
        return;
    ggml_set_abort_callback(onAbort);
#if defined(_WIN32)
    // abort() 不弹 CRT 的对话框、不走 Windows 错误报告（同一个 UCRT，ggml 的 abort 也受它管）
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
}

void setLibraryErrorsQuiet(bool quiet)
{
    g_quietLibraryErrors.store(quiet, std::memory_order_relaxed);
}

uint64_t logMark()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_sequence;
}

std::string lastErrorSince(uint64_t mark)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_lastErrorSequence <= mark)
        return {};
    return trimmed(g_lastError);
}

std::string recentLog(size_t maxLines)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    return recentLocked(maxLines);
}

void log(int level, const std::string& text)
{
    deliver(level, text.c_str());
}

void logf(int level, const char* format, ...)
{
    // 最多 2046 个字符 + 补上的换行 + '\0'
    char buffer[2048];
    va_list args;
    va_start(args, format);
    const int written = std::vsnprintf(buffer, sizeof(buffer) - 1, format, args);
    va_end(args);
    if (written < 0)
        return;
    size_t length = size_t(written) < sizeof(buffer) - 2 ? size_t(written) : sizeof(buffer) - 2;
    if (length == 0 || buffer[length - 1] != '\n')
        buffer[length++] = '\n';
    buffer[length] = '\0';
    deliver(level, buffer);
}

} // namespace flr
