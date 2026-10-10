// friday-llama 的日志接管（WP2，docs/LOCAL_INFERENCE.md §A.1、§B.2、§E.6）：运行时自己的日志（internal.h 的 flr::log / logf）与
// llama / ggml / mtmd 的输出都转给 flr_init 的回调，绝不写 stdout / stderr（friday-cli 的 stderr 是给用户看的）。
// common 的日志（common_log）没有回调接口，只能写控制台：整个暂停它（common_log_pause），一行都不出。
// 另外留着最近的日志：加载失败时取最后一条错误给用户看，GGML_ABORT 时连同消息写进崩溃记录（§E.6 第 4 条）。
#pragma once

#include "Codes.h"

#include <cstdint>
#include <string>

namespace flr {

/// flr_init 设置的回调与最低级别（FLR_LOG_*）；fn 为空 = 丢掉全部日志（最近日志与错误照样记）
void setLogSink(flr_log_fn fn, void* user, int minLevel);
/// 把 llama / ggml（llama_log_set）、mtmd（mtmd_helper_log_set）的日志接到 flr::log，暂停 common_log 的控制台输出。
/// 进程内调用一次即可（重复调用无害）。注意：common_init() 会把 llama 的日志改回 common_log，运行时不要调用它
void installLibraryLogging();
/// 再压一次 common_log：common_params_parse 会按 -lv / -v 改阈值，--log-file / --log-colors 会重启它的输出线程
void silenceCommonLog();
/// GGML_ABORT / GGML_ASSERT / CUDA_CHECK 失败时把消息与最近 200 行推理日志写进 path（UTF-8；空 = 不写），
/// Windows 上顺带关掉 CRT 的 abort 对话框与错误报告（_set_abort_behavior）。之后进程照样结束（ggml 随后调 abort()）
void setCrashLog(const std::string& path);

/// 有意中止 llama_process（取消、卸载）时 llama 会照常报 ERROR（graph_compute failed…）：这期间把库的 WARN / ERROR 降成 DEBUG，
/// 免得应用把它们当成错误显示。推理线程在 abort 回调返回 true 时打开，llama_process 返回后关上
void setLibraryErrorsQuiet(bool quiet);

/// 日志序号（每条 +1）。加载前记一个，失败后用 lastErrorSince 取这之后最后一条 ERROR
uint64_t logMark();
/// mark 之后最后一条 ERROR 级日志（去掉首尾空白）；没有为空
std::string lastErrorSince(uint64_t mark);
/// 最近的日志（最多 maxLines 条，INFO 及以上，按时间顺序拼起来）
std::string recentLog(size_t maxLines);

} // namespace flr
