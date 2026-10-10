// 推理内核的返回码、日志级别与回调类型（原 friday-llama 的 C ABI 头文件 friday_llama.h 里的这一部分；v2.7 起内核编进 friday-llm，
// 不再是应用加载的库，C 函数都去掉了，HTTP 接口见 docs/llm-protocol.md）。
#pragma once

/* 日志级别。NOTICE 是内核自己写的要点（加载用时、一次请求的 cached / prefilled / generated、检查点、前缀共享）；
 * DEBUG / INFO 是 llama / ggml / mtmd 的原样输出 */
enum {
    FLR_LOG_DEBUG = 0,
    FLR_LOG_INFO = 1,
    FLR_LOG_NOTICE = 2,
    FLR_LOG_WARN = 3,
    FLR_LOG_ERROR = 4
};

/* 返回码 */
enum {
    FLR_OK = 0,
    FLR_CANCELLED,          /* 调用方要停（连接断了），或加载进度回调返回 false */
    FLR_CONTEXT_OVERFLOW,   /* 提示超过单序列上限（建流前）；result.error 带 n_prompt_tokens / n_ctx */
    FLR_KV_FULL,            /* 共用的 KV 池放不下，这个请求被选为牺牲者；其余请求照常 */
    FLR_INVALID_REQUEST,    /* 请求 / 参数 / JSON 不对 */
    FLR_UNSUPPORTED_MEDIA,  /* 纯文本模型收到图片（或模型不支持的媒体） */
    FLR_COMPUTE_ERROR,      /* decode 出错（llama_process 返回 < 0） */
    FLR_NOT_READY,          /* 正在卸载 / 换模型 */
    FLR_IO_ERROR,           /* 读写文件失败，或文件格式不对 */
    FLR_MODEL_MISMATCH,     /* 状态文件属于别的模型 */
    FLR_INTERNAL            /* 其它（内核里的异常，已经接住） */
};

/* level 见 FLR_LOG_*；text 是一段原样输出（可能含多行、可能不以换行结尾）。任意线程 */
typedef void (*flr_log_fn)(int level, const char* text, void* user);
/* 加载进度 0…1；返回 false = 取消加载 */
typedef bool (*flr_progress_fn)(double progress, void* user);
/* 一个 OpenAI 流式 chunk（JSON 文本，== SSE 的 data: 负载）；返回 false = 停止这个请求 */
typedef bool (*flr_chunk_fn)(const char* chunk_json, void* user);
/* 等待时每 ≤ 10 ms 调一次；返回 true = 停止这个请求 */
typedef bool (*flr_should_stop_fn)(void* user);
