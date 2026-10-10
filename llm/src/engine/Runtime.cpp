// 一个已加载的模型（见 Runtime.h）。
// 加载流程照 llama.cpp 的 tools/server/server.cpp main 与 server_context_impl::load_model（MIT License，
// Copyright (c) 2023-2026 The ggml authors）。
#include "Runtime.h"

#include "RuntimeLog.h"

#include "arg.h"
#include "gguf.h"
#include "preset.h"

#include <algorithm>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>

#if defined(_WIN32)
#include <windows.h>
#include <shellapi.h>
#endif

namespace flr {
namespace {

/// llama_context 上的 MemoryOps（internal.h）：失败返回 false / 空，不用会 GGML_ABORT 的 common 辅助函数（§E.6）
class LlamaMemory final : public MemoryOps {
public:
    LlamaMemory(llama_context* ctx, common_context_seq_rm_type type)
        : m_ctx(ctx)
        , m_type(type)
        , m_seqMax(int(llama_n_seq_max(ctx)))
    {
    }

    using MemoryOps::restorePartial;

    bool partialRemovable() const override { return m_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART; }

    bool seqRemove(int seq, llama_pos p0, llama_pos p1) override
    {
        return valid(seq) && llama_memory_seq_rm(memory(), seq, p0, p1);
    }

    void seqCopy(int src, int dst) override
    {
        if (!valid(src) || !valid(dst) || src == dst)
            return;
        llama_memory_seq_rm(memory(), dst, -1, -1);
        llama_memory_seq_cp(memory(), src, dst, -1, -1);
    }

    void seqClear(int seq) override
    {
        if (valid(seq))
            llama_memory_seq_rm(memory(), seq, -1, -1);
    }

    void seqShare(int src, int dst, llama_pos p1) override
    {
        if (!valid(src) || !valid(dst) || src == dst)
            return;
        llama_memory_seq_rm(memory(), dst, -1, -1);
        llama_memory_seq_cp(memory(), src, dst, 0, p1);
    }

    llama_pos seqPosMax(int seq) const override
    {
        return valid(seq) ? llama_memory_seq_pos_max(llama_get_memory(m_ctx), seq) : -1;
    }

    std::vector<uint8_t> partialState(int seq) override { return state(seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY); }
    bool restorePartial(int seq, const std::vector<uint8_t>& data) override { return restore(seq, data, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY); }
    std::vector<uint8_t> fullState(int seq) override { return state(seq, LLAMA_STATE_SEQ_FLAGS_NONE); }
    bool restoreFull(int seq, const std::vector<uint8_t>& data) override { return restore(seq, data, LLAMA_STATE_SEQ_FLAGS_NONE); }

private:
    bool valid(int seq) const { return seq >= 0 && seq < m_seqMax; }
    llama_memory_t memory() const { return llama_get_memory(m_ctx); }

    std::vector<uint8_t> state(int seq, llama_state_seq_flags flags)
    {
        if (!valid(seq))
            return {};
        std::vector<uint8_t> data(llama_state_seq_get_size_ext(m_ctx, seq, flags));
        if (data.empty())
            return data;
        const size_t written = llama_state_seq_get_data_ext(m_ctx, data.data(), data.size(), seq, flags);
        data.resize(written);
        return data;
    }

    bool restore(int seq, const std::vector<uint8_t>& data, llama_state_seq_flags flags)
    {
        // 数据不对、放不下时返回 0（spike 实测进程无事）
        return valid(seq) && !data.empty() && llama_state_seq_set_data_ext(m_ctx, data.data(), data.size(), seq, flags) != 0;
    }

    llama_context* m_ctx;
    common_context_seq_rm_type m_type;
    int m_seqMax;
};

const char* seqRmName(common_context_seq_rm_type type)
{
    switch (type) {
    case COMMON_CONTEXT_SEQ_RM_TYPE_PART: return "part";
    case COMMON_CONTEXT_SEQ_RM_TYPE_FULL: return "full";
    case COMMON_CONTEXT_SEQ_RM_TYPE_RS: return "rs";
    default: return "no";
    }
}

std::string memoryKindOf(const llama_model* model)
{
    if (llama_model_is_hybrid(model))
        return "hybrid";
    if (llama_model_is_recurrent(model))
        return "recurrent";
    if (llama_model_n_swa(model) > 0)
        return "swa";
    return "attention";
}

bool fileExists(const std::string& path)
{
    std::error_code error;
    return !path.empty() && std::filesystem::is_regular_file(std::filesystem::u8path(path), error);
}

/// 显卡（独显 + 核显）已用的显存，MiB（日志用）
double usedVramMiB()
{
    size_t used = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU)
            continue;
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        used += total - free;
    }
    return double(used) / 1048576.0;
}

/// 加载进度：主模型与 mmproj 两段，等分拼成一条（== LlamaServer 按 stages 拼的算法）；只报增长，最多每 50 ms 一次
struct LoadProgress {
    flr_progress_fn fn = nullptr;
    void* user = nullptr;
    int stage = 0;
    int stages = 1;
    double last = -1;
    int64_t lastCall = 0;
    bool cancelled = false;

    bool report(float value)
    {
        if (cancelled)
            return false;
        if (!fn)
            return true;
        const double clamped = std::min(1.0, std::max(0.0, double(value)));
        const double total = (double(stage) + clamped) / double(stages);
        const int64_t now = ggml_time_ms();
        if (total <= last || (clamped < 1.0 && now - lastCall < 50 && last >= 0))
            return true;
        last = total;
        lastCall = now;
        if (!fn(total, user))
            cancelled = true;
        return !cancelled;
    }

    static bool trampoline(float value, void* data) { return static_cast<LoadProgress*>(data)->report(value); }
};

/// 每 token 的 KV 字节数与每序列的循环状态字节数（flr_info）：在工作序列上算两个 token，用 llama_state_seq_get_size 量，量完清掉
void measureState(llama_context* ctx, const llama_vocab* vocab, int seq, uint64_t& perToken, uint64_t& perSequence)
{
    llama_token token = llama_vocab_bos(vocab);
    if (token == LLAMA_TOKEN_NULL)
        token = 0;
    llama_batch_ext* batch = llama_batch_ext_init(ctx);
    if (!batch)
        return;
    const auto decodeOne = [&](llama_pos pos) {
        llama_batch_ext_clear(batch);
        const int32_t index = llama_batch_ext_add_token(batch, seq, token);
        if (index < 0)
            return false;
        llama_batch_ext_set_pos(batch, index, &pos);
        return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0;
    };
    if (decodeOne(0)) {
        const size_t one = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (decodeOne(1)) {
            const size_t two = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
            perToken = two > one ? uint64_t(two - one) : 0;
        }
        perSequence = uint64_t(llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    }
    llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
    llama_batch_ext_free(batch);
}

std::string format(const char* pattern, ...)
{
    char buffer[1024];
    va_list args;
    va_start(args, pattern);
    std::vsnprintf(buffer, sizeof(buffer), pattern, args);
    va_end(args);
    return buffer;
}

json errorJson(int seq, const std::string& message)
{
    return json { { "seq", seq }, { "error", json { { "message", message } } } };
}

/// llama.cpp 的系统配置文件（%PROGRAMDATA%\llama.cpp\config.ini 或 /etc/llama.cpp/config.ini、用户配置目录里的 config.ini）：
/// common_params_parse 会读它们（== llama-server），格式或值不对时抛的不是 invalid_argument → exit(1)。
/// 照 common_params_apply_system_config（common/arg.cpp:719-760）先在草稿参数上读一遍，读不了就报错，不让进程退出
bool checkSystemConfig(std::string* error)
{
    std::vector<std::filesystem::path> paths;
#if defined(_WIN32)
    const std::filesystem::path programData = common_get_path_from_env("PROGRAMDATA");
    if (!programData.empty())
        paths.push_back(programData / "llama.cpp" / "config.ini");
#else
    paths.push_back("/etc/llama.cpp/config.ini");
#endif
    try {
        paths.push_back(fs_get_config_directory() / "config.ini");
    } catch (const std::exception&) {
    }
    for (const std::filesystem::path& path : paths) {
        std::error_code missing;
        if (!std::filesystem::exists(path, missing))
            continue;
        try {
            common_preset_context context(LLAMA_EXAMPLE_SERVER);
            context.ignore_unknown_keys = true;
            common_preset global;
            const common_presets presets = context.load_from_ini(path, global);
            common_params scratch;
            global.apply_to_params(scratch);
            const auto preset = presets.find(COMMON_PRESET_DEFAULT_NAME);
            if (preset != presets.end())
                preset->second.apply_to_params(scratch);
        } catch (const std::exception& exception) {
            if (error)
                *error = "llama.cpp 的配置文件读不了（" + path.u8string() + "）：" + exception.what();
            return false;
        }
    }
    return true;
}

} // namespace

std::unique_ptr<MemoryOps> makeMemoryOps(llama_context* ctx, common_context_seq_rm_type seqRmType)
{
    return std::make_unique<LlamaMemory>(ctx, seqRmType);
}

bool prepareArguments(const std::vector<std::string>& args, std::vector<std::string>& out, std::string* error)
{
    const auto fail = [error](const std::string& message) {
        if (error)
            *error = message;
        return false;
    };
    if (!checkSystemConfig(error))
        return false;
    // 会让进程 exit() 的（common/arg.cpp）与只管 common_log 控制台输出的：都去掉（参数表里第一个名字）
    static const std::set<std::string> dropped {
        "-h", "--version", "-cl", "--completion-bash", "--list-devices",
        "--log-disable", "--log-file", "--log-jsonl", "--log-prompts-dir", "--log-colors", "--log-prefix", "--log-timestamps",
    };
    common_params scratch;
    common_params_context context = common_params_parser_init(scratch, LLAMA_EXAMPLE_SERVER);
    std::map<std::string, const common_arg*> options;
    for (const common_arg& option : context.options) {
        for (const char* name : option.args)
            options[name] = &option;
        for (const char* name : option.args_neg)
            options[name] = &option;
    }

    out.clear();
    out.push_back("friday");
    // common_log 写控制台、没有回调：整个暂停（运行时自己也会暂停，这里是保险）
    out.push_back("--log-disable");
    for (size_t i = 0; i < args.size(); ++i) {
        std::string name = args[i];
        if (name.compare(0, 2, "--") == 0)
            std::replace(name.begin(), name.end(), '_', '-');
        const auto found = options.find(name);
        if (found == options.end())
            return fail("不认识的参数：" + args[i]);
        const common_arg& option = *found->second;
        const size_t arity = option.handler_str_str ? 2 : (option.handler_string || option.handler_int) ? 1 : 0;
        if (i + arity >= args.size())
            return fail("参数 " + args[i] + " 缺值");
        const std::string first = option.args.empty() ? name : std::string(option.args.front());
        if (dropped.count(first)) {
            i += arity;
            continue;
        }
        // 先在一份草稿参数上跑一遍处理函数：值不对时这里接住，不让 common_params_parse 往 stderr 打一行（或者 exit）。
        // --rpc 会注册后端设备，跑两遍会注册两次，不预跑
        if (first != "--rpc") {
            try {
                if (option.handler_void) {
                    option.handler_void(scratch);
                } else if (option.handler_bool) {
                    const bool negative = std::find_if(option.args_neg.begin(), option.args_neg.end(),
                                                       [&](const char* candidate) { return name == candidate; })
                        != option.args_neg.end();
                    option.handler_bool(scratch, !negative);
                } else if (option.handler_int) {
                    option.handler_int(scratch, std::stoi(args[i + 1]));
                } else if (option.handler_string) {
                    option.handler_string(scratch, args[i + 1]);
                } else if (option.handler_str_str) {
                    option.handler_str_str(scratch, args[i + 1], args[i + 2]);
                }
            } catch (const std::exception& exception) {
                std::string what = exception.what();
                while (!what.empty() && (what.back() == '\n' || what.back() == ' '))
                    what.pop_back();
                return fail("参数 " + args[i] + " 的值不对：" + what);
            }
        }
        out.push_back(args[i]);
        for (size_t k = 1; k <= arity; ++k)
            out.push_back(args[i + k]);
        i += arity;
    }
#if defined(_WIN32)
    // common_params_parse 在 Windows 上：参数个数等于本进程命令行的个数时改用进程的命令行（为了修 UTF-8）。错开它
    int processArgc = 0;
    if (LPWSTR* processArgv = CommandLineToArgvW(GetCommandLineW(), &processArgc))
        LocalFree(processArgv);
    if (int(out.size()) == processArgc)
        out.insert(out.begin() + 1, "--log-disable");
#endif
    // 预跑可能动了 common_log（-v / -lv 改阈值）
    silenceCommonLog();
    return true;
}

std::string guessMemoryKind(const std::string& ggufPath)
{
    gguf_init_params init {};
    init.no_alloc = true;
    init.ctx = nullptr;
    gguf_context* ctx = gguf_init_from_file(ggufPath.c_str(), init);
    if (!ctx)
        return "attention";
    bool recurrent = false;
    bool attention = false;
    bool swa = false;
    const int64_t count = gguf_get_n_kv(ctx);
    for (int64_t i = 0; i < count; ++i) {
        const std::string key = gguf_get_key(ctx, i);
        if (key.find(".ssm.") != std::string::npos || key.find(".wkv.") != std::string::npos || key.find(".shortconv.") != std::string::npos
            || key.find("rwkv") != std::string::npos)
            recurrent = true;
        if (key.find(".attention.head_count") != std::string::npos)
            attention = true;
        if (key.find(".attention.sliding_window") != std::string::npos)
            swa = true;
    }
    gguf_free(ctx);
    if (recurrent)
        return attention ? "hybrid" : "recurrent";
    return swa ? "swa" : "attention";
}

std::unique_ptr<Runtime> Runtime::load(const LoadOptions& options, flr_progress_fn progress, void* user, LoadError& error)
{
    const auto fail = [&error](int code, const std::string& message) -> std::unique_ptr<Runtime> {
        error.code = code;
        error.message = message;
        logf(FLR_LOG_ERROR, "模型加载失败：%s", message.c_str());
        return nullptr;
    };
    const int64_t started = ggml_time_us();
    const uint64_t mark = logMark();
    std::unique_ptr<Runtime> runtime(new Runtime());
    runtime->m_modelId = options.modelId;

    if (options.modelPath.empty())
        return fail(FLR_INVALID_REQUEST, "没有给模型文件");
    if (!fileExists(options.modelPath))
        return fail(FLR_IO_ERROR, "找不到模型文件：" + options.modelPath);
    if (!options.mmprojPath.empty() && !fileExists(options.mmprojPath))
        return fail(FLR_IO_ERROR, "找不到视觉投影文件：" + options.mmprojPath);

    // 序列数：纯注意力模型的序列不占额外显存（统一 KV 池里只是格子上的标记），多留一些缓存着的对话 / 前缀；
    // 混合 / 循环模型每个序列一份循环状态（Qwen3.5-9B 约 50 MiB），滑窗模型每个序列一份窗口，少留
    const int parallel = std::max(1, options.parallel);
    const std::string guessed = guessMemoryKind(options.modelPath);
    int sequences = options.sequences > 0 ? options.sequences : (guessed == "attention" ? 32 : parallel + 4);
    sequences = std::max(sequences, parallel);
    // llama 的序列号上限 LLAMA_MAX_SEQ = 256（src/llama-cparams.h，不在公开头文件里），留一个给工作序列
    sequences = std::min(sequences, 255);

    // MARK: 参数（== llama-server 的命令行；-np 是序列数，内部工作序列下面再加）
    // fit 显存不够时先把上下文往下压，最低压到 --fit-ctx（llama.cpp 缺省 4096），还不够才把层挪到内存里。
    // Friday 的系统提示加工具定义就有一万多 token，4096 的池子什么请求都装不下：宁可少放几层到显卡，也保住 32K
    // （为生图 / 生视频留显存时就会压到这里；模型的训练上下文本来就更小时 fit 不改它）。FRIDAY_LLAMA_ARGS 可以覆盖
    std::vector<std::string> args { "-m", options.modelPath, "-np", std::to_string(sequences), "-kvu", "--jinja", "--fit-ctx", "32768" };
    if (!options.mmprojPath.empty()) {
        args.push_back("--mmproj");
        args.push_back(options.mmprojPath);
    }
    if (options.contextTokens > 0) {
        args.push_back("-c");
        args.push_back(std::to_string(options.contextTokens));
    }
    if (options.perSequence > 0) {
        args.push_back("--kv-unified-per-slot");
        args.push_back(std::to_string(options.perSequence));
    }
    args.insert(args.end(), options.extraArgs.begin(), options.extraArgs.end());

    std::vector<std::string> argv;
    std::string message;
    if (!prepareArguments(args, argv, &message))
        return fail(FLR_INVALID_REQUEST, message);
    std::vector<char*> pointers;
    for (std::string& arg : argv)
        pointers.push_back(arg.data());
    pointers.push_back(nullptr);
    common_params& params = runtime->m_params;
    const bool parsed = common_params_parse(int(argv.size()), pointers.data(), params, LLAMA_EXAMPLE_SERVER);
    silenceCommonLog();
    if (!parsed)
        return fail(FLR_INVALID_REQUEST, "模型参数不对");
    const std::string mmprojRequest = options.mmprojPath;

    LoadProgress loadProgress;
    loadProgress.fn = progress;
    loadProgress.user = user;
    if (!loadProgress.report(0.0f))
        return fail(FLR_CANCELLED, "加载已取消");

    // 池子大小：给了 -c 用它；没给就是模型的训练上下文，fit 按显存往下调（序列之间共用前缀，池子不必是 序列数 × 单序列上限）
    if (params.n_parallel < 1)
        params.n_parallel = sequences;
    if (!mmprojRequest.empty() && params.mmproj.path.empty())
        params.mmproj.path = mmprojRequest;
    const bool hasMmproj = !params.mmproj.path.empty();
    runtime->m_modelPath = params.model.path;
    runtime->m_mmprojPath = params.mmproj.path;

    // 序列：np 个 + 内部工作序列。统一 KV 池：序列之间 seq_cp 只改元数据、不占池（不统一的话 n_seq_max 会把池子分成几份）
    const int nSeq = std::max(1, params.n_parallel);
    params.n_parallel = nSeq + 1;
    params.kv_unified = true;
    // 建上下文时 llama 先 output_reserve(n_seq_max)，要求 n_outputs_max ≥ n_seq_max（llama-context.cpp:376 → 2258，不满足就
    // GGML_ASSERT、整个应用退出）。多了一个工作序列，FRIDAY_LLAMA_ARGS 给的 -b 小于序列数时（-b 4 -np 4：llama-server 照样起得来）
    // 把批大小提上来（评审修正）
    if (params.n_batch < params.n_parallel) {
        logf(FLR_LOG_NOTICE, "批大小 -b %d 小于序列数 %d（-np %d + 工作序列 1），提到 %d", params.n_batch, params.n_parallel, nSeq,
             params.n_parallel);
        params.n_batch = params.n_parallel;
    }
    // == server_output_limits（无投机解码）：每个序列一行 logits（n_vocab = 248320 时 n_batch 行会占 2 GB）
    params.n_outputs_max = std::min(params.n_batch, params.n_parallel);
    params.n_outputs_max_per_seq = 1;
    // 运行时自己采样（common_sampler），不用后端采样
    params.sampling.backend_sampling = false;
    if (hasMmproj)
        Media::reserveForFit(params.mmproj.path, params);

    loadProgress.stages = hasMmproj ? 2 : 1;
    params.load_progress_callback = &LoadProgress::trampoline;
    params.load_progress_callback_user_data = &loadProgress;

    logf(FLR_LOG_NOTICE, "加载模型：%s（%s，-c %d，序列 %d + 工作序列 1，同时 %d 个请求，-b %d，-ub %d，fit %s）", params.model.path.c_str(),
         guessed.c_str(), params.n_ctx, nSeq, parallel, params.n_batch, params.n_ubatch, params.fit_params ? "开" : "关");
    const double vramBefore = usedVramMiB();
    runtime->m_init = common_init_from_params(params);
    params.load_progress_callback = nullptr;
    params.load_progress_callback_user_data = nullptr;
    runtime->m_model = runtime->m_init ? runtime->m_init->model() : nullptr;
    runtime->m_ctx = runtime->m_init ? runtime->m_init->context() : nullptr;
    if (loadProgress.cancelled)
        return fail(FLR_CANCELLED, "加载已取消");
    if (!runtime->m_model) {
        const std::string reason = lastErrorSince(mark);
        return fail(FLR_IO_ERROR, "读不了模型文件" + (reason.empty() ? std::string() : "：" + reason));
    }
    if (!runtime->m_ctx) {
        const std::string reason = lastErrorSince(mark);
        return fail(FLR_INTERNAL, "建推理上下文失败" + (reason.empty() ? std::string() : "：" + reason));
    }
    llama_model* model = runtime->m_model;
    llama_context* ctx = runtime->m_ctx;
    runtime->m_vocab = llama_model_get_vocab(model);
    const double tModel = double(ggml_time_us() - started) / 1000.0;

    // MARK: mmproj（WP4）
    if (hasMmproj) {
        loadProgress.stage = 1;
        std::string mediaError;
        runtime->m_media = Media::load(params.mmproj.path, model, params,
                                       [&loadProgress](float value) { return loadProgress.report(value); }, &mediaError,
                                       int(llama_n_ubatch(ctx)));
        if (loadProgress.cancelled)
            return fail(FLR_CANCELLED, "加载已取消");
        if (!runtime->m_media)
            return fail(FLR_IO_ERROR, "视觉投影（mmproj）加载失败" + (mediaError.empty() ? std::string() : "：" + mediaError));
    }

    // 只测一次：会清空 memory。Qwen3.5（混合）= FULL（不能部分截断）
    runtime->m_seqRm = common_context_can_seq_rm(ctx);

    // MARK: 聊天模板（== server init 的 chat_params）
    try {
        runtime->m_templates = common_chat_templates_init(model, params.chat_template);
    } catch (const std::exception& exception) {
        return fail(FLR_INVALID_REQUEST, std::string("聊天模板解析失败：") + exception.what());
    }
    if (!runtime->m_templates)
        return fail(FLR_INVALID_REQUEST, "聊天模板解析失败");
    runtime->m_caps = common_chat_templates_get_caps(runtime->m_templates.get());
    runtime->m_supportsThinking = params.use_jinja && common_chat_templates_support_enable_thinking(runtime->m_templates.get());

    // MARK: 尺寸
    runtime->m_nCtx = int(llama_n_ctx(ctx));
    runtime->m_nBatch = int(llama_n_batch(ctx));
    runtime->m_nUbatch = int(llama_n_ubatch(ctx));
    runtime->m_nSeqMax = int(llama_n_seq_max(ctx));
    runtime->m_nSeq = std::max(1, runtime->m_nSeqMax - 1);
    runtime->m_nParallel = std::min(parallel, runtime->m_nSeq);
    runtime->m_nCtxTrain = llama_model_n_ctx_train(model);
    // == server n_ctx_slot()：单序列上限 = min(池子能给一个序列的, --kv-unified-per-slot, 训练上下文)
    int nCtxSeq = int(llama_n_ctx_seq(ctx));
    if (params.kv_unified_per_slot > 0)
        nCtxSeq = std::min(nCtxSeq, params.kv_unified_per_slot);
    if (runtime->m_nCtxTrain > 0)
        nCtxSeq = std::min(nCtxSeq, runtime->m_nCtxTrain);
    runtime->m_nCtxSeq = nCtxSeq;
    runtime->m_memoryKind = memoryKindOf(model);
    // 滑动窗口注意力（Gemma 3 一类，没开 --swa-full）：common 测出来是 PART，但窗口外的 KV 已经裁掉，从中间截断后接着算会缺 KV
    // （llama-server 按 pos_min 判断后改走检查点）。按不能截断处理：复用走检查点（PARTIAL_ONLY 存的就是 SWA 那部分），
    // flr_info.seq_rm 也报 full，应用按混合模型的规则判断要不要恢复对话缓存（WP3）
    if (runtime->m_seqRm == COMMON_CONTEXT_SEQ_RM_TYPE_PART && runtime->m_memoryKind == "swa" && !params.swa_full)
        runtime->m_seqRm = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    runtime->m_memory = makeMemoryOps(ctx, runtime->m_seqRm);
    measureState(ctx, runtime->m_vocab, runtime->m_nSeqMax - 1, runtime->m_kvBytesPerToken, runtime->m_stateBytesPerSeq);
    // 纯注意力模型没有「不能回退的那部分」（PARTIAL_ONLY 对它整段都算），每序列的固定状态记 0
    if (runtime->m_memoryKind == "attention")
        runtime->m_stateBytesPerSeq = 0;
    runtime->m_modelSha = modelFingerprint(params.model.path);

    // MARK: 模板 / 采样 / 解析环境（== server 的 chat_params + params_base）
    ChatEnvironment environment;
    environment.model = model;
    environment.vocab = runtime->m_vocab;
    environment.templates = runtime->m_templates.get();
    environment.caps = runtime->m_caps;
    environment.useJinja = params.use_jinja;
    environment.prefillAssistant = params.prefill_assistant;
    environment.enableThinking = params.enable_reasoning != 0 && runtime->m_supportsThinking;
    environment.forcePureContent = params.force_pure_content_parser;
    environment.special = params.special;
    environment.reasoningFormat = params.reasoning_format;
    environment.templateKwargs = params.default_template_kwargs;
    environment.reasoningBudget = params.sampling.reasoning_budget_tokens;
    environment.reasoningBudgetMessage = params.sampling.reasoning_budget_message;
    environment.sampling = params.sampling;
    environment.antiprompt = params.antiprompt;
    environment.nPredict = params.n_predict;
    environment.media = runtime->m_media.get();
    environment.modelName = runtime->m_modelId.empty() ? std::filesystem::u8path(params.model.path).filename().u8string() : runtime->m_modelId;
    runtime->m_format = std::make_unique<ChatFormat>(std::move(environment));

    // MARK: 推理线程
    Scheduler::Config config;
    config.ctx = ctx;
    config.vocab = runtime->m_vocab;
    config.memory = runtime->m_memory.get();
    config.media = runtime->m_media.get();
    config.nBatch = runtime->m_nBatch;
    config.nUbatch = runtime->m_nUbatch;
    config.nCtxSeq = runtime->m_nCtxSeq;
    config.nSeq = runtime->m_nSeq;
    config.nSeqMax = runtime->m_nSeqMax;
    config.nParallel = runtime->m_nParallel;
    config.special = params.special;
    config.modelName = runtime->m_format->environment().modelName;
    runtime->m_scheduler = std::make_unique<Scheduler>(std::move(config));
    runtime->m_scheduler->start();

    loadProgress.stage = loadProgress.stages - 1;
    loadProgress.report(1.0f);
    runtime->m_loadMs = double(ggml_time_us() - started) / 1000.0;
    logf(FLR_LOG_NOTICE,
         "模型已加载：%.0f ms（主模型 %.0f ms），显存 %.0f → %.0f MiB；n_ctx %d，单序列上限 %d，序列 %d + 1（同时 %d 个请求），n_batch %d，"
         "n_ubatch %d，memory %s，seq_rm %s，KV %.1f KiB/token，循环状态 %.2f MiB/序列，思考 %s，视觉 %s",
         runtime->m_loadMs, tModel, vramBefore, usedVramMiB(), runtime->m_nCtx, runtime->m_nCtxSeq, runtime->m_nSeq, runtime->m_nParallel,
         runtime->m_nBatch, runtime->m_nUbatch, runtime->m_memoryKind.c_str(), seqRmName(runtime->m_seqRm),
         double(runtime->m_kvBytesPerToken) / 1024.0, double(runtime->m_stateBytesPerSeq) / 1048576.0,
         runtime->m_supportsThinking ? "支持" : "不支持", runtime->m_media && runtime->m_media->supportsImages() ? "有" : "无");
    return runtime;
}

Runtime::~Runtime()
{
    shutdown();
    // 先停调度器（它用着上下文），再放模板、媒体、上下文与模型
    m_scheduler.reset();
    m_format.reset();
    m_memory.reset();
    m_media.reset();
    m_templates.reset();
    m_init.reset();
}

void Runtime::shutdown()
{
    {
        std::lock_guard<std::mutex> lock(m_callMutex);
        if (m_closing)
            return;
        m_closing = true;
    }
    // 在途请求以 FLR_NOT_READY 结束（正在跑的 llama_process 经 abort 回调中止），推理线程跑完当前一步退出
    if (m_scheduler)
        m_scheduler->stop();
    // 等所有还在 flr_* 里的调用方离开
    std::unique_lock<std::mutex> lock(m_callMutex);
    m_callsDone.wait(lock, [this] { return m_calls == 0; });
}

bool Runtime::enter()
{
    std::lock_guard<std::mutex> lock(m_callMutex);
    if (m_closing)
        return false;
    ++m_calls;
    return true;
}

void Runtime::leave()
{
    std::lock_guard<std::mutex> lock(m_callMutex);
    if (--m_calls == 0)
        m_callsDone.notify_all();
}

json Runtime::info() const
{
    json caps = json::object();
    for (const auto& [name, value] : m_caps)
        caps[name] = value;
    json result {
        { "model_id", m_modelId },
        { "model_path", m_modelPath },
        { "mmproj_path", m_mmprojPath },
        { "load_ms", m_loadMs },
        { "n_ctx", m_nCtx },
        { "n_ctx_seq", m_nCtxSeq },
        { "n_ctx_train", m_nCtxTrain },
        { "parallel", m_nParallel },
        { "sequences", m_nSeq },
        { "vision", m_media && m_media->supportsImages() },
        { "memory", m_memoryKind },
        { "seq_rm", seqRmName(m_seqRm) },
        { "supports_thinking", m_supportsThinking },
        { "template_caps", caps },
        { "kv_bytes_per_token", m_kvBytesPerToken },
        { "state_bytes_per_seq", m_stateBytesPerSeq },
        { "n_batch", m_nBatch },
        { "n_ubatch", m_nUbatch },
    };
    if (m_scheduler)
        result["cache"] = m_scheduler->stats();
    return result;
}

bool Runtime::busy() const
{
    return m_scheduler && m_scheduler->busy();
}

int Runtime::chat(json body, flr_chunk_fn chunk, flr_should_stop_fn shouldStop, void* user, json& result)
{
    const int64_t started = ggml_time_us();
    ChatRequest request;
    std::string error;
    const int status = m_format->prepare(std::move(body), request, &error);
    if (status != FLR_OK) {
        result = errorJson(-1, error);
        return status;
    }
    const int64_t preparedAt = ggml_time_us();
    const Prepared& prepared = request.prepared;
    if (prepared.items.empty()) {
        result = errorJson(-1, "提示为空");
        return FLR_INVALID_REQUEST;
    }
    // 提示超过单序列上限：不进推理线程（== server 的 exceed_context_size_error，带两个数）
    if (request.promptTokens >= m_nCtxSeq) {
        result = json {
            { "error",
              json {
                  { "message", format("request (%lld tokens) exceeds the available context size (%d tokens), try increasing it",
                                      (long long) request.promptTokens, m_nCtxSeq) },
                  { "type", "exceed_context_size_error" },
                  { "n_prompt_tokens", request.promptTokens },
                  { "n_ctx", m_nCtxSeq },
              } },
        };
        return FLR_CONTEXT_OVERFLOW;
    }
    std::optional<Sampler> sampler;
    try {
        sampler.emplace(m_model, request.prepared.sampling);
        sampler->acceptPrompt(request.prepared.items);
    } catch (const std::exception& exception) {
        result = errorJson(-1, std::string("Failed to initialize samplers: ") + exception.what());
        return FLR_INVALID_REQUEST;
    }
    request.prepareMs = double(preparedAt - started) / 1000.0;
    request.samplerMs = double(ggml_time_us() - preparedAt) / 1000.0;
    logf(FLR_LOG_DEBUG,
         "请求（会话 %s，优先级 %d）：%zu 项（%lld token），准备 %.1f ms（模板 %.1f、分词 %.1f、采样参数 %.1f、边界 %.1f），采样器 %.1f ms，"
         "P %d，B %d，%s，max_tokens %d，temperature %.2f",
         request.session.empty() ? "-" : request.session.c_str(), request.priority, prepared.items.size(), (long long) request.promptTokens,
         request.prepareMs, prepared.t_template_ms, prepared.t_tokenize_ms, request.paramsMs, request.boundaryMs, request.samplerMs,
         prepared.prefix_boundary, prepared.transcript_boundary, prepared.thinking ? "思考开" : "思考关", prepared.n_predict,
         prepared.sampling.temp);
    return m_scheduler->chat(std::move(request), std::move(*sampler), chunk, shouldStop, user, result);
}

int Runtime::withSequence(int seq, const std::function<int(SequenceCache& cache, MemoryOps& memory)>& job)
{
    if (!m_scheduler)
        return FLR_NOT_READY;
    return m_scheduler->withSequence(seq, job);
}

bool Runtime::evictIdleSequence(int keep)
{
    return m_scheduler && m_scheduler->evictIdleFromJob(keep);
}

} // namespace flr
