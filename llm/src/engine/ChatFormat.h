// 聊天模板、请求解析与流式 chunk。照 llama-server（b11541）的 /v1/chat/completions（--jinja）：
//   ChatFormat::prepare == oaicompat_chat_params_parse（server-common.cpp）+ 采样参数（server-schema.cpp）+ 分词 + 检查点边界 P / B，
//                          在调用线程上做（模板 2–3 ms、6k token 分词约 6 ms）；模板渲染、工具调用 / 思考的解析器都在 common_chat_session 里；
//   ChatStream           == task_result_state::update_chat_msg（server-task.cpp）+ to_json_oaicompat_chat(_stream)：
//                          生成的文字逐段喂给 common_chat_session，增量解析成 OpenAI 流式 chunk，也在调用线程上；
//   RequestStats         == server_slot_stats：timings 的几个数与算法。
// v2.7 加：请求里的 x_friday（session / priority / main，docs/llm-protocol.md），调度器按它挑序列、排队。
#pragma once

#include "internal.h"
#include "Sampler.h"

#include <cstdint>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace flr {

/// 模型级的模板 / 采样 / 解析环境：加载时建好，之后只读，多个调用线程共用（== server 的 params_base + chat_params）
struct ChatEnvironment {
    const llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    const common_chat_templates* templates = nullptr;
    /// common_chat_templates_get_caps
    std::map<std::string, bool> caps;
    bool useJinja = true;
    /// 结尾是 assistant 消息时接着它写（params.prefill_assistant）
    bool prefillAssistant = true;
    /// 模板支持 enable_thinking 且没有 --reasoning off（== server chat_params.enable_thinking）
    bool enableThinking = false;
    bool forcePureContent = false;
    /// params.special：生成的特殊 token 照样输出文字
    bool special = false;
    common_reasoning_format reasoningFormat = COMMON_REASONING_FORMAT_DEEPSEEK;
    /// 命令行的模板 kwargs，值是 JSON 文本
    std::map<std::string, std::string> templateKwargs;
    int reasoningBudget = -1;
    std::string reasoningBudgetMessage;
    /// 采样默认值（common_init_from_params 按模型元数据改过之后的 params.sampling，含 logit_bias_eog）
    common_params_sampling sampling;
    /// 命令行的停止词（请求没给有效停止词时用）
    std::vector<std::string> antiprompt;
    /// 命令行的 -n（请求没给 max_tokens 时用）
    int nPredict = -1;
    /// 看图；没有 mmproj 时为空
    Media* media = nullptr;
    /// chunk 的 model 字段（Friday 的模型 id）
    std::string modelName;
};

/// prepare 的结果
struct ChatRequest {
    Prepared prepared;
    /// 模板渲染的结果与这次生成的解析器（工具调用、思考块），交给 ChatStream
    common_chat_session chatSession;
    /// 请求的 stream（false：攒成一个 chat.completion 一次回）
    bool stream = false;
    /// 请求的 timings_per_token：每一块都带 timings（== server；缺省只有停止词那块与最后一块带）
    bool timingsPerToken = false;
    /// 提示占的 KV 单元数（Σ items.n_tokens）：单序列上限判断、usage.prompt_tokens
    int64_t promptTokens = 0;
    /// x_friday.session：同一段对话 / 同一个子智能体的请求带同一个值，调度器把它们放在同一个序列上接着算（docs/llm-protocol.md）
    std::string session;
    /// x_friday.priority：排队时大的先（主智能体 1000，子智能体用层数）
    int priority = 0;
    /// x_friday.main：主智能体的对话（KV 池满时最后腾它）
    bool mainAgent = false;
    /// 调用线程上的准备用时（解析 + 模板 + 分词 + 边界、建采样器），日志用
    double prepareMs = 0;
    double samplerMs = 0;
    /// prepare 里各段的用时：采样参数、检查点边界（P / B）
    double paramsMs = 0;
    double boundaryMs = 0;
};

class ChatFormat {
public:
    explicit ChatFormat(ChatEnvironment environment);

    /// 调用线程：OpenAI chat 请求体 → 模板 → 分词 → 采样参数 → P / B。
    /// 返回 FLR_OK / FLR_INVALID_REQUEST / FLR_UNSUPPORTED_MEDIA / FLR_INTERNAL，error 写原因（== llama-server 的报错原文）
    int prepare(json body, ChatRequest& out, std::string* error) const;

    const ChatEnvironment& environment() const { return m_env; }

private:
    /// P：两个只差用户消息的探针（「甲」「乙」）不带生成提示渲染、分词，取公共前缀，再与本轮提示取公共前缀
    /// （== Swift ChatSession.systemPrefixBoundary）。探针结果按「系统消息 + 工具 + 模板参数」缓存（新请求只做一次比较）
    int prefixBoundary(const common_chat_templates_inputs& inputs, const std::vector<LedgerItem>& items) const;
    /// B：生成提示之前（不带生成提示的那段分词后是 items 的严格前缀）；不是就 -1。
    /// withMedia：提示里有媒体块（items 不是纯文本，不能重新分词比对）——只走快路（items 的末尾正好是生成提示自己的分词）
    int transcriptBoundary(const common_chat_templates_inputs& inputs, const std::string& prompt, const std::string& generation,
                           const std::vector<LedgerItem>& items, bool withMedia) const;

    ChatEnvironment m_env;

    struct ProbeEntry {
        std::string key;
        std::vector<llama_token> tokens;
    };
    mutable std::mutex m_mutex;
    mutable std::list<ProbeEntry> m_probes;
};

/// == server_slot_stats：时间戳是 ggml_time_us
struct RequestStats {
    uint64_t promptCached = 0;
    uint64_t promptProcessed = 0;
    uint64_t generated = 0;
    int64_t tStart = 0;
    int64_t tPromptLast = 0;
    int64_t tGenLast = 0;

    double promptMs() const { return tPromptLast == 0 ? 0.0 : double(tPromptLast - tStart) / 1000.0; }
    int64_t genUs() const;
    double genMs() const { return double(genUs()) / 1000.0; }
    /// 生成用的 decode 步数（第一个 token 来自提示最后一批的 logits，不算）
    uint64_t genSteps() const { return generated > 0 ? generated - 1 : 0; }
    double promptPerSecond() const;
    double genPerSecond() const;
    /// {"cache_n","prompt_n","prompt_ms","prompt_per_token_ms","prompt_per_second","predicted_n","predicted_ms",
    ///  "predicted_per_token_ms","predicted_per_second"}（== server_slot_stats::to_json）
    json toJson() const;
};

/// 一次请求的流式解析与 chunk（调用线程）
class ChatStream {
public:
    ChatStream(common_chat_session session, std::string completionId, std::string model);

    /// 一段生成的文字（带 token，可能为空）→ 0..n 个 chunk 的 JSON 文本。nDecoded == 1 时先发 {"role":"assistant","content":null}；
    /// stats 非空（停止词那一块）时挂在最后一块上。解析出错抛 std::exception
    void partial(const common_chat_input& text, uint64_t nDecoded, const RequestStats* stats, std::vector<std::string>& chunks);
    /// 结束：最终解析（is_partial = false）的增量 + finish_reason 那块 + usage 块（带 timings）。返回 finish_reason
    std::string finish(StopType stop, int64_t promptTokens, const RequestStats& stats, std::vector<std::string>& chunks);
    /// 不流式时的整个回复（== to_json_oaicompat_chat）：finish 之后调用
    json completion(const std::string& finishReason, int64_t promptTokens, const RequestStats& stats) const;

    const common_chat_msg& message() const { return m_message; }

    /// usage（== server_task_result_cmpl_final::usage_json_oaicompat）
    static json usage(int64_t promptTokens, const RequestStats& stats);
    /// 新的 chunk / 工具调用 id（== gen_chatcmplid / gen_tool_call_id：32 位字母数字）
    static std::string randomId();

private:
    std::vector<common_chat_msg_diff> update(const common_chat_input& text, bool partial);
    json chunk(json choices) const;
    static json deltaJson(const common_chat_msg_diff& diff);

    common_chat_session m_session;
    common_chat_msg m_message;
    std::vector<std::string> m_toolCallIds;
    std::string m_id;
    std::string m_model;
    std::string m_fingerprint;
    bool m_finished = false;
};

} // namespace flr
