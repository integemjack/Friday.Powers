// 聊天模板、请求解析与流式 chunk（见 ChatFormat.h）。
// 照抄 llama.cpp b11541 tools/server 的语义（MIT License，Copyright (c) 2023-2026 The ggml authors）：
// oaicompat_chat_params_parse、server_schema 的采样字段、task_result_state、server_task_result_cmpl_*::to_json_oaicompat_chat*。
#include "ChatFormat.h"

#include "RuntimeLog.h"

#include "build-info.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <random>
#include <stdexcept>

namespace flr {
namespace {

std::string fieldError(const char* name, const std::string& what)
{
    return std::string("Field '") + name + "': " + what;
}

template <class T>
std::string numberText(T value)
{
    if constexpr (std::is_floating_point<T>::value) {
        char text[64];
        std::snprintf(text, sizeof(text), "%f", double(value));
        return text;
    } else {
        return std::to_string(value);
    }
}

/// 数值字段（== server_schema::field_num::eval）：第一个有值（非 null）的名字生效；hard = 越界报错，否则夹到范围里
template <class T>
void readNumber(const json& data, std::initializer_list<const char*> names, T& value, T low = std::numeric_limits<T>::lowest(),
                T high = std::numeric_limits<T>::max(), bool hard = false)
{
    for (const char* name : names) {
        const auto it = data.find(name);
        if (it == data.end() || it->is_null())
            continue;
        T read {};
        try {
            read = it->template get<T>();
        } catch (const std::exception& error) {
            throw std::invalid_argument(fieldError(name, error.what()));
        }
        if (hard) {
            if (read < low || read > high) {
                throw std::invalid_argument(fieldError(name, "Value must be between " + numberText(low) + " <= value <= " + numberText(high)
                                                                 + ", but got " + numberText(read)));
            }
            value = read;
        } else {
            value = std::max(low, std::min(high, read));
        }
        return;
    }
}

void readBool(const json& data, const char* name, bool& value)
{
    const auto it = data.find(name);
    if (it == data.end() || it->is_null())
        return;
    try {
        value = it->get<bool>();
    } catch (const std::exception& error) {
        throw std::invalid_argument(fieldError(name, error.what()));
    }
}

/// == server 的 json_value：没有、null 或类型不对都用缺省值
template <class T>
T valueOr(const json& data, const char* key, const T& fallback)
{
    const auto it = data.find(key);
    if (it == data.end() || it->is_null())
        return fallback;
    try {
        return it->template get<T>();
    } catch (const std::exception&) {
        return fallback;
    }
}

/// 交给 common 的 JSON（common_json，llama-common.dll 里的另一份 nlohmann）
common_json toCommon(const json& value)
{
    return common_json::parse(value.dump());
}

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text)
{
    return common_tokenize(vocab, text, true, true);
}

/// 文本 token 的公共前缀长度（遇到媒体块就停）
size_t commonTextPrefix(const std::vector<llama_token>& tokens, const std::vector<LedgerItem>& items)
{
    size_t n = 0;
    while (n < tokens.size() && n < items.size() && !items[n].isMedia() && items[n].token == tokens[n])
        ++n;
    return n;
}

/// 探针缓存的键：会影响系统提示渲染的全部输入（日期也算：模板可能写今天的日期）
std::string probeKey(const common_chat_templates_inputs& inputs, size_t nSystem)
{
    std::string key;
    key.reserve(4096);
    const auto add = [&key](const std::string& part) {
        key += std::to_string(part.size());
        key += ':';
        key += part;
    };
    for (size_t i = 0; i < nSystem; ++i) {
        const common_chat_msg& message = inputs.messages[i];
        add(message.role);
        add(message.content);
        for (const common_chat_msg_content_part& part : message.content_parts) {
            add(part.type);
            add(part.text);
        }
    }
    key += "|tools";
    for (const common_chat_tool& tool : inputs.tools) {
        add(tool.name);
        add(tool.description);
        add(tool.parameters);
    }
    key += "|kwargs";
    for (const auto& [name, value] : inputs.chat_template_kwargs) {
        add(name);
        add(value);
    }
    add(inputs.json_schema);
    add(inputs.grammar);
    key += '|' + std::to_string(int(inputs.tool_choice)) + ',' + std::to_string(int(inputs.parallel_tool_calls)) + ','
        + std::to_string(int(inputs.reasoning_format)) + ',' + std::to_string(int(inputs.enable_thinking)) + ','
        + std::to_string(int(inputs.use_jinja)) + ',' + std::to_string(int(inputs.force_pure_content));
    const std::time_t now = std::chrono::system_clock::to_time_t(inputs.now);
    std::tm local {};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char day[32];
    std::strftime(day, sizeof(day), "%Y-%m-%d", &local);
    key += day;
    return key;
}

} // namespace

ChatFormat::ChatFormat(ChatEnvironment environment)
    : m_env(std::move(environment))
{
}

int ChatFormat::prepare(json body, ChatRequest& out, std::string* error) const
{
    const auto fail = [error](int code, const std::string& message) {
        if (error)
            *error = message;
        return code;
    };
    if (!body.is_object())
        return fail(FLR_INVALID_REQUEST, "请求体不是 JSON 对象");
    try {
        Prepared& p = out.prepared;
        const double t0 = double(ggml_time_us());

        // MARK: oaicompat_chat_params_parse
        const json tools = body.contains("tools") ? body["tools"] : json();
        const bool hasTools = tools.is_array() && !tools.empty();
        const std::string toolChoice = valueOr<std::string>(body, "tool_choice", "auto");
        if (!m_env.useJinja) {
            if (hasTools)
                throw std::runtime_error("tools param requires --jinja flag");
            if (toolChoice != "auto")
                throw std::runtime_error("tool_choice param requires --jinja flag");
        }

        // 停止词：请求的 stop（字符串或数组）+ 模板的 additional_stops
        json stop = json::array();
        if (body.contains("stop") && body["stop"].is_string())
            stop.push_back(body["stop"].get<std::string>());
        else if (body.contains("stop") && body["stop"].is_array())
            stop = body["stop"];

        json jsonSchema = body.contains("json_schema") ? body["json_schema"] : json();
        const std::string grammar = valueOr<std::string>(body, "grammar", std::string());
        if (!jsonSchema.is_null() && !grammar.empty())
            throw std::runtime_error("Cannot use both json_schema and grammar");
        if (body.contains("response_format")) {
            const json responseFormat = valueOr<json>(body, "response_format", json::object());
            const std::string type = valueOr<std::string>(responseFormat, "type", std::string());
            if (type == "json_object") {
                if (responseFormat.contains("schema") || jsonSchema.empty())
                    jsonSchema = valueOr<json>(responseFormat, "schema", json::object());
            } else if (type == "json_schema") {
                const json wrapper = valueOr<json>(responseFormat, "json_schema", json::object());
                jsonSchema = valueOr<json>(wrapper, "schema", json::object());
            } else if (!type.empty() && type != "text") {
                throw std::invalid_argument("response_format type must be one of \"text\" or \"json_object\", but got: " + type);
            }
        }
        if (jsonSchema.is_object() && jsonSchema.empty())
            jsonSchema["type"] = "object";

        if (!body.contains("messages"))
            throw std::invalid_argument("'messages' is required");
        json& messages = body["messages"];
        if (!messages.is_array())
            throw std::invalid_argument("Expected 'messages' to be an array");
        for (json& message : messages) {
            const std::string role = valueOr<std::string>(message, "role", std::string());
            if (role != "assistant" && !message.contains("content"))
                throw std::invalid_argument("All non-assistant messages must contain 'content'");
            if (role == "assistant") {
                if (!message.contains("content") && !message.contains("tool_calls"))
                    throw std::invalid_argument("Assistant message must contain either 'content' or 'tool_calls'!");
                if (!message.contains("content"))
                    continue;
            }
            const json& content = message["content"];
            if (content.is_string() || content.is_null())
                continue;
            if (!content.is_array())
                throw std::invalid_argument("Expected 'content' to be a string or an array");
        }

        // 媒体（WP4）：image_url / input_audio 解码进 inputs，消息里换成随机标记
        std::vector<MediaInput> mediaInputs;
        {
            std::string mediaError;
            const bool images = m_env.media && m_env.media->supportsImages();
            const bool audio = m_env.media && m_env.media->supportsAudio();
            const std::string marker = m_env.media ? m_env.media->marker() : std::string();
            const int status = media::extract(messages, marker, images, audio, mediaInputs, &mediaError);
            if (status != FLR_OK)
                return fail(status, mediaError.empty() ? std::string("请求里的媒体不能处理") : mediaError);
        }

        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(toCommon(messages));
        if (tools.is_array())
            inputs.tools = common_chat_tools_parse_oaicompat(toCommon(tools));
        inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(toolChoice);
        inputs.json_schema = jsonSchema.is_null() ? std::string() : jsonSchema.dump();
        inputs.grammar = grammar;
        inputs.use_jinja = m_env.useJinja;
        const auto capability = [this](const char* name) {
            const auto it = m_env.caps.find(name);
            return it != m_env.caps.end() && it->second;
        };
        inputs.parallel_tool_calls = valueOr<bool>(body, "parallel_tool_calls", capability("supports_parallel_tool_calls"));
        inputs.add_generation_prompt = valueOr<bool>(body, "add_generation_prompt", true);
        inputs.continue_final_message = body.contains("continue_final_message")
            ? common_chat_continuation_parse(toCommon(body["continue_final_message"]))
            : COMMON_CHAT_CONTINUATION_NONE;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_NONE && m_env.prefillAssistant && !inputs.messages.empty()
            && inputs.messages.back().role == "assistant") {
            if (inputs.messages.size() >= 2 && inputs.messages[inputs.messages.size() - 2].role == "assistant")
                throw std::invalid_argument("Cannot have 2 or more assistant messages at the end of the list.");
            inputs.continue_final_message = COMMON_CHAT_CONTINUATION_AUTO;
            inputs.add_generation_prompt = false;
        }
        if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && inputs.add_generation_prompt)
            throw std::invalid_argument("Cannot set both add_generation_prompt and continue_final_message to true.");
        if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !inputs.messages.empty()
            && inputs.messages.back().role == "assistant" && !inputs.messages.back().tool_calls.empty())
            throw std::invalid_argument("Cannot continue an assistant message that contains tool calls.");
        inputs.reasoning_format = m_env.reasoningFormat;
        if (body.contains("reasoning_format") && body["reasoning_format"].is_string())
            inputs.reasoning_format = common_reasoning_format_from_name(body["reasoning_format"].get<std::string>());
        inputs.enable_thinking = m_env.enableThinking;
        if (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && body.contains("grammar"))
            throw std::invalid_argument("Cannot use custom grammar constraints with tools.");

        // 模板 kwargs：命令行的 + 请求的（值按 JSON 文本存）
        inputs.chat_template_kwargs = m_env.templateKwargs;
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
            for (const auto& item : body["chat_template_kwargs"].items())
                inputs.chat_template_kwargs[item.key()] = item.value().dump();
        }
        {
            const auto it = inputs.chat_template_kwargs.find("enable_thinking");
            const std::string kwarg = it == inputs.chat_template_kwargs.end() ? std::string() : it->second;
            if (kwarg == "true")
                inputs.enable_thinking = true;
            else if (kwarg == "false")
                inputs.enable_thinking = false;
            else if (!kwarg.empty() && kwarg[0] == '"')
                throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");
        }
        if (body.contains("reasoning_effort")) {
            const std::string effort = valueOr<std::string>(body, "reasoning_effort", std::string());
            if (effort == "none") {
                inputs.enable_thinking = false;
                inputs.chat_template_kwargs.erase("reasoning_effort");
            } else if (!effort.empty()) {
                inputs.chat_template_kwargs["reasoning_effort"] = json(effort).dump();
            }
        }
        inputs.force_pure_content = m_env.forcePureContent;

        common_chat_session_params sessionParams;
        sessionParams.echo = valueOr<bool>(body, "echo", false);
        out.chatSession = common_chat_session(m_env.templates, m_env.vocab, inputs, sessionParams);
        p.prompt = out.chatSession.prompt();
        p.generation_prompt = out.chatSession.generation_prompt();
        const double t1 = double(ggml_time_us());
        p.t_template_ms = (t1 - t0) / 1000.0;

        // MARK: 分词
        p.items.clear();
        if (!mediaInputs.empty()) {
            if (!m_env.media)
                return fail(FLR_UNSUPPORTED_MEDIA, "这个模型不能看图（没有加载视觉投影 mmproj）");
            std::string mediaError;
            const int status = m_env.media->tokenize(p.prompt, mediaInputs, p, &mediaError);
            if (status != FLR_OK)
                return fail(status, mediaError);
        } else {
            const std::vector<llama_token> tokens = tokenize(m_env.vocab, p.prompt);
            p.items.reserve(tokens.size());
            for (const llama_token token : tokens)
                p.items.push_back(LedgerItem::text(token));
        }
        p.t_tokenize_ms = (double(ggml_time_us()) - t1) / 1000.0;
        out.promptTokens = 0;
        for (const LedgerItem& item : p.items)
            out.promptTokens += item.n_tokens;
        if (!p.items.empty() && p.items.back().isMedia())
            throw std::invalid_argument("提示不能以媒体结尾");

        // MARK: eval_llama_cmpl_schema（只取对话补全用得到的字段，名字、范围、报错原文同 server）
        const double t2 = double(ggml_time_us());
        common_params_sampling& sampling = p.sampling;
        sampling = m_env.sampling;
        sampling.grammar = common_grammar();
        sampling.grammar_lazy = false;
        sampling.grammar_triggers.clear();
        sampling.preserved_tokens.clear();
        sampling.generation_prompt.clear();
        sampling.reasoning_budget_start.clear();
        sampling.reasoning_budget_end.clear();
        sampling.reasoning_budget_forced.clear();
        p.n_predict = m_env.nPredict;
        readNumber<int32_t>(body, { "n_predict", "max_completion_tokens", "max_tokens" }, p.n_predict, -1, INT32_MAX, true);
        readNumber<int32_t>(body, { "top_k" }, sampling.top_k, 0, INT32_MAX);
        readNumber<float>(body, { "top_p" }, sampling.top_p, 0.0f, 1.0f);
        readNumber<float>(body, { "min_p" }, sampling.min_p, 0.0f, 1.0f);
        readNumber<float>(body, { "top_n_sigma" }, sampling.top_n_sigma);
        readNumber<float>(body, { "xtc_probability" }, sampling.xtc_probability, 0.0f, 1.0f);
        readNumber<float>(body, { "xtc_threshold" }, sampling.xtc_threshold, 0.0f, 1.0f);
        readNumber<float>(body, { "typical_p" }, sampling.typ_p);
        readNumber<float>(body, { "temperature" }, sampling.temp, 0.0f, std::numeric_limits<float>::infinity());
        readNumber<float>(body, { "dynatemp_range" }, sampling.dynatemp_range);
        readNumber<float>(body, { "dynatemp_exponent" }, sampling.dynatemp_exponent);
        readNumber<int32_t>(body, { "repeat_last_n" }, sampling.penalty_last_n, 0, INT32_MAX, true);
        readNumber<float>(body, { "repeat_penalty" }, sampling.penalty_repeat);
        readNumber<float>(body, { "frequency_penalty" }, sampling.penalty_freq);
        readNumber<float>(body, { "presence_penalty" }, sampling.penalty_present);
        readNumber<float>(body, { "dry_multiplier" }, sampling.dry_multiplier);
        if (body.contains("dry_base") && !body["dry_base"].is_null()) {
            float base = sampling.dry_base;
            readNumber<float>(body, { "dry_base" }, base);
            sampling.dry_base = base < 1.0f ? m_env.sampling.dry_base : base;
        }
        readNumber<int32_t>(body, { "dry_allowed_length" }, sampling.dry_allowed_length, 0, INT32_MAX, true);
        readNumber<int32_t>(body, { "dry_penalty_last_n" }, sampling.dry_penalty_last_n, 0, INT32_MAX, true);
        readNumber<int32_t>(body, { "mirostat" }, sampling.mirostat, 0, 2);
        readNumber<float>(body, { "mirostat_tau" }, sampling.mirostat_tau);
        readNumber<float>(body, { "mirostat_eta" }, sampling.mirostat_eta);
        readNumber<float>(body, { "adaptive_target" }, sampling.adaptive_target, -std::numeric_limits<float>::max(), 1.0f);
        readNumber<float>(body, { "adaptive_decay" }, sampling.adaptive_decay, 0.0f, 0.99f, true);
        readNumber<uint32_t>(body, { "seed" }, sampling.seed);
        readNumber<int32_t>(body, { "min_keep" }, sampling.min_keep, 0, INT32_MAX, true);
        if (body.contains("dry_sequence_breakers") && !body["dry_sequence_breakers"].is_null()) {
            sampling.dry_sequence_breakers = valueOr<std::vector<std::string>>(body, "dry_sequence_breakers", {});
            if (sampling.dry_sequence_breakers.empty())
                throw std::invalid_argument(fieldError("dry_sequence_breakers", "Error: dry_sequence_breakers must be a non-empty array of strings"));
        }
        if (body.contains("samplers") && !body["samplers"].is_null()) {
            const json& samplers = body["samplers"];
            if (samplers.is_array())
                sampling.samplers = common_sampler_types_from_names(samplers.get<std::vector<std::string>>());
            else if (samplers.is_string())
                sampling.samplers = common_sampler_types_from_chars(samplers.get<std::string>());
        }
        if (body.contains("logit_bias") && !body["logit_bias"].is_null()) {
            sampling.logit_bias.clear();
            const json& bias = body["logit_bias"];
            const int vocabSize = llama_vocab_n_tokens(m_env.vocab);
            const auto parseBias = [](const json& value, float& out) {
                if (value.is_number()) {
                    out = value.get<float>();
                    return true;
                }
                if (value.is_boolean() && !value.get<bool>()) {
                    out = -INFINITY;
                    return true;
                }
                return false;
            };
            if (bias.is_array()) {
                for (const json& element : bias) {
                    float value = 0;
                    if (!element.is_array() || element.size() != 2 || !parseBias(element[1], value))
                        continue;
                    if (element[0].is_number_integer()) {
                        const llama_token token = element[0].get<llama_token>();
                        if (token >= 0 && token < vocabSize)
                            sampling.logit_bias.push_back({ token, value });
                    } else if (element[0].is_string()) {
                        for (const llama_token token : common_tokenize(m_env.vocab, element[0].get<std::string>(), false))
                            sampling.logit_bias.push_back({ token, value });
                    }
                }
            } else if (bias.is_object()) {
                for (const auto& element : bias.items()) {
                    float value = 0;
                    if (!parseBias(element.value(), value))
                        continue;
                    char* end = nullptr;
                    const long token = std::strtol(element.key().c_str(), &end, 10);
                    if (end && *end == 0) {
                        if (token >= 0 && token < vocabSize)
                            sampling.logit_bias.push_back({ llama_token(token), value });
                    } else {
                        for (const llama_token piece : common_tokenize(m_env.vocab, element.key(), false))
                            sampling.logit_bias.push_back({ piece, value });
                    }
                }
            }
        }
        if (body.contains("ignore_eos") && !body["ignore_eos"].is_null()) {
            readBool(body, "ignore_eos", sampling.ignore_eos);
            if (sampling.ignore_eos)
                sampling.logit_bias.insert(sampling.logit_bias.end(), m_env.sampling.logit_bias_eog.begin(), m_env.sampling.logit_bias_eog.end());
        }

        // 用户给的语法；模板给的语法（工具调用 / json_schema）、惰性触发词、保留 token、生成提示由 common_chat_session 填
        // （== server 的 task.apply_chat_session）
        if (!grammar.empty())
            sampling.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, grammar);
        out.chatSession.apply_sampling(sampling);
        if (sampling.grammar_lazy && sampling.grammar_triggers.empty())
            throw std::runtime_error("Error: no triggers set for lazy grammar!");

        // 思考预算采样器（模板有思考结束标记时；惰性语法要靠它避开 <think> 里的触发词）
        if (!out.chatSession.thinking_end_tags().empty()) {
            int budget = valueOr<int>(body, "reasoning_budget_tokens", valueOr<int>(body, "thinking_budget_tokens", -1));
            if (budget == -1)
                budget = m_env.reasoningBudget;
            if (budget < -1)
                throw std::invalid_argument(fieldError("reasoning_budget_tokens", "Value must be between -1 <= value <= "
                                                                                     + std::to_string(INT32_MAX) + ", but got " + std::to_string(budget)));
            sampling.reasoning_budget_tokens = budget;
            sampling.reasoning_control = valueOr<bool>(body, "reasoning_control", false);
            sampling.reasoning_budget_start = common_tokenize(m_env.vocab, out.chatSession.thinking_start_tag(), false, true);
            for (const std::string& tag : out.chatSession.thinking_end_tags()) {
                if (!tag.empty())
                    sampling.reasoning_budget_end.push_back(common_tokenize(m_env.vocab, tag, false, true));
            }
            if (!sampling.reasoning_budget_end.empty()) {
                llama_tokens forced = sampling.reasoning_budget_end.front();
                const std::string message = valueOr<std::string>(body, "reasoning_budget_message", m_env.reasoningBudgetMessage);
                if (!message.empty()) {
                    const llama_tokens messageTokens = common_tokenize(m_env.vocab, message, false, true);
                    forced.insert(forced.begin(), messageTokens.begin(), messageTokens.end());
                }
                sampling.reasoning_budget_forced = std::move(forced);
            }
        }

        // 停止词：请求的 + 模板的；都没有就用命令行的
        p.stops.clear();
        for (const json& word : stop) {
            if (word.is_string() && !word.get<std::string>().empty())
                p.stops.push_back(word.get<std::string>());
        }
        for (const std::string& word : out.chatSession.additional_stops()) {
            if (!word.empty())
                p.stops.push_back(word);
        }
        if (p.stops.empty())
            p.stops = m_env.antiprompt;

        const double t3 = double(ggml_time_us());
        out.paramsMs = (t3 - t2) / 1000.0;
        out.stream = valueOr<bool>(body, "stream", false);
        out.timingsPerToken = false;
        readBool(body, "timings_per_token", out.timingsPerToken);
        p.thinking = inputs.enable_thinking;
        const double t4 = t3;

        // MARK: 检查点边界（§F.2）
        p.user_driven = !inputs.messages.empty() && inputs.messages.back().role == "user";
        p.prefix_boundary = prefixBoundary(inputs, p.items);
        p.transcript_boundary = p.user_driven ? transcriptBoundary(inputs, p.prompt, p.generation_prompt, p.items, !mediaInputs.empty()) : -1;
        p.prefix_fingerprint = p.prefix_boundary > 0 ? prefixFingerprint(p.items, p.prefix_boundary) : std::string();
        out.boundaryMs = (double(ggml_time_us()) - t4) / 1000.0;
        // v2.7：调度用的提示（docs/llm-protocol.md）
        out.session.clear();
        out.priority = 0;
        out.mainAgent = false;
        if (body.contains("x_friday") && body["x_friday"].is_object()) {
            const json& extra = body["x_friday"];
            out.session = valueOr<std::string>(extra, "session", std::string());
            out.priority = valueOr<int>(extra, "priority", 0);
            out.mainAgent = valueOr<bool>(extra, "main", false);
        }
        return FLR_OK;
    } catch (const std::exception& exception) {
        return fail(FLR_INVALID_REQUEST, exception.what());
    } catch (...) {
        return fail(FLR_INTERNAL, "解析请求时出了未知错误");
    }
}

int ChatFormat::prefixBoundary(const common_chat_templates_inputs& inputs, const std::vector<LedgerItem>& items) const
{
    size_t nSystem = 0;
    while (nSystem < inputs.messages.size() && inputs.messages[nSystem].role == "system")
        ++nSystem;
    if (nSystem == 0 || nSystem >= inputs.messages.size())
        return -1;

    const std::string key = probeKey(inputs, nSystem);
    std::vector<llama_token> shared;
    bool cached = false;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto it = m_probes.begin(); it != m_probes.end(); ++it) {
            if (it->key == key) {
                shared = it->tokens;
                m_probes.splice(m_probes.begin(), m_probes, it);
                cached = true;
                break;
            }
        }
    }
    if (!cached) {
        try {
            const auto probe = [&](const char* text) {
                common_chat_templates_inputs probeInputs = inputs;
                probeInputs.messages.assign(inputs.messages.begin(), inputs.messages.begin() + std::ptrdiff_t(nSystem));
                common_chat_msg user;
                user.role = "user";
                user.content = text;
                probeInputs.messages.push_back(std::move(user));
                probeInputs.add_generation_prompt = false;
                probeInputs.continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
                return tokenize(m_env.vocab, common_chat_templates_apply(m_env.templates, probeInputs).prompt);
            };
            const std::vector<llama_token> first = probe("甲");
            const std::vector<llama_token> second = probe("乙");
            size_t n = 0;
            while (n < first.size() && n < second.size() && first[n] == second[n])
                ++n;
            shared.assign(first.begin(), first.begin() + std::ptrdiff_t(n));
        } catch (const std::exception& exception) {
            logf(FLR_LOG_DEBUG, "系统前缀探针渲染失败（不设 P）：%s", exception.what());
            shared.clear();
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        m_probes.push_front({ key, shared });
        while (m_probes.size() > 4)
            m_probes.pop_back();
    }
    const size_t length = commonTextPrefix(shared, items);
    if (length == 0 || length >= items.size())
        return -1;
    return int(length);
}

int ChatFormat::transcriptBoundary(const common_chat_templates_inputs& inputs, const std::string& prompt, const std::string& generation,
                                   const std::vector<LedgerItem>& items, bool withMedia) const
{
    // 快路：生成提示以特殊 token 开头时，分词按特殊 token 切开，整段的分词 == 前面一段 ++ 生成提示自己的分词，
    // 只要核对 items 的末尾就是生成提示
    if (!generation.empty() && prompt.size() > generation.size()
        && prompt.compare(prompt.size() - generation.size(), generation.size(), generation) == 0) {
        const std::vector<llama_token> tail = common_tokenize(m_env.vocab, generation, false, true);
        if (!tail.empty() && tail.size() < items.size()
            && (llama_vocab_get_attr(m_env.vocab, tail.front()) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED))) {
            bool matches = true;
            const size_t start = items.size() - tail.size();
            for (size_t i = 0; i < tail.size() && matches; ++i)
                matches = !items[start + i].isMedia() && items[start + i].token == tail[i];
            if (matches)
                return int(start);
        }
    }
    // 有媒体块时不能重新分词比对（分出来的只有文字）：快路不行就不设 B
    if (withMedia)
        return -1;
    // 慢路：不带生成提示的那段重新分词，必须是 items 的严格前缀
    std::string head;
    if (!generation.empty() && prompt.size() >= generation.size()
        && prompt.compare(prompt.size() - generation.size(), generation.size(), generation) == 0) {
        head = prompt.substr(0, prompt.size() - generation.size());
    } else {
        try {
            common_chat_templates_inputs headInputs = inputs;
            headInputs.add_generation_prompt = false;
            headInputs.continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
            head = common_chat_templates_apply(m_env.templates, headInputs).prompt;
        } catch (const std::exception&) {
            return -1;
        }
    }
    const std::vector<llama_token> tokens = tokenize(m_env.vocab, head);
    if (tokens.empty() || tokens.size() >= items.size() || commonTextPrefix(tokens, items) != tokens.size())
        return -1;
    return int(tokens.size());
}

// MARK: - RequestStats

int64_t RequestStats::genUs() const
{
    if (tGenLast == 0)
        return 0;
    // 第一个 token 可能和提示最后一批落在同一微秒
    return std::max<int64_t>(1, tGenLast - tPromptLast);
}

double RequestStats::promptPerSecond() const
{
    const double ms = promptMs();
    return ms > 0.0 ? 1e3 / ms * double(promptProcessed) : 0.0;
}

double RequestStats::genPerSecond() const
{
    const double ms = genMs();
    return ms > 0.0 ? 1e3 / ms * double(genSteps()) : 0.0;
}

json RequestStats::toJson() const
{
    const double promptPerToken = promptProcessed > 0 ? promptMs() / double(promptProcessed) : 0.0;
    const double genPerToken = genSteps() > 0 ? genMs() / double(genSteps()) : 0.0;
    return json {
        { "cache_n", promptCached },
        { "prompt_n", promptProcessed },
        { "prompt_ms", promptMs() },
        { "prompt_per_token_ms", promptPerToken },
        { "prompt_per_second", promptPerSecond() },
        { "predicted_n", generated },
        { "predicted_ms", genMs() },
        { "predicted_per_token_ms", genPerToken },
        { "predicted_per_second", genPerSecond() },
    };
}

// MARK: - ChatStream

ChatStream::ChatStream(common_chat_session session, std::string completionId, std::string model)
    : m_session(std::move(session))
    , m_message(m_session.msg())
    , m_id(std::move(completionId))
    , m_model(std::move(model))
    , m_fingerprint(llama_build_info())
{
    // 接着 assistant 消息写（不 echo）时，session 已经从预填的那段开始解析，那段不会当增量发出去（== task_result_state 的构造）
}

std::string ChatStream::randomId()
{
    static const char characters[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    thread_local std::mt19937 generator(std::random_device {}());
    std::string id(32, ' ');
    for (char& c : id)
        c = characters[generator() % (sizeof(characters) - 1)];
    return id;
}

std::vector<common_chat_msg_diff> ChatStream::update(const common_chat_input& text, bool partial)
{
    // == task_result_state::update_chat_msg（filter_tool_calls = false，对话补全接口就是这样）
    if (m_finished)
        return {};
    const common_chat_msg previous = m_message;
    common_chat_msg parsed = partial ? m_session.feed(text) : m_session.finish(text);
    if (!partial)
        m_finished = true;
    if (parsed.empty())
        return {};
    parsed.set_tool_call_ids(m_toolCallIds, [] { return randomId(); });
    m_message = std::move(parsed);
    return common_chat_msg_diff::compute_diffs(previous, m_message);
}

json ChatStream::deltaJson(const common_chat_msg_diff& diff)
{
    // == server_chat_msg_diff_to_json_oaicompat（server-chat.cpp:621-649）
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty())
        delta["reasoning_content"] = diff.reasoning_content_delta;
    if (!diff.content_delta.empty())
        delta["content"] = diff.content_delta;
    if (diff.tool_call_index != std::string::npos) {
        json call;
        call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            call["id"] = diff.tool_call_delta.id;
            call["type"] = "function";
        }
        if (!diff.tool_call_delta.name.empty() || !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty())
                function["name"] = diff.tool_call_delta.name;
            if (!diff.tool_call_delta.arguments.empty())
                function["arguments"] = diff.tool_call_delta.arguments;
            call["function"] = function;
        }
        delta["tool_calls"] = json::array({ call });
    }
    return delta;
}

json ChatStream::chunk(json choices) const
{
    return json {
        { "choices", std::move(choices) },
        { "created", std::time(nullptr) },
        { "id", m_id },
        { "model", m_model },
        { "system_fingerprint", m_fingerprint },
        { "object", "chat.completion.chunk" },
    };
}

void ChatStream::partial(const common_chat_input& text, uint64_t nDecoded, const RequestStats* stats, std::vector<std::string>& chunks)
{
    const std::vector<common_chat_msg_diff> diffs = update(text, true);
    std::vector<json> deltas;
    const auto add = [&](json delta) {
        deltas.push_back(chunk(json::array({ json { { "finish_reason", nullptr }, { "index", 0 }, { "delta", std::move(delta) } } })));
    };
    // OpenAI 的流第一块带 role（== to_json_oaicompat_chat 的 first）
    if (nDecoded == 1)
        add(json { { "role", "assistant" }, { "content", nullptr } });
    for (const common_chat_msg_diff& diff : diffs)
        add(deltaJson(diff));
    if (!deltas.empty() && stats)
        deltas.back()["timings"] = stats->toJson();
    for (const json& delta : deltas)
        chunks.push_back(delta.dump(-1, ' ', false, json::error_handler_t::replace));
}

json ChatStream::usage(int64_t promptTokens, const RequestStats& stats)
{
    return json {
        { "completion_tokens", stats.generated },
        { "prompt_tokens", promptTokens },
        { "total_tokens", int64_t(stats.generated) + promptTokens },
        { "prompt_tokens_details", json { { "cached_tokens", stats.promptCached } } },
    };
}

std::string ChatStream::finish(StopType stop, int64_t promptTokens, const RequestStats& stats, std::vector<std::string>& chunks)
{
    const std::vector<common_chat_msg_diff> diffs = update(common_chat_input(), false);
    std::string reason = "length";
    if (stop == StopType::Word || stop == StopType::Eos)
        reason = m_message.tool_calls.empty() ? "stop" : "tool_calls";
    std::vector<json> out;
    for (const common_chat_msg_diff& diff : diffs)
        out.push_back(chunk(json::array({ json { { "finish_reason", nullptr }, { "index", 0 }, { "delta", deltaJson(diff) } } })));
    out.push_back(chunk(json::array({ json { { "finish_reason", reason }, { "index", 0 }, { "delta", json::object() } } })));
    // include_usage：最后一块 choices 为空、带 usage；timings 挂在最后一块上
    json last = chunk(json::array());
    last["usage"] = usage(promptTokens, stats);
    last["timings"] = stats.toJson();
    out.push_back(std::move(last));
    for (const json& value : out)
        chunks.push_back(value.dump(-1, ' ', false, json::error_handler_t::replace));
    return reason;
}

json ChatStream::completion(const std::string& finishReason, int64_t promptTokens, const RequestStats& stats) const
{
    // == to_json_oaicompat_chat
    json message;
    try {
        message = json::parse(m_message.to_json_oaicompat().dump());
    } catch (const std::exception&) {
        message = json { { "role", "assistant" }, { "content", m_message.content } };
    }
    return json {
        { "choices", json::array({ json { { "finish_reason", finishReason }, { "index", 0 }, { "message", std::move(message) } } }) },
        { "created", std::time(nullptr) },
        { "model", m_model },
        { "system_fingerprint", m_fingerprint },
        { "object", "chat.completion" },
        { "usage", usage(promptTokens, stats) },
        { "id", m_id },
        { "timings", stats.toJson() },
    };
}

} // namespace flr
