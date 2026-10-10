// 采样与逐 token 的文字处理。照 llama-server（b11541）：
//   Sampler   == server_slot 的 smpl（common_sampler）+ init_sampler（提示 token 进采样器的历史，惩罚项用，不推进语法）；
//   TokenText == server_context_impl::process_token 的文字部分（停止词、末尾不完整的 UTF-8 先不发、部分停止词先压住），
//                server-context.cpp 的 process_token。生成的文字带着 token（common_chat_input），交给 common_chat_session 解析。
// 文件名不叫 Sampling.*：Windows 不分大小写，会挡住 common 的 sampling.h（internal.h 开头的说明）。
#pragma once

#include "internal.h"

#include <string>
#include <vector>

namespace flr {

/// common_sampler 的 RAII 包装（一个请求一个）
class Sampler {
public:
    /// common_sampler_init（语法、惰性触发词、思考预算都在 params 里）。参数不对时抛 std::exception
    /// （调用方转 FLR_INVALID_REQUEST「Failed to initialize samplers: …」，== server launch_slot_with_task）
    Sampler(const llama_model* model, common_params_sampling& params);
    Sampler(Sampler&&) noexcept = default;
    Sampler& operator=(Sampler&&) noexcept = default;

    /// 提示里的文本 token 进采样器的历史（== server_slot::init_sampler：common_sampler_reset + accept(tok, false)）
    void acceptPrompt(const std::vector<LedgerItem>& items);
    /// 推理线程：用 batch 里下标 idx 那个输出的 logits 采样，并 accept（推进语法与思考预算）。语法出错时抛 std::exception
    llama_token sample(llama_context* ctx, int idx);

private:
    common_sampler_ptr m_sampler;
};

/// == server-common.cpp validate_utf8：去掉末尾不完整的多字节字符之后的长度
size_t validUtf8Length(const std::string& text);

/// 停止的原因（== server 的 stop_type）
enum class StopType { None, Eos, Word, Limit };

/// 一个请求生成的文字与停止词处理（== process_token 的文字部分）
class TokenText {
public:
    /// 一个 token 之后要发出去的文字（可能为空：部分停止词先压住、不完整的 UTF-8 先不发）
    struct Step {
        /// false：末尾是不完整的 UTF-8，这一步不发（server 不调 send_partial_response）
        bool send = false;
        /// 这一步要发的文字（带 token）
        common_chat_input text;
        /// 遇到停止词（文字已截掉停止词）
        bool stopWord = false;
    };
    /// piece 是这个 token 的文字（common_token_to_piece）；eog：这个 token 是结束符（结束符不压部分停止词）
    Step push(const std::string& piece, llama_token token, bool eog, const std::vector<std::string>& stops);

    const common_chat_input& generated() const { return m_generated; }
    const std::string& stoppingWord() const { return m_stoppingWord; }

private:
    common_chat_input m_generated;
    size_t m_sent = 0;
    std::string m_stoppingWord;
};

} // namespace flr
