// 采样与逐 token 的文字处理（见 Sampler.h）
#include "Sampler.h"

#include <algorithm>
#include <stdexcept>

namespace flr {

Sampler::Sampler(const llama_model* model, common_params_sampling& params)
    : m_sampler(common_sampler_init(model, params))
{
    if (!m_sampler)
        throw std::runtime_error("common_sampler_init 失败");
}

void Sampler::acceptPrompt(const std::vector<LedgerItem>& items)
{
    common_sampler_reset(m_sampler.get());
    for (const LedgerItem& item : items) {
        if (!item.isMedia() && item.token != LLAMA_TOKEN_NULL)
            common_sampler_accept(m_sampler.get(), item.token, false);
    }
}

llama_token Sampler::sample(llama_context* ctx, int idx)
{
    const llama_token token = common_sampler_sample(m_sampler.get(), ctx, idx);
    common_sampler_accept(m_sampler.get(), token, true);
    return token;
}

size_t validUtf8Length(const std::string& text)
{
    const size_t length = text.size();
    if (length == 0)
        return 0;
    // 从末尾往前看最多 4 个字节，找被截断的多字节字符的开头
    for (size_t i = 1; i <= 4 && i <= length; ++i) {
        const unsigned char c = static_cast<unsigned char>(text[length - i]);
        if ((c & 0xE0) == 0xC0) {
            if (i < 2)
                return length - i;
        } else if ((c & 0xF0) == 0xE0) {
            if (i < 3)
                return length - i;
        } else if ((c & 0xF8) == 0xF0) {
            if (i < 4)
                return length - i;
        }
    }
    return length;
}

TokenText::Step TokenText::push(const std::string& piece, llama_token token, bool eog, const std::vector<std::string>& stops)
{
    Step step;
    m_generated.append(piece, token);
    const std::string& text = m_generated.text;
    if (validUtf8Length(text) < text.size())
        return step;
    step.send = true;

    size_t pos = std::min(m_sent, text.size());
    const std::string test = text.substr(pos);
    bool sendText = true;

    // 完整的停止词：只在最后一个 token 能影响到的范围里找（== find_stopping_strings(…, is_full_stop = true)）
    size_t stopPos = std::string::npos;
    for (const std::string& word : stops) {
        if (word.empty())
            continue;
        const size_t window = word.size() + piece.size();
        const size_t from = test.size() > window ? test.size() - window : 0;
        const size_t found = test.find(word, from);
        if (found != std::string::npos && (stopPos == std::string::npos || found < stopPos)) {
            stopPos = found;
            m_stoppingWord = word;
        }
    }
    if (stopPos != std::string::npos) {
        // 停止词本身不发
        m_generated.truncate(pos + stopPos);
        pos = std::min(m_sent, m_generated.text.size());
        step.stopWord = true;
    } else if (!eog) {
        // 末尾可能是停止词的开头：先压住，等下一个 token 再说
        for (const std::string& word : stops) {
            if (!word.empty() && string_find_partial_stop(test, word) != std::string::npos) {
                sendText = false;
                break;
            }
        }
    }
    if (sendText) {
        step.text = m_generated.substr(pos);
        m_sent += step.text.size();
    }
    return step;
}

} // namespace flr
