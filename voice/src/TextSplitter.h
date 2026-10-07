// 边写边读的断句：大模型一个字一个字地吐，攒够一句就交给合成。
//   - 句末（。！？；… 换行，英文句点后跟空白）切开；
//   - 一轮回复的头一段在第一个逗号处就先切（让第一声早点出来），太长的句子在最后一个逗号处切；
//   - 去掉 Markdown 记号、代码块、链接地址、表情，只留念得出来的字。
#pragma once

#include <string>
#include <vector>

namespace voice {

/// 一段 Markdown / 纯文字 → 念得出来的文字（没有可念的就返回空）
std::string cleanForSpeech(const std::string& text);

class TextSplitter {
public:
    /// 追加新吐出来的字；返回已经可以念的段（已清理）
    std::vector<std::string> push(const std::string& delta);
    /// 这一轮说完了：剩下的都切出来，状态复位
    std::vector<std::string> flush();
    void reset();
    /// 还有没念的字
    bool pending() const { return !m_raw.empty() || !m_text.empty(); }

private:
    void absorb(bool final);
    void cut(std::vector<std::string>& out, bool final);
    void emit(std::vector<std::string>& out, size_t length);

    std::string m_raw;      // 还没过代码块筛的原文
    std::string m_text;     // 过了筛、等着切的
    bool m_inCode = false;  // 在 ``` 代码块里
    int m_emitted = 0;      // 这一轮已经切出去几段
};

} // namespace voice
