#include "TextSplitter.h"

#include <algorithm>
#include <cctype>
#include <cstdint>

namespace voice {

namespace {

constexpr int kFirstUnits = 5;    // 一轮的头一段：到第一个逗号、且至少这么多字（词）就先念
constexpr int kSoftMax = 40;      // 一段超过这么多字（词），在最后一个逗号处切
constexpr int kHardMax = 80;      // 再长还没逗号就硬切

/// 解 UTF-8 的一个字符；坏字节当一个字节的字符
uint32_t decode(const std::string& s, size_t i, size_t* length)
{
    const unsigned char c = static_cast<unsigned char>(s[i]);
    size_t n = 1;
    uint32_t cp = c;
    if (c >= 0xF0 && c < 0xF8) {
        n = 4, cp = c & 0x07;
    } else if (c >= 0xE0) {
        n = 3, cp = c & 0x0F;
    } else if (c >= 0xC0) {
        n = 2, cp = c & 0x1F;
    }
    if (i + n > s.size()) {
        *length = 1;
        return c;
    }
    for (size_t k = 1; k < n; ++k)
        cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    *length = n;
    return cp;
}

bool isCjk(uint32_t cp)
{
    return (cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x3400 && cp <= 0x9FFF) || (cp >= 0xAC00 && cp <= 0xD7AF)
        || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x20000 && cp <= 0x2FFFF);
}

bool isWordChar(uint32_t cp)
{
    return (cp < 0x80 && (std::isalnum(int(cp)) != 0)) || (cp >= 0xC0 && cp < 0x2000);
}

bool isSpace(uint32_t cp)
{
    return cp == ' ' || cp == '\t' || cp == '\r' || cp == '\n' || cp == 0x3000 || cp == 0xA0;
}

bool isEmoji(uint32_t cp)
{
    return (cp >= 0x1F000 && cp <= 0x1FAFF) || (cp >= 0x2600 && cp <= 0x27BF) || cp == 0xFE0F || cp == 0x200D
        || (cp >= 0x2B00 && cp <= 0x2BFF) || (cp >= 0xE0000 && cp <= 0xE007F);
}

/// 句末之后紧跟着的收尾符（引号、括号）归到这一句
bool isCloser(uint32_t cp)
{
    switch (cp) {
    case 0x201D: case 0x2019: case 0x300D: case 0x300F: case 0xFF09: case 0x3011: case 0x300B:
    case '"': case '\'': case ')': case ']': case '*': case '_':
        return true;
    default:
        return false;
    }
}

bool isWideHard(uint32_t cp)
{
    return cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F || cp == 0xFF1B || cp == 0x2026 || cp == '\n';
}

bool isWideSoft(uint32_t cp)
{
    return cp == 0xFF0C || cp == 0x3001 || cp == 0xFF1A || cp == 0x2014;
}

/// 英文标点（. ! ? ; , :）后面跟空白、中日韩字、收尾符才算断开（3.14、10:30、1,000 不切）；后面还没字时 final 才算
bool asciiBreak(const std::string& s, size_t next, bool final)
{
    if (next >= s.size())
        return final;
    size_t n = 0;
    const uint32_t cp = decode(s, next, &n);
    return isSpace(cp) || isCjk(cp) || isCloser(cp);
}

/// 去掉 [文字](地址) 的地址、![图](地址) 整个、裸网址、HTML 标签
std::string stripLinks(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const char c = s[i];
        if (c == '!' && i + 1 < s.size() && s[i + 1] == '[') {
            ++i;   // 图片：留下替代文字
            continue;
        }
        if (c == '[') {
            const size_t close = s.find(']', i + 1);
            if (close != std::string::npos && close + 1 < s.size() && s[close + 1] == '(') {
                const size_t end = s.find(')', close + 2);
                if (end != std::string::npos) {
                    out.append(s, i + 1, close - i - 1);
                    i = end + 1;
                    continue;
                }
            }
        }
        if (s.compare(i, 7, "http://") == 0 || s.compare(i, 8, "https://") == 0 || s.compare(i, 4, "www.") == 0) {
            while (i < s.size()) {
                size_t n = 0;
                const uint32_t cp = decode(s, i, &n);
                if (isSpace(cp) || isCjk(cp) || cp == ')' || cp >= 0x3000)
                    break;
                i += n;
            }
            continue;
        }
        if (c == '<' && i + 1 < s.size() && (std::isalpha(static_cast<unsigned char>(s[i + 1])) || s[i + 1] == '/')) {
            const size_t end = s.find('>', i + 1);
            if (end != std::string::npos && end - i < 120) {
                i = end + 1;
                continue;
            }
        }
        out += c;
        ++i;
    }
    return out;
}

/// 一行开头的 Markdown 记号：# 标题、> 引用、- * + 1. 列表；整行是分隔线 / 表格分隔就不要
std::string stripLineMarkers(const std::string& line)
{
    size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t'))
        ++i;
    bool rule = true;
    for (size_t k = i; k < line.size(); ++k) {
        const char c = line[k];
        if (c != '-' && c != '*' && c != '_' && c != '=' && c != '|' && c != ':' && c != ' ' && c != '\t' && c != '\r') {
            rule = false;
            break;
        }
    }
    if (rule)
        return {};
    while (i < line.size() && (line[i] == '#' || line[i] == '>'))
        ++i;
    if (i + 1 < line.size() && (line[i] == '-' || line[i] == '*' || line[i] == '+') && line[i + 1] == ' ') {
        i += 2;
    } else {
        size_t k = i;
        while (k < line.size() && std::isdigit(static_cast<unsigned char>(line[k])))
            ++k;
        if (k > i && k + 1 < line.size() && (line[k] == '.' || line[k] == ')') && line[k + 1] == ' ')
            i = k + 2;
    }
    return line.substr(i);
}

} // namespace

std::string cleanForSpeech(const std::string& text)
{
    const std::string linked = stripLinks(text);
    std::string joined;
    size_t start = 0;
    while (start <= linked.size()) {
        size_t end = linked.find('\n', start);
        if (end == std::string::npos)
            end = linked.size();
        const std::string line = stripLineMarkers(linked.substr(start, end - start));
        if (!line.empty()) {
            if (!joined.empty())
                joined += ' ';
            joined += line;
        }
        start = end + 1;
    }

    std::string out;
    out.reserve(joined.size());
    bool speakable = false, space = false;
    for (size_t i = 0; i < joined.size();) {
        size_t n = 0;
        const uint32_t cp = decode(joined, i, &n);
        const size_t at = i;
        i += n;
        if (isEmoji(cp) || cp == '*' || cp == '`' || cp == '~' || (cp == '#' && (out.empty() || space)))
            continue;
        if (cp == '_' && i < joined.size() && joined[i] == '_') {
            ++i;
            continue;
        }
        if (isSpace(cp) || cp == '|') {
            space = !out.empty();
            continue;
        }
        if (space) {
            out += ' ';
            space = false;
        }
        out.append(joined, at, n);
        if (isCjk(cp) || isWordChar(cp))
            speakable = true;
    }
    return speakable ? out : std::string();
}

void TextSplitter::reset()
{
    m_raw.clear();
    m_text.clear();
    m_inCode = false;
    m_emitted = 0;
}

std::vector<std::string> TextSplitter::push(const std::string& delta)
{
    m_raw += delta;
    absorb(false);
    std::vector<std::string> out;
    cut(out, false);
    return out;
}

std::vector<std::string> TextSplitter::flush()
{
    absorb(true);
    std::vector<std::string> out;
    cut(out, true);
    reset();
    return out;
}

void TextSplitter::absorb(bool final)
{
    // ``` 代码块整个不念；分隔符可能被拆在两次 push 里，结尾的反引号先留着
    while (true) {
        const size_t fence = m_raw.find("```");
        if (m_inCode) {
            if (fence == std::string::npos) {
                const size_t keep = final ? 0 : std::min<size_t>(2, m_raw.size());
                m_raw.erase(0, m_raw.size() - keep);
                return;
            }
            m_raw.erase(0, fence + 3);
            m_inCode = false;
            continue;
        }
        if (fence == std::string::npos) {
            size_t keep = 0;
            while (!final && keep < 2 && keep < m_raw.size() && m_raw[m_raw.size() - 1 - keep] == '`')
                ++keep;
            m_text.append(m_raw, 0, m_raw.size() - keep);
            m_raw.erase(0, m_raw.size() - keep);
            return;
        }
        m_text.append(m_raw, 0, fence);
        m_text += '\n';
        m_raw.erase(0, fence + 3);
        m_inCode = true;
    }
}

void TextSplitter::cut(std::vector<std::string>& out, bool final)
{
    while (!m_text.empty()) {
        size_t cutAt = std::string::npos, lastSoft = std::string::npos, lastSpace = std::string::npos;
        int units = 0;
        bool inWord = false;
        for (size_t i = 0; i < m_text.size();) {
            size_t n = 0;
            const uint32_t cp = decode(m_text, i, &n);
            const size_t next = i + n;
            const bool ascii = cp == '.' || cp == '!' || cp == '?' || cp == ';';
            const bool hard = isWideHard(cp) || (ascii && asciiBreak(m_text, next, final));
            const bool soft = isWideSoft(cp) || ((cp == ',' || cp == ':') && asciiBreak(m_text, next, final));
            if (hard || soft) {
                // 连着的句末符、收尾的引号括号都归这一段
                size_t end = next;
                while (end < m_text.size()) {
                    size_t m = 0;
                    const uint32_t c2 = decode(m_text, end, &m);
                    if (!isCloser(c2) && !isWideHard(c2) && c2 != '.' && c2 != '!' && c2 != '?')
                        break;
                    end += m;
                }
                if (hard && units >= 2) {
                    cutAt = end;
                    break;
                }
                if (soft) {
                    lastSoft = end;
                    if (m_emitted == 0 && units >= kFirstUnits) {
                        cutAt = end;
                        break;
                    }
                }
            }
            if (isSpace(cp))
                lastSpace = next;
            if (isCjk(cp)) {
                ++units;
                inWord = false;
            } else if (isWordChar(cp)) {
                if (!inWord)
                    ++units;
                inWord = true;
            } else {
                inWord = false;
            }
            if (units >= kSoftMax && lastSoft != std::string::npos) {
                cutAt = lastSoft;
                break;
            }
            if (units >= kHardMax) {
                cutAt = lastSpace != std::string::npos ? lastSpace : next;
                break;
            }
            i = next;
        }
        if (cutAt == std::string::npos) {
            if (final)
                emit(out, m_text.size());
            return;
        }
        emit(out, cutAt);
    }
}

void TextSplitter::emit(std::vector<std::string>& out, size_t length)
{
    const std::string segment = cleanForSpeech(m_text.substr(0, length));
    m_text.erase(0, length);
    if (!segment.empty()) {
        out.push_back(segment);
        ++m_emitted;
    }
}

} // namespace voice
