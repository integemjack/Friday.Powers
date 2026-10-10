// 假的 KV（MemoryOps）：SequenceCache / 快照 / 分叉的单测用，不需要模型。
// 每个序列记「KV 里有哪些位置」和一份「循环状态」（= 它最后算到的位置，-1 = 没有）：
//   - hybrid = false（PART 型，普通 Transformer）：seqRemove 能删任意区间；
//   - hybrid = true（Qwen3.5 这类混合模型）：循环状态不能回退——seqRemove 只能整段删（p0 ≤ 0 且 p1 < 0），或者删循环状态之后的
//     部分（p0 > 循环状态的位置，== llama-memory-recurrent.cpp:161-240）；先 restorePartial 把循环状态退回到检查点，再删尾巴。
// decode(seq, n) 模拟一次成功的 decode：在序列末尾加 n 个位置，循环状态前进到最后一个位置。
// partialState / fullState 是可以恢复的字节串；calls 记下每次操作（断言调用顺序用）。
#pragma once

#include "internal.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

class FakeMemory : public flr::MemoryOps {
public:
    explicit FakeMemory(bool hybrid)
        : m_hybrid(hybrid)
    {
    }

    /// 模拟一次成功的 decode：位置 posMax+1 … posMax+n 进 KV，循环状态前进到最后一个位置
    void decode(int seq, int n)
    {
        llama_pos next = seqPosMax(seq) + 1;
        for (int i = 0; i < n; ++i)
            m_cells[seq].insert(next++);
        m_recurrent[seq] = next - 1;
    }
    /// 序列的循环状态停在哪个位置（-1 = 没有）
    llama_pos recurrent(int seq) const
    {
        const auto it = m_recurrent.find(seq);
        return it == m_recurrent.end() ? -1 : it->second;
    }
    /// 序列在 KV 里的位置数
    size_t cellCount(int seq) const
    {
        const auto it = m_cells.find(seq);
        return it == m_cells.end() ? 0 : it->second.size();
    }

    std::vector<std::string> calls;
    /// 接下来这么多次 restoreFull 直接失败（模拟 KV 池被别的序列占着、装不进去；评审修正的用例）
    int rejectRestores = 0;

    /// 模拟一次失败的 decode（中止 / 出错）：llama 先把 n 个位置的 ubatch 记进序列（注意力 cell 占上、循环状态的尾 cell 改到
    /// ubatch 末尾），算到一半失败后再 seq_rm(seq, 起点, -1)——混合模型上这一步删不掉（循环状态已经在起点之后），留下没算过的位置
    void failedDecode(int seq, int n)
    {
        const llama_pos start = seqPosMax(seq) + 1;
        decode(seq, n);
        seqRemove(seq, start, -1);
    }

    // MARK: MemoryOps

    using flr::MemoryOps::restorePartial;

    bool partialRemovable() const override { return !m_hybrid; }

    bool seqRemove(int seq, llama_pos p0, llama_pos p1) override
    {
        calls.push_back("seqRemove " + std::to_string(seq) + " " + std::to_string(p0) + " " + std::to_string(p1));
        if (p0 < 0)
            p0 = 0;
        const bool everything = p0 == 0 && p1 < 0;
        if (m_hybrid && !everything && p0 <= recurrent(seq))
            return false;
        auto& cells = m_cells[seq];
        for (auto it = cells.begin(); it != cells.end();) {
            if (*it >= p0 && (p1 < 0 || *it < p1))
                it = cells.erase(it);
            else
                ++it;
        }
        if (everything)
            m_recurrent[seq] = -1;
        return true;
    }

    void seqCopy(int src, int dst) override
    {
        calls.push_back("seqCopy " + std::to_string(src) + " " + std::to_string(dst));
        m_cells[dst] = m_cells[src];
        m_recurrent[dst] = recurrent(src);
    }

    void seqShare(int src, int dst, llama_pos p1) override
    {
        calls.push_back("seqShare " + std::to_string(src) + " " + std::to_string(dst) + " " + std::to_string(p1));
        std::set<llama_pos> cells;
        for (llama_pos pos : m_cells[src]) {
            if (p1 < 0 || pos < p1)
                cells.insert(pos);
        }
        m_cells[dst] = cells;
        m_recurrent[dst] = recurrent(src);
    }

    void seqClear(int seq) override
    {
        calls.push_back("seqClear " + std::to_string(seq));
        m_cells[seq].clear();
        m_recurrent[seq] = -1;
    }

    llama_pos seqPosMax(int seq) const override
    {
        const auto it = m_cells.find(seq);
        return it == m_cells.end() || it->second.empty() ? -1 : *it->second.rbegin();
    }

    std::vector<uint8_t> partialState(int seq) override
    {
        calls.push_back("partialState " + std::to_string(seq));
        return encode("P", { recurrent(seq) });
    }

    bool restorePartial(int seq, const std::vector<uint8_t>& data) override
    {
        calls.push_back("restorePartial " + std::to_string(seq));
        std::vector<llama_pos> values;
        if (!decode("P", data, values) || values.size() != 1)
            return false;
        m_recurrent[seq] = values[0];
        return true;
    }

    std::vector<uint8_t> fullState(int seq) override
    {
        calls.push_back("fullState " + std::to_string(seq));
        std::vector<llama_pos> values { recurrent(seq) };
        for (llama_pos pos : m_cells[seq])
            values.push_back(pos);
        return encode("F", values);
    }

    bool restoreFull(int seq, const std::vector<uint8_t>& data) override
    {
        calls.push_back("restoreFull " + std::to_string(seq));
        if (rejectRestores > 0) {
            --rejectRestores;
            return false;
        }
        std::vector<llama_pos> values;
        if (!decode("F", data, values) || values.empty())
            return false;
        m_recurrent[seq] = values[0];
        m_cells[seq] = std::set<llama_pos>(values.begin() + 1, values.end());
        return true;
    }

private:
    static std::vector<uint8_t> encode(const std::string& tag, const std::vector<llama_pos>& values)
    {
        std::ostringstream out;
        out << tag;
        for (llama_pos value : values)
            out << ' ' << value;
        const std::string text = out.str();
        return std::vector<uint8_t>(text.begin(), text.end());
    }

    static bool decode(const std::string& tag, const std::vector<uint8_t>& data, std::vector<llama_pos>& values)
    {
        std::istringstream in(std::string(data.begin(), data.end()));
        std::string head;
        if (!(in >> head) || head != tag)
            return false;
        llama_pos value = 0;
        while (in >> value)
            values.push_back(value);
        return true;
    }

    bool m_hybrid;
    std::map<int, std::set<llama_pos>> m_cells;
    std::map<int, llama_pos> m_recurrent;
};
