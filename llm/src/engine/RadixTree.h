// 基数树（radix tree，压缩前缀树）：所有序列账本的索引（docs/llm-design.md §3）。
//
// 每个序列（llama 的 seq_id）的账本是从根到某个节点的一条路径；几条账本的公共前缀是同一段路径——KV 池里那段格子
// 同时挂着这几个序列号（统一 KV 里 llama_memory_seq_cp 只改元数据，格子只存一份），树只是它们的索引：
//   - match：新请求的提示在树上走，找到最长的公共前缀和经过那里的序列（分叉的来源）；
//   - exclusive：一个序列独占的尾巴有多长（腾掉它能空出多少格子）；
//   - 节点的边是一段账本项（不是单个 token），一条路径上只有分叉处才断开，所以几万 token 的对话只有几个节点。
// 树不碰 KV：Scheduler 改了账本之后调 sync，树跟着改。纯数据结构，只在推理线程上用（单测直接测它）。
#pragma once

#include "internal.h"

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace flr {

class RadixTree {
public:
    RadixTree();
    ~RadixTree();
    RadixTree(const RadixTree&) = delete;
    RadixTree& operator=(const RadixTree&) = delete;

    /// 把序列 seq 现在的账本同步进树。generation 是账本的「代」（SequenceCache::generation：截断、清空、换内容时变，只追加时不变）：
    /// 和上次一样且账本只长不短 → 只把新加的那段接上；否则整条拿掉重插
    void sync(int seq, const std::vector<LedgerItem>& ledger, uint64_t generation);
    /// 序列清空了 / 不再用
    void remove(int seq);
    bool contains(int seq) const;

    struct Match {
        /// 与提示的最长公共前缀（项数）
        size_t length = 0;
        /// 账本经过这段前缀的序列（它们的账本前 length 项都与提示相同）
        std::vector<int> holders;
    };
    /// 提示在树上能走多远；exclude 里的序列不算（Scheduler 用它排除自己要用的那个序列）
    Match match(const std::vector<LedgerItem>& items, const std::set<int>& exclude = {}) const;

    /// 两个序列账本的公共前缀长度（都在树里时）
    size_t shared(int a, int b) const;
    /// seq 账本末尾只有它自己经过的那段有多少项（腾掉它能空出的）；不在树里为 0
    size_t exclusive(int seq) const;
    /// seq 的账本有多少项与别的序列共用（= 账本长度 − exclusive）
    size_t sharedWithOthers(int seq) const;

    struct Stats {
        size_t nodes = 0;
        size_t sequences = 0;
        /// 树里的项数（共用的只算一次，≈ KV 里真正占的格子数）
        size_t items = 0;
        /// 所有序列账本的总项数（不共用时要占的）
        size_t logical = 0;
    };
    Stats stats() const;

    /// 调试：树的样子（每个节点一行：深度、边长、经过的序列）
    std::string dump() const;

private:
    struct Node;
    struct Entry {
        Node* leaf = nullptr;
        size_t length = 0;
        uint64_t generation = 0;
    };

    /// 从 node 往下接 items[from, end)，返回最后一个节点（路径正好在它的边末尾结束）
    Node* insert(Node* node, const std::vector<LedgerItem>& items, size_t from, int seq);
    /// 把 node 的边在 at 项处断开：前半段留在 node，后半段连同孩子挪到新孩子里。返回新孩子
    Node* split(Node* node, size_t at);
    /// seq 从 leaf 到根的路径上去掉；没有人经过的节点删掉，能合并的合并
    void detach(Node* leaf, int seq);
    void compact(Node* node);

    std::unique_ptr<Node> m_root;
    std::map<int, Entry> m_entries;
};

} // namespace flr
