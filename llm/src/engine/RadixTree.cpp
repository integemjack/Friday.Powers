// 基数树（见 RadixTree.h）
#include "RadixTree.h"

#include <algorithm>
#include <cstdio>

namespace flr {
namespace {

/// 孩子按边的第一项找（整项比较：文本比 token，媒体块比 media_id + 单元数 + 位置数），同一个键的孩子只有一个
struct ItemKey {
    llama_token token = LLAMA_TOKEN_NULL;
    std::string media;
    int n_tokens = 1;
    int n_pos = 1;

    static ItemKey of(const LedgerItem& item) { return { item.token, item.media_id, item.n_tokens, item.n_pos }; }
    bool operator<(const ItemKey& other) const
    {
        if (token != other.token)
            return token < other.token;
        if (media != other.media)
            return media < other.media;
        if (n_tokens != other.n_tokens)
            return n_tokens < other.n_tokens;
        return n_pos < other.n_pos;
    }
};

} // namespace

struct RadixTree::Node {
    /// 这条边上的账本项
    std::vector<LedgerItem> edge;
    /// 边之前有多少项（根到这条边开头的长度）
    size_t start = 0;
    Node* parent = nullptr;
    std::map<ItemKey, std::unique_ptr<Node>> children;
    /// 经过这条边的序列（账本至少走到这条边末尾）
    std::set<int> holders;

    size_t end() const { return start + edge.size(); }
};

RadixTree::RadixTree()
    : m_root(std::make_unique<Node>())
{
}

RadixTree::~RadixTree() = default;

bool RadixTree::contains(int seq) const
{
    return m_entries.count(seq) > 0;
}

void RadixTree::sync(int seq, const std::vector<LedgerItem>& ledger, uint64_t generation)
{
    if (ledger.empty()) {
        remove(seq);
        return;
    }
    const auto it = m_entries.find(seq);
    if (it != m_entries.end() && it->second.generation == generation && ledger.size() >= it->second.length) {
        Entry& entry = it->second;
        if (ledger.size() == entry.length)
            return;
        Node* leaf = entry.leaf;
        if (leaf != m_root.get() && leaf->holders.size() == 1 && leaf->children.empty()) {
            // 只有它自己的叶子：直接接在边上（逐 token 生成时走这条）
            leaf->edge.insert(leaf->edge.end(), ledger.begin() + std::ptrdiff_t(entry.length), ledger.end());
        } else {
            leaf = insert(leaf, ledger, entry.length, seq);
        }
        entry.leaf = leaf;
        entry.length = ledger.size();
        return;
    }
    remove(seq);
    m_root->holders.insert(seq);
    Node* leaf = insert(m_root.get(), ledger, 0, seq);
    m_entries[seq] = Entry { leaf, ledger.size(), generation };
}

RadixTree::Node* RadixTree::insert(Node* node, const std::vector<LedgerItem>& items, size_t from, int seq)
{
    while (from < items.size()) {
        const auto found = node->children.find(ItemKey::of(items[from]));
        if (found == node->children.end()) {
            auto child = std::make_unique<Node>();
            child->edge.assign(items.begin() + std::ptrdiff_t(from), items.end());
            child->start = node->end();
            child->parent = node;
            child->holders.insert(seq);
            Node* raw = child.get();
            node->children.emplace(ItemKey::of(items[from]), std::move(child));
            return raw;
        }
        Node* child = found->second.get();
        size_t k = 0;
        while (k < child->edge.size() && from + k < items.size() && child->edge[k] == items[from + k])
            ++k;
        // 键就是整个第一项，k ≥ 1
        if (k < child->edge.size())
            split(child, k);
        child->holders.insert(seq);
        node = child;
        from += k;
    }
    return node;
}

RadixTree::Node* RadixTree::split(Node* node, size_t at)
{
    auto tail = std::make_unique<Node>();
    tail->edge.assign(node->edge.begin() + std::ptrdiff_t(at), node->edge.end());
    tail->start = node->start + at;
    tail->parent = node;
    tail->holders = node->holders;
    tail->children = std::move(node->children);
    for (auto& [key, child] : tail->children)
        child->parent = tail.get();
    node->edge.resize(at);
    node->children.clear();
    Node* raw = tail.get();
    // 在这条边末尾结束的序列现在在后半段的末尾结束
    for (auto& [seq, entry] : m_entries) {
        if (entry.leaf == node)
            entry.leaf = raw;
    }
    node->children.emplace(ItemKey::of(raw->edge.front()), std::move(tail));
    return raw;
}

void RadixTree::remove(int seq)
{
    const auto it = m_entries.find(seq);
    if (it == m_entries.end())
        return;
    Node* leaf = it->second.leaf;
    m_entries.erase(it);
    detach(leaf, seq);
}

void RadixTree::detach(Node* leaf, int seq)
{
    std::vector<Node*> path;
    for (Node* node = leaf; node; node = node->parent)
        path.push_back(node);
    // 从叶子往上：去掉这个序列，没人经过的节点删掉（它的孩子一定也没人经过了）
    Node* survivor = nullptr;
    for (Node* node : path) {
        node->holders.erase(seq);
        if (node != m_root.get() && node->holders.empty()) {
            Node* parent = node->parent;
            parent->children.erase(ItemKey::of(node->edge.front()));
            continue;
        }
        if (!survivor)
            survivor = node;
    }
    // 留下来的最深的节点往上：只剩一个孩子、经过的序列一样的，合成一条边
    for (Node* node = survivor; node && node != m_root.get();) {
        Node* parent = node->parent;
        compact(node);
        node = parent;
    }
}

void RadixTree::compact(Node* node)
{
    while (node != m_root.get() && node->children.size() == 1) {
        Node* child = node->children.begin()->second.get();
        if (child->holders != node->holders)
            break;
        node->edge.insert(node->edge.end(), child->edge.begin(), child->edge.end());
        for (auto& [seq, entry] : m_entries) {
            if (entry.leaf == child)
                entry.leaf = node;
        }
        std::map<ItemKey, std::unique_ptr<Node>> grandchildren = std::move(child->children);
        for (auto& [key, grandchild] : grandchildren)
            grandchild->parent = node;
        node->children = std::move(grandchildren);
    }
}

RadixTree::Match RadixTree::match(const std::vector<LedgerItem>& items, const std::set<int>& exclude) const
{
    const auto usable = [&exclude](const Node* node) {
        for (const int seq : node->holders) {
            if (!exclude.count(seq))
                return true;
        }
        return false;
    };
    Match result;
    const Node* node = m_root.get();
    const Node* last = nullptr;
    size_t pos = 0;
    while (pos < items.size()) {
        const auto found = node->children.find(ItemKey::of(items[pos]));
        if (found == node->children.end() || !usable(found->second.get()))
            break;
        const Node* child = found->second.get();
        size_t k = 0;
        while (k < child->edge.size() && pos + k < items.size() && child->edge[k] == items[pos + k])
            ++k;
        if (k == 0)
            break;
        pos += k;
        last = child;
        if (k < child->edge.size())
            break;
        node = child;
    }
    if (!last)
        return result;
    result.length = pos;
    for (const int seq : last->holders) {
        if (!exclude.count(seq))
            result.holders.push_back(seq);
    }
    return result;
}

size_t RadixTree::shared(int a, int b) const
{
    const auto ia = m_entries.find(a);
    const auto ib = m_entries.find(b);
    if (ia == m_entries.end() || ib == m_entries.end())
        return 0;
    if (a == b)
        return ia->second.length;
    std::set<const Node*> ancestors;
    for (const Node* node = ia->second.leaf; node; node = node->parent)
        ancestors.insert(node);
    for (const Node* node = ib->second.leaf; node; node = node->parent) {
        if (ancestors.count(node))
            return node->end();
    }
    return 0;
}

size_t RadixTree::exclusive(int seq) const
{
    const auto it = m_entries.find(seq);
    if (it == m_entries.end())
        return 0;
    size_t count = 0;
    for (const Node* node = it->second.leaf; node && node != m_root.get() && node->holders.size() == 1; node = node->parent)
        count += node->edge.size();
    return count;
}

size_t RadixTree::sharedWithOthers(int seq) const
{
    const auto it = m_entries.find(seq);
    if (it == m_entries.end())
        return 0;
    return it->second.length - std::min(it->second.length, exclusive(seq));
}

RadixTree::Stats RadixTree::stats() const
{
    Stats stats;
    std::vector<const Node*> stack { m_root.get() };
    while (!stack.empty()) {
        const Node* node = stack.back();
        stack.pop_back();
        if (node != m_root.get()) {
            ++stats.nodes;
            stats.items += node->edge.size();
        }
        for (const auto& [key, child] : node->children)
            stack.push_back(child.get());
    }
    stats.sequences = m_entries.size();
    for (const auto& [seq, entry] : m_entries)
        stats.logical += entry.length;
    return stats;
}

std::string RadixTree::dump() const
{
    std::string out;
    std::vector<std::pair<const Node*, int>> stack { { m_root.get(), 0 } };
    while (!stack.empty()) {
        const auto [node, depth] = stack.back();
        stack.pop_back();
        if (node != m_root.get()) {
            char line[160];
            std::snprintf(line, sizeof(line), "%*s[%zu, %zu) 序列", depth * 2, "", node->start, node->end());
            out += line;
            for (const int seq : node->holders)
                out += " " + std::to_string(seq);
            out += "\n";
        }
        for (auto it = node->children.rbegin(); it != node->children.rend(); ++it)
            stack.push_back({ it->second.get(), node == m_root.get() ? 0 : depth + 1 });
    }
    return out;
}

} // namespace flr
