# friday-llm 的设计：多并发与基数树前缀共享

## 1. 为什么自己调度

Friday 的请求天然是一棵树：主智能体和同时派出的几个子智能体共用系统提示 + 工具说明（几千 token）；完成核对、预测下一步、
起标题这些旁路请求都是从当前对话分出去的叉；多轮对话每轮都是上一轮的延长。

llama-server（b11541）有多槽位、连续批处理、统一 KV、每个槽位自己的前缀复用、内存里的提示缓存和混合模型的检查点，
但同一段前缀在每个槽位里各算一遍、各存一份，槽位之间不共享。SGLang 的 RadixAttention（2024）和 2026-08 的 Unified Radix Cache
用一棵基数树把所有请求的 KV 串起来：公共前缀只算一次、只存一份。friday-llm 在 llama.cpp 上做同样的事。

内核来自 Friday v1 的进程内运行时 friday-llama（Friday 仓库 `7555bc9^:qt/src/runtime/llama`，当时在 Qwen3.5-9B 上实测过）：
连续批处理的调度器、混合模型的 P / B / E 检查点（`SequenceCache`）、落盘（`StateFile`）、看图（`Media`）；
聊天模板与解析改用 b11541 的 `common_chat_session`。v2.7 在它上面加了基数树（`RadixTree`）和序列的自动分配。

## 2. llama.cpp 的版本

`CMakeLists.txt` 的 `LLM_LLAMA_TAG`（现在 b11541）。照着 server 写的几处（换版本时对一遍）：
`ChatFormat::prepare` ↔ `server-common.cpp` 的 `oaicompat_chat_params_parse` 与 `server-schema.cpp` 的采样字段；
`ChatStream` ↔ `server-task.cpp` 的 `task_result_state::update_chat_msg`、`to_json_oaicompat_chat*`；
`Scheduler` ↔ `server-context.cpp` 的 `update_slots` / `process_token`；`Runtime::load` ↔ `server.cpp` 的 `main` / `load_model`。

## 3. 基数树

### 3.1 KV 怎么共享

开统一 KV（`kv_unified`）。统一池里一个格子可以同时属于几个序列，`llama_memory_seq_cp(src, dst, p0, p1)` 只给 `[p0, p1)` 的格子
加上 `dst` 这个序列号，不拷数据（`llama-kv-cache.cpp` 的 `seq_cp`，同一个 stream 时只改元数据）。所以：

- 每条缓存着的对话 / 前缀占一个序列号（`--sequences`，最多 255），它的账本是从根到某个节点的一条路径；
- 几条账本的公共前缀，KV 里那段格子同时挂着这几个序列号——只存一份；
- 腾掉一个序列（`seq_rm`）只去掉它的序列号：别的序列还挂着的格子留着，只有它独占的那段真的空出来。

`RadixTree` 是这些账本的索引（纯数据结构，不碰 KV）：节点的边是一段账本项（文本 token 或整个图片块），分叉处才断开；
每个节点记经过它的序列。`match` 找新提示的最长公共前缀和经过那里的序列，`exclusive` 算一个序列独占的尾巴有多长。

### 3.2 排到一个请求时给它挑序列（`Scheduler::place`）

1. **同一会话**（`x_friday.session`）上次用的空闲序列：原地接着用（账本截到公共前缀 / 退回检查点，== v1）。
2. **正在预填的请求里有和它共用得更多的**（同时派出去的几个子智能体）：等那个请求算过分叉点，再从它分叉，不重复算。
3. **整个账本就是这次提示前缀的空闲序列**：原地接着算，什么都不丢。
4. **分叉**：树上的公共前缀 ≥ 64 项时，新序列先 `seq_cp` 那段前缀（零拷贝），只预填剩下的。
5. 都没有：空序列，或者腾掉最久没用的空闲序列（非主会话优先）。

同时在算的请求最多 `--parallel` 个，排队时优先级高的先（`x_friday.priority`），同级先来先走；预填也按这个顺序填批。

### 3.3 混合 / 循环 / 滑窗模型

Qwen3.5 这类线性注意力（Gated DeltaNet）+ 全注意力的混合模型，循环状态是每个序列一份、固定大小、不能回退。
分叉只能在存了检查点的地方：先 `seq_cp` 注意力那段（循环状态这时跟着来源序列的末尾），再用检查点
（`llama_state_seq_set_data_ext(PARTIAL_ONLY)`）盖掉循环部分——它先 `seq_rm(dst)` 再分配新的格子，不会改到来源序列。
检查点放在 v1 实测过的边界：系统提示末尾 P、每轮用户消息之前 B、提示末尾 E；分叉出来的序列共用来源的检查点字节（`StateBlob`，不拷贝）。
这类模型每个序列多占一份循环状态（Qwen3.5-9B 约 50 MiB），所以序列数少（parallel + 4）。滑窗模型（Gemma）同理（PARTIAL_ONLY 存的是窗口）。
这就是 SGLang Unified Radix Cache 里的 MAMBA / SWA 组件：只有检查点处能复用。

### 3.4 KV 池满

`llama_process` 返回 1：先腾空闲序列（只腾自己独占格子的、非主会话、最久没用的优先），再减半批大小重试，
批大小 1 仍放不下就只让一个请求失败（非主会话里 token 最多的，回 429）。

## 4. 以后

- 落盘一层：被腾掉的会话序列按回合边界存到 `<数据目录>/kv/`（v1 的 `StateFile`，已经编进来了），同一会话再来时先读回来；
- 投机解码：b11541 有 `draft-mtp` / `draft-eagle3` / `ngram-*`，带 MTP 头的模型（Qwen3.5 MTP 版）可以直接用。
