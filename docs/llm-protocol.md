# friday-llm 协议

本地大模型 Power。OpenAI 兼容的对话补全，加上几个给 Friday 用的扩展。启动、就绪、stdin、令牌照 [power-spec.md](power-spec.md) §2。

```
friday-llm serve --models <模型库目录> [--models <另一个目录>]… --host 127.0.0.1 --port 0 --token <T> --parent-stdin
                 [--parallel 4] [--sequences 0] [--ctx 0] [--per-sequence 0] [--llama-arg <llama-server 的参数>]…
```

- `--models`：Friday 模型库的目录（魔搭布局 `<目录>/<组织>/<仓库>/*.gguf`，HF 布局 `models--组织--仓库/snapshots/*/`）。
- `--parallel`：同时在算的请求最多几个，其余排队（缺省 4）。
- `--sequences`：缓存着的序列（对话 / 前缀）最多几条；0 = 按模型自动（纯注意力 32，混合 / 循环 / 滑窗模型 parallel + 4）。
- `--ctx`：KV 池大小；0 = 模型的训练上下文，按显存往下调（llama.cpp 的 fit）。序列之间共用前缀，池子不必是「序列数 × 单序列上限」。
- 环境变量 `FRIDAY_LLAMA_ARGS`（空格分开）追加给 llama 的参数，调试用。

## 1. 接口

| 接口 | 说明 |
|---|---|
| `GET /health` | `{"status":"ok"}`；模型按请求加载，不等模型 |
| `GET /v1/info` | `{id, version, "llama.cpp", capabilities, devices[{name,description,type,total,free}], model{loaded, loading, last_error, reserved_mib, loaded_reserve_mib}, in_flight}`；`model.loaded` 有调度统计 `cache`（见 §3） |
| `GET /v1/models` | 模型库目录里的 GGUF：`{data:[{id:"组织/仓库:量化档", path, size, vision, loaded}]}` |
| `POST /v1/chat/completions` | OpenAI 兼容（流式 SSE / 不流式），工具调用、思考（`reasoning_content`）、看图（`image_url` 的 data URL），字段与 llama-server 的 `/v1/chat/completions` 一样 |
| `POST /v1/models/load` | 先把模型加载好（用户在 Friday 里选了它时预热）：`{model, x_friday}` → `{model: <loaded 的信息>}` |
| `POST /v1/memory/release` | 卸载模型、腾显存：→ `{unloaded: <模型 id 或 null>}` |
| `POST /v1/memory/reserve` | 给别的程序留显存（Friday 生图 / 生视频前调，见 §1.1）：`{mib, seconds}` → `{reserved_mib, seconds, unloaded}`；`mib: 0` 取消 |

### 1.1 显存预留

本地大模型和 friday-diffusion 共用一张显卡。大模型按 llama.cpp 的 fit 加载：上下文开到模型的训练长度、按空闲显存往下调，
几乎会占满显卡，生图 / 生视频就只剩自己的权重挪到内存里、计算空间都拿不到（Wan2.2 直接失败）。光卸载也不行：
智能体紧接着的下一个请求又会把模型按满显存加载回来。所以 Friday 在提交生图 / 生视频任务前调 `reserve`：

- 之后加载模型时每张卡至少空出 `mib` MiB（`--fit-target`，放不下就少放几层到显卡、上下文小一些），最多留到整张卡的九成；
- 现在加载着的模型留得不够就卸掉（等在途请求结束），下个请求按新的预留加载；
- `seconds`（缺省 900）后失效；到期或调小后，下个请求来时没有别的请求在算就重新加载，把显存要回来。

Friday 的预留 = 生图 / 生视频模型文件的大小（主模型 + VAE + 文本编码器）+ 计算空间（生图 1.5 GB、生视频 5 GB），
15 分钟（friday-diffusion 闲置 10 分钟才放模型），生视频轮询时续、做完就取消（同时让 friday-diffusion 放掉视频模型）。

## 2. 请求里的 `model` 与 `x_friday`

```json
{
  "model": "unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
  "messages": [...], "tools": [...], "stream": true,
  "x_friday": {
    "model_path": "D:/…/Qwen3.5-9B-Q4_K_M.gguf",
    "mmproj_path": "D:/…/mmproj-F16.gguf",
    "session": "conversation-<uuid>",
    "priority": 1000,
    "main": true
  }
}
```

- 模型：`x_friday.model_path` 直接给文件；没有就按 `model`（文件路径，或「组织/仓库[:量化档]」在 `--models` 目录里找）。
  `mmproj_path` 不给时在模型旁边找 `*mmproj*.gguf`，给空字符串 = 不看图。同一时间只加载一个模型：要的和加载着的不一样时，
  等加载着的那个的请求都结束再换，换的时候来的请求等换完。
- `session`：同一段对话 / 同一个子智能体的请求带同一个值。调度器把它们放在同一个序列上接着算（上一轮的 KV 原样复用）。
  **旁路请求（完成核对、预测下一步、起标题…）不要带主对话的 session**（带了会原地截断主对话的缓存）：不带或带自己的，
  调度器会从主对话分叉出去，只算最后那几百 token。
- `priority`：排队时大的先（主智能体 1000，子智能体用层数，旁路请求放中间）。
- `main`：主智能体的对话。KV 池满时最后腾它。

## 3. 返回

和 llama-server 一样（`usage.prompt_tokens_details.cached_tokens`、`timings`）。另外：

- 流式：`[DONE]` 之前多一块 `{"object":"chat.completion.chunk","choices":[],"x_friday":{"cache":{…}}}`；
- 不流式：回复里多一个 `x_friday.cache`；
- `cache`：`{seq, how, reused, prefilled}`——用的哪个序列、怎么放的（同一会话 / 接着 / 分叉自序列 N（共用 K 项）/ 新序列）、
  复用了多少、实际预填了多少。

出错：`{"error":{"message","type","code"}}`。HTTP 状态：请求不对 400（超长 `type: exceed_context_size_error`）、
模型找不到 404、**KV 池放不下 / 排队太多 429**（Friday 的请求闸门见到 429 会减并发、等一会儿重试）、正在换模型 503、其它 500。
流式已经开始以后出错：发一块 `data: {"error":…}` 再 `[DONE]`。

`/v1/info` 的 `model.loaded.cache`：

```json
{"parallel":4, "sequences":32, "sequences_used":7, "active":2, "queued":0, "requests":42, "forks":11, "dedupe_waits":3,
 "evictions":1, "tokens_reused":182340, "tokens_shared":66012, "tokens_prefilled":20391,
 "tree":{"nodes":12, "items":41022, "logical":203118}}
```

`tree.items` 是树里的项数（共用的只算一次，≈ KV 里真正占的格子），`tree.logical` 是各序列账本加起来（不共用时要占的）。
