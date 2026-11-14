# claudecode Power（friday-claudecode）

把用户**自己装的、原样不改的** Claude Code 命令行（`claude`，Anthropic）接进 Friday：模型菜单里多一组「Claude Code」；
选了它的对话交给 **Claude Code 自己的智能体**（它自己的系统提示、工具、登录和额度）在这段对话的工作目录里做，内容实时流回 Friday。
Friday 只传用户的话，不跑自己的智能体循环。模型列表、每个模型能选的思考档都问 Claude Code 自己（不写死），它更新出了新模型 Friday 这边跟着有。

## 0. 约束（必须守住）

- 只拉起用户本机装好的 `claude`，原样调用：不改它、不打包它、不分发它。
- **不读、不复制、不碰** Claude Code 的登录凭据（钥匙串、`~/.claude/.credentials.json`、OAuth 令牌）。
  登录状态只看 `claude auth status` 自己打印的 JSON，而且只留 `loggedIn`、`authMethod`、`apiProvider`、`subscriptionType` 四项
  （邮箱、组织 id 之类一概丢掉）。
- 不加自定义系统提示：发给 Claude Code 的只有用户的话（新开的 Claude Code 会话前面带上这段对话之前的内容）。
  工具结果里给模型的话只有权限确认的回答（「用户没有同意：…」「Friday 现在是计划档（只读）…」）。

## 1. 找 claude（backend.rs）

1. `FRIDAY_CLAUDE_BIN`；
2. PATH 上的 `claude`（Windows：`claude.exe`、`claude.cmd`）；
3. `~/.local/bin/claude`、`~/.claude/local/claude`、`/opt/homebrew/bin/claude`、`/usr/local/bin/claude`；
   Windows：`%USERPROFILE%\.local\bin\claude.exe`、`%APPDATA%\npm\claude.cmd`；
4. 都没有：`/v1/models`、`/v1/chat/completions` 回 503，说要装 Claude Code、`claude auth login`。

版本看 `claude --version`（「2.1.276 (Claude Code)」，按程序路径 + 修改时间缓存）；登录状态 `claude auth status --json`（缓存 60 秒，
没登录时退出码 1 但照样打印 JSON；出现登录错误后丢掉缓存）。

拉起 claude 时继承本程序的环境（`CLAUDE_CONFIG_DIR`、`ANTHROPIC_*`、代理、`PATH`、`HOME` 都留着），但去掉从上层 Claude Code 会话
漏下来的 `CLAUDECODE`、`CLAUDE_CODE_*`、`CLAUDE_AGENT_SDK_*`、`CLAUDE_PID`、`CLAUDE_EFFORT`、`CLAUDE_PREVIEW_*`（带着它们 claude 会认错登录）。
Windows 上不弹控制台窗口；macOS / Linux 上 claude 自己一个进程组，停的时候连它跑的命令、MCP 子进程一起停。

## 2. 接口

宿主协议同 power-spec.md §2（`serve --host 127.0.0.1 --port 0 --data-dir {models} --token {token} --parent-stdin`），
stdin 关了就停掉所有 claude 再退出。

| 接口 | 说明 |
|---|---|
| `GET /health` | `{"status":"ok"}` |
| `GET /v1/info` | `id`、`version`、`capabilities: ["agentChat"]`、`claude`（`path`、`version`）、`auth`（上面那四项）、`conversations`（记着几段对话）、`running`（常驻的 claude 进程数）、`active`（正在跑的对话请求数，宿主停闲置的 Power 之前看它）、`rateLimit`（最近一次见到的 `rate_limit_info`，没有为 null）、`error`（没找到 claude 时） |
| `GET /v1/models` | **问 Claude Code 要**（0.2.0 起，不写死）：起一个 `claude -p --input-format stream-json --output-format stream-json --verbose --no-session-persistence --tools "" --strict-mcp-config`（`<data-dir>/work` 里），发流式输入的 `initialize` 控制请求（`{"type":"control_request","request_id":…,"request":{"subtype":"initialize"}}`，Agent SDK 的 `supportedModels()` 就是它），回的 `control_response` 里的 `models` 是 Claude Code 自己 `/model` 里常用的那几个（跟着装的版本和登录的账号变，约 1 秒）。0.3.0 起再加 **Anthropic 公开的 Claude Code 模型目录**（`https://downloads.claude.ai/model-catalog/v1/catalog.json` 的 `surfaces.cc`，Claude Code 自己的 `/model` 也用它；公开签名文件，不用登录，6 小时缓存、存 `<data-dir>/catalog.json`，`FRIDAY_CLAUDE_CATALOG` 可指本地文件）里它没列的具体版本：`section` 不是 `deprecated`、`offered_on` 含当前登录方式（`first_party` / `bedrock` / `vertex` / `foundry`）、按 `resolvedModel` 去重；`min_claude_code_version` 比装的 claude 新的不列，名字写进 `backend.needsUpdate` 和 `detail`（「更新 Claude Code 后还能选：Opus 5.5」）。每项：`id`（= 它的 `value`，`--model` 的值：`default` 不传 `--model`、`opus[1m]`、`sonnet`、`haiku`、完整模型名…）、`name`（`displayName`；`default` 写「默认」）、`description`、`resolved_model`（别名现在指哪个）、`efforts`（`supportsEffort` 时的 `supportedEffortLevels`，目录里的取 `runtime.effort_levels`：`--effort` 能取的值，空 = 不分档，只能开 / 关思考）、`adaptive_thinking`、`group`（`main` 常用：Claude Code 自己列的和目录 main 组的；`overflow` 其他版本：Opus 4.7、Sonnet 4.6……，Friday 放进「更多版本」），另有 `input_modalities: ["text","image"]`、`free: false`、`billing`（`authMethod` 是 `claude.ai` 时 `plan`，否则 `metered`）。60 秒内用缓存；取到的存进 `<data-dir>/models.json`，取不到时用存下的（`backend.stale: true`，`detail` 后面注明）。顶层 `backend`：`detail`（「Claude Code 2.1.276 · Max 订阅」/「未登录：在终端运行 claude auth login」）、`version`、`loggedIn`、`stale`。没装 claude 回 503，取不到又没有存下的回 502 |
| `GET /v1/usage` | 0.3.0：额度。起一个不留会话的 claude 发 `get_usage` 控制请求（`skip_behaviors: true`；就是 `/usage` 的数据，claude 用它自己的登录去查，本程序不碰凭据，约 2 秒，缓存 30 秒，`?fresh=1` 不用缓存），换成 Friday 的额度报告：`{report: {plan（"Max"…）, subscription, windows: [{title（5 小时 / 每周 / 每周 Fable…，按 `rate_limits.limits[]` 的 kind / scope，旧版按 five_hour / seven_day…）, remainingPercent, resetAt（ISO 8601，整秒 UTC）}], fetchedAt}, available, plan}`；API 密钥 / Bedrock 这类 `rate_limits_available: false` 的没有窗口。没装 claude 503，问不到 502 |
| `POST /v1/chat/completions` | OpenAI 兼容（流式 / 非流式），见下 |
| `POST /internal/permission` | 内部：`mcp` 子命令转来的权限确认（带内部口令 `X-Friday-Internal`，不是宿主的 token），见 §3 |

### 2.1 chat/completions

- `model`：`/v1/models` 报的 id 之一（`claudecode/` 前缀也认；别的字符串原样当 `--model` 传，Claude Code 认别名和完整模型名）。
- **只用最后一句用户话**（文字 + `image_url` 图片：`data:` 地址转成 Anthropic 的 base64 image 块，网址转成 url image 块）；
  Friday 的系统提示、`tools` 都不传。
- `x_friday`：
  - `session`：这段对话（`conversation-<uuid>`）或子智能体的标识。同一个值（同一个工作目录）→ 同一个常驻 claude 进程 / 同一个 Claude Code 会话
    （映射 key = sha256(session + "\n" + cwd) 前 24 位，会话 id 存在 `<data-dir>/sessions.json`，60 天没用的忘掉）。
    全新的 Claude Code 会话第一句前面带上这段对话之前的内容（末尾 12000 字）。
    没有 `session` = 一次性的辅助请求（起标题、总结…）：Friday 的要求和前文写进这一句，起一个 `--tools "" --strict-mcp-config --no-session-persistence`
    的 claude，答完就退。
  - `cwd`：工作目录（绝对路径、存在），claude 的当前目录；没有就用 `<data-dir>/work`。
  - `permission`：会话框下的权限档 → claude 的 `--permission-mode`（变了就重开进程、`--resume` 接回）：
    - `auto`（自动，默认）→ `default`：改**工作目录里的**文件（Edit / Write / MultiEdit / NotebookEdit）和待办由本程序直接放行，
      其余要确认的（跑命令、上网、改工作目录以外的文件、派子智能体…）在对话里问用户。
      不用 `acceptEdits`：它连工作目录里的 `rm`、`echo >` 都不问（2.1.276 实测）；
    - `plan`（计划）→ `plan`：只读，要确认的一律拒（ExitPlanMode 也拒：只出方案、不动手）；
    - `full`（完全）→ `bypassPermissions`：都不问。
  - `toolEvents`：`true` = Friday 认结构化的工具事件（见下）；没有（旧版 Friday）就写进思考区（`▸ 运行命令 …` / `  ✓ …` / `  ✗ …`）。
  - `questions`（0.1.1，Friday 2.97.25 起）：`true` = Friday 把等回答的提问行（`AskUserQuestion`）画成选项卡片，提问时正文不再列问题。
  - `thinking`（0.2.0）：思考档。`off` = 不想（拉起时设环境变量 `MAX_THINKING_TOKENS=0`，请求里 `thinking: {type: disabled}`）；
    `on` = 想、不指定档（不分档的模型，如 Haiku）；其余（`low` / `medium` / `high` / `xhigh` / `max`，按 `/v1/models` 里这个模型的 `efforts`）
    → `--effort <档>`（请求里 `output_config.effort`）；选了要想时去掉环境里继承的 `MAX_THINKING_TOKENS`。没有 = 用 Claude Code 自己的设置。
    和模型、权限档一样，变了就重开进程、`--resume` 接回（在等确认 / 回答时不重开，下一句再换）。
  - `continuing`（0.1.2，Friday 2.97.26 起）：`true` = 这句是 Friday 自己接着做的话（出错后自动恢复、点了重试、换了模型接着做），不是用户说的：
    还在等的提问 / 确认不拿它当回答，那一行再发一次 `waiting`、正文同第一次问时，这次回答到此结束（claude 那边照旧停着等）。
- 流式输出：主智能体的文字 → `content`；思考 → `reasoning_content`（只在有字时：0.2.0 起拉起时带隐藏参数 `--thinking-display summarized`，
  思考块里是摘要；不带时无头模式默认 `omitted`，只有签名、没有字）；最后一块带 `usage`
  （`prompt_tokens` = 最后一次调用的 input + cache_read + cache_creation，`completion_tokens` = 各轮 output 相加），
  和顶层 `x_friday.rateLimit`（最近的 `rate_limit_info`，Friday 不认就忽略）。
  一轮里没经模型（本地 / 命令之类）时把 `result` 里的文字当正文。
- 工具活动（`toolEvents` 时）：单独一块，`delta` 为空、顶层 `x_friday.tool`：
  `{"id"（tool_use_id）, "name"（Claude Code 的工具名：Read / Write / Edit / Bash / Grep / Glob / WebFetch / WebSearch / Agent / TodoWrite …）,
  "title"（中文名：读取文件 / 写入文件 / 修改文件 / 运行命令 / 搜索内容 / 查找文件 / 打开网页 / 搜索网页 / 子智能体 / 待办…）,
  "summary"（文件路径（工作目录下的写成相对路径）/ 命令第一行 / 搜索的模式 / 网址 / 子智能体的描述）, "input"（参数，路径同样相对）, "status": "running"}`；
  之后同一个 `id` 再来 `success` / `failed`（带 `output`：结果全文，截到 8000 字）/ `waiting`（等用户确认，`output` 写等什么）。
  每块都带着 `name` / `title` / `summary` / `input`（放行后接着的那次请求里也带）。
  子智能体里的工具也画成工具行，`metadata.parent` = 派它的那个 Agent 工具的 id；子智能体的文字不显示。
  后台子智能体：Agent 那行先是 `running`（`output`「在后台运行…」），它做完（`task_notification`）才报 `success` / `failed`，`output` 是它的总结。
- **权限确认**：claude 要问的（见 §3）：`toolEvents` 时给那个工具发一条 `waiting`，正文追加一句
  「Claude Code 想运行命令，等你确认：点上面的「允许执行」或回复「允许」（「总是允许」以后同类不再问）；回复别的就取消，按你新说的做。」
  （没有 `toolEvents` 时是一段带命令的说明），**这次回答到此结束**，claude 那边的权限调用一直挂着。同一段对话的下一句：
  - `允许` / `可以` / `好` / `继续`…（12 个字以内）= 放行这一次；`总是允许` = 这个工具名以后都放行（本程序活着时一直算），同类也在等的一起放行；
    然后接着把这一轮后面的内容流回来；
  - 说别的 = 拒绝（`{"behavior":"deny","message":"用户没有同意：<那句话>"}`，其它也在等的一起拒），再把这句话写给 claude：
    Claude Code 会把一轮里写进去的话并进正在跑的这一轮（模型看到「The user sent a new message while you were working」），或者排成下一轮；
  - Friday 附在后面的「（发出时间：…）」不算。
  - `AskUserQuestion`：工具行 `waiting`「等你回答」（没有 `questions` 时正文列出问题和选项），用户下一句就是回答：几题用「；」隔开
    （段数对不上就整句当每一题的回答），每段写选项名或序号，多选用「、」或逗号，对不上的原样当自己写的，「（跳过）」= 这题不答；
    以放行交回（0.1.1；同 Agent SDK：`{"behavior":"allow","updatedInput":{…原参数, "answers": {"<问题>": "<选项名>"}}}`，多选用「, 」连），
    它的提问工具照常成功（结果是「User has answered your questions: …」）；0.1.2 起它做完不再报给 Friday（那一行已经记着「回答：…」）。
- 出错：还没写出任何内容时回 HTTP 错误码——额度用完（`rate_limit_event` 的 `status` 是 `rejected`，或文字「hit your … limit」，或 API 429）→ **429**，
  `Retry-After` = `resetsAt` − 现在（只有订阅登录时 Claude Code 才报 `rate_limit_event`；用 API 密钥时没有重置时间，不带 Retry-After）；
  没登录 / 登录过期 → **401**，说要在终端运行 `claude auth login`；Claude 太忙（529）→ 503；其它 → 502。
  Claude Code 对 401 会退避重试 10 次（约 3 分钟）：见到第一个 `api_retry`（`authentication_failed`）就停下直接回 401；
  429 且额度状态是 `rejected` 时同样不等。已经在流了就在正文末尾写「（Claude Code 出错：…）」。
- Friday 断开（点了停止，写不出去）→ 停掉这段对话的 claude 进程（下一句 `--resume` 接回）。同一段对话来了新的请求而旧的还在流 → 旧的停下让位。
  15 分钟没有任何输出 → 停掉、504。
  15 秒没往 Friday 写过东西就发一块空的 `delta`（不是 SSE 注释：EDITH / Friday 的读流按数据块算没超时）。

## 3. 实现要点

### 3.1 常驻进程（conv.rs）

每段对话一个常驻进程（每句新起一个要 2.4–3.9 秒，常驻的第一句以后几乎没有开销）：

```
claude -p --input-format stream-json --output-format stream-json --include-partial-messages --verbose --forward-subagent-text
       [--model <m>] [--effort <档>] [--thinking-display summarized] --permission-mode <mode>
       --mcp-config <data-dir>/mcp/<key>.json --permission-prompt-tool mcp__friday__approve
       [--resume <会话 id>]
```

`--effort`、`--thinking-display` 是后来才有的参数（后者不在 `--help` 里）：每个 claude（按程序路径和修改时间）探一次——关着输入跑
`claude -p … --effort high --thinking-display summarized`，认得的 1 秒内正常退出，不认的立刻报 `unknown option '--x'`，去掉再试；不认的就不带。

当前目录 = 对话的工作目录。每句话往 stdin 写一行
`{"type":"user","uuid":"<随机 UUID>","message":{"role":"user","content":[{"type":"text",…},{"type":"image",…}]}}`。
换模型 / 权限档、Friday 停止、被外面杀掉、闲置 15 分钟（有等用户确认的放 12 小时）→ 停掉，下一句 `--resume` 接回；
`--resume` 的会话找不到了（`result` 的 `errors` 里「No conversation found」，还没 `init`）→ 忘掉旧 id、开新会话、带上之前的对话重发。

### 3.2 一次请求什么时候算完（chat.rs 的 pump）

Claude Code 2.1.276 的行为（实测）：

- 每句带 `uuid` 的话有 `command_lifecycle`：`queued` → `started` → `completed`。**一轮在用工具时写进去的话会并进正在跑的那一轮**
  （不单独开一轮、没有自己的 `result`），排在文字流着时写的则排成下一轮；两种情况都报 `completed`。所以「写进去的话都答完了」按 uuid 的 `completed` 算
  （老版本没有 `command_lifecycle` 时退回按 `result` 计数）。
- 子智能体（Agent）默认在**后台**跑：这一轮先出 `result`，子智能体做完时报 `task_notification`，Claude Code 再自己开一轮、出第二个 `result`。
  后台子智能体在这一轮结束前就做完的，`result` 之后也会再开一轮。Bash 命令也报 `task_started` / `task_notification`（`is_backgrounded: false`，在一轮之内）。
- `system` `status` = `requesting`：开始一次模型调用；`init`：每轮开头。

所以一次请求结束于：写进去的话都 `completed`、主智能体不在一轮里（最后见到的是 `result`）、要等的后台任务都做完了
（子智能体要等；`run_in_background` 的命令——开发服务器之类——不等）；`result` 之后报了 `task_notification` 的（或这一轮里有后台任务做完的）
再等 Claude Code 自己开的那一轮（`result` 后 3 秒 / `task_notification` 后 10 秒内没开就不等了）。
两次请求之间攒下的输出（后台命令做完后 Claude Code 自己开的一轮）在下一次请求开头只记账（会话 id、任务、`result`）、不发给 Friday。

### 3.3 权限（mcp.rs + conv.rs）

- `--permission-prompt-tool mcp__friday__approve`：Claude Code 要确认时调 MCP 服务 `friday` 的工具 `approve`，参数 `{tool_name, input, tool_use_id}`，
  要的回答是工具结果文字里的 JSON：`{"behavior":"allow","updatedInput":<input>}` / `{"behavior":"deny","message":"…"}`。
  调用可以挂很久（实测 180 秒），Claude Code 一直等；带 tool_use 的 `assistant` 消息在权限调用之前就到了。拒绝 → 一个出错的 `tool_result`，模型接着做。
- MCP 服务就是本程序：`friday-claudecode mcp --url http://127.0.0.1:<端口> --key <对话 key>`（stdio JSON-RPC：`initialize` → `capabilities.tools`，
  `tools/list` → 一个 `approve`，`tools/call` → POST 本程序的 `/internal/permission` 等到回答；别的带 id 的请求回空结果）。
  配置写在 `<data-dir>/mcp/<key>.json`（Unix 上 0600），内部口令放在配置的 `env`（`FRIDAY_CLAUDECODE_TOKEN`），不上命令行。
- `/internal/permission` 按对话 key 找到这段对话：能自己定的马上回（自动档改工作目录里的文件、待办；计划档一律拒；「总是允许」过的）；
  其余插进这段对话的输出队列（和 claude 的输出排在一起），正在流的请求在对话里问用户（§2.1），HTTP 请求一直挂着等，没有超时；
  进程停了就回「拒绝」。

### 3.4 其它

- 诊断日志：`<data-dir>/friday-claudecode.log`（每个请求、拉起 / 停掉进程、每次 `result`、权限确认怎么回的、claude 的 stderr；超过 1 MB 从头写）。
- 测法（不用真订阅）：假的 Anthropic API（`ANTHROPIC_BASE_URL` 指过去、`ANTHROPIC_API_KEY=sk-ant-fake`、隔离的 `CLAUDE_CONFIG_DIR`），
  桥在干净的环境里起（`env -i …`，不带上层 Claude Code 会话的 `CLAUDE_*`），按 Friday 的请求格式发：文字、常驻复用、
  跑命令 → 等确认 → 允许 / 拒绝 / 总是允许、自动档写文件不问、工作目录以外要问、后台子智能体、后台命令、停止后 `--resume`、
  会话丢了重开、换模型 / 权限档、一次性请求、非流式、没有 toolEvents、AskUserQuestion、401 / 429 / 502、心跳、退出时停掉所有 claude。
  `cargo test` 覆盖确认的话、Retry-After、路径判断、工具行。
