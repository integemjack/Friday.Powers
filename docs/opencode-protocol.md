# opencode Power（friday-opencode）

把本机的 OpenCode（https://opencode.ai，V2）接进 Friday：模型菜单里多一组「OpenCode」，列出用户在 OpenCode 里能用的模型；
选了它的对话交给 **OpenCode 自己的智能体**（它自己的系统提示和工具）在这段对话的工作目录里做，内容实时流回 Friday。

为什么不让 OpenCode 只当「模型」、跑 Friday 自己的智能体循环：OpenCode 的免费模型只给 OpenCode 自己的智能体用，
网关按请求内容判断，换成别的系统提示 / 工具就回 429 `FreeUsageLimitError`（2026-11-11 实测）。所以 Friday 只传用户的话。

## 1. 连哪个 OpenCode（backend.rs）

1. `FRIDAY_OPENCODE_URL`（+ `FRIDAY_OPENCODE_PASSWORD`）：测试用，直接连这个服务；
2. 用户的后台服务：`$XDG_STATE_HOME/opencode/service.json`（默认 `~/.local/state/opencode/service.json`，桌面版开着时就是它）
   里的 `url`、`password`、`version`；`GET /api/agent` 通了就用——登录、额度、身份都是用户自己的；
3. 没有就找装着的 OpenCode：`FRIDAY_OPENCODE_BIN`、PATH 上的 `opencode`、`~/.opencode/bin/opencode`、
   桌面版自带的 `…/ai.opencode.desktop/cli/<版本>-<架构>/opencode-cli`，自己拉起 `opencode serve --hostname 127.0.0.1 --port 0`
   （随机密码，跟着本程序退出；数据目录和用户共用，登录一致）。1.x（V1）不支持；
4. 都没有：`/v1/models`、`/v1/chat/completions` 回 503，说要装 OpenCode。

连不上（桌面版重开换了端口）时丢掉、重新找一次。

## 2. 接口

宿主协议同 power-spec.md §2（`serve --host 127.0.0.1 --port 0 --data-dir {models} --token {token} --parent-stdin`）。

| 接口 | 说明 |
|---|---|
| `GET /health` | `{"status":"ok"}` |
| `GET /v1/info` | 版本、连的哪个 OpenCode（`backend`）、找到的 opencode 程序、记着几段对话、`active`（正在跑的对话请求数：宿主停闲置的 Power 之前看它，OpenCode 一轮可能跑很久） |
| `GET /v1/models` | OpenCode 的 `/api/model`：`id` = `服务商/模型`，`name`、`context_length`、`input_modalities`、`free`（价格都是 0）、`backend` |
| `POST /v1/chat/completions` | OpenAI 兼容（流式 / 非流式），见下 |

### 2.1 chat/completions

- **只用最后一句用户话**（文字 + `image_url` 图片，图片作为附件给 OpenCode）；Friday 的系统提示、`tools` 都不传。
- `x_friday`：
  - `session`：这段对话（`conversation-<uuid>`）或子智能体的标识。同一个值（同一个工作目录）→ 同一个 OpenCode 会话
    （映射存在 `<data-dir>/sessions.json`，60 天没用的忘掉），OpenCode 记得前面聊过的；
    新建会话时前面的对话（在别的模型上聊的）整理成一段带上（末尾 12000 字）。
    没有 `session` = 一次性的辅助请求（起标题、总结…）：Friday 的要求和前文写进这一句，OpenCode 的工具全禁，答完删会话。
  - `cwd`：工作目录（绝对路径、存在）。OpenCode 在这里读写、跑命令（会话的 location）；没有就用 `<data-dir>/work`。
  - `permission`：会话框下的权限档 → OpenCode 会话的规则（每次请求都按当前的改）：
    - `auto`（自动，默认）：读、改文件不问，跑命令、上网、出工作目录等先问；
    - `plan`（计划）：只读（改文件、跑命令都不许），用 OpenCode 的 plan 智能体；
    - `full`（完全）：都不问。
  - `toolEvents`：`true` = Friday 认结构化的工具事件（见下），工具活动画成 Friday 原生的工具行；没有（旧版 Friday）就写进思考区。
  - `questions`（0.1.2，Friday 2.97.25 起）：`true` = Friday 把等回答的提问行画成选项卡片，提问时正文不再列问题。
- 流式输出：文字 → `content`；OpenCode 的思考 → `reasoning_content`；最后一块带 `usage`
  （`prompt_tokens` = 最后一步的上下文长度，输出各步相加）。工具活动：
  - `toolEvents` 时：单独一块，`delta` 为空、顶层 `x_friday.tool`：
    `{"id", "name"（OpenCode 的工具名：read / write / edit / shell / grep / glob / list / webfetch / websearch …）, "title"（中文名）,
    "summary"（路径 / 命令等，工作目录下的写成相对路径）, "input"（原样参数）, "status": "running"}`，
    之后同一个 `id` 再来一块 `"status": "success" | "failed" | "waiting"`（等用户确认）带 `"output"`（结果全文，截到 8000 字）、`"metadata"`，
    每块都带着 `name` / `title` / `summary` / `input`（放行后接着那次请求里报它做完时也带：客户端那边原来那行可能已经不在了）；
    Friday 只显示，不执行。
  - 否则写进 `reasoning_content`（`▸ 读取 numbers.txt` / `  ✓ …` / `  ✗ …`）。
- **权限确认**：OpenCode 要问（`permission.asked`）时（`toolEvents` 时先给对应的工具发一条 `"status": "waiting"`，正文只一句），
  这次回答以一段说明结束（「OpenCode 想运行命令 … 回复『允许』…」），
  OpenCode 那边停着等。同一段对话的下一句：`允许` / `可以` / `好` / `继续`…（12 个字以内）= 放行这一次，
  `总是允许` = 以后同类不再问，然后接着把后面的内容流回来；说别的 = 拒绝，再按新的话做（Friday 附在后面的「（发出时间：…）」不算）。
  Friday 那边对 OpenCode 模型不做交付检查（它追的提醒会被当成「不同意」）。
- **提问**（0.1.2）：OpenCode 的 `question` 工具建一张表单（事件 `form.created`，会话 id 在 `data.form.sessionID`；
  `form.metadata.tool.id` 是那次工具调用，`fields` 每题一项：`key`（q0、q1…）、`description`（问题）、`type`（`string` 单选 / `multiselect` 多选）、
  `options[{value, label, description}]`、`custom`（能自己写））然后一直等。桥给提问那一行发 `"status": "waiting"`（「等你回答」），
  没有 `questions` 时正文写出问题和带序号的选项，**这次回答到此结束**。同一段对话的下一句就是回答：几题用「；」隔开（段数对不上就整句当每一题的回答），
  每段写选项名或序号（「2」「第二个」），多选用「、」或逗号隔开，对不上选项的原样当自己写的，「（跳过）」= 这题不答；
  填成 `{"q0": "中文", "q1": ["表格", "摘要"]}` 回 `POST /api/session/<id>/form/<表单 id>/reply {"answer": …}`，OpenCode 接着做，内容接着流回来。
  填不进去（选项外的回答而这题不许自己写、表单已经没了）→ 取消表单（`DELETE …/form/<id>`）、停下这次执行，再把这句话当新的话发。
- 等确认 / 等回答记在内存里；本程序重启过（闲置 15 分钟被停）时，下一句话进来先问 OpenCode 这段会话有没有还在等的
  （`GET /api/session/<id>/form`、`…/permission`，只列没答的），有就照上面处理，不会把回答当成新的话排在一直等着的表单后面。
- 出错：还没写出任何内容时回 HTTP 错误码（OpenCode 报的 `status`，如 429 免费额度用完 / 402 余额不足），
  带 `Retry-After`（OpenCode 的 `retryAfterMs`）；已经在流了就在正文末尾写一句。
- Friday 断开（点了停止）→ 中止 OpenCode 那次执行。15 分钟没有任何事件 → 中止、504。
  OpenCode 没动静时每 15 秒发一块空的 `delta`（不是 SSE 注释：EDITH / Friday 的读流按数据块算没超时）。

## 3. 实现要点

- 诊断日志：`<data-dir>/friday-opencode.log`（每个请求的会话、有没有等着的确认、确认怎么回的；超过 1 MB 从头写）。

- OpenCode 的事件流（`GET /api/event`）按目录分、没有心跳：每个（服务, 目录）只开一条长连接，按 `sessionID` 分给各个请求。
- 会话里的工具调用 OpenCode 自己执行；Friday 看到的只是活动记录。
- 测法：`FRIDAY_OPENCODE_URL` 指一个隔离的 `opencode serve`（配一个假的 OpenAI 兼容服务商），按 Friday 的请求格式发；
  Friday 两端的端到端：本地 `index-v2.json`（`file://` 包）+ `FRIDAY_POWERS_INDEX`，装上后选「OpenCode ▸ …」发消息。
