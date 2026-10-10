# Power 规范（v2）

Power 是 Friday 按需下载、按需启动、可以热加载 / 卸载的本地能力。一个 Power 就是**一个可执行文件**（各平台各编一份），
每个 Power 单独下载、单独用；模型不打进文件里：
- 清单 schema 1（voice）：清单列出要的模型，Friday 装 Power 时一起下到它的 models 目录；
- 清单 schema 2（v2.7 起，llm、diffusion）：模型在 Friday 的**模型库**里选（魔搭 / Hugging Face 上的 GGUF，同 v1.x），
  下到模型库的缓存，请求里告诉 Power 用哪几个文件。

## 1. 发布与目录

- 源码：本仓库 `powers/<id>/`，每个 Power 一个 CMake 工程，带 `power.json`。
- 发布：GitHub Release 的附件，文件名 `<id>-<version>-<target>.zip`（里面只有那一个可执行文件 + `power.json` + `LICENSES/`）。
- 索引：`index` Release 里的两份——`index-v2.json`（所有 Power，Friday 2.7 起读）和 `index.json`（只有 schema 1 的，
  Friday 2.6 及以前读，免得旧版看到用不了的 llm / diffusion）。`ci/make_index.py` 发哪个 Power 就只换它那一项：

```json
{
  "schema": 1,
  "powers": [
    {
      "id": "voice",
      "title": "本地 GPU 实时语音",
      "version": "0.1.0",
      "detail": "实时听写（边说边出字幕，说完一句出定稿）、流式合成（边写边读）",
      "capabilities": ["transcription", "realtimeTranscription"],
      "targets": {
        "windows-x64-vulkan": { "url": "…", "sha256": "…", "size": 0 },
        "windows-x64-cuda":   { "url": "…", "sha256": "…", "size": 0 },
        "macos-arm64-metal":  { "url": "…", "sha256": "…", "size": 0 },
        "linux-x64-vulkan":   { "url": "…", "sha256": "…", "size": 0 },
        "linux-x64-cuda":     { "url": "…", "sha256": "…", "size": 0 },
        "linux-arm64-jetson-orin": { "url": "…", "sha256": "…", "size": 0 },
        "linux-arm64-vulkan": { "url": "…", "sha256": "…", "size": 0 }
      }
    }
  ]
}
```

target 命名：`<os>-<arch>-<加速>`。Friday 按本机情况挑：Windows / Linux x64 默认 vulkan，有 NVIDIA 驱动时可选 cuda；
Jetson 按 `/etc/nv_tegra_release` 认 Orin（R36）；初代 Jetson Nano（R32）v2.7 起不支持。

schema 2 的清单多两个字段：`library`（`{format: "gguf", kinds: ["text","multimodal"] | ["image","video"]}`：模型在模型库里选）、
`families`（diffusion 的模型家族，见 [diffusion-protocol.md](diffusion-protocol.md) §1）。

## 2. 和 Friday 的约定（宿主协议）

启动：

```
<exe> serve --host 127.0.0.1 --port 0 --data-dir <目录> [--device auto|cpu|<序号>] [--token <随机串>]
```

- 参数模板里的占位：`{models}`（Power 自己的 models 目录）、`{token}`、`{library}`（schema 2：模型库的目录，魔搭和 HF 两个缓存；
  Friday 把它前面的参数名跟着每个目录重复一遍，`--models A --models B`）。
- `--port 0` 由系统分配端口；就绪后在 **stdout 打一行 JSON**：`{"event":"listening","url":"http://127.0.0.1:53211"}`。
- `GET /health` → `200 {"status":"ok"}` 表示可以用了（模型懒加载：第一次请求才加载，`/health` 不等模型）。
- **stdin 被关掉就退出**（Friday 退出、崩溃时 Power 跟着走）；Windows 上 Friday 再用 Job Object 兜底。
- `--token`：给了的话，所有请求要带 `Authorization: Bearer <token>`（局域网里给别的机器用时）。
- 日志写 stderr；进度、状态写 stdout（每行一个 JSON 事件）。
- 其他子命令：`info`（打印设备、后端、显存，JSON）、`models list|pull <名>`（管理模型）、`--version`。

热加载：Friday 停掉旧进程、换文件、再拉起来即可；Power 不在 Friday 进程里，不需要重启 Friday。

## 3. 能力

| 能力 id | 接口 | 说明 |
|---|---|---|
| `transcription` | `POST /v1/audio/transcriptions`（OpenAI 兼容） | 一次性听录音；Friday 现有的「OpenAI 兼容」提供方直接能用 |
| `realtimeTranscription` | `GET /v1/realtime/transcribe`（WebSocket） | 实时听写，见 `voice-protocol.md` |
| `speech` | `POST /v1/audio/speech`（OpenAI 兼容） | 合成，边合成边分块返回（wav / pcm） |
| `realtimeSpeech` | `GET /v1/realtime/speak`（WebSocket） | 边写边读：文字流进去、音频流出来，可打断，见 `voice-protocol.md` |
| `chat` | `POST /v1/chat/completions`（OpenAI 兼容 + `x_friday`） | 本地大模型（llm），多并发 + 基数树前缀共享，见 `llm-protocol.md` |
| `imageGeneration` / `videoGeneration` | `POST /v1/jobs`、`POST /v1/images/generations` | 本地生图 / 生视频（diffusion），见 `diffusion-protocol.md` |

吃显存的 Power（llm、diffusion）另有 `POST /v1/memory/release`：卸载模型、腾显存。llm 还有 `POST /v1/memory/reserve`：给别的程序留显存（Friday 生图 / 生视频前让 llm 留出地方，见 [llm-protocol.md](llm-protocol.md) §1.1）。

`speech` / `realtimeSpeech` 只在装了合成模型（CosyVoice3 + 至少一个音色）时出现在 `/v1/info` 的 `capabilities` 里。`speakerEmbedding`（声纹，v0.2）只在装了 `campplus.onnx` 时出现。

`GET /v1/info` 返回 Power 的 id、版本、能力、设备、已加载的模型。
