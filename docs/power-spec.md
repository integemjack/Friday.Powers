# Power 规范（v1 草案）

Power 是 Friday 按需下载、按需启动、可以热加载 / 卸载的本地能力。一个 Power 就是**一个可执行文件**（各平台各编一份），
模型不打进文件里：第一次用到时由 Power 自己下载到 Friday 指定的数据目录。

## 1. 发布与目录

- 源码：本仓库 `powers/<id>/`，每个 Power 一个 CMake 工程，带 `power.json`。
- 发布：GitHub Release 的附件，文件名 `<id>-<version>-<target>.zip`（里面只有那一个可执行文件 + `power.json` + `LICENSES/`）。
- 索引：Release 里的 `index.json`，Friday 从这里取「有哪些 Power、每个平台下哪个、校验值」：

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
        "linux-arm64-jetson-orin": { "url": "…", "sha256": "…", "size": 0 },
        "linux-arm64-jetson-nano": { "url": "…", "sha256": "…", "size": 0 },
        "linux-arm64-vulkan": { "url": "…", "sha256": "…", "size": 0 }
      }
    }
  ]
}
```

target 命名：`<os>-<arch>-<加速>`。Friday 按本机情况挑：NVIDIA 优先 cuda，其次 vulkan；Jetson 按 `/etc/nv_tegra_release` 认 nano / orin。

## 2. 和 Friday 的约定（宿主协议）

启动：

```
<exe> serve --host 127.0.0.1 --port 0 --data-dir <目录> [--device auto|cpu|<序号>] [--token <随机串>]
```

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

`speech` / `realtimeSpeech` 只在装了合成模型（CosyVoice3 + 至少一个音色）时出现在 `/v1/info` 的 `capabilities` 里。`speakerEmbedding`（声纹，v0.2）只在装了 `campplus.onnx` 时出现。

`GET /v1/info` 返回 Power 的 id、版本、能力、设备、已加载的模型。
