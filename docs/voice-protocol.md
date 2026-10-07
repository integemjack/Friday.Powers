# voice：实时语音协议（v1）

两条 WebSocket：`/v1/realtime/transcribe`（实时听写）和 `/v1/realtime/speak`（边写边读）。
设了 `--token` 时要带 `Authorization: Bearer <token>`，或者在地址后面加 `?token=<token>`。

## 一、实时听写 `GET /v1/realtime/transcribe`

### 客户端 → 服务端

- **二进制帧**：音频。默认 16 kHz、单声道、16 位小端 PCM；任意长度，按实时节奏连续发。
- **文本帧**（JSON）：
  - `{"type":"start","sample_rate":16000,"language":"auto","partial_interval_ms":300,"end_silence_ms":450}`
    —— 可选，在第一段音频之前发；`language` 取 `auto|zh|en|yue|ja|ko`；不发就用默认值。
    `sample_rate` 不是 16000 时服务端自己重采样。其他可选字段：`max_segment_ms`（默认 20000）、
    `vad_threshold`（默认 0.6）、`partials`（默认 true）、`itn`（标点和数字，默认 true）。
  - `{"type":"flush"}` —— 正在说的这句马上定稿（比如用户按了「说完了」）。
  - `{"type":"stop"}` —— 算完剩下的、正在说的定稿。

### 服务端 → 客户端（文本帧，JSON）

| type | 字段 | 说明 |
|---|---|---|
| `ready` | `model`, `device`, `sample_rate` | 连上、模型就绪 |
| `speech_start` | `segment`, `start_ms` | 检测到开口（打断助手用） |
| `partial` | `segment`, `text` | 这句目前听到的（实时字幕），只在变了时发 |
| `final` | `segment`, `text`, `start_ms`, `end_ms`, `language`, `emotion`, `event`, `latency_ms`, `infer_ms` | 一句说完：带标点的定稿；`latency_ms` = 判定说完到出结果 |
| `speech_end` | `segment` | 这句没听出字（噪音、咳嗽）时代替 final |
| `error` | `message` | |

`start_ms` / `end_ms` 从这条连接收到的第一个采样算起。

### 断句

- FSMN-VAD 每 10 毫秒判一帧；最近 200 毫秒里 ≥ 120 毫秒在说话算开口，往前多带 300 毫秒；
- 说完：连着 `end_silence_ms`（默认 450）没说话；一句最长 `max_segment_ms`；
- 说话中每 `partial_interval_ms` 对这句已有的录音整段重识别一次出 `partial`（SenseVoice 非自回归，5080 上一次约 40 毫秒）；
  刚开口那零点几秒模型常把气声猜成「Yeah.」之类，这种不发；
- 说完那一刻对整句识别一次（带逆文本正则化：标点、数字）出 `final`。

## 二、边写边读 `GET /v1/realtime/speak`

大模型一边吐字一边发过来，服务端攒够一句就合成，音频边出边发回去；随时可以打断。

### 客户端 → 服务端（文本帧，JSON）

| type | 字段 | 说明 |
|---|---|---|
| `start` | `voice`, `speed`, `sample_rate`, `instruction` | 可选；`voice` 见 `ready.voices`（空 = 默认），`speed` 0.5–2，`sample_rate` 是发回来的音频的采样率（默认模型的 24000），`instruction` 是 CosyVoice3 的指令（如「用四川话说」，空 = 普通朗读） |
| `text` | `text` | 追加一段字（大模型新吐的 delta，可以是 Markdown） |
| `flush` | | 这一轮说完了：剩下的字念完后回 `done` |
| `say` | `text` | 等于 `text` + `flush` |
| `cancel` | | 打断：没念的丢掉，正在念的马上停；回 `cancelled`，之后不会再有这一轮的音频 |

### 服务端 → 客户端

- **二进制帧**：16 位小端 PCM、单声道，采样率按 `start.sample_rate`，按顺序直接播。
- **文本帧**（JSON）：

| type | 字段 | 说明 |
|---|---|---|
| `ready` | `model`, `device`, `sample_rate`, `voices`, `voice` | 连上、模型就绪；`voice` 是默认音色 |
| `segment_start` | `index`, `text` | 开始念这一段（后面跟着它的音频） |
| `segment_end` | `index`, `audio_ms`, `first_audio_ms`, `infer_ms`, `queued_ms`, `stopped` | 这一段念完（`stopped` = 被打断） |
| `done` | | `flush` 之前的字都念完了 |
| `cancelled` | | 打断完成 |
| `error` | `message`, `index`? | |

### 断句（服务端做）

- 句末（。！？；… 换行，英文 `. ! ? ;` 后面跟空白）切开；数字里的点、冒号、逗号（3.14、10:30、1,000）不切；
- 一轮的头一段到第一个逗号、且有 5 个字（词）就先念，让第一声早点出来；一段超过 40 个字在最后一个逗号处切，80 个字还没逗号硬切；
- 去掉 Markdown 记号（标题、列表、粗体、行内代码的反引号）、``` 代码块整个不念、链接只念文字、网址和表情不念。

## 三、一次性接口（OpenAI 兼容）

- `POST /v1/audio/transcriptions`（multipart：`file`、`language`、`response_format` = `json|text|verbose_json`）。
- `POST /v1/audio/speech`（JSON：`input`、`voice`、`speed`、`instructions`、`response_format` = `wav|pcm`，额外的 `sample_rate`）：
  分块传输，边合成边返回；`wav` 的长度字段写成最大值（流式 WAV）；要 mp3 等也给 wav（没带编码器）；
  认不出的音色名（`alloy` 等）用默认音色。
