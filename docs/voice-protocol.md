# voice：实时听写协议（v1 草案）

`GET /v1/realtime/transcribe`，升级成 WebSocket。

## 客户端 → 服务端

- **二进制帧**：音频。默认 16 kHz、单声道、16 位小端 PCM；任意长度，按实时节奏连续发。
- **文本帧**（JSON）：
  - `{"type":"start","sample_rate":16000,"language":"auto","partial_interval_ms":300,"end_silence_ms":450}`
    —— 可选，开始前发；`language` 取 `auto|zh|en|yue|ja|ko`；不发就用默认值。
    `sample_rate` 不是 16000 时服务端自己重采样。
  - `{"type":"flush"}` —— 把正在说的这句马上定稿（比如用户按了「说完了」）。
  - `{"type":"stop"}` —— 定稿后关闭。

## 服务端 → 客户端（文本帧，JSON）

| type | 字段 | 说明 |
|---|---|---|
| `ready` | `model`, `device` | 连上、模型就绪 |
| `speech_start` | `segment`, `start_ms` | 检测到开口（打断助手用） |
| `partial` | `segment`, `text` | 这句话目前听到的（实时字幕），只在变了时发 |
| `final` | `segment`, `text`, `start_ms`, `end_ms`, `language`, `latency_ms` | 一句说完：带标点的定稿；`latency_ms` = 判定说完到出结果 |
| `speech_end` | `segment` | 这句没听出字（噪音、咳嗽）时代替 final |
| `error` | `message` | |

`start_ms` / `end_ms` 从这条连接收到的第一个采样算起。

## 断句

- Silero VAD（每 32 毫秒一帧），开口：语音概率 > 0.5 连续 ≥ 100 毫秒；
- 说完：低于阈值连续 `end_silence_ms`（默认 450）；一句最长 20 秒，到了在最安静的地方切开；
- 说话中每 `partial_interval_ms` 对这句已有的录音整段重识别一次出 `partial`（SenseVoice 非自回归，一次几十毫秒）；
- 说完那一刻对整句识别一次（带逆文本正则化：标点、数字）出 `final`。
