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
    `vad_threshold`（默认 0.6）、`partials`（默认 true）、`itn`（标点和数字，默认 true）、
    `speaker`（默认 false：true 时 final / partial 带这句的声纹，见下文「声纹」；`ready.speaker` 为 false 时没有声纹模型、这个字段不起作用）、
    `whisper`（默认 true：装了 Whisper 时定稿用它，见下文「Whisper 定稿」）、`prompt`（交给 Whisper 的热词、上文，如「Secret Chat、Claude Code」）。
  - `{"type":"flush"}` —— 正在说的这句马上定稿（比如用户按了「说完了」）。
  - `{"type":"stop"}` —— 算完剩下的、正在说的定稿。

### 服务端 → 客户端（文本帧，JSON）

| type | 字段 | 说明 |
|---|---|---|
| `ready` | `model`, `device`, `sample_rate`, `speaker`, `whisper` | 连上、模型就绪；`speaker`：有没有声纹模型；`whisper`：定稿用不用 Whisper（v0.3） |
| `speech_start` | `segment`, `start_ms` | 检测到开口（打断助手用） |
| `partial` | `segment`, `text`, `speaker`? | 这句目前听到的（实时字幕），只在变了时发 |
| `final` | `segment`, `text`, `start_ms`, `end_ms`, `language`, `emotion`, `event`, `latency_ms`, `infer_ms`, `model`, `whisper_ms`?, `speaker`? | 一句说完：带标点的定稿；`latency_ms` = 判定说完到出结果；`model`：sensevoice / whisper（v0.3） |
| `speech_end` | `segment` | 这句没听出字（噪音、咳嗽）时代替 final |
| `error` | `message` | |

`start_ms` / `end_ms` 从这条连接收到的第一个采样算起。

### 断句

- FSMN-VAD 每 10 毫秒判一帧；最近 200 毫秒里 ≥ 120 毫秒在说话算开口，往前多带 300 毫秒；
- 说完：连着 `end_silence_ms`（默认 450）没说话；一句最长 `max_segment_ms`；
- 说话中每 `partial_interval_ms` 对这句已有的录音整段重识别一次出 `partial`（SenseVoice 非自回归，5080 上一次约 40 毫秒）；
  刚开口那零点几秒模型常把气声猜成「Yeah.」之类，这种不发；
- 说完那一刻对整句识别一次（带逆文本正则化：标点、数字）出 `final`。

### Whisper 定稿（v0.3，用户 2026-10-10「识别不清楚我说的英语」）

- 中文为主的一句里夹着英文词（「帮我看看 Secret Chat」），SenseVoice 按中文听，常把英文按读音写成汉字（「C亏欠」）。
  装了 Whisper（`whisper/ggml-large-v3-turbo-q5_0.bin`，whisper.cpp v1.9.5，和这里同一份 ggml）时：
  实时字幕、断句、判断是不是人声、认语种照旧用 SenseVoice；认定是人声的一句再交给 Whisper（语种用 SenseVoice 认出的，
  中文先给一句「以下是普通话的句子，中间可能夹着英文。」让它出简体带标点，后面接 `prompt` 的热词）。
- 两边合起来（`Whisper::merge`）：按字 / 英文词对齐（编辑距离），一样的地方中文和标点用 SenseVoice 的、英文词用 Whisper 的（大小写、拼写）；
  不一样的一段——Whisper 那边是英文词、SenseVoice 那边是汉字（把英文写成了汉字）或两边都是英文 → 用 Whisper 的，
  别的（中文对中文、繁简、数字写法）→ 用 SenseVoice 的（它中文更准）。整句是英文（`language` = en）时直接用 Whisper 的。
- Whisper 出错、出空、像是幻觉（「字幕由…提供」「谢谢观看」这类，或比 SenseVoice 长出三倍多）时用 SenseVoice 的。
- 编码窗口：Whisper 默认按 30 秒算，一句几秒的话 M1 上要 4 秒多。备两个状态、窗口各自固定：放得下的句子（约 7.7 秒以内）用 512 帧（约 10 秒）
  的，放不下的用 30 秒的（同一个状态前后两次窗口长度不一样时，后面的句子越来越不准——热词不起作用、英文全小写，whisper.cpp v1.9.5 实测）。
  不做温度回退（M1 上一句会拖到 8 秒多），一句最多出 每秒 15 个 + 24 个 token。
- 用时：5080（CUDA）上一句 4–5 秒的话 Whisper 多花约 0.03–0.05 秒；M1（Metal，不开 flash attention）约 1 秒，10 秒的长句约 4 秒。
  `friday-voice merge 文字A 文字B` 看两边怎么合。
- 一次性听写（`/v1/audio/transcriptions`）同样处理，认 OpenAI 的 `prompt` 字段；带 `whisper=false` 时只用 SenseVoice（`model` 字段不看：Friday 一直传的是 sensevoice-small）。

### 声纹（v0.2，`start.speaker: true`）

- 模型：CAM++（3D-Speaker，CosyVoice 带的 `campplus.onnx`，ggml 上跑，权重直接读 ONNX）；`speaker` 是 192 维单位向量（保留 4 位小数），
  两句的点积就是余弦相似度：同一个人 3 秒左右的一句一般 0.7–0.98，不同人一般 < 0.3；1 秒左右的短句不稳（同一个人 0.4–0.7）。
- `final` 带整句的声纹（不到 0.5 秒的不带）；`partial` 说到 1 秒以上时带、之后每多 0.8 秒再带一次（判断插话的是不是登记过的人）。
- 登记：几段录音分别 `POST /v1/audio/speaker` 取平均再归一化，比一句准；比较、阈值由客户端定（Friday：默认 0.45，短句放宽）。

## 二、边写边读 `GET /v1/realtime/speak`

大模型一边吐字一边发过来，服务端攒够一句就合成，音频边出边发回去；随时可以打断。

### 客户端 → 服务端（文本帧，JSON）

| type | 字段 | 说明 |
|---|---|---|
| `start` | `voice`, `speed`, `sample_rate`, `instruction` | 可选；`voice` 见 `ready.voices`（空 = 默认），`speed` 0.5–2，`sample_rate` 是发回来的音频的采样率（默认模型的 24000），`instruction` 是 CosyVoice3 的指令（如「用四川话说」，空 = 普通朗读）。可以随时再发一次换音色、语气：只管之后交过去的字，还没念完的照原样念（0.4 起；之前的版本连还没开始念的也换） |
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
| `segment_end` | `index`, `audio_ms`, `first_audio_ms`, `infer_ms`, `queued_ms`, `stopped`, `chunks`, `gap_ms` | 这一段念完（`stopped` = 被打断）。0.5.1 起多 `chunks`（每块要的语音 token 数）、`gap_ms`（客户端到了就放的话，这段中间会放空多久，服务端估的）；`infer_ms / audio_ms` 大于 1 就是合成跟不上实时，客户端该先攒一些再放 |
| `done` | | `flush` 之前的字都念完了 |
| `cancelled` | | 打断完成 |
| `error` | `message`, `index`? | |

### 断句（服务端做）

- 句末（。！？；… 换行，英文 `. ! ? ;` 后面跟空白）切开；数字里的点、冒号、逗号（3.14、10:30、1,000）不切；
- 一轮的头一段到第一个逗号、且有 5 个字（词）就先念，让第一声早点出来；一段超过 40 个字在最后一个逗号处切，80 个字还没逗号硬切；
- 去掉 Markdown 记号（标题、列表、粗体、行内代码的反引号）、``` 代码块整个不念、链接只念文字、网址和表情不念。

### 分块（0.5.1，服务端做）

每一段边合成边按块发（CosyVoice3 的语音 token 25 Hz，1 倍速一个 0.04 秒）。0.5.1 时每块都有一笔大的固定开销（flow 带着音色的提示 mel
和这段前面念过的全部重算一遍），块越小第一声越早、总速度越慢：RTX 2060 + Vulkan、克隆音色、1.3 倍速时每块 28 个（模型默认）实时率 1.6——
到了就放会说一截停一下；84 个 0.76，112 个 0.63（2026-11-10 实测）。0.5.2 开了 DiT 缓存（见下一节）后这笔开销几乎没了，块大小主要管第一声早晚。
0.5.1 起块大小不再固定：

- 服务端按「客户端到了就放」估它手里还剩多少没放（`ahead`）。一段的第一块：剩的够等一个大块就直接用大块；不然挑最小的、
  「念完它（加上剩的）时下一块已经合成好」的（28 起、14 一档）；之后每块 84，剩得多时最大到 140。
- 每块的耗时边念边学（按音色，`实测 / 参考` 的系数）。参考（0.5.2 起）是 M1 + Metal、gentle-female、1.3 倍速、10 步、开着 DiT 缓存量的：
  第一块约 0.36 + 0.038·n 秒，之后每块约 0.19 + 0.040·n 秒（0.5.1 用的是 2060 上不开缓存量的 0.77 + 0.0083·n、1.10 + 0.0098·n）。
  快的机器系数小，第一块就小、第一声早；慢的机器、CPU 吃满时系数大，第一块就大些，后面不断。
- `--chunk-tokens N` 仍是每块固定 N 个（调参、对比用）。`speak` 命令多打印每段的分块、到了就放会断几次、要等多久开口才一路不断。

### flow 的 DiT 缓存和合成步数（0.5.2）

用户 2026-11-10：M1 上和本机助手通话，念的话一卡一卡。M1 + Metal 上量（gentle-female，1.3 倍速）：大模型出语音 token 一个约 19 毫秒
（一块 84 个约 1.6 秒），HiFT 一块 0.1–0.5 秒，慢的是 flow 的 DiT——10 个扩散步，每步约 0.5 秒（CFG 两路 × 600 帧，M1 的 GPU
已经算到峰值的一半多，算子没有掉到 CPU 上的），一块 5 秒多。原因是 DiT 的 KV 缓存没开（cosyvoice.cpp 默认不开）：每块的每一步都把
音色的提示 mel（gentle-female 8.8 秒 = 440 帧）和这段前面念过的重算一遍，一段 4 秒的话要 13–15 秒。0.5.2：

- **DiT KV 缓存**：每个扩散步一份（cosyvoice.cpp 的 fixed slots = 步数），每份 2048 个 mel 帧位置（K Q8_0、V Q4_0，10 步约 1.4 GB 显存 /
  统一内存，5 步约 0.7 GB；设备上剩的不够放「缓存 + 模型 + 1 GB」时退到 1280，再不够就不开、退回 0.5.1 的样子，日志里说）。块与块之间只算新的帧。
- **音色提示前缀**（我们给 cosyvoice.cpp 打的补丁，`voice/patches/cosyvoice/0001-dit-prompt-prefix.patch`，配置时自动打）：一个音色第一次念时
  单独对提示跑一遍 DiT（M1 上 gentle-female 10 步约 3.4 秒，只这一次；服务启动预热时默认音色就做了），各步的 KV 留在缓存最前面；之后每段
  都接着它往下算，不再重算提示。提示最后 3 个 token（要看后面的字）随每段重算、从输出里切掉。和原来的差别：提示不再看这段的第一块
  （块因果，和上游 CosyVoice 流式训练时一样）。换音色时重建前缀（缓存里只留一个音色的）。
- **合成步数**：`--diffusion-steps N`（没给时看环境变量 `FRIDAY_VOICE_DIFFUSION_STEPS`，Friday 的「设置 › 能力 › 本地语音」里选：
  标准 10 步 / 快 5 步，见 power.json 的 `options`）。0 = 模型自带的 10 步。`--no-dit-cache` 关掉缓存和前缀（0.5.1 的样子，对比用）。
- 实测（2026-11-10，`speak` 同一段三句话、1.3 倍速、自适应分块，第 2、3 次的平均；实时率 = 合成用时 / 音频长度，首块 = 第一段的第一块）：

  | 机器 | 音色 | 不开缓存（`--no-dit-cache`，≈ 0.5.1） | 0.5.2 10 步 | 0.5.2 5 步 |
  |---|---|---|---|---|
  | M1 16 GB，Metal | gentle-female | 3.18，首块 7.0 秒 | 1.42，3.6 秒 | 1.11，2.8 秒 |
  | M1 16 GB，Metal | default | 2.55，4.7 秒 | 1.44，3.4 秒 | 1.13，2.7 秒 |
  | RTX 2060，CUDA（不是发的 Vulkan 版） | 克隆音色 | 1.58，3.8 秒 | 1.45，3.6 秒 | 1.30，3.3 秒 |

  M1 上 DiT 不再是大头：一块里大模型约占一半（Metal 上大模型不开 flash attention，一个 token 19 毫秒；开了 13 毫秒但念出来是乱的，
  见下）。2060 的 CUDA 版本来就慢在 DiT 以外（10 步比 5 步只多 2 秒），Vulkan 版没量（这台机器没装 Vulkan SDK）。
- 声音：同一种子、同样的语音 token 下和 0.5.1 的输出比，mel 倒谱失真 0.5.2 10 步约 17、5 步约 19，换一个噪声种子（0.5.1 自己）约 27——
  改动带来的差别比扩散采样本身的随机还小；Whisper 听写几种都一字不差（只差标点）。
- Metal 上大模型的 flash attention（0.2.1 关掉的）又查了一遍：开着时念出来是乱的，和图复用、全 0 掩码、ggml-metal 的融合 / 并发 /
  图优化都无关；ggml 的 test-backend-ops 按这个模型的形状（头 64、14 个 q 头对 2 个 kv 头、Q8_0 / F16）在 M1 上全部通过。原因没找到，仍关着。

## 三、一次性接口（OpenAI 兼容）

- `POST /v1/audio/transcriptions`（multipart：`file`、`language`、`response_format` = `json|text|verbose_json`）。
- `POST /v1/audio/speaker`（multipart：`file`）→ `{"embedding": [192 维单位向量], "dim": 192, "duration": 秒}`：一段录音的声纹（登记用；
  不到 0.5 秒回 400；超过 20 秒只用中间 20 秒）。只在装了声纹模型时有（`/v1/info` 的 `capabilities` 有 `speakerEmbedding`）。
- `POST /v1/audio/speech`（JSON：`input`、`voice`、`speed`、`instructions`、`response_format` = `wav|pcm`，额外的 `sample_rate`）：
  分块传输，边合成边返回；`wav` 的长度字段写成最大值（流式 WAV）；要 mp3 等也给 wav（没带编码器）；
  认不出的音色名（`alloy` 等）用默认音色。

## 四、用户音色（v0.5）

用户自己的音色：一段录音克隆出来的，或者别处克隆好的 CosyVoice3 音色文件（prompt_speech gguf，比如「克隆声音」工具音色库里的 `<名字>.gguf`）
导入进来。存在 `<models>/user-voices/<名字>.gguf`（模型目录各版本共用，更新不丢；`--user-voices` 可以换目录），启动时和包里的 `voices/`
一起加载，运行中增删立刻生效（`ready.voices`、`/v1/voices` 里马上有）；和内置音色重名的不收。

- `GET /v1/voices` → `{"voices": [全部], "user_voices": [用户的], "default": "default", "clone": 能不能克隆}`。
- `POST /v1/voice/clone`（multipart）：`file` 录音（wav / mp3 / flac；3 秒起，最好 8～15 秒、只有这一个人、安静；超过 30 秒只用前 30 秒）、
  `name` 音色名（1–40 个字，不能有 `\ / : * ? " < > |`，不以点开头）、可选 `text`（这段录音说了什么；空 = 自己听写，口音重时手填更准）、
  `language`（听写用，默认 auto）、`replace`（`true` 时同名的用户音色换掉）→ `{"voice", "transcript", "seconds", "took"}`。
  同一段录音解两份：16 kHz 给语音分词器和听写，24 kHz 给说话人向量和 mel（从原文件直接解，从 16 kHz 升上来声音发闷）。
  要语音分词器（`frontend-onnx/speech_tokenizer_v3.int8.onnx`）和 campplus 都在、包带了 ONNX Runtime，否则回 501（`/v1/voices` 的 `clone` 为 false）。
- `POST /v1/voice/import`（multipart）：`file`（.gguf）、`name`、`replace` → `{"voice", "took"}`；不是音色文件回 400
  （先用 ggml 的 gguf 解析器检查四个张量 feat / embedding / tokens / text，cosyvoice.cpp 读到坏文件会直接崩）。
- `DELETE /v1/voice/<名字>`（名字按 URL 编码）→ `{"deleted"}`；内置的 403、没有的 404。正在念的那段念完才删。
- 命令行：`friday-voice clone --models <目录> --reference 录音 [--voice 名] [--text 底稿] [--output 音色.gguf]`。
- `/v1/info` 的 `capabilities`：有用户音色目录时 `voiceImport`，能克隆时再加 `voiceCloning`；`models.user_voices` 列用户音色。
