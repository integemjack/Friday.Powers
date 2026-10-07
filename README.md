# Friday.Powers

Friday 的**按需能力**（Powers）。

Friday 主程序只接在线模型、保持小巧；要用到本机 GPU 的重能力（实时语音识别、流式语音合成……）做成一个个独立的 Power，
放在这个仓库里，由 Friday **按需下载、启动，随时热加载 / 卸载**，不打进主程序。

## 一个 Power 是什么

- 一个目录 `<id>/`：清单 `power.json` + C++ 代码，每个平台编成**一个可执行文件**（推理库静态链接进去，不带 Python、不装运行库）。
- 运行起来是一个本地服务（独立进程）：Friday 用 `serve --port 0 --parent-stdin` 拉起，从 stdout 读到地址，
  Friday 退出它就跟着退出；换版本就是停掉旧进程、换文件、再拉起。
- 通过 HTTP / WebSocket 提供能力：一次性的（听录音、说话，OpenAI 兼容接口）和实时的（边说边识别、边写边读）。
- 模型不打进包里：清单里列出各平台要的模型（ModelScope 地址、sha256），Friday 第一次用时下载。
- 发布：推 `<id>-vX.Y.Z` 标签由 GitHub Actions 编各平台、发 Release；`index` Release 里的 `index.json`
  列出所有 Power、各平台下载地址和校验值，Friday 从这里取。

规范见 [docs/power-spec.md](docs/power-spec.md)。

## 第一个 Power：`voice`（本地实时语音）

- **实时听写**：SenseVoice-Small + FSMN-VAD。边说边出字幕，一句说完马上给出带标点的定稿（RTX 5080 上一句约 40 毫秒）。
- **边写边读**：CosyVoice3（[cosyvoice.cpp](https://github.com/Lourdle/cosyvoice.cpp)）。大模型边出字，服务端攒够一句就合成、
  音频边出边发，随时可以打断（5080 上实时率约 0.22，第一块音频约 250 毫秒）。
- 用在 Friday 的：桌面端语音模式、助手的 SChat 通话、会议 / 听课的连续听写 + 逐句分析。

| 平台 | 加速 | 说明 |
|---|---|---|
| `windows-x64-vulkan` | Vulkan | 单个 exe（约 49 MB），NVIDIA / AMD / Intel 都能用；默认 |
| `windows-x64-cuda` | CUDA 13 | 带 cuBLAS 的 DLL（约 420 MB）；RTX 5080 上 Vulkan 反而更快，留作备选 |
| `macos-arm64-metal` | Metal | Apple 芯片的 Mac |
| `linux-arm64-jetson-orin` | CUDA（L4T） | JetPack 6.x；CUDA 运行库和 cuBLAS 静态链接 |
| `linux-arm64-jetson-nano` | CPU | 初代 Nano（JetPack 4）编不了 ggml 的 CUDA：只用 CPU 听写，合成用在线服务 |
| `linux-arm64-vulkan` | Vulkan | 高通 Adreno 等 ARM64 板子（如 Arduino VENTUNO Q），要 ARMv8.2 |

接口和协议见 [docs/voice-protocol.md](docs/voice-protocol.md)。本机开发：

```
pwsh voice/scripts/build-windows.ps1 -Backend cuda      # 或 vulkan / cpu
voice/build-cuda/bin/friday-voice serve --models <模型目录> --port 0
python voice/tools/stream_test.py --url ws://127.0.0.1:端口/v1/realtime/transcribe 一段.wav
python voice/tools/speak_test.py  --url ws://127.0.0.1:端口/v1/realtime/speak "要念的文字"
```
