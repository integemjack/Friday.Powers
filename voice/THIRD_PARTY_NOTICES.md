# friday-voice 用到的第三方代码和模型

## 编进可执行文件的代码

| 项目 | 许可 | 用途 |
|---|---|---|
| [ggml](https://github.com/ggml-org/ggml) | MIT | 推理（CPU / CUDA / Vulkan / Metal / OpenCL） |
| [cosyvoice.cpp](https://github.com/Lourdle/cosyvoice.cpp) | MIT | CosyVoice3 合成 |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp)（v1.9.5，只用 src/whisper.cpp） | MIT | 听写定稿（中英混说） |
| [FunASR](https://github.com/modelscope/FunASR)（llama.cpp runtime 的 SenseVoice / FSMN-VAD 实现，作参考移植） | MIT | 听写、断句 |
| [civetweb](https://github.com/civetweb/civetweb) | MIT | HTTP / WebSocket 服务 |
| [PCRE2](https://github.com/PCRE2Project/pcre2) | BSD-3-Clause（带 PCRE2 例外） | 合成分词的正则 |
| [nlohmann/json](https://github.com/nlohmann/json) | MIT | JSON |
| [miniaudio](https://github.com/mackron/miniaudio) | 公有领域 / MIT-0 | 音频解码、重采样 |

各项目的许可全文在 `LICENSES/`。

## 随包附带的库

- Windows CUDA 包里的 `cublas64_*.dll`、`cublasLt64_*.dll` 来自 NVIDIA CUDA Toolkit，按
  [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/) 附录 A 允许再分发的部分随包分发。
- Jetson 包静态链接了 CUDA 运行库和 cuBLAS（同一 EULA）。

## 模型（不在包里，第一次用时下载：Whisper 从 Hugging Face，别的从 ModelScope）

| 模型 | 许可 |
|---|---|
| SenseVoice-Small、FSMN-VAD（FunAudioLLM 的 GGUF） | [FunASR 模型许可](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE) |
| Fun-CosyVoice3-0.5B-2512（Lourdle 转的 GGUF） | Apache-2.0 |
| Whisper large-v3-turbo（OpenAI；whisper.cpp 转的 ggml，q5_0） | [MIT](https://github.com/openai/whisper/blob/main/LICENSE) |

## 默认音色

`voices/default.gguf` 是用 cosyvoice.cpp 的前端，从 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)（Apache-2.0）
仓库的 `asset/zero_shot_prompt.wav`（文字「希望你以后能够做的比我还好呦。」）提取的 prompt speech。
