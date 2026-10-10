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

## 音色

`voices/*.gguf` 都是用 cosyvoice.cpp 的前端（`cosyvoice-cli --frontend-only`，CosyVoice3 的 speech_tokenizer_v3.onnx + campplus.onnx）
从下面的录音提取的 prompt speech（只含语音 token、声学特征和说话人向量，不含录音本身）：

| 文件 | 音色 | 录音来源 | 许可 |
|---|---|---|---|
| `default.gguf` | 默认女声 | [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 仓库的 `asset/zero_shot_prompt.wav`（「希望你以后能够做的比我还好呦。」） | Apache-2.0 |
| `young-male.gguf` | 青年男声 | [AISHELL-3](https://www.openslr.org/93/) 说话人 SSB0073 的 `SSB00730025`、`SSB00730028` 两句 | Apache-2.0 |
| `calm-male.gguf` | 沉稳男声 | AISHELL-3 说话人 SSB0434 的 `SSB04340029`、`SSB04340066` 两句 | Apache-2.0 |
| `gentle-female.gguf` | 温柔女声 | AISHELL-3 说话人 SSB0005 的 `SSB00050027` | Apache-2.0 |
| `lively-female.gguf` | 活泼女声 | AISHELL-3 说话人 SSB0267 的 `SSB02670001` | Apache-2.0 |

AISHELL-3：Shi, Yao, et al. "AISHELL-3: A Multi-speaker Mandarin TTS Corpus and the Baselines." arXiv:2010.11567（北京希尔贝壳科技，
[Hugging Face 上的官方发布](https://huggingface.co/datasets/AISHELL/AISHELL-3)，Apache-2.0）。
