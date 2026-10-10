# 音色

`*.gguf` 是 CosyVoice3 的 prompt speech（参考录音的语音 token、声学特征和说话人向量），文件名就是音色名
（`ready.voices`、`start.voice` 用的就是它）。随包放在 `friday-voice` 旁边的 `voices/`；来源和许可见 `../../THIRD_PARTY_NOTICES.md`。
Friday 的设置 › 助手「音色」按名字认（`default` 默认女声、`young-male` 青年男声、`calm-male` 沉稳男声、`gentle-female` 温柔女声、
`lively-female` 活泼女声），不认识的名字原样列出来。

## 做一个新音色

1. 准备 5–10 秒干净的人声（单声道 WAV；太长每次合成都要多处理、第一声出来更晚），和它一字不差的文字稿。
   录音要有授权：自己的声音，或者许可允许再分发的语料（AISHELL-3 是 Apache-2.0）。
2. 编带 ONNX 前端的 `cosyvoice-cli`（friday-voice 自己只编 cosyvoice.cpp 的核心、不带前端）：
   用 friday-voice 构建目录里 `_deps/cosyvoice-src` 那份源码（`vendor/pcre2` 换成 `_deps/pcre2-src`），
   `cmake -G Ninja -DGGML_SOURCE_DIR=<_deps/ggml-src> -DORT_PREBUILT_DIR=<onnxruntime 预编译包> -DCOSYVOICE_NO_ICU=ON
   -DCOSYVOICE_CLI_NO_PLAYBACK=ON -DCOSYVOICE_SERVER_NO_WEBUI=ON`，只编 `cosyvoice-cli` 这个目标。
3. 提取：

   ```
   cosyvoice-cli --frontend-only --speech-tokenizer speech_tokenizer_v3.onnx --campplus campplus.onnx \
       --prompt-audio ref.wav --prompt-text "文字稿" --prompt-speech-output <音色名>.gguf
   ```

   两个 ONNX 在 Fun-CosyVoice3-0.5B-2512 的模型仓库里（`campplus.onnx` 也是声纹用的那个）。文字稿写原文就行，
   `You are a helpful assistant.<|endofprompt|>` 这类前缀合成时库自己加。
4. 验：`friday-voice speak --tts-model CosyVoice3*.gguf --voices <目录> --voice <音色名> --text … --output out.wav` 念一句，
   `friday-voice transcribe --no-itn out.wav` 转写回来看有没有念错，`friday-voice speaker ref.wav out.wav` 看像不像
   （同一个人一般 0.7 以上；和别的音色最好低于 0.6）。路径别带中文（Windows 上命令行参数按本地代码页传）。
