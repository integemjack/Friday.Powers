# friday-llm 用到的第三方代码

| 组件 | 版本 | 许可证 | 用在哪 |
|---|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp)（含 ggml、common、mtmd、vendor 里的 nlohmann/json、minja、stb） | b11541 | MIT | 推理、聊天模板与解析、看图；`src/engine` 里照 tools/server 改写的部分同样是 MIT（Copyright (c) 2023-2026 The ggml authors） |
| [civetweb](https://github.com/civetweb/civetweb) | 1.16 | MIT | HTTP 服务 |

模型不随包发：用户在 Friday 的模型库里选、从魔搭 / Hugging Face 下载，许可证各看各的模型页。
