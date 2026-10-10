# friday-diffusion 用到的第三方代码

| 组件 | 版本 | 许可证 | 用在哪 |
|---|---|---|---|
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)（含它的 ggml 分支、examples/common 的 media_io、stb、oniguruma、utf8proc、darts-clone、miniz） | master-951-f89d9b1 | MIT（各组件见上游 thirdparty） | 生图、生视频、读写图片 |
| [libwebp](https://github.com/webmproject/libwebp) | 0c9546f | BSD-3-Clause | 视频帧编码（VP8） |
| [libwebm](https://github.com/webmproject/libwebm) | 5bf1226 | BSD-3-Clause | WebM 封装 |
| [civetweb](https://github.com/civetweb/civetweb) | 1.16 | MIT | HTTP 服务 |

模型不随包发：用户在 Friday 的模型库里选、从魔搭 / Hugging Face 下载，许可证各看各的模型页。
