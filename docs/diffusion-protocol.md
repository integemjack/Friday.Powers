# friday-diffusion 协议

本地生图 / 生视频 Power（stable-diffusion.cpp）。启动、就绪、stdin、令牌照 [power-spec.md](power-spec.md) §2。

```
friday-diffusion serve --host 127.0.0.1 --port 0 --token <T> --parent-stdin [--idle 600] [--manifest power.json]
```

- `--idle`：空闲多少秒释放模型（腾显存给大模型），0 = 不释放；缺省 600。
- 模型家族的缺省参数从可执行文件旁边的 `power.json`（`families`）读。

## 1. 模型家族

一个生图 / 生视频模型除了主文件（扩散模型的 GGUF），还要配套文件（VAE、文本编码器）。`power.json` 的 `families` 列出认得的家族：

| 字段 | 说明 |
|---|---|
| `id` / `title` / `kind` / `detail` | 家族 id、名字、`image` 或 `video`、一句话说明 |
| `match` | 正则（不区分大小写）：仓库 id 能匹配上的就是这一家（Friday 的模型库据此把魔搭 / HF 上的 GGUF 仓库归到生图 / 生视频） |
| `main` | 主文件在 `x_friday.files` 里的键（`diffusion_model`，SD 1.x / SDXL 这种整个模型一个文件的是 `model`） |
| `recommended` | 推荐的仓库 / 文件（`repo`、`file`、`size`、`sha256`） |
| `components` | 配套文件：键是 `x_friday.files` 的键（`vae`、`llm`、`t5xxl`、`clip_l`…），值是 `{repo, file, size, sha256}`（魔搭和 HF 上同名） |
| `defaults` | `image` / `video` 的缺省参数（steps、cfg_scale、sampler、flow_shift、width、height、frames、fps、negative_prompt） |
| `edits` | 能按参考图改图 |

现在有：Z-Image-Turbo、FLUX.2 klein 4B、Qwen-Image（生图），Wan2.2 TI2V 5B（生视频）。

## 2. 接口

| 接口 | 说明 |
|---|---|
| `GET /health` | `{"status":"ok"}` |
| `GET /v1/info` | `{id, version, "stable-diffusion.cpp", capabilities, generator{loaded, family, queued, running, idle_unload_seconds}, system_info}` |
| `POST /v1/jobs` | 提交任务（异步，见 §3）→ `202` 任务状态 |
| `GET /v1/jobs/<id>` | 任务状态：`{id, kind, family, status, progress, step, steps, outputs[{index, mime, bytes, width, height, url}], error, elapsed_ms}` |
| `GET /v1/jobs/<id>/outputs/<n>` | 第 n 个结果的字节（`image/png`、`video/webm`） |
| `POST /v1/jobs/<id>/cancel` | 取消 |
| `POST /v1/images/generations` | OpenAI 兼容的生图（同步）：`{data:[{b64_json}], x_friday:<任务状态>}` |
| `POST /v1/memory/release` | 释放模型（正在跑的做完再放）→ `{released}` |

`status`：`queued` → `loading`（加载模型）→ `running`（采样，`progress` 0…1）→ `encoding` → `done` / `failed` / `cancelled`。
任务一个个做（显卡一次只跑一个）；结果在内存里留最近 32 个任务。

## 3. 任务请求

```json
{
  "kind": "image",
  "prompt": "一只戴着围巾的橘猫坐在窗台上",
  "negative_prompt": "",
  "size": "1024x1024",
  "n": 1,
  "steps": 8, "cfg_scale": 1.0, "seed": -1, "sampler": "euler", "scheduler": "", "flow_shift": 3.0,
  "image": "data:image/png;base64,…",
  "reference_images": ["data:image/png;base64,…"],
  "strength": 0.75,
  "frames": 33, "fps": 16, "end_image": "data:…",
  "x_friday": {
    "family": "z-image-turbo",
    "files": {
      "diffusion_model": "D:/…/z_image_turbo-Q4_K.gguf",
      "vae": "D:/…/ae.safetensors",
      "llm": "D:/…/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    }
  }
}
```

- 参数：请求里给的优先，其次家族的 `defaults`，再其次 stable-diffusion.cpp 的缺省。`size` 和 `width` / `height` 二选一。
- `image`：图生图的起始图（生视频时是起始帧，即图生视频）；`reference_images`：按参考图改图（FLUX.2 klein、Qwen-Image-Edit 这类）。
- 模型上下文按「用到的那几个文件」缓存：同一组文件的任务接着用，换一组就释放旧的、加载新的。显存不够时 stable-diffusion.cpp
  自己把权重分到内存里（auto-fit），VAE 解码放不下时自己分块。
- 视频出 WebM（VP8，libwebp 编码、libwebm 封装）。

实测（RTX 5080）：Z-Image-Turbo Q4_K 768×768 一张 5.5 秒（加载 0.5 秒）；Wan2.2 TI2V 5B Q4_K_M 832×480 17 帧约 65 秒。
