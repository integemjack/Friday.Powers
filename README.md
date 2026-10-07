# Friday.Powers

Friday 的**按需能力**（Powers）。

Friday 主程序只接在线模型、保持小巧；要用到本机 GPU 的重能力（实时语音识别、流式语音合成……）做成一个个独立的 Power，
放在这个仓库里，由 Friday **按需下载、安装、启动，随时热加载 / 卸载**，不打进主程序。

## 一个 Power 是什么

- 一个目录 `powers/<id>/`：清单 `power.json` + 代码 + 安装脚本。
- 装好后是一个**常驻的本地服务**（独立进程，自己管理 Python 环境、模型和显存），Friday 按需拉起、闲置时停掉。
- 通过 HTTP / WebSocket 向 Friday 提供能力：既有一次性的（听录音、说话，OpenAI 兼容接口），也有实时的（边说边识别、边写边读）。
- 发布：每个 Power 打成 zip 挂在本仓库的 Release 上，`index.json` 列出所有 Power、版本、下载地址和校验值，Friday 从这里取。

## 第一个 Power：`voice`（本地 GPU 实时语音）

- **实时听写**：边说边出字幕，一句说完马上给出带标点的定稿（FunASR：流式 VAD + Paraformer 流式识别 + 句末精修）。
- **流式合成**：文字边来边合成、边出声（CosyVoice2）。
- 用在 Friday 的：桌面端语音模式（麦克风 → 实时字幕 → 助手听到一句就开始想 → 回复边写边读）、
  助手的 SChat 通话、会议 / 听课的连续听写 + 逐句分析。

> 设计和协议正在定，见 `docs/`（随后补上）。
