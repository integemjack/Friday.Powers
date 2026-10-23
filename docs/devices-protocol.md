# devices：客户端设备（Power 协议）

`friday-devices` 让 Friday 的智能体用**远程客户端那台电脑上的硬件**：Friday Work 桌面客户端连着的 ESP / Arduino / STM32 开发板（串口）、
摄像头、安卓 / iOS 手机、客户端局域网里的设备（树莓派……）。不用显卡、不要模型，每个系统一种包。

```
客户端那台电脑                         Friday（服务端）                          friday-devices（这个 Power）
WebSerial / getUserMedia / adb …  ⇆  WebRTC 工作通道 ⇆ remote/ClientDevices  ⇆  /v1/bridge（WebSocket）
  device.* 请求（客户端回）           （只中转）                                  本机端口 127.0.0.1:…（rfc2217 / adb / usbmux / TCP）
  friday-serial / friday-tcp 通道                                                /mcp：给智能体的工具（服务名 client）
```

- 客户端和 Friday 之间的部分（`device.*` 反向请求、专用通道、授权）见 Friday 的 `qt/docs/REMOTE.md` §6「客户端设备」
  和客户端仓库的 `docs/方案-客户端设备.md`。
- 这个 Power 不碰任何硬件：打开串口、拍照、连 adb 都是客户端做的（先问拿着设备的人），这里只开本机端口、读固件文件、编排请求。

## 1. 宿主约定

同 [power-spec.md](power-spec.md) §2：`serve --host 127.0.0.1 --port 0 --token T --parent-stdin`，就绪打一行
`{"event":"listening","url":…}`，`GET /health`、`GET /v1/info`（`bridge`：Friday 接没接上、`clients`、`ports`）。能力 id `clientDevices`。
清单 schema 2，平台包是不带加速后缀的 `windows-x64`、`macos-arm64`、`linux-x64`、`linux-arm64`（Friday 2.78 起认：放在加速包后面）。

## 2. 桥：`GET /v1/bridge`（WebSocket，带 token）

Friday 接进来一条（新的顶掉旧的）。文本帧是 JSON，`type` 区分：

| 方向 | type | 内容 |
|---|---|---|
| Friday → | `hello` | 刚接上 |
| Friday → | `clients` | `{clients: [{id, name, kinds}]}`：声明了 clientDevices 的客户端全表，每次有增减都发整张 |
| Friday → | `response` | `{id, result}` 或 `{id, error: {code, message}}`：`request` 的结果（客户端回的，或 Friday 自己的超时 / 断开） |
| Friday → | `notify` | `{client, method, params}`：客户端的 `device.*` 通知 |
| Friday → | `channel.open` / `channel.closed` | `{client, label}`：客户端开 / 关了一条专用通道 |
| Friday → | `channel.pause` / `channel.resume` | `{client, label}`：那条 DataChannel 积压超过 4 MB / 发完了：先别往里写 |
| → Friday | `request` | `{id, client, method, params, timeoutMs}`：转给那个客户端的 `device.*` 请求（只转 `device.*`） |
| → Friday | `channel.close` | `{client, label}`：关一条通道 |

二进制帧是专用通道上的字节，两个方向一样：`[1 字节头长 H][H 字节 "<client>\n<label>"][数据]`。

开会话的请求（`device.serial.open` / `device.tcp.open`）回来时才知道通道的 label，而客户端开通道、发协商字节可能比回复先到：
没认领的通道先把字节攒着（最多 4 MB），认领时一起交出去。

## 3. 本机端口

每个端口只听 `127.0.0.1`；每来一个 TCP 连接，向客户端开一次会话、拿到通道，socket ⇆ 通道原样搬字节，任一边关了另一边跟着关。
开会话失败（客户端那边拒绝、设备拔了）就关掉这个连接，错误记在端口上（`ports` 工具里看得到）。

| 种类 | 每个连接发的请求 | 用法 |
|---|---|---|
| serial | `device.serial.open {deviceId, mode, baudRate}` | `rfc2217://127.0.0.1:端口`（客户端是 RFC 2217 服务器：波特率、DTR/RTS 都带）；`mode: raw` 时是 `socket://` |
| tcp | `device.tcp.open {host, port}` | 客户端只放行本机 / 局域网地址 |
| adb | `device.tcp.open {service: adb, deviceId}` | `ADB_SERVER_SOCKET=tcp:127.0.0.1:端口`；客户端只放行授权的那一台、拒绝 kill / connect |
| usbmux | `device.tcp.open {service: usbmux}` | `USBMUXD_SOCKET_ADDRESS=127.0.0.1:端口` |

## 4. MCP：`POST /mcp`

Streamable HTTP，只回 `application/json`（不开 SSE）；`initialize`、`tools/list`、`tools/call`、`ping`，通知回 202。
Friday 把它当成名叫 `client` 的 MCP 服务（工具是 `client__devices` 这样），信任它（不逐次确认：客户端那头会问拿着设备的人）。

| 工具 | 做什么 |
|---|---|
| `devices` | 列客户端和它们的设备（`device.list`）：`serial:…`、`camera:…`、`android:…`、`ios:…`、`uf2:…` |
| `serial_open` | 开串口端口（rfc2217 / raw），给服务端的程序用 |
| `serial_monitor` | 开串口监视：输出缓存在这个能力里（最近 512 KB），智能体直接读写，不用写脚本（见下） |
| `serial_read` | 取新输出：从上次读到的地方接着读或给 `cursor`；`wait_s` 最多等多久，`until` 等到出现某段文字；`hex` 按十六进制回 |
| `serial_write` | 发数据：`text` 原样、`line` 补换行（默认 `
`）、`hex` 字节 |
| `serial_reset` | 复位：`rts`（ESP：RTS 拉 EN 低 100 ms、DTR 放开 = 正常启动，默认）/ `dtr`（Arduino：DTR 拉一下） |
| `serial_close` | 关掉监视、把口还给客户端 |
| `tunnel_open` | TCP 隧道到客户端那边的 `host:port` |
| `android_attach` | adb 端口（只有一台安卓时不用填 device_id） |
| `ios_attach` | usbmuxd 端口 |
| `ports` / `port_close` | 看 / 关开着的端口；`ports` 也列后台跑着的客户端工具 |
| `flash` | 在客户端本机烧：`esp`（多文件带地址）、`uf2`、`stk500`（.hex / .bin）、`stm32`；固件是这台机器上的文件（读出来 base64 给客户端） |
| `esp_info` | 读 ESP 芯片型号、MAC、flash 大小 |
| `camera_snapshot` | 拍一张，结果是 MCP 图片内容 |
| `run_tool` / `stop_tool` | 在客户端跑白名单里的烧录 / 调试工具（每次都问人）；`background` 时不等结束 |
| `job_wait` | 等后台任务 |

**串口监视**：`serial_monitor` 向客户端开一个 `rfc2217` 串口会话（同 `serial_open` 的端口，同一个口同时只能有一个），通道认领到能力里：
这里是 RFC 2217 的客户端一头——开场协商、`SET-BAUDRATE`、复位用 `SET-CONTROL`（DTR 8/9、RTS 11/12）；收到的数据去掉 Telnet 转义
（`IAC IAC` → 0xFF，夹在里面的协商 / 子协商丢掉）后进缓冲，按会话开始以来的字节数编游标，只留最近 512 KB（读到被挤掉的部分时回里带 `dropped`）。
客户端在这个口上烧录时会话挂起、烧完按原参数恢复，监视不断：「烧录 → `serial_reset` → `serial_read until=…`」就能看新固件起没起来。

Friday 调一个工具最多等 120 秒：烧录、前台跑工具做成后台任务，100 秒内做完直接回结果，否则先回 `job_id` 和进度
（客户端的 `device.esp.progress` / `device.flash.progress` 通知里带 `token` = job id），智能体用 `job_wait` 接着等。

## 5. 测试

`devices/tools/bridge_test.py`：同时扮演 Friday（连桥）和一台客户端（回 `device.*`、开通道、通道上的字节转大写回来），
走一遍 MCP 工具，再真用 TCP 连开出来的端口看两个方向都通。只用标准库：

```
cmake -S devices -B devices/build -G Ninja && cmake --build devices/build
python devices/tools/bridge_test.py devices/build/bin/friday-devices[.exe]
```
