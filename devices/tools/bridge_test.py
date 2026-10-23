"""friday-devices 的端到端测试：这个脚本同时扮演 Friday（连 /v1/bridge）和一台客户端（回 device.*、开专用通道），
再像 Friday 的 MCP 客户端那样调工具，最后真用 TCP 连开出来的本机端口，看字节两个方向都通。

    python devices/tools/bridge_test.py devices/build/bin/friday-devices[.exe]

只用标准库（WebSocket 客户端自己写了一个最小的）。全部通过时最后一行打印 OK，退出码 0。
"""
import base64
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

TOKEN = "t0ken"


class WebSocket:
    """最小的 WebSocket 客户端（RFC 6455）：文本 / 二进制帧，客户端掩码，收分片拼整条"""

    def __init__(self, host, port, path):
        self.sock = socket.create_connection((host, port))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(1)
            if not chunk:
                raise RuntimeError("握手时断了")
            head += chunk
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise RuntimeError("握手失败：" + head.decode(errors="replace"))
        self.lock = threading.Lock()

    def send(self, data, binary=False):
        payload = data if isinstance(data, bytes) else data.encode()
        header = bytearray([0x80 | (0x2 if binary else 0x1)])
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = os.urandom(4)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        with self.lock:
            self.sock.sendall(bytes(header) + masked)

    def _exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise EOFError
            buf += chunk
        return buf

    def receive(self):
        """返回 (opcode, payload)：1 文本、2 二进制；拼好分片、跳过 ping / pong"""
        message, opcode = b"", None
        while True:
            b1, b2 = self._exact(2)
            fin, op = b1 & 0x80, b1 & 0x0F
            n = b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._exact(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._exact(8))[0]
            payload = self._exact(n)
            if op == 0x8:
                raise EOFError
            if op in (0x9, 0xA):
                continue
            if op != 0:
                opcode = op
            message += payload
            if fin:
                return opcode, message

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class FakeFridayAndClient:
    """桥那一头：Friday + 一台客户端（id 7）。专用通道上收到什么就转成大写回过去"""

    CLIENT = "7"

    def __init__(self, port):
        self.ws = WebSocket("127.0.0.1", port, f"/v1/bridge?token={TOKEN}")
        self.channel_closes = []
        self.requests = []
        self.ws.send(json.dumps({"type": "hello"}))
        self.ws.send(json.dumps({"type": "clients", "clients": [
            {"id": self.CLIENT, "name": "测试客户端 0.5.85（windows）", "kinds": ["serial", "tcp", "camera"]}]}))
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def respond(self, request, result=None, error=None):
        message = {"type": "response", "id": request["id"]}
        if error:
            message["error"] = error
        else:
            message["result"] = result
        self.ws.send(json.dumps(message))

    def open_channel(self, label):
        self.ws.send(json.dumps({"type": "channel.open", "client": self.CLIENT, "label": label}))

    def loop(self):
        try:
            while True:
                opcode, payload = self.ws.receive()
                if opcode == 2:
                    h = payload[0]
                    key, data = payload[1:1 + h], payload[1 + h:]
                    self.ws.send(bytes([h]) + key + data.upper(), binary=True)
                    continue
                message = json.loads(payload)
                if message["type"] == "channel.close":
                    self.channel_closes.append(message["label"])
                    continue
                if message["type"] != "request":
                    continue
                self.requests.append(message)
                threading.Thread(target=self.handle, args=(message,), daemon=True).start()
        except (EOFError, OSError):
            pass

    def handle(self, request):
        method, params = request["method"], request["params"]
        if request["client"] != self.CLIENT:
            self.respond(request, error={"code": 21, "message": "没有这个客户端"})
        elif method == "device.list":
            self.respond(request, {"devices": [
                {"id": "serial:4292:60000:0", "kind": "serial", "label": "COM9 · CP210x", "espHint": True},
                {"id": "android:R5CT", "kind": "android", "label": "Pixel 8（R5CT）"},
                {"id": "camera:abc", "kind": "camera", "label": "测试摄像头"}]})
        elif method == "device.serial.open":
            assert params["deviceId"] == "serial:4292:60000:0" and params["mode"] == "rfc2217", params
            label = "friday-serial:s%d" % request["id"]
            self.open_channel(label)
            # 客户端开通道后先发协商字节，回复后到：这时还没认领，要攒着
            key = f"{self.CLIENT}\n{label}".encode()
            self.ws.send(bytes([len(key)]) + key + b"\xff\xfb\x2c", binary=True)
            time.sleep(0.2)
            self.respond(request, {"sessionId": "s1", "channel": label})
        elif method == "device.tcp.open":
            if params.get("service") == "adb":
                assert params["deviceId"] == "android:R5CT", params
            label = "friday-tcp:t%d" % request["id"]
            self.open_channel(label)
            self.respond(request, {"sessionId": "t1", "channel": label})
        elif method == "device.esp.flash":
            assert params["files"][0]["address"] == "0x10000", params
            assert base64.b64decode(params["files"][0]["data"]) == b"firmware-bytes", params
            self.ws.send(json.dumps({"type": "notify", "client": self.CLIENT, "method": "device.esp.progress",
                                     "params": {"token": params["progressToken"], "fileIndex": 0, "written": 5, "total": 10}}))
            time.sleep(0.3)
            self.respond(request, {"chip": "ESP32-S3", "files": [{"address": 65536, "size": 14, "md5": "x"}]})
        elif method == "device.camera.snapshot":
            self.respond(request, {"mime": "image/jpeg", "data": base64.b64encode(b"\xff\xd8\xff\xd9").decode(),
                                   "width": 2, "height": 1, "label": "测试摄像头", "takenAt": "2026-10-23T00:00:00Z"})
        elif method == "device.esp.info":
            self.respond(request, error={"code": 20, "message": "客户端用户没有允许"})
        else:
            self.respond(request, error={"code": -32601, "message": "不认识 " + method})


def mcp(port, method, params=None, rid=[0]):
    rid[0] += 1
    body = json.dumps({"jsonrpc": "2.0", "id": rid[0], "method": method, "params": params or {}}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/mcp", body, {
        "Content-Type": "application/json", "Accept": "application/json, text/event-stream", "Authorization": "Bearer " + TOKEN})
    with urllib.request.urlopen(request, timeout=130) as response:
        return json.loads(response.read())


def call(port, name, arguments=None):
    reply = mcp(port, "tools/call", {"name": name, "arguments": arguments or {}})
    return reply["result"]


def text_of(result):
    return "\n".join(c["text"] for c in result["content"] if c["type"] == "text")


def json_tail(text):
    return json.loads(text[text.index("{"):])


def main():
    exe = sys.argv[1]
    proc = subprocess.Popen([exe, "serve", "--port", "0", "--token", TOKEN, "--parent-stdin"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
    try:
        listening = json.loads(proc.stdout.readline())
        assert listening["event"] == "listening", listening
        port = int(listening["url"].rsplit(":", 1)[1])
        assert json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health").read())["status"] == "ok"

        # 桥没接上时：工具回错误
        result = call(port, "devices")
        assert result.get("isError") and "还没接上" in text_of(result), result

        bridge = FakeFridayAndClient(port)
        time.sleep(0.3)
        init = mcp(port, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}})
        assert init["result"]["serverInfo"]["name"] == "friday-devices", init
        names = [t["name"] for t in mcp(port, "tools/list")["result"]["tools"]]
        for expected in ("devices", "serial_open", "tunnel_open", "android_attach", "ios_attach", "flash", "camera_snapshot",
                         "run_tool", "job_wait", "ports", "port_close"):
            assert expected in names, names

        listed = json.loads(text_of(call(port, "devices")))
        assert listed[0]["devices"][0]["label"] == "COM9 · CP210x", listed

        # 串口：开端口 → TCP 连上 → 先收到攒着的协商字节 → 发 hello 收 HELLO → 断开 → 桥收到 channel.close
        opened = text_of(call(port, "serial_open", {"device_id": "serial:4292:60000:0"}))
        assert "rfc2217://127.0.0.1:" in opened and "COM9 · CP210x" in opened, opened
        serial_port = json_tail(opened)["port"]
        conn = socket.create_connection(("127.0.0.1", serial_port))
        conn.settimeout(10)
        assert conn.recv(3) == b"\xff\xfb\x2c"
        conn.sendall(b"hello")
        got = b""
        while len(got) < 5:
            got += conn.recv(5 - len(got))
        assert got == b"HELLO", got
        conn.close()
        deadline = time.time() + 5
        while not bridge.channel_closes and time.time() < deadline:
            time.sleep(0.05)
        assert bridge.channel_closes and bridge.channel_closes[0].startswith("friday-serial:"), bridge.channel_closes

        # adb：只有一台安卓时不用填 device_id
        attached = text_of(call(port, "android_attach"))
        assert "ADB_SERVER_SOCKET=tcp:127.0.0.1:" in attached and "Pixel 8" in attached, attached
        adb_port = json_tail(attached)["port"]
        conn = socket.create_connection(("127.0.0.1", adb_port))
        conn.settimeout(10)
        conn.sendall(b"000chost:version")
        assert conn.recv(100) == b"000CHOST:VERSION"
        conn.close()

        # 烧录：读本机文件、base64、带进度令牌；做完直接回结果
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"firmware-bytes")
        flashed = call(port, "flash", {"device_id": "serial:4292:60000:0", "files": [{"path": f.name, "address": "0x10000"}]})
        assert not flashed.get("isError") and "ESP32-S3" in text_of(flashed), flashed
        os.unlink(f.name)
        missing = call(port, "flash", {"device_id": "serial:4292:60000:0", "files": [{"path": f.name + ".nope", "address": 0}]})
        assert missing.get("isError") and "读不了文件" in text_of(missing), missing

        # 客户端拒绝：错误码带回来
        denied = call(port, "esp_info", {"device_id": "serial:4292:60000:0"})
        assert denied.get("isError") and "错误码 20" in text_of(denied), denied

        # 摄像头：MCP 的图片内容
        shot = call(port, "camera_snapshot")
        images = [c for c in shot["content"] if c["type"] == "image"]
        assert images and images[0]["mimeType"] == "image/jpeg", shot

        # 端口表、关端口
        ports = json.loads(text_of(call(port, "ports")))["ports"]
        assert {p["kind"] for p in ports} == {"serial", "adb"}, ports
        for p in ports:
            assert "已关" in text_of(call(port, "port_close", {"id": p["id"]}))
        assert json.loads(text_of(call(port, "ports")))["ports"] == []

        # 桥断了：再调工具回错误
        bridge.ws.close()
        time.sleep(0.5)
        result = call(port, "devices")
        assert result.get("isError"), result
        print("OK")
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
