"""实时听写的测试客户端（开发用，不进发布包）。

按真实说话的节奏把一段 WAV 发给 /v1/realtime/transcribe，打印收到的事件和延迟：
    python tools/stream_test.py --url ws://127.0.0.1:PORT/v1/realtime/transcribe test.wav
WAV 要 16 位单声道（任意采样率；不是 16 kHz 时在 start 里告诉服务端）。
"""
import argparse
import asyncio
import json
import time
import wave

import websockets


async def run(url, path, chunk_ms, speed, language):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, "要 16 位单声道 WAV"
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    total_ms = len(pcm) / 2 / rate * 1000
    async with websockets.connect(url, max_size=None) as ws:
        print("<", await ws.recv())
        await ws.send(json.dumps({"type": "start", "sample_rate": rate, "language": language}))
        t0 = time.perf_counter()
        fed_ms = 0.0
        finals = []

        async def receiver():
            async for message in ws:
                event = json.loads(message)
                now_ms = (time.perf_counter() - t0) * 1000 * speed
                kind = event["type"]
                if kind == "final":
                    # 说完（end_ms，音频时间）到收到定稿
                    lag = now_ms - event["end_ms"]
                    finals.append((event, lag))
                    print(f"[{now_ms/1000:6.2f}s] FINAL #{event['segment']} {event['start_ms']/1000:.2f}-{event['end_ms']/1000:.2f}s "
                          f"lang={event['language']} 推理={event['infer_ms']}ms 判定说完后={event['latency_ms']}ms "
                          f"音频说完到收到={lag:.0f}ms\n           {event['text']}")
                elif kind == "partial":
                    print(f"[{now_ms/1000:6.2f}s] partial #{event['segment']}: {event['text']}")
                else:
                    print(f"[{now_ms/1000:6.2f}s] {kind} {json.dumps({k: v for k, v in event.items() if k != 'type'}, ensure_ascii=False)}")
                if kind in ("final", "speech_end") and fed_ms >= total_ms and event.get("segment") == last_segment[0]:
                    return

        last_segment = [None]
        task = asyncio.create_task(receiver())
        step = int(rate * chunk_ms / 1000) * 2
        for offset in range(0, len(pcm), step):
            await ws.send(pcm[offset:offset + step])
            fed_ms = min(total_ms, (offset + step) / 2 / rate * 1000)
            target = t0 + fed_ms / 1000 / speed
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(task, timeout=5)
        except asyncio.TimeoutError:
            task.cancel()
        if finals:
            lags = [lag for _, lag in finals]
            print(f"\n{len(finals)} 句；音频说完到收到定稿：平均 {sum(lags)/len(lags):.0f} ms，最大 {max(lags):.0f} ms"
                  f"（含判定说完要等的静音 end_silence_ms）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("--url", required=True)
    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--speed", type=float, default=1.0, help="大于 1 时比实时快（压测）")
    parser.add_argument("--language", default="auto")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.wav, args.chunk_ms, args.speed, args.language))


if __name__ == "__main__":
    main()
