"""边写边读的测试客户端（开发用，不进发布包）。

模拟大模型按 token 吐字（每 --delta-ms 毫秒吐 --delta-chars 个字），发给 /v1/realtime/speak，
打印事件、第一声延迟，把收到的音频存成 wav：
    python tools/speak_test.py --url ws://127.0.0.1:PORT/v1/realtime/speak --out out.wav "要念的文字"
--cancel-after 秒：到时间发 cancel，看打断有多快。
"""
import argparse
import asyncio
import json
import time
import wave

import websockets


async def run(args):
    async with websockets.connect(args.url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        print("<", ready)
        rate = args.sample_rate or ready["sample_rate"]
        await ws.send(json.dumps({"type": "start", "voice": args.voice, "speed": args.speed, "sample_rate": rate,
                                  "instruction": args.instruction}))
        audio = bytearray()
        t0 = time.perf_counter()
        first_audio = None
        cancel_sent = None
        done = asyncio.Event()

        def now():
            return (time.perf_counter() - t0) * 1000

        async def receiver():
            nonlocal first_audio
            async for message in ws:
                if isinstance(message, bytes):
                    if first_audio is None:
                        first_audio = now()
                        print(f"[{first_audio:7.0f}ms] 第一块音频（{len(message)} 字节）")
                    audio.extend(message)
                    continue
                event = json.loads(message)
                print(f"[{now():7.0f}ms] {event['type']} {json.dumps({k: v for k, v in event.items() if k != 'type'}, ensure_ascii=False)}")
                if event["type"] in ("done", "cancelled"):
                    if event["type"] == "cancelled" and cancel_sent is not None:
                        print(f"           打断用时 {now() - cancel_sent:.0f}ms")
                    done.set()
                    return

        task = asyncio.create_task(receiver())
        text = args.text
        for i in range(0, len(text), args.delta_chars):
            await ws.send(json.dumps({"type": "text", "text": text[i:i + args.delta_chars]}))
            await asyncio.sleep(args.delta_ms / 1000)
            if args.cancel_after and now() >= args.cancel_after * 1000 and cancel_sent is None:
                break
        if args.cancel_after:
            while now() < args.cancel_after * 1000:
                await asyncio.sleep(0.01)
            cancel_sent = now()
            await ws.send(json.dumps({"type": "cancel"}))
        else:
            await ws.send(json.dumps({"type": "flush"}))
        print(f"[{now():7.0f}ms] 文字发完")
        try:
            await asyncio.wait_for(done.wait(), timeout=120)
        except asyncio.TimeoutError:
            print("等 done 超时")
        task.cancel()
        seconds = len(audio) / 2 / rate
        print(f"\n音频 {seconds:.2f} 秒；第一声 {first_audio or -1:.0f}ms；总用时 {now():.0f}ms")
        if args.out:
            with wave.open(args.out, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(bytes(audio))
            print("已存", args.out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--url", required=True)
    parser.add_argument("--out")
    parser.add_argument("--voice", default="")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=0)
    parser.add_argument("--delta-chars", type=int, default=2)
    parser.add_argument("--delta-ms", type=int, default=30)
    parser.add_argument("--cancel-after", type=float, default=0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
