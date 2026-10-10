"""照 Friday 的用法压 friday-llm：主智能体一段对话 + 同时派出的几个子智能体（共用很长的系统提示和工具说明）+ 从主对话分出去的旁路请求
（完成核对、预测下一步），看基数树省下了多少预填。

    python llm/tools/concurrency_test.py --url http://127.0.0.1:端口 --model 组织/仓库:量化档 [--agents 4] [--tools 60] [--token T]

每个请求打一行：会话、提示 token、复用 / 预填、首字时间、调度怎么放的（x_friday.cache.how）；最后合计「提示总量 / 实际预填」。
只用标准库。
"""
import argparse
import json
import random
import threading
import time
import urllib.request


def tools_block(count):
    """一批像 Friday 那样的工具说明（让系统前缀有几千 token）"""
    tools = []
    for i in range(count):
        tools.append({
            "type": "function",
            "function": {
                "name": f"tool_{i:02d}",
                "description": f"第 {i} 个工具：读写工作目录里的文件、运行命令、查资料，参数要按 JSON Schema 给。"
                               f"这一段描述故意写长一点，模拟真实的工具说明，让系统提示有足够多的 token（编号 {i}）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对工作目录的路径"},
                        "content": {"type": "string", "description": "要写入的内容"},
                        "mode": {"type": "string", "enum": ["read", "write", "append"], "description": "读、写还是追加"},
                    },
                    "required": ["path"],
                },
            },
        })
    return tools


SYSTEM = ("你是 Friday，一个在用户电脑上工作的智能体。回答简短，用中文。需要时调用工具；工具结果回来以后接着做，"
          "做完说一句结论。不确定的先查再说，不要编造。") * 8


def chat(url, token, body, results, label, lock):
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url + "/v1/chat/completions", data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
    if token:
        request.add_header("Authorization", "Bearer " + token)
    started = time.time()
    first = None
    text = ""
    usage = {}
    cache = {}
    timings = {}
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"].get("message"))
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if piece and first is None:
                        first = time.time() - started
                    text += delta.get("content") or ""
                if "usage" in chunk:
                    usage = chunk["usage"]
                if "timings" in chunk:
                    timings = chunk["timings"]
                if "x_friday" in chunk:
                    cache = chunk["x_friday"].get("cache", {})
    except Exception as error:  # noqa: BLE001
        with lock:
            print(f"{label:10s} 出错：{error}")
        return
    elapsed = time.time() - started
    with lock:
        results.append({"label": label, "prompt": usage.get("prompt_tokens", 0), "reused": cache.get("reused", timings.get("cache_n", 0)),
                        "prefilled": cache.get("prefilled", timings.get("prompt_n", 0)), "ttft": first or elapsed, "elapsed": elapsed,
                        "how": cache.get("how", "")})
        print(f"{label:10s} 提示 {usage.get('prompt_tokens', 0):6d}  复用 {cache.get('reused', 0):6d}  预填 {cache.get('prefilled', 0):6d}  "
              f"首字 {first or elapsed:6.2f}s  共 {elapsed:6.2f}s  {cache.get('how', '')}  | {text[:30]!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--token", default="")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--tools", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()

    tools = tools_block(args.tools)
    lock = threading.Lock()
    results = []
    base = {"model": args.model, "stream": True, "max_tokens": args.max_tokens, "tools": tools, "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": args.think}}

    def body(messages, session, priority, main=False):
        b = dict(base)
        b["messages"] = messages
        b["x_friday"] = {"session": session, "priority": priority, "main": main}
        return b

    main_history = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "帮我看看这个项目的构建脚本有什么问题。"}]

    print("── 第 1 轮：主智能体（冷启动，系统前缀第一次算）")
    chat(args.url, args.token, body(main_history, "main", 1000, True), results, "主·1", lock)
    main_history.append({"role": "assistant", "content": "我先派几个子智能体分头看。"})
    main_history.append({"role": "user", "content": "好的，分头看吧。"})

    print(f"── 同时派出 {args.agents} 个子智能体（共用系统提示 + 工具说明）")
    threads = []
    for i in range(args.agents):
        task = f"子任务 {i}：检查第 {i} 个模块的 CMakeLists，列出可疑的地方。随机数 {random.randint(0, 99999)}"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
        t = threading.Thread(target=chat, args=(args.url, args.token, body(messages, f"agent-{i}", 1), results, f"子·{i}", lock))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("── 第 2 轮：主智能体 + 同时两个旁路请求（完成核对、预测下一步，从主对话分出去）")
    threads = [
        threading.Thread(target=chat, args=(args.url, args.token, body(main_history, "main", 1000, True), results, "主·2", lock)),
        threading.Thread(target=chat, args=(args.url, args.token,
                                            body(main_history + [{"role": "user", "content": "（核对）上面这件事做完了吗？只答是或否。"}],
                                                 "", 500), results, "核对", lock)),
        threading.Thread(target=chat, args=(args.url, args.token,
                                            body(main_history + [{"role": "user", "content": "（预测）用户下一步最可能说什么？一句话。"}],
                                                 "", 500), results, "预测", lock)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"── 冷启动：换一份系统提示，同时派出 {args.agents} 个子智能体（只该算一次前缀，其余等它算完再分叉）")
    salt = f"（这一批的编号 {random.randint(0, 10**9)}）"
    threads = []
    for i in range(args.agents):
        messages = [{"role": "system", "content": salt + SYSTEM}, {"role": "user", "content": f"冷启动子任务 {i}：说一句话。"}]
        threads.append(threading.Thread(target=chat, args=(args.url, args.token, body(messages, f"cold-{i}", 1), results, f"冷·{i}", lock)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    prompt = sum(r["prompt"] for r in results)
    prefilled = sum(r["prefilled"] for r in results)
    print(f"── 合计 {len(results)} 个请求：提示 {prompt} token，实际预填 {prefilled}（省下 {100 * (1 - prefilled / max(prompt, 1)):.0f}%）")
    try:
        with urllib.request.urlopen(urllib.request.Request(args.url + "/v1/info", headers={"Authorization": "Bearer " + args.token}
                                                           if args.token else {})) as response:
            info = json.loads(response.read())
            print("── 调度统计：", json.dumps((info.get("model") or {}).get("loaded", {}).get("cache", {}), ensure_ascii=False))
    except Exception as error:  # noqa: BLE001
        print("读 /v1/info 失败：", error)


if __name__ == "__main__":
    main()
