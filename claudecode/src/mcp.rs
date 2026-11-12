//! `friday-claudecode mcp --url http://127.0.0.1:<端口> --key <对话 key>`：claude 的 `--permission-prompt-tool mcp__friday__approve`
//! 用的 MCP 服务（stdio 上的 JSON-RPC，一行一条）。只有一个工具 approve（参数 {tool_name, input, tool_use_id}）：
//! 收到调用就 POST 给本程序（serve 那个进程）的 /internal/permission，一直等到它回答（可能要等用户很久），
//! 把回答（{"behavior":"allow","updatedInput":…} / {"behavior":"deny","message":…}）作为工具结果的文字交回 claude。
//! 内部口令从环境变量 FRIDAY_CLAUDECODE_TOKEN 拿（写在这段对话的 MCP 配置文件里，不放命令行）。

use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::http;

fn send(out: &Arc<Mutex<std::io::Stdout>>, msg: &Value) {
    let mut o = out.lock().unwrap();
    let _ = writeln!(o, "{msg}");
    let _ = o.flush();
}

fn reply(out: &Arc<Mutex<std::io::Stdout>>, id: &Value, result: Value) {
    send(out, &json!({ "jsonrpc": "2.0", "id": id, "result": result }));
}

fn deny(message: &str) -> Value {
    json!({ "behavior": "deny", "message": message })
}

fn ask(url: &str, token: &str, key: &str, args: &Value) -> Value {
    let body = json!({ "key": key, "tool_name": args["tool_name"], "input": args["input"], "tool_use_id": args["tool_use_id"] });
    match http::post_json(url, "/internal/permission", &[("X-Friday-Internal", token)], &body) {
        Ok((200, v)) if v["behavior"].is_string() => v,
        Ok((status, v)) => deny(&format!("Friday 没有回答这次权限确认（HTTP {status}：{v}），这次没有执行。")),
        Err(e) => deny(&format!("连不上 Friday（{e}），这次没有执行。")),
    }
}

pub fn run(args: &[String]) {
    let mut url = String::new();
    let mut key = String::new();
    let mut i = 0;
    while i < args.len() {
        let next = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--url" => {
                url = next;
                i += 1;
            }
            "--key" => {
                key = next;
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    let token = std::env::var("FRIDAY_CLAUDECODE_TOKEN").unwrap_or_default();
    let out = Arc::new(Mutex::new(std::io::stdout()));
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let Ok(msg) = serde_json::from_str::<Value>(line.trim()) else { continue };
        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        let method = msg["method"].as_str().unwrap_or("");
        match method {
            "initialize" => {
                let version = msg["params"]["protocolVersion"].as_str().unwrap_or("2025-06-18").to_string();
                reply(&out, &id, json!({ "protocolVersion": version, "capabilities": { "tools": {} }, "serverInfo": { "name": "friday", "version": crate::VERSION } }));
            }
            "tools/list" => reply(
                &out,
                &id,
                json!({ "tools": [{
                    "name": "approve",
                    "description": "Friday：在对话里问用户批不批这次工具调用",
                    "inputSchema": {
                        "type": "object",
                        "properties": { "tool_name": { "type": "string" }, "input": { "type": "object" }, "tool_use_id": { "type": "string" } },
                        "required": ["tool_name", "input"],
                    },
                }] }),
            ),
            "tools/call" => {
                // 每个调用一个线程：等用户回答时别的请求（ping 之类）照常回
                let out = out.clone();
                let (url, token, key) = (url.clone(), token.clone(), key.clone());
                std::thread::spawn(move || {
                    let params = &msg["params"];
                    let decision = if params["name"].as_str() == Some("approve") {
                        ask(&url, &token, &key, &params["arguments"])
                    } else {
                        deny("没有这个工具")
                    };
                    reply(&out, &id, json!({ "content": [{ "type": "text", "text": decision.to_string() }], "isError": false }));
                });
            }
            "" => {} // 回应（没有 method）不用理
            _ => {
                if !id.is_null() {
                    reply(&out, &id, json!({}));
                }
            }
        }
    }
}
