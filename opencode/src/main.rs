//! friday-opencode：Friday 的 OpenCode 能力（Power）。
//! 连上本机的 OpenCode（桌面版 / 命令行的后台服务；没有就自己拉起一个 `opencode serve`），把 OpenCode 自己的智能体
//! （它自己的提示词和工具）接成 OpenAI 兼容的流式接口：Friday 只传用户的话，OpenCode 在对话的工作目录里干活，内容实时返回。
//! 宿主协议见 docs/power-spec.md §2：
//!   friday-opencode serve --host 127.0.0.1 --port 0 --data-dir <目录> [--token <串>] [--parent-stdin]
//! 就绪后 stdout 打一行 {"event":"listening","url":…}；--parent-stdin 时 stdin 关了就退出（拉起的 opencode 一起停）。
//! 接口：GET /health、GET /v1/info、GET /v1/models、POST /v1/chat/completions（见 chat.rs）。
//! 测试用：FRIDAY_OPENCODE_URL + FRIDAY_OPENCODE_PASSWORD 指定要连的 opencode 服务。

mod backend;
mod chat;
mod http;
mod oc;
mod store;

use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

use serde_json::{json, Value};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version") | Some("-v") | Some("version") => println!("friday-opencode {VERSION}"),
        Some("serve") => serve(&args[1..]),
        Some("info") => {
            let data = std::env::temp_dir().join("friday-opencode");
            let _ = std::fs::create_dir_all(&data);
            backend::init(&data);
            println!("{}", serde_json::to_string_pretty(&info()).unwrap());
        }
        _ => {
            eprintln!("用法：friday-opencode serve --host 127.0.0.1 --port 0 --data-dir <目录> [--token <串>] [--parent-stdin] | info | --version");
            std::process::exit(2);
        }
    }
}

struct Options {
    host: String,
    port: u16,
    data_dir: PathBuf,
    token: String,
    parent_stdin: bool,
}

fn options(args: &[String]) -> Options {
    let mut o = Options { host: "127.0.0.1".into(), port: 0, data_dir: std::env::temp_dir().join("friday-opencode"), token: String::new(), parent_stdin: false };
    let mut i = 0;
    while i < args.len() {
        let next = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--host" => {
                o.host = next;
                i += 1;
            }
            "--port" => {
                o.port = next.parse().unwrap_or(0);
                i += 1;
            }
            "--data-dir" => {
                o.data_dir = PathBuf::from(next);
                i += 1;
            }
            "--token" => {
                o.token = next;
                i += 1;
            }
            "--parent-stdin" => o.parent_stdin = true,
            // 宿主协议里的其它参数（--device 等）这里用不到
            "--device" | "--models" => i += 1,
            _ => {}
        }
        i += 1;
    }
    o
}

fn serve(args: &[String]) {
    let o = options(args);
    let _ = std::fs::create_dir_all(&o.data_dir);
    backend::init(&o.data_dir);
    store::init(&o.data_dir);
    let listener = match TcpListener::bind((o.host.as_str(), o.port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("监听 {}:{} 失败：{e}", o.host, o.port);
            std::process::exit(1);
        }
    };
    let addr = listener.local_addr().unwrap();
    println!("{}", json!({ "event": "listening", "url": format!("http://{}:{}", o.host, addr.port()) }));
    if o.parent_stdin {
        std::thread::spawn(|| {
            let mut buf = [0u8; 256];
            let mut stdin = std::io::stdin();
            loop {
                match stdin.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
            }
            backend::shutdown();
            std::process::exit(0);
        });
    }
    // 先在后台把 opencode 找好 / 拉起来（第一个请求就不用等）
    std::thread::spawn(|| {
        if let Err(e) = backend::current() {
            eprintln!("{e}");
        }
    });
    let token = std::sync::Arc::new(o.token);
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let token = token.clone();
        std::thread::spawn(move || handle(stream, &token));
    }
}

fn info() -> Value {
    let backend = backend::current();
    json!({
        "id": "opencode",
        "version": VERSION,
        "capabilities": ["agentChat"],
        "backend": match &backend { Ok(b) => b.describe(), Err(_) => Value::Null },
        "error": match &backend { Ok(_) => Value::Null, Err(e) => Value::String(e.clone()) },
        "binary": backend::find_binary().map(|p| p.to_string_lossy().to_string()),
        "conversations": store::count(),
        // 正在跑的对话请求（宿主停闲置的 Power 之前看它：OpenCode 一轮可能跑很久）
        "active": chat::ACTIVE.load(std::sync::atomic::Ordering::SeqCst),
    })
}

fn models() -> Result<Value, String> {
    let be = backend::current()?;
    let value = match oc::ok(oc::get(&be.conn, "/api/model", None), "列模型") {
        Ok(v) => v,
        Err(e) if e.starts_with(oc::CONNECT_ERROR) => {
            backend::reset();
            let be = backend::current()?;
            oc::ok(oc::get(&be.conn, "/api/model", None), "列模型")?
        }
        Err(e) => return Err(e),
    };
    let mut data = Vec::new();
    for m in value["data"].as_array().into_iter().flatten() {
        if m["enabled"].as_bool() == Some(false) {
            continue;
        }
        let provider = m["providerID"].as_str().unwrap_or("");
        let id = m["id"].as_str().unwrap_or("");
        if provider.is_empty() || id.is_empty() {
            continue;
        }
        let cost = m["cost"].as_array().and_then(|c| c.first());
        let free = match cost {
            Some(c) => c["input"].as_f64().unwrap_or(0.0) == 0.0 && c["output"].as_f64().unwrap_or(0.0) == 0.0,
            None => false,
        };
        data.push(json!({
            "id": format!("{provider}/{id}"),
            "object": "model",
            "owned_by": provider,
            "name": m["name"].as_str().unwrap_or(id),
            "context_length": m["limit"]["context"],
            "max_output_tokens": m["limit"]["output"],
            "input_modalities": m["capabilities"]["input"],
            "tools": m["capabilities"]["tools"],
            "free": free,
            "status": m["status"],
            "released": m["time"]["released"],
        }));
    }
    Ok(json!({ "object": "list", "data": data, "backend": be.describe() }))
}

fn handle(mut stream: TcpStream, token: &str) {
    let Ok(Some(req)) = http::read_request(&stream) else { return };
    if !token.is_empty() {
        let ok = req.header("authorization").map(|v| v.trim() == format!("Bearer {token}")).unwrap_or(false);
        if !ok {
            let _ = http::error(&mut stream, 401, "unauthorized", "要带 Authorization: Bearer <token>", &[]);
            return;
        }
    }
    let path = req.path.trim_end_matches('/');
    match (req.method.as_str(), path) {
        ("GET", "/health") => {
            let _ = http::json(&mut stream, 200, &json!({ "status": "ok" }));
        }
        ("GET", "/v1/info") => {
            let _ = http::json(&mut stream, 200, &info());
        }
        ("GET", "/v1/models") => match models() {
            Ok(v) => {
                let _ = http::json(&mut stream, 200, &v);
            }
            Err(e) => {
                let _ = http::error(&mut stream, 503, "opencode_unavailable", &e, &[]);
            }
        },
        ("POST", "/v1/chat/completions") => chat::handle(&req, &mut stream),
        _ => {
            let _ = http::error(&mut stream, 404, "not_found", &format!("没有 {} {}", req.method, req.path), &[]);
        }
    }
}
