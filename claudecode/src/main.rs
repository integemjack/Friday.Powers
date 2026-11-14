//! friday-claudecode：Friday 的 Claude Code 能力（Power）。
//! 把一段对话交给用户自己装的、原样不改的 Claude Code 命令行（`claude`，它自己的提示词、工具、登录和额度），
//! 接成 OpenAI 兼容的流式接口：Friday 只传用户的话，Claude Code 在对话的工作目录里干活，内容实时返回。
//! 本程序不读、不碰 Claude Code 的登录凭据；登录状态只看 `claude auth status` 自己的输出（见 backend.rs）。
//! 宿主协议见 docs/power-spec.md §2：
//!   friday-claudecode serve --host 127.0.0.1 --port 0 --data-dir <目录> [--token <串>] [--parent-stdin]
//! 就绪后 stdout 打一行 {"event":"listening","url":…}；--parent-stdin 时 stdin 关了就退出（拉起的 claude 一起停）。
//! 接口：GET /health、GET /v1/info、GET /v1/models、POST /v1/chat/completions（见 chat.rs）；
//! 内部：POST /internal/permission（mcp 子命令转来的权限确认，见 conv.rs）。
//! 子命令 `mcp`：claude 的 --permission-prompt-tool 用的 MCP 服务（stdio），见 mcp.rs。
//! 测试用：FRIDAY_CLAUDE_BIN 指定 claude；CLAUDE_CONFIG_DIR / ANTHROPIC_BASE_URL 等照常传给 claude。

mod backend;
mod chat;
mod conv;
mod http;
mod mcp;
mod store;

use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

use serde_json::{json, Value};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version") | Some("-v") | Some("version") => println!("friday-claudecode {VERSION}"),
        Some("serve") => serve(&args[1..]),
        Some("mcp") => mcp::run(&args[1..]),
        Some("info") => {
            let data = std::env::temp_dir().join("friday-claudecode");
            let _ = std::fs::create_dir_all(&data);
            backend::init(&data);
            println!("{}", serde_json::to_string_pretty(&info()).unwrap());
        }
        _ => {
            eprintln!("用法：friday-claudecode serve --host 127.0.0.1 --port 0 --data-dir <目录> [--token <串>] [--parent-stdin] | info | --version");
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
    let mut o = Options { host: "127.0.0.1".into(), port: 0, data_dir: std::env::temp_dir().join("friday-claudecode"), token: String::new(), parent_stdin: false };
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
    // mcp 子命令回连本程序：监听在所有地址上时用 127.0.0.1
    let local_host = match o.host.as_str() {
        "" | "0.0.0.0" | "::" | "[::]" => "127.0.0.1".to_string(),
        h if h.contains(':') && !h.starts_with('[') => format!("[{h}]"),
        h => h.to_string(),
    };
    backend::set_server(&format!("http://{local_host}:{}", addr.port()), &backend::random_hex(40));
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
            conv::shutdown();
            std::process::exit(0);
        });
    }
    // 先在后台把 claude 的路径、版本、登录状态、认哪些参数、模型查好（第一个 /v1/models、第一段对话就不用等）
    std::thread::spawn(|| {
        if let Some(bin) = backend::find_binary() {
            let _ = backend::version(&bin);
            let _ = backend::auth(&bin);
            let _ = backend::flags(&bin);
            let _ = backend::models(&bin);
        }
    });
    std::thread::spawn(conv::reaper);
    let token = std::sync::Arc::new(o.token);
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let token = token.clone();
        std::thread::spawn(move || handle(stream, &token));
    }
}

fn info() -> Value {
    let bin = backend::find_binary();
    let version = bin.as_deref().and_then(backend::version);
    let auth = bin.as_deref().and_then(backend::auth);
    json!({
        "id": "claudecode",
        "version": VERSION,
        "capabilities": ["agentChat"],
        "claude": bin.as_ref().map(|b| json!({ "path": b.to_string_lossy(), "version": version })),
        "auth": auth,
        "error": if bin.is_none() { Value::String(backend::NOT_FOUND.into()) } else { Value::Null },
        "conversations": store::count(),
        "running": conv::running(),
        // 正在跑的对话请求（宿主停闲置的 Power 之前看它：Claude Code 一轮可能跑很久）
        "active": chat::ACTIVE.load(std::sync::atomic::Ordering::SeqCst),
        "rateLimit": conv::rate_limit(),
    })
}

/// 模型列表不写死：问 Claude Code 自己（backend::models，initialize 回的 models），它更新了新模型、换了默认这里跟着变。
/// 每个模型：id = --model 的值（default 不传 --model，用 Claude Code 自己的默认）、name、description（Claude Code 的说明）、
/// resolved_model（default 这类别名现在指哪个）、efforts（能选的思考档 = --effort 的值；空 = 不分档，只能开 / 关思考）
fn models() -> Result<Value, (u16, String)> {
    let bin = backend::find_binary().ok_or((503, backend::NOT_FOUND.to_string()))?;
    let version = backend::version(&bin).unwrap_or_default();
    let auth = backend::auth(&bin);
    let logged_in = auth.as_ref().map(|a| a["loggedIn"].as_bool() == Some(true)).unwrap_or(false);
    let plan = auth.as_ref().map(|a| a["authMethod"].as_str() == Some("claude.ai")).unwrap_or(false);
    let (list, stale) = backend::models(&bin).map_err(|e| (502, e))?;
    let data: Vec<Value> = list.iter().filter_map(|m| model_entry(m, plan)).collect();
    let mut detail = backend::detail(&version, auth.as_ref());
    if stale {
        detail.push_str(" · 模型列表是上次取到的");
    }
    Ok(json!({
        "object": "list",
        "data": data,
        "backend": { "detail": detail, "version": version, "loggedIn": logged_in, "path": bin.to_string_lossy(), "stale": stale },
    }))
}

/// Claude Code 报的一个模型 → /v1/models 的一项
fn model_entry(m: &Value, plan: bool) -> Option<Value> {
    let id = m["value"].as_str().map(str::trim).filter(|s| !s.is_empty())?;
    let name = if id == "default" { "默认" } else { m["displayName"].as_str().map(str::trim).filter(|s| !s.is_empty()).unwrap_or(id) };
    let efforts: Vec<&str> = if m["supportsEffort"].as_bool() == Some(true) {
        m["supportedEffortLevels"].as_array().into_iter().flatten().filter_map(Value::as_str).filter(|l| conv::is_word(l)).collect()
    } else {
        Vec::new()
    };
    Some(json!({
        "id": id,
        "object": "model",
        "owned_by": "claudecode",
        "name": name,
        "description": m["description"].as_str().unwrap_or(""),
        "resolved_model": m["resolvedModel"].as_str().unwrap_or(""),
        "efforts": efforts,
        "adaptive_thinking": m["supportsAdaptiveThinking"].as_bool().unwrap_or(false),
        "input_modalities": ["text", "image"],
        "tools": true,
        "free": false,
        "billing": if plan { "plan" } else { "metered" },
    }))
}

fn handle(mut stream: TcpStream, token: &str) {
    let Ok(Some(req)) = http::read_request(&stream) else { return };
    let path = req.path.trim_end_matches('/').to_string();
    // 内部接口：只给本程序拉起的 mcp 子命令用（它带的是内部口令，不是宿主的 token）
    if path == "/internal/permission" {
        let (_, internal) = backend::server();
        if req.method != "POST" || req.header("x-friday-internal") != Some(internal.as_str()) {
            let _ = http::error(&mut stream, 401, "unauthorized", "内部接口", &[]);
            return;
        }
        let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        let answer = conv::ask_permission(&body);
        let _ = http::json(&mut stream, 200, &answer);
        return;
    }
    if !token.is_empty() {
        let ok = req.header("authorization").map(|v| v.trim() == format!("Bearer {token}")).unwrap_or(false);
        if !ok {
            let _ = http::error(&mut stream, 401, "unauthorized", "要带 Authorization: Bearer <token>", &[]);
            return;
        }
    }
    match (req.method.as_str(), path.as_str()) {
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
            Err((status, e)) => {
                let code = if status == 503 { "claude_not_installed" } else { "claude_error" };
                let _ = http::error(&mut stream, status, code, &e, &[]);
            }
        },
        ("POST", "/v1/chat/completions") => chat::handle(&req, &mut stream),
        _ => {
            let _ = http::error(&mut stream, 404, "not_found", &format!("没有 {} {}", req.method, req.path), &[]);
        }
    }
}
