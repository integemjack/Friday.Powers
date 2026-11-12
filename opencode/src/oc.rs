//! opencode V2 服务的客户端（/api/...，Basic opencode:<密码>）。所有请求都带 location[directory]：
//! opencode 按目录分实例（项目配置、AGENTS.md、会话归属都按它），Friday 一段对话的工作目录就是这里的目录。

use std::io::{BufRead, BufReader, Read};
use std::sync::OnceLock;
use std::time::Duration;

use serde_json::Value;

#[derive(Clone, Debug)]
pub struct Conn {
    pub base: String,
    /// Authorization 头的值（Basic …）
    pub auth: String,
}

/// 连不上（不是 HTTP 错误）：后端可能换了端口 / 退出了，调用方据此重新找
pub const CONNECT_ERROR: &str = "连不上 opencode";

fn agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .http_status_as_error(false)
            .proxy(None)
            .timeout_global(Some(Duration::from_secs(120)))
            .build()
            .into()
    })
}

fn stream_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .http_status_as_error(false)
            .proxy(None)
            .timeout_connect(Some(Duration::from_secs(10)))
            .build()
            .into()
    })
}

pub fn basic_auth(password: &str) -> String {
    use base64::Engine;
    format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("opencode:{password}")))
}

fn pct(text: &str) -> String {
    let mut out = String::new();
    for b in text.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

pub fn url(conn: &Conn, path: &str, dir: Option<&str>) -> String {
    let mut u = format!("{}{}", conn.base.trim_end_matches('/'), path);
    if let Some(d) = dir {
        u.push(if u.contains('?') { '&' } else { '?' });
        u.push_str("location%5Bdirectory%5D=");
        u.push_str(&pct(d));
    }
    u
}

fn finish(result: Result<ureq::http::Response<ureq::Body>, ureq::Error>) -> Result<(u16, Value), String> {
    match result {
        Ok(mut resp) => {
            let status = resp.status().as_u16();
            let text = resp.body_mut().with_config().limit(64 * 1024 * 1024).read_to_string().unwrap_or_default();
            let value = if text.trim().is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap_or(Value::String(text)) };
            Ok((status, value))
        }
        Err(e) => Err(format!("{CONNECT_ERROR}：{e}")),
    }
}

pub fn get(conn: &Conn, path: &str, dir: Option<&str>) -> Result<(u16, Value), String> {
    finish(agent().get(&url(conn, path, dir)).header("Authorization", &conn.auth).call())
}

pub fn get_quick(conn: &Conn, path: &str) -> Result<(u16, Value), String> {
    finish(
        agent()
            .get(&url(conn, path, None))
            .header("Authorization", &conn.auth)
            .config()
            .timeout_global(Some(Duration::from_secs(3)))
            .build()
            .call(),
    )
}

pub fn post(conn: &Conn, path: &str, dir: Option<&str>, body: &Value) -> Result<(u16, Value), String> {
    finish(
        agent()
            .post(&url(conn, path, dir))
            .header("Authorization", &conn.auth)
            .header("Content-Type", "application/json")
            .send(body.to_string()),
    )
}

pub fn post_patch(conn: &Conn, path: &str, dir: Option<&str>, body: &Value) -> Result<(u16, Value), String> {
    finish(
        agent()
            .patch(&url(conn, path, dir))
            .header("Authorization", &conn.auth)
            .header("Content-Type", "application/json")
            .send(body.to_string()),
    )
}

pub fn delete(conn: &Conn, path: &str, dir: Option<&str>) -> Result<(u16, Value), String> {
    finish(agent().delete(&url(conn, path, dir)).header("Authorization", &conn.auth).call())
}

/// 2xx 才算成功；否则把 opencode 的报错拿出来
pub fn ok(result: Result<(u16, Value), String>, what: &str) -> Result<Value, String> {
    let (status, value) = result?;
    if (200..300).contains(&status) {
        return Ok(value);
    }
    let message = value
        .pointer("/message")
        .or_else(|| value.pointer("/error/message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string());
    Err(format!("{what}失败（HTTP {status}）：{message}"))
}

/// /api/event 的 SSE：一行行读 data:，阻塞
pub struct Events {
    reader: BufReader<Box<dyn Read + Send + Sync>>,
}

impl Events {
    pub fn open(conn: &Conn, dir: Option<&str>) -> Result<Events, String> {
        let resp = stream_agent()
            .get(&url(conn, "/api/event", dir))
            .header("Authorization", &conn.auth)
            .header("Accept", "text/event-stream")
            .call()
            .map_err(|e| format!("{CONNECT_ERROR}：{e}"))?;
        if resp.status().as_u16() != 200 {
            return Err(format!("订阅 opencode 事件失败（HTTP {}）", resp.status().as_u16()));
        }
        let reader: Box<dyn Read + Send + Sync> = Box::new(resp.into_body().into_reader());
        Ok(Events { reader: BufReader::new(reader) })
    }

    /// 下一条事件；连接断了返回 None
    pub fn next(&mut self) -> Option<Value> {
        let mut line = String::new();
        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) | Err(_) => return None,
                Ok(_) => {}
            }
            let t = line.trim();
            if let Some(data) = t.strip_prefix("data:") {
                if let Ok(v) = serde_json::from_str::<Value>(data.trim()) {
                    return Some(v);
                }
            }
        }
    }
}

// ---- 事件总线：每个（服务, 目录）一条长连接，按 sessionID 分给各个请求 ----
// opencode 的事件流没有心跳，空闲时什么都不发；一个请求开一条会留下读不完的线程，所以共用。

use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

pub struct Bus {
    subs: Mutex<HashMap<String, Sender<Value>>>,
    alive: std::sync::atomic::AtomicBool,
}

impl Bus {
    pub fn subscribe(&self, session: &str) -> Receiver<Value> {
        let (tx, rx) = channel();
        self.subs.lock().unwrap().insert(session.to_string(), tx);
        rx
    }

    pub fn unsubscribe(&self, session: &str) {
        self.subs.lock().unwrap().remove(session);
    }

    /// 另一个会话（这次请求派到后台的子智能体）的事件也送到 from 那个请求的收件箱：看它在做什么、有没有动静
    pub fn follow(&self, session: &str, from: &str) {
        let mut subs = self.subs.lock().unwrap();
        if let Some(tx) = subs.get(from).cloned() {
            subs.insert(session.to_string(), tx);
        }
    }
}

fn buses() -> &'static Mutex<HashMap<String, Arc<Bus>>> {
    static BUSES: OnceLock<Mutex<HashMap<String, Arc<Bus>>>> = OnceLock::new();
    BUSES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 拿到这个目录的总线（没有就连上；连上 = 收到响应头，之后发的事件都收得到）
pub fn bus(conn: &Conn, dir: &str) -> Result<Arc<Bus>, String> {
    let key = format!("{}\n{}", conn.base, dir);
    if let Some(b) = buses().lock().unwrap().get(&key) {
        if b.alive.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(b.clone());
        }
    }
    let mut events = Events::open(conn, Some(dir))?;
    let bus = Arc::new(Bus { subs: Mutex::new(HashMap::new()), alive: std::sync::atomic::AtomicBool::new(true) });
    buses().lock().unwrap().insert(key.clone(), bus.clone());
    let b = bus.clone();
    std::thread::spawn(move || {
        while let Some(event) = events.next() {
            // 提问的表单事件（form.created）会话 id 在 data.form.sessionID 里
            let session = event
                .pointer("/data/sessionID")
                .or_else(|| event.pointer("/data/form/sessionID"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if session.is_empty() {
                continue;
            }
            let tx = b.subs.lock().unwrap().get(&session).cloned();
            if let Some(tx) = tx {
                let _ = tx.send(event);
            }
        }
        // 断了：标记、通知还在等的请求（它们会按出错结束），下次用到时重连
        b.alive.store(false, std::sync::atomic::Ordering::SeqCst);
        let subs: Vec<Sender<Value>> = b.subs.lock().unwrap().values().cloned().collect();
        for tx in subs {
            let _ = tx.send(serde_json::json!({ "type": "bridge.disconnected" }));
        }
        let mut all = buses().lock().unwrap();
        if all.get(&key).map(|x| Arc::ptr_eq(x, &b)).unwrap_or(false) {
            all.remove(&key);
        }
    });
    Ok(bus)
}
