//! POST /v1/chat/completions：把 Friday 的一句话交给 opencode 自己的智能体（build：它自己的系统提示和工具）在对话的工作目录里做，
//! 内容实时返回：文字 → content，思考 → reasoning_content，工具活动 → reasoning_content 里的一行行，最后 usage。
//! Friday 的系统提示和工具不传（opencode 的免费模型只给它自己的智能体用，换了提示词会被网关拒）。
//!
//! 请求里 Friday 的扩展字段 x_friday：
//!   session     这段对话（或子智能体）的标识：同一个值 → 同一个 opencode 会话（opencode 记得前面聊过的）；没有 = 一次性的辅助请求
//!   cwd         工作目录（opencode 在这里读写、跑命令）；没有就用本程序的数据目录下的 work
//!   permission  Friday 会话框下的权限档：auto（默认：读写工作目录里的文件不问，跑命令等先问）| plan（只读）| full（都不问）
//! 要用户确认的操作（opencode 的 permission.asked）：这次回答以一段说明结束，opencode 那边停着等；用户下一句说「允许」/「总是允许」
//! 就放行并接着把后面的内容流回来，说别的就拒绝、再按新的话做。

use std::net::TcpStream;
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::backend::{self, Backend};
use crate::http::{self, Request, Sse};
use crate::{oc, store};

struct Ask {
    provider: String,
    model_id: String,
    model: String,
    /// 存会话映射用；一次性请求为空
    key: String,
    dir: String,
    permission: String,
    /// 发给 opencode 的话
    text: String,
    /// 用户这句话本身（判断是不是在回答权限确认）
    said: String,
    files: Vec<Value>,
    /// 这段对话之前的内容（新建 opencode 会话时带上，免得它不知道前面聊过什么）
    history: String,
    /// Friday 认结构化的工具事件（x_friday.toolEvents）：工具活动发成 x_friday.tool，画成原生工具行；否则写进思考区
    tool_events: bool,
}

fn text_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter(|p| matches!(p["type"].as_str(), Some("text") | Some("input_text")))
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn images_of(content: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    if let Value::Array(parts) = content {
        for p in parts {
            if p["type"].as_str() != Some("image_url") {
                continue;
            }
            let url = p["image_url"].as_str().or_else(|| p["image_url"]["url"].as_str()).unwrap_or("");
            if url.is_empty() {
                continue;
            }
            let ext = url.strip_prefix("data:image/").and_then(|r| r.split(';').next()).unwrap_or("png");
            out.push(json!({ "uri": url, "name": format!("image-{}.{}", out.len() + 1, ext) }));
        }
    }
    out
}

pub fn cut(s: &str, n: usize) -> String {
    let mut it = s.chars();
    let head: String = it.by_ref().take(n).collect();
    if it.next().is_some() {
        format!("{head}…")
    } else {
        head
    }
}

/// 末尾 n 个字
fn tail(s: &str, n: usize) -> String {
    let count = s.chars().count();
    if count <= n {
        s.to_string()
    } else {
        format!("…{}", s.chars().skip(count - n).collect::<String>())
    }
}

fn transcript(messages: &[Value]) -> String {
    let mut out = String::new();
    for m in messages {
        let role = m["role"].as_str().unwrap_or("");
        let text = text_of(&m["content"]);
        match role {
            "user" if !text.trim().is_empty() => out.push_str(&format!("用户：{}\n\n", text.trim())),
            "assistant" => {
                if !text.trim().is_empty() {
                    out.push_str(&format!("助手：{}\n\n", text.trim()));
                }
                for c in m["tool_calls"].as_array().into_iter().flatten() {
                    out.push_str(&format!("（助手调用了工具 {}）\n\n", c["function"]["name"].as_str().unwrap_or("?")));
                }
            }
            "tool" => out.push_str(&format!("（工具结果：{}）\n\n", cut(text.trim(), 300))),
            _ => {}
        }
    }
    out
}

fn parse(body: &Value) -> Result<Ask, String> {
    let model = body["model"].as_str().unwrap_or("").trim().to_string();
    if model.is_empty() {
        return Err("缺 model（写成 服务商/模型，如 opencode/glm-5.3-flash-free）".into());
    }
    let (provider, model_id) = match model.split_once('/') {
        Some((p, m)) => (p.to_string(), m.to_string()),
        None => ("opencode".to_string(), model.clone()),
    };
    let xf = &body["x_friday"];
    let default_dir = backend::data_dir().join("work");
    let _ = std::fs::create_dir_all(&default_dir);
    let dir = ["cwd", "workingDirectory", "directory"]
        .iter()
        .filter_map(|k| xf[*k].as_str())
        .find(|d| std::path::Path::new(d).is_absolute() && std::path::Path::new(d).is_dir())
        .map(str::to_string)
        .unwrap_or_else(|| default_dir.to_string_lossy().to_string());
    let session = xf["session"].as_str().unwrap_or("").to_string();
    let permission = xf["permission"].as_str().unwrap_or("auto").to_string();
    let tool_events = xf["toolEvents"].as_bool().unwrap_or(false);
    let messages: Vec<Value> = body["messages"].as_array().cloned().unwrap_or_default();
    let last_user = messages.iter().rposition(|m| m["role"].as_str() == Some("user"));
    let Some(last_user) = last_user else { return Err("messages 里没有用户的话".into()) };
    let said = text_of(&messages[last_user]["content"]);
    let files = images_of(&messages[last_user]["content"]);
    let before = &messages[..last_user];
    let key = if session.is_empty() {
        String::new()
    } else {
        let mut h = Sha256::new();
        h.update(session.as_bytes());
        h.update(b"\n");
        h.update(dir.as_bytes());
        format!("{:x}", h.finalize())[..24].to_string()
    };
    let (text, history) = if key.is_empty() {
        // 一次性的辅助请求（起标题、总结……）：Friday 的要求和前文都写进这一句里，让 opencode 的智能体直接回文字
        let system: Vec<String> = messages
            .iter()
            .filter(|m| matches!(m["role"].as_str(), Some("system") | Some("developer")))
            .map(|m| text_of(&m["content"]))
            .filter(|t| !t.trim().is_empty())
            .collect();
        let mut t = String::new();
        if !system.is_empty() {
            t.push_str(&format!("要求：\n{}\n\n", system.join("\n\n")));
        }
        let before_text = transcript(before);
        if !before_text.is_empty() {
            t.push_str(&format!("前文：\n{}\n", tail(&before_text, 12000)));
        }
        t.push_str(&said);
        t.push_str("\n\n（只用文字直接回答，不要用工具。）");
        (t, String::new())
    } else {
        (said.clone(), tail(&transcript(before), 12000))
    };
    Ok(Ask { provider, model_id, model, key, dir, permission, text, said, files, history, tool_events })
}

fn rule(action: &str, effect: &str) -> Value {
    json!({ "action": action, "resource": "*", "effect": effect })
}

/// Friday 的权限档（会话框下的 自动 / 计划 / 完全）→ opencode 的规则（后面的盖前面的）：
///   full  完全：都不问；auto 自动（默认）：读、改工作目录里的文件不问，跑命令、上网、出工作目录先问；
///   plan  计划：只读（改文件、跑命令都不许），配 opencode 的 plan 智能体；deny：一次性辅助请求，什么工具都不许
pub fn ruleset(mode: &str) -> Value {
    const READ: [&str; 6] = ["read", "glob", "grep", "list", "question", "todowrite"];
    let mut rules = Vec::new();
    match mode {
        "full" | "bypass" => rules.push(rule("*", "allow")),
        "deny" => rules.push(rule("*", "deny")),
        "plan" => {
            rules.push(rule("*", "deny"));
            for a in READ {
                rules.push(rule(a, "allow"));
            }
        }
        "auto" | "edits" | "acceptEdits" => {
            rules.push(rule("*", "ask"));
            for a in READ {
                rules.push(rule(a, "allow"));
            }
            rules.push(rule("edit", "allow"));
        }
        _ => {
            rules.push(rule("*", "ask"));
            for a in READ {
                rules.push(rule(a, "allow"));
            }
        }
    }
    Value::Array(rules)
}

/// 计划档用 opencode 的 plan 智能体（只分析、出方案），其余用 build
fn agent_for(mode: &str) -> &'static str {
    if mode == "plan" {
        "plan"
    } else {
        "build"
    }
}

fn model_ref(ask: &Ask) -> Value {
    json!({ "providerID": ask.provider, "id": ask.model_id })
}

/// 这段对话的 opencode 会话（没有、换了目录、在那边被删了 → 新建）；返回（会话 id，是不是新建的）
fn ensure_session(be: &Backend, ask: &Ask) -> Result<(String, bool), String> {
    let dir = Some(ask.dir.as_str());
    if !ask.key.is_empty() {
        if let Some(e) = store::get(&ask.key) {
            if e.dir == ask.dir && e.base == be.conn.base {
                let (status, _) = oc::get(&be.conn, &format!("/api/session/{}", e.session), dir)?;
                if status == 200 {
                    if e.model != ask.model {
                        oc::ok(oc::post(&be.conn, &format!("/api/session/{}/model", e.session), dir, &json!({ "model": model_ref(ask) })), "换模型")?;
                    }
                    // 权限档每次都按 Friday 当前的设（用户可能改了），智能体跟着（计划 ↔ 其它）
                    let _ = oc::post_patch(&be.conn, &format!("/api/session/{}", e.session), dir, &json!({ "permissions": ruleset(&ask.permission) }));
                    let _ = oc::post(&be.conn, &format!("/api/session/{}/agent", e.session), dir, &json!({ "agent": agent_for(&ask.permission) }));
                    store::touch(&ask.key, &ask.model);
                    return Ok((e.session, false));
                }
            }
        }
    }
    let title = if ask.key.is_empty() { "Friday（辅助）".to_string() } else { format!("Friday · {}", cut(ask.said.trim(), 30)) };
    let permissions = if ask.key.is_empty() { ruleset("deny") } else { ruleset(&ask.permission) };
    let created = oc::ok(
        oc::post(
            &be.conn,
            "/api/session",
            dir,
            &json!({ "agent": agent_for(&ask.permission), "model": model_ref(ask), "location": { "directory": ask.dir }, "permissions": permissions, "title": title }),
        ),
        "建 opencode 会话",
    )?;
    let id = created["data"]["id"].as_str().ok_or("opencode 建会话没返回 id")?.to_string();
    if !ask.key.is_empty() {
        store::put(&ask.key, store::Entry { session: id.clone(), dir: ask.dir.clone(), model: ask.model.clone(), base: be.conn.base.clone(), used: store::now() });
    }
    Ok((id, true))
}

/// 用户这句话是不是在回答权限确认：Some("once" | "always")；不是返回 None（= 拒绝，按新的话做）
fn approval(said: &str) -> Option<&'static str> {
    // Friday 在用户的话后面附了「（发出时间：…）」（v2.92）：只看前面用户自己说的
    let said = said.split("\n\n（发出时间").next().unwrap_or(said);
    let t = said.trim().trim_end_matches(['。', '！', '!', '.', '，', ',']).to_lowercase();
    if t.chars().count() > 12 {
        return None;
    }
    const ALWAYS: [&str; 5] = ["总是允许", "一直允许", "始终允许", "以后都允许", "always"];
    const ONCE: [&str; 14] = ["允许", "允许一次", "同意", "可以", "好", "好的", "继续", "行", "是", "确认", "yes", "y", "ok", "allow"];
    if ALWAYS.iter().any(|w| t == *w) {
        Some("always")
    } else if ONCE.iter().any(|w| t == *w) {
        Some("once")
    } else {
        None
    }
}

fn tool_label(name: &str) -> &str {
    match name {
        "read" => "读取",
        "write" => "写入",
        "edit" => "修改",
        "shell" | "bash" => "运行",
        "grep" => "搜索",
        "glob" => "查找文件",
        "list" => "列目录",
        "webfetch" => "打开网页",
        "websearch" => "搜索网页",
        "subagent" | "task" => "派子智能体",
        "question" => "提问",
        "skill" => "用技能",
        "execute" => "执行脚本",
        "todowrite" => "更新待办",
        other => other,
    }
}

fn permission_label(action: &str) -> &str {
    match action {
        "shell" | "bash" => "运行命令",
        "edit" | "write" => "修改文件",
        "read" => "读取文件",
        "webfetch" => "打开网页",
        "websearch" => "搜索网页",
        "external_directory" => "访问工作目录以外的文件",
        "subagent" | "task" => "派子智能体",
        other => other,
    }
}

fn tool_arg(input: &Value, dir: &str) -> String {
    for k in ["command", "path", "filePath", "file_path", "pattern", "url", "query", "description", "prompt"] {
        if let Some(s) = input[k].as_str() {
            let s = s.strip_prefix(dir).map(|r| r.trim_start_matches(['/', '\\'])).filter(|r| !r.is_empty()).unwrap_or(s);
            return cut(s, 160);
        }
    }
    if input.is_null() || input.as_object().map(|o| o.is_empty()).unwrap_or(false) {
        String::new()
    } else {
        cut(&input.to_string(), 160)
    }
}

/// 参数里工作目录下的路径写成相对路径（工具行上短一些）
fn relative(value: &Value, dir: &str) -> Value {
    match value {
        Value::String(s) => {
            let prefix = format!("{}/", dir.trim_end_matches(['/', '\\']));
            match s.strip_prefix(&prefix) {
                Some(rest) if !rest.is_empty() => Value::String(rest.to_string()),
                _ => value.clone(),
            }
        }
        Value::Object(map) => Value::Object(map.iter().map(|(k, v)| (k.clone(), relative(v, dir))).collect()),
        Value::Array(items) => Value::Array(items.iter().map(|v| relative(v, dir)).collect()),
        _ => value.clone(),
    }
}

/// 工具结果的全文（opencode 的 content 是 [{type:text,text}…]）
fn full_text(content: &Value) -> String {
    match content {
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n"),
        Value::String(s) => s.clone(),
        _ => String::new(),
    }
}

fn first_line(content: &Value) -> String {
    let text = match content {
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n"),
        Value::String(s) => s.clone(),
        _ => String::new(),
    };
    cut(text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or(""), 100)
}

fn chunk(model: &str, delta: Value, finish: Option<&str>, usage: Option<&Value>) -> Value {
    let mut c = json!({
        "id": "chatcmpl-opencode",
        "object": "chat.completion.chunk",
        "created": store::now(),
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
    });
    if let Some(u) = usage {
        c["usage"] = u.clone();
    }
    c
}

#[derive(Default)]
struct Usage {
    prompt: u64,
    cached: u64,
    completion: u64,
    reasoning: u64,
}

impl Usage {
    /// prompt 取最后一步的（= 现在的上下文有多长），输出各步相加
    fn add(&mut self, tokens: &Value) {
        let input = tokens["input"].as_u64().unwrap_or(0);
        let read = tokens["cache"]["read"].as_u64().unwrap_or(0);
        let write = tokens["cache"]["write"].as_u64().unwrap_or(0);
        self.prompt = input + read + write;
        self.cached = read;
        self.completion += tokens["output"].as_u64().unwrap_or(0) + tokens["reasoning"].as_u64().unwrap_or(0);
        self.reasoning += tokens["reasoning"].as_u64().unwrap_or(0);
    }

    fn json(&self) -> Value {
        json!({
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
            "total_tokens": self.prompt + self.completion,
            "prompt_tokens_details": { "cached_tokens": self.cached },
            "completion_tokens_details": { "reasoning_tokens": self.reasoning },
        })
    }
}

struct Failure {
    status: u16,
    kind: String,
    message: String,
    retry_after_ms: u64,
}

/// 往 Friday 写：流式时懒开 SSE（出错在开头的话还能回 HTTP 错误码），非流式时攒着
struct Out<'a> {
    stream: Option<&'a mut TcpStream>,
    sse: Option<Sse<'a>>,
    streaming: bool,
    model: String,
    content: String,
    reasoning: String,
    gone: bool,
    /// 发过的工具事件条数（算「已经写出东西了」：之后出错不能再回 HTTP 错误码）
    tools: usize,
}

impl<'a> Out<'a> {
    fn emit(&mut self, delta: Value) {
        if let Some(s) = delta["content"].as_str() {
            self.content.push_str(s);
        }
        if let Some(s) = delta["reasoning_content"].as_str() {
            self.reasoning.push_str(s);
        }
        if !self.streaming || self.gone {
            return;
        }
        if self.sse.is_none() {
            let stream = self.stream.take().unwrap();
            match Sse::start(stream) {
                Ok(mut sse) => {
                    sse.data(&chunk(&self.model, json!({ "role": "assistant", "content": "" }), None, None));
                    self.sse = Some(sse);
                }
                Err(_) => {
                    self.gone = true;
                    return;
                }
            }
        }
        let sse = self.sse.as_mut().unwrap();
        if !sse.data(&chunk(&self.model, delta, None, None)) {
            self.gone = true;
        }
    }

    /// 一条工具事件（只在流式、Friday 认的时候发）：chunk 顶层 x_friday.tool，delta 为空
    fn tool(&mut self, tool: Value) {
        if !self.streaming || self.gone {
            return;
        }
        // 还没开流：先开（发 role）
        if self.sse.is_none() {
            self.emit(json!({}));
            if self.gone {
                return;
            }
        }
        let mut c = chunk(&self.model, json!({}), None, None);
        c["x_friday"] = json!({ "tool": tool });
        self.tools += 1;
        if !self.sse.as_mut().unwrap().data(&c) {
            self.gone = true;
        }
    }

    fn ping(&mut self) {
        if let Some(sse) = self.sse.as_mut() {
            // SSE 注释行：保活，也顺便发现客户端走了
            if !sse.comment() {
                self.gone = true;
            }
        }
    }

    fn started(&self) -> bool {
        !self.content.is_empty() || !self.reasoning.is_empty() || self.tools > 0
    }
}

pub fn handle(req: &Request, stream: &mut TcpStream) {
    let body: Value = match serde_json::from_slice(&req.body) {
        Ok(v) => v,
        Err(e) => {
            let _ = http::error(stream, 400, "invalid_request", &format!("请求体不是 JSON：{e}"), &[]);
            return;
        }
    };
    let ask = match parse(&body) {
        Ok(a) => a,
        Err(e) => {
            let _ = http::error(stream, 400, "invalid_request", &e, &[]);
            return;
        }
    };
    let streaming = body["stream"].as_bool().unwrap_or(false);
    // 连不上（桌面版重开换了端口）就重新找一次后端再来
    for attempt in 0..2 {
        let be = match backend::current() {
            Ok(b) => b,
            Err(e) => {
                let _ = http::error(stream, 503, "opencode_unavailable", &e, &[]);
                return;
            }
        };
        match run(&be, &ask, stream, streaming) {
            Err(e) if e.starts_with(oc::CONNECT_ERROR) && attempt == 0 => {
                backend::reset();
                continue;
            }
            Err(e) => {
                if e.starts_with(oc::CONNECT_ERROR) {
                    backend::reset();
                }
                let _ = http::error(stream, 502, "opencode_error", &e, &[]);
                return;
            }
            Ok(()) => return,
        }
    }
}

/// 返回 Err 时还没往 Friday 写过任何东西（调用方回 HTTP 错误）
fn run(be: &Backend, ask: &Ask, stream: &mut TcpStream, streaming: bool) -> Result<(), String> {
    let (sid, created) = ensure_session(be, ask)?;
    backend::log(&format!(
        "chat key={} session={sid} new={created} model={} permission={} pending={} said={:?}",
        if ask.key.is_empty() { "-" } else { &ask.key[..8] },
        ask.model,
        ask.permission,
        !ask.key.is_empty() && store::pending(&ask.key).is_some(),
        cut(&ask.said, 40)
    ));
    let dir = Some(ask.dir.as_str());
    let bus = oc::bus(&be.conn, &ask.dir)?;
    let rx = bus.subscribe(&sid);
    let result = (|| -> Result<(), String> {
        // 上一句停在权限确认上：这句是「允许」就放行接着收，否则拒掉再按新的话做
        let mut resumed = false;
        if !ask.key.is_empty() {
            if let Some(p) = store::pending(&ask.key) {
                store::set_pending(&ask.key, None);
                let decision = approval(&ask.said).unwrap_or("reject");
                let reply = oc::post(&be.conn, &format!("/api/session/{sid}/permission/{}/reply", p.request), dir, &json!({ "decision": decision }));
                let replied = matches!(reply, Ok((s, _)) if (200..300).contains(&s));
                backend::log(&format!("permission {} → {decision}（{}）", p.request, match &reply { Ok((s, _)) => format!("HTTP {s}"), Err(e) => e.clone() }));
                if decision != "reject" && replied {
                    resumed = true;
                } else if replied {
                    // 拒掉后 opencode 会中止这次执行：等它停下再发新的话（最多 5 秒）
                    let deadline = Instant::now() + Duration::from_secs(5);
                    while Instant::now() < deadline {
                        match rx.recv_timeout(Duration::from_millis(200)) {
                            Ok(e) if e["type"].as_str().map(|t| t.starts_with("session.execution.") && t != "session.execution.started").unwrap_or(false) => break,
                            _ => {}
                        }
                    }
                }
            }
        }
        if !resumed {
            let mut text = ask.text.clone();
            if created && !ask.history.trim().is_empty() {
                text = format!("下面是这段对话之前的内容（供参考）：\n<之前的对话>\n{}\n</之前的对话>\n\n现在用户说：\n{}", ask.history.trim(), ask.text);
            }
            let mut prompt = json!({ "text": text });
            if !ask.files.is_empty() {
                prompt["files"] = Value::Array(ask.files.clone());
            }
            oc::ok(oc::post(&be.conn, &format!("/api/session/{sid}/prompt"), dir, &prompt), "发给 opencode")?;
        }
        pump(be, ask, &sid, &rx, stream, streaming)
    })();
    bus.unsubscribe(&sid);
    if ask.key.is_empty() {
        let _ = oc::delete(&be.conn, &format!("/api/session/{sid}"), dir);
    }
    result
}

fn pump(be: &Backend, ask: &Ask, sid: &str, rx: &std::sync::mpsc::Receiver<Value>, stream: &mut TcpStream, streaming: bool) -> Result<(), String> {
    let dir = Some(ask.dir.as_str());
    let mut out = Out { stream: Some(stream), sse: None, streaming, model: ask.model.clone(), content: String::new(), reasoning: String::new(), gone: false, tools: 0 };
    let structured = ask.tool_events && streaming;
    let mut names: std::collections::HashMap<String, String> = Default::default();
    let mut usage = Usage::default();
    let mut failure: Option<Failure> = None;
    let mut last_event = Instant::now();
    let mut waiting_permission = false;
    loop {
        if out.gone {
            // Friday 停了 / 断了：opencode 也停
            let _ = oc::post(&be.conn, &format!("/api/session/{sid}/interrupt"), dir, &json!({}));
            if !ask.key.is_empty() {
                store::set_pending(&ask.key, None);
            }
            return Ok(());
        }
        let event = match rx.recv_timeout(Duration::from_secs(15)) {
            Ok(e) => e,
            Err(RecvTimeoutError::Timeout) => {
                out.ping();
                if last_event.elapsed() > Duration::from_secs(900) {
                    let _ = oc::post(&be.conn, &format!("/api/session/{sid}/interrupt"), dir, &json!({}));
                    failure = Some(Failure { status: 504, kind: "timeout".into(), message: "opencode 15 分钟没有动静，已停下".into(), retry_after_ms: 0 });
                    break;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => {
                failure = Some(Failure { status: 502, kind: "opencode_error".into(), message: "和 opencode 的连接断了".into(), retry_after_ms: 0 });
                break;
            }
        };
        last_event = Instant::now();
        let t = event["type"].as_str().unwrap_or("");
        let d = &event["data"];
        match t {
            "session.text.delta" => out.emit(json!({ "content": d["delta"].as_str().unwrap_or("") })),
            "session.reasoning.delta" => out.emit(json!({ "reasoning_content": d["delta"].as_str().unwrap_or("") })),
            "session.tool.input.started" => {
                names.insert(d["id"].as_str().unwrap_or("").to_string(), d["name"].as_str().unwrap_or("").to_string());
            }
            "session.tool.called" => {
                let id = d["id"].as_str().unwrap_or("").to_string();
                let name = names.get(&id).cloned().unwrap_or_default();
                let arg = tool_arg(&d["input"], &ask.dir);
                if structured {
                    out.tool(json!({ "id": id, "name": name, "title": tool_label(&name), "summary": arg, "input": relative(&d["input"], &ask.dir), "status": "running" }));
                } else {
                    out.emit(json!({ "reasoning_content": format!("\n▸ {} {}\n", tool_label(&name), arg) }));
                }
            }
            "session.tool.success" => {
                if structured {
                    let id = d["id"].as_str().unwrap_or("").to_string();
                    let name = names.get(&id).cloned().unwrap_or_default();
                    out.tool(json!({ "id": id, "name": name, "status": "success", "output": cut(&full_text(&d["content"]), 8000), "metadata": d["metadata"] }));
                } else {
                    let line = first_line(&d["content"]);
                    out.emit(json!({ "reasoning_content": if line.is_empty() { "  ✓\n".to_string() } else { format!("  ✓ {line}\n") } }));
                }
            }
            "session.tool.failed" => {
                let msg = d["error"]["message"].as_str().unwrap_or("失败");
                if structured {
                    let id = d["id"].as_str().unwrap_or("").to_string();
                    let name = names.get(&id).cloned().unwrap_or_default();
                    out.tool(json!({ "id": id, "name": name, "status": "failed", "output": cut(msg, 2000) }));
                } else {
                    out.emit(json!({ "reasoning_content": format!("  ✗ {}\n", cut(msg, 160)) }));
                }
            }
            "permission.asked" => {
                let action = d["action"].as_str().unwrap_or("").to_string();
                let resources: Vec<String> = d["resources"].as_array().into_iter().flatten().filter_map(|r| r.as_str().map(str::to_string)).collect();
                let request = d["id"].as_str().unwrap_or("").to_string();
                if ask.key.is_empty() {
                    // 一次性请求不该用工具：直接拒
                    let _ = oc::post(&be.conn, &format!("/api/session/{sid}/permission/{request}/reply"), dir, &json!({ "decision": "reject" }));
                    continue;
                }
                // 对应的工具行标成「等确认」（不然一直显示运行中）
                if structured {
                    if let Some(call) = d["source"]["id"].as_str() {
                        out.tool(json!({ "id": call, "status": "waiting", "output": format!("等你确认：{}", permission_label(&action)) }));
                    }
                }
                let list = resources.iter().map(|r| cut(r, 300)).collect::<Vec<_>>().join("\n");
                if structured {
                    // 要做的事已经在等确认的工具行上了（带允许 / 拒绝按钮）：正文只说一句
                    out.emit(json!({ "content": format!(
                        "\n\nOpenCode 想{}，等你确认：点上面的「允许执行」或回复「允许」（「总是允许」以后同类不再问）；回复别的就取消，按你新说的做。",
                        permission_label(&action)) }));
                } else {
                    out.emit(json!({ "content": format!(
                        "\n\n> **OpenCode 想{}**\n\n```\n{}\n```\n\n回复「允许」放行这一次，回复「总是允许」以后同类操作不再问；回复别的就取消，按你新说的做。",
                        permission_label(&action), list) }));
                }
                backend::log(&format!("permission asked {request} action={action} session={sid}"));
                store::set_pending(&ask.key, Some(store::Pending { request }));
                waiting_permission = true;
                break;
            }
            "session.step.ended" => usage.add(&d["tokens"]),
            "session.step.failed" => {
                let err = &d["error"];
                if err["type"].as_str() != Some("aborted") {
                    failure = Some(Failure {
                        status: err["status"].as_u64().unwrap_or(502) as u16,
                        kind: err["type"].as_str().unwrap_or("opencode_error").to_string(),
                        message: err["message"].as_str().unwrap_or("opencode 出错").to_string(),
                        retry_after_ms: err["retryAfterMs"].as_u64().unwrap_or(0),
                    });
                }
            }
            "session.execution.failed" => {
                if failure.is_none() {
                    let err = &d["error"];
                    failure = Some(Failure {
                        status: err["status"].as_u64().unwrap_or(502) as u16,
                        kind: err["type"].as_str().unwrap_or("opencode_error").to_string(),
                        message: err["message"].as_str().unwrap_or("opencode 出错").to_string(),
                        retry_after_ms: err["retryAfterMs"].as_u64().unwrap_or(0),
                    });
                }
                break;
            }
            "session.execution.succeeded" | "session.execution.interrupted" => break,
            "bridge.disconnected" => {
                failure = Some(Failure { status: 502, kind: "opencode_error".into(), message: "和 opencode 的连接断了".into(), retry_after_ms: 0 });
                backend::reset();
                break;
            }
            _ => {}
        }
    }
    let _ = waiting_permission;
    if let Some(f) = &failure {
        if !out.started() {
            // 什么都还没写：回 HTTP 错误码（429 / 402 让 Friday 按额度处理，带上多久后再试）
            let mut extra = Vec::new();
            if f.retry_after_ms > 0 {
                extra.push(("Retry-After", ((f.retry_after_ms + 999) / 1000).to_string()));
            }
            if let Some(s) = out.stream.take() {
                let status = if (400..600).contains(&f.status) { f.status } else { 502 };
                let _ = http::error(s, status, &f.kind, &f.message, &extra);
            }
            return Ok(());
        }
        out.emit(json!({ "content": format!("\n\n（OpenCode 出错：{}）", f.message) }));
    }
    let usage = usage.json();
    if streaming {
        if out.sse.is_none() && !out.gone {
            out.emit(json!({ "content": "" }));
        }
        if let Some(sse) = out.sse.as_mut() {
            sse.data(&chunk(&ask.model, json!({}), Some("stop"), Some(&usage)));
            sse.done();
        }
    } else if let Some(s) = out.stream.take() {
        let mut message = json!({ "role": "assistant", "content": out.content });
        if !out.reasoning.is_empty() {
            message["reasoning_content"] = Value::String(out.reasoning.clone());
        }
        let _ = http::json(
            s,
            200,
            &json!({
                "id": "chatcmpl-opencode", "object": "chat.completion", "created": store::now(), "model": ask.model,
                "choices": [{ "index": 0, "message": message, "finish_reason": "stop" }],
                "usage": usage,
            }),
        );
    }
    Ok(())
}
