//! POST /v1/chat/completions：把 Friday 的一句话交给 opencode 自己的智能体（build：它自己的系统提示和工具）在对话的工作目录里做，
//! 内容实时返回：文字 → content，思考 → reasoning_content，工具活动 → reasoning_content 里的一行行，最后 usage。
//! Friday 的系统提示和工具不传（opencode 的免费模型只给它自己的智能体用，换了提示词会被网关拒）。
//!
//! 请求里 Friday 的扩展字段 x_friday：
//!   session     这段对话（或子智能体）的标识：同一个值 → 同一个 opencode 会话（opencode 记得前面聊过的）；没有 = 一次性的辅助请求
//!   cwd         工作目录（opencode 在这里读写、跑命令）；没有就用本程序的数据目录下的 work
//!   permission  Friday 会话框下的权限档：auto（默认：读写工作目录里的文件不问，跑命令等先问）| plan（只读）| full（都不问）
//!   questions   Friday 会把等回答的提问行画成选项卡片（v2.97.25 起）：提问时正文不再列一遍问题
//!   continuing  这句是 Friday 自己接着做的话（出错后自动恢复、点了重试、换了模型接着做），不是用户说的：
//!               还在等的提问 / 确认不拿它当回答，原样再交回去（工具行再标一次等回答，这次到此为止）
//! 要用户确认的操作（opencode 的 permission.asked）：这次回答以一段说明结束，opencode 那边停着等；用户下一句说「允许」/「总是允许」
//! 就放行并接着把后面的内容流回来，说别的就拒绝、再按新的话做。
//! 提问（opencode 的 question 工具建一张表单 form.created，一直等到有人回答）：提问那一行标成等回答、这次回答到此结束；
//! 用户下一句话就是回答——几题用「；」隔开、选项写名字或序号，填进表单（POST …/form/<id>/reply），opencode 接着做；
//! 填不进去（选项外的回答而这题不许自己写）就取消提问、按新的话做。
//! 派到后台的子智能体（subagent 工具 background: true）：父会话那一轮先结束，子智能体做完后 OpenCode 往父会话塞一条合成消息、自己再开一轮；
//! 还有没交回的，这次回答就不收尾——那一行显示「在后台运行：它在做什么」，交回后标成功 / 失败（结果是它的回报），接着的那一轮照常流回来。
//! 子智能体要确认 / 提问时同样转给用户（回复发到子会话），下一句话进来接着等它们。

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
    /// Friday 把等回答的提问行画成选项卡片（x_friday.questions）：提问时正文不再列问题
    questions: bool,
    /// 这句是 Friday 自己接着做的话（x_friday.continuing）：不是对提问 / 确认的回答
    continuing: bool,
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
    let questions = xf["questions"].as_bool().unwrap_or(false);
    let continuing = xf["continuing"].as_bool().unwrap_or(false);
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
    Ok(Ask { provider, model_id, model, key, dir, permission, text, said, files, history, tool_events, questions, continuing })
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

/// Friday 在用户的话后面附了「（发出时间：…）」（v2.92）：只看前面用户自己说的
fn own_words(said: &str) -> &str {
    said.split("\n\n（发出时间").next().unwrap_or(said).trim()
}

/// 用户这句话是不是在回答权限确认：Some("once" | "always")；不是返回 None（= 拒绝，按新的话做）
fn approval(said: &str) -> Option<&'static str> {
    let t = own_words(said).trim_end_matches(['。', '！', '!', '.', '，', ',']).to_lowercase();
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
            // 空的数据块（不是注释行）：保活、发现客户端走了；EDITH 只在收到数据时才看要不要停，OpenCode 安静时点停止也能马上停
            if !sse.data(&chunk(&self.model, json!({}), None, None)) {
                self.gone = true;
            }
        }
    }

    fn started(&self) -> bool {
        !self.content.is_empty() || !self.reasoning.is_empty() || self.tools > 0
    }
}

/// 正在跑的对话请求有几个（/v1/info 的 active：宿主据此不把在干活的桥当成闲置停掉）
pub static ACTIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct ActiveGuard;
impl ActiveGuard {
    fn new() -> Self {
        ACTIVE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        ActiveGuard
    }
}
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

pub fn handle(req: &Request, stream: &mut TcpStream) {
    let _active = ActiveGuard::new();
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
        let mut carried = None;
        // 回答过的提问：opencode 之后报它做完不再转给 Friday（那一行已经记着「回答：…」，再报会盖掉；Friday 重开过的还会多出一行空的）
        let mut quiet = None;
        if !ask.key.is_empty() {
            // 内存里没记着（本程序闲置被停过、重启了）：问 opencode 这个会话还有没有在等的提问 / 确认
            let pending = store::pending(&ask.key).or_else(|| waiting_in_opencode(be, &sid, dir));
            if let Some(p) = pending.as_ref().filter(|_| ask.continuing) {
                // Friday 自己接着做的话（出错后自动恢复、点了重试）不是回答：还在等的提问 / 确认原样再交回去，这次到此为止
                backend::log(&format!("session {sid} 还在等 {}，这句是 Friday 接着做的话：再交回去", p.request));
                store::set_pending(&ask.key, Some(p.clone()));
                reshow(ask, p, stream, streaming);
                return Ok(());
            }
            if let Some(p) = pending.as_ref().filter(|p| p.form.is_some()) {
                // 上一句停在提问上：这句话就是回答
                quiet = p.tool.as_ref().and_then(|t| t["id"].as_str()).map(str::to_string);
                store::set_pending(&ask.key, None);
                let form = p.form.as_ref().unwrap();
                // 后台子智能体问的：回复发到子会话
                let asker = p.session.clone().unwrap_or_else(|| sid.clone());
                let answer = form_answer(form, own_words(&ask.said));
                let reply = match &answer {
                    Some(a) => oc::post(&be.conn, &format!("/api/session/{asker}/form/{}/reply", p.request), dir, &json!({ "answer": a })),
                    None => Err("回答对不上表单（选项外的回答，而这题不许自己写）".to_string()),
                };
                let replied = matches!(reply, Ok((s, _)) if (200..300).contains(&s));
                backend::log(&format!(
                    "form {} → {}（{}）",
                    p.request,
                    answer.as_ref().map(|a| a.to_string()).unwrap_or_default(),
                    match &reply { Ok((s, _)) => format!("HTTP {s}"), Err(e) => e.clone() }
                ));
                if replied {
                    resumed = true;
                } else {
                    // 填不进去：取消提问、停下这次执行，再把这句话当新的话发
                    let _ = oc::delete(&be.conn, &format!("/api/session/{asker}/form/{}", p.request), dir);
                    if p.session.is_none() {
                        let _ = oc::post(&be.conn, &format!("/api/session/{sid}/interrupt"), dir, &json!({}));
                        wait_stopped(&rx);
                    }
                }
            } else if let Some(p) = pending {
                carried = p.tool.clone();
                store::set_pending(&ask.key, None);
                let decision = approval(&ask.said).unwrap_or("reject");
                let asker = p.session.clone().unwrap_or_else(|| sid.clone());
                let reply = oc::post(&be.conn, &format!("/api/session/{asker}/permission/{}/reply", p.request), dir, &json!({ "decision": decision }));
                let replied = matches!(reply, Ok((s, _)) if (200..300).contains(&s));
                backend::log(&format!("permission {} → {decision}（{}）", p.request, match &reply { Ok((s, _)) => format!("HTTP {s}"), Err(e) => e.clone() }));
                if decision != "reject" && replied {
                    resumed = true;
                } else if replied && p.session.is_none() {
                    // 拒掉后 opencode 会中止这次执行：等它停下再发新的话
                    wait_stopped(&rx);
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
        pump(be, ask, &sid, &bus, &rx, stream, streaming, if resumed { carried } else { None }, quiet)
    })();
    bus.unsubscribe(&sid);
    if ask.key.is_empty() {
        let _ = oc::delete(&be.conn, &format!("/api/session/{sid}"), dir);
    }
    result
}

/// carried：放行的那个等确认的工具（上一次请求里报过开始），这次报它做完时带上它的名字和参数；
/// quiet：回答过的提问（它做完不再报）
#[allow(clippy::too_many_arguments)]
fn pump(
    be: &Backend,
    ask: &Ask,
    sid: &str,
    bus: &oc::Bus,
    rx: &std::sync::mpsc::Receiver<Value>,
    stream: &mut TcpStream,
    streaming: bool,
    carried: Option<Value>,
    quiet: Option<String>,
) -> Result<(), String> {
    let dir = Some(ask.dir.as_str());
    let mut out = Out { stream: Some(stream), sse: None, streaming, model: ask.model.clone(), content: String::new(), reasoning: String::new(), gone: false, tools: 0 };
    let structured = ask.tool_events && streaming;
    let mut names: std::collections::HashMap<String, String> = Default::default();
    // 这次（和放行的那个）报过开始的工具：id → {id, name, title, summary, input}；之后的状态都带上这几项
    let mut called: std::collections::HashMap<String, Value> = Default::default();
    // 这次派到后台、还没报子会话的 subagent 调用
    let mut bg_calls: std::collections::HashSet<String> = Default::default();
    if let Some(tool) = carried {
        if let Some(id) = tool["id"].as_str() {
            names.insert(id.to_string(), tool["name"].as_str().unwrap_or("").to_string());
            called.insert(id.to_string(), tool.clone());
            // 自动档下派子智能体要确认：放行的那次 subagent 调用是派到后台的（参数在上一次请求里报过）
            if tool["name"].as_str() == Some("subagent") && tool["input"]["background"].as_bool() == Some(true) {
                bg_calls.insert(id.to_string());
            }
        }
    }
    let update = |called: &std::collections::HashMap<String, Value>, id: &str, name: &str, extra: Value| -> Value {
        let mut tool = called.get(id).cloned().unwrap_or_else(|| json!({ "id": id, "name": name, "title": tool_label(name) }));
        for (key, value) in extra.as_object().cloned().unwrap_or_default() {
            tool[key] = value;
        }
        tool
    };
    let mut usage = Usage::default();
    let mut failure: Option<Failure> = None;
    let mut last_event = Instant::now();
    let mut waiting_permission = false;
    // 后台子智能体：上一次请求停在确认 / 提问上时还没交回的、opencode 里这个会话还在跑的子会话，接着等
    let mut background: Vec<Background> = Vec::new();
    let mut followed = Followed { bus, ids: Vec::new() };
    if !ask.key.is_empty() {
        let known = store::children(&ask.key);
        for child in running_children(be, sid, dir) {
            let call = known.iter().find(|c| c.session == child).map(|c| c.call.clone()).unwrap_or_default();
            backend::log(&format!("session {sid} 还有在后台跑的子智能体 {child}，接着等"));
            adopt(&mut background, &mut followed, sid, &child, &call);
        }
    }
    // 父会话又开了一轮（后台子智能体交回后）：接着的文字和前面空一行
    let mut gap = false;
    let mut last_check = Instant::now();
    loop {
        if out.gone {
            // Friday 停了 / 断了：opencode 也停（派到后台的子智能体一起停）
            let _ = oc::post(&be.conn, &format!("/api/session/{sid}/interrupt"), dir, &json!({}));
            for b in background.iter().filter(|b| !b.handled) {
                let _ = oc::post(&be.conn, &format!("/api/session/{}/interrupt", b.child), dir, &json!({}));
            }
            if !ask.key.is_empty() {
                store::set_pending(&ask.key, None);
                store::set_children(&ask.key, Vec::new());
            }
            return Ok(());
        }
        let event = match rx.recv_timeout(Duration::from_secs(15)) {
            Ok(e) => e,
            Err(RecvTimeoutError::Timeout) => {
                out.ping();
                let waiting = background.iter().any(|b| !b.handled) || !bg_calls.is_empty();
                if waiting && last_check.elapsed() > Duration::from_secs(60) {
                    // 事件漏了也不干等：父会话、子智能体都不在跑，也没有没送达的消息，就收尾
                    last_check = Instant::now();
                    if !still_waiting(be, sid, dir, &background) {
                        backend::log(&format!("session {sid} 后台子智能体都不在跑了，收尾"));
                        break;
                    }
                }
                // 等后台子智能体时它跑长命令可以很久没动静（Friday 点停止随时能停）
                let limit = if waiting { 6 * 3600 } else { 900 };
                if last_event.elapsed() > Duration::from_secs(limit) {
                    let _ = oc::post(&be.conn, &format!("/api/session/{sid}/interrupt"), dir, &json!({}));
                    let message = format!("opencode {}没有动静，已停下", if waiting { "6 小时" } else { "15 分钟" });
                    failure = Some(Failure { status: 504, kind: "timeout".into(), message, retry_after_ms: 0 });
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
        // 别的会话（这次等着的后台子智能体）的事件：看它在做什么（写到父会话那一行上），它要确认 / 提问时转给用户；别的不管
        let from = d["sessionID"].as_str().or_else(|| d["form"]["sessionID"].as_str()).unwrap_or(sid).to_string();
        let child = from != sid;
        if child {
            match t {
                "session.tool.input.started" => {
                    names.insert(d["id"].as_str().unwrap_or("").to_string(), d["name"].as_str().unwrap_or("").to_string());
                    continue;
                }
                "session.tool.called" => {
                    let id = d["id"].as_str().unwrap_or("").to_string();
                    let name = names.get(&id).cloned().unwrap_or_default();
                    let arg = tool_arg(&d["input"], &ask.dir);
                    called.insert(id.clone(), json!({ "id": id, "name": name, "title": tool_label(&name), "summary": arg, "input": relative(&d["input"], &ask.dir) }));
                    if let Some(b) = background.iter().find(|b| b.child == from && !b.call.is_empty() && b.inbox.is_none()) {
                        if structured {
                            out.tool(update(&called, &b.call, "subagent", json!({ "status": "running", "output": format!("在后台运行：{} {}", tool_label(&name), arg) })));
                        }
                    }
                    continue;
                }
                "permission.asked" | "form.created" => {}
                _ => continue,
            }
        }
        // 后台子智能体要确认 / 提问时正文里这么称呼它
        let who = if child { "OpenCode 的后台子智能体" } else { "OpenCode" };
        match t {
            "session.text.delta" => {
                let mut text = d["delta"].as_str().unwrap_or("").to_string();
                if gap && !text.is_empty() {
                    gap = false;
                    let sep = if out.content.ends_with("\n\n") || out.content.is_empty() { "" } else if out.content.ends_with('\n') { "\n" } else { "\n\n" };
                    text.insert_str(0, sep);
                }
                out.emit(json!({ "content": text }));
            }
            "session.reasoning.delta" => out.emit(json!({ "reasoning_content": d["delta"].as_str().unwrap_or("") })),
            "session.tool.input.started" => {
                names.insert(d["id"].as_str().unwrap_or("").to_string(), d["name"].as_str().unwrap_or("").to_string());
            }
            "session.tool.called" => {
                let id = d["id"].as_str().unwrap_or("").to_string();
                let name = names.get(&id).cloned().unwrap_or_default();
                let arg = tool_arg(&d["input"], &ask.dir);
                if name == "subagent" && d["input"]["background"].as_bool() == Some(true) {
                    bg_calls.insert(id.clone());
                }
                if structured {
                    let tool = json!({ "id": id, "name": name, "title": tool_label(&name), "summary": arg, "input": relative(&d["input"], &ask.dir) });
                    called.insert(id.clone(), tool.clone());
                    out.tool(update(&called, &id, &name, json!({ "status": "running" })));
                } else {
                    out.emit(json!({ "reasoning_content": format!("\n▸ {} {}\n", tool_label(&name), arg) }));
                }
            }
            "session.tool.success" | "session.tool.failed" if quiet.as_deref().is_some_and(|q| d["id"].as_str() == Some(q)) => {}
            "session.tool.progress" => {
                // 派到后台的子智能体报了子会话 id：开始跟着它
                let id = d["id"].as_str().unwrap_or("").to_string();
                if let Some(child) = d["metadata"]["sessionID"].as_str().filter(|_| bg_calls.contains(&id)) {
                    adopt(&mut background, &mut followed, sid, child, &id);
                }
            }
            "session.tool.success"
                if bg_calls.contains(d["id"].as_str().unwrap_or(""))
                    || (names.get(d["id"].as_str().unwrap_or("")).map(String::as_str) == Some("subagent")
                        && full_text(&d["content"]).contains("working in the background")) =>
            {
                // 派到后台了（工具马上就「成功」，结果说它在后台跑）：那一行留着运行中，等它交回
                let id = d["id"].as_str().unwrap_or("").to_string();
                bg_calls.remove(&id);
                let text = full_text(&d["content"]);
                match d["metadata"]["sessionID"].as_str().map(str::to_string).or_else(|| child_in(&text)) {
                    Some(child) => {
                        adopt(&mut background, &mut followed, sid, &child, &id);
                        backend::log(&format!("subagent {child} 在后台跑 session={sid}"));
                        if structured {
                            out.tool(update(&called, &id, "subagent", json!({ "status": "running", "output": "在后台运行…" })));
                        } else {
                            out.emit(json!({ "reasoning_content": "  … 在后台运行\n" }));
                        }
                    }
                    None => {
                        if structured {
                            out.tool(update(&called, &id, "subagent", json!({ "status": "success", "output": cut(&text, 8000) })));
                        }
                    }
                }
            }
            "session.inbox.enqueued" => {
                // 后台子智能体交回了（父会话收到一条合成消息，送达后它接着做）：那一行标成功 / 失败，结果是它的回报
                let payload = &d["item"]["payload"];
                let meta = &payload["metadata"];
                if meta["source"].as_str() == Some("subagent") {
                    let kid = meta["childID"].as_str().unwrap_or("");
                    if let Some(b) = background.iter_mut().find(|b| b.child == kid) {
                        b.inbox = d["inboxID"].as_str().map(str::to_string);
                        let state = meta["state"].as_str().unwrap_or("");
                        let report = subagent_report(payload["text"].as_str().unwrap_or(""));
                        backend::log(&format!("subagent {kid} 交回（{state}）session={sid}"));
                        if structured && !b.call.is_empty() {
                            let status = if state == "completed" { "success" } else { "failed" };
                            out.tool(update(&called, &b.call, "subagent", json!({ "status": status, "output": cut(&report, 8000) })));
                        } else if !structured {
                            out.emit(json!({ "reasoning_content": format!("\n▸ 后台子智能体交回了（{state}）\n") }));
                        }
                        followed.bus.unsubscribe(kid);
                    }
                }
            }
            "session.inbox.delivered" => {
                if let Some(inbox) = d["inboxID"].as_str() {
                    for b in background.iter_mut().filter(|b| b.inbox.as_deref() == Some(inbox)) {
                        b.handled = true;
                    }
                }
            }
            "session.execution.started" => {
                if !out.content.is_empty() {
                    gap = true;
                }
            }
            "session.tool.success" => {
                if structured {
                    let id = d["id"].as_str().unwrap_or("").to_string();
                    let name = names.get(&id).cloned().unwrap_or_default();
                    out.tool(update(&called, &id, &name, json!({ "status": "success", "output": cut(&full_text(&d["content"]), 8000), "metadata": d["metadata"] })));
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
                    out.tool(update(&called, &id, &name, json!({ "status": "failed", "output": cut(msg, 2000) })));
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
                    let _ = oc::post(&be.conn, &format!("/api/session/{from}/permission/{request}/reply"), dir, &json!({ "decision": "reject" }));
                    continue;
                }
                // 对应的工具行标成「等确认」（不然一直显示运行中）
                let call = d["source"]["id"].as_str().unwrap_or("").to_string();
                if structured && !call.is_empty() {
                    let name = names.get(&call).cloned().unwrap_or_default();
                    out.tool(update(&called, &call, &name, json!({ "status": "waiting", "output": format!("等你确认：{}", permission_label(&action)) })));
                }
                let list = resources.iter().map(|r| cut(r, 300)).collect::<Vec<_>>().join("\n");
                if structured {
                    // 要做的事已经在等确认的工具行上了（带允许 / 拒绝按钮）：正文只说一句
                    out.emit(json!({ "content": format!(
                        "\n\n{who}想{}，等你确认：点上面的「允许执行」或回复「允许」（「总是允许」以后同类不再问）；回复别的就取消，按你新说的做。",
                        permission_label(&action)) }));
                } else {
                    out.emit(json!({ "content": format!(
                        "\n\n> **{who}想{}**\n\n```\n{}\n```\n\n回复「允许」放行这一次，回复「总是允许」以后同类操作不再问；回复别的就取消，按你新说的做。",
                        permission_label(&action), list) }));
                }
                backend::log(&format!("permission asked {request} action={action} session={from}"));
                store::set_pending(&ask.key, Some(store::Pending { request, tool: called.get(&call).cloned(), form: None, session: child.then(|| from.clone()) }));
                waiting_permission = true;
                break;
            }
            "form.created" => {
                let form = d["form"].clone();
                let form_id = form["id"].as_str().unwrap_or("").to_string();
                if ask.key.is_empty() {
                    // 一次性请求不该问：取消
                    let _ = oc::delete(&be.conn, &format!("/api/session/{from}/form/{form_id}"), dir);
                    continue;
                }
                // 提问那一行标成等回答（Friday 认 questions 时就地画成选项卡片）
                let call = form["metadata"]["tool"]["id"].as_str().unwrap_or("").to_string();
                if structured && !call.is_empty() {
                    let name = names.get(&call).cloned().filter(|n| !n.is_empty()).unwrap_or_else(|| "question".to_string());
                    // 没见过这次调用的参数（报开始在更早的请求里）：按表单补上，Friday 照它画选项卡片
                    if !called.get(&call).is_some_and(|t| t["input"]["questions"].is_array()) {
                        let tool = update(&called, &call, &name, json!({ "input": question_input(&form) }));
                        called.insert(call.clone(), tool);
                    }
                    out.tool(update(&called, &call, &name, json!({ "status": "waiting", "output": "等你回答" })));
                }
                if !(structured && ask.questions) {
                    out.emit(json!({ "content": form_text(&form).replacen("OpenCode", who, 1) }));
                } else if child {
                    out.emit(json!({ "content": "\n\nOpenCode 的后台子智能体在问你（上面的卡片）。" }));
                }
                backend::log(&format!("form asked {form_id} fields={} session={from}", form["fields"].as_array().map(Vec::len).unwrap_or(0)));
                let tool = called.get(&call).cloned().or_else(|| (!call.is_empty()).then(|| json!({ "id": call, "name": "question", "title": tool_label("question") })));
                store::set_pending(&ask.key, Some(store::Pending { request: form_id, tool, form: Some(form), session: child.then(|| from.clone()) }));
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
            "session.execution.succeeded"
                if (background.iter().any(|b| !b.handled) || !bg_calls.is_empty()) && still_waiting(be, sid, dir, &background) =>
            {
                // 这一轮完了，派到后台的子智能体还没交回：不收尾，等它交回后 opencode 接着开的那一轮
                backend::log(&format!("session {sid} 这一轮完了，还在等 {} 个后台子智能体", background.iter().filter(|b| !b.handled).count()));
                last_check = Instant::now();
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
    if !ask.key.is_empty() {
        // 停在确认 / 提问上、出错结束时还没交回的：下一次请求接着等（用来更新那一行）
        store::set_children(&ask.key, background.iter().filter(|b| !b.handled).map(|b| store::Child { session: b.child.clone(), call: b.call.clone() }).collect());
    }
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
    close(&mut out, ask, &usage.json());
    Ok(())
}

/// 这次回答收尾：流式的发结束块（带用量），非流式的回整段
fn close(out: &mut Out, ask: &Ask, usage: &Value) {
    if out.streaming {
        if out.sse.is_none() && !out.gone {
            out.emit(json!({ "content": "" }));
        }
        if let Some(sse) = out.sse.as_mut() {
            sse.data(&chunk(&ask.model, json!({}), Some("stop"), Some(usage)));
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
}

/// 还在等的提问 / 确认再交回去（这句是 Friday 自己接着做的话，不是回答）：工具行再标一次等回答 / 等确认
///（重开过的 Friday 照它重新画选项卡片 / 允许按钮），正文同第一次问时
fn reshow(ask: &Ask, p: &store::Pending, stream: &mut TcpStream, streaming: bool) {
    let mut out = Out { stream: Some(stream), sse: None, streaming, model: ask.model.clone(), content: String::new(), reasoning: String::new(), gone: false, tools: 0 };
    let structured = ask.tool_events && streaming;
    let row = p.tool.clone().filter(|_| structured);
    match &p.form {
        Some(form) => {
            if let Some(mut tool) = row.clone() {
                if !tool["input"]["questions"].is_array() {
                    tool["input"] = question_input(form);
                }
                tool["status"] = json!("waiting");
                tool["output"] = json!("等你回答");
                out.tool(tool);
            }
            if !(row.is_some() && ask.questions) {
                out.emit(json!({ "content": form_text(form) }));
            }
        }
        None => {
            if let Some(mut tool) = row.clone() {
                tool["status"] = json!("waiting");
                tool["output"] = json!("等你确认");
                out.tool(tool);
            }
            out.emit(json!({ "content": if row.is_some() {
                "\n\nOpenCode 还在等你确认上一步要做的操作：点上面的「允许执行」或回复「允许」（「总是允许」以后同类不再问）；回复别的就取消，按你新说的做。"
            } else {
                "\n\nOpenCode 还在等你确认上一步要做的操作：回复「允许」放行这一次，回复「总是允许」以后同类操作不再问；回复别的就取消，按你新说的做。"
            } }));
        }
    }
    close(&mut out, ask, &Usage::default().json());
}

/// 表单 → question 工具的参数（{questions: [{header, question, options: [{label, description}], multiple}]}）：Friday 照它画选项卡片
fn question_input(form: &Value) -> Value {
    let questions: Vec<Value> = form_fields(form)
        .iter()
        .map(|f| {
            let options: Vec<Value> = f["options"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|o| json!({ "label": o["label"].as_str().or_else(|| o["value"].as_str()).unwrap_or(""), "description": o["description"].as_str().unwrap_or("") }))
                .collect();
            json!({
                "header": f["title"].as_str().unwrap_or(""),
                "question": f["description"].as_str().filter(|s| !s.is_empty()).or_else(|| f["title"].as_str()).unwrap_or(""),
                "options": options,
                "multiple": f["type"].as_str() == Some("multiselect"),
            })
        })
        .collect();
    json!({ "questions": questions })
}

/// 拒掉确认 / 取消提问后 opencode 会停下这次执行：等它停下再发新的话（最多 5 秒）
fn wait_stopped(rx: &std::sync::mpsc::Receiver<Value>) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(e) if e["type"].as_str().map(|t| t.starts_with("session.execution.") && t != "session.execution.started").unwrap_or(false) => break,
            _ => {}
        }
    }
}

/// 这段对话在 opencode 那边有没有还在等的提问（表单）或权限确认（本程序重启过、内存里没记着时用）
fn waiting_in_opencode(be: &Backend, sid: &str, dir: Option<&str>) -> Option<store::Pending> {
    if let Some(p) = waiting_in(be, sid, dir) {
        return Some(p);
    }
    // 派到后台、还在跑的子智能体在等确认 / 回答：回复要发到子会话
    for child in running_children(be, sid, dir) {
        if let Some(mut p) = waiting_in(be, &child, dir) {
            p.session = Some(child);
            return Some(p);
        }
    }
    None
}

/// 这个会话自己有没有在等的提问（表单）或权限确认
fn waiting_in(be: &Backend, sid: &str, dir: Option<&str>) -> Option<store::Pending> {
    let pending = |item: &Value| item["state"]["status"].as_str().map(|s| s == "pending").unwrap_or(true) && item["id"].as_str().is_some();
    if let Ok((200, v)) = oc::get(&be.conn, &format!("/api/session/{sid}/form"), dir) {
        if let Some(form) = v["data"].as_array().into_iter().flatten().find(|f| pending(f)) {
            let call = form["metadata"]["tool"]["id"].as_str().unwrap_or("");
            let tool = (!call.is_empty()).then(|| json!({ "id": call, "name": "question", "title": tool_label("question"), "input": question_input(form) }));
            backend::log(&format!("session {sid} 还有没答的提问 {}", form["id"].as_str().unwrap_or("")));
            return Some(store::Pending { request: form["id"].as_str().unwrap_or("").to_string(), tool, form: Some(form.clone()), session: None });
        }
    }
    if let Ok((200, v)) = oc::get(&be.conn, &format!("/api/session/{sid}/permission"), dir) {
        if let Some(r) = v["data"].as_array().into_iter().flatten().find(|r| pending(r)) {
            backend::log(&format!("session {sid} 还有没确认的权限请求 {}", r["id"].as_str().unwrap_or("")));
            return Some(store::Pending { request: r["id"].as_str().unwrap_or("").to_string(), tool: None, form: None, session: None });
        }
    }
    None
}

/// 这次请求等着的后台子智能体（subagent 工具 background: true）：父会话那一轮先结束，子智能体做完后 OpenCode 往父会话塞一条
/// 合成消息（metadata.source = subagent、childID、state，正文是它的回报），送达后父会话自己再开一轮接着做
struct Background {
    child: String,
    /// 父会话里那次 subagent 调用（Friday 那边的工具行）；本程序重启后从 opencode 找回来的不知道，为空
    call: String,
    /// 报它交回的那条合成消息
    inbox: Option<String>,
    /// 那条消息送达了：父会话接着的那一轮在处理它
    handled: bool,
}

/// 这次请求跟着的子会话：结束时从总线上摘掉
struct Followed<'a> {
    bus: &'a oc::Bus,
    ids: Vec<String>,
}

impl Drop for Followed<'_> {
    fn drop(&mut self) {
        for id in &self.ids {
            self.bus.unsubscribe(id);
        }
    }
}

/// 开始等一个后台子智能体（已经在等的补上它那一行的 id）
fn adopt(background: &mut Vec<Background>, followed: &mut Followed, sid: &str, child: &str, call: &str) {
    if let Some(b) = background.iter_mut().find(|b| b.child == child) {
        if b.call.is_empty() {
            b.call = call.to_string();
        }
        return;
    }
    followed.bus.follow(child, sid);
    followed.ids.push(child.to_string());
    background.push(Background { child: child.to_string(), call: call.to_string(), inbox: None, handled: false });
}

/// subagent 工具的结果「…working in the background (sessionID: ses_…)…」里的子会话 id
fn child_in(text: &str) -> Option<String> {
    let at = text.find("sessionID: ")? + "sessionID: ".len();
    let id: String = text[at..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
    id.starts_with("ses_").then_some(id)
}

/// 子智能体交回的合成消息去掉外面的 <subagent …> 标签，只留它的回报
fn subagent_report(text: &str) -> String {
    let t = text.trim();
    let t = match t.find('>') {
        Some(i) if t.starts_with("<subagent") => &t[i + 1..],
        _ => t,
    };
    t.trim_end().trim_end_matches("</subagent>").trim().to_string()
}

/// opencode 里正在跑的会话（/api/session/active）；问不到 = None
fn active_sessions(be: &Backend, dir: Option<&str>) -> Option<std::collections::HashSet<String>> {
    match oc::get(&be.conn, "/api/session/active", dir) {
        Ok((200, v)) => Some(v["data"].as_object().map(|m| m.keys().cloned().collect()).unwrap_or_default()),
        _ => None,
    }
}

/// 这个会话派出去、还在跑的子会话（正在跑的 ∩ parentID 是它的）
fn running_children(be: &Backend, sid: &str, dir: Option<&str>) -> Vec<String> {
    let Some(active) = active_sessions(be, dir).filter(|a| !a.is_empty()) else { return Vec::new() };
    match oc::get(&be.conn, &format!("/api/session?parentID={sid}&limit=100"), dir) {
        Ok((200, v)) => v["data"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|s| s["id"].as_str())
            .filter(|id| active.contains(*id))
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

/// 还要不要接着等：父会话在跑、没交回的后台子智能体还在跑，或者父会话还有没送达的消息（子智能体交回的那条）；问不到就接着等
fn still_waiting(be: &Backend, sid: &str, dir: Option<&str>, background: &[Background]) -> bool {
    let Some(active) = active_sessions(be, dir) else { return true };
    if active.contains(sid) || background.iter().any(|b| !b.handled && active.contains(&b.child)) {
        return true;
    }
    match oc::get(&be.conn, &format!("/api/session/{sid}/inbox"), dir) {
        Ok((200, v)) => v["data"].as_array().is_some_and(|a| !a.is_empty()),
        _ => true,
    }
}

fn form_fields(form: &Value) -> Vec<Value> {
    form["fields"].as_array().cloned().unwrap_or_default().into_iter().filter(|f| f["type"].as_str() != Some("external")).collect()
}

/// 提问写成正文（旧版 Friday、没开结构化工具事件时）：问题 + 带序号的选项，怎么回答
fn form_text(form: &Value) -> String {
    let fields = form_fields(form);
    let mut t = String::from("\n\n**OpenCode 在问你：**\n");
    for (i, f) in fields.iter().enumerate() {
        let question = f["description"].as_str().filter(|s| !s.is_empty()).or_else(|| f["title"].as_str()).unwrap_or("");
        if fields.len() > 1 {
            t.push_str(&format!("\n{}. {question}\n", i + 1));
        } else {
            t.push_str(&format!("\n{question}\n"));
        }
        for (j, o) in f["options"].as_array().into_iter().flatten().enumerate() {
            let label = o["label"].as_str().or_else(|| o["value"].as_str()).unwrap_or("");
            match o["description"].as_str().filter(|d| !d.is_empty()) {
                Some(d) => t.push_str(&format!("   {}) {label}：{d}\n", j + 1)),
                None => t.push_str(&format!("   {}) {label}\n", j + 1)),
            }
        }
        if f["type"].as_str() == Some("multiselect") {
            t.push_str("   （可以选几个）\n");
        }
    }
    if fields.len() > 1 {
        t.push_str("\n回复选项的名字或序号，几题之间用「；」隔开；也可以直接说你的想法。");
    } else {
        t.push_str("\n回复选项的名字或序号，也可以直接说你的想法。");
    }
    t
}

/// 用户的回答 → 表单的 answer（{q0: "中文", q1: ["表格", "摘要"]}）；对不上（选项外的回答而这题不许自己写）返回 None。
/// 几题时按「；」拆成每题一段（段数对不上就整句当每一题的回答）；「（跳过）」/ 空 = 这题不答
fn form_answer(form: &Value, said: &str) -> Option<Value> {
    let fields = form_fields(form);
    let said = said.trim();
    let parts: Vec<&str> = if fields.len() > 1 {
        let parts: Vec<&str> = said.split(['；', ';', '\n']).map(str::trim).collect();
        if parts.len() == fields.len() {
            parts
        } else {
            vec![said; fields.len()]
        }
    } else {
        vec![said]
    };
    let mut answer = serde_json::Map::new();
    for (field, part) in fields.iter().zip(parts) {
        let key = field["key"].as_str()?;
        let part = part.trim().trim_end_matches(['。', '.', '！', '!']);
        if part.is_empty() || part == "（跳过）" || part == "跳过" {
            continue;
        }
        answer.insert(key.to_string(), field_value(field, part)?);
    }
    Some(Value::Object(answer))
}

fn field_value(field: &Value, text: &str) -> Option<Value> {
    let options = field["options"].as_array().cloned().unwrap_or_default();
    let custom = field["custom"].as_bool().unwrap_or(false) || options.is_empty();
    match field["type"].as_str().unwrap_or("string") {
        "multiselect" => {
            let items: Vec<&str> = text.split(['、', ',', '，']).map(str::trim).filter(|s| !s.is_empty()).collect();
            let matched: Vec<Value> = items.iter().filter_map(|i| match_option(&options, i)).map(Value::String).collect();
            if !items.is_empty() && matched.len() == items.len() {
                Some(Value::Array(matched))
            } else if let Some(one) = match_option(&options, text) {
                Some(json!([one]))
            } else if custom {
                Some(json!([text]))
            } else {
                None
            }
        }
        "boolean" => {
            let t = text.to_lowercase();
            if ["是", "对", "好", "要", "yes", "y", "true", "需要", "可以"].contains(&t.as_str()) {
                Some(Value::Bool(true))
            } else if ["否", "不", "不要", "no", "n", "false", "不需要", "不用"].contains(&t.as_str()) {
                Some(Value::Bool(false))
            } else {
                None
            }
        }
        "number" | "integer" => text.parse::<f64>().ok().and_then(serde_json::Number::from_f64).map(Value::Number),
        _ => match match_option(&options, text) {
            Some(v) => Some(Value::String(v)),
            None if custom => Some(Value::String(text.to_string())),
            None => None,
        },
    }
}

/// 一段回答对上哪个选项（返回它的 value）：名字（不分大小写，也认去掉「（推荐）」「(Recommended)」的）、或序号「2」「第二个」「选 2」
fn match_option(options: &[Value], text: &str) -> Option<String> {
    fn bare(s: &str) -> String {
        let mut s = s.trim().to_lowercase();
        for tag in ["(recommended)", "（推荐）", "(推荐)", "【推荐】"] {
            if let Some(stripped) = s.strip_suffix(tag) {
                s = stripped.trim().to_string();
            }
        }
        s
    }
    let t = bare(text);
    if t.is_empty() {
        return None;
    }
    for o in options {
        let value = o["value"].as_str().unwrap_or("");
        let label = o["label"].as_str().unwrap_or(value);
        if bare(value) == t || bare(label) == t {
            return Some(value.to_string());
        }
    }
    let n = t.trim_start_matches(['选', '第']).trim().trim_end_matches(['个', '项', '号', '.', '、', ')', '）']).trim();
    let index = match n {
        "一" => Some(1),
        "二" | "两" => Some(2),
        "三" => Some(3),
        "四" => Some(4),
        "五" => Some(5),
        "六" => Some(6),
        _ => n.parse::<usize>().ok(),
    }?;
    options.get(index.checked_sub(1)?).and_then(|o| o["value"].as_str()).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form() -> Value {
        json!({ "id": "frm_1", "fields": [
            { "key": "q0", "type": "string", "description": "报告用什么语言？", "custom": true,
              "options": [{ "value": "中文", "label": "中文" }, { "value": "英文 (Recommended)", "label": "英文 (Recommended)" }] },
            { "key": "q1", "type": "multiselect", "description": "要哪些附件？", "custom": true,
              "options": [{ "value": "表格", "label": "表格" }, { "value": "图表", "label": "图表" }, { "value": "摘要", "label": "摘要" }] }
        ]})
    }

    #[test]
    fn answers_each_question_by_label_or_number() {
        assert_eq!(form_answer(&form(), "中文；表格、摘要"), Some(json!({ "q0": "中文", "q1": ["表格", "摘要"] })));
        assert_eq!(form_answer(&form(), "2；第一个"), Some(json!({ "q0": "英文 (Recommended)", "q1": ["表格"] })));
        assert_eq!(form_answer(&form(), "英文；1, 3"), Some(json!({ "q0": "英文 (Recommended)", "q1": ["表格", "摘要"] })));
    }

    #[test]
    fn free_reply_goes_to_every_question() {
        let a = form_answer(&form(), "你看着办，越详细越好").unwrap();
        assert_eq!(a, json!({ "q0": "你看着办，越详细越好", "q1": ["你看着办，越详细越好"] }));
        // 跳过的题不答
        assert_eq!(form_answer(&form(), "中文；（跳过）"), Some(json!({ "q0": "中文" })));
    }

    #[test]
    fn background_subagent_texts() {
        let text = "The subagent is working in the background (sessionID: ses_e31668e14ffeDczMI20uo39aQg). You will be notified automatically when it finishes.";
        assert_eq!(child_in(text).as_deref(), Some("ses_e31668e14ffeDczMI20uo39aQg"));
        assert_eq!(child_in("Task done"), None);
        let report = "<subagent sessionID=\"ses_x\" state=\"completed\" description=\"写 a.txt\">\n任务已完成。\n\n**文件内容：** hello\n</subagent>";
        assert_eq!(subagent_report(report), "任务已完成。\n\n**文件内容：** hello");
        assert_eq!(subagent_report("没有标签的回报"), "没有标签的回报");
    }

    #[test]
    fn question_input_from_form() {
        // 重开过的 Friday 照它重新画选项卡片：问题、选项、多选都在
        let f = json!({ "fields": [
            { "key": "q0", "title": "语言", "description": "报告用什么语言？", "type": "string", "options": [{ "value": "zh", "label": "中文", "description": "默认" }, { "value": "en", "label": "英文" }] },
            { "key": "q1", "title": "附件", "description": "", "type": "multiselect", "options": [{ "value": "表格", "label": "表格" }] },
            { "key": "x", "type": "external" },
        ] });
        let input = question_input(&f);
        assert_eq!(input["questions"].as_array().map(Vec::len), Some(2));
        assert_eq!(input["questions"][0]["question"], "报告用什么语言？");
        assert_eq!(input["questions"][0]["options"][0], json!({ "label": "中文", "description": "默认" }));
        assert_eq!(input["questions"][0]["multiple"], false);
        assert_eq!((input["questions"][1]["question"].clone(), input["questions"][1]["multiple"].clone()), (json!("附件"), json!(true)));
    }

    #[test]
    fn continuing_flag_parsed() {
        let body = json!({ "model": "opencode/x", "messages": [{ "role": "user", "content": "（上一步因出错中断：超时。）" }], "x_friday": { "session": "s", "continuing": true } });
        assert!(parse(&body).unwrap().continuing);
        let body = json!({ "model": "opencode/x", "messages": [{ "role": "user", "content": "中文" }], "x_friday": { "session": "s" } });
        assert!(!parse(&body).unwrap().continuing);
    }

    #[test]
    fn closed_question_without_match_is_none() {
        let f = json!({ "fields": [{ "key": "q0", "type": "string", "options": [{ "value": "a", "label": "A" }], "custom": false }] });
        assert_eq!(form_answer(&f, "a"), Some(json!({ "q0": "a" })));
        assert_eq!(form_answer(&f, "别的"), None);
    }

    #[test]
    fn form_text_lists_options() {
        let t = form_text(&form());
        assert!(t.contains("1. 报告用什么语言？") && t.contains("2) 英文 (Recommended)") && t.contains("（可以选几个）") && t.contains("「；」"));
    }

    #[test]
    fn own_words_drops_sent_time() {
        assert_eq!(own_words("中文；表格\n\n（发出时间：2026年11月12日 11:17）"), "中文；表格");
        assert_eq!(approval("允许\n\n（发出时间：…）"), Some("once"));
    }
}
