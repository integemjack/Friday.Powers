//! POST /v1/chat/completions：把 Friday 的一句话交给用户自己的 Claude Code（它自己的系统提示、工具、登录、额度）在对话的工作目录里做，
//! 内容实时返回：文字 → content，思考（有字的话）→ reasoning_content，工具活动 → x_friday.tool（或思考区里的一行行），最后 usage。
//! 本程序不加任何系统提示，只传用户的话（新开的 Claude Code 会话带上这段对话之前的内容）。
//!
//! 请求里 Friday 的扩展字段 x_friday：
//!   session     这段对话（或子智能体）的标识：同一个值 → 同一个常驻 claude 进程 / 同一个 Claude Code 会话；没有 = 一次性的辅助请求
//!   cwd         工作目录（claude 在这里读写、跑命令）；没有就用本程序的数据目录下的 work
//!   permission  Friday 会话框下的权限档：auto（默认：改工作目录里的文件不问，跑命令等先问）| plan（只读）| full（都不问）
//!   toolEvents  Friday 认结构化的工具事件（x_friday.tool）
//!   questions   Friday 把等回答的提问行（AskUserQuestion）画成选项卡片（v2.97.25 起）：提问时正文不再列一遍问题
//!   continuing  这句是 Friday 自己接着做的话（出错后自动恢复、点了重试、换了模型接着做），不是用户说的：
//!               还在等的提问 / 确认不拿它当回答，原样再交回去（工具行再标一次等回答，这次到此为止）
//! 要用户确认的操作：这次回答以一句说明结束，claude 那边停着等；用户下一句说「允许」/「总是允许」就放行并接着把后面的内容流回来，
//! 说别的就拒绝、再把这句话当成新的一句交给它。

use std::net::TcpStream;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::backend;
use crate::conv::{self, Conv, Pending, Proc, Task};
use crate::http::{self, Request, Sse};
use crate::store;

struct Ask {
    /// 请求里的模型 id：default（不传 --model）/ fable / opus / sonnet / haiku（也认 Claude Code 认的完整模型名）
    model: String,
    /// 这段对话的 key（session + 工作目录的哈希）；一次性请求为空
    key: String,
    dir: String,
    /// auto | plan | full
    permission: String,
    /// 发给 claude 的话（一次性请求：Friday 的要求、前文和这句拼在一起）
    text: String,
    /// 用户这句话本身（判断是不是在回答权限确认）
    said: String,
    /// Anthropic 的 image 内容块
    images: Vec<Value>,
    /// 这段对话之前的内容（新开 Claude Code 会话时带上）
    history: String,
    tool_events: bool,
    /// Friday 把等回答的提问行画成选项卡片（x_friday.questions）
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

/// OpenAI 的 image_url（data: 地址或网址）→ Anthropic 的 image 块
fn images_of(content: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    if let Value::Array(parts) = content {
        for p in parts {
            if p["type"].as_str() != Some("image_url") {
                continue;
            }
            let url = p["image_url"].as_str().or_else(|| p["image_url"]["url"].as_str()).unwrap_or("");
            if let Some(rest) = url.strip_prefix("data:") {
                let Some((meta, data)) = rest.split_once(',') else { continue };
                let media = meta.split(';').next().filter(|m| m.starts_with("image/")).unwrap_or("image/png");
                out.push(json!({ "type": "image", "source": { "type": "base64", "media_type": media, "data": data } }));
            } else if url.starts_with("http://") || url.starts_with("https://") {
                out.push(json!({ "type": "image", "source": { "type": "url", "url": url } }));
            }
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
    let model = body["model"].as_str().unwrap_or("").trim();
    let model = if model.is_empty() { "default".to_string() } else { model.strip_prefix("claudecode/").unwrap_or(model).to_string() };
    let xf = &body["x_friday"];
    let default_dir = backend::data_dir().join("work");
    let _ = std::fs::create_dir_all(&default_dir);
    let dir = ["cwd", "workingDirectory", "directory"]
        .iter()
        .filter_map(|k| xf[*k].as_str())
        .find(|d| Path::new(d).is_absolute() && Path::new(d).is_dir())
        .map(str::to_string)
        .unwrap_or_else(|| default_dir.to_string_lossy().to_string());
    let session = xf["session"].as_str().unwrap_or("").to_string();
    let permission = match xf["permission"].as_str().unwrap_or("auto") {
        "full" | "bypass" | "bypassPermissions" => "full",
        "plan" => "plan",
        _ => "auto",
    }
    .to_string();
    let tool_events = xf["toolEvents"].as_bool().unwrap_or(false);
    let questions = xf["questions"].as_bool().unwrap_or(false);
    let continuing = xf["continuing"].as_bool().unwrap_or(false);
    let messages: Vec<Value> = body["messages"].as_array().cloned().unwrap_or_default();
    let Some(last_user) = messages.iter().rposition(|m| m["role"].as_str() == Some("user")) else {
        return Err("messages 里没有用户的话".into());
    };
    let said = text_of(&messages[last_user]["content"]);
    let images = images_of(&messages[last_user]["content"]);
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
        // 一次性的辅助请求（起标题、总结……）：Friday 的要求和前文都写进这一句里，让 Claude Code 直接回文字（工具也关了）
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
    if text.trim().is_empty() && images.is_empty() {
        return Err("最后一句用户的话是空的".into());
    }
    Ok(Ask { model, key, dir, permission, text, said, images, history, tool_events, questions, continuing })
}

fn content(text: &str, images: &[Value]) -> Value {
    let mut parts = Vec::new();
    if !text.is_empty() {
        parts.push(json!({ "type": "text", "text": text }));
    }
    parts.extend(images.iter().cloned());
    Value::Array(parts)
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

/// Claude Code 的工具名 → 工具行上的中文名
fn title(name: &str) -> String {
    let t = match name {
        "Read" => "读取文件",
        "Write" => "写入文件",
        "Edit" | "MultiEdit" => "修改文件",
        "NotebookEdit" => "修改笔记本",
        "NotebookRead" => "读取笔记本",
        "Bash" | "PowerShell" => "运行命令",
        "BashOutput" | "TaskOutput" => "查看输出",
        "KillShell" | "KillBash" | "TaskStop" => "停止任务",
        "Grep" => "搜索内容",
        "Glob" => "查找文件",
        "LS" => "列目录",
        "WebFetch" => "打开网页",
        "WebSearch" => "搜索网页",
        "Task" | "Agent" => "子智能体",
        "TodoWrite" | "TaskCreate" | "TaskUpdate" | "TaskList" | "TaskGet" => "待办",
        "Skill" => "用技能",
        "AskUserQuestion" => "提问",
        "ExitPlanMode" => "计划",
        "EnterPlanMode" => "进入计划",
        "SendMessage" => "发消息",
        "Monitor" => "监视",
        "Workflow" => "工作流",
        "CronCreate" | "CronDelete" | "CronList" => "定时任务",
        "ScheduleWakeup" => "定时唤醒",
        "EnterWorktree" | "ExitWorktree" => "工作树",
        "ListMcpResourcesTool" | "ReadMcpResourceTool" => "MCP 资源",
        other => return other.strip_prefix("mcp__").map(|r| r.rsplit("__").next().unwrap_or(r).to_string()).unwrap_or_else(|| other.to_string()),
    };
    t.to_string()
}

/// 权限确认那句话里说「Claude Code 想……」
pub fn permission_label(name: &str) -> String {
    match name {
        "Bash" | "PowerShell" => "运行命令".into(),
        "Edit" | "Write" | "MultiEdit" | "NotebookEdit" => "修改工作目录以外的文件".into(),
        "WebFetch" => "打开网页".into(),
        "WebSearch" => "搜索网页".into(),
        "Task" | "Agent" => "派子智能体".into(),
        "Skill" => "用技能".into(),
        "KillShell" | "KillBash" | "TaskStop" => "停止任务".into(),
        other => match other.strip_prefix("mcp__") {
            Some(r) => format!("用工具 {}", r.replace("__", " · ")),
            None => format!("用 {other} 工具"),
        },
    }
}

/// 工作目录下的路径写成相对路径
fn rel(s: &str, dir: &str) -> String {
    let prefix = format!("{}/", dir.trim_end_matches(['/', '\\']));
    let prefix_win = format!("{}\\", dir.trim_end_matches(['/', '\\']));
    match s.strip_prefix(&prefix).or_else(|| s.strip_prefix(&prefix_win)) {
        Some(rest) if !rest.is_empty() => rest.to_string(),
        _ => s.to_string(),
    }
}

/// 参数里工作目录下的路径写成相对路径（工具行上短一些）
fn relative(value: &Value, dir: &str) -> Value {
    match value {
        Value::String(s) => Value::String(rel(s, dir)),
        Value::Object(map) => Value::Object(map.iter().map(|(k, v)| (k.clone(), relative(v, dir))).collect()),
        Value::Array(items) => Value::Array(items.iter().map(|v| relative(v, dir)).collect()),
        _ => value.clone(),
    }
}

/// 工具行上的一句：文件路径（相对工作目录）、命令、搜索的模式、网址、子智能体的描述……
fn summary(name: &str, input: &Value, dir: &str) -> String {
    let s = |k: &str| input[k].as_str().unwrap_or("").to_string();
    let first = |t: String| t.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("").to_string();
    let text = match name {
        "Read" | "Write" | "Edit" | "MultiEdit" | "NotebookRead" => rel(&s("file_path"), dir),
        "NotebookEdit" => rel(&s("notebook_path"), dir),
        "Bash" | "PowerShell" => first(s("command")),
        "Grep" => {
            let path = s("path");
            if path.is_empty() {
                s("pattern")
            } else {
                format!("{} · {}", s("pattern"), rel(&path, dir))
            }
        }
        "Glob" => s("pattern"),
        "LS" => rel(&s("path"), dir),
        "WebFetch" => s("url"),
        "WebSearch" => s("query"),
        "Task" | "Agent" => s("description"),
        "TodoWrite" => input["todos"].as_array().map(|t| format!("{} 项", t.len())).unwrap_or_default(),
        "TaskCreate" | "TaskUpdate" => {
            let subject = s("subject");
            if subject.is_empty() {
                s("taskId")
            } else {
                subject
            }
        }
        "Skill" => s("skill"),
        "AskUserQuestion" => input["questions"][0]["question"].as_str().unwrap_or("").to_string(),
        _ => String::new(),
    };
    if !text.is_empty() {
        return cut(&text, 160);
    }
    for k in ["command", "file_path", "path", "pattern", "url", "query", "description", "prompt"] {
        if let Some(v) = input[k].as_str() {
            return cut(&rel(v, dir), 160);
        }
    }
    if input.is_null() || input.as_object().map(|o| o.is_empty()).unwrap_or(false) {
        String::new()
    } else {
        cut(&input.to_string(), 160)
    }
}

/// tool_result 的 content：字符串，或 [{type:text,text}…]
fn result_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p["type"].as_str() {
                Some("text") => p["text"].as_str().map(str::to_string),
                Some("image") => Some("[图片]".to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn first_line(text: &str) -> String {
    cut(text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or(""), 100)
}

fn chunk(model: &str, delta: Value, finish: Option<&str>, usage: Option<&Value>) -> Value {
    let mut c = json!({
        "id": "chatcmpl-claudecode",
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
    /// 一次 API 调用的用量 → 现在的上下文有多长
    fn context(&mut self, u: &Value) {
        let input = u["input_tokens"].as_u64().unwrap_or(0);
        let read = u["cache_read_input_tokens"].as_u64().unwrap_or(0);
        let write = u["cache_creation_input_tokens"].as_u64().unwrap_or(0);
        if input + read + write > 0 {
            self.prompt = input + read + write;
            self.cached = read;
        }
    }

    /// 一轮的 result：prompt 取最后一次调用的（iterations 的最后一项），输出各轮相加
    fn result(&mut self, event: &Value) {
        let u = &event["usage"];
        self.completion += u["output_tokens"].as_u64().unwrap_or(0);
        self.reasoning += u["output_tokens_details"]["thinking_tokens"].as_u64().unwrap_or(0);
        match u["iterations"].as_array().and_then(|a| a.last()) {
            Some(last) => self.context(last),
            None if self.prompt == 0 => self.context(u),
            None => {}
        }
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
    retry_after_secs: u64,
}

impl Failure {
    fn new(status: u16, kind: &str, message: impl Into<String>) -> Failure {
        Failure { status, kind: kind.into(), message: message.into(), retry_after_secs: 0 }
    }
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
    /// 最近一次往 Friday 写的时候（心跳按它算：子智能体在跑时 claude 一直有输出，但不一定有要发给 Friday 的）
    wrote: Instant,
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
        self.wrote = Instant::now();
    }

    /// 一条工具事件（只在流式、Friday 认的时候发）：chunk 顶层 x_friday.tool，delta 为空
    fn tool(&mut self, tool: Value) {
        if !self.streaming || self.gone {
            return;
        }
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
        self.wrote = Instant::now();
    }

    /// 15 秒没往 Friday 写过：发一块空的数据（不是注释行）保活、发现客户端走了（EDITH / Friday 的读流按数据块算超时）
    fn keepalive(&mut self) {
        if self.wrote.elapsed() < Duration::from_secs(15) {
            return;
        }
        if let Some(sse) = self.sse.as_mut() {
            if !sse.data(&chunk(&self.model, json!({}), None, None)) {
                self.gone = true;
            }
        }
        self.wrote = Instant::now();
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
        ACTIVE.fetch_add(1, Ordering::SeqCst);
        ActiveGuard
    }
}
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE.fetch_sub(1, Ordering::SeqCst);
    }
}

/// 一次请求流下来的结果
#[derive(Default)]
struct Pumped {
    usage: Usage,
    failure: Option<Failure>,
    /// 这次请求里最近的额度信息
    rate: Option<Value>,
    /// --resume 起的进程还没 init 就退了（会话找不到了）：什么都还没写，换新会话重来
    resume_failed: bool,
}

impl Pumped {
    fn failed(status: u16, kind: &str, message: impl Into<String>) -> Pumped {
        Pumped { failure: Some(Failure::new(status, kind, message)), ..Default::default() }
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
    let Some(bin) = backend::find_binary() else {
        let _ = http::error(stream, 503, "claude_not_installed", backend::NOT_FOUND, &[]);
        return;
    };
    let streaming = body["stream"].as_bool().unwrap_or(false);
    let mut out = Out {
        stream: Some(stream),
        sse: None,
        streaming,
        model: ask.model.clone(),
        content: String::new(),
        reasoning: String::new(),
        gone: false,
        tools: 0,
        wrote: Instant::now(),
    };
    let result = if ask.key.is_empty() { oneshot(&bin, &ask, &mut out) } else { conversation(&bin, &ask, &mut out) };
    finish(&mut out, &ask, result);
}

/// 一次性的辅助请求：起一个不留会话、没有工具的 claude，答完就退
fn oneshot(bin: &Path, ask: &Ask, out: &mut Out) -> Pumped {
    let conv = Conv::new("oneshot", &ask.dir, true);
    let mut p = conv.proc.lock().unwrap();
    if let Err(e) = conv.spawn(&mut p, bin, &ask.model, "default", None) {
        return Pumped::failed(502, "claude_error", e);
    }
    backend::log(&format!("oneshot model={} said={:?}", ask.model, cut(&ask.said, 40)));
    if let Err(e) = conv.write(&mut p, content(&ask.text, &ask.images)) {
        conv.stop(&mut p, "");
        return Pumped::failed(502, "claude_error", e);
    }
    conv.close_input(&mut p);
    let r = pump(&conv, &mut p, ask, out, 0);
    conv.stop(&mut p, "");
    r
}

/// 没在跑 / 模型或权限档变了：（重新）拉起，能接回的用 --resume
fn ensure(conv: &Conv, p: &mut Proc, bin: &Path, ask: &Ask) -> Result<(), String> {
    let mode = conv::claude_mode(&ask.permission);
    if conv.alive(p) && (p.model != ask.model || p.mode != mode) {
        let why = format!("换了模型 / 权限档（{} {} → {} {}），重开", p.model, p.mode, ask.model, mode);
        conv.stop(p, &why);
    }
    if !conv.alive(p) {
        conv.stop(p, "");
        let resume = if !p.session.is_empty() {
            Some(p.session.clone())
        } else {
            store::get(&ask.key).filter(|e| e.dir == ask.dir).map(|e| e.session)
        };
        conv.spawn(p, bin, &ask.model, mode, resume.as_deref())?;
    }
    Ok(())
}

/// 上一个请求结束以后攒下的输出（后台命令做完后 Claude Code 自己开的一轮之类）：只记账、不发给 Friday
fn drain(conv: &Conv, p: &mut Proc) {
    let mut n = 0;
    while let Some(Ok(event)) = p.rx.as_ref().map(|rx| rx.try_recv()) {
        n += 1;
        let parent = event["parent_tool_use_id"].as_str().map(|s| !s.is_empty()).unwrap_or(false);
        match event["type"].as_str().unwrap_or("") {
            "friday_permission" => p.deferred.push_back(event),
            "friday_exit" => {
                conv.stop(p, "claude 在两次请求之间退出了");
                break;
            }
            "system" => match event["subtype"].as_str().unwrap_or("") {
                "init" => {
                    note_session(conv, p, &event);
                    p.in_turn = true;
                }
                "status" if event["status"].as_str() == Some("requesting") => p.in_turn = true,
                "task_started" => task_started(p, &event),
                "task_notification" => {
                    if let Some(id) = event["task_id"].as_str() {
                        p.tasks.remove(id);
                    }
                }
                _ => {}
            },
            "stream_event" | "assistant" if !parent => p.in_turn = true,
            "result" => {
                p.unanswered = p.unanswered.saturating_sub(1);
                p.in_turn = false;
            }
            "rate_limit_event" => conv::set_rate_limit(&event["rate_limit_info"]),
            "command_lifecycle" => lifecycle(p, &event),
            _ => {}
        }
    }
    // （老版本没有 command_lifecycle 时）正在跑 Claude Code 自己开的一轮：这句话排在它后面，要多等一个 result
    if !p.lifecycle && p.in_turn && p.unanswered == 0 {
        p.unanswered = 1;
    }
    if n > 0 {
        backend::log(&format!("drain key={} 丢掉上次之后攒下的 {n} 条输出", conv::short(&conv.key)));
    }
}

fn note_session(conv: &Conv, p: &mut Proc, event: &Value) {
    p.started = true;
    if let Some(sid) = event["session_id"].as_str().filter(|s| !s.is_empty()) {
        if p.session != sid {
            p.session = sid.to_string();
        }
        if !conv.oneshot {
            store::put(&conv.key, sid, &conv.dir, &p.model);
        }
    }
}

/// command_lifecycle：写进去的那句话 queued → started → completed（并进别的一轮的也报 completed）
fn lifecycle(p: &mut Proc, event: &Value) {
    p.lifecycle = true;
    if !matches!(event["state"].as_str(), Some("queued") | Some("started")) {
        if let Some(id) = event["command_uuid"].as_str() {
            p.open.remove(id);
        }
    }
}

fn task_started(p: &mut Proc, event: &Value) {
    let Some(id) = event["task_id"].as_str() else { return };
    let kind = event["task_type"].as_str().unwrap_or("");
    let background = event["is_backgrounded"].as_bool().unwrap_or(false);
    let agent = kind.contains("agent") || kind.contains("workflow");
    p.tasks.insert(
        id.to_string(),
        Task { tool_use_id: event["tool_use_id"].as_str().unwrap_or("").to_string(), background, wait: !background || agent, agent },
    );
}

fn conversation(bin: &Path, ask: &Ask, out: &mut Out) -> Pumped {
    let conv = conv::get_or_create(&ask.key, &ask.dir);
    let my_turn = conv.turn.fetch_add(1, Ordering::SeqCst) + 1;
    conv.touch();
    // 同一段对话上一个请求还在流：它看到 turn 变了会停下让位
    let mut p = conv.proc.lock().unwrap();
    conv.touch();
    *conv.permission.lock().unwrap() = ask.permission.clone();
    let asked = conv.take_asked();
    backend::log(&format!(
        "chat key={} model={} permission={} pending={} said={:?}",
        conv::short(&ask.key),
        ask.model,
        ask.permission,
        asked.as_ref().map(|a| a.tool.as_str()).unwrap_or("-"),
        cut(&ask.said, 40)
    ));
    let r = (|| {
        if let Some(pending) = asked {
            if conv.alive(&mut p) {
                if ask.continuing {
                    // Friday 自己接着做的话（出错后自动恢复、点了重试）不是回答：还在等的提问 / 确认原样再交回去，这次到此为止
                    backend::log(&format!("key={} 还在等 {} {}，这句是 Friday 接着做的话：再交回去", conv::short(&ask.key), pending.id, pending.tool));
                    let structured = ask.tool_events && out.streaming;
                    show_ask(&mut p, ask, out, structured, &pending.id, &pending.tool, &pending.input);
                    conv.pending.lock().unwrap().insert(0, pending);
                    return Pumped::default();
                }
                if let Err(e) = answer(&conv, &mut p, ask, pending) {
                    return Pumped::failed(502, "claude_error", e);
                }
                return pump(&conv, &mut p, ask, out, my_turn);
            }
        }
        for attempt in 0..2 {
            drain(&conv, &mut p);
            if let Err(e) = ensure(&conv, &mut p, bin, ask) {
                return Pumped::failed(502, "claude_error", e);
            }
            // 全新的 Claude Code 会话：带上这段对话之前的内容（在别的模型上聊的）
            let fresh = !p.resumed && p.session.is_empty();
            let text = if fresh && !ask.history.trim().is_empty() {
                format!("下面是这段对话之前的内容（供参考）：\n<之前的对话>\n{}\n</之前的对话>\n\n现在用户说：\n{}", ask.history.trim(), ask.text)
            } else {
                ask.text.clone()
            };
            if let Err(e) = conv.write(&mut p, content(&text, &ask.images)) {
                if attempt == 0 {
                    conv.stop(&mut p, &e);
                    continue;
                }
                return Pumped::failed(502, "claude_error", e);
            }
            let r = pump(&conv, &mut p, ask, out, my_turn);
            if r.resume_failed && attempt == 0 {
                backend::log(&format!("resume key={} 会话 {} 接不回，开新会话", conv::short(&ask.key), p.session));
                store::remove(&ask.key);
                p.session.clear();
                continue;
            }
            return r;
        }
        Pumped::failed(502, "claude_error", "Claude Code 起不来")
    })();
    conv.touch();
    r
}

/// 上一次回答停在权限确认上：这句是「允许」就放行，否则拒掉（把这句话当成新的一句交给它）
fn answer(conv: &Conv, p: &mut Proc, ask: &Ask, pending: Pending) -> Result<(), String> {
    let words = own_words(&ask.said).to_string();
    if pending.tool == "AskUserQuestion" {
        // 放行并带上回答（同 Agent SDK 的做法：updatedInput.answers = {问题: 选的选项名}），它的提问工具照常成功
        let mut input = if pending.input.is_object() { pending.input.clone() } else { json!({}) };
        input["answers"] = question_answers(&pending.input, &words);
        backend::log(&format!("question {} → {}", pending.id, input["answers"]));
        // 它做完不再报（Friday 那一行已经记着「回答：…」，再报会盖掉；Friday 重开过的还会多出一行）
        p.tools.remove(&pending.id);
        let _ = pending.reply.send(conv::allow_json(&input));
        return Ok(());
    }
    match approval(&ask.said) {
        Some(kind) => {
            if kind == "always" {
                conv.allow.lock().unwrap().insert(pending.tool.clone());
                // 同类的、也在等的一起放行
                let mut all = conv.pending.lock().unwrap();
                let (same, rest): (Vec<Pending>, Vec<Pending>) = all.drain(..).partition(|x| x.tool == pending.tool);
                *all = rest;
                for x in same {
                    let _ = x.reply.send(conv::allow_json(&x.input));
                }
            }
            backend::log(&format!("permission {} {} → allow（{kind}）", pending.id, pending.tool));
            let _ = pending.reply.send(conv::allow_json(&pending.input));
            Ok(())
        }
        None => {
            let deny = conv::deny_json(&format!("用户没有同意：{words}"));
            let others: Vec<Pending> = std::mem::take(&mut *conv.pending.lock().unwrap());
            for x in others {
                let _ = x.reply.send(deny.clone());
            }
            backend::log(&format!("permission {} {} → deny，再把这句话交给它", pending.id, pending.tool));
            let _ = pending.reply.send(deny);
            conv.write(p, content(&ask.text, &ask.images))
        }
    }
}

/// 收 claude 的输出、翻成 OpenAI 的流，直到这次请求该结束：写进去的每句话都有了 result、要等的后台任务都做完了、
/// 最后见到的是 result（后台任务做完后 Claude Code 会自己再开一轮，等它）；或者停在权限确认上。
fn pump(conv: &Conv, p: &mut Proc, ask: &Ask, out: &mut Out, my_turn: u64) -> Pumped {
    let structured = ask.tool_events && out.streaming;
    let mut r = Pumped::default();
    // result 之后还等 Claude Code 自己开的一轮（后台任务做完了）等到几点
    let mut expect_until: Option<Instant> = None;
    // 这一轮里有后台任务做完了（result 之后多半还有一轮）
    let mut notified = false;
    // 这一轮主智能体流过东西（调过模型）；没有的话 result 里的文字补上（本地的 / 命令之类不经模型、不流式出字）
    let mut turn_streamed = false;
    let mut last_event = Instant::now();
    loop {
        out.keepalive();
        if out.gone {
            conv.stop(p, "Friday 断开了（点了停止）");
            return r;
        }
        if Conv::answered(p) && !p.in_turn && !p.tasks.values().any(|t| t.wait) && expect_until.map(|t| Instant::now() >= t).unwrap_or(true) {
            return r;
        }
        if my_turn != 0 && conv.turn.load(Ordering::SeqCst) != my_turn {
            conv.stop(p, "同一段对话来了新的请求，停下让位");
            return r;
        }
        let event = match p.deferred.pop_front() {
            Some(e) => e,
            None => {
                let Some(rx) = p.rx.as_ref() else {
                    r.failure = Some(Failure::new(502, "claude_error", "Claude Code 进程没在跑"));
                    return r;
                };
                match rx.recv_timeout(Duration::from_secs(1)) {
                    Ok(e) => e,
                    Err(RecvTimeoutError::Timeout) => {
                        if last_event.elapsed() > Duration::from_secs(900) {
                            conv.stop(p, "15 分钟没有动静，停下");
                            r.failure = Some(Failure::new(504, "timeout", "Claude Code 15 分钟没有动静，已停下"));
                            return r;
                        }
                        continue;
                    }
                    Err(RecvTimeoutError::Disconnected) => json!({ "type": "friday_exit" }),
                }
            }
        };
        last_event = Instant::now();
        let parent = event["parent_tool_use_id"].as_str().filter(|s| !s.is_empty()).map(str::to_string);
        match event["type"].as_str().unwrap_or("") {
            "friday_exit" => {
                let resume_failed = p.resumed && !p.started && !out.started();
                let busy = !Conv::answered(p) || p.in_turn || p.tasks.values().any(|t| t.wait);
                let tail = conv.stderr_tail(p);
                conv.stop(p, "claude 退出了");
                if resume_failed {
                    r.resume_failed = true;
                } else if busy {
                    let mut msg = "Claude Code 意外退出了".to_string();
                    if !tail.trim().is_empty() {
                        msg.push_str(&format!("：{}", cut(tail.trim(), 600)));
                    }
                    r.failure = Some(Failure::new(502, "claude_exited", msg));
                }
                return r;
            }
            "friday_permission" => {
                if ask_user(conv, p, ask, out, structured, &event) {
                    return r;
                }
            }
            "system" => match event["subtype"].as_str().unwrap_or("") {
                "init" => {
                    note_session(conv, p, &event);
                    p.in_turn = true;
                    expect_until = None;
                }
                "status" if event["status"].as_str() == Some("requesting") => {
                    p.in_turn = true;
                    expect_until = None;
                }
                "task_started" => task_started(p, &event),
                "task_notification" => {
                    let id = event["task_id"].as_str().unwrap_or("");
                    let task = p.tasks.remove(id);
                    let background = task.as_ref().map(|t| t.background).unwrap_or(true);
                    if background {
                        if p.in_turn {
                            notified = true;
                        } else {
                            // result 之后才做完：Claude Code 会自己再开一轮（10 秒内没开就不等了）
                            expect_until = Some(Instant::now() + Duration::from_secs(10));
                        }
                    }
                    let tool_use_id = task.as_ref().map(|t| t.tool_use_id.clone()).filter(|s| !s.is_empty()).or_else(|| event["tool_use_id"].as_str().map(str::to_string));
                    let agent = task.as_ref().map(|t| t.agent).unwrap_or(false);
                    if let Some(tid) = tool_use_id.filter(|_| agent) {
                        // 后台子智能体做完了：它那行这时才算做完
                        let ok = event["status"].as_str() == Some("completed");
                        let summary = event["summary"].as_str().unwrap_or("");
                        let base = p.tools.get(&tid).cloned().unwrap_or_else(|| json!({ "id": tid, "name": "Agent", "title": title("Agent") }));
                        if structured {
                            out.tool(with(&base, json!({ "status": if ok { "success" } else { "failed" }, "output": cut(summary, 8000) })));
                        } else {
                            out.emit(json!({ "reasoning_content": format!("  {} 子智能体{}\n", if ok { "✓" } else { "✗" }, if ok { "做完了" } else { "没做完" }) }));
                        }
                    }
                }
                "api_retry" => {
                    backend::log(&format!("api_retry key={} {}", conv::short(&conv.key), cut(&event.to_string(), 300)));
                    // 重试救不回来的（没登录 / 登录过期、订阅额度用完）：Claude Code 自己会退避重试 10 次（要好几分钟），
                    // 不等：停下（下一句 --resume 接回），直接告诉 Friday（还没写东西就回 401 / 429）
                    let status = event["error_status"].as_u64().unwrap_or(0);
                    let rate = r.rate.clone().or_else(conv::rate_limit);
                    let rejected = rate.as_ref().map(|x| x["status"].as_str() == Some("rejected")).unwrap_or(false);
                    let fatal = if status == 401 || event["error"].as_str() == Some("authentication_failed") {
                        Some(classify(&json!({ "api_error_status": 401 }), event["error"].as_str().unwrap_or("authentication_failed"), None))
                    } else if status == 429 && rejected {
                        Some(classify(&json!({ "api_error_status": 429 }), "", rate.as_ref()))
                    } else {
                        None
                    };
                    if let Some(f) = fatal {
                        conv.stop(p, &format!("重试救不回来（{}），停下", f.kind));
                        r.failure = Some(f);
                        return r;
                    }
                }
                _ => {}
            },
            "stream_event" => {
                // 子智能体的文字不显示（它的工具活动按工具行显示）
                if parent.is_some() {
                    continue;
                }
                p.in_turn = true;
                expect_until = None;
                turn_streamed = true;
                let e = &event["event"];
                if e["type"].as_str() == Some("content_block_delta") {
                    let d = &e["delta"];
                    match d["type"].as_str().unwrap_or("") {
                        "text_delta" => {
                            let t = d["text"].as_str().unwrap_or("");
                            if !t.is_empty() {
                                out.emit(json!({ "content": t }));
                            }
                        }
                        "thinking_delta" => {
                            let t = d["thinking"].as_str().unwrap_or("");
                            if !t.is_empty() {
                                out.emit(json!({ "reasoning_content": t }));
                            }
                        }
                        _ => {}
                    }
                }
            }
            "assistant" => {
                if parent.is_none() {
                    p.in_turn = true;
                    expect_until = None;
                    r.usage.context(&event["message"]["usage"]);
                }
                for block in event["message"]["content"].as_array().into_iter().flatten() {
                    if block["type"].as_str() == Some("tool_use") {
                        tool_started(p, ask, out, structured, block, parent.as_deref());
                    }
                }
            }
            "user" => {
                for block in event["message"]["content"].as_array().into_iter().flatten() {
                    if block["type"].as_str() == Some("tool_result") {
                        tool_finished(p, out, structured, block);
                    }
                }
            }
            "command_lifecycle" => lifecycle(p, &event),
            "rate_limit_event" => {
                let info = &event["rate_limit_info"];
                conv::set_rate_limit(info);
                if info.is_object() {
                    r.rate = Some(info.clone());
                }
            }
            "result" => {
                p.unanswered = p.unanswered.saturating_sub(1);
                p.in_turn = false;
                r.usage.result(&event);
                let text = event["result"].as_str().map(str::to_string).unwrap_or_else(|| {
                    let errors: Vec<String> = event["errors"].as_array().into_iter().flatten().filter_map(|e| e.as_str().map(str::to_string)).collect();
                    if errors.is_empty() {
                        event["subtype"].as_str().unwrap_or("").to_string()
                    } else {
                        errors.join("\n")
                    }
                });
                // 订阅额度用完时 Claude Code 不调模型、直接给一句「You've hit your … limit」：也按出错算
                let lower = text.to_lowercase();
                let is_error = event["is_error"].as_bool() == Some(true) || (!turn_streamed && lower.contains("hit your") && lower.contains("limit"));
                // --resume 的会话找不到了（Claude Code 那边删了 / 换了配置目录）：还没 init 就报 No conversation found，换新会话重来
                if is_error && p.resumed && !p.started && !out.started() && lower.contains("no conversation found") {
                    conv.stop(p, &format!("会话接不回：{}", cut(&text, 200)));
                    r.resume_failed = true;
                    return r;
                }
                backend::log(&format!(
                    "result key={} subtype={} error={is_error} turns={} 还等={} 后台={}",
                    conv::short(&conv.key),
                    event["subtype"].as_str().unwrap_or(""),
                    event["num_turns"],
                    if p.lifecycle { p.open.len() } else { p.unanswered },
                    p.tasks.values().filter(|t| t.wait).count()
                ));
                if is_error {
                    let f = classify(&event, &text, r.rate.as_ref());
                    if !out.started() {
                        r.failure = Some(f);
                        return r;
                    }
                    out.emit(json!({ "content": format!("\n\n（Claude Code 出错：{}）", f.message) }));
                } else if !turn_streamed && !text.trim().is_empty() {
                    out.emit(json!({ "content": text }));
                }
                turn_streamed = false;
                if notified {
                    expect_until = Some(Instant::now() + Duration::from_secs(3));
                }
                notified = false;
            }
            _ => {}
        }
    }
}

/// result 出错：额度用完 → 429（带多久后再试），没登录 / 登录过期 → 401，其余 502
fn classify(event: &Value, text: &str, rate: Option<&Value>) -> Failure {
    let lower = text.to_lowercase();
    let status = event["api_error_status"].as_u64().unwrap_or(0);
    let rate = rate.cloned().or_else(conv::rate_limit);
    let rejected = rate.as_ref().map(|r| r["status"].as_str() == Some("rejected")).unwrap_or(false);
    if status == 429 || rejected || (lower.contains("hit your") && lower.contains("limit")) || lower.contains("usage limit") || lower.contains("rate limit") {
        let reset = rate.as_ref().and_then(|r| r["resetsAt"].as_u64()).unwrap_or(0);
        let message = if text.trim().is_empty() { "Claude Code 的用量到上限了".to_string() } else { format!("Claude Code 的用量到上限了：{}", text.trim()) };
        return Failure { status: 429, kind: "rate_limit".into(), message, retry_after_secs: reset.saturating_sub(store::now()) };
    }
    if status == 401
        || lower.contains("invalid api key")
        || lower.contains("/login")
        || lower.contains("not logged in")
        || lower.contains("auth login")
        || lower.contains("oauth token")
        || lower.contains("authentication_error")
    {
        backend::forget_auth();
        return Failure::new(401, "claude_auth", format!("Claude Code 没有登录或登录过期了：请在终端运行 claude auth login 登录后再试。（{}）", text.trim()));
    }
    if status == 529 || lower.contains("overloaded") {
        return Failure::new(503, "overloaded", format!("Claude 现在太忙：{}", text.trim()));
    }
    Failure::new(502, "claude_error", if text.trim().is_empty() { "Claude Code 出错了".to_string() } else { text.trim().to_string() })
}

fn with(base: &Value, extra: Value) -> Value {
    let mut tool = base.clone();
    for (k, v) in extra.as_object().cloned().unwrap_or_default() {
        tool[k] = v;
    }
    tool
}

fn tool_started(p: &mut Proc, ask: &Ask, out: &mut Out, structured: bool, block: &Value, parent: Option<&str>) {
    let id = block["id"].as_str().unwrap_or("").to_string();
    if id.is_empty() || p.tools.contains_key(&id) {
        return;
    }
    let name = block["name"].as_str().unwrap_or("").to_string();
    let input = &block["input"];
    let mut base = json!({ "id": id, "name": name, "title": title(&name), "summary": summary(&name, input, &ask.dir), "input": relative(input, &ask.dir) });
    if let Some(parent) = parent {
        base["metadata"] = json!({ "parent": parent });
    }
    if p.tools.len() > 4000 {
        p.tools.clear();
    }
    p.tools.insert(id, base.clone());
    if structured {
        out.tool(with(&base, json!({ "status": "running" })));
    } else {
        let indent = if parent.is_some() { "  ↳ " } else { "▸ " };
        out.emit(json!({ "reasoning_content": format!("\n{indent}{} {}\n", base["title"].as_str().unwrap_or(""), base["summary"].as_str().unwrap_or("")) }));
    }
}

fn tool_finished(p: &mut Proc, out: &mut Out, structured: bool, block: &Value) {
    let id = block["tool_use_id"].as_str().unwrap_or("").to_string();
    let Some(base) = p.tools.get(&id).cloned() else { return };
    let failed = block["is_error"].as_bool() == Some(true);
    let text = result_text(&block["content"]);
    // 后台子智能体：这里的结果只是「已经在后台跑了」，等 task_notification 再报做完
    if !failed && p.tasks.values().any(|t| t.agent && t.tool_use_id == id) {
        if structured {
            out.tool(with(&base, json!({ "status": "running", "output": "在后台运行…" })));
        }
        return;
    }
    if structured {
        out.tool(with(&base, json!({ "status": if failed { "failed" } else { "success" }, "output": cut(&text, 8000) })));
    } else {
        let line = first_line(&text);
        let mark = if failed { "✗" } else { "✓" };
        out.emit(json!({ "reasoning_content": if line.is_empty() { format!("  {mark}\n") } else { format!("  {mark} {line}\n") } }));
    }
}

/// 一个要问用户的权限请求轮到了：在对话里问，这次回答到此为止（返回 true）；已经不用问了（总是允许过 / 被一起答掉了）返回 false
fn ask_user(conv: &Conv, p: &mut Proc, ask: &Ask, out: &mut Out, structured: bool, event: &Value) -> bool {
    let id = event["tool_use_id"].as_str().unwrap_or("");
    let (tool, input) = {
        let mut pending = conv.pending.lock().unwrap();
        let Some(i) = pending.iter().position(|x| x.id == id && !x.asked) else { return false };
        if conv.allow.lock().unwrap().contains(&pending[i].tool) {
            let x = pending.remove(i);
            let _ = x.reply.send(conv::allow_json(&x.input));
            return false;
        }
        pending[i].asked = true;
        (pending[i].tool.clone(), pending[i].input.clone())
    };
    show_ask(p, ask, out, structured, id, &tool, &input);
    backend::log(&format!("permission asked key={} id={id} tool={tool}", conv::short(&conv.key)));
    true
}

/// 在对话里问用户：工具行标成等回答 / 等确认，正文（没画卡片时）列出问题或要做的事
fn show_ask(p: &mut Proc, ask: &Ask, out: &mut Out, structured: bool, id: &str, tool: &str, input: &Value) {
    let base = match p.tools.get(id) {
        Some(b) => b.clone(),
        None => {
            let b = json!({ "id": id, "name": tool, "title": title(tool), "summary": summary(tool, input, &ask.dir), "input": relative(input, &ask.dir) });
            p.tools.insert(id.to_string(), b.clone());
            b
        }
    };
    let question = tool == "AskUserQuestion";
    if structured {
        let note = if question { "等你回答".to_string() } else { format!("等你确认：{}", permission_label(tool)) };
        out.tool(with(&base, json!({ "status": "waiting", "output": note })));
    }
    let text = if question && structured && ask.questions {
        // Friday 把这一行画成选项卡片，正文不再列一遍
        String::new()
    } else if question {
        let mut t = String::from("\n\n");
        for q in input["questions"].as_array().into_iter().flatten() {
            t.push_str(&format!("**{}**\n", q["question"].as_str().unwrap_or("")));
            for o in q["options"].as_array().into_iter().flatten() {
                let label = o["label"].as_str().unwrap_or("");
                match o["description"].as_str().filter(|d| !d.is_empty()) {
                    Some(d) => t.push_str(&format!("- {label}：{d}\n")),
                    None => t.push_str(&format!("- {label}\n")),
                }
            }
            t.push('\n');
        }
        if input["questions"].as_array().map(Vec::len).unwrap_or(0) > 1 {
            t.push_str("（Claude Code 在问你：回复选项的名字或序号，几题之间用「；」隔开；也可以直接说你的想法。）");
        } else {
            t.push_str("（Claude Code 在问你：回复选项的名字或序号，也可以直接说你的想法。）");
        }
        t
    } else if structured {
        format!(
            "\n\nClaude Code 想{}，等你确认：点上面的「允许执行」或回复「允许」（「总是允许」以后同类不再问）；回复别的就取消，按你新说的做。",
            permission_label(tool)
        )
    } else {
        let detail = match input["command"].as_str() {
            Some(c) => cut(c, 2000),
            None => cut(&relative(&input, &ask.dir).to_string(), 2000),
        };
        format!(
            "\n\n> **Claude Code 想{}**\n\n```\n{}\n```\n\n回复「允许」放行这一次，回复「总是允许」以后同类操作不再问；回复别的就取消，按你新说的做。",
            permission_label(tool),
            detail
        )
    };
    if !text.is_empty() {
        out.emit(json!({ "content": text }));
    }
}

fn finish(out: &mut Out, ask: &Ask, r: Pumped) {
    if let Some(f) = &r.failure {
        if !out.started() {
            // 什么都还没写：回 HTTP 错误码（429 让 Friday 按额度处理，带上多久后再试）
            let mut extra = Vec::new();
            if f.retry_after_secs > 0 {
                extra.push(("Retry-After", f.retry_after_secs.to_string()));
            }
            if let Some(s) = out.stream.take() {
                let status = if (400..600).contains(&f.status) { f.status } else { 502 };
                let _ = http::error(s, status, &f.kind, &f.message, &extra);
            }
            return;
        }
        out.emit(json!({ "content": format!("\n\n（Claude Code 出错：{}）", f.message) }));
    }
    let usage = r.usage.json();
    let rate = r.rate.clone().or_else(conv::rate_limit);
    if out.streaming {
        if out.sse.is_none() && !out.gone {
            out.emit(json!({ "content": "" }));
        }
        if let Some(sse) = out.sse.as_mut() {
            let mut last = chunk(&ask.model, json!({}), Some("stop"), Some(&usage));
            if let Some(rate) = rate {
                last["x_friday"] = json!({ "rateLimit": rate });
            }
            sse.data(&last);
            sse.done();
        }
    } else if let Some(s) = out.stream.take() {
        let mut message = json!({ "role": "assistant", "content": out.content });
        if !out.reasoning.is_empty() {
            message["reasoning_content"] = Value::String(out.reasoning.clone());
        }
        let mut body = json!({
            "id": "chatcmpl-claudecode", "object": "chat.completion", "created": store::now(), "model": ask.model,
            "choices": [{ "index": 0, "message": message, "finish_reason": "stop" }],
            "usage": usage,
        });
        if let Some(rate) = rate {
            body["x_friday"] = json!({ "rateLimit": rate });
        }
        let _ = http::json(s, 200, &body);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approval_words() {
        assert_eq!(approval("允许"), Some("once"));
        assert_eq!(approval("好的。"), Some("once"));
        assert_eq!(approval("总是允许\n\n（发出时间：2026-11-12 10:00）"), Some("always"));
        assert_eq!(approval("不用了，改成说个你好"), None);
        assert_eq!(approval("允许，然后把结果写进 README 里再提交一下"), None);
    }

    #[test]
    fn rate_limit_retry_after() {
        let reset = store::now() + 3600;
        let rate = json!({ "status": "rejected", "resetsAt": reset, "rateLimitType": "five_hour" });
        let f = classify(&json!({ "is_error": true }), "You've hit your session limit · resets 3:45pm", Some(&rate));
        assert_eq!(f.status, 429);
        assert!(f.retry_after_secs > 3590 && f.retry_after_secs <= 3600, "{}", f.retry_after_secs);
        let f = classify(&json!({ "api_error_status": 401 }), "Invalid API key · Please run /login", None);
        assert_eq!(f.status, 401);
        assert!(f.message.contains("claude auth login"));
        let f = classify(&json!({}), "something else broke", None);
        assert_eq!(f.status, 502);
    }

    #[test]
    fn tool_rows() {
        let dir = "/w/proj";
        assert_eq!(title("Bash"), "运行命令");
        assert_eq!(title("mcp__github__create_issue"), "create_issue");
        assert_eq!(summary("Write", &json!({ "file_path": "/w/proj/src/a.rs" }), dir), "src/a.rs");
        assert_eq!(summary("Bash", &json!({ "command": "ls -la\nmore" }), dir), "ls -la");
        assert_eq!(summary("Agent", &json!({ "description": "查一下" }), dir), "查一下");
        assert_eq!(relative(&json!({ "file_path": "/w/proj/x.txt", "n": 1 }), dir), json!({ "file_path": "x.txt", "n": 1 }));
    }

    #[test]
    fn data_url_images() {
        let parts = json!([{ "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,QUJD" } }, { "type": "text", "text": "看图" }]);
        let images = images_of(&parts);
        assert_eq!(images, vec![json!({ "type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": "QUJD" } })]);
    }
}

/// 用户对 AskUserQuestion 的回答 → updatedInput.answers（{问题: 选的选项名}，多选用「, 」连起来，选项外的原样）。
/// 几题时按「；」拆成每题一段（段数对不上就整句当每一题的回答）；「（跳过）」/ 空 = 这题不答
fn question_answers(input: &Value, said: &str) -> Value {
    let questions: Vec<&Value> = input["questions"].as_array().into_iter().flatten().collect();
    let said = said.trim();
    let parts: Vec<&str> = if questions.len() > 1 {
        let parts: Vec<&str> = said.split(['；', ';', '\n']).map(str::trim).collect();
        if parts.len() == questions.len() {
            parts
        } else {
            vec![said; questions.len()]
        }
    } else {
        vec![said]
    };
    let mut answers = serde_json::Map::new();
    for (q, part) in questions.iter().zip(parts) {
        let Some(question) = q["question"].as_str() else { continue };
        let part = part.trim().trim_end_matches(['。', '.', '！', '!']);
        if part.is_empty() || part == "（跳过）" || part == "跳过" {
            continue;
        }
        let labels: Vec<&str> = q["options"].as_array().into_iter().flatten().filter_map(|o| o["label"].as_str()).collect();
        let value = if q["multiSelect"].as_bool() == Some(true) {
            let items: Vec<&str> = part.split(['、', ',', '，']).map(str::trim).filter(|s| !s.is_empty()).collect();
            let matched: Vec<&str> = items.iter().filter_map(|i| match_label(&labels, i)).collect();
            if !items.is_empty() && matched.len() == items.len() {
                matched.join(", ")
            } else {
                match_label(&labels, part).unwrap_or(part).to_string()
            }
        } else {
            match_label(&labels, part).unwrap_or(part).to_string()
        };
        answers.insert(question.to_string(), Value::String(value));
    }
    Value::Object(answers)
}

/// 一段回答对上哪个选项名：名字（不分大小写，也认去掉「(Recommended)」「（推荐）」的）、或序号「2」「第二个」「选 2」
fn match_label<'a>(labels: &[&'a str], text: &str) -> Option<&'a str> {
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
    if let Some(l) = labels.iter().find(|l| bare(l) == t) {
        return Some(l);
    }
    let n = t.trim_start_matches(['选', '第']).trim().trim_end_matches(['个', '项', '号', '.', '、', ')', '）']).trim();
    let index = match n {
        "一" => Some(1),
        "二" | "两" => Some(2),
        "三" => Some(3),
        "四" => Some(4),
        _ => n.parse::<usize>().ok(),
    }?;
    labels.get(index.checked_sub(1)?).copied()
}

#[cfg(test)]
mod question_tests {
    use super::*;

    fn input() -> Value {
        json!({ "questions": [
            { "question": "用哪种语言？", "header": "语言", "multiSelect": false,
              "options": [{ "label": "Rust (Recommended)", "description": "快" }, { "label": "Python", "description": "简单" }] },
            { "question": "要哪些附件？", "header": "附件", "multiSelect": true,
              "options": [{ "label": "表格" }, { "label": "图表" }, { "label": "摘要" }] }
        ]})
    }

    #[test]
    fn answers_by_label_number_and_free_text() {
        assert_eq!(question_answers(&input(), "rust；1、3"), json!({ "用哪种语言？": "Rust (Recommended)", "要哪些附件？": "表格, 摘要" }));
        assert_eq!(question_answers(&input(), "2；图表"), json!({ "用哪种语言？": "Python", "要哪些附件？": "图表" }));
        assert_eq!(question_answers(&input(), "随你"), json!({ "用哪种语言？": "随你", "要哪些附件？": "随你" }));
        assert_eq!(question_answers(&input(), "Python；（跳过）"), json!({ "用哪种语言？": "Python" }));
    }
}
