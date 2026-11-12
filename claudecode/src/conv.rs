//! 每段对话一个常驻的 claude 进程（`claude -p --input-format stream-json …`，工作目录 = 对话的工作目录）：
//! 每句话往它的 stdin 写一行，输出一行一个 JSON 读进一个队列；第一句以后几乎没有额外开销（每句新起一个进程要 2.4–3.9 秒）。
//! 换模型 / 权限档、Friday 点了停止、闲置 15 分钟 → 停掉，下一句用 `--resume <会话 id>` 接回。
//!
//! 权限：claude 带 `--permission-prompt-tool mcp__friday__approve`，要确认的操作调本程序的 `mcp` 子命令（每段对话一份 MCP 配置，
//! 带对话的 key），子命令把请求 POST 到本程序的 /internal/permission，这里挂着等：能自己定的（自动档改工作目录里的文件、待办、
//! 「总是允许」过的）马上回，别的插进这段对话的输出队列（和 claude 的输出排在一起），由正在流的请求在对话里问用户，
//! 下一句话再回答它（见 chat.rs）。

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::backend;

/// 一个挂着等用户回答的权限请求（MCP 那边的 HTTP 请求还没回）
pub struct Pending {
    /// Claude Code 的 tool_use_id
    pub id: String,
    pub tool: String,
    pub input: Value,
    pub reply: Sender<Value>,
    /// 已经在对话里问过用户了（那次回答以这个问题结束，下一句话就是回答）
    pub asked: bool,
}

/// Claude Code 报的后台任务（task_started → task_notification）
pub struct Task {
    pub tool_use_id: String,
    pub background: bool,
    /// 这次请求要等它做完才算完（子智能体要等；放到后台的命令（开发服务器之类）不等）
    pub wait: bool,
    pub agent: bool,
}

/// 常驻进程和它的状态（只有正在处理这段对话的那个请求拿着锁）
#[derive(Default)]
pub struct Proc {
    pub child: Option<Child>,
    pub stdin: Option<ChildStdin>,
    pub rx: Option<Receiver<Value>>,
    /// 起进程时的模型（请求里的 id）和 claude 的 --permission-mode
    pub model: String,
    pub mode: String,
    /// Claude Code 的 session_id（init 里报的；停了以后留着，下次 --resume）
    pub session: String,
    /// 这个进程是 --resume 起的；started = 见过 init（接回成功）
    pub resumed: bool,
    pub started: bool,
    /// 写进去、还没等到 result 的话（老版本 claude 没有 command_lifecycle 时用它算）
    pub unanswered: usize,
    /// 写进去、还没做完的话（每句带 uuid，claude 报 command_lifecycle：queued → started → completed；
    /// 一轮里插进去的话会并进正在跑的那一轮，不会单独有 result，所以按它算才准）
    pub open: HashSet<String>,
    /// 这个进程报过 command_lifecycle
    pub lifecycle: bool,
    /// 主智能体正在一轮里（init / 开始请求之后、result 之前）
    pub in_turn: bool,
    pub tasks: HashMap<String, Task>,
    /// 报过开始的工具：tool_use_id → {id, name, title, summary, input[, metadata]}（之后每条状态都带上；放行后接着的请求也用它）
    pub tools: HashMap<String, Value>,
    /// 上一个请求收尾时没处理的权限请求事件（排在新的输出前面）
    pub deferred: VecDeque<Value>,
    pub stderr: Arc<Mutex<VecDeque<String>>>,
}

pub struct Conv {
    pub key: String,
    pub dir: String,
    pub oneshot: bool,
    /// Friday 的权限档（auto / plan / full）：权限请求按它判断
    pub permission: Mutex<String>,
    inject: Mutex<Option<Sender<Value>>>,
    pub pending: Mutex<Vec<Pending>>,
    /// 用户说过「总是允许」的工具名（本程序活着时一直算）
    pub allow: Mutex<HashSet<String>>,
    /// 每来一个请求 +1：还在流的旧请求看到变了就停下让位
    pub turn: AtomicU64,
    pub proc: Mutex<Proc>,
    used: Mutex<Instant>,
    /// 正在跑的 claude 的 pid（退出时不拿锁也能停它）
    pid: AtomicU32,
}

fn registry() -> &'static Mutex<HashMap<String, Arc<Conv>>> {
    static R: OnceLock<Mutex<HashMap<String, Arc<Conv>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 最近一次见到的额度信息（rate_limit_event 的 rate_limit_info）
fn rate_slot() -> &'static Mutex<Option<Value>> {
    static R: OnceLock<Mutex<Option<Value>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(None))
}

pub fn set_rate_limit(info: &Value) {
    if info.is_object() {
        *rate_slot().lock().unwrap() = Some(info.clone());
    }
}

pub fn rate_limit() -> Option<Value> {
    rate_slot().lock().unwrap().clone()
}

pub fn get_or_create(key: &str, dir: &str) -> Arc<Conv> {
    let mut r = registry().lock().unwrap();
    if let Some(c) = r.get(key) {
        if c.dir == dir {
            return c.clone();
        }
    }
    let c = Arc::new(Conv::new(key, dir, false));
    r.insert(key.to_string(), c.clone());
    c
}

/// 正在跑的常驻进程数
pub fn running() -> usize {
    registry().lock().unwrap().values().filter(|c| c.pid.load(Ordering::SeqCst) != 0).count()
}

/// Friday 的权限档 → claude 的 --permission-mode。自动档用 default（acceptEdits 连工作目录里的 rm、echo > 都不问），
/// 改文件由本程序自己放行
pub fn claude_mode(permission: &str) -> &'static str {
    match permission {
        "full" => "bypassPermissions",
        "plan" => "plan",
        _ => "default",
    }
}

pub fn allow_json(input: &Value) -> Value {
    let input = if input.is_object() { input.clone() } else { json!({}) };
    json!({ "behavior": "allow", "updatedInput": input })
}

pub fn deny_json(message: &str) -> Value {
    json!({ "behavior": "deny", "message": message })
}

impl Conv {
    pub fn new(key: &str, dir: &str, oneshot: bool) -> Conv {
        Conv {
            key: key.to_string(),
            dir: dir.to_string(),
            oneshot,
            permission: Mutex::new("auto".into()),
            inject: Mutex::new(None),
            pending: Mutex::new(Vec::new()),
            allow: Mutex::new(HashSet::new()),
            turn: AtomicU64::new(0),
            proc: Mutex::new(Proc::default()),
            used: Mutex::new(Instant::now()),
            pid: AtomicU32::new(0),
        }
    }

    pub fn touch(&self) {
        *self.used.lock().unwrap() = Instant::now();
    }

    pub fn alive(&self, p: &mut Proc) -> bool {
        p.child.as_mut().map(|c| matches!(c.try_wait(), Ok(None))).unwrap_or(false)
    }

    /// 拿走已经在对话里问过用户的那个权限请求
    pub fn take_asked(&self) -> Option<Pending> {
        let mut pending = self.pending.lock().unwrap();
        let i = pending.iter().position(|p| p.asked)?;
        Some(pending.remove(i))
    }

    pub fn stderr_tail(&self, p: &Proc) -> String {
        p.stderr.lock().unwrap().iter().cloned().collect::<Vec<_>>().join("\n")
    }

    /// 这段对话的 MCP 配置（权限确认用的 friday 服务 = 本程序的 mcp 子命令），写进数据目录，返回路径
    fn mcp_config(&self) -> Result<PathBuf, String> {
        let exe = std::env::current_exe().map_err(|e| format!("找不到本程序的路径：{e}"))?;
        let (url, token) = backend::server();
        let config = json!({
            "mcpServers": {
                "friday": {
                    "type": "stdio",
                    "command": exe.to_string_lossy(),
                    "args": ["mcp", "--url", url, "--key", self.key],
                    "env": { "FRIDAY_CLAUDECODE_TOKEN": token },
                }
            }
        });
        let dir = backend::data_dir().join("mcp");
        std::fs::create_dir_all(&dir).map_err(|e| format!("建 {} 失败：{e}", dir.display()))?;
        let path = dir.join(format!("{}.json", self.key));
        let mut options = std::fs::OpenOptions::new();
        options.create(true).write(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut f = options.open(&path).map_err(|e| format!("写 {} 失败：{e}", path.display()))?;
        f.write_all(config.to_string().as_bytes()).map_err(|e| format!("写 {} 失败：{e}", path.display()))?;
        Ok(path)
    }

    /// 拉起 claude。resume：接回的会话 id
    pub fn spawn(&self, p: &mut Proc, bin: &Path, model: &str, mode: &str, resume: Option<&str>) -> Result<(), String> {
        let mut args: Vec<String> = ["-p", "--input-format", "stream-json", "--output-format", "stream-json", "--include-partial-messages", "--verbose"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        if self.oneshot {
            // 一次性的辅助请求（起标题、总结……）：不要工具、不加载 MCP、不留会话
            args.extend(["--tools", "", "--strict-mcp-config", "--no-session-persistence"].iter().map(|s| s.to_string()));
        } else {
            let config = self.mcp_config()?;
            args.push("--forward-subagent-text".into());
            args.push("--mcp-config".into());
            args.push(config.to_string_lossy().to_string());
            args.push("--permission-prompt-tool".into());
            args.push("mcp__friday__approve".into());
        }
        if !model.is_empty() && model != "default" {
            args.push("--model".into());
            args.push(model.to_string());
        }
        args.push("--permission-mode".into());
        args.push(mode.to_string());
        if let Some(id) = resume.filter(|s| !s.is_empty()) {
            args.push("--resume".into());
            args.push(id.to_string());
        }
        let mut cmd = backend::command(bin);
        cmd.args(&args).current_dir(&self.dir).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        #[cfg(unix)]
        {
            // 自己一个进程组：停的时候连它拉起的命令、MCP 子进程一起停
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let mut child = cmd.spawn().map_err(|e| format!("拉起 Claude Code 失败（{}）：{e}", bin.display()))?;
        backend::log(&format!(
            "spawn key={} pid={} dir={} model={model} mode={mode} resume={}",
            short(&self.key),
            child.id(),
            self.dir,
            resume.unwrap_or("-")
        ));
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let (tx, rx) = channel::<Value>();
        let reader_tx = tx.clone();
        let key = short(&self.key).to_string();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
                let line = String::from_utf8_lossy(&buf);
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<Value>(line) {
                    Ok(v) => {
                        if reader_tx.send(v).is_err() {
                            return;
                        }
                    }
                    Err(_) => backend::log(&format!("[claude {key}] {}", cut_line(line, 300))),
                }
            }
            let _ = reader_tx.send(json!({ "type": "friday_exit" }));
        });
        let tail = Arc::new(Mutex::new(VecDeque::new()));
        let tail_w = tail.clone();
        let key = short(&self.key).to_string();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if line.trim().is_empty() {
                    continue;
                }
                backend::log(&format!("[claude {key} stderr] {}", cut_line(&line, 500)));
                let mut t = tail_w.lock().unwrap();
                t.push_back(cut_line(&line, 300));
                while t.len() > 8 {
                    t.pop_front();
                }
            }
        });
        self.pid.store(child.id(), Ordering::SeqCst);
        *self.inject.lock().unwrap() = Some(tx);
        p.stdin = child.stdin.take();
        p.child = Some(child);
        p.rx = Some(rx);
        p.model = model.to_string();
        p.mode = mode.to_string();
        p.resumed = resume.map(|s| !s.is_empty()).unwrap_or(false);
        p.started = false;
        p.unanswered = 0;
        p.open.clear();
        p.lifecycle = false;
        p.in_turn = false;
        p.tasks.clear();
        p.tools.clear();
        p.deferred.clear();
        p.stderr = tail;
        Ok(())
    }

    /// 写一句用户的话（content：Anthropic 的内容块数组）
    pub fn write(&self, p: &mut Proc, content: Value) -> Result<(), String> {
        let uuid = uuid_v4();
        let line = json!({ "type": "user", "uuid": uuid, "message": { "role": "user", "content": content } }).to_string() + "\n";
        let stdin = p.stdin.as_mut().ok_or("Claude Code 进程没在跑")?;
        stdin.write_all(line.as_bytes()).and_then(|_| stdin.flush()).map_err(|e| format!("写给 Claude Code 失败：{e}"))?;
        p.unanswered += 1;
        p.open.insert(uuid);
        Ok(())
    }

    /// 写进去的话都答完了吗
    pub fn answered(p: &Proc) -> bool {
        if p.lifecycle {
            p.open.is_empty()
        } else {
            p.unanswered == 0
        }
    }

    /// 一次性请求写完就关 stdin（claude 答完自己退出）
    pub fn close_input(&self, p: &mut Proc) {
        p.stdin = None;
    }

    /// 停掉进程（连它的进程组），清掉挂着的权限请求（MCP 那边收到「拒绝」）；会话 id 留着下次接回
    pub fn stop(&self, p: &mut Proc, why: &str) {
        if let Some(mut child) = p.child.take() {
            p.stdin = None;
            let pid = child.id();
            if matches!(child.try_wait(), Ok(None)) {
                terminate(pid);
                let deadline = Instant::now() + Duration::from_secs(3);
                while Instant::now() < deadline && matches!(child.try_wait(), Ok(None)) {
                    std::thread::sleep(Duration::from_millis(50));
                }
                if matches!(child.try_wait(), Ok(None)) {
                    kill_hard(pid);
                    let _ = child.kill();
                }
            }
            let _ = child.wait();
            if !why.is_empty() {
                backend::log(&format!("stop key={} pid={pid}：{why}", short(&self.key)));
            }
        }
        self.pid.store(0, Ordering::SeqCst);
        p.stdin = None;
        p.rx = None;
        *self.inject.lock().unwrap() = None;
        let dropped: Vec<Pending> = std::mem::take(&mut *self.pending.lock().unwrap());
        drop(dropped);
        p.unanswered = 0;
        p.open.clear();
        p.lifecycle = false;
        p.in_turn = false;
        p.started = false;
        p.resumed = false;
        p.tasks.clear();
        p.tools.clear();
        p.deferred.clear();
    }
}

/// 随机的 UUID（v4 的格式）：每句话的 uuid，claude 的 command_lifecycle 按它报
fn uuid_v4() -> String {
    let h = backend::random_hex(32);
    let b = h.as_bytes();
    let variant = ["8", "9", "a", "b"][(b[16] as usize) % 4];
    format!(
        "{}-{}-4{}-{}{}-{}",
        &h[0..8],
        &h[8..12],
        &h[13..16],
        variant,
        &h[17..20],
        &h[20..32]
    )
}

pub fn short(key: &str) -> &str {
    &key[..key.len().min(8)]
}

fn cut_line(s: &str, n: usize) -> String {
    let mut it = s.chars();
    let head: String = it.by_ref().take(n).collect();
    if it.next().is_some() {
        format!("{head}…")
    } else {
        head
    }
}

#[cfg(unix)]
fn terminate(pid: u32) {
    // SAFETY: 只是给自己拉起的进程组发信号
    unsafe {
        libc::kill(-(pid as i32), libc::SIGTERM);
    }
}

#[cfg(unix)]
fn kill_hard(pid: u32) {
    // SAFETY: 同上
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
    }
}

#[cfg(windows)]
fn terminate(pid: u32) {
    use std::os::windows::process::CommandExt;
    let _ = std::process::Command::new("taskkill")
        .args(["/T", "/F", "/PID", &pid.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(0x0800_0000)
        .status();
}

#[cfg(windows)]
fn kill_hard(pid: u32) {
    terminate(pid);
}

#[cfg(not(any(unix, windows)))]
fn terminate(_pid: u32) {}

#[cfg(not(any(unix, windows)))]
fn kill_hard(_pid: u32) {}

/// 闲置收掉：15 分钟没用的常驻进程停掉（有挂着等用户确认的放 12 小时）
pub fn reaper() {
    loop {
        std::thread::sleep(Duration::from_secs(30));
        let convs: Vec<Arc<Conv>> = registry().lock().unwrap().values().cloned().collect();
        for c in convs {
            if c.pid.load(Ordering::SeqCst) == 0 {
                continue;
            }
            let idle = c.used.lock().unwrap().elapsed();
            let waiting = !c.pending.lock().unwrap().is_empty();
            let limit = if waiting { Duration::from_secs(12 * 3600) } else { Duration::from_secs(15 * 60) };
            if idle < limit {
                continue;
            }
            if let Ok(mut p) = c.proc.try_lock() {
                c.stop(&mut p, "闲置太久，收掉（下一句用 --resume 接回）");
            }
        }
    }
}

/// 本程序要退了：停掉所有 claude
pub fn shutdown() {
    let convs: Vec<Arc<Conv>> = registry().lock().unwrap().values().cloned().collect();
    for c in convs {
        let pid = c.pid.swap(0, Ordering::SeqCst);
        if pid != 0 {
            terminate(pid);
        }
    }
}

const EDITS: [&str; 4] = ["Edit", "Write", "MultiEdit", "NotebookEdit"];
const TODOS: [&str; 5] = ["TodoWrite", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet"];

/// 去掉 . 和 ..（文件可能还不存在，不能 canonicalize）
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// 要改的文件在工作目录里
fn inside(input: &Value, dir: &str) -> bool {
    let Some(path) = ["file_path", "notebook_path", "path"].iter().find_map(|k| input[*k].as_str()) else { return false };
    let path = Path::new(path);
    let full = if path.is_absolute() { path.to_path_buf() } else { Path::new(dir).join(path) };
    normalize(&full).starts_with(normalize(Path::new(dir)))
}

/// 本程序自己能定的（不用问用户）：Some(回答)
fn decide(conv: &Conv, permission: &str, tool: &str, input: &Value) -> Option<Value> {
    if tool == "AskUserQuestion" {
        return None;
    }
    if permission == "full" || TODOS.contains(&tool) {
        return Some(allow_json(input));
    }
    if permission == "plan" {
        if tool == "ExitPlanMode" {
            return Some(deny_json("Friday 现在是计划档：只出方案、不动手。方案已经显示给用户了，用户换到别的权限档后会让你接着做。"));
        }
        return Some(deny_json(&format!("Friday 现在是计划档（只读）：不能{}。", crate::chat::permission_label(tool))));
    }
    if EDITS.contains(&tool) && inside(input, &conv.dir) {
        return Some(allow_json(input));
    }
    if conv.allow.lock().unwrap().contains(tool) {
        return Some(allow_json(input));
    }
    None
}

/// POST /internal/permission（mcp 子命令转来的）：{key, tool_name, input, tool_use_id} → {"behavior": …}；要问用户的一直挂着等
pub fn ask_permission(body: &Value) -> Value {
    let key = body["key"].as_str().unwrap_or("");
    let tool = body["tool_name"].as_str().unwrap_or("").to_string();
    let input = body["input"].clone();
    let id = body["tool_use_id"].as_str().unwrap_or("").to_string();
    let conv = registry().lock().unwrap().get(key).cloned();
    let Some(conv) = conv else {
        backend::log(&format!("permission key={} tool={tool}：没有这段对话，拒绝", short(key)));
        return deny_json("Friday 里找不到这段对话，没法问用户，这次没有执行。");
    };
    let permission = conv.permission.lock().unwrap().clone();
    if let Some(answer) = decide(&conv, &permission, &tool, &input) {
        backend::log(&format!("permission key={} tool={tool} id={id} 档={permission} → {}", short(key), answer["behavior"].as_str().unwrap_or("")));
        return answer;
    }
    let (tx, rx) = channel();
    {
        let inject = conv.inject.lock().unwrap();
        let Some(sender) = inject.as_ref() else { return deny_json("Claude Code 进程已经停了，这次没有执行。") };
        conv.pending.lock().unwrap().push(Pending { id: id.clone(), tool: tool.clone(), input, reply: tx, asked: false });
        let _ = sender.send(json!({ "type": "friday_permission", "tool_use_id": id }));
    }
    backend::log(&format!("permission key={} tool={tool} id={id}：问用户", short(key)));
    let answer = rx.recv().unwrap_or_else(|_| deny_json("Friday 那边停下了（对话被停止或重开），这次没有执行。"));
    backend::log(&format!("permission key={} tool={tool} id={id} → {}", short(key), answer["behavior"].as_str().unwrap_or("")));
    answer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_inside_working_dir() {
        assert!(inside(&json!({ "file_path": "/w/proj/a.txt" }), "/w/proj"));
        assert!(inside(&json!({ "file_path": "sub/a.txt" }), "/w/proj"));
        assert!(!inside(&json!({ "file_path": "/w/proj/../other/a.txt" }), "/w/proj"));
        assert!(!inside(&json!({ "file_path": "/w/project2/a.txt" }), "/w/proj"));
        assert!(!inside(&json!({ "command": "rm -rf /" }), "/w/proj"));
    }

    #[test]
    fn uuid_format() {
        let u = uuid_v4();
        let parts: Vec<&str> = u.split('-').collect();
        assert_eq!(parts.iter().map(|p| p.len()).collect::<Vec<_>>(), vec![8, 4, 4, 4, 12]);
        assert!(parts[2].starts_with('4'));
        assert!(matches!(&parts[3][..1], "8" | "9" | "a" | "b"));
        assert_ne!(u, uuid_v4());
    }

    #[test]
    fn decisions() {
        let c = Conv::new("k", "/w/proj", false);
        assert_eq!(decide(&c, "auto", "Write", &json!({ "file_path": "/w/proj/a" })).unwrap()["behavior"], "allow");
        assert!(decide(&c, "auto", "Write", &json!({ "file_path": "/etc/hosts" })).is_none());
        assert!(decide(&c, "auto", "Bash", &json!({ "command": "ls" })).is_none());
        assert_eq!(decide(&c, "auto", "TodoWrite", &json!({})).unwrap()["behavior"], "allow");
        assert_eq!(decide(&c, "plan", "Bash", &json!({ "command": "ls" })).unwrap()["behavior"], "deny");
        assert!(decide(&c, "plan", "AskUserQuestion", &json!({})).is_none());
        c.allow.lock().unwrap().insert("Bash".into());
        assert_eq!(decide(&c, "auto", "Bash", &json!({ "command": "ls" })).unwrap()["behavior"], "allow");
    }
}
