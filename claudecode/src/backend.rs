//! 找本机装着的 Claude Code（`claude` 命令行，用户自己装、自己登录的那个，原样调用，不改它）：
//! FRIDAY_CLAUDE_BIN、PATH、~/.local/bin、~/.claude/local、Homebrew、/usr/local/bin，Windows 上 %USERPROFILE%\.local\bin\claude.exe、%APPDATA%\npm\claude.cmd。
//! 版本看 `claude --version`；登录状态只看 `claude auth status` 自己打印的 JSON，而且只留 loggedIn / authMethod / apiProvider / subscriptionType
//! 四项（邮箱、组织 id 等一概不留）。本程序**不读、不碰** Claude Code 的登录凭据（钥匙串、~/.claude/.credentials.json、OAuth 令牌）。
//! 拉起 claude 时继承本程序的环境，但去掉从上层 Claude Code 会话漏下来的 CLAUDECODE、CLAUDE_CODE_* 等（带着它们 claude 会认错登录）。
//! 模型列表不写死：问 claude 自己（流式输入的 initialize 控制请求，见 models()），Claude Code 更新出了新模型、换了默认，这里跟着变；
//! 具体版本（Opus 4.7、Sonnet 4.6 这些 Claude Code 的 /model 里「更多」那一组）取 Anthropic 公开的 Claude Code 模型目录（catalog()）。
//! 额度：claude 的 get_usage 控制请求（就是 /usage 的数据，claude 用它自己的登录去查），见 usage()。

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::channel;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

use serde_json::{json, Value};

use crate::store;

struct State {
    data_dir: PathBuf,
    /// 本程序的地址（给 mcp 子命令回连）和内部口令（只给 mcp 子命令用，和宿主的 --token 分开）
    server: String,
    internal_token: String,
    /// claude --version 的缓存：（程序路径, 修改时间）→ 版本
    version: Option<(PathBuf, Option<std::time::SystemTime>, String)>,
    /// claude auth status 的缓存（60 秒）
    auth: Option<(Instant, Value)>,
    /// claude 认不认后来加的参数：（程序路径, 修改时间）→ Flags
    flags: Option<(PathBuf, Option<SystemTime>, Flags)>,
    /// Claude Code 报的模型（60 秒）：（程序路径, 修改时间, 取到的时候, 模型）
    models: Option<(PathBuf, Option<SystemTime>, Instant, Vec<Value>)>,
    /// 公开的模型目录里 Claude Code 那一份的模型（6 小时）
    catalog: Option<(Instant, Vec<Value>)>,
    /// get_usage 的结果（30 秒）
    usage: Option<(Instant, Value)>,
}

fn state() -> &'static Mutex<State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(|| {
        Mutex::new(State {
            data_dir: PathBuf::from("."),
            server: String::new(),
            internal_token: String::new(),
            version: None,
            auth: None,
            flags: None,
            models: None,
            catalog: None,
            usage: None,
        })
    })
}

pub fn init(data_dir: &Path) {
    state().lock().unwrap().data_dir = data_dir.to_path_buf();
}

pub fn data_dir() -> PathBuf {
    state().lock().unwrap().data_dir.clone()
}

pub fn set_server(url: &str, internal_token: &str) {
    let mut s = state().lock().unwrap();
    s.server = url.to_string();
    s.internal_token = internal_token.to_string();
}

/// （本程序的地址, 内部口令）
pub fn server() -> (String, String) {
    let s = state().lock().unwrap();
    (s.server.clone(), s.internal_token.clone())
}

/// 诊断日志：<data-dir>/friday-claudecode.log（也打到 stderr）；超过 1 MB 从头写
pub fn log(line: &str) {
    use std::io::Write;
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    eprintln!("{line}");
    let path = data_dir().join("friday-claudecode.log");
    let truncate = std::fs::metadata(&path).map(|m| m.len() > 1 << 20).unwrap_or(false);
    let file = std::fs::OpenOptions::new().create(true).append(!truncate).write(true).truncate(truncate).open(&path);
    if let Ok(mut f) = file {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let _ = writeln!(f, "{now} {line}");
    }
}

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from)
}

fn names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["claude.exe", "claude.cmd"]
    } else {
        &["claude"]
    }
}

pub fn find_binary() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("FRIDAY_CLAUDE_BIN") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            for name in names() {
                let p = dir.join(name);
                if p.is_file() {
                    return Some(p);
                }
            }
        }
    }
    let mut candidates = Vec::new();
    if let Some(home) = home() {
        for name in names() {
            candidates.push(home.join(".local").join("bin").join(name));
            candidates.push(home.join(".claude").join("local").join(name));
        }
    }
    if cfg!(windows) {
        if let Some(d) = std::env::var_os("APPDATA") {
            candidates.push(PathBuf::from(d).join("npm").join("claude.cmd"));
        }
    } else {
        candidates.push(PathBuf::from("/opt/homebrew/bin/claude"));
        candidates.push(PathBuf::from("/usr/local/bin/claude"));
    }
    candidates.into_iter().find(|p| p.is_file())
}

pub const NOT_FOUND: &str = "没找到 Claude Code（claude 命令行）：请先装 Claude Code（macOS / Linux：curl -fsSL https://claude.ai/install.sh | bash；或 npm install -g @anthropic-ai/claude-code），装好后在终端运行 claude auth login 登录";

/// 上层 Claude Code 会话漏下来的环境变量：带着它们拉起的 claude 会认错登录、以为自己是被嵌套调用的
fn leaked(name: &str) -> bool {
    name == "CLAUDECODE"
        || name == "CLAUDE_PID"
        || name == "CLAUDE_EFFORT"
        || name.starts_with("CLAUDE_CODE_")
        || name.starts_with("CLAUDE_AGENT_SDK_")
        || name.starts_with("CLAUDE_PREVIEW_")
}

/// 拉起 claude 的命令：继承本程序的环境（CLAUDE_CONFIG_DIR、ANTHROPIC_*、代理、PATH、HOME 都留着），去掉漏下来的那些；
/// Windows 上不弹控制台窗口
pub fn command(bin: &Path) -> Command {
    let mut cmd = Command::new(bin);
    for (name, _) in std::env::vars_os() {
        if let Some(name) = name.to_str() {
            if leaked(name) {
                cmd.env_remove(name);
            }
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

/// 跑一个很快的命令，最多等 secs 秒，返回 stdout
fn run_quick(mut cmd: Command, secs: u64) -> Option<String> {
    let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;
    let mut stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stdout.read_to_string(&mut s);
        s
    });
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
    reader.join().ok()
}

fn mtime(bin: &Path) -> Option<SystemTime> {
    std::fs::metadata(bin).and_then(|m| m.modified()).ok()
}

/// `claude --version` → "2.1.276"（按程序路径和修改时间缓存）
pub fn version(bin: &Path) -> Option<String> {
    let mtime = std::fs::metadata(bin).and_then(|m| m.modified()).ok();
    if let Some((p, t, v)) = &state().lock().unwrap().version {
        if p == bin && *t == mtime {
            return Some(v.clone());
        }
    }
    let mut cmd = command(bin);
    cmd.arg("--version");
    let text = run_quick(cmd, 20)?;
    let v = text.split_whitespace().map(|w| w.trim_start_matches('v')).find(|w| w.starts_with(|c: char| c.is_ascii_digit()))?.to_string();
    state().lock().unwrap().version = Some((bin.to_path_buf(), mtime, v.clone()));
    Some(v)
}

/// 登录状态：`claude auth status`（默认就是 JSON；没登录时退出码 1，但照样打印 JSON），只留四项。60 秒内用缓存。
pub fn auth(bin: &Path) -> Option<Value> {
    if let Some((at, v)) = &state().lock().unwrap().auth {
        if at.elapsed() < Duration::from_secs(60) {
            return Some(v.clone());
        }
    }
    let mut cmd = command(bin);
    cmd.args(["auth", "status", "--json"]);
    let text = run_quick(cmd, 20)?;
    let start = text.find('{')?;
    let full: Value = serde_json::from_str(text[start..].trim()).ok()?;
    let mut kept = json!({});
    for key in ["loggedIn", "authMethod", "apiProvider", "subscriptionType"] {
        if let Some(v) = full.get(key) {
            if !v.is_null() {
                kept[key] = v.clone();
            }
        }
    }
    state().lock().unwrap().auth = Some((Instant::now(), kept.clone()));
    Some(kept)
}

/// 登录过期 / 出错后丢掉缓存，下次重查
pub fn forget_auth() {
    state().lock().unwrap().auth = None;
}

/// claude 认不认这几个后来才有的参数（老版本没有；--thinking-display 不在 --help 里）
#[derive(Clone, Copy, Debug)]
pub struct Flags {
    /// --effort low|medium|high|xhigh|max（思考档）
    pub effort: bool,
    /// --thinking-display summarized（无头模式默认 omitted：思考只有签名、没有字）
    pub display: bool,
}

/// 探一次：关着输入跑 `claude -p … <参数>`——认得的 1 秒内正常退出（没话可答），不认的立刻报 `unknown option '--x'`，去掉它再试。
/// 按程序路径和修改时间缓存（Claude Code 更新了重探）；探不出结果（超时、起不来）当都认，不缓存
pub fn flags(bin: &Path) -> Flags {
    let mtime = mtime(bin);
    if let Some((p, t, f)) = &state().lock().unwrap().flags {
        if p == bin && *t == mtime {
            return *f;
        }
    }
    let mut f = Flags { effort: true, display: true };
    for _ in 0..3 {
        let mut cmd = command(bin);
        cmd.args(QUIET);
        if f.effort {
            cmd.args(["--effort", "high"]);
        }
        if f.display {
            cmd.args(["--thinking-display", "summarized"]);
        }
        cmd.current_dir(work_dir());
        let Some((_, err)) = run_capture(cmd, 20) else { return f };
        if f.effort && err.contains("unknown option '--effort'") {
            f.effort = false;
        } else if f.display && err.contains("unknown option '--thinking-display'") {
            f.display = false;
        } else {
            break;
        }
    }
    log(&format!("claude 参数：--effort {} --thinking-display {}", f.effort, f.display));
    state().lock().unwrap().flags = Some((bin.to_path_buf(), mtime, f));
    f
}

/// 只起 claude 不干活时的参数：不留会话、不要工具、不加载用户的 MCP 服务
const QUIET: [&str; 11] = [
    "-p",
    "--input-format",
    "stream-json",
    "--output-format",
    "stream-json",
    "--verbose",
    "--no-session-persistence",
    "--tools",
    "",
    "--strict-mcp-config",
    "--include-partial-messages",
];

/// 没给工作目录时 claude 在这里跑（<data-dir>/work）
pub fn work_dir() -> PathBuf {
    let dir = data_dir().join("work");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// 跑一个命令，关着输入，最多等 secs 秒：（正常退出了没有, stderr）；超时、起不来返回 None
fn run_capture(mut cmd: Command, secs: u64) -> Option<(bool, String)> {
    let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().ok()?;
    let mut stderr = child.stderr.take()?;
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });
    let deadline = Instant::now() + Duration::from_secs(secs);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    Some((status.success(), reader.join().ok().unwrap_or_default()))
}

/// Claude Code 自己报的模型：流式输入的 initialize 控制请求（Agent SDK 的 supportedModels() 就是它）回的 models——
/// [{value（--model 的值：default / opus[1m] / sonnet / haiku / 完整模型名…）, displayName, description, resolvedModel,
///   supportsEffort, supportedEffortLevels（low … max）, supportsAdaptiveThinking, …}]，跟着装的 Claude Code 版本和登录的账号变。
/// 60 秒内用缓存；取到的存进 <data-dir>/models.json，取不到（claude 起不来、超时）时用存下的，第二项为 true
pub fn models(bin: &Path) -> Result<(Vec<Value>, bool), String> {
    let mtime = mtime(bin);
    if let Some((p, t, at, list)) = &state().lock().unwrap().models {
        if p == bin && *t == mtime && at.elapsed() < Duration::from_secs(60) {
            return Ok((list.clone(), false));
        }
    }
    let saved = data_dir().join("models.json");
    match ask_models(bin) {
        Ok(list) => {
            state().lock().unwrap().models = Some((bin.to_path_buf(), mtime, Instant::now(), list.clone()));
            let _ = std::fs::write(&saved, json!({ "at": store::now(), "models": list }).to_string());
            Ok((list, false))
        }
        Err(e) => {
            log(&format!("问 Claude Code 要模型失败：{e}"));
            let old = std::fs::read_to_string(&saved).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok());
            match old.and_then(|v| v["models"].as_array().cloned()).filter(|l| !l.is_empty()) {
                Some(list) => Ok((list, true)),
                None => Err(format!("问 Claude Code 要模型列表失败：{e}")),
            }
        }
    }
}

fn ask_models(bin: &Path) -> Result<Vec<Value>, String> {
    let answer = control(bin, json!({ "subtype": "initialize" }), 30)?;
    match answer["models"].as_array() {
        Some(list) if !list.is_empty() => Ok(list.clone()),
        _ => Err("Claude Code 没报模型".into()),
    }
}

/// 起一个只用来问话的 claude（不留会话、不要工具），发一个控制请求，等它回（最多 secs 秒），回的 response 就是结果；问完就停
pub fn control(bin: &Path, request: Value, secs: u64) -> Result<Value, String> {
    let mut cmd = command(bin);
    cmd.args(QUIET).current_dir(work_dir()).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("拉起 {} 失败：{e}", bin.display()))?;
    let mut stdin = child.stdin.take().ok_or("没有 stdin")?;
    let stdout = child.stdout.take().ok_or("没有 stdout")?;
    let mut stderr = child.stderr.take().ok_or("没有 stderr")?;
    let err_reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });
    let (tx, rx) = channel::<Value>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(v) = serde_json::from_str::<Value>(line.trim()) {
                if tx.send(v).is_err() {
                    return;
                }
            }
        }
    });
    let subtype = request["subtype"].as_str().unwrap_or("").to_string();
    let id = format!("friday-{subtype}");
    let line = json!({ "type": "control_request", "request_id": id, "request": request });
    let wrote = writeln!(stdin, "{line}").and_then(|_| stdin.flush());
    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut answer: Result<Value, String> = Err(format!("Claude Code 没回 {subtype}"));
    if let Err(e) = wrote {
        answer = Err(format!("写不进 claude：{e}"));
    } else {
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                answer = Err(format!("等 Claude Code 回 {subtype} 超时（{secs} 秒）"));
                break;
            }
            let Ok(v) = rx.recv_timeout(left.min(Duration::from_millis(500))) else {
                if matches!(child.try_wait(), Ok(Some(_))) {
                    break;
                }
                continue;
            };
            if v["type"].as_str() != Some("control_response") || v["response"]["request_id"].as_str() != Some(id.as_str()) {
                continue;
            }
            let r = &v["response"];
            answer = if r["subtype"].as_str() == Some("success") {
                Ok(r["response"].clone())
            } else {
                Err(r["error"].as_str().unwrap_or("Claude Code 回了出错").to_string())
            };
            break;
        }
    }
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();
    let err = err_reader.join().unwrap_or_default();
    answer.map_err(|e| match err.lines().map(str::trim).rfind(|l| !l.is_empty()) {
        Some(last) => format!("{e}（{}）", last.chars().take(200).collect::<String>()),
        None => e,
    })
}

/// Anthropic 公开的模型目录（Claude Code 自己的 /model 也用它；签过名的公开文件，不用登录）
pub const CATALOG_URL: &str = "https://downloads.claude.ai/model-catalog/v1/catalog.json";

/// 公开目录里 Claude Code（surface cc）那一份的模型：[{id, name, description, section（main / overflow / deprecated）,
/// runtime.effort_levels, min_claude_code_version, offered_on…}]。6 小时内用缓存；取到的存进 <data-dir>/catalog.json，取不到用存下的；
/// 都没有返回空（只列 Claude Code 自己报的那几个）
pub fn catalog() -> Vec<Value> {
    if let Some((at, list)) = &state().lock().unwrap().catalog {
        if at.elapsed() < Duration::from_secs(6 * 3600) {
            return list.clone();
        }
    }
    let saved = data_dir().join("catalog.json");
    let fetched = fetch_catalog();
    let list = match fetched {
        Ok(list) => {
            let _ = std::fs::write(&saved, Value::Array(list.clone()).to_string());
            list
        }
        Err(e) => {
            log(&format!("取模型目录失败：{e}"));
            std::fs::read_to_string(&saved).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).and_then(|v| v.as_array().cloned()).unwrap_or_default()
        }
    };
    // 取不到时也记一下，6 小时内不再试（每次问模型都等它会拖慢菜单）
    state().lock().unwrap().catalog = Some((Instant::now(), list.clone()));
    list
}

fn fetch_catalog() -> Result<Vec<Value>, String> {
    if let Ok(path) = std::env::var("FRIDAY_CLAUDE_CATALOG") {
        // 测试用：本地文件
        let text = std::fs::read_to_string(&path).map_err(|e| format!("读 {path} 失败：{e}"))?;
        return parse_catalog(&text);
    }
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(20))).http_status_as_error(false).build().into();
    let mut response = agent.get(CATALOG_URL).header("User-Agent", format!("friday-claudecode/{}", env!("CARGO_PKG_VERSION"))).call().map_err(|e| e.to_string())?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(format!("HTTP {status}"));
    }
    let text = response.body_mut().with_config().limit(8 << 20).read_to_string().map_err(|e| e.to_string())?;
    parse_catalog(&text)
}

fn parse_catalog(text: &str) -> Result<Vec<Value>, String> {
    let doc: Value = serde_json::from_str(text).map_err(|e| format!("模型目录不是 JSON：{e}"))?;
    let configs = doc["surfaces"]["cc"]["model_selector_config"].as_array().ok_or("模型目录里没有 Claude Code 那一份")?;
    let config = configs.iter().find(|c| c["id"] == "cc").or(configs.first()).ok_or("模型目录里没有 Claude Code 那一份")?;
    Ok(config["models"].as_array().cloned().unwrap_or_default())
}

/// 版本号比大小（「2.1.276」）：a 比 b 旧
pub fn older(a: &str, b: &str) -> bool {
    let parts = |v: &str| v.split(|c: char| !c.is_ascii_digit()).filter(|p| !p.is_empty()).map(|p| p.parse::<u64>().unwrap_or(0)).collect::<Vec<_>>();
    parts(a) < parts(b)
}

/// 额度：claude 的 get_usage（/usage 的数据，claude 用它自己的登录去查；skip_behaviors 不扫本机记录），换成 Friday 的额度报告：
/// {report: {plan, subscription, windows: [{title, remainingPercent, resetAt}], warnings?, fetchedAt}, available}。30 秒内用缓存
pub fn usage(bin: &Path, fresh: bool) -> Result<Value, String> {
    if !fresh {
        if let Some((at, v)) = &state().lock().unwrap().usage {
            if at.elapsed() < Duration::from_secs(30) {
                return Ok(v.clone());
            }
        }
    }
    let answer = control(bin, json!({ "subtype": "get_usage", "skip_behaviors": true }), 40)?;
    let v = usage_report(&answer);
    state().lock().unwrap().usage = Some((Instant::now(), v.clone()));
    Ok(v)
}

/// get_usage 的回答 → Friday 的额度报告（ServiceUsage.Report 的 JSON）
pub fn usage_report(answer: &Value) -> Value {
    let limits = &answer["rate_limits"];
    let mut windows: Vec<Value> = Vec::new();
    let mut push = |title: String, percent: Option<f64>, resets: Option<&str>| {
        let Some(percent) = percent else { return };
        let mut w = json!({ "title": title, "remainingPercent": (100.0 - percent).clamp(0.0, 100.0) });
        if let Some(at) = resets.and_then(iso_seconds) {
            w["resetAt"] = json!(at);
        }
        windows.push(w);
    };
    if let Some(list) = limits["limits"].as_array().filter(|l| !l.is_empty()) {
        // 新版的统一列表：session（5 小时）、weekly_all（每周）、weekly_scoped（某个模型的每周）……
        for item in list {
            let kind = item["kind"].as_str().unwrap_or("");
            let scope = item["scope"]["model"]["display_name"].as_str().or(item["scope"]["surface"].as_str()).unwrap_or("");
            let title = match kind {
                "session" => "5 小时".to_string(),
                "weekly_all" => "每周".to_string(),
                "weekly_scoped" if !scope.is_empty() => format!("每周 {scope}"),
                _ if !scope.is_empty() => format!("{kind} {scope}"),
                _ => kind.to_string(),
            };
            push(title, item["percent"].as_f64(), item["resets_at"].as_str());
        }
    } else if limits.is_object() {
        for (key, title) in [("five_hour", "5 小时"), ("seven_day", "每周"), ("seven_day_opus", "每周 Opus"), ("seven_day_sonnet", "每周 Sonnet")] {
            push(title.to_string(), limits[key]["utilization"].as_f64(), limits[key]["resets_at"].as_str());
        }
        for item in limits["model_scoped"].as_array().into_iter().flatten() {
            push(format!("每周 {}", item["display_name"].as_str().unwrap_or("")), item["utilization"].as_f64(), item["resets_at"].as_str());
        }
    }
    let extra = &limits["extra_usage"];
    if extra["is_enabled"].as_bool() == Some(true) {
        push("额外用量".to_string(), extra["utilization"].as_f64(), None);
    }
    let plan = answer["subscription_type"].as_str().filter(|s| !s.is_empty()).map(|s| {
        let mut c = s.chars();
        c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
    });
    let available = answer["rate_limits_available"].as_bool().unwrap_or(false);
    let mut report = json!({ "balances": [], "windows": windows, "freeTier": false, "fetchedAt": iso_seconds_now() });
    if let Some(plan) = &plan {
        report["plan"] = json!(plan);
        report["subscription"] = json!(true);
    }
    json!({ "report": report, "available": available, "plan": plan })
}

/// 「2026-11-14T12:10:00.079912+00:00」→「2026-11-14T12:10:00Z」（Friday 解 ISO 8601 时不认小数秒和时区偏移）
fn iso_seconds(text: &str) -> Option<String> {
    let text = text.trim();
    let (date, rest) = text.split_once('T')?;
    let time = rest.get(..8)?;
    let tail = &rest[8..];
    let tail = tail.trim_start_matches(|c: char| c == '.' || c.is_ascii_digit());
    let offset_minutes: i64 = match tail {
        "" | "Z" | "z" | "+00:00" | "-00:00" => 0,
        t if t.len() >= 6 && (t.starts_with('+') || t.starts_with('-')) => {
            let sign = if t.starts_with('-') { -1 } else { 1 };
            let h: i64 = t[1..3].parse().ok()?;
            let m: i64 = t[4..6].parse().ok()?;
            sign * (h * 60 + m)
        }
        _ => return None,
    };
    let secs = unix_of(date, time)? - offset_minutes * 60;
    Some(iso_of(secs))
}

fn iso_seconds_now() -> String {
    iso_of(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0))
}

/// 「2026-11-14」「12:10:00」→ Unix 秒（UTC）
fn unix_of(date: &str, time: &str) -> Option<i64> {
    let mut d = date.split('-').map(|p| p.parse::<i64>().ok());
    let (y, mo, da) = (d.next()??, d.next()??, d.next()??);
    let mut t = time.split(':').map(|p| p.parse::<i64>().ok());
    let (h, mi, se) = (t.next()??, t.next()??, t.next()??);
    // 公历日期 → 自 1970-01-01 的天数（Howard Hinnant 的 days_from_civil）
    let y2 = if mo <= 2 { y - 1 } else { y };
    let era = if y2 >= 0 { y2 } else { y2 - 399 } / 400;
    let yoe = y2 - era * 400;
    let doy = (153 * (if mo > 2 { mo - 3 } else { mo + 9 }) + 2) / 5 + da - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Some(days * 86400 + h * 3600 + mi * 60 + se)
}

fn iso_of(secs: i64) -> String {
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// 给人看的一句：「Claude Code 2.1.276 · Max 订阅」/「未登录：在终端运行 claude auth login」
pub fn detail(version: &str, auth: Option<&Value>) -> String {
    let head = if version.is_empty() { "Claude Code".to_string() } else { format!("Claude Code {version}") };
    let Some(a) = auth else { return head };
    if a["loggedIn"].as_bool() != Some(true) {
        return "未登录：在终端运行 claude auth login".into();
    }
    let method = a["authMethod"].as_str().unwrap_or("");
    let provider = a["apiProvider"].as_str().unwrap_or("");
    let how = match (method, provider) {
        (_, "bedrock") => "Amazon Bedrock".to_string(),
        (_, "vertex") => "Google Vertex AI".to_string(),
        (_, "foundry") => "Microsoft Foundry".to_string(),
        ("claude.ai", _) => match a["subscriptionType"].as_str().unwrap_or("") {
            "" => "Claude 订阅".to_string(),
            s => {
                let mut c = s.chars();
                let name: String = c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default();
                format!("{name} 订阅")
            }
        },
        ("api_key", _) | ("apiKey", _) => "API 密钥（按量计费）".to_string(),
        ("", _) => String::new(),
        (m, _) => m.to_string(),
    };
    if how.is_empty() {
        head
    } else {
        format!("{head} · {how}")
    }
}

pub fn random_hex(n: usize) -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut out = String::new();
    let mut i = 0u64;
    while out.len() < n {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
        h.write_u64(i);
        h.write_u32(std::process::id());
        out.push_str(&format!("{:016x}", h.finish()));
        i += 1;
    }
    out.truncate(n);
    out
}
