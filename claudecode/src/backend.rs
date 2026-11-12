//! 找本机装着的 Claude Code（`claude` 命令行，用户自己装、自己登录的那个，原样调用，不改它）：
//! FRIDAY_CLAUDE_BIN、PATH、~/.local/bin、~/.claude/local、Homebrew、/usr/local/bin，Windows 上 %USERPROFILE%\.local\bin\claude.exe、%APPDATA%\npm\claude.cmd。
//! 版本看 `claude --version`；登录状态只看 `claude auth status` 自己打印的 JSON，而且只留 loggedIn / authMethod / apiProvider / subscriptionType
//! 四项（邮箱、组织 id 等一概不留）。本程序**不读、不碰** Claude Code 的登录凭据（钥匙串、~/.claude/.credentials.json、OAuth 令牌）。
//! 拉起 claude 时继承本程序的环境，但去掉从上层 Claude Code 会话漏下来的 CLAUDECODE、CLAUDE_CODE_* 等（带着它们 claude 会认错登录）。

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

struct State {
    data_dir: PathBuf,
    /// 本程序的地址（给 mcp 子命令回连）和内部口令（只给 mcp 子命令用，和宿主的 --token 分开）
    server: String,
    internal_token: String,
    /// claude --version 的缓存：（程序路径, 修改时间）→ 版本
    version: Option<(PathBuf, Option<std::time::SystemTime>, String)>,
    /// claude auth status 的缓存（60 秒）
    auth: Option<(Instant, Value)>,
}

fn state() -> &'static Mutex<State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(|| {
        Mutex::new(State { data_dir: PathBuf::from("."), server: String::new(), internal_token: String::new(), version: None, auth: None })
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
