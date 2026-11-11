//! 找到要连的 opencode：
//! 1. 正在跑的后台服务（桌面版或命令行的 `opencode service`）：`<state>/opencode/service.json` 记着 url、pid、password、version；
//!    用它就是用户自己的 opencode（登录、额度、身份都是它的），优先；
//! 2. 没有就找本机装着的 opencode（FRIDAY_OPENCODE_BIN、PATH、~/.opencode/bin、桌面版自带的 opencode-cli），自己拉起一个
//!    `opencode serve`（随机密码，跟着本程序退出），和用户共用数据目录（登录一致）；
//! 3. 都没有：报「需要装 OpenCode」。
//! 连不上时（桌面版重开换了端口）调用方 reset()，下次重新找。

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::oc::{self, Conn};

#[derive(Clone, Debug)]
pub struct Backend {
    pub conn: Conn,
    /// service（用户的后台服务）| private（本程序拉起的）
    pub kind: &'static str,
    pub version: String,
    /// 给人看：哪个程序、在哪
    pub detail: String,
}

impl Backend {
    pub fn describe(&self) -> Value {
        json!({ "kind": self.kind, "url": self.conn.base, "version": self.version, "detail": self.detail })
    }
}

struct State {
    current: Option<Backend>,
    /// 本程序拉起的 opencode 和连它的方式
    child: Option<Child>,
    private: Option<Backend>,
    data_dir: PathBuf,
}

fn state() -> &'static Mutex<State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(State { current: None, child: None, private: None, data_dir: PathBuf::from(".") }))
}

pub fn init(data_dir: &Path) {
    state().lock().unwrap().data_dir = data_dir.to_path_buf();
}

pub fn data_dir() -> PathBuf {
    state().lock().unwrap().data_dir.clone()
}

/// 诊断日志：<data-dir>/friday-opencode.log（也打到 stderr）；超过 1 MB 从头写
pub fn log(line: &str) {
    use std::io::Write;
    eprintln!("{line}");
    let path = data_dir().join("friday-opencode.log");
    let truncate = std::fs::metadata(&path).map(|m| m.len() > 1 << 20).unwrap_or(false);
    let file = std::fs::OpenOptions::new().create(true).append(!truncate).write(true).truncate(truncate).open(&path);
    if let Ok(mut f) = file {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let _ = writeln!(f, "{now} {line}");
    }
}

/// 当前能用的后端（没有就找 / 拉起）
pub fn current() -> Result<Backend, String> {
    let mut s = state().lock().unwrap();
    if let Some(b) = &s.current {
        return Ok(b.clone());
    }
    let found = discover(&mut s);
    if let Ok(b) = &found {
        s.current = Some(b.clone());
    }
    found
}

/// 连不上了：丢掉，下次重新找（自己拉起的那个已经退了的话也清掉）
pub fn reset() {
    let mut s = state().lock().unwrap();
    s.current = None;
    if let Some(child) = s.child.as_mut() {
        if !matches!(child.try_wait(), Ok(None)) {
            s.child = None;
            s.private = None;
        }
    }
}

pub fn shutdown() {
    let mut s = state().lock().unwrap();
    if let Some(mut child) = s.child.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from)
}

fn xdg(var: &str, fallback: &[&str]) -> Option<PathBuf> {
    if let Some(v) = std::env::var_os(var) {
        if !v.is_empty() {
            return Some(PathBuf::from(v));
        }
    }
    let mut p = home()?;
    for part in fallback {
        p.push(part);
    }
    Some(p)
}

fn discover(s: &mut State) -> Result<Backend, String> {
    // 测试 / 手动指定：FRIDAY_OPENCODE_URL（+ FRIDAY_OPENCODE_PASSWORD）
    if let Some(url) = std::env::var("FRIDAY_OPENCODE_URL").ok().filter(|u| !u.is_empty()) {
        let password = std::env::var("FRIDAY_OPENCODE_PASSWORD").unwrap_or_default();
        let conn = Conn { base: url.trim_end_matches('/').to_string(), auth: oc::basic_auth(&password) };
        return match oc::get_quick(&conn, "/api/agent") {
            Ok((200, _)) => Ok(Backend { conn, kind: "configured", version: String::new(), detail: format!("FRIDAY_OPENCODE_URL（{url}）") }),
            Ok((status, _)) => Err(format!("FRIDAY_OPENCODE_URL 指的 opencode 回了 HTTP {status}（密码对吗？）")),
            Err(e) => Err(e),
        };
    }
    if let Some(b) = from_service() {
        return Ok(b);
    }
    // 自己拉起的那个还活着就接着用
    if let Some(child) = s.child.as_mut() {
        if matches!(child.try_wait(), Ok(None)) {
            if let Some(b) = s.private.clone() {
                return Ok(b);
            }
        }
        s.child = None;
        s.private = None;
    }
    let Some(bin) = find_binary() else {
        return Err("没找到 OpenCode：请装 OpenCode 桌面版（https://opencode.ai/download），或命令行版（curl -fsSL https://opencode.ai/v2/install | bash），装好后打开一次".into());
    };
    let version = binary_version(&bin).unwrap_or_default();
    // V1（1.x）的接口不一样；V2 的测试版是 0.0.0-beta-…，也算 V2
    if version.starts_with("1.") {
        return Err(format!("本机的 OpenCode 是 {version}，要 2.x（OpenCode V2）：请升级"));
    }
    let (backend, child) = spawn_private(&bin, &version, &s.data_dir)?;
    s.child = Some(child);
    s.private = Some(backend.clone());
    Ok(backend)
}

/// 用户的后台服务（桌面版开着时就是它）
fn from_service() -> Option<Backend> {
    let path = xdg("XDG_STATE_HOME", &[".local", "state"])?.join("opencode").join("service.json");
    let text = std::fs::read_to_string(&path).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let url = v.get("url")?.as_str()?.trim_end_matches('/').to_string();
    let password = v.get("password")?.as_str()?.to_string();
    let conn = Conn { base: url.clone(), auth: oc::basic_auth(&password) };
    match oc::get_quick(&conn, "/api/agent") {
        Ok((200, _)) => Some(Backend {
            conn,
            kind: "service",
            version: v.get("version").and_then(Value::as_str).unwrap_or("").to_string(),
            detail: format!("OpenCode 后台服务（{url}）"),
        }),
        _ => None,
    }
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

/// 某个目录下按名字排序最大的子目录（桌面版自带的 cli/<版本>-<架构>/）
fn latest_child(dir: &Path) -> Option<PathBuf> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    entries.sort_by(|a, b| version_key(a).cmp(&version_key(b)));
    entries.pop()
}

fn version_key(p: &Path) -> Vec<u64> {
    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
    name.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).map(|s| s.parse().unwrap_or(0)).collect()
}

pub fn find_binary() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("FRIDAY_OPENCODE_BIN") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let p = dir.join(exe("opencode"));
            if p.is_file() {
                return Some(p);
            }
        }
    }
    let home = home()?;
    let p = home.join(".opencode").join("bin").join(exe("opencode"));
    if p.is_file() {
        return Some(p);
    }
    // 桌面版自带的命令行（macOS：~/Library/Application Support/ai.opencode.desktop/cli/<版本>-<架构>/opencode-cli）
    let mut roots = vec![home.join("Library").join("Application Support").join("ai.opencode.desktop").join("cli")];
    if let Some(d) = std::env::var_os("LOCALAPPDATA").or_else(|| std::env::var_os("APPDATA")) {
        roots.push(PathBuf::from(d).join("ai.opencode.desktop").join("cli"));
    }
    if let Some(d) = xdg("XDG_CONFIG_HOME", &[".config"]) {
        roots.push(d.join("ai.opencode.desktop").join("cli"));
    }
    if let Some(d) = xdg("XDG_DATA_HOME", &[".local", "share"]) {
        roots.push(d.join("ai.opencode.desktop").join("cli"));
    }
    for root in roots {
        if let Some(dir) = latest_child(&root) {
            let p = dir.join(exe("opencode-cli"));
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// `opencode --version` → 2.2.7
pub fn binary_version(bin: &Path) -> Option<String> {
    let out = Command::new(bin).arg("--version").stdin(Stdio::null()).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    text.split_whitespace().map(|w| w.trim_start_matches('v')).find(|w| w.starts_with(|c: char| c.is_ascii_digit())).map(str::to_string)
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

fn spawn_private(bin: &Path, version: &str, data_dir: &Path) -> Result<(Backend, Child), String> {
    let work = data_dir.join("work");
    let _ = std::fs::create_dir_all(&work);
    let password = random_hex(40);
    let mut child = Command::new(bin)
        .args(["serve", "--hostname", "127.0.0.1", "--port", "0"])
        .current_dir(&work)
        .env("OPENCODE_SERVER_PASSWORD", &password)
        .env("OPENCODE_DISABLE_AUTOUPDATE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("拉起 opencode 失败（{}）：{e}", bin.display()))?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            eprintln!("[opencode] {line}");
        }
    });
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        let mut sent = false;
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if !sent {
                if let Some(i) = line.find("http://") {
                    let url: String = line[i..].split_whitespace().next().unwrap_or("").to_string();
                    if line.contains("listening") {
                        let _ = tx.send(url);
                        sent = true;
                        continue;
                    }
                }
            }
            if !line.contains("password") {
                eprintln!("[opencode] {line}");
            }
        }
    });
    let deadline = Instant::now() + Duration::from_secs(60);
    let url = loop {
        if let Ok(url) = rx.recv_timeout(Duration::from_millis(200)) {
            break url;
        }
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!("opencode 启动后退出了（{status}）"));
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            return Err("opencode 60 秒内没起来".into());
        }
    };
    let conn = Conn { base: url.trim_end_matches('/').to_string(), auth: oc::basic_auth(&password) };
    Ok((Backend { conn, kind: "private", version: version.to_string(), detail: format!("本程序拉起的 {}", bin.display()) }, child))
}
