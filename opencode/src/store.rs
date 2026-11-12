//! Friday 的一段对话 → opencode 的一个会话（存在 <data-dir>/sessions.json，重启后接着用同一个会话，opencode 记得前面聊过的）；
//! 以及停在「等用户确认权限 / 等用户回答提问」的会话（只在内存里；本程序重启后丢了，下一句话进来时再问 opencode 这个会话有没有在等的，见 chat.rs）。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct Entry {
    pub session: String,
    pub dir: String,
    pub model: String,
    /// 连的哪个服务（换了服务，旧会话 id 在新服务上可能不存在，按 404 重建）
    pub base: String,
    pub used: u64,
}

#[derive(Clone, Debug)]
pub struct Pending {
    /// opencode 的权限请求 id（回复它）；提问时是表单 id（frm_…）
    pub request: String,
    /// 等确认的那个工具（结构化事件里的 {id, name, title, summary, input}）：放行后接着那次请求报它做完时带上名字和参数
    /// （客户端那边原来那行可能已经不在了：重开过、远程的存档里那一轮已经结束）
    pub tool: Option<serde_json::Value>,
    /// 提问（opencode 的 question 工具建的表单 form.created 的 form：{id, title, metadata, fields}）；None = 权限确认
    pub form: Option<serde_json::Value>,
}

struct Store {
    path: PathBuf,
    entries: HashMap<String, Entry>,
    pending: HashMap<String, Pending>,
}

fn store() -> &'static Mutex<Store> {
    static S: OnceLock<Mutex<Store>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Store { path: PathBuf::new(), entries: HashMap::new(), pending: HashMap::new() }))
}

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// 60 天没用过的对话不再记（opencode 那边的会话留着，用户在 opencode 里还看得到）
const KEEP_SECS: u64 = 60 * 24 * 3600;

pub fn init(data_dir: &std::path::Path) {
    let mut s = store().lock().unwrap();
    s.path = data_dir.join("sessions.json");
    if let Ok(text) = std::fs::read_to_string(&s.path) {
        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&text) {
            let cutoff = now().saturating_sub(KEEP_SECS);
            for (key, v) in map {
                let e = Entry {
                    session: v["session"].as_str().unwrap_or("").to_string(),
                    dir: v["dir"].as_str().unwrap_or("").to_string(),
                    model: v["model"].as_str().unwrap_or("").to_string(),
                    base: v["base"].as_str().unwrap_or("").to_string(),
                    used: v["used"].as_u64().unwrap_or(0),
                };
                if !e.session.is_empty() && e.used >= cutoff {
                    s.entries.insert(key, e);
                }
            }
        }
    }
}

fn save(s: &Store) {
    let map: serde_json::Map<String, Value> = s
        .entries
        .iter()
        .map(|(k, e)| (k.clone(), json!({ "session": e.session, "dir": e.dir, "model": e.model, "base": e.base, "used": e.used })))
        .collect();
    let tmp = s.path.with_extension("json.tmp");
    if std::fs::write(&tmp, Value::Object(map).to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, &s.path);
    }
}

pub fn get(key: &str) -> Option<Entry> {
    store().lock().unwrap().entries.get(key).cloned()
}

pub fn put(key: &str, entry: Entry) {
    let mut s = store().lock().unwrap();
    s.entries.insert(key.to_string(), entry);
    save(&s);
}

pub fn touch(key: &str, model: &str) {
    let mut s = store().lock().unwrap();
    if let Some(e) = s.entries.get_mut(key) {
        e.used = now();
        e.model = model.to_string();
        save(&s);
    }
}

pub fn pending(key: &str) -> Option<Pending> {
    store().lock().unwrap().pending.get(key).cloned()
}

pub fn set_pending(key: &str, p: Option<Pending>) {
    let mut s = store().lock().unwrap();
    match p {
        Some(p) => s.pending.insert(key.to_string(), p),
        None => s.pending.remove(key),
    };
}

pub fn count() -> usize {
    store().lock().unwrap().entries.len()
}
