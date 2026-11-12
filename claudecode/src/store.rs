//! Friday 的一段对话 → Claude Code 的一个会话（session id，存在 <data-dir>/sessions.json）：
//! 常驻的 claude 进程停了（闲置收掉、换模型 / 权限档、Friday 点了停止、本程序重启）以后用 `--resume <id>` 接回同一个会话，
//! Claude Code 记得前面聊过的。会话本身是 Claude Code 自己存的（在用户的 Claude Code 配置目录里），这里只记 id。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct Entry {
    /// Claude Code 的 session_id
    pub session: String,
    /// 工作目录（Claude Code 的会话按目录存，换了目录就接不回）
    pub dir: String,
    pub model: String,
    pub used: u64,
}

struct Store {
    path: PathBuf,
    entries: HashMap<String, Entry>,
}

fn store() -> &'static Mutex<Store> {
    static S: OnceLock<Mutex<Store>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Store { path: PathBuf::new(), entries: HashMap::new() }))
}

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// 60 天没用过的对话不再记（Claude Code 那边的会话留着，用户用 claude --resume 还看得到）
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
    if s.path.as_os_str().is_empty() {
        return;
    }
    let map: serde_json::Map<String, Value> = s
        .entries
        .iter()
        .map(|(k, e)| (k.clone(), json!({ "session": e.session, "dir": e.dir, "model": e.model, "used": e.used })))
        .collect();
    let tmp = s.path.with_extension("json.tmp");
    if std::fs::write(&tmp, Value::Object(map).to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, &s.path);
    }
}

pub fn get(key: &str) -> Option<Entry> {
    store().lock().unwrap().entries.get(key).cloned()
}

/// 记下（或更新）这段对话的会话 id
pub fn put(key: &str, session: &str, dir: &str, model: &str) {
    let mut s = store().lock().unwrap();
    let changed = s.entries.get(key).map(|e| e.session != session || e.model != model || now().saturating_sub(e.used) > 60).unwrap_or(true);
    if !changed {
        return;
    }
    s.entries.insert(key.to_string(), Entry { session: session.to_string(), dir: dir.to_string(), model: model.to_string(), used: now() });
    save(&s);
}

pub fn remove(key: &str) {
    let mut s = store().lock().unwrap();
    if s.entries.remove(key).is_some() {
        save(&s);
    }
}

pub fn count() -> usize {
    store().lock().unwrap().entries.len()
}
