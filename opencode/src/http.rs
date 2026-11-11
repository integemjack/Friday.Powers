//! 极简 HTTP/1.1 服务端：一个连接一个线程、读一个请求、回一个响应就关（Connection: close）。
//! 自己写而不用库：流式（SSE）每段要立刻刷出去，库的分块缓冲会攒着不发。

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

const MAX_HEADER: usize = 64 * 1024;
const MAX_BODY: usize = 256 * 1024 * 1024;

/// 读一个请求；连接一上来就关了返回 None
pub fn read_request(stream: &TcpStream) -> io::Result<Option<Request>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.trim_end().splitn(3, ' ');
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    let path = target.split('?').next().unwrap_or("/").to_string();
    let mut headers = Vec::new();
    let mut total = line.len();
    loop {
        let mut h = String::new();
        let n = reader.read_line(&mut h)?;
        total += n;
        if n == 0 || total > MAX_HEADER {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "请求头太长或不完整"));
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    let length = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse::<usize>().ok())
        .unwrap_or(0);
    if length > MAX_BODY {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "请求体太大"));
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body)?;
    Ok(Some(Request { method, path, headers, body }))
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Status",
    }
}

/// 一次性回完
pub fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8], extra: &[(&str, String)]) -> io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n",
        reason(status),
        body.len()
    );
    for (k, v) in extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

pub fn json(stream: &mut TcpStream, status: u16, value: &serde_json::Value) -> io::Result<()> {
    respond(stream, status, "application/json; charset=utf-8", value.to_string().as_bytes(), &[])
}

/// OpenAI 样式的错误体：{"error": {"message", "type", "code"}}
pub fn error(stream: &mut TcpStream, status: u16, kind: &str, message: &str, extra: &[(&str, String)]) -> io::Result<()> {
    let body = serde_json::json!({ "error": { "message": message, "type": kind, "code": kind } });
    respond(stream, status, "application/json; charset=utf-8", body.to_string().as_bytes(), extra)
}

/// SSE 响应（分块编码）：每段写完立刻刷；写不出去 = 客户端走了
pub struct Sse<'a> {
    stream: &'a mut TcpStream,
    pub alive: bool,
}

impl<'a> Sse<'a> {
    pub fn start(stream: &'a mut TcpStream) -> io::Result<Self> {
        stream.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
        )?;
        stream.flush()?;
        Ok(Sse { stream, alive: true })
    }

    fn chunk(&mut self, payload: &[u8]) -> bool {
        if !self.alive {
            return false;
        }
        let mut out = format!("{:x}\r\n", payload.len()).into_bytes();
        out.extend_from_slice(payload);
        out.extend_from_slice(b"\r\n");
        if self.stream.write_all(&out).and_then(|_| self.stream.flush()).is_err() {
            self.alive = false;
        }
        self.alive
    }

    pub fn data(&mut self, value: &serde_json::Value) -> bool {
        self.chunk(format!("data: {value}\n\n").as_bytes())
    }

    /// SSE 注释行（客户端忽略）：保活、探测客户端还在不在
    pub fn comment(&mut self) -> bool {
        self.chunk(b": ping\n\n")
    }

    pub fn done(&mut self) {
        self.chunk(b"data: [DONE]\n\n");
        if self.alive {
            let _ = self.stream.write_all(b"0\r\n\r\n");
            let _ = self.stream.flush();
        }
    }
}
