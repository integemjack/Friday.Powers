# friday-claudecode 用到的第三方代码

静态链接进 friday-claudecode 的 Rust 包（都在 crates.io，许可证见各包）：

- serde_json（MIT / Apache-2.0）
- sha2、digest 等 RustCrypto 包（MIT / Apache-2.0）
- libc（MIT / Apache-2.0，只在 macOS / Linux 上）

friday-claudecode 不带 Claude Code 本身：它调用用户自己装好、自己登录的 Claude Code 命令行（Anthropic），原样使用，不修改、不分发。
