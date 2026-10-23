// 本机 TCP 的一层薄封装（Windows Winsock / POSIX）：只监听 127.0.0.1，阻塞读写，各连接一个线程。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace devices::net {

#ifdef _WIN32
using Handle = SOCKET;
inline const Handle kInvalid = INVALID_SOCKET;
#else
using Handle = int;
inline const Handle kInvalid = -1;
#endif

/// 进程里第一次用 socket 前调一次（Windows 的 WSAStartup；civetweb 也会调，重复无害）
void initialize();

/// 在 127.0.0.1:port 上监听（port 0 = 系统分配）；成功时 *actualPort 是实际端口，失败返回 kInvalid、*error 是原因
Handle listenLocal(int port, int* actualPort, std::string* error);

/// 等一个连接进来；监听被关掉时返回 kInvalid
Handle acceptOne(Handle listener);

/// 读一块：返回读到的字节数，0 = 对方关了，-1 = 出错 / 被关掉
long receiveSome(Handle socket, char* buffer, size_t size);

/// 全部写出去：失败返回 false
bool sendAll(Handle socket, const char* data, size_t size);

/// 两个方向都停（阻塞在读写上的线程会醒）
void shutdownBoth(Handle socket);

void closeHandle(Handle socket);

} // namespace devices::net
