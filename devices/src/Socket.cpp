// 见 Socket.h
#include "Socket.h"

#include <cstring>
#include <mutex>

namespace devices::net {

void initialize()
{
#ifdef _WIN32
    static std::once_flag once;
    std::call_once(once, [] {
        WSADATA data {};
        WSAStartup(MAKEWORD(2, 2), &data);
    });
#endif
}

Handle listenLocal(int port, int* actualPort, std::string* error)
{
    initialize();
    const Handle handle = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (handle == kInvalid) {
        if (error)
            *error = "建不了 socket";
        return kInvalid;
    }
#ifdef _WIN32
    // 不让别的进程抢同一个端口
    BOOL exclusive = TRUE;
    setsockopt(handle, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&exclusive), sizeof(exclusive));
#else
    int reuse = 1;
    setsockopt(handle, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#endif
    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_port = htons(uint16_t(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(handle, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 || ::listen(handle, 16) != 0) {
        if (error)
            *error = "端口 " + std::to_string(port) + " 用不了（被占用？）";
        closeHandle(handle);
        return kInvalid;
    }
    socklen_t length = sizeof(address);
    if (::getsockname(handle, reinterpret_cast<sockaddr*>(&address), &length) == 0 && actualPort)
        *actualPort = ntohs(address.sin_port);
    return handle;
}

Handle acceptOne(Handle listener)
{
    sockaddr_in peer {};
    socklen_t length = sizeof(peer);
    const Handle handle = ::accept(listener, reinterpret_cast<sockaddr*>(&peer), &length);
    if (handle == kInvalid)
        return kInvalid;
    // 交互式的串口 / adb：别攒包
    int noDelay = 1;
    setsockopt(handle, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&noDelay), sizeof(noDelay));
    return handle;
}

long receiveSome(Handle socket, char* buffer, size_t size)
{
#ifdef _WIN32
    const int n = ::recv(socket, buffer, int(size), 0);
#else
    const ssize_t n = ::recv(socket, buffer, size, 0);
#endif
    return n < 0 ? -1 : long(n);
}

bool sendAll(Handle socket, const char* data, size_t size)
{
    while (size > 0) {
#ifdef _WIN32
        const int n = ::send(socket, data, int(size), 0);
#else
        const ssize_t n = ::send(socket, data, size, MSG_NOSIGNAL);
#endif
        if (n <= 0)
            return false;
        data += n;
        size -= size_t(n);
    }
    return true;
}

void shutdownBoth(Handle socket)
{
#ifdef _WIN32
    ::shutdown(socket, SD_BOTH);
#else
    ::shutdown(socket, SHUT_RDWR);
#endif
}

void closeHandle(Handle socket)
{
#ifdef _WIN32
    ::closesocket(socket);
#else
    ::close(socket);
#endif
}

} // namespace devices::net
