#pragma once
// BSD sockets on Linux / macOS and Winsock on Windows behind one small surface. Sockets stay `int` (-1 = invalid) so
// the callers' `fd < 0` checks keep working; on Windows the SOCKET handle is narrowed, which holds for the handle values
// Windows hands out (small kernel handle indices). Only the calls that differ are wrapped.
#include <cstdint>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <BaseTsd.h>
#pragma comment(lib, "ws2_32.lib")
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef SSIZE_T ssize_t;
#endif
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0  // Windows has no SIGPIPE: a send on a closed socket returns an error instead
#endif
#ifndef MSG_DONTWAIT
#define MSG_DONTWAIT 0  // callers only peek after poll() reported the socket readable
#endif
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace tqnet {
#ifdef _WIN32
inline bool startup() {
    static const bool ok = [] { WSADATA d; return WSAStartup(MAKEWORD(2, 2), &d) == 0; }();
    return ok;
}
inline SOCKET raw(int s) { return SOCKET(uintptr_t(unsigned(s))); }
inline int from_raw(SOCKET s) { return s == INVALID_SOCKET ? -1 : int(s); }
inline int socket(int family, int type, int protocol) { startup(); return from_raw(::socket(family, type, protocol)); }
inline int accept(int s) { return from_raw(::accept(raw(s), nullptr, nullptr)); }
inline int close(int s) { return ::closesocket(raw(s)); }
inline ssize_t send(int s, const void * data, size_t bytes, int flags) {
    return ::send(raw(s), static_cast<const char *>(data), int(bytes > 0x7fffffff ? 0x7fffffff : bytes), flags);
}
inline ssize_t recv(int s, void * data, size_t bytes, int flags) {
    return ::recv(raw(s), static_cast<char *>(data), int(bytes > 0x7fffffff ? 0x7fffffff : bytes), flags);
}
using pollfd_t = WSAPOLLFD;
inline int poll(pollfd_t * fds, unsigned n, int timeout_ms) { return ::WSAPoll(fds, ULONG(n), timeout_ms); }
inline pollfd_t poll_entry(int s, short events) { pollfd_t p{}; p.fd = raw(s); p.events = events; return p; }
inline int last_error() { return WSAGetLastError(); }
inline bool interrupted(int e) { return e == WSAEINTR; }
inline bool would_block(int e) { return e == WSAEWOULDBLOCK; }
inline bool connect_pending(int e) { return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS; }
inline bool set_nonblocking(int s, bool on) { u_long v = on ? 1 : 0; return ::ioctlsocket(raw(s), FIONBIO, &v) == 0; }
inline bool set_recv_timeout(int s, int seconds) {
    const DWORD ms = DWORD(seconds) * 1000;
    return ::setsockopt(raw(s), SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char *>(&ms), sizeof(ms)) == 0;
}
inline bool set_send_timeout(int s, int seconds) {
    const DWORD ms = DWORD(seconds) * 1000;
    return ::setsockopt(raw(s), SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char *>(&ms), sizeof(ms)) == 0;
}
inline int shutdown_both(int s) { return ::shutdown(raw(s), SD_BOTH); }
inline bool set_reuse_address(int s) {
    const BOOL excl = TRUE;  // Windows SO_REUSEADDR lets another socket steal the port: exclusive use instead
    return ::setsockopt(raw(s), SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char *>(&excl), sizeof(excl)) == 0;
}
inline int socket_error(int s) {
    int err = 0; int len = sizeof(err);
    return ::getsockopt(raw(s), SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&err), &len) == 0 ? err : -1;
}
#else
inline bool startup() { return true; }
inline int raw(int s) { return s; }
inline int socket(int family, int type, int protocol) { return ::socket(family, type, protocol); }
inline int accept(int s) { return ::accept(s, nullptr, nullptr); }
inline int close(int s) { return ::close(s); }
inline ssize_t send(int s, const void * data, size_t bytes, int flags) { return ::send(s, data, bytes, flags); }
inline ssize_t recv(int s, void * data, size_t bytes, int flags) { return ::recv(s, data, bytes, flags); }
using pollfd_t = struct pollfd;
inline int poll(pollfd_t * fds, unsigned n, int timeout_ms) { return ::poll(fds, nfds_t(n), timeout_ms); }
inline pollfd_t poll_entry(int s, short events) { pollfd_t p{}; p.fd = s; p.events = events; return p; }
inline int last_error() { return errno; }
inline bool interrupted(int e) { return e == EINTR; }
inline bool would_block(int e) { return e == EAGAIN || e == EWOULDBLOCK; }
inline bool connect_pending(int e) { return e == EINPROGRESS; }
inline bool set_nonblocking(int s, bool on) {
    const int fl = ::fcntl(s, F_GETFL, 0);
    return fl >= 0 && ::fcntl(s, F_SETFL, on ? (fl | O_NONBLOCK) : (fl & ~O_NONBLOCK)) == 0;
}
inline bool set_recv_timeout(int s, int seconds) {
    timeval t{}; t.tv_sec = seconds;
    return ::setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &t, sizeof(t)) == 0;
}
inline bool set_send_timeout(int s, int seconds) {
    timeval t{}; t.tv_sec = seconds;
    return ::setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &t, sizeof(t)) == 0;
}
inline int shutdown_both(int s) { return ::shutdown(s, SHUT_RDWR); }
inline bool set_reuse_address(int s) { const int one = 1; return ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) == 0; }
inline int socket_error(int s) {
    int err = 0; socklen_t len = sizeof(err);
    return ::getsockopt(s, SOL_SOCKET, SO_ERROR, &err, &len) == 0 ? err : -1;
}
#endif
}  // namespace tqnet
