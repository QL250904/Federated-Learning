"""
Network Utilities cho Federated Learning trên mạng LAN thực.
Hỗ trợ: gửi/nhận dữ liệu lớn qua TCP socket với:
  - Header 8 bytes (hỗ trợ dữ liệu >4GB)
  - Gửi theo chunk để tránh tràn bộ đệm
  - Timeout và retry tự động
  - Keep-alive cho kết nối dài
"""

import socket
import struct
import pickle
import time
import sys

# Tương thích Numpy 2.x <-> 1.x
try:
    import numpy.core.numeric
    import numpy.core.multiarray
    sys.modules['numpy._core'] = sys.modules.get('numpy._core', sys.modules['numpy.core'])
    sys.modules['numpy._core.numeric'] = sys.modules.get('numpy._core.numeric', sys.modules['numpy.core.numeric'])
    sys.modules['numpy._core.multiarray'] = sys.modules.get('numpy._core.multiarray', sys.modules['numpy.core.multiarray'])
except Exception:
    pass

CHUNK_SIZE = 65536  # 64KB per send
HEADER_FORMAT = '>Q'  # unsigned long long, 8 bytes
HEADER_SIZE = 8


def configure_socket(sock, timeout=300):
    """Cấu hình socket cho kết nối ổn định trên LAN."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Tăng buffer size
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB
    # Keep-alive
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    # Disable Nagle algorithm for lower latency
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # Timeout
    if timeout:
        sock.settimeout(timeout)


def send_msg(sock, msg):
    """Gói dữ liệu bằng pickle và gửi kèm header 8 bytes (kích thước)."""
    msg_bytes = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack(HEADER_FORMAT, len(msg_bytes))
    sock.sendall(header)
    # Gửi theo chunk để tránh tràn bộ đệm OS
    sent = 0
    while sent < len(msg_bytes):
        end = min(sent + CHUNK_SIZE, len(msg_bytes))
        sock.sendall(msg_bytes[sent:end])
        sent = end


def recvall(sock, n):
    """Nhận đủ n bytes từ socket."""
    data = bytearray()
    while len(data) < n:
        try:
            packet = sock.recv(min(CHUNK_SIZE, n - len(data)))
        except socket.timeout:
            print(f"[WARN] Socket timeout khi nhận dữ liệu ({len(data)}/{n} bytes)")
            raise
        if not packet:
            if len(data) == 0:
                return None
            raise ConnectionError(f"Kết nối bị đóng khi đang nhận ({len(data)}/{n} bytes)")
        data.extend(packet)
    return bytes(data)


def recv_msg(sock):
    """Đọc header 8 bytes và nhận toàn bộ dữ liệu."""
    raw_header = recvall(sock, HEADER_SIZE)
    if raw_header is None:
        return None
    msglen = struct.unpack(HEADER_FORMAT, raw_header)[0]
    msg_bytes = recvall(sock, msglen)
    if msg_bytes is None:
        return None
    return pickle.loads(msg_bytes)


def create_server(port, num_clients, timeout=300):
    """Tạo server socket và chờ kết nối từ clients."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    configure_socket(server, timeout=None)  # Server socket không cần timeout
    server.bind(('0.0.0.0', port))
    server.listen(num_clients + 1)
    print(f"[SERVER] Đang lắng nghe trên port {port}...")

    clients = []
    for i in range(num_clients):
        print(f"[SERVER] Chờ Client {i+1}/{num_clients}...")
        conn, addr = server.accept()
        configure_socket(conn, timeout=timeout)
        print(f"[SERVER] ✓ Client {i+1} đã kết nối từ {addr}")
        clients.append(conn)

    return server, clients


def connect_to_server(server_ip, port, max_retries=10, retry_delay=3, timeout=300):
    """Kết nối đến server với retry tự động."""
    for attempt in range(1, max_retries + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            configure_socket(sock, timeout=timeout)
            print(f"[CLIENT] Đang kết nối đến {server_ip}:{port} (lần {attempt}/{max_retries})...")
            sock.connect((server_ip, port))
            print(f"[CLIENT] ✓ Đã kết nối thành công!")
            return sock
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f"[CLIENT] ✗ Lỗi: {e}")
            if attempt < max_retries:
                print(f"[CLIENT] Thử lại sau {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                raise ConnectionError(
                    f"Không thể kết nối đến {server_ip}:{port} sau {max_retries} lần thử.\n"
                    f"Kiểm tra: (1) Server đã chạy chưa? (2) IP đúng chưa? (3) Firewall?"
                )
