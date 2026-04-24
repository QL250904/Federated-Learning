import socket
import struct
import pickle

def send_msg(sock, msg):
    """Gói dữ liệu bằng pickle và gửi kèm kích thước ở header."""
    msg_bytes = pickle.dumps(msg)
    # Pack kích thước (4 bytes) vào đầu tin nhắn
    sock.sendall(struct.pack('>I', len(msg_bytes)))
    sock.sendall(msg_bytes)

def recvall(sock, n):
    """Nhận đủ n bytes."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def recv_msg(sock):
    """Đọc kích thước header và nhận toàn bộ dữ liệu."""
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    msg_bytes = recvall(sock, msglen)
    return pickle.loads(msg_bytes)
