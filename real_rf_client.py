"""
Real RF Client - Chạy trên MÁY TRẠM (Client)
Gửi toàn bộ dữ liệu GỐC cho server (không bảo mật).

CÁCH DÙNG:
  python real_rf_client.py --server_ip 192.168.1.100 --port 6000 --client_id 0
  python real_rf_client.py --server_ip 192.168.1.100 --port 6000 --client_id 1
"""

import argparse
import socket
import pickle
import time
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def main():
    parser = argparse.ArgumentParser(description='RF Client (Real Network)')
    parser.add_argument('--server_ip', type=str, required=True, help='IP of RF server')
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--client_id', type=int, required=True, help='0 or 1')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  📤 RF CLIENT {args.client_id} (Truyền thống)")
    print(f"  Gửi toàn bộ DATA GỐC tới server {args.server_ip}:{args.port}")
    print("=" * 60)

    trainset = MNIST('./data', train=True, download=True, transform=ToTensor())
    total = len(trainset)
    per_client = total // 2

    start_idx = args.client_id * per_client
    end_idx = start_idx + per_client
    X = trainset.data[start_idx:end_idx].numpy().reshape(-1, 28*28) / 255.0
    y = trainset.targets[start_idx:end_idx].numpy()

    encoded = pickle.dumps((X, y))
    data_mb = len(encoded) / (1024*1024)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print(f"[Client {args.client_id}] Đang kết nối...")
        client_socket.connect((args.server_ip, args.port))
        print(f"[Client {args.client_id}] Truyền {len(X)} samples (~{data_mb:.1f}MB RAW DATA)...")

        t0 = time.time()
        client_socket.sendall(encoded + b"END_OF_TRANSMISSION")
        t1 = time.time()

        print(f"[Client {args.client_id}] ✅ Gửi xong trong {t1-t0:.2f}s")
    except ConnectionRefusedError:
        print(f"[Client {args.client_id}] ❌ Server chưa bật! Chạy real_rf_server.py trước")
    finally:
        client_socket.close()


if __name__ == "__main__":
    main()
