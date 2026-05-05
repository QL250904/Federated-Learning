"""
Mô hình 1: Tập Trung Hoàn Toàn (Fully Centralized) - CLIENT
Client gửi RAW DATA (ảnh gốc) lên Server.

Chạy: python real_1_fully_centralized_client.py --server_ip <IP> --port 5001 --client_id 0 --num_clients 2
"""

import argparse
import time
import numpy as np
from torchvision.datasets import MNIST
from network_utils import connect_to_server, send_msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, required=True)
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--num_clients', type=int, default=2)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  MÔ HÌNH 1: TẬP TRUNG HOÀN TOÀN - CLIENT {args.client_id}")
    print("  Gửi RAW DATA lên Server")
    print("=" * 60)

    # Load và chia data
    trainset = MNIST('./data', train=True, download=True)
    X = trainset.data.numpy().reshape(-1, 28*28).astype(np.float32) / 255.0
    y = trainset.targets.numpy()

    per_client = len(X) // args.num_clients
    start = args.client_id * per_client
    end = start + per_client
    X_client = X[start:end]
    y_client = y[start:end]

    data_size = X_client.nbytes + y_client.nbytes
    print(f"  Data: {len(X_client)} samples, {data_size/1024/1024:.1f}MB")

    # Kết nối và gửi
    sock = connect_to_server(args.server_ip, args.port)
    
    t_start = time.time()
    send_msg(sock, {'X': X_client, 'y': y_client})
    t_send = time.time() - t_start
    
    print(f"  ✓ Gửi thành công! Thời gian: {t_send:.2f}s")
    print(f"  Throughput: {data_size/1024/1024/t_send:.1f} MB/s")
    sock.close()


if __name__ == '__main__':
    main()
