"""
Mô hình 2: Tập Trung Có Tiền Xử Lý (Semi-Centralized) - CLIENT
Client tiền xử lý data (chuẩn hóa + PCA giảm chiều) rồi gửi lên Server.

Chạy: python real_2_semi_centralized_client.py --server_ip <IP> --port 5002 --client_id 0 --num_clients 2
"""

import argparse
import time
import numpy as np
from torchvision.datasets import MNIST
from sklearn.decomposition import PCA
from network_utils import connect_to_server, send_msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, required=True)
    parser.add_argument('--port', type=int, default=5002)
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--pca_components', type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  MÔ HÌNH 2: SEMI-CENTRALIZED - CLIENT {args.client_id}")
    print("  Tiền xử lý (PCA) → Gửi data nén lên Server")
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

    raw_size = X_client.nbytes + y_client.nbytes
    print(f"  Raw data: {len(X_client)} samples, {raw_size/1024/1024:.1f}MB")

    # Tiền xử lý
    t_pre = time.time()
    
    # 1. Lọc outlier
    pixel_sums = X_client.sum(axis=1)
    valid_mask = pixel_sums > 10
    X_client = X_client[valid_mask]
    y_client = y_client[valid_mask]

    # 2. Z-score normalize
    mean = X_client.mean(axis=0)
    std = X_client.std(axis=0) + 1e-8
    X_client = (X_client - mean) / std

    # 3. PCA giảm chiều
    pca = PCA(n_components=args.pca_components, random_state=42)
    X_pca = pca.fit_transform(X_client)

    preprocess_time = time.time() - t_pre
    processed_size = X_pca.nbytes + y_client.nbytes
    print(f"  Processed: {len(X_pca)} samples, {processed_size/1024/1024:.1f}MB")
    print(f"  Giảm {(1-processed_size/raw_size)*100:.0f}% bandwidth, PCA time={preprocess_time:.2f}s")

    # Kết nối và gửi
    sock = connect_to_server(args.server_ip, args.port)

    t_start = time.time()
    send_msg(sock, {
        'X': X_pca,
        'y': y_client,
        'preprocess_time': preprocess_time
    })
    t_send = time.time() - t_start

    print(f"  ✓ Gửi thành công! Thời gian: {t_send:.2f}s")
    sock.close()


if __name__ == '__main__':
    main()
