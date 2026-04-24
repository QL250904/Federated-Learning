"""
Real RF Server (Truyền thống) - Chạy trên MÁY CHỦ
Thu gom dữ liệu GỐC từ clients qua mạng, train Random Forest.

CÁCH DÙNG:
  Máy Server: python real_rf_server.py --host 0.0.0.0 --port 6000
  Máy Client: python real_rf_client.py --server_ip <IP_SERVER> --port 6000 --client_id 0
"""

import argparse
import socket
import pickle
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def main():
    parser = argparse.ArgumentParser(description='RF Server (Real Network)')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--num_clients', type=int, default=2)
    args = parser.parse_args()

    print("=" * 60)
    print("  🗄️  RF SERVER (Truyền thống - Thu gom data GỐC)")
    print(f"  Address: {args.host}:{args.port}")
    print(f"  Chờ {args.num_clients} clients gửi dữ liệu...")
    print("=" * 60)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((args.host, args.port))
    server_socket.listen(args.num_clients)

    all_X, all_y = [], []
    start_time = time.time()

    for i in range(args.num_clients):
        conn, addr = server_socket.accept()
        print(f"[Server] Client {i+1} kết nối từ {addr}")

        data = b""
        while True:
            packet = conn.recv(1024 * 1024)
            if not packet:
                break
            data += packet
            if data.endswith(b"END_OF_TRANSMISSION"):
                data = data[:-19]
                break

        X_client, y_client = pickle.loads(data)
        all_X.append(X_client)
        all_y.append(y_client)
        data_mb = len(pickle.dumps((X_client, y_client))) / (1024*1024)
        print(f"  -> Nhận {len(X_client)} samples (~{data_mb:.1f}MB RAW DATA) từ Client {i+1}")
        conn.close()

    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)
    print(f"\n[Server] Tổng dữ liệu gom: {len(X_train)} samples")
    print("[Server] Training Random Forest...")

    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, verbose=1)
    rf.fit(X_train, y_train)

    # Test
    testset = MNIST('./data', train=False, download=True, transform=ToTensor())
    X_test = testset.data.numpy().reshape(-1, 28*28) / 255.0
    y_test = testset.targets.numpy()
    acc = accuracy_score(y_test, rf.predict(X_test))

    total_time = time.time() - start_time
    print(f"\n✅ Hoàn tất!")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Total time: {total_time:.2f}s")
    server_socket.close()


if __name__ == "__main__":
    main()
