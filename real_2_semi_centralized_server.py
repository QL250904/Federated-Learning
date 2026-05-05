"""
Mô hình 2: Tập Trung Có Tiền Xử Lý (Semi-Centralized) - SERVER
Server nhận data ĐÃ QUA PCA từ clients → gom → train.

Chạy: python real_2_semi_centralized_server.py --port 5002 --num_clients 2
"""

import argparse
import time
import json
import tracemalloc
import psutil
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

from network_utils import create_server, recv_msg


class NetSmall(nn.Module):
    """Mô hình cho dữ liệu đã giảm chiều qua PCA."""
    def __init__(self, input_dim=100):
        super(NetSmall, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5002)
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--pca_components', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    print("=" * 60)
    print("  MÔ HÌNH 2: SEMI-CENTRALIZED - SERVER")
    print("  Nhận data đã qua PCA từ clients → Train")
    print("=" * 60)

    # === Phase 1: Nhận data đã tiền xử lý từ clients ===
    server_sock, clients = create_server(args.port, args.num_clients)

    total_comm_bytes = 0
    all_X, all_y = [], []

    t_comm_start = time.time()
    for i, conn in enumerate(clients):
        print(f"\n[SERVER] Đang nhận data từ Client {i}...")
        data = recv_msg(conn)
        X_client = data['X']
        y_client = data['y']
        data_size = X_client.nbytes + y_client.nbytes
        total_comm_bytes += data_size
        all_X.append(X_client)
        all_y.append(y_client)
        print(f"[SERVER] ✓ Client {i}: {len(X_client)} samples (PCA), {data_size/1024/1024:.1f}MB")
        conn.close()

    comm_time = time.time() - t_comm_start
    
    # Tính raw data size để so sánh
    raw_data_per_client = (len(all_X[0]) * 28 * 28 * 4)
    total_raw_bytes = raw_data_per_client * args.num_clients
    
    print(f"\n[SERVER] Data nhận: {total_comm_bytes/1024/1024:.1f}MB (vs {total_raw_bytes/1024/1024:.1f}MB raw)")
    print(f"[SERVER] Giảm {(1-total_comm_bytes/total_raw_bytes)*100:.0f}% bandwidth!")

    # === Phase 2: Train ===
    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)

    # Test set cần PCA transform cùng chiều
    testset = MNIST('./data', train=False, download=True)
    X_test = testset.data.numpy().reshape(-1, 28*28).astype(np.float32) / 255.0
    y_test = testset.targets.numpy()
    
    # Fit PCA trên train data để transform test
    X_train_raw = MNIST('./data', train=True, download=True).data.numpy().reshape(-1, 28*28).astype(np.float32) / 255.0
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0) + 1e-8
    pca = PCA(n_components=args.pca_components, random_state=42)
    pca.fit((X_train_raw - mean) / std)
    X_test_pca = pca.transform((X_test - mean) / std)

    train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    test_tensor = TensorDataset(torch.tensor(X_test_pca, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=64)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = NetSmall(input_dim=args.pca_components).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    tracemalloc.start()
    t_train_start = time.time()

    epoch_metrics = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        correct, test_loss = 0, 0.0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                outputs = model(X_b)
                test_loss += criterion(outputs, y_b).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_b).sum().item()

        accuracy = correct / len(y_test)
        epoch_time = time.time() - epoch_start

        epoch_metrics.append({
            'epoch': epoch,
            'train_loss': epoch_loss / len(train_loader),
            'test_loss': test_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time,
        })
        print(f"  Epoch {epoch}/{args.epochs}: Acc={accuracy*100:.2f}%, "
              f"Loss={test_loss:.4f}, Time={epoch_time:.2f}s")

    train_time = time.time() - t_train_start
    _, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_usage = psutil.cpu_percent(interval=0.5)

    results = {
        'method': 'Semi-Centralized (Real Network)',
        'mode': 'real',
        'num_clients': args.num_clients,
        'epochs': args.epochs,
        'pca_components': args.pca_components,
        'final_accuracy': epoch_metrics[-1]['accuracy'],
        'final_loss': epoch_metrics[-1]['test_loss'],
        'total_time': comm_time + train_time,
        'comm_time': comm_time,
        'train_time': train_time,
        'total_comm_bytes': total_comm_bytes,
        'total_comm_mb': total_comm_bytes / (1024*1024),
        'raw_comm_mb': total_raw_bytes / (1024*1024),
        'bandwidth_reduction': (1 - total_comm_bytes/total_raw_bytes) * 100,
        'peak_memory_mb': final_peak / (1024*1024),
        'cpu_percent': cpu_usage,
        'epoch_metrics': epoch_metrics,
    }

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ - Semi-Centralized (Real Network)")
    print(f"{'='*60}")
    print(f"  Accuracy:      {results['final_accuracy']*100:.2f}%")
    print(f"  Total Time:    {results['total_time']:.2f}s")
    print(f"  Comm Size:     {results['total_comm_mb']:.1f}MB (vs {results['raw_comm_mb']:.1f}MB raw)")
    print(f"  BW Reduction:  {results['bandwidth_reduction']:.0f}%")

    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'real_semi_centralized_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    server_sock.close()


if __name__ == '__main__':
    main()
