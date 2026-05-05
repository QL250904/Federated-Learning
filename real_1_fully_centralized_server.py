"""
Mô hình 1: Tập Trung Hoàn Toàn (Fully Centralized) - SERVER
Server nhận RAW DATA từ clients → gom lại → train CNN.

Chạy: python real_1_fully_centralized_server.py --port 5001 --num_clients 2
"""

import argparse
import time
import json
import tracemalloc
import psutil
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

from network_utils import create_server, recv_msg, send_msg


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    print("=" * 60)
    print("  MÔ HÌNH 1: TẬP TRUNG HOÀN TOÀN - SERVER")
    print("  Nhận RAW DATA từ clients → Gom → Train CNN")
    print("=" * 60)

    # === Phase 1: Nhận data từ clients ===
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
        print(f"[SERVER] ✓ Client {i}: {len(X_client)} samples, {data_size/1024/1024:.1f}MB")
        conn.close()
    
    comm_time = time.time() - t_comm_start
    print(f"\n[SERVER] Tổng data nhận: {total_comm_bytes/1024/1024:.1f}MB trong {comm_time:.2f}s")

    # === Phase 2: Gom data và train ===
    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)
    
    tensor_X = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28)
    tensor_y = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(tensor_X, tensor_y), batch_size=64, shuffle=True)

    # Test set
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    testset = MNIST('./data', train=False, download=True, transform=tr)
    test_loader = DataLoader(testset, batch_size=64)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
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

        # Evaluate
        model.eval()
        correct, test_loss = 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        accuracy = correct / len(testset)
        epoch_time = time.time() - epoch_start
        _, peak_mem = tracemalloc.get_traced_memory()
        
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

    # === Kết quả ===
    results = {
        'method': 'Fully Centralized (Real Network)',
        'mode': 'real',
        'num_clients': args.num_clients,
        'epochs': args.epochs,
        'final_accuracy': epoch_metrics[-1]['accuracy'],
        'final_loss': epoch_metrics[-1]['test_loss'],
        'total_time': comm_time + train_time,
        'comm_time': comm_time,
        'train_time': train_time,
        'total_comm_bytes': total_comm_bytes,
        'total_comm_mb': total_comm_bytes / (1024*1024),
        'peak_memory_mb': final_peak / (1024*1024),
        'cpu_percent': cpu_usage,
        'epoch_metrics': epoch_metrics,
    }

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ - Fully Centralized (Real Network)")
    print(f"{'='*60}")
    print(f"  Accuracy:    {results['final_accuracy']*100:.2f}%")
    print(f"  Total Time:  {results['total_time']:.2f}s (comm={comm_time:.2f}s, train={train_time:.2f}s)")
    print(f"  Comm Size:   {results['total_comm_mb']:.1f}MB")
    print(f"  Peak Memory: {results['peak_memory_mb']:.1f}MB")

    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'real_fully_centralized_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Kết quả đã lưu vào outputs/benchmark/real_fully_centralized_results.json")

    server_sock.close()


if __name__ == '__main__':
    main()
