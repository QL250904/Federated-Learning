"""
Benchmark 1: Tập Trung Hoàn Toàn (Fully Centralized)
- Mọi client gửi DỮ LIỆU THÔ (raw data) về server
- Server gom hết data → train CNN trên toàn bộ
- ❌ Rủi ro bảo mật cao (data rời khỏi client)
- ❌ Tốn băng thông (truyền toàn bộ ảnh gốc)
- ✅ Dễ triển khai
"""

import time
import tracemalloc
import pickle
import json
import psutil
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def run_fully_centralized(num_clients=2, epochs=5, lr=0.01, momentum=0.9, batch_size=64):
    print("=" * 60)
    print("  MÔ HÌNH 1: TẬP TRUNG HOÀN TOÀN (Fully Centralized)")
    print("  Client gửi DATA THÔ → Server gom → Server train")
    print("=" * 60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # --- Mô phỏng: Clients gửi raw data về server ---
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST('./data', train=True, download=True, transform=tr)
    testset = MNIST('./data', train=False, download=True, transform=tr)

    # Chia data cho clients
    per_client = len(trainset) // num_clients
    client_indices = [list(range(i*per_client, (i+1)*per_client)) for i in range(num_clients)]

    # Mô phỏng truyền tải: tính kích thước raw data mỗi client gửi
    total_comm_bytes = 0
    comm_times = []
    for i in range(num_clients):
        raw_data_size = per_client * 28 * 28 * 4  # float32 = 4 bytes per pixel
        label_size = per_client * 8  # int64 = 8 bytes
        client_data_size = raw_data_size + label_size
        total_comm_bytes += client_data_size
        # Mô phỏng thời gian truyền (giả lập ~100MB/s LAN)
        transfer_time = client_data_size / (100 * 1024 * 1024)
        comm_times.append(transfer_time)
        print(f"  Client {i}: gửi {per_client} samples ({client_data_size/1024/1024:.1f}MB RAW DATA)")

    print(f"  Tổng bandwidth: {total_comm_bytes/1024/1024:.1f}MB")

    # --- Server: Gom data và train ---
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64)

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    tracemalloc.start()
    cpu_before = psutil.cpu_percent(interval=None)
    t_start = time.time()

    results = {
        'method': 'Tập Trung Hoàn Toàn (Fully Centralized)',
        'group': 'centralized',
        'num_clients': num_clients,
        'epochs': epochs,
        'epoch_metrics': [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
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
        current_mem, peak_mem = tracemalloc.get_traced_memory()

        results['epoch_metrics'].append({
            'epoch': epoch,
            'train_loss': epoch_loss / len(train_loader),
            'test_loss': test_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time,
            'peak_memory_mb': peak_mem / (1024*1024),
        })
        print(f"  Epoch {epoch}/{epochs}: Acc={accuracy*100:.2f}%, Loss={test_loss:.4f}, Time={epoch_time:.2f}s")

    total_time = time.time() - t_start
    cpu_after = psutil.cpu_percent(interval=0.5)
    _, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results['total_time'] = total_time
    results['total_comm_bytes'] = total_comm_bytes
    results['total_comm_mb'] = total_comm_bytes / (1024*1024)
    results['peak_memory_mb'] = final_peak / (1024*1024)
    results['cpu_percent'] = cpu_after
    results['final_accuracy'] = results['epoch_metrics'][-1]['accuracy']
    results['final_loss'] = results['epoch_metrics'][-1]['test_loss']

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ - Tập Trung Hoàn Toàn")
    print(f"{'='*60}")
    print(f"  Accuracy:    {results['final_accuracy']*100:.2f}%")
    print(f"  Total Time:  {total_time:.2f}s")
    print(f"  Peak Memory: {results['peak_memory_mb']:.1f}MB")
    print(f"  Comm Size:   {results['total_comm_mb']:.1f}MB (raw data)")
    print(f"  CPU Usage:   {cpu_after:.1f}%")

    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'fully_centralized_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results


if __name__ == '__main__':
    run_fully_centralized(num_clients=2, epochs=5)
