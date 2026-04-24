"""
Benchmark 4: Song Song Mô Hình (Model Parallelism)
- Chia MODEL thành nhiều phần, mỗi worker xử lý 1 phần
- Dữ liệu chảy qua pipeline: Worker0 (conv layers) → Worker1 (fc layers)
- ✅ Dùng khi model quá lớn không vừa 1 máy
- ❌ Phức tạp, bottleneck tại worker chậm nhất
"""

import time
import tracemalloc
import pickle
import psutil
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose


# === Chia model thành 2 phần ===
class ModelPart1(nn.Module):
    """Phần 1: Feature Extraction (Conv layers) - chạy trên Worker 0"""
    def __init__(self):
        super(ModelPart1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        return x


class ModelPart2(nn.Module):
    """Phần 2: Classification (FC layers) - chạy trên Worker 1"""
    def __init__(self, num_classes=10):
        super(ModelPart2, self).__init__()
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FullModelForReference(nn.Module):
    """Model đầy đủ để so sánh param count."""
    def __init__(self, num_classes=10):
        super().__init__()
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


def run_model_parallelism(num_workers=2, epochs=5, lr=0.01, momentum=0.9, batch_size=64):
    print("=" * 60)
    print("  MÔ HÌNH 4: SONG SONG MÔ HÌNH (Model Parallelism)")
    print("  Chia model → Worker 0: Conv | Worker 1: FC")
    print("  Data pipeline: Input → W0 → activations → W1 → Output")
    print("=" * 60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST('./data', train=True, download=True, transform=tr)
    testset = MNIST('./data', train=False, download=True, transform=tr)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64)

    # Tạo 2 phần model
    part1 = ModelPart1().to(device)
    part2 = ModelPart2().to(device)

    p1_params = sum(p.numel() for p in part1.parameters())
    p2_params = sum(p.numel() for p in part2.parameters())
    total_params = p1_params + p2_params

    print(f"  Worker 0 (Conv): {p1_params:,} params ({p1_params*100/total_params:.1f}%)")
    print(f"  Worker 1 (FC):   {p2_params:,} params ({p2_params*100/total_params:.1f}%)")
    print(f"  Total:           {total_params:,} params")

    criterion = nn.CrossEntropyLoss()
    # Mỗi worker có optimizer riêng cho phần model của mình
    opt1 = torch.optim.SGD(part1.parameters(), lr=lr, momentum=momentum)
    opt2 = torch.optim.SGD(part2.parameters(), lr=lr, momentum=momentum)

    tracemalloc.start()
    cpu_before = psutil.cpu_percent(interval=None)
    t_start = time.time()
    total_comm_bytes = 0

    results = {
        'method': 'Song Song Mô Hình (Model Parallelism)',
        'group': 'distributed',
        'num_workers': num_workers,
        'epochs': epochs,
        'part1_params': p1_params,
        'part2_params': p2_params,
        'epoch_metrics': [],
    }

    for epoch in range(1, epochs + 1):
        part1.train()
        part2.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        epoch_comm = 0
        part1_time = 0.0
        part2_time = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward Part 1 (Worker 0)
            t1 = time.time()
            activations = part1(images)
            part1_time += time.time() - t1

            # Mô phỏng truyền activations qua mạng (Worker 0 → Worker 1)
            activation_size = activations.numel() * 4  # float32
            epoch_comm += activation_size

            # Forward Part 2 (Worker 1) - nhận activations
            t2 = time.time()
            # Cần retain_graph để backward qua pipeline
            activations_detached = activations.detach().requires_grad_(True)
            outputs = part2(activations_detached)
            loss = criterion(outputs, labels)

            # Backward Part 2
            opt2.zero_grad()
            loss.backward()
            opt2.step()
            part2_time += time.time() - t2

            # Truyền gradient ngược: Worker 1 → Worker 0
            grad_size = activations_detached.grad.numel() * 4
            epoch_comm += grad_size

            # Backward Part 1
            t1b = time.time()
            opt1.zero_grad()
            activations.backward(activations_detached.grad)
            opt1.step()
            part1_time += time.time() - t1b

            epoch_loss += loss.item()

        total_comm_bytes += epoch_comm

        # Evaluate
        part1.eval()
        part2.eval()
        correct, test_loss = 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                act = part1(images)
                outputs = part2(act)
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
            'part1_time': part1_time,
            'part2_time': part2_time,
            'comm_bytes': epoch_comm,
            'peak_memory_mb': peak_mem / (1024*1024),
        })

        print(f"  Epoch {epoch}/{epochs}: Acc={accuracy*100:.2f}%, "
              f"Time={epoch_time:.2f}s (W0={part1_time:.2f}s, W1={part2_time:.2f}s), "
              f"Comm={epoch_comm/1024/1024:.1f}MB")

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
    print(f"  KẾT QUẢ - Model Parallelism")
    print(f"{'='*60}")
    print(f"  Accuracy:    {results['final_accuracy']*100:.2f}%")
    print(f"  Total Time:  {total_time:.2f}s")
    print(f"  Peak Memory: {results['peak_memory_mb']:.1f}MB")
    print(f"  Total Comm:  {results['total_comm_mb']:.1f}MB (activations + gradients)")

    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'model_parallel_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results


if __name__ == '__main__':
    run_model_parallelism(num_workers=2, epochs=5)
