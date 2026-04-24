"""
Benchmark 3: Song Song Dữ Liệu (Data Parallelism)
- Chia DỮ LIỆU ra nhiều worker (máy)
- Mỗi worker train cùng 1 MÔ HÌNH giống nhau
- Sau mỗi bước → AllReduce đồng bộ gradient/tham số
- ✅ Tăng tốc training (phổ biến nhất)
- ✅ Dễ scale với nhiều GPU/máy
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
from torch.utils.data import DataLoader, random_split
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


def get_params(model):
    return [p.data.clone() for p in model.parameters()]

def set_params(model, params):
    with torch.no_grad():
        for p, new_val in zip(model.parameters(), params):
            p.data.copy_(new_val)

def allreduce_average(worker_params_list):
    """Mô phỏng AllReduce: trung bình tham số từ tất cả workers."""
    num_workers = len(worker_params_list)
    avg_params = []
    for layer_idx in range(len(worker_params_list[0])):
        stacked = torch.stack([worker_params_list[w][layer_idx] for w in range(num_workers)])
        avg_params.append(stacked.mean(dim=0))
    return avg_params


def run_data_parallelism(num_workers=2, epochs=5, lr=0.01, momentum=0.9, batch_size=32):
    print("=" * 60)
    print("  MÔ HÌNH 3: SONG SONG DỮ LIỆU (Data Parallelism)")
    print("  Chia data → mỗi worker train cùng model → AllReduce sync")
    print("=" * 60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST('./data', train=True, download=True, transform=tr)
    testset = MNIST('./data', train=False, download=True, transform=tr)

    # Chia data cho workers
    per_worker = len(trainset) // num_workers
    partition_lens = [per_worker] * num_workers
    worker_datasets = random_split(trainset, partition_lens, torch.Generator().manual_seed(42))
    worker_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in worker_datasets]
    test_loader = DataLoader(testset, batch_size=64)

    for i in range(num_workers):
        print(f"  Worker {i}: {len(worker_datasets[i])} samples")

    # Tạo model cho mỗi worker (cùng kiến trúc, cùng init)
    workers = [Net().to(device) for _ in range(num_workers)]
    # Đồng bộ init params
    init_params = get_params(workers[0])
    for w in workers[1:]:
        set_params(w, init_params)

    worker_optimizers = [
        torch.optim.SGD(w.parameters(), lr=lr, momentum=momentum)
        for w in workers
    ]
    criterion = nn.CrossEntropyLoss()

    tracemalloc.start()
    cpu_before = psutil.cpu_percent(interval=None)
    t_start = time.time()
    total_comm_bytes = 0

    results = {
        'method': 'Song Song Dữ Liệu (Data Parallelism)',
        'group': 'distributed',
        'num_workers': num_workers,
        'epochs': epochs,
        'epoch_metrics': [],
    }

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        worker_losses = []

        # Mỗi worker train trên phần data riêng
        worker_train_times = []
        for w_idx in range(num_workers):
            workers[w_idx].train()
            w_loss = 0.0
            t_w = time.time()

            for images, labels in worker_loaders[w_idx]:
                images, labels = images.to(device), labels.to(device)
                worker_optimizers[w_idx].zero_grad()
                loss = criterion(workers[w_idx](images), labels)
                loss.backward()
                worker_optimizers[w_idx].step()
                w_loss += loss.item()

            worker_train_times.append(time.time() - t_w)
            worker_losses.append(w_loss / len(worker_loaders[w_idx]))

        # AllReduce: đồng bộ tham số (trung bình)
        t_sync = time.time()
        all_params = [get_params(w) for w in workers]

        # Tính communication size (mỗi worker gửi params cho tất cả → AllReduce)
        param_size = sum(p.numel() * 4 for p in workers[0].parameters())  # float32
        # Ring-AllReduce: mỗi worker gửi/nhận 2*(N-1)/N * param_size
        epoch_comm = param_size * 2 * (num_workers - 1)
        total_comm_bytes += epoch_comm

        avg_params = allreduce_average(all_params)
        for w in workers:
            set_params(w, avg_params)
        sync_time = time.time() - t_sync

        # Evaluate (dùng worker 0 vì đã sync)
        workers[0].eval()
        correct, test_loss = 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = workers[0](images)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        accuracy = correct / len(testset)
        epoch_time = time.time() - epoch_start
        current_mem, peak_mem = tracemalloc.get_traced_memory()

        results['epoch_metrics'].append({
            'epoch': epoch,
            'train_loss': np.mean(worker_losses),
            'test_loss': test_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time,
            'sync_time': sync_time,
            'avg_worker_time': np.mean(worker_train_times),
            'max_worker_time': max(worker_train_times),
            'comm_bytes': epoch_comm,
            'peak_memory_mb': peak_mem / (1024*1024),
        })

        print(f"  Epoch {epoch}/{epochs}: Acc={accuracy*100:.2f}%, "
              f"Time={epoch_time:.2f}s (train={np.mean(worker_train_times):.2f}s, sync={sync_time:.4f}s), "
              f"Comm={epoch_comm/1024:.1f}KB")

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
    print(f"  KẾT QUẢ - Data Parallelism")
    print(f"{'='*60}")
    print(f"  Accuracy:    {results['final_accuracy']*100:.2f}%")
    print(f"  Total Time:  {total_time:.2f}s")
    print(f"  Peak Memory: {results['peak_memory_mb']:.1f}MB")
    print(f"  Total Comm:  {results['total_comm_mb']:.3f}MB (model params only)")

    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'data_parallel_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results


if __name__ == '__main__':
    run_data_parallelism(num_workers=2, epochs=5)
