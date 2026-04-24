"""
Benchmark 5: Hybrid Parallelism (Data + Model Parallelism)
- Kết hợp CHIA DỮ LIỆU + CHIA MÔ HÌNH
- Mỗi data-group có nhiều workers chia model pipeline
- Giữa các data-group: AllReduce đồng bộ tham số
- ✅ Hiệu năng cao nhất khi có nhiều tài nguyên
- ❌ Phức tạp nhất về triển khai
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


class ModelPart1(nn.Module):
    """Conv layers (feature extraction)"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(-1, 16*4*4)


class ModelPart2(nn.Module):
    """FC layers (classification)"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_parts_params(part1, part2):
    return ([p.data.clone() for p in part1.parameters()],
            [p.data.clone() for p in part2.parameters()])

def set_parts_params(part1, part2, params1, params2):
    with torch.no_grad():
        for p, v in zip(part1.parameters(), params1):
            p.data.copy_(v)
        for p, v in zip(part2.parameters(), params2):
            p.data.copy_(v)


def run_hybrid_parallelism(num_data_groups=2, epochs=5, lr=0.01, momentum=0.9, batch_size=64):
    """
    Hybrid: 2 data groups × 2 model parts = 4 workers tổng cộng.
    Group 0: Worker A0 (conv) + Worker A1 (fc) → data shard 0
    Group 1: Worker B0 (conv) + Worker B1 (fc) → data shard 1
    Sau mỗi epoch: AllReduce giữa 2 groups.
    """
    print("=" * 60)
    print("  MÔ HÌNH 5: HYBRID (Data Parallel + Model Parallel)")
    print(f"  {num_data_groups} data groups × 2 model parts = {num_data_groups*2} workers")
    print("=" * 60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST('./data', train=True, download=True, transform=tr)
    testset = MNIST('./data', train=False, download=True, transform=tr)

    # Data Parallelism: chia data
    per_group = len(trainset) // num_data_groups
    group_datasets = random_split(trainset, [per_group]*num_data_groups,
                                   torch.Generator().manual_seed(42))
    group_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in group_datasets]
    test_loader = DataLoader(testset, batch_size=64)

    # Model Parallelism: mỗi group có 2 phần model
    groups = []
    for g in range(num_data_groups):
        p1 = ModelPart1().to(device)
        p2 = ModelPart2().to(device)
        groups.append((p1, p2))
        print(f"  Group {g}: {len(group_datasets[g])} samples, "
              f"Part1={sum(p.numel() for p in p1.parameters())} params, "
              f"Part2={sum(p.numel() for p in p2.parameters())} params")

    # Đồng bộ init params giữa groups
    init_p1, init_p2 = get_parts_params(groups[0][0], groups[0][1])
    for g in range(1, num_data_groups):
        set_parts_params(groups[g][0], groups[g][1], init_p1, init_p2)

    criterion = nn.CrossEntropyLoss()
    group_optimizers = []
    for p1, p2 in groups:
        opt1 = torch.optim.SGD(p1.parameters(), lr=lr, momentum=momentum)
        opt2 = torch.optim.SGD(p2.parameters(), lr=lr, momentum=momentum)
        group_optimizers.append((opt1, opt2))

    tracemalloc.start()
    cpu_before = psutil.cpu_percent(interval=None)
    t_start = time.time()
    total_comm_bytes = 0

    results = {
        'method': 'Hybrid (Data + Model Parallelism)',
        'group': 'distributed',
        'num_data_groups': num_data_groups,
        'num_workers': num_data_groups * 2,
        'epochs': epochs,
        'epoch_metrics': [],
    }

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        group_losses = []
        epoch_comm = 0

        # === Mỗi group train pipeline song song ===
        for g in range(num_data_groups):
            p1, p2 = groups[g]
            opt1, opt2 = group_optimizers[g]
            p1.train(); p2.train()
            g_loss = 0.0

            for images, labels in group_loaders[g]:
                images, labels = images.to(device), labels.to(device)

                # Forward Part1 → activations → Part2
                activations = p1(images)
                act_detached = activations.detach().requires_grad_(True)
                outputs = p2(act_detached)
                loss = criterion(outputs, labels)

                # Backward Part2
                opt2.zero_grad()
                loss.backward()
                opt2.step()

                # Pipeline comm: activations + gradients
                act_size = activations.numel() * 4
                grad_size = act_detached.grad.numel() * 4
                epoch_comm += act_size + grad_size

                # Backward Part1
                opt1.zero_grad()
                activations.backward(act_detached.grad)
                opt1.step()

                g_loss += loss.item()

            group_losses.append(g_loss / len(group_loaders[g]))

        # === AllReduce giữa các data groups (sync params) ===
        t_sync = time.time()
        all_p1_params = []
        all_p2_params = []
        for p1, p2 in groups:
            pp1, pp2 = get_parts_params(p1, p2)
            all_p1_params.append(pp1)
            all_p2_params.append(pp2)

        # Average
        avg_p1 = [torch.stack([all_p1_params[g][l] for g in range(num_data_groups)]).mean(0)
                   for l in range(len(all_p1_params[0]))]
        avg_p2 = [torch.stack([all_p2_params[g][l] for g in range(num_data_groups)]).mean(0)
                   for l in range(len(all_p2_params[0]))]

        for p1, p2 in groups:
            set_parts_params(p1, p2, avg_p1, avg_p2)
        sync_time = time.time() - t_sync

        # AllReduce comm size
        param_bytes = sum(p.numel()*4 for p in groups[0][0].parameters())
        param_bytes += sum(p.numel()*4 for p in groups[0][1].parameters())
        allreduce_comm = param_bytes * 2 * (num_data_groups - 1)
        epoch_comm += allreduce_comm
        total_comm_bytes += epoch_comm

        # Evaluate
        p1, p2 = groups[0]
        p1.eval(); p2.eval()
        correct, test_loss = 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = p2(p1(images))
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        accuracy = correct / len(testset)
        epoch_time = time.time() - epoch_start
        current_mem, peak_mem = tracemalloc.get_traced_memory()

        results['epoch_metrics'].append({
            'epoch': epoch,
            'train_loss': np.mean(group_losses),
            'test_loss': test_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time,
            'sync_time': sync_time,
            'comm_bytes': epoch_comm,
            'peak_memory_mb': peak_mem / (1024*1024),
        })

        print(f"  Epoch {epoch}/{epochs}: Acc={accuracy*100:.2f}%, "
              f"Time={epoch_time:.2f}s (sync={sync_time:.4f}s), "
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
    print(f"  KẾT QUẢ - Hybrid Parallelism")
    print(f"{'='*60}")
    print(f"  Accuracy:    {results['final_accuracy']*100:.2f}%")
    print(f"  Total Time:  {total_time:.2f}s")
    print(f"  Peak Memory: {results['peak_memory_mb']:.1f}MB")
    print(f"  Total Comm:  {results['total_comm_mb']:.1f}MB")

    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'hybrid_parallel_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results


if __name__ == '__main__':
    run_hybrid_parallelism(num_data_groups=2, epochs=5)
