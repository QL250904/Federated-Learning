"""
Mô hình 5: Hybrid (Data + Model Parallelism) - SERVER
Server giữ Part2 (FC) cho mỗi data group, nhận activations từ clients,
và thực hiện AllReduce giữa các groups.

Kiến trúc 3 máy:
  - Server: Giữ Part2 cho cả 2 groups + AllReduce
  - Client 0: Part1 + Data shard 0
  - Client 1: Part1 + Data shard 1

Chạy: python real_5_hybrid_server.py --port 5005 --num_clients 2 --epochs 5
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
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

from network_utils import create_server, send_msg, recv_msg


class ModelPart2(nn.Module):
    """FC layers"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FullNet(nn.Module):
    """Full model cho evaluation"""
    def __init__(self):
        super().__init__()
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
    parser.add_argument('--port', type=int, default=5005)
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    print("=" * 60)
    print("  MÔ HÌNH 5: HYBRID (Data + Model Parallel) - SERVER")
    print(f"  {args.num_clients} data groups, Server giữ Part2 + AllReduce")
    print("=" * 60)

    server_sock, clients = create_server(args.port, args.num_clients)

    # Mỗi group có Part2 riêng
    parts2 = [ModelPart2() for _ in range(args.num_clients)]
    optimizers = [torch.optim.SGD(p.parameters(), lr=args.lr, momentum=0.9) for p in parts2]
    criterion = nn.CrossEntropyLoss()

    # Đồng bộ init Part2 weights
    init_state = parts2[0].state_dict()
    for p2 in parts2[1:]:
        p2.load_state_dict(init_state)

    # Test set
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    testset = MNIST('./data', train=False, download=True, transform=tr)
    test_loader = DataLoader(testset, batch_size=64)

    tracemalloc.start()
    t_start = time.time()
    total_comm_bytes = 0
    epoch_metrics = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        epoch_comm = 0
        epoch_loss_sum = 0
        epoch_batch_count = 0

        # === Phase 1: Pipeline training mỗi group ===
        for g in range(args.num_clients):
            parts2[g].train()
            
            while True:
                data = recv_msg(clients[g])
                if data is None:
                    break
                    
                action = data.get('action', '')
                if action == 'epoch_done':
                    break

                act_np = data['activations']
                y_np = data['labels']

                activations = torch.tensor(act_np, dtype=torch.float32, requires_grad=True)
                labels = torch.tensor(y_np, dtype=torch.long)

                optimizers[g].zero_grad()
                outputs = parts2[g](activations)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizers[g].step()

                grad_np = activations.grad.numpy()
                send_msg(clients[g], {'gradients': grad_np, 'loss': loss.item()})

                act_size = act_np.nbytes + y_np.nbytes + grad_np.nbytes
                epoch_comm += act_size
                epoch_loss_sum += loss.item()
                epoch_batch_count += 1

        # === Phase 2: AllReduce Part2 weights giữa các groups ===
        all_part2_params = []
        for p2 in parts2:
            params = [p.data.clone() for p in p2.parameters()]
            all_part2_params.append(params)

        # Average
        avg_params = []
        for layer_idx in range(len(all_part2_params[0])):
            stacked = torch.stack([all_part2_params[g][layer_idx] for g in range(args.num_clients)])
            avg_params.append(stacked.mean(dim=0))

        # Set averaged params
        for p2 in parts2:
            with torch.no_grad():
                for p, v in zip(p2.parameters(), avg_params):
                    p.data.copy_(v)

        # AllReduce comm size (tham số Part2)
        p2_param_size = sum(p.numel() * 4 for p in parts2[0].parameters())
        allreduce_comm = p2_param_size * 2 * (args.num_clients - 1)
        epoch_comm += allreduce_comm

        # === Phase 3: Nhận Part1 weights từ clients và AllReduce ===
        all_part1_weights = []
        for g in range(args.num_clients):
            data = recv_msg(clients[g])
            if data and data.get('action') == 'sync_part1':
                all_part1_weights.append(data['part1_weights'])

        # Average Part1 weights
        if all_part1_weights:
            avg_part1 = []
            for layer_idx in range(len(all_part1_weights[0])):
                layer_avg = np.mean([w[layer_idx] for w in all_part1_weights], axis=0)
                avg_part1.append(layer_avg)

            # Gửi averaged Part1 weights trở lại clients
            for g in range(args.num_clients):
                send_msg(clients[g], {'action': 'sync_part1', 'part1_weights': avg_part1})

            # p1 comm
            p1_size = sum(w.nbytes for w in all_part1_weights[0])
            epoch_comm += p1_size * 2 * args.num_clients

        total_comm_bytes += epoch_comm

        # === Evaluate ===
        if all_part1_weights:
            full_model = FullNet()
            part1_names = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias']
            for i, name in enumerate(part1_names):
                dict(full_model.named_parameters())[name].data.copy_(torch.tensor(avg_part1[i]))
            part2_params = list(parts2[0].parameters())
            part2_names = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
            for i, name in enumerate(part2_names):
                dict(full_model.named_parameters())[name].data.copy_(part2_params[i].data)

            full_model.eval()
            correct, test_loss = 0, 0.0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = full_model(images)
                    test_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / len(testset)
        else:
            accuracy, test_loss = 0, 0

        epoch_time = time.time() - epoch_start
        _, peak_mem = tracemalloc.get_traced_memory()
        avg_loss = epoch_loss_sum / max(epoch_batch_count, 1)

        epoch_metrics.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'test_loss': test_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time,
            'comm_bytes': epoch_comm,
        })
        print(f"  Epoch {epoch}/{args.epochs}: Acc={accuracy*100:.2f}%, "
              f"Loss={avg_loss:.4f}, Time={epoch_time:.2f}s, "
              f"Comm={epoch_comm/1024/1024:.1f}MB")

    # Done
    for conn in clients:
        send_msg(conn, {'action': 'done'})  # Báo clients kết thúc (cho epoch_done check)
        conn.close()

    total_time = time.time() - t_start
    _, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_usage = psutil.cpu_percent(interval=0.5)

    results = {
        'method': 'Hybrid Parallelism (Real Network)',
        'mode': 'real',
        'num_data_groups': args.num_clients,
        'num_workers': args.num_clients * 2,
        'epochs': args.epochs,
        'final_accuracy': epoch_metrics[-1]['accuracy'],
        'final_loss': epoch_metrics[-1]['test_loss'],
        'total_time': total_time,
        'total_comm_bytes': total_comm_bytes,
        'total_comm_mb': total_comm_bytes / (1024*1024),
        'peak_memory_mb': final_peak / (1024*1024),
        'cpu_percent': cpu_usage,
        'epoch_metrics': epoch_metrics,
    }

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ - Hybrid Parallelism (Real Network)")
    print(f"{'='*60}")
    print(f"  Accuracy:    {results['final_accuracy']*100:.2f}%")
    print(f"  Total Time:  {total_time:.2f}s")
    print(f"  Total Comm:  {results['total_comm_mb']:.1f}MB")

    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'real_hybrid_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    server_sock.close()


if __name__ == '__main__':
    main()
