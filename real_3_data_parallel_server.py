"""
Mô hình 3: Song Song Dữ Liệu (Data Parallelism) - SERVER (Parameter Server)
Server broadcast global weights → clients train local → server FedAvg.

Chạy: python real_3_data_parallel_server.py --port 5003 --num_clients 2 --epochs 5
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


def get_params(model):
    return [p.data.cpu().numpy() for p in model.parameters()]

def set_params(model, params):
    with torch.no_grad():
        for p, v in zip(model.parameters(), params):
            p.data.copy_(torch.tensor(v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5003)
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    print("=" * 60)
    print("  MÔ HÌNH 3: DATA PARALLELISM - PARAMETER SERVER")
    print("  Broadcast weights → Clients train → FedAvg")
    print("=" * 60)

    server_sock, clients = create_server(args.port, args.num_clients)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)

    # Test set
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    testset = MNIST('./data', train=False, download=True, transform=tr)
    test_loader = DataLoader(testset, batch_size=64)
    criterion = nn.CrossEntropyLoss()

    param_size = sum(p.numel() * 4 for p in model.parameters())
    print(f"  Model size: {param_size/1024:.1f}KB")

    tracemalloc.start()
    t_start = time.time()
    total_comm_bytes = 0
    epoch_metrics = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        # Broadcast global weights
        global_weights = get_params(model)
        t_comm = time.time()
        for conn in clients:
            send_msg(conn, {'action': 'train', 'weights': global_weights})
        broadcast_comm = param_size * args.num_clients
        total_comm_bytes += broadcast_comm

        # Nhận local weights từ clients
        local_weights_list = []
        worker_train_times = []
        for i, conn in enumerate(clients):
            data = recv_msg(conn)
            local_weights_list.append(data['weights'])
            worker_train_times.append(data.get('train_time', 0.0))
            print(f"  ✓ Nhận weights từ Client {i}")
        
        receive_comm = param_size * args.num_clients
        total_comm_bytes += receive_comm
        comm_time = time.time() - t_comm

        # FedAvg - trung bình các local weights
        avg_weights = []
        for layer_idx in range(len(global_weights)):
            layer_avg = np.mean([w[layer_idx] for w in local_weights_list], axis=0)
            avg_weights.append(layer_avg)
        set_params(model, avg_weights)

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
        epoch_comm = broadcast_comm + receive_comm

        epoch_metrics.append({
            'epoch': epoch,
            'test_loss': test_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time,
            'sync_time': comm_time,
            'avg_worker_time': np.mean(worker_train_times),
            'comm_bytes': epoch_comm,
        })
        print(f"  Epoch {epoch}: Acc={accuracy*100:.2f}%, "
              f"Time={epoch_time:.2f}s (comm={comm_time:.2f}s), "
              f"Comm={epoch_comm/1024:.1f}KB")

    # Gửi lệnh kết thúc
    for conn in clients:
        send_msg(conn, {'action': 'done'})
        conn.close()

    total_time = time.time() - t_start
    _, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_usage = psutil.cpu_percent(interval=0.5)

    results = {
        'method': 'Data Parallelism (Real Network)',
        'mode': 'real',
        'num_clients': args.num_clients,
        'num_workers': args.num_clients,
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
    print(f"  KẾT QUẢ - Data Parallelism (Real Network)")
    print(f"{'='*60}")
    print(f"  Accuracy:    {results['final_accuracy']*100:.2f}%")
    print(f"  Total Time:  {total_time:.2f}s")
    print(f"  Total Comm:  {results['total_comm_mb']:.3f}MB")

    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'real_data_parallel_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    server_sock.close()


if __name__ == '__main__':
    main()
