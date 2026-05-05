"""
Mô hình 4: Song Song Mô Hình (Model Parallelism) - SERVER
Server giữ Phần 2 (FC layers). Client giữ Phần 1 (Conv layers).
Pipeline: Client forward Part1 → gửi activations → Server forward Part2 + backward → gửi grads về Client.

Chạy: python real_4_model_parallel_server.py --port 5004
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
    """Phần 2: Classification (FC layers) - chạy trên Server"""
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
    parser.add_argument('--port', type=int, default=5004)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    print("=" * 60)
    print("  MÔ HÌNH 4: MODEL PARALLELISM - SERVER (Part 2: FC Layers)")
    print("  Pipeline: Client(Conv) → activations → Server(FC) → grads → Client")
    print("=" * 60)

    server_sock, clients = create_server(args.port, 1)  # 1 client
    conn = clients[0]

    model = ModelPart2()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Test set cho evaluation
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    testset = MNIST('./data', train=False, download=True, transform=tr)
    test_loader = DataLoader(testset, batch_size=64)

    tracemalloc.start()
    t_start = time.time()
    total_comm_bytes = 0
    epoch_metrics = []
    batch_count = 0

    print("\n  Bắt đầu pipeline training...")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_comm = 0
        epoch_batches = 0
        model.train()

        while True:
            data = recv_msg(conn)
            if data is None:
                break

            action = data.get('action', '')
            if action == 'epoch_done':
                break
            if action == 'all_done':
                # Nhận Part1 weights cho evaluation
                part1_weights = data.get('part1_weights')
                break

            # Nhận activations và labels
            act_np = data['activations']
            y_np = data['labels']
            
            activations = torch.tensor(act_np, dtype=torch.float32, requires_grad=True)
            labels = torch.tensor(y_np, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(activations)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Gửi gradient ngược về client
            grad_np = activations.grad.numpy()
            send_msg(conn, {'gradients': grad_np, 'loss': loss.item()})

            # Tính communication
            act_size = act_np.nbytes
            label_size = y_np.nbytes
            grad_size = grad_np.nbytes
            epoch_comm += act_size + label_size + grad_size
            epoch_loss += loss.item()
            epoch_batches += 1

        total_comm_bytes += epoch_comm
        epoch_time = time.time() - epoch_start

        # Evaluate - nhận Part1 weights từ client để tạo full model
        if data and data.get('action') == 'epoch_done':
            # Nhận evaluation request
            eval_data = recv_msg(conn)
            if eval_data and eval_data.get('action') == 'eval':
                part1_weights = eval_data['part1_weights']
                
                # Tạo full model để evaluate
                full_model = FullNet()
                # Set Part1 weights (conv layers)
                part1_param_names = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias']
                for i, name in enumerate(part1_param_names):
                    param = dict(full_model.named_parameters())[name]
                    param.data.copy_(torch.tensor(part1_weights[i]))
                # Set Part2 weights (fc layers)
                part2_params = list(model.parameters())
                part2_param_names = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
                for i, name in enumerate(part2_param_names):
                    param = dict(full_model.named_parameters())[name]
                    param.data.copy_(part2_params[i].data)

                full_model.eval()
                correct, test_loss = 0, 0.0
                test_criterion = nn.CrossEntropyLoss()
                with torch.no_grad():
                    for images, labels in test_loader:
                        outputs = full_model(images)
                        test_loss += test_criterion(outputs, labels).item()
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == labels).sum().item()

                accuracy = correct / len(testset)
                
                # Gửi kết quả eval về client
                send_msg(conn, {'accuracy': accuracy, 'test_loss': test_loss})
            else:
                accuracy = 0
                test_loss = 0
        else:
            accuracy = 0
            test_loss = 0

        _, peak_mem = tracemalloc.get_traced_memory()
        avg_loss = epoch_loss / max(epoch_batches, 1)

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

    total_time = time.time() - t_start
    _, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_usage = psutil.cpu_percent(interval=0.5)

    conn.close()
    server_sock.close()

    results = {
        'method': 'Model Parallelism (Real Network)',
        'mode': 'real',
        'num_workers': 2,
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
    print(f"  KẾT QUẢ - Model Parallelism (Real Network)")
    print(f"{'='*60}")
    print(f"  Accuracy:    {results['final_accuracy']*100:.2f}%")
    print(f"  Total Time:  {total_time:.2f}s")
    print(f"  Total Comm:  {results['total_comm_mb']:.1f}MB")

    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'real_model_parallel_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
