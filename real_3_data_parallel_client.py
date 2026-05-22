"""
Mô hình 3: Song Song Dữ Liệu (Data Parallelism) - CLIENT
Client nhận global weights → train local → gửi local weights về server.

Chạy: python real_3_data_parallel_client.py --server_ip <IP> --port 5003 --client_id 0 --num_clients 2
"""

import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

from network_utils import connect_to_server, send_msg, recv_msg


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
    parser.add_argument('--server_ip', type=str, required=True)
    parser.add_argument('--port', type=int, default=5003)
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  MÔ HÌNH 3: DATA PARALLELISM - CLIENT {args.client_id}")
    print("  Nhận weights → Train local → Gửi weights về Server")
    print("=" * 60)

    # Load data
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST('./data', train=True, download=True, transform=tr)
    per_client = len(trainset) // args.num_clients
    indices = list(range(args.client_id * per_client, (args.client_id + 1) * per_client))
    loader = DataLoader(Subset(trainset, indices), batch_size=32, shuffle=True)
    print(f"  Data: {len(indices)} samples")

    # Connect
    sock = connect_to_server(args.server_ip, args.port)

    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    epoch = 0
    while True:
        data = recv_msg(sock)
        if data is None or data.get('action') == 'done':
            print("  Nhận lệnh kết thúc từ Server.")
            break

        epoch += 1
        print(f"\n  === Epoch {epoch} ===")
        
        # Set global weights
        set_params(model, data['weights'])

        # Train local
        model.train()
        total_loss = 0.0
        t_train = time.time()
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_time = time.time() - t_train
        avg_loss = total_loss / len(loader)
        print(f"  Train: Loss={avg_loss:.4f}, Time={train_time:.2f}s")

        # Gửi local weights
        send_msg(sock, {'weights': get_params(model), 'train_time': train_time})
        print(f"  ✓ Đã gửi local weights về Server")

    sock.close()
    print("Hoàn tất!")


if __name__ == '__main__':
    main()
