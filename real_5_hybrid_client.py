"""
Mô hình 5: Hybrid (Data + Model Parallelism) - CLIENT
Client giữ Part1 (Conv layers) + data shard riêng.
Pipeline forward → gửi activations → nhận grads → backward.
Sau mỗi epoch: AllReduce Part1 weights qua Server.

Chạy: python real_5_hybrid_client.py --server_ip <IP> --port 5005 --client_id 0 --num_clients 2
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


class ModelPart1(nn.Module):
    """Conv layers - chạy trên Client"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(-1, 16*4*4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, required=True)
    parser.add_argument('--port', type=int, default=5005)
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  MÔ HÌNH 5: HYBRID - CLIENT {args.client_id} (Part1: Conv)")
    print(f"  Data shard {args.client_id}/{args.num_clients}")
    print("=" * 60)

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST('./data', train=True, download=True, transform=tr)
    per_client = len(trainset) // args.num_clients
    indices = list(range(args.client_id * per_client, (args.client_id + 1) * per_client))
    loader = DataLoader(Subset(trainset, indices), batch_size=64, shuffle=True)
    print(f"  Data: {len(indices)} samples")

    sock = connect_to_server(args.server_ip, args.port)

    model = ModelPart1()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        batch_losses = []

        for X, y in loader:
            optimizer.zero_grad()
            activations = model(X)

            act_np = activations.detach().numpy()
            y_np = y.numpy()
            send_msg(sock, {
                'action': 'forward',
                'activations': act_np,
                'labels': y_np
            })

            response = recv_msg(sock)
            grad_np = response['gradients']
            batch_losses.append(response['loss'])

            grad_tensor = torch.tensor(grad_np, dtype=torch.float32)
            activations.backward(grad_tensor)
            optimizer.step()

        # Báo hết epoch
        send_msg(sock, {'action': 'epoch_done'})

        # Gửi Part1 weights cho AllReduce
        part1_weights = [p.data.cpu().numpy() for p in model.parameters()]
        send_msg(sock, {'action': 'sync_part1', 'part1_weights': part1_weights})

        # Nhận averaged Part1 weights
        sync_data = recv_msg(sock)
        if sync_data and sync_data.get('action') == 'sync_part1':
            avg_weights = sync_data['part1_weights']
            with torch.no_grad():
                for p, v in zip(model.parameters(), avg_weights):
                    p.data.copy_(torch.tensor(v))

        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(batch_losses) if batch_losses else 0
        print(f"  Epoch {epoch}/{args.epochs}: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s")

    sock.close()
    print("Hoàn tất!")


if __name__ == '__main__':
    main()
