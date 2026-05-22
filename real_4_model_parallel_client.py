"""
Mô hình 4: Song Song Mô Hình (Model Parallelism) - CLIENT
Client giữ Phần 1 (Conv layers). Server giữ Phần 2 (FC layers).
Pipeline: forward Part1 → gửi activations → nhận gradients → backward Part1.

Chạy: python real_4_model_parallel_client.py --server_ip <IP> --port 5004
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
    """Phần 1: Feature Extraction (Conv layers) - chạy trên Client"""
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
    parser.add_argument('--port', type=int, default=5004)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    print("=" * 60)
    print("  MÔ HÌNH 4: MODEL PARALLELISM - CLIENT (Part 1: Conv Layers)")
    print("  Pipeline: forward Conv → gửi activations → nhận grads → backward Conv")
    print("=" * 60)

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST('./data', train=True, download=True, transform=tr)
    loader = DataLoader(trainset, batch_size=64, shuffle=True)
    print(f"  Data: {len(trainset)} samples")

    sock = connect_to_server(args.server_ip, args.port)

    model = ModelPart1()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    print("\n  Bắt đầu pipeline training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        batch_losses = []
        part1_time = 0.0

        for batch_idx, (X, y) in enumerate(loader):
            t_p1_start = time.time()
            optimizer.zero_grad()
            activations = model(X)
            act_np = activations.detach().numpy()
            y_np = y.numpy()
            part1_time += time.time() - t_p1_start

            # Gửi activations + labels đến Server (Part 2)
            send_msg(sock, {
                'action': 'forward',
                'activations': act_np,
                'labels': y_np
            })

            # Nhận gradient từ Server
            response = recv_msg(sock)
            grad_np = response['gradients']
            batch_losses.append(response['loss'])

            # Backward Part 1
            t_p1_back_start = time.time()
            grad_tensor = torch.tensor(grad_np, dtype=torch.float32)
            activations.backward(grad_tensor)
            optimizer.step()
            part1_time += time.time() - t_p1_back_start

        # Báo hết epoch
        send_msg(sock, {'action': 'epoch_done'})
        
        # Gửi Part1 weights để server evaluate
        part1_weights = [p.data.cpu().numpy() for p in model.parameters()]
        send_msg(sock, {'action': 'eval', 'part1_weights': part1_weights, 'part1_time': part1_time})
        
        # Nhận kết quả eval
        eval_result = recv_msg(sock)
        accuracy = eval_result['accuracy']
        test_loss = eval_result['test_loss']

        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(batch_losses)
        print(f"  Epoch {epoch}/{args.epochs}: Acc={accuracy*100:.2f}%, "
              f"Train Loss={avg_loss:.4f}, Time={epoch_time:.2f}s")

    sock.close()
    print("\nHoàn tất!")


if __name__ == '__main__':
    main()
