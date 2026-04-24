"""
Real Decentralized Node - Chạy trên MỖI MÁY CLIENT
Giao tiếp P2P (Ring Topology) qua mạng thực, KHÔNG CẦN SERVER.

CÁCH DÙNG (2 máy):
  Máy A: python real_decentralized_node.py --node_id 0 --my_port 7001 --neighbor_ip <IP_B> --neighbor_port 7001 --rounds 5
  Máy B: python real_decentralized_node.py --node_id 1 --my_port 7001 --neighbor_ip <IP_A> --neighbor_port 7001 --rounds 5

  Node 0 sẽ listen trước, rồi Node 1 connect vào.
"""

import argparse
import socket
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import prepare_datasets
import numpy as np


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
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def get_params(model):
    return [val.cpu().detach().numpy() for val in model.parameters()]

def set_params(model, params):
    with torch.no_grad():
        for p, new_val in zip(model.parameters(), params):
            p.copy_(torch.tensor(new_val))

def train_local(model, trainloader):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for images, labels in trainloader:
        optimizer.zero_grad()
        nn.CrossEntropyLoss()(model(images), labels).backward()
        optimizer.step()

def evaluate_model(model, testloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

def recv_all(conn):
    """Nhận toàn bộ dữ liệu cho tới khi gặp marker END"""
    data = b""
    while True:
        packet = conn.recv(1024 * 1024)
        if not packet:
            break
        data += packet
        if b"END_PARAMS" in data:
            data = data.split(b"END_PARAMS")[0]
            break
    return data

def send_params(sock, params):
    """Gửi params qua socket"""
    sock.sendall(pickle.dumps(params) + b"END_PARAMS")


def main():
    parser = argparse.ArgumentParser(description='Decentralized FL Node (Real Network)')
    parser.add_argument('--node_id', type=int, required=True, help='Node ID (0 or 1)')
    parser.add_argument('--my_port', type=int, default=7001, help='Port to listen on')
    parser.add_argument('--neighbor_ip', type=str, required=True, help='Neighbor IP')
    parser.add_argument('--neighbor_port', type=int, default=7001, help='Neighbor port')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--num_clients', type=int, default=2)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  🔗 DECENTRALIZED NODE {args.node_id} (Real Network P2P)")
    print(f"  My port: {args.my_port}")
    print(f"  Neighbor: {args.neighbor_ip}:{args.neighbor_port}")
    print(f"  Rounds: {args.rounds}")
    print("=" * 60)

    model = Net()
    train_loaders, _, test_loader = prepare_datasets(args.num_clients, 32)
    my_loader = train_loaders[args.node_id]

    if args.node_id == 0:
        # Node 0: Listen trước, đợi Node 1 connect
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', args.my_port))
        server.listen(1)
        print(f"[Node 0] Đang chờ Node 1 kết nối trên port {args.my_port}...")
        conn_in, addr = server.accept()
        print(f"[Node 0] Node 1 đã kết nối từ {addr}")

        # Connect ngược lại
        conn_out = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        time.sleep(1)
        while True:
            try:
                conn_out.connect((args.neighbor_ip, args.neighbor_port))
                break
            except ConnectionRefusedError:
                time.sleep(1)
        print("[Node 0] Đã thiết lập Ring Topology!")

    else:
        # Node 1: Connect trước
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', args.my_port))
        server.listen(1)

        conn_out = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[Node 1] Đang kết nối tới Node 0 ({args.neighbor_ip}:{args.neighbor_port})...")
        while True:
            try:
                conn_out.connect((args.neighbor_ip, args.neighbor_port))
                break
            except ConnectionRefusedError:
                time.sleep(1)
        print("[Node 1] Đã kết nối! Chờ kết nối ngược...")

        conn_in, addr = server.accept()
        print(f"[Node 1] Đã thiết lập Ring Topology! ({addr})")

    # ===== Training Loop =====
    start_time = time.time()
    for rnd in range(args.rounds):
        print(f"\n{'='*40} Vòng {rnd+1}/{args.rounds} {'='*40}")

        # Train local
        t0 = time.time()
        train_local(model, my_loader)
        train_time = time.time() - t0
        print(f"[Node {args.node_id}] Train local: {train_time:.2f}s")

        my_params = get_params(model)

        if args.node_id == 0:
            # Node 0: Gửi trước, nhận sau
            send_params(conn_out, my_params)
            print(f"[Node 0] Đã gửi tham số -> Node 1")

            data = recv_all(conn_in)
            neighbor_params = pickle.loads(data)
            print(f"[Node 0] Nhận tham số <- Node 1")
        else:
            # Node 1: Nhận trước, gửi sau
            data = recv_all(conn_in)
            neighbor_params = pickle.loads(data)
            print(f"[Node 1] Nhận tham số <- Node 0")

            send_params(conn_out, my_params)
            print(f"[Node 1] Đã gửi tham số -> Node 0")

        # FedAvg
        avg_params = [np.mean([p1, p2], axis=0) for p1, p2 in zip(my_params, neighbor_params)]
        set_params(model, avg_params)

        # Evaluate
        acc = evaluate_model(model, test_loader)
        print(f"[Node {args.node_id}] Accuracy sau vòng {rnd+1}: {acc*100:.2f}%")

    total_time = time.time() - start_time
    print(f"\n✅ Node {args.node_id} hoàn tất! Tổng thời gian: {total_time:.2f}s")

    conn_in.close()
    conn_out.close()
    server.close()


if __name__ == "__main__":
    main()
