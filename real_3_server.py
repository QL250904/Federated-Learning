import argparse
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from network_utils import send_msg, recv_msg
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def get_params(model): return [p.data.cpu().numpy() for p in model.parameters()]
def set_params(model, params):
    with torch.no_grad():
        for p, v in zip(model.parameters(), params): p.data.copy_(torch.tensor(v))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5003)
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    print("=== PARAMETER SERVER (DATA PARALLELISM) ===")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', args.port))
    server.listen(args.num_clients)

    clients = []
    for i in range(args.num_clients):
        conn, addr = server.accept()
        print(f"Client {addr} connected.")
        clients.append(conn)

    model = Net()
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        global_weights = get_params(model)
        
        # Broadcast weights
        for conn in clients:
            send_msg(conn, global_weights)
        
        # Nhận local weights
        local_weights = []
        for conn in clients:
            local_weights.append(recv_msg(conn))
        
        # FedAvg
        avg_weights = [np.mean([w[layer] for w in local_weights], axis=0) 
                       for layer in range(len(global_weights))]
        set_params(model, avg_weights)
        print("Đã tổng hợp weights từ các clients.")

    for conn in clients:
        send_msg(conn, "DONE")
        conn.close()
    print("Hoàn tất!")

if __name__ == '__main__':
    main()
