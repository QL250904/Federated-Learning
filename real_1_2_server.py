import argparse
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from network_utils import recv_msg
import numpy as np

class Net(nn.Module):
    def __init__(self, input_dim=None):
        super(Net, self).__init__()
        self.is_pca = input_dim is not None
        if self.is_pca:
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
        else:
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16*4*4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        if self.is_pca:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16*4*4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--mode', type=str, choices=['fully', 'semi'], required=True)
    args = parser.parse_args()

    print(f"=== SERVER TẬP TRUNG ({args.mode.upper()}) ===")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', args.port))
    server.listen(args.num_clients)

    all_X, all_y = [], []
    for i in range(args.num_clients):
        print(f"Chờ Client {i+1}...")
        conn, addr = server.accept()
        print(f"Client {addr} đã kết nối!")
        data = recv_msg(conn)
        all_X.append(data['X'])
        all_y.append(data['y'])
        conn.close()

    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)
    print(f"Đã gom đủ data từ {args.num_clients} clients. Bắt đầu train...")

    tensor_X = torch.tensor(X_train, dtype=torch.float32)
    if args.mode == 'fully':
        tensor_X = tensor_X.view(-1, 1, 28, 28)
    tensor_y = torch.tensor(y_train, dtype=torch.long)
    
    loader = DataLoader(TensorDataset(tensor_X, tensor_y), batch_size=64, shuffle=True)
    model = Net(input_dim=100 if args.mode == 'semi' else None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, 6):
        loss_sum = 0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"Epoch {epoch}/5 | Loss: {loss_sum/len(loader):.4f}")
    
    print("Hoàn tất Training trên Server!")

if __name__ == '__main__':
    main()
