import argparse
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from network_utils import send_msg, recv_msg

class ModelPart1(nn.Module):
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
    parser.add_argument('--client_id', type=int, default=0)
    parser.add_argument('--num_clients', type=int, default=1)
    args = parser.parse_args()

    mode = "MODEL PARALLEL" if args.num_clients == 1 else "HYBRID"
    print(f"=== CLIENT {args.client_id} ({mode}) - Giữ Phần 1 (Conv Layers) ===")

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST('./data', train=True, download=True, transform=tr)
    
    # Rút gọn số lượng để demo mạng LAN tránh timeout
    per_client = 1000
    indices = list(range(args.client_id * per_client, (args.client_id+1) * per_client))
    loader = DataLoader(torch.utils.data.Subset(trainset, indices), batch_size=32, shuffle=True)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((args.server_ip, args.port))

    model = ModelPart1()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    print("Bắt đầu train và pipeline qua mạng...")
    for epoch in range(1, 6):
        print(f"Epoch {epoch}/5...")
        for X, y in loader:
            optimizer.zero_grad()
            activations = model(X)
            
            # Gửi activations qua mạng
            act_data = activations.detach().numpy()
            y_data = y.numpy()
            send_msg(client, (act_data, y_data))

            # Nhận gradients từ Server (Part 2)
            grad_data = recv_msg(client)
            grad_tensor = torch.tensor(grad_data)

            # Tiếp tục backward pass
            activations.backward(grad_tensor)
            optimizer.step()

    send_msg(client, "DONE")
    client.close()
    print("Hoàn tất!")

if __name__ == '__main__':
    main()
