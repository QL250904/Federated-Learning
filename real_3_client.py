import argparse
import socket
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from network_utils import send_msg, recv_msg
from real_3_server import Net, set_params, get_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, required=True)
    parser.add_argument('--port', type=int, default=5003)
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--num_clients', type=int, default=2)
    args = parser.parse_args()

    print(f"=== CLIENT {args.client_id} (DATA PARALLELISM) ===")
    
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST('./data', train=True, download=True, transform=tr)
    per_client = len(trainset) // args.num_clients
    indices = list(range(args.client_id * per_client, (args.client_id+1) * per_client))
    loader = DataLoader(torch.utils.data.Subset(trainset, indices), batch_size=32, shuffle=True)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((args.server_ip, args.port))
    
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    epoch = 1
    while True:
        data = recv_msg(client)
        if data == "DONE":
            break
        print(f"\nNhận Global Weights (Epoch {epoch})")
        set_params(model, data)

        model.train()
        for X, y in loader:
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

        print("Gửi Local Weights lên Server...")
        send_msg(client, get_params(model))
        epoch += 1

    client.close()
    print("Hoàn tất!")

if __name__ == '__main__':
    main()
