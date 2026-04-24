import argparse
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from network_utils import send_msg, recv_msg

class ModelPart2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5004)
    parser.add_argument('--num_clients', type=int, default=1, help='1=ModelParallel, 2=Hybrid')
    args = parser.parse_args()

    mode = "MODEL PARALLEL" if args.num_clients == 1 else "HYBRID"
    print(f"=== SERVER ({mode}) - Giữ Phần 2 (FC Layers) ===")
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', args.port))
    server.listen(args.num_clients)

    clients = []
    for i in range(args.num_clients):
        conn, addr = server.accept()
        clients.append(conn)
        print(f"Client {addr} connected.")

    model = ModelPart2()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    print("Đang xử lý luồng Pipeline từ các clients...")
    try:
        while True:
            # Lặp qua từng client để xử lý batch
            for i, conn in enumerate(clients):
                data = recv_msg(conn)
                if data == "DONE":
                    continue
                if data is None:
                    continue

                act_data, y_data = data
                # Khôi phục tensor và cho phép tính gradient
                activations = torch.tensor(act_data, requires_grad=True)
                labels = torch.tensor(y_data, dtype=torch.long)

                optimizer.zero_grad()
                outputs = model(activations)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Gửi gradient của activations trả lại cho Client (Part 1)
                grad_data = activations.grad.numpy()
                send_msg(conn, grad_data)
                
    except Exception as e:
        print("Kết thúc quá trình training.")

    for conn in clients:
        conn.close()

if __name__ == '__main__':
    main()
