import socket
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import prepare_datasets
import numpy as np

MY_PORT = 7002
NEIGHBOR_PORT = 7001
HOST = '127.0.0.1'
NUM_ROUNDS = 5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5); self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*4*4,120); self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*4*4)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

def get_params(model): return [val.cpu().detach().numpy() for val in model.parameters()]
def set_params(model, params):
    with torch.no_grad():
        for p, new_val in zip(model.parameters(), params): p.copy_(torch.tensor(new_val))

def train(model, trainloader):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for images, labels in trainloader:
        optimizer.zero_grad()
        nn.CrossEntropyLoss()(model(images), labels).backward()
        optimizer.step()

def start_node2():
    print("="*50)
    print(" MÁY TRẠM ĐỘC LẬP NODE 2 (DECENTRALIZED - RING)")
    print(" Cơ chế: Học độc lập, Chủ động chia sẻ kinh nghiệm Node 1")
    print("="*50)
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, MY_PORT))
    server.listen(1)
    
    model = Net()
    train_loaders, _, _ = prepare_datasets(2, 32)
    my_loader = train_loaders[1]
    
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[Node 2] Đang dò tìm cổng mở của Node 1 trên Port {NEIGHBOR_PORT}...")
    while True:
        try:
            client.connect((HOST, NEIGHBOR_PORT))
            break
        except ConnectionRefusedError:
            time.sleep(1)
            
    print("[Node 2] Chờ thiết lập socket ngược chiều từ Node 1...")
    conn, addr = server.accept()
    print("[Node 2] Đã thành công Ring Topology Mạng Vòng!")

    # Trao đổi liên hồi (Sẽ chờ phản hồi xong mới gộp AllReduce)
    start_time = time.time()
    for rnd in range(NUM_ROUNDS):
        print(f"\n[Vòng {rnd+1}] Đang train (Dataset cục bộ, cấm Share)...")
        train(model, my_loader)
        my_params = get_params(model)
        
        # Nhận tổng hợp trước (tráng đụng chạm cổng vì Node 1 gửi trước)
        print(f"[Vòng {rnd+1}] Đợi tổng hợp kinh nghiệm từ Node 1 truyền sang...")
        data = b""
        while True:
            packet = conn.recv(1024*1024)
            data += packet
            if b"END" in data: 
                data = data.split(b"END")[0]
                break
                
        neighbor_params = pickle.loads(data)
        
        # Gửi params qua cho nó gộp
        print(f"[Vòng {rnd+1}] Bắn Tham số ngược qua cho Node 1 gộp (Ring Topology)")
        client.sendall(pickle.dumps(my_params) + b"END")
        
        # Nó tự gộp phần của nhận từ hàng xóm
        avg_params = [np.mean([p1, p2], axis=0) for p1, p2 in zip(my_params, neighbor_params)]
        set_params(model, avg_params)
        print(f"[Vòng {rnd+1}] Gộp kinh nghiệm thành công!")
        
    print(f"\n[Node 2] ✅ Máy trạm đã hội tụ học xong, tốn {time.time()-start_time:.2f}s!")
    conn.close(); server.close(); client.close()

if __name__ == "__main__":
    start_node2()
