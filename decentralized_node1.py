import socket
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import prepare_datasets
import numpy as np

MY_PORT = 7001
NEIGHBOR_PORT = 7002
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

def start_node1():
    print("="*50)
    print(" MÁY TRẠM ĐỘC LẬP NODE 1 (DECENTRALIZED - RING)")
    print(" Cơ chế: Không cần Server, giao tiếp với hàng xóm (Node 2)")
    print("="*50)
    
    # 1. Khởi chạy lưới chờ TCP
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, MY_PORT))
    server.listen(1)
    
    model = Net()
    train_loaders, _, _ = prepare_datasets(2, 32)
    my_loader = train_loaders[0]
    
    # 2. Đợi hàng xóm
    print(f"[Node 1] Đang chờ Node 2 kết nối trên Port {MY_PORT}...")
    conn, addr = server.accept()
    print(f"[Node 1] Đã mở khoá kết nối liên lưới với: {addr}")

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            client.connect((HOST, NEIGHBOR_PORT))
            break
        except ConnectionRefusedError:
            time.sleep(1)

    # 3. Trao đổi
    start_time = time.time()
    for rnd in range(NUM_ROUNDS):
        print(f"\n[Vòng {rnd+1}] Đang train (Dataset cục bộ, cấm Share)...")
        train(model, my_loader)
        my_params = get_params(model)
        
        # Gửi params qua Hàng xóm
        print(f"[Vòng {rnd+1}] Gửi Tham số Trí tuệ (KHÔNG GỬI DATA GỐC) -> Node 2")
        client.sendall(pickle.dumps(my_params) + b"END")
        
        # Chờ phản hồi
        print(f"[Vòng {rnd+1}] Đợi tổng hợp từ Node 2 trả về...")
        data = b""
        while True:
            packet = conn.recv(1024*1024)
            data += packet
            if b"END" in data: 
                data = data.split(b"END")[0]
                break
                
        neighbor_params = pickle.loads(data)
        
        # FEDAVG tự động trên máy (Ring-AllReduce All-Gather)
        avg_params = [np.mean([p1, p2], axis=0) for p1, p2 in zip(my_params, neighbor_params)]
        set_params(model, avg_params)
        print(f"[Vòng {rnd+1}] Gộp kinh nghiệm thành công!")
        
    print(f"\n[Node 1] ✅ Máy trạm đã hội tụ học xong, tốn {time.time()-start_time:.2f}s!")
    conn.close(); server.close(); client.close()

if __name__ == "__main__":
    start_node1()
