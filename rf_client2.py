import socket
import pickle
import time
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np

HOST = '127.0.0.1'
PORT = 6000

def start_rf_client():
    print("="*50)
    print(" MÁY TRẠM CLIENT 2 (KIỂU TRUYỀN THỐNG TRONG AI)")
    print(" Cơ chế: Nộp toàn bộ file gốc cho Server (Không bảo mật)")
    print("="*50)
    
    # Load dataset cá nhân (nửa còn lại)
    trainset = MNIST('./data', train=True, download=True, transform=ToTensor())
    X = trainset.data[30000:].numpy().reshape(-1, 28 * 28) / 255.0
    y = trainset.targets[30000:].numpy()
    
    encoded_data = pickle.dumps((X, y))
    
    # Nộp cho server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        print(f"[Client 2] Đang truyền {len(X)} hình ảnh GỐC (~{len(encoded_data)/1024/1024:.1f} MB) cho Server...")
        client_socket.sendall(encoded_data + b"END_OF_TRANSMISSION")
        print("[Client 2] Nộp thành công. Dừng tiến trình!")
    except ConnectionRefusedError:
        print("Lỗi: Server (rf_server.py) chưa được bật!")
    finally:
        client_socket.close()

if __name__ == "__main__":
    start_rf_client()
