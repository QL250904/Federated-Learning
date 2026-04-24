import socket
import pickle
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

HOST = '127.0.0.1'
PORT = 6000
NUM_CLIENTS = 2

def start_rf_server():
    print("="*50)
    print(" MÁY CHỦ TRUNG TÂM (TRUYỀN THỐNG TRONG MACHINE LEARNING)")
    print(" Cơ chế: Bắt buộc thu gom file gốc của Mọi Client về học")
    print("="*50)
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(NUM_CLIENTS)
    print(f"[Server] Đang đợi dữ liệu nhạy cảm thô từ {NUM_CLIENTS} máy Clients (Port {PORT})...")

    all_X = []
    all_y = []
    
    start_time = time.time()
    for i in range(NUM_CLIENTS):
        conn, addr = server_socket.accept()
        print(f"[Server] Đã kết nối với Client {i+1} tại {addr}")
        
        # Nhận dữ liệu (Tối đa từng đợt để chống nghẽn RAM)
        data = b""
        while True:
            packet = conn.recv(1024 * 1024)
            if not packet: break
            data += packet
            if data.endswith(b"END_OF_TRANSMISSION"): 
                data = data[:-19]
                break
        
        X_client, y_client = pickle.loads(data)
        all_X.append(X_client)
        all_y.append(y_client)
        print(f" -> Đã Copy {len(X_client)} hình ảnh GỐC từ Client {i+1}")
        conn.close()

    # Gộp data về 1 chỗ ở Server (Big Data Centralized)
    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)
    print(f"\n[Server] Tổng hình ảnh đã gom trên Server: {len(X_train)}")
    print("[Server] Đang huấn luyện Random Forest...")
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    total_time = time.time() - start_time
    print(f"[Server] ✅ Hoàn tất! (Múi giờ: {total_time:.2f}s)")
    
if __name__ == "__main__":
    start_rf_server()
