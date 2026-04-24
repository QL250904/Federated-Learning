import argparse
import socket
import numpy as np
from torchvision.datasets import MNIST
from network_utils import send_msg
from sklearn.decomposition import PCA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, required=True)
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--mode', type=str, choices=['fully', 'semi'], required=True)
    args = parser.parse_args()

    print(f"=== CLIENT {args.client_id} ({args.mode.upper()}) ===")
    trainset = MNIST('./data', train=True, download=True)
    X = trainset.data.numpy().reshape(-1, 28*28).astype(np.float32) / 255.0
    y = trainset.targets.numpy()

    # Chia data
    per_client = len(X) // args.num_clients
    start = args.client_id * per_client
    X_client = X[start:start+per_client]
    y_client = y[start:start+per_client]

    if args.mode == 'semi':
        print("Đang tiền xử lý PCA giảm xuống 100 chiều...")
        pca = PCA(n_components=100, random_state=42)
        X_client = pca.fit_transform(X_client)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((args.server_ip, args.port))
    print("Đã kết nối Server. Đang gửi dữ liệu...")
    
    send_msg(client, {'X': X_client, 'y': y_client})
    print("Gửi thành công! Client ngắt kết nối.")
    client.close()

if __name__ == '__main__':
    main()
