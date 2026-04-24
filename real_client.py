"""
Real FL Client (Centralized) - Chạy trên MÁY TRẠM (Client)
Kết nối tới Server Flower qua mạng thực.

CÁCH DÙNG:
  python real_client.py --server_ip 192.168.1.100 --port 8080 --client_id 0
  python real_client.py --server_ip 192.168.1.100 --port 8080 --client_id 1
"""

import argparse
import flwr as fl
from clients import FlowerClient
from dataset import prepare_datasets


def main():
    parser = argparse.ArgumentParser(description='FL Client (Real Network)')
    parser.add_argument('--server_ip', type=str, required=True, help='IP address of FL server')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--client_id', type=int, required=True, help='Client ID (0, 1, ...)')
    parser.add_argument('--num_clients', type=int, default=2, help='Total number of clients')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()

    trainloaders, valloaders, _ = prepare_datasets(args.num_clients, args.batch_size)

    client = FlowerClient(
        trainloader=trainloaders[args.client_id],
        valoader=valloaders[args.client_id],
        num_classes=args.num_classes,
    ).to_client()

    server_addr = f"{args.server_ip}:{args.port}"
    print("=" * 60)
    print(f"  🖥️  FL CLIENT {args.client_id} (Real Network)")
    print(f"  Connecting to server: {server_addr}")
    print(f"  Dataset partition: {args.client_id}/{args.num_clients}")
    print("=" * 60)

    fl.client.start_client(
        server_address=server_addr,
        client=client,
    )
    print(f"\n✅ Client {args.client_id} hoàn tất!")


if __name__ == "__main__":
    main()
