"""
Real FL Server (Centralized) - Chạy trên MÁY CHỦ
Dùng Flower framework, kết nối thực với 2 máy clients qua mạng.

CÁCH DÙNG:
  Máy Server: python real_server.py --host 0.0.0.0 --port 8080 --rounds 5
  Máy Client: python real_client.py --server_ip <IP_SERVER> --port 8080 --client_id 0
"""

import argparse
import time
import json
import flwr as fl
from dataset import prepare_datasets
from model import Net, test
from collections import OrderedDict
import torch


def get_on_fit_config(config_fit: dict):
    def fit_config_fn(server_round: int):
        return {
            'lr': config_fit['lr'],
            'momentum': config_fit['momentum'],
            'local_epochs': config_fit['local_epochs'],
        }
    return fit_config_fn


def get_evaluate_fn(test_loader, num_classes: int):
    def evaluate_fn(server_round: int, parameters, config):
        model = Net(num_classes)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(model, test_loader, device)
        print(f"[Server] Round {server_round} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return float(loss), {"accuracy": float(accuracy)}
    return evaluate_fn


def main():
    parser = argparse.ArgumentParser(description='FL Centralized Server (Real Network)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server bind address')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--rounds', type=int, default=5, help='Number of FL rounds')
    parser.add_argument('--num_clients', type=int, default=2, help='Min clients to start')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.995)
    parser.add_argument('--local_epochs', type=int, default=1)
    args = parser.parse_args()

    config_fit = {'lr': args.lr, 'momentum': args.momentum, 'local_epochs': args.local_epochs}
    _, _, test_loader = prepare_datasets(args.num_clients, 32)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,
        min_fit_clients=args.num_clients,
        fraction_evaluate=0.00001,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        on_fit_config_fn=get_on_fit_config(config_fit),
        evaluate_fn=get_evaluate_fn(test_loader, 10),
    )

    addr = f"{args.host}:{args.port}"
    print("=" * 60)
    print("  🌐 FL CENTRALIZED SERVER (Real Network)")
    print(f"  Address: {addr}")
    print(f"  Chờ {args.num_clients} clients kết nối...")
    print(f"  Rounds: {args.rounds}")
    print("=" * 60)

    start_time = time.time()
    history = fl.server.start_server(
        server_address=addr,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    total_time = time.time() - start_time

    metrics = {
        "total_time_s": total_time,
        "num_rounds": args.rounds,
        "num_clients": args.num_clients,
        "metrics_distributed": str(history.metrics_distributed),
        "losses_distributed": str(history.losses_distributed),
    }
    with open("real_server_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n✅ Hoàn tất! Tổng thời gian: {total_time:.2f}s")
    print(f"Metrics saved to: real_server_metrics.json")


if __name__ == "__main__":
    main()
