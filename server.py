import flwr as fl
from dataset import prepare_datasets
from model import Net, test
from collections import OrderedDict
import torch


# ============ Cấu hình gửi cho client mỗi round fit ============
def get_on_fit_config(config_fit: dict):
    """Trả về hàm config gửi cho client mỗi round training."""
    def fit_config_fn(server_round: int):
        return {
            'lr': config_fit['lr'],
            'momentum': config_fit['momentum'],
            'local_epochs': config_fit['local_epochs'],
        }
    return fit_config_fn


# ============ Hàm evaluate trung tâm trên server ============
def get_evaluate_fn(test_loader, num_classes: int):
    """Trả về hàm evaluate chạy trên server sau mỗi round."""
    def evaluate_fn(server_round: int, parameters, config):
        model = Net(num_classes)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load parameters vào model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Evaluate
        loss, accuracy = test(model, test_loader, device)
        print(f"[Server] Round {server_round} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate_fn


# ============ Khởi động server ============
def start():
    # === Cấu hình (lấy từ base.yaml) ===
    NUM_CLIENTS = 2
    BATCH_SIZE = 32
    NUM_CLASSES = 10
    NUM_ROUNDS = 8
    NUM_CLIENTS_PER_ROUND_FIT = 2
    NUM_CLIENTS_PER_ROUND_EVAL = 2

    config_fit = {
        'lr': 0.01,
        'momentum': 0.995,
        'local_epochs': 1,
    }

    # === Chuẩn bị test dataset cho server evaluate ===
    _, _, test_loader = prepare_datasets(NUM_CLIENTS, BATCH_SIZE)

    # === Chiến lược FedAvg ===
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,
        min_fit_clients=NUM_CLIENTS_PER_ROUND_FIT,
        fraction_evaluate=0.00001,
        min_evaluate_clients=NUM_CLIENTS_PER_ROUND_EVAL,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=get_on_fit_config(config_fit),
        evaluate_fn=get_evaluate_fn(test_loader, NUM_CLASSES),
    )

    # === Khởi động Flower server ===
    print("=" * 50)
    print("  Flower Server đang khởi động...")
    print(f"  Address: 0.0.0.0:8080")
    print(f"  Chờ {NUM_CLIENTS} client kết nối...")
    print(f"  Số rounds: {NUM_ROUNDS}")
    print("=" * 50)

    import time
    import json
    
    start_time = time.time()
    
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    
    total_time = time.time() - start_time
    
    # Lưu thời gian ra file để đánh giá sau
    metrics = {
        "total_time_s": total_time,
        "metrics_distributed": str(history.metrics_distributed),
        "losses_distributed": str(history.losses_distributed)
    }
    with open("server_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\n[Hoàn tất] Tổng thời gian FL Network chạy: {total_time:.2f} giây")
    print(f"File log được lưu tại: server_metrics.json")


if __name__ == "__main__":
    start()