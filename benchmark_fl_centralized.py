"""
Benchmark: Federated Learning (Centralized Server) - Không mã hóa (No FHE)
- Máy chủ trung gian tập hợp tham số từ các local client.
- Chiến lược: FedAvg (mô phỏng trong code cho dễ thu metrics).
- Dùng mô hình CNN trên tập dataset MNIST.
"""

import time
import tracemalloc
import pickle
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose


# ===== Model =====
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ===== Data Loading =====
def prepare_data(num_clients=2, batch_size=32, val_ratio=0.1):
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST('./data', train=True, download=True, transform=tr)
    testset = MNIST('./data', train=False, download=True, transform=tr)

    num_sample = len(trainset) // num_clients
    partition_len = [num_sample] * num_clients
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    train_loaders, val_loaders = [], []
    for ts_part in trainsets:
        num_total = len(ts_part)
        num_val = int(num_total * val_ratio)
        num_train = num_total - num_val
        for_train, for_val = random_split(ts_part, (num_train, num_val),
                                          torch.Generator().manual_seed(2023))
        train_loaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False))

    test_loader = DataLoader(testset, batch_size=64)
    return train_loaders, val_loaders, test_loader


# ===== Helper Functions =====
def get_model_params_numpy(model):
    """Lấy parameters dạng numpy arrays."""
    return [val.cpu().detach().numpy() for val in model.parameters()]

def set_model_params_numpy(model, params):
    """Set parameters từ numpy arrays."""
    with torch.no_grad():
        for p, new_val in zip(model.parameters(), params):
            p.copy_(torch.tensor(new_val))

def aggregate_parameters(client_params_list):
    """Tính trung bình (FedAvg) các numpy array parameter."""
    num_clients = len(client_params_list)
    avg_params = []
    
    for layer_idx in range(len(client_params_list[0])):
        sum_layer = np.sum([client_params[layer_idx] for client_params in client_params_list], axis=0)
        avg_params.append(sum_layer / num_clients)
        
    return avg_params


# ===== Local Training =====
def train_local(model, train_loader, lr=0.01, momentum=0.9, epochs=1):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def evaluate(model, test_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return total_loss, accuracy


# ===== Main Benchmark =====
def run_fl_centralized(num_clients=2, num_rounds=5, local_epochs=1, lr=0.01, momentum=0.9):
    """
    Chạy FL Centralized (Server mô phỏng tập trung FedAvg) và thu thập metrics.
    """
    print("=" * 60)
    print("  FL (Centralized Server) - Mô phỏng FedAvg (No Encryption)")
    print("=" * 60)

    # Load data
    train_loaders, val_loaders, test_loader = prepare_data(num_clients)
    print(f"\nSố clients: {num_clients}")
    for i, tl in enumerate(train_loaders):
        print(f"  Client {i}: {len(tl.dataset)} samples")
    print(f"Test set: {len(test_loader.dataset)} samples")

    results = {
        'method': 'FL Centralized (FedAvg)',
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'rounds': [],
        'total_time': 0,
        'total_memory_peak_mb': 0,
    }

    tracemalloc.start()
    total_start = time.time()

    # Initialize global model
    global_model = Net()
    global_params = get_model_params_numpy(global_model)

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")
        round_start = time.time()

        all_client_params = []
        client_train_times = []
        client_comm_sizes = []

        for i in range(num_clients):
            # 1. Client nhận global params & set vào local model
            local_model = Net()
            set_model_params_numpy(local_model, global_params)

            # 2. Local training
            t_train_start = time.time()
            train_local(local_model, train_loaders[i], lr=lr, momentum=momentum, epochs=local_epochs)
            t_train_end = time.time()
            
            train_time = t_train_end - t_train_start
            client_train_times.append(train_time)

            # 3. Tính communication size (gửi mảng unencrypted params)
            local_params = get_model_params_numpy(local_model)
            param_size_bytes = sum(p.nbytes for p in local_params)
            client_comm_sizes.append(param_size_bytes)
            
            all_client_params.append(local_params)
            print(f"  Client {i}: train={train_time:.2f}s, comm_size={param_size_bytes / 1024:.1f}KB")

        # 4. Server Aggregate (FedAvg)
        t_agg_start = time.time()
        global_params = aggregate_parameters(all_client_params)
        t_agg_end = time.time()
        agg_time = t_agg_end - t_agg_start
        print(f"  Server aggregation: {agg_time:.4f}s")

        # 5. Evaluate
        set_model_params_numpy(global_model, global_params)
        loss, accuracy = evaluate(global_model, test_loader)

        round_time = time.time() - round_start
        current_mem, peak_mem = tracemalloc.get_traced_memory()

        round_result = {
            'round': round_num,
            'accuracy': accuracy,
            'loss': loss,
            'round_time': round_time,
            'agg_time': agg_time,
            'avg_client_train_time': np.mean(client_train_times),
            'avg_communication_size_bytes': np.mean(client_comm_sizes),
            'total_communication_size_bytes': sum(client_comm_sizes),
            'encryption_time': 0, # None
            'decryption_time': 0, # None
            'current_memory_mb': current_mem / (1024 * 1024),
            'peak_memory_mb': peak_mem / (1024 * 1024),
        }
        results['rounds'].append(round_result)

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Loss: {loss:.4f}")
        print(f"  Round time: {round_time:.2f}s")

    results['total_time'] = time.time() - total_start
    _, peak = tracemalloc.get_traced_memory()
    results['total_memory_peak_mb'] = peak / (1024 * 1024)
    tracemalloc.stop()

    # Final metrics
    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS - FL Centralized")
    print(f"{'=' * 60}")
    print(f"  Final Accuracy: {results['rounds'][-1]['accuracy']:.4f}")
    print(f"  Final Loss: {results['rounds'][-1]['loss']:.4f}")
    print(f"  Total Time: {results['total_time']:.2f}s")
    print(f"  Peak Memory: {results['total_memory_peak_mb']:.2f} MB")

    # Save results
    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'fl_centralized_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {save_path / 'fl_centralized_results.pkl'}")

    return results


if __name__ == '__main__':
    results = run_fl_centralized(
        num_clients=2,
        num_rounds=5,
        local_epochs=1,
        lr=0.01,
        momentum=0.995,
    )
