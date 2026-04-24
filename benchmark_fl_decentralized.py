"""
Benchmark: Federated Learning (Decentralized - Peer-to-peer)
- Không có Server điều phối (No Server).
- Các clients tổ chức theo mô hình Ring-Topology (Vòng).
- Dữ liệu tham số được truyền vòng tròn giữa các máy để tính tổng và trung bình (AllReduce).
- Không mã hóa tham số (No FHE)
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
    return [val.cpu().detach().numpy() for val in model.parameters()]

def set_model_params_numpy(model, params):
    with torch.no_grad():
        for p, new_val in zip(model.parameters(), params):
            p.copy_(torch.tensor(new_val))


def ring_allreduce(client_params_list):
    """
    Mô phỏng cơ chế Ring-AllReduce.
    Mỗi node giao tiếp với Node kế tiếp trong vòng tròn.
    Hàm này mô phỏng kết quả bằng cách tính trực tiếp và đếm số bước nhảy (hops) thực tế ở ngoài tự nhiên.
    """
    num_clients = len(client_params_list)
    num_params = len(client_params_list[0])
    
    # Số bước để truyền hết thông tin (Scatter-Reduce & AllGather) = 2*(N-1) bước
    comm_hops = 2 * (num_clients - 1)
    
    avg_params = []
    
    # Tính trung bình trên từng Layer
    for layer_idx in range(num_params):
        sum_layer = np.sum([client_params[layer_idx] for client_params in client_params_list], axis=0)
        avg_params.append(sum_layer / num_clients)
        
    return avg_params, comm_hops


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
def run_fl_decentralized(num_clients=2, num_rounds=5, local_epochs=1, lr=0.01, momentum=0.9):
    print("=" * 60)
    print("  FL (Decentralized - Peer-to-Peer Ring) - Không mã hóa (No FHE)")
    print("=" * 60)

    train_loaders, val_loaders, test_loader = prepare_data(num_clients)
    print(f"\nSố clients: {num_clients}")
    print(f"Topology: Ring (Client 0 → Client 1 → ... → Client 0)")
    for i, tl in enumerate(train_loaders):
        print(f"  Client {i}: {len(tl.dataset)} samples")

    results = {
        'method': 'FL Decentralized (Ring-AllReduce)',
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'topology': 'ring',
        'rounds': [],
        'total_time': 0,
        'total_memory_peak_mb': 0,
    }

    tracemalloc.start()
    total_start = time.time()

    # Initialize models cho mỗi client
    client_models = [Net() for _ in range(num_clients)]
    
    # Sync initial params
    init_params = get_model_params_numpy(client_models[0])
    for m in client_models[1:]:
        set_model_params_numpy(m, init_params)

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")
        round_start = time.time()

        client_train_times = []
        all_client_params = []
        total_comm_bytes = 0

        # Phase 1: Mỗi client train local
        for i in range(num_clients):
            t0 = time.time()
            train_local(client_models[i], train_loaders[i],
                        lr=lr, momentum=momentum, epochs=local_epochs)
            t1 = time.time()
            client_train_times.append(t1 - t0)

            local_params = get_model_params_numpy(client_models[i])
            all_client_params.append(local_params)
            
            # Tính băng thông cần trong mạng ngang hàng Network
            param_size_bytes = sum(p.nbytes for p in local_params)
            # Trong một Ring-Allreduce với N Node, mỗi parameter được chuyển 2 lần
            # do đó dữ liệu Data transfer in + out = 2 * param_size_bytes (cho mỗi client)
            client_comm_size = 2 * param_size_bytes
            total_comm_bytes += client_comm_size

            print(f"  Client {i}: train={t1 - t0:.2f}s, comm_size_per_client={client_comm_size / 1024:.1f}KB")

        # Phase 2: Decentralized Aggregation (Ring-AllReduce Protocol)
        t_agg_start = time.time()
        agg_params, comm_hops = ring_allreduce(all_client_params)
        t_agg_end = time.time()
        agg_time = t_agg_end - t_agg_start
        print(f"  Decentralized Ring-AllReduce: {agg_time:.4f}s, Hops: {comm_hops}")

        # Phase 3: Set Global parameters cho tất cả Client Models (sync-ed)
        for m in client_models:
            set_model_params_numpy(m, agg_params)

        # Evaluate (Sau khi merge vòng, các client có chung model tham số, nên ta lấy bất kỳ Client 0 để test)
        loss, accuracy = evaluate(client_models[0], test_loader)

        round_time = time.time() - round_start
        current_mem, peak_mem = tracemalloc.get_traced_memory()

        round_result = {
            'round': round_num,
            'accuracy': accuracy,
            'loss': loss,
            'round_time': round_time,
            'agg_time': agg_time,
            'avg_client_train_time': np.mean(client_train_times),
            'avg_communication_size_bytes': total_comm_bytes / num_clients,
            'total_communication_size_bytes': total_comm_bytes,
            'communication_hops': comm_hops,
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

    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS - FL Decentralized (Ring-Topology)")
    print(f"{'=' * 60}")
    print(f"  Final Accuracy: {results['rounds'][-1]['accuracy']:.4f}")
    print(f"  Final Loss:    {results['rounds'][-1]['loss']:.4f}")
    print(f"  Total Time:    {results['total_time']:.2f}s")
    print(f"  Peak Memory:   {results['total_memory_peak_mb']:.2f} MB")

    save_path = Path('./outputs/benchmark')
    with open(save_path / 'fl_decentralized_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {save_path / 'fl_decentralized_results.pkl'}")

    return results


if __name__ == '__main__':
    results = run_fl_decentralized(
        num_clients=2,
        num_rounds=5,
        local_epochs=1,
        lr=0.01,
        momentum=0.995,
    )
