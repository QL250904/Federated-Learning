"""
Benchmark: Federated Learning + FHE (Fully Homomorphic Encryption) - Decentralized (Peer-to-Peer)
- KHÔNG có server trung tâm
- Các client giao tiếp trực tiếp với nhau (ring topology)
- Mỗi client mã hóa, gửi cho neighbor, neighbor aggregate trên ciphertext
- Dùng Ring-AllReduce pattern
"""

import time
import tracemalloc
import pickle
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

try:
    import tenseal as ts
except ImportError:
    print("=" * 60)
    print("  ERROR: Chưa cài tenseal!")
    print("  Chạy: pip install tenseal")
    print("=" * 60)
    sys.exit(1)


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


# ===== FHE Helper Functions =====
def create_fhe_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 40
    return context


def encrypt_parameters(context, parameters):
    encrypted_params = []
    for param in parameters:
        flat = param.flatten().tolist()
        chunk_size = 4096
        chunks = [flat[i:i + chunk_size] for i in range(0, len(flat), chunk_size)]
        encrypted_chunks = [ts.ckks_vector(context, chunk) for chunk in chunks]
        encrypted_params.append({
            'chunks': encrypted_chunks,
            'shape': param.shape,
            'total_len': len(flat),
        })
    return encrypted_params


def decrypt_parameters(encrypted_params):
    decrypted = []
    for ep in encrypted_params:
        flat = []
        for chunk in ep['chunks']:
            flat.extend(chunk.decrypt())
        flat = flat[:ep['total_len']]
        param = np.array(flat).reshape(ep['shape'])
        decrypted.append(param)
    return decrypted


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


def get_model_params_numpy(model):
    return [val.cpu().detach().numpy() for val in model.parameters()]


def set_model_params_numpy(model, params):
    with torch.no_grad():
        for p, new_val in zip(model.parameters(), params):
            p.copy_(torch.tensor(new_val))


# ===== Decentralized Aggregation (Ring-AllReduce) =====
def ring_allreduce_encrypted(client_encrypted_params_list, num_clients):
    """
    Mô phỏng Ring-AllReduce trên encrypted parameters.

    Ring topology: Client 0 → Client 1 → Client 2 → ... → Client 0

    Phase 1 (Scatter-Reduce): Mỗi client gửi 1 phần params cho neighbor bên phải,
    neighbor cộng vào phần tương ứng của mình.
    Sau N-1 bước, mỗi client có 1 phần đã được sum đầy đủ.

    Phase 2 (AllGather): Mỗi client gửi phần đã sum đầy đủ cho neighbor,
    sau N-1 bước tất cả đều có kết quả.

    Để đơn giản, mô phỏng bằng cách:
    - Mỗi client gửi encrypted params cho tất cả neighbor trong ring
    - Mỗi hop là 1 bước communication
    """
    num_params = len(client_encrypted_params_list[0])
    total_comm_hops = 0
    total_comm_bytes = 0

    # ===== Phase 1: Scatter-Reduce =====
    # Mỗi client chia params thành num_clients chunks
    # Mỗi bước, gửi 1 chunk cho neighbor bên phải
    # Sau num_clients - 1 bước, mỗi position i có sum từ tất cả clients

    # Simplified: sum tất cả encrypted params tại mỗi param position
    aggregated = []
    for param_idx in range(num_params):
        num_chunks = len(client_encrypted_params_list[0][param_idx]['chunks'])
        agg_chunks = []

        for chunk_idx in range(num_chunks):
            result = client_encrypted_params_list[0][param_idx]['chunks'][chunk_idx]
            for step in range(1, num_clients):
                # Mỗi step = 1 hop trong ring
                sender = step
                result = result + client_encrypted_params_list[sender][param_idx]['chunks'][chunk_idx]
                total_comm_hops += 1

                # Tính communication size cho hop này
                comm_size = len(
                    client_encrypted_params_list[sender][param_idx]['chunks'][chunk_idx].serialize()
                )
                total_comm_bytes += comm_size

            # Chia trung bình
            result = result * (1.0 / num_clients)
            agg_chunks.append(result)

        aggregated.append({
            'chunks': agg_chunks,
            'shape': client_encrypted_params_list[0][param_idx]['shape'],
            'total_len': client_encrypted_params_list[0][param_idx]['total_len'],
        })

    # ===== Phase 2: AllGather =====
    # Mỗi client broadcast reduced chunk cho tất cả → thêm (N-1) hops per chunk
    allgather_hops = (num_clients - 1) * num_params
    total_comm_hops += allgather_hops

    return aggregated, total_comm_hops, total_comm_bytes


def run_fhe_decentralized(num_clients=2, num_rounds=5, local_epochs=1, lr=0.01, momentum=0.9):
    """
    Chạy FL + FHE Decentralized (Ring-AllReduce) và thu thập metrics.
    """
    print("=" * 60)
    print("  FL + FHE (Decentralized - Ring-AllReduce)")
    print("=" * 60)

    # Load data
    train_loaders, val_loaders, test_loader = prepare_data(num_clients)
    print(f"\nSố clients: {num_clients}")
    print(f"Topology: Ring (Client 0 → Client 1 → ... → Client 0)")
    for i, tl in enumerate(train_loaders):
        print(f"  Client {i}: {len(tl.dataset)} samples")
    print(f"Test set: {len(test_loader.dataset)} samples")

    # Mỗi client có FHE context riêng (decentralized = mỗi client tự quản lý key)
    print("\nTạo FHE context cho mỗi client...")
    # Trong thực tế, mỗi client có key riêng. Ở đây dùng shared context để demo
    # vì Ring-AllReduce cần operate trên cùng encryption space
    context = create_fhe_context()
    print("  Shared CKKS context (for demo - thực tế dùng threshold FHE)")

    results = {
        'method': 'FL + FHE Decentralized (Ring-AllReduce)',
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
        client_encrypt_times = []
        all_encrypted_params = []

        # Phase 1: Mỗi client train local
        for i in range(num_clients):
            t0 = time.time()
            train_local(client_models[i], train_loaders[i],
                        lr=lr, momentum=momentum, epochs=local_epochs)
            t1 = time.time()
            client_train_times.append(t1 - t0)

            # Encrypt local params
            local_params = get_model_params_numpy(client_models[i])
            t_enc_start = time.time()
            enc_params = encrypt_parameters(context, local_params)
            t_enc_end = time.time()
            client_encrypt_times.append(t_enc_end - t_enc_start)

            all_encrypted_params.append(enc_params)
            print(f"  Client {i}: train={t1 - t0:.2f}s, encrypt={t_enc_end - t_enc_start:.2f}s")

        # Phase 2: Ring-AllReduce Aggregation (decentralized, no central server!)
        t_agg_start = time.time()
        agg_encrypted, comm_hops, comm_bytes = ring_allreduce_encrypted(
            all_encrypted_params, num_clients
        )
        t_agg_end = time.time()
        agg_time = t_agg_end - t_agg_start
        print(f"  Ring-AllReduce: {agg_time:.4f}s, {comm_hops} hops, {comm_bytes / 1024:.1f}KB total comm")

        # Phase 3: Decrypt & update tất cả client models
        t_dec_start = time.time()
        new_params = decrypt_parameters(agg_encrypted)
        t_dec_end = time.time()
        decrypt_time = t_dec_end - t_dec_start

        for m in client_models:
            set_model_params_numpy(m, new_params)

        # Evaluate (dùng client 0 vì tất cả đã sync)
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
            'avg_communication_size_bytes': comm_bytes / num_clients,
            'total_communication_size_bytes': comm_bytes,
            'communication_hops': comm_hops,
            'encryption_time': np.mean(client_encrypt_times),
            'decryption_time': decrypt_time,
            'current_memory_mb': current_mem / (1024 * 1024),
            'peak_memory_mb': peak_mem / (1024 * 1024),
        }
        results['rounds'].append(round_result)

        print(f"  Decryption: {decrypt_time:.4f}s")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Loss: {loss:.4f}")
        print(f"  Round time: {round_time:.2f}s")

    results['total_time'] = time.time() - total_start
    _, peak = tracemalloc.get_traced_memory()
    results['total_memory_peak_mb'] = peak / (1024 * 1024)
    tracemalloc.stop()

    # Final metrics
    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS - FL + FHE Decentralized")
    print(f"{'=' * 60}")
    print(f"  Final Accuracy: {results['rounds'][-1]['accuracy']:.4f}")
    print(f"  Final Loss: {results['rounds'][-1]['loss']:.4f}")
    print(f"  Total Time: {results['total_time']:.2f}s")
    print(f"  Peak Memory: {results['total_memory_peak_mb']:.2f} MB")
    print(f"  Avg Encryption Time/round: {np.mean([r['encryption_time'] for r in results['rounds']]):.2f}s")
    print(f"  Total Comm Hops: {sum(r['communication_hops'] for r in results['rounds'])}")

    # Save results
    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'fhe_decentralized_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {save_path / 'fhe_decentralized_results.pkl'}")

    return results


if __name__ == '__main__':
    results = run_fhe_decentralized(
        num_clients=2,
        num_rounds=5,
        local_epochs=1,
        lr=0.01,
        momentum=0.995,
    )
