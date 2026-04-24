"""
Benchmark 2: Tập Trung Có Tiền Xử Lý (Semi-Centralized)
- Client xử lý sơ bộ data (chuẩn hóa, lọc, PCA giảm chiều)
- Gửi data ĐÃ XỬ LÝ (nhỏ hơn) về server
- Server train trên dữ liệu đã tiền xử lý
- ✅ Giảm tải server & bandwidth
- ❌ Vẫn phụ thuộc server trung tâm
"""

import time
import tracemalloc
import pickle
import psutil
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose


class NetSmall(nn.Module):
    """Mô hình nhỏ hơn vì input đã giảm chiều qua PCA."""
    def __init__(self, input_dim=100, num_classes=10):
        super(NetSmall, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def run_semi_centralized(num_clients=2, epochs=5, lr=0.01, momentum=0.9,
                          batch_size=64, pca_components=100):
    print("=" * 60)
    print("  MÔ HÌNH 2: TẬP TRUNG CÓ TIỀN XỬ LÝ (Semi-Centralized)")
    print("  Client tiền xử lý (PCA) → gửi data nén → Server train")
    print("=" * 60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST('./data', train=True, download=True, transform=tr)
    testset = MNIST('./data', train=False, download=True, transform=tr)

    # === PHASE 1: Client-side preprocessing ===
    per_client = len(trainset) // num_clients
    total_comm_bytes = 0
    all_X_processed = []
    all_y = []
    preprocess_times = []

    # Flatten full datasets
    X_train_full = trainset.data.numpy().reshape(-1, 28*28).astype(np.float32) / 255.0
    y_train_full = trainset.targets.numpy()
    X_test_full = testset.data.numpy().reshape(-1, 28*28).astype(np.float32) / 255.0
    y_test_full = testset.targets.numpy()

    # Tính raw data size để so sánh
    raw_data_size_per_client = per_client * 28 * 28 * 4
    total_raw_bytes = raw_data_size_per_client * num_clients

    for i in range(num_clients):
        start_idx = i * per_client
        end_idx = start_idx + per_client
        X_client = X_train_full[start_idx:end_idx]
        y_client = y_train_full[start_idx:end_idx]

        t_preprocess = time.time()

        # Tiền xử lý 1: Lọc outlier (loại bỏ ảnh gần trống)
        pixel_sums = X_client.sum(axis=1)
        valid_mask = pixel_sums > 10  # Loại ảnh quá tối
        X_client = X_client[valid_mask]
        y_client = y_client[valid_mask]

        # Tiền xử lý 2: Chuẩn hóa Z-score
        mean = X_client.mean(axis=0)
        std = X_client.std(axis=0) + 1e-8
        X_client = (X_client - mean) / std

        # Tiền xử lý 3: PCA giảm chiều (784 → pca_components)
        pca = PCA(n_components=pca_components, random_state=42)
        X_pca = pca.fit_transform(X_client)

        preprocess_time = time.time() - t_preprocess
        preprocess_times.append(preprocess_time)

        # Tính communication: chỉ gửi data đã giảm chiều
        processed_size = X_pca.nbytes + y_client.nbytes
        total_comm_bytes += processed_size

        all_X_processed.append(X_pca)
        all_y.append(y_client)

        print(f"  Client {i}: {len(X_client)} samples")
        print(f"    Raw: {raw_data_size_per_client/1024/1024:.1f}MB → Processed: {processed_size/1024/1024:.1f}MB")
        print(f"    Giảm {(1-processed_size/raw_data_size_per_client)*100:.0f}% bandwidth")
        print(f"    Preprocess time: {preprocess_time:.2f}s")

    # === PHASE 2: Server gom & train ===
    X_train = np.concatenate(all_X_processed, axis=0)
    y_train = np.concatenate(all_y, axis=0)

    # PCA cho test set (dùng PCA fit trên toàn bộ train để transform test)
    pca_full = PCA(n_components=pca_components, random_state=42)
    X_train_norm = (X_train_full - X_train_full.mean(axis=0)) / (X_train_full.std(axis=0) + 1e-8)
    pca_full.fit(X_train_norm)
    X_test_norm = (X_test_full - X_test_full.mean(axis=0)) / (X_test_full.std(axis=0) + 1e-8)
    X_test_pca = pca_full.transform(X_test_norm)

    train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    test_tensor = TensorDataset(torch.tensor(X_test_pca, dtype=torch.float32),
                                 torch.tensor(y_test_full, dtype=torch.long))
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=64)

    model = NetSmall(input_dim=pca_components).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    tracemalloc.start()
    cpu_before = psutil.cpu_percent(interval=None)
    t_start = time.time()

    results = {
        'method': 'Tập Trung Tiền Xử Lý (Semi-Centralized)',
        'group': 'centralized',
        'num_clients': num_clients,
        'epochs': epochs,
        'pca_components': pca_components,
        'epoch_metrics': [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        correct, test_loss = 0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                test_loss += criterion(outputs, y_batch).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / len(testset)
        epoch_time = time.time() - epoch_start
        current_mem, peak_mem = tracemalloc.get_traced_memory()

        results['epoch_metrics'].append({
            'epoch': epoch,
            'train_loss': epoch_loss / len(train_loader),
            'test_loss': test_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time,
            'peak_memory_mb': peak_mem / (1024*1024),
        })
        print(f"  Epoch {epoch}/{epochs}: Acc={accuracy*100:.2f}%, Loss={test_loss:.4f}, Time={epoch_time:.2f}s")

    total_time = time.time() - t_start
    cpu_after = psutil.cpu_percent(interval=0.5)
    _, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results['total_time'] = total_time + sum(preprocess_times)
    results['train_time_only'] = total_time
    results['preprocess_time'] = sum(preprocess_times)
    results['total_comm_bytes'] = total_comm_bytes
    results['total_comm_mb'] = total_comm_bytes / (1024*1024)
    results['raw_comm_mb'] = total_raw_bytes / (1024*1024)
    results['bandwidth_reduction'] = (1 - total_comm_bytes/total_raw_bytes) * 100
    results['peak_memory_mb'] = final_peak / (1024*1024)
    results['cpu_percent'] = cpu_after
    results['final_accuracy'] = results['epoch_metrics'][-1]['accuracy']
    results['final_loss'] = results['epoch_metrics'][-1]['test_loss']

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ - Semi-Centralized")
    print(f"{'='*60}")
    print(f"  Accuracy:      {results['final_accuracy']*100:.2f}%")
    print(f"  Total Time:    {results['total_time']:.2f}s (preprocess={results['preprocess_time']:.2f}s)")
    print(f"  Peak Memory:   {results['peak_memory_mb']:.1f}MB")
    print(f"  Comm Size:     {results['total_comm_mb']:.1f}MB (vs {results['raw_comm_mb']:.1f}MB raw)")
    print(f"  BW Reduction:  {results['bandwidth_reduction']:.0f}%")

    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'semi_centralized_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results


if __name__ == '__main__':
    run_semi_centralized(num_clients=2, epochs=5)
