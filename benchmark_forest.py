"""
Benchmark: Random Forest (Truyền thống - Centralized)
- Train trên TOÀN BỘ dataset tập trung (không chia cho clients)
- Dùng để so sánh với Federated Learning
- Metrics: Accuracy, Precision, Recall, F1, Training Time, Memory
"""

import time
import tracemalloc
import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, log_loss
)
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def load_mnist_numpy():
    """Load toàn bộ MNIST dataset dạng numpy."""
    print("  Loading MNIST dataset...")
    trainset = MNIST('./data', train=True, download=True, transform=ToTensor())
    testset = MNIST('./data', train=False, download=True, transform=ToTensor())

    # Flatten 28x28 → 784 features, normalize 0-1
    X_train = trainset.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_train = trainset.targets.numpy()
    X_test = testset.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_test = testset.targets.numpy()

    print(f"  Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test set:  {X_test.shape[0]} samples")

    return X_train, y_train, X_test, y_test


def run_random_forest(n_estimators=100, max_depth=20):
    """
    Train Random Forest truyền thống và thu thập metrics.
    """
    print("=" * 60)
    print("  RANDOM FOREST (Truyền thống - Centralized)")
    print("=" * 60)

    # Load data
    X_train, y_train, X_test, y_test = load_mnist_numpy()

    results = {
        'method': 'Random Forest',
        'model_type': 'RandomForestClassifier',
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'num_features': X_train.shape[1],
    }

    # ===== Training =====
    print(f"\n  Training Random Forest...")
    print(f"    n_estimators: {n_estimators}")
    print(f"    max_depth: {max_depth}")

    tracemalloc.start()
    t_train_start = time.time()

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    rf.fit(X_train, y_train)

    t_train_end = time.time()
    train_time = t_train_end - t_train_start

    current_mem, peak_mem = tracemalloc.get_traced_memory()

    results['training_time'] = train_time
    results['peak_memory_mb'] = peak_mem / (1024 * 1024)

    print(f"\n  Training time: {train_time:.2f}s")
    print(f"  Peak memory: {peak_mem / (1024 * 1024):.1f} MB")

    # ===== Evaluation =====
    print(f"\n  Evaluating on test set...")
    t_eval_start = time.time()

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    t_eval_end = time.time()
    eval_time = t_eval_end - t_eval_start

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    loss = log_loss(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    _, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results['eval_time'] = eval_time
    results['accuracy'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['f1_score'] = f1
    results['loss'] = loss
    results['confusion_matrix'] = cm.tolist()
    results['total_time'] = train_time + eval_time
    results['final_peak_memory_mb'] = final_peak / (1024 * 1024)

    # ===== Print Results =====
    print(f"\n{'=' * 60}")
    print(f"  KẾT QUẢ - Random Forest")
    print(f"{'=' * 60}")
    print(f"  Accuracy:       {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1-Score:       {f1:.4f}")
    print(f"  Log Loss:       {loss:.4f}")
    print(f"  Training Time:  {train_time:.2f}s")
    print(f"  Eval Time:      {eval_time:.2f}s")
    print(f"  Total Time:     {train_time + eval_time:.2f}s")
    print(f"  Peak Memory:    {final_peak / (1024 * 1024):.1f} MB")

    print(f"\n  Classification Report:")
    print(report)

    print(f"\n  Confusion Matrix:")
    print(cm)

    # ===== Save Results =====
    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / 'forest_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Cũng save dạng JSON để dễ đọc
    json_results = {k: v for k, v in results.items() if k != 'confusion_matrix'}
    json_results['confusion_matrix'] = cm.tolist()
    with open(save_path / 'forest_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to:")
    print(f"    {save_path / 'forest_results.pkl'}")
    print(f"    {save_path / 'forest_results.json'}")

    return results


if __name__ == '__main__':
    results = run_random_forest(
        n_estimators=100,
        max_depth=20,
    )
