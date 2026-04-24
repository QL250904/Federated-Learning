"""
Benchmark Runner: Chạy tất cả 5 mô hình và thu thập kết quả.

  I. MÔ HÌNH TẬP TRUNG (Centralized):
     1. Tập trung hoàn toàn (Fully Centralized)
     2. Tập trung có tiền xử lý (Semi-Centralized)

  II. MÔ HÌNH PHÂN TÁN (Distributed):
     3. Song song dữ liệu (Data Parallelism)
     4. Song song mô hình (Model Parallelism)
     5. Kết hợp (Hybrid Parallelism)
"""

import time
from pathlib import Path

NUM_CLIENTS = 2
EPOCHS = 5
LR = 0.01
MOMENTUM = 0.9


def run_all():
    print("=" * 70)
    print("  BENCHMARK RUNNER - So sánh 5 Mô Hình Tập Trung vs Phân Tán")
    print(f"  Cấu hình: {NUM_CLIENTS} clients/workers, {EPOCHS} epochs")
    print("=" * 70)

    all_results = {}
    total_start = time.time()

    configs = [
        ('fully_centralized', '[1/5] Tập Trung Hoàn Toàn',
         'bench_1_fully_centralized', 'run_fully_centralized',
         {'num_clients': NUM_CLIENTS, 'epochs': EPOCHS, 'lr': LR, 'momentum': MOMENTUM}),

        ('semi_centralized', '[2/5] Tập Trung Có Tiền Xử Lý',
         'bench_2_semi_centralized', 'run_semi_centralized',
         {'num_clients': NUM_CLIENTS, 'epochs': EPOCHS, 'lr': LR, 'momentum': MOMENTUM}),

        ('data_parallel', '[3/5] Song Song Dữ Liệu',
         'bench_3_data_parallel', 'run_data_parallelism',
         {'num_workers': NUM_CLIENTS, 'epochs': EPOCHS, 'lr': LR, 'momentum': MOMENTUM}),

        ('model_parallel', '[4/5] Song Song Mô Hình',
         'bench_4_model_parallel', 'run_model_parallelism',
         {'num_workers': NUM_CLIENTS, 'epochs': EPOCHS, 'lr': LR, 'momentum': MOMENTUM}),

        ('hybrid', '[5/5] Hybrid (Data + Model)',
         'bench_5_hybrid', 'run_hybrid_parallelism',
         {'num_data_groups': NUM_CLIENTS, 'epochs': EPOCHS, 'lr': LR, 'momentum': MOMENTUM}),
    ]

    for key, title, module_name, func_name, kwargs in configs:
        print("\n" + "▓" * 70)
        print(f"  {title}")
        print("▓" * 70)
        try:
            import importlib
            mod = importlib.import_module(module_name)
            func = getattr(mod, func_name)
            all_results[key] = func(**kwargs)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[key] = None

    total_elapsed = time.time() - total_start

    # Tổng kết
    print("\n" + "=" * 70)
    print("  TỔNG KẾT NHANH")
    print("=" * 70)
    print(f"  Tổng thời gian: {total_elapsed:.1f}s\n")

    for key, r in all_results.items():
        if r is None:
            print(f"  ✗ {key}: FAILED")
        else:
            acc = r.get('final_accuracy', 0) * 100
            t = r.get('total_time', 0)
            mem = r.get('peak_memory_mb', 0)
            comm = r.get('total_comm_mb', 0)
            print(f"  ✓ {key}: Acc={acc:.2f}%, Time={t:.1f}s, Mem={mem:.1f}MB, Comm={comm:.1f}MB")

    import pickle
    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n💡 Chạy 'python compare_results.py' để vẽ biểu đồ so sánh!")
    return all_results


if __name__ == '__main__':
    run_all()
