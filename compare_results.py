"""
Compare Results: Vẽ biểu đồ và bảng so sánh 5 mô hình.
  I.  Tập trung: (1) Fully Centralized, (2) Semi-Centralized
  II. Phân tán:  (3) Data Parallel, (4) Model Parallel, (5) Hybrid
"""

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['axes.labelsize'] = 11

ALL_KEYS = ['fully_centralized', 'semi_centralized', 'data_parallel', 'model_parallel', 'hybrid']
ALL_NAMES = ['Tập Trung\nHoàn Toàn', 'Semi-\nCentralized', 'Data\nParallel', 'Model\nParallel', 'Hybrid']
SHORT_NAMES = ['Fully Central', 'Semi-Central', 'Data Parallel', 'Model Parallel', 'Hybrid']
COLORS = ['#e74c3c', '#e67e22', '#3498db', '#9b59b6', '#2ecc71']
GROUPS = ['centralized', 'centralized', 'distributed', 'distributed', 'distributed']


def load_results():
    save_path = Path('./outputs/benchmark')
    results = {}
    files = {
        'fully_centralized': 'fully_centralized_results.pkl',
        'semi_centralized': 'semi_centralized_results.pkl',
        'data_parallel': 'data_parallel_results.pkl',
        'model_parallel': 'model_parallel_results.pkl',
        'hybrid': 'hybrid_parallel_results.pkl',
    }
    for key, fn in files.items():
        fp = save_path / fn
        if fp.exists():
            with open(fp, 'rb') as f:
                results[key] = pickle.load(f)
            print(f"  ✓ {fn}")
        else:
            print(f"  ✗ {fn}")
            results[key] = None
    return results


def _get(r, field, default=0):
    if r is None: return default
    return r.get(field, default)


def plot_overall(results):
    """Biểu đồ tổng hợp: Accuracy, Time, Memory, Communication."""
    avail = [(k, n, c) for k, n, c in zip(ALL_KEYS, ALL_NAMES, COLORS) if results.get(k)]
    if len(avail) < 2:
        print("⚠ Cần ít nhất 2 mô hình!")
        return

    keys, names, cols = zip(*avail)
    data = [results[k] for k in keys]
    x = np.arange(len(keys))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('TỔNG HỢP: So sánh 5 Mô Hình (Tập Trung vs Phân Tán)',
                 fontsize=16, fontweight='bold')

    metrics = [
        (axes[0,0], 'Accuracy (%)', [_get(d,'final_accuracy',0)*100 for d in data], '%'),
        (axes[0,1], 'Thời gian huấn luyện (s)', [_get(d,'total_time') for d in data], 's'),
        (axes[1,0], 'Peak Memory (MB)', [_get(d,'peak_memory_mb') for d in data], 'MB'),
        (axes[1,1], 'Dữ liệu truyền tải (MB)', [_get(d,'total_comm_mb') for d in data], 'MB'),
    ]

    for ax, title, vals, unit in metrics:
        bars = ax.bar(x, vals, color=cols, width=0.55, edgecolor='black', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                    f'{v:.1f}{unit}', ha='center', fontweight='bold', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        # Thêm đường chia nhóm
        if len(keys) >= 3:
            ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
            ax.text(0.75, ax.get_ylim()[1]*0.95, 'TẬP TRUNG', ha='center',
                    fontsize=8, color='red', fontstyle='italic')
            ax.text(x[-1]-0.5, ax.get_ylim()[1]*0.95, 'PHÂN TÁN', ha='center',
                    fontsize=8, color='blue', fontstyle='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = Path('./outputs/benchmark')
    plt.savefig(save_path / 'overall_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: overall_comparison.png")


def plot_accuracy_over_epochs(results):
    """Biểu đồ accuracy theo epoch cho tất cả mô hình."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Accuracy theo Epoch - Tất cả Mô Hình', fontsize=14, fontweight='bold')
    markers = ['o', 's', '^', 'D', 'v']

    for i, (k, n, c) in enumerate(zip(ALL_KEYS, SHORT_NAMES, COLORS)):
        r = results.get(k)
        if r and 'epoch_metrics' in r:
            epochs = [m['epoch'] for m in r['epoch_metrics']]
            accs = [m['accuracy']*100 for m in r['epoch_metrics']]
            ax.plot(epochs, accs, f'{markers[i]}-', color=c, lw=2, ms=8, label=n)

    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    save_path = Path('./outputs/benchmark')
    plt.savefig(save_path / 'accuracy_over_epochs.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: accuracy_over_epochs.png")


def plot_centralized_vs_distributed(results):
    """So sánh nhóm tập trung vs phân tán."""
    central_keys = [k for k, g in zip(ALL_KEYS, GROUPS) if g == 'centralized' and results.get(k)]
    dist_keys = [k for k, g in zip(ALL_KEYS, GROUPS) if g == 'distributed' and results.get(k)]

    if not central_keys or not dist_keys:
        print("⚠ Thiếu dữ liệu!")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Tập Trung vs Phân Tán: So sánh Trung bình Nhóm',
                 fontsize=16, fontweight='bold')

    def avg_metric(keys, field):
        vals = [_get(results[k], field) for k in keys if results.get(k)]
        return np.mean(vals) if vals else 0

    group_names = ['Tập Trung', 'Phân Tán']
    group_colors = ['#e74c3c', '#3498db']

    comparisons = [
        ('Accuracy (%)', 'final_accuracy', 100, '%'),
        ('Thời gian (s)', 'total_time', 1, 's'),
        ('Peak Memory (MB)', 'peak_memory_mb', 1, 'MB'),
    ]

    for ax, (title, field, mult, unit) in zip(axes, comparisons):
        v_c = avg_metric(central_keys, field) * mult
        v_d = avg_metric(dist_keys, field) * mult
        vals = [v_c, v_d]
        bars = ax.bar(group_names, vals, color=group_colors, width=0.4, edgecolor='black')
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                    f'{v:.1f}{unit}', ha='center', fontweight='bold')
        ax.set_title(title); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = Path('./outputs/benchmark')
    plt.savefig(save_path / 'centralized_vs_distributed.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: centralized_vs_distributed.png")


def print_table(results):
    print("\n" + "=" * 110)
    print("  BẢNG SO SÁNH 5 MÔ HÌNH")
    print("=" * 110)

    print(f"\n{'Metric':<25}", end='')
    for n in SHORT_NAMES:
        print(f" {n:<17}", end='')
    print()
    print("-" * 110)

    rows = [
        ('Nhóm', lambda r, k: 'Tập Trung' if GROUPS[ALL_KEYS.index(k)]=='centralized' else 'Phân Tán'),
        ('Accuracy (%)', lambda r, k: f"{_get(r,'final_accuracy')*100:.2f}"),
        ('Total Time (s)', lambda r, k: f"{_get(r,'total_time'):.2f}"),
        ('Peak Memory (MB)', lambda r, k: f"{_get(r,'peak_memory_mb'):.1f}"),
        ('Comm Size (MB)', lambda r, k: f"{_get(r,'total_comm_mb'):.1f}"),
        ('CPU Usage (%)', lambda r, k: f"{_get(r,'cpu_percent'):.1f}"),
    ]

    for name, fn in rows:
        print(f"  {name:<23}", end='')
        for key in ALL_KEYS:
            r = results.get(key)
            try:
                val = fn(r, key) if r else 'N/A'
            except Exception:
                val = 'N/A'
            print(f" {val:<17}", end='')
        print()

    # Phân tích
    print(f"\n{'='*110}")
    print("  PHÂN TÍCH")
    print(f"{'='*110}")

    fc = results.get('fully_centralized')
    sc = results.get('semi_centralized')
    dp = results.get('data_parallel')
    mp_ = results.get('model_parallel')
    hy = results.get('hybrid')

    if fc and sc:
        bw_red = _get(sc, 'bandwidth_reduction', 0)
        print(f"\n  [1] Fully vs Semi-Centralized:")
        print(f"    • Semi-Centralized giảm {bw_red:.0f}% bandwidth nhờ tiền xử lý PCA")
        print(f"    • Accuracy: Fully={_get(fc,'final_accuracy')*100:.1f}% vs Semi={_get(sc,'final_accuracy')*100:.1f}%")

    if dp and mp_:
        print(f"\n  [2] Data Parallel vs Model Parallel:")
        print(f"    • Data Parallel: chia data, mỗi worker train full model → comm nhỏ (chỉ params)")
        print(f"    • Model Parallel: chia model, pipeline activations/gradients → comm lớn hơn")
        print(f"    • Comm: DP={_get(dp,'total_comm_mb'):.1f}MB vs MP={_get(mp_,'total_comm_mb'):.1f}MB")

    if hy:
        print(f"\n  [3] Hybrid:")
        print(f"    • Kết hợp chia data + chia model → phức tạp nhất")
        print(f"    • Phù hợp khi model lớn VÀ data nhiều")


def main():
    print("=" * 70)
    print("  📊 SO SÁNH 5 MÔ HÌNH: Tập Trung vs Phân Tán")
    print("=" * 70)

    results = load_results()
    print_table(results)
    plot_overall(results)
    plot_accuracy_over_epochs(results)
    plot_centralized_vs_distributed(results)


if __name__ == '__main__':
    main()
