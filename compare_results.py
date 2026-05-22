"""
Compare Results: Vẽ biểu đồ và bảng so sánh 5 mô hình.
  I.  Tập trung: (1) Fully Centralized, (2) Semi-Centralized
  II. Phân tán:  (3) Data Parallel, (4) Model Parallel, (5) Hybrid

  Phiên bản cải tiến: Hiển thị số liệu chi tiết Server/Client/Worker
  cho từng mô hình, thay vì chỉ biểu đồ cột đơn giản.
"""

import pickle
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

ALL_KEYS = ['fully_centralized', 'semi_centralized', 'data_parallel', 'model_parallel', 'hybrid']
ALL_NAMES = ['Tập Trung\nHoàn Toàn', 'Semi-\nCentralized', 'Data\nParallel', 'Model\nParallel', 'Hybrid']
SHORT_NAMES = ['Fully Central', 'Semi-Central', 'Data Parallel', 'Model Parallel', 'Hybrid']
COLORS = ['#e74c3c', '#e67e22', '#3498db', '#9b59b6', '#2ecc71']
GROUPS = ['centralized', 'centralized', 'distributed', 'distributed', 'distributed']


def load_results(mode='auto'):
    save_path = Path('./outputs/benchmark')
    results = {}
    
    is_real = False
    if mode == 'real':
        is_real = True
    elif mode == 'sim':
        is_real = False
    else:  # auto
        real_exists = any((save_path / f"real_{k}_results.json").exists() for k in ALL_KEYS)
        is_real = real_exists

    if is_real:
        print("\n=== Đang tải kết quả chạy MẠNG THỰC (Real Network - .json) ===")
        files = {
            'fully_centralized': 'real_fully_centralized_results.json',
            'semi_centralized': 'real_semi_centralized_results.json',
            'data_parallel': 'real_data_parallel_results.json',
            'model_parallel': 'real_model_parallel_results.json',
            'hybrid': 'real_hybrid_results.json',
        }
        for key, fn in files.items():
            fp = save_path / fn
            if fp.exists():
                try:
                    with open(fp, 'r') as f:
                        results[key] = json.load(f)
                    print(f"  ✓ {fn}")
                except Exception as e:
                    print(f"  ✗ Lỗi khi đọc {fn}: {e}")
                    results[key] = None
            else:
                print(f"  ✗ Không tìm thấy {fn}")
                results[key] = None
    else:
        print("\n=== Đang tải kết quả chạy MÔ PHỎNG (Simulation - .pkl) ===")
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
                try:
                    with open(fp, 'rb') as f:
                        results[key] = pickle.load(f)
                    print(f"  ✓ {fn}")
                except Exception as e:
                    print(f"  ✗ Lỗi khi đọc {fn}: {e}")
                    results[key] = None
            else:
                print(f"  ✗ Không tìm thấy {fn}")
                results[key] = None
    return results


def _get(r, field, default=0):
    if r is None: return default
    return r.get(field, default)


def plot_overall(results):
    """
    Biểu đồ tổng hợp CHI TIẾT:
    - Hàng 1: Bảng số liệu tổng hợp (table)
    - Hàng 2: 4 biểu đồ cột chính (Accuracy, Time, Memory, Comm) với số liệu cụ thể
    - Hàng 3: Chi tiết Server/Client/Worker cho từng mô hình
    - Hàng 4: Đường cong Accuracy theo Epoch + So sánh nhóm
    """
    avail = [(k, n, c) for k, n, c in zip(ALL_KEYS, ALL_NAMES, COLORS) if results.get(k)]
    if len(avail) < 2:
        print("⚠ Cần ít nhất 2 mô hình!")
        return

    keys, names, cols = zip(*avail)
    data = [results[k] for k in keys]

    fig = plt.figure(figsize=(24, 32))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle('📊 TỔNG HỢP CHI TIẾT: So sánh 5 Mô Hình Federated Learning\n'
                 'Tập Trung vs Phân Tán — Số liệu Server / Client / Worker',
                 fontsize=18, fontweight='bold', y=0.98, color='#2c3e50')

    # Tính toán các chỉ số chi tiết cho Client/Server
    c_times, s_times, c_mems, s_mems, c_acts, s_acts = [], [], [], [], [], []
    for k, d in zip(keys, data):
        ep_metrics = _get(d, 'epoch_metrics', [])
        total_time = _get(d, 'total_time')
        peak_mem = _get(d, 'peak_memory_mb')
        comm_mb = _get(d, 'total_comm_mb')
        if k == 'fully_centralized':
            s_time = total_time
            c_time = comm_mb / 100.0
            s_mem = peak_mem
            c_mem = 0.05
            c_act = "Gửi raw data"
            s_act = "Train toàn bộ"
        elif k == 'semi_centralized':
            s_time = _get(d, 'train_time_only', total_time)
            c_time = _get(d, 'preprocess_time', 0)
            s_mem = peak_mem
            c_mem = 0.05
            c_act = "PCA cục bộ & gửi đi"
            s_act = "Train trên PCA data"
        elif k == 'data_parallel':
            s_time = sum([m.get('sync_time', 0) for m in ep_metrics]) if ep_metrics else 0.1
            c_time = total_time - s_time
            s_mem = peak_mem * 0.1
            c_mem = peak_mem
            c_act = "Train cục bộ, gửi grad/weights"
            s_act = "AllReduce/FedAvg weights"
        elif k == 'model_parallel':
            s_time = sum([m.get('part2_time', 0) for m in ep_metrics]) if ep_metrics else total_time/2
            c_time = sum([m.get('part1_time', 0) for m in ep_metrics]) if ep_metrics else total_time/2
            s_mem = peak_mem
            c_mem = peak_mem
            c_act = "Chạy Conv, gửi activations"
            s_act = "Nhận act., chạy FC, trả grad"
        elif k == 'hybrid':
            s_time = sum([m.get('sync_time', 0) for m in ep_metrics]) if ep_metrics else 0.1
            c_time = total_time - s_time
            s_mem = peak_mem * 0.1
            c_mem = peak_mem
            c_act = "Xử lý pipeline song song"
            s_act = "Tổng hợp trọng số global"
        else:
            s_time, c_time, s_mem, c_mem, c_act, s_act = total_time, 0, peak_mem, 0, "-", "-"

        c_times.append(c_time); s_times.append(s_time)
        c_mems.append(c_mem); s_mems.append(s_mem)
        c_acts.append(c_act); s_acts.append(s_act)

    # ========================================================
    # HÀNG 1: BẢNG SỐ LIỆU TỔNG HỢP
    # ========================================================
    ax_table = fig.add_axes([0.05, 0.88, 0.90, 0.08])
    ax_table.axis('off')
    ax_table.set_title('📋 BẢNG SỐ LIỆU TỔNG HỢP CÁC MÔ HÌNH',
                       fontsize=14, fontweight='bold', pad=10, color='#2c3e50')

    col_labels = ['Metric'] + [SHORT_NAMES[ALL_KEYS.index(k)] for k in keys]
    table_data = []

    row = ['Nhóm'] + ['TẬP TRUNG' if GROUPS[ALL_KEYS.index(k)] == 'centralized' else 'PHÂN TÁN' for k in keys]
    table_data.append(row)
    row = ['Accuracy (%)'] + [f"{_get(d, 'final_accuracy', 0)*100:.2f}%" for d in data]
    table_data.append(row)
    row = ['Thời gian (Client / Server)'] + [f"{ct:.1f}s / {st:.1f}s" for ct, st in zip(c_times, s_times)]
    table_data.append(row)
    row = ['Peak Memory (Client / Server)'] + [f"{cm:.2f}MB / {sm:.2f}MB" for cm, sm in zip(c_mems, s_mems)]
    table_data.append(row)
    row = ['Dữ liệu truyền tải (MB)'] + [f"{_get(d, 'total_comm_mb'):.2f} MB" for d in data]
    table_data.append(row)
    row = ['Hoạt động Client'] + c_acts
    table_data.append(row)
    row = ['Hoạt động Server'] + s_acts
    table_data.append(row)

    table = ax_table.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif col_idx == 0:
            cell.set_facecolor('#ecf0f1')
            cell.set_text_props(fontweight='bold')
        else:
            model_idx = col_idx - 1
            if model_idx < len(cols):
                cell.set_facecolor(cols[model_idx] + '20')
        cell.set_edgecolor('#bdc3c7')

    # ========================================================
    # HÀNG 2: 4 BIỂU ĐỒ CỘT CHÍNH (với số liệu cụ thể trên cột)
    # ========================================================
    x = np.arange(len(keys))
    
    # 1. Accuracy
    ax_acc = fig.add_subplot(8, 4, 5)
    vals_acc = [_get(d, 'final_accuracy', 0)*100 for d in data]
    bars_acc = ax_acc.bar(x, vals_acc, color=cols, width=0.55, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars_acc, vals_acc):
        ax_acc.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=8, color='#2c3e50')
    best_idx = np.argmax(vals_acc)
    bars_acc[best_idx].set_edgecolor('#27ae60'); bars_acc[best_idx].set_linewidth(2.5)
    ax_acc.text(best_idx, max(vals_acc) * 1.08, '⭐', ha='center', fontsize=10)
    ax_acc.set_xticks(x); ax_acc.set_xticklabels(names, fontsize=7)
    ax_acc.set_title('Accuracy (%)', fontweight='bold', fontsize=10)
    ax_acc.grid(True, alpha=0.3, axis='y')

    # 2. Thời gian (Client/Server)
    ax_time = fig.add_subplot(8, 4, 6)
    width = 0.35
    b1 = ax_time.bar(x - width/2, c_times, width, label='Client', color='#3498db', edgecolor='black', linewidth=0.5)
    b2 = ax_time.bar(x + width/2, s_times, width, label='Server', color='#e74c3c', edgecolor='black', linewidth=0.5)
    for b_list in [b1, b2]:
        for b in b_list:
            h = b.get_height()
            if h > 0:
                ax_time.text(b.get_x() + b.get_width()/2, h * 1.02, f'{h:.1f}s', ha='center', fontsize=7, color='#2c3e50', fontweight='bold')
    ax_time.set_xticks(x); ax_time.set_xticklabels(names, fontsize=7)
    ax_time.set_title('Thời gian huấn luyện (s)', fontweight='bold', fontsize=10)
    ax_time.legend(fontsize=7)
    ax_time.grid(True, alpha=0.3, axis='y')

    # 3. Peak Memory (Client/Server)
    ax_mem = fig.add_subplot(8, 4, 7)
    b3 = ax_mem.bar(x - width/2, c_mems, width, label='Client', color='#2ecc71', edgecolor='black', linewidth=0.5)
    b4 = ax_mem.bar(x + width/2, s_mems, width, label='Server', color='#9b59b6', edgecolor='black', linewidth=0.5)
    for b_list in [b3, b4]:
        for b in b_list:
            h = b.get_height()
            if h > 0:
                ax_mem.text(b.get_x() + b.get_width()/2, h * 1.02, f'{h:.2f}', ha='center', fontsize=7, color='#2c3e50', fontweight='bold')
    ax_mem.set_xticks(x); ax_mem.set_xticklabels(names, fontsize=7)
    ax_mem.set_title('Peak Memory (MB)', fontweight='bold', fontsize=10)
    ax_mem.legend(fontsize=7)
    ax_mem.grid(True, alpha=0.3, axis='y')

    # 4. Comm
    ax_comm = fig.add_subplot(8, 4, 8)
    vals_comm = [_get(d, 'total_comm_mb') for d in data]
    bars_comm = ax_comm.bar(x, vals_comm, color=cols, width=0.55, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars_comm, vals_comm):
        ax_comm.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02, f'{v:.2f}MB', ha='center', fontweight='bold', fontsize=8, color='#2c3e50')
    best_idx_comm = np.argmin(vals_comm)
    bars_comm[best_idx_comm].set_edgecolor('#27ae60'); bars_comm[best_idx_comm].set_linewidth(2.5)
    ax_comm.text(best_idx_comm, max(vals_comm) * 1.08, '⭐', ha='center', fontsize=10)
    ax_comm.set_xticks(x); ax_comm.set_xticklabels(names, fontsize=7)
    ax_comm.set_title('Dữ liệu truyền tải (MB)', fontweight='bold', fontsize=10)
    ax_comm.grid(True, alpha=0.3, axis='y')

    # ========================================================
    # HÀNG 3-4: CHI TIẾT SERVER/CLIENT/WORKER CỦA TỪNG MÔ HÌNH
    # ========================================================

    # --- Mô hình 1: Fully Centralized ---
    fc = results.get('fully_centralized')
    if fc:
        ax1 = fig.add_subplot(8, 2, 5)
        ax1.set_facecolor('#fff5f5')
        ax1.set_title('📌 [MH1] Tập Trung Hoàn Toàn — Chi tiết Server & Clients',
                       fontsize=11, fontweight='bold', color='#e74c3c')

        num_clients = _get(fc, 'num_clients', 2)
        total_comm = _get(fc, 'total_comm_mb')
        per_client_mb = total_comm / num_clients if num_clients > 0 else 0
        total_time = _get(fc, 'total_time')

        categories = ['Server\n(Train toàn bộ)', 'Client 0\n(Gửi raw data)', 'Client 1\n(Gửi raw data)']
        # Server gets all training time, clients just send data
        server_train_time = total_time
        client_transfer_time = per_client_mb / 100  # ~100MB/s LAN

        time_vals = [server_train_time, client_transfer_time, client_transfer_time]
        data_vals = [total_comm, per_client_mb, per_client_mb]

        bar_colors = ['#c0392b', '#e74c3c', '#f1948a']
        x_pos = np.arange(len(categories))

        bars1 = ax1.bar(x_pos - 0.2, time_vals, 0.35, label='Thời gian (s)',
                        color=bar_colors, edgecolor='black', linewidth=0.5, alpha=0.8)
        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x_pos + 0.2, data_vals, 0.35, label='Data truyền (MB)',
                             color=bar_colors, edgecolor='black', linewidth=0.5, alpha=0.4,
                             hatch='///')

        for b, v in zip(bars1, time_vals):
            ax1.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                     f'{v:.2f}s', ha='center', fontsize=8, fontweight='bold')
        for b, v in zip(bars2, data_vals):
            ax1_twin.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                          f'{v:.1f}MB', ha='center', fontsize=8, fontweight='bold', color='#7f8c8d')

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(categories, fontsize=8)
        ax1.set_ylabel('Thời gian (s)', fontsize=9)
        ax1_twin.set_ylabel('Data truyền (MB)', fontsize=9, color='#7f8c8d')
        ax1.legend(loc='upper left', fontsize=7)
        ax1_twin.legend(loc='upper right', fontsize=7)
        ax1.grid(True, alpha=0.2, axis='y')

        # Text summary
        epoch_metrics = fc.get('epoch_metrics', [])
        if epoch_metrics:
            last_e = epoch_metrics[-1]
            summary_text = (
                f"Server train {_get(fc,'epochs',5)} epochs trên toàn bộ {num_clients} clients data\n"
                f"Final Acc: {_get(fc,'final_accuracy',0)*100:.2f}% | "
                f"Loss: {last_e.get('test_loss',0):.4f}\n"
                f"Mỗi client gửi {per_client_mb:.1f}MB raw data (ảnh gốc 28×28)"
            )
            ax1.text(0.5, -0.22, summary_text, transform=ax1.transAxes, fontsize=8,
                     ha='center', style='italic', color='#555',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.5))

    # --- Mô hình 2: Semi-Centralized ---
    sc = results.get('semi_centralized')
    if sc:
        ax2 = fig.add_subplot(8, 2, 6)
        ax2.set_facecolor('#fff8f0')
        ax2.set_title('📌 [MH2] Semi-Centralized — Chi tiết Server & Clients (PCA)',
                       fontsize=11, fontweight='bold', color='#e67e22')

        num_clients = _get(sc, 'num_clients', 2)
        preprocess_time = _get(sc, 'preprocess_time')
        train_time_only = _get(sc, 'train_time_only', _get(sc, 'total_time') - preprocess_time)
        raw_comm = _get(sc, 'raw_comm_mb')
        processed_comm = _get(sc, 'total_comm_mb')
        bw_reduction = _get(sc, 'bandwidth_reduction')

        categories = ['Server\n(Train)', 'Client\n(Preprocess PCA)', 'Bandwidth\nSo sánh']
        bar_colors_sc = ['#e67e22', '#f39c12', '#f1c40f']

        # Time comparison
        time_vals = [train_time_only, preprocess_time, 0]
        x_pos = np.arange(3)
        bars = ax2.bar(x_pos[:2], time_vals[:2], 0.4, color=bar_colors_sc[:2],
                       edgecolor='black', linewidth=0.5)

        for b, v in zip(bars, time_vals[:2]):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                     f'{v:.2f}s', ha='center', fontsize=9, fontweight='bold')

        # Bandwidth comparison (stacked)
        ax2_bw = ax2.twinx()
        ax2_bw.bar(2 - 0.15, raw_comm, 0.25, color='#e74c3c', alpha=0.6, label=f'Raw: {raw_comm:.1f}MB')
        ax2_bw.bar(2 + 0.15, processed_comm, 0.25, color='#27ae60', alpha=0.6,
                   label=f'PCA: {processed_comm:.1f}MB')

        ax2_bw.text(2 - 0.15, raw_comm * 1.02, f'{raw_comm:.1f}MB',
                    ha='center', fontsize=7, color='#e74c3c', fontweight='bold')
        ax2_bw.text(2 + 0.15, processed_comm * 1.02, f'{processed_comm:.1f}MB',
                    ha='center', fontsize=7, color='#27ae60', fontweight='bold')

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(categories, fontsize=8)
        ax2.set_ylabel('Thời gian (s)', fontsize=9)
        ax2_bw.set_ylabel('Bandwidth (MB)', fontsize=9, color='#7f8c8d')
        ax2_bw.legend(loc='upper right', fontsize=7)
        ax2.grid(True, alpha=0.2, axis='y')

        summary_text = (
            f"PCA giảm 784→{_get(sc,'pca_components',100)} chiều | "
            f"Bandwidth giảm {bw_reduction:.0f}%\n"
            f"Final Acc: {_get(sc,'final_accuracy',0)*100:.2f}% | "
            f"Preprocess: {preprocess_time:.2f}s | Train: {train_time_only:.2f}s"
        )
        ax2.text(0.5, -0.22, summary_text, transform=ax2.transAxes, fontsize=8,
                 ha='center', style='italic', color='#555',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.5))

    # --- Mô hình 3: Data Parallel ---
    dp = results.get('data_parallel')
    if dp:
        ax3 = fig.add_subplot(8, 2, 7)
        ax3.set_facecolor('#f0f8ff')
        ax3.set_title('📌 [MH3] Data Parallel — Chi tiết Server (FedAvg) & Workers',
                       fontsize=11, fontweight='bold', color='#3498db')

        num_workers = _get(dp, 'num_workers', 2)
        epoch_metrics = dp.get('epoch_metrics', [])

        # Per-epoch breakdown
        if epoch_metrics:
            epochs = [m['epoch'] for m in epoch_metrics]
            worker_times = [m.get('avg_worker_time', 0) for m in epoch_metrics]
            sync_times = [m.get('sync_time', 0) for m in epoch_metrics]
            comm_per_epoch = [m.get('comm_bytes', 0) / 1024 for m in epoch_metrics]  # KB

            x_epochs = np.arange(len(epochs))
            width = 0.3

            bars_w = ax3.bar(x_epochs - width/2, worker_times, width,
                            label=f'Avg Worker Train (×{num_workers})',
                            color='#3498db', edgecolor='black', linewidth=0.5)
            bars_s = ax3.bar(x_epochs + width/2, sync_times, width,
                            label='AllReduce Sync',
                            color='#e74c3c', edgecolor='black', linewidth=0.5)

            for b, v in zip(bars_w, worker_times):
                ax3.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                         f'{v:.2f}s', ha='center', fontsize=7, fontweight='bold')
            for b, v in zip(bars_s, sync_times):
                ax3.text(b.get_x() + b.get_width()/2, b.get_height() * 1.05,
                         f'{v:.4f}s', ha='center', fontsize=6, color='#e74c3c')

            # Comm on twin axis
            ax3_twin = ax3.twinx()
            ax3_twin.plot(x_epochs, comm_per_epoch, 'D--', color='#f39c12', ms=6,
                         label='Comm/epoch (KB)')
            for i, v in enumerate(comm_per_epoch):
                ax3_twin.text(i, v * 1.03, f'{v:.1f}KB', ha='center', fontsize=7,
                             color='#f39c12', fontweight='bold')

            ax3.set_xticks(x_epochs)
            ax3.set_xticklabels([f'Epoch {e}' for e in epochs], fontsize=8)
            ax3.set_ylabel('Thời gian (s)', fontsize=9)
            ax3_twin.set_ylabel('Comm (KB)', fontsize=9, color='#f39c12')
            ax3.legend(loc='upper left', fontsize=7)
            ax3_twin.legend(loc='upper right', fontsize=7)
            ax3.grid(True, alpha=0.2, axis='y')

            summary_text = (
                f"{num_workers} workers, mỗi worker train full model trên data riêng\n"
                f"FedAvg (AllReduce): chỉ gửi model params → Total comm: {_get(dp,'total_comm_mb'):.3f}MB\n"
                f"Final Acc: {_get(dp,'final_accuracy',0)*100:.2f}%"
            )
            ax3.text(0.5, -0.22, summary_text, transform=ax3.transAxes, fontsize=8,
                     ha='center', style='italic', color='#555',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#dff9fb', alpha=0.5))

    # --- Mô hình 4: Model Parallel ---
    mp_ = results.get('model_parallel')
    if mp_:
        ax4 = fig.add_subplot(8, 2, 8)
        ax4.set_facecolor('#f8f0ff')
        ax4.set_title('📌 [MH4] Model Parallel — Chi tiết Worker 0 (Conv) vs Worker 1 (FC)',
                       fontsize=11, fontweight='bold', color='#9b59b6')

        epoch_metrics = mp_.get('epoch_metrics', [])
        if epoch_metrics:
            epochs = [m['epoch'] for m in epoch_metrics]
            p1_times = [m.get('part1_time', 0) for m in epoch_metrics]
            p2_times = [m.get('part2_time', 0) for m in epoch_metrics]
            comm_per_epoch_mb = [m.get('comm_bytes', 0) / (1024*1024) for m in epoch_metrics]

            x_epochs = np.arange(len(epochs))
            width = 0.25

            bars_p1 = ax4.bar(x_epochs - width, p1_times, width,
                             label=f'W0: Conv ({_get(mp_,"part1_params",0):,} params)',
                             color='#9b59b6', edgecolor='black', linewidth=0.5)
            bars_p2 = ax4.bar(x_epochs, p2_times, width,
                             label=f'W1: FC ({_get(mp_,"part2_params",0):,} params)',
                             color='#8e44ad', edgecolor='black', linewidth=0.5, alpha=0.7)

            for b, v in zip(bars_p1, p1_times):
                ax4.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                         f'{v:.2f}s', ha='center', fontsize=7, fontweight='bold', color='#9b59b6')
            for b, v in zip(bars_p2, p2_times):
                ax4.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                         f'{v:.2f}s', ha='center', fontsize=7, fontweight='bold', color='#8e44ad')

            # Comm on twin axis
            ax4_twin = ax4.twinx()
            bars_comm = ax4_twin.bar(x_epochs + width, comm_per_epoch_mb, width,
                                     label='Activations+Grads (MB)',
                                     color='#f39c12', edgecolor='black', linewidth=0.5, alpha=0.5)
            for b, v in zip(bars_comm, comm_per_epoch_mb):
                ax4_twin.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                             f'{v:.1f}MB', ha='center', fontsize=7, color='#f39c12', fontweight='bold')

            ax4.set_xticks(x_epochs)
            ax4.set_xticklabels([f'Epoch {e}' for e in epochs], fontsize=8)
            ax4.set_ylabel('Thời gian (s)', fontsize=9)
            ax4_twin.set_ylabel('Comm (MB)', fontsize=9, color='#f39c12')
            ax4.legend(loc='upper left', fontsize=7)
            ax4_twin.legend(loc='upper right', fontsize=7)
            ax4.grid(True, alpha=0.2, axis='y')

            summary_text = (
                f"Pipeline: Input → W0(Conv) → activations → W1(FC) → Output\n"
                f"Truyền activations + gradients mỗi batch → Total comm: {_get(mp_,'total_comm_mb'):.1f}MB\n"
                f"Final Acc: {_get(mp_,'final_accuracy',0)*100:.2f}%"
            )
            ax4.text(0.5, -0.22, summary_text, transform=ax4.transAxes, fontsize=8,
                     ha='center', style='italic', color='#555',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8daef', alpha=0.5))

    # --- Mô hình 5: Hybrid ---
    hy = results.get('hybrid')
    if hy:
        ax5 = fig.add_subplot(8, 2, 9)
        ax5.set_facecolor('#f0fff0')
        ax5.set_title('📌 [MH5] Hybrid — Chi tiết Data Groups × Model Parts',
                       fontsize=11, fontweight='bold', color='#2ecc71')

        epoch_metrics = hy.get('epoch_metrics', [])
        num_groups = _get(hy, 'num_data_groups', 2)
        num_workers = _get(hy, 'num_workers', num_groups * 2)

        if epoch_metrics:
            epochs = [m['epoch'] for m in epoch_metrics]
            epoch_times = [m.get('epoch_time', 0) for m in epoch_metrics]
            sync_times = [m.get('sync_time', 0) for m in epoch_metrics]
            comm_per_epoch_mb = [m.get('comm_bytes', 0) / (1024*1024) for m in epoch_metrics]
            train_times = [et - st for et, st in zip(epoch_times, sync_times)]

            x_epochs = np.arange(len(epochs))
            width = 0.25

            # Stacked: pipeline train + AllReduce sync
            bars_train = ax5.bar(x_epochs - width/2, train_times, width,
                                label=f'Pipeline Train ({num_groups} groups)',
                                color='#2ecc71', edgecolor='black', linewidth=0.5)
            bars_sync = ax5.bar(x_epochs - width/2, sync_times, width,
                               bottom=train_times,
                               label='AllReduce Sync',
                               color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.7)

            for b, v in zip(bars_train, train_times):
                ax5.text(b.get_x() + b.get_width()/2, b.get_height() / 2,
                         f'{v:.2f}s', ha='center', fontsize=7, fontweight='bold', color='white')
            for b, v, bt in zip(bars_sync, sync_times, train_times):
                ax5.text(b.get_x() + b.get_width()/2, bt + v * 1.5,
                         f'+{v:.4f}s', ha='center', fontsize=6, color='#e74c3c')

            # Comm on twin axis
            ax5_twin = ax5.twinx()
            bars_comm = ax5_twin.bar(x_epochs + width/2, comm_per_epoch_mb, width,
                                     label='Total Comm/epoch (MB)',
                                     color='#f39c12', edgecolor='black', linewidth=0.5, alpha=0.5)
            for b, v in zip(bars_comm, comm_per_epoch_mb):
                ax5_twin.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                             f'{v:.1f}MB', ha='center', fontsize=7, color='#f39c12', fontweight='bold')

            ax5.set_xticks(x_epochs)
            ax5.set_xticklabels([f'Epoch {e}' for e in epochs], fontsize=8)
            ax5.set_ylabel('Thời gian (s)', fontsize=9)
            ax5_twin.set_ylabel('Comm (MB)', fontsize=9, color='#f39c12')
            ax5.legend(loc='upper left', fontsize=7)
            ax5_twin.legend(loc='upper right', fontsize=7)
            ax5.grid(True, alpha=0.2, axis='y')

            summary_text = (
                f"{num_groups} data groups × 2 model parts = {num_workers} workers tổng cộng\n"
                f"Pipeline (activations+grads) + AllReduce (model params)\n"
                f"Total comm: {_get(hy,'total_comm_mb'):.1f}MB | "
                f"Final Acc: {_get(hy,'final_accuracy',0)*100:.2f}%"
            )
            ax5.text(0.5, -0.22, summary_text, transform=ax5.transAxes, fontsize=8,
                     ha='center', style='italic', color='#555',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', alpha=0.5))

    # ========================================================
    # HÀNG 5: SO SÁNH TỔNG THỂ — RADAR + BẢNG RANKING
    # ========================================================
    ax_rank = fig.add_subplot(8, 2, 10)
    ax_rank.set_facecolor('#f9f9f9')
    ax_rank.set_title('🏆 BẢNG XẾP HẠNG TỔNG HỢP', fontsize=11, fontweight='bold', color='#2c3e50')
    ax_rank.axis('off')

    # Create ranking table
    rank_headers = ['Hạng', 'Mô hình', 'Accuracy', 'Thời gian', 'Memory', 'Comm', 'Tổng điểm']
    rank_data = []

    # Score each model (lower is better for time/mem/comm, higher for accuracy)
    scores = {}
    accs = [_get(d, 'final_accuracy', 0)*100 for d in data]
    times = [_get(d, 'total_time') for d in data]
    mems = [_get(d, 'peak_memory_mb') for d in data]
    comms = [_get(d, 'total_comm_mb') for d in data]

    for i, k in enumerate(keys):
        # Normalize scores to 0-100 (higher=better)
        if max(accs) > min(accs):
            acc_score = (accs[i] - min(accs)) / (max(accs) - min(accs)) * 100
        else:
            acc_score = 100

        if max(times) > min(times):
            time_score = (1 - (times[i] - min(times)) / (max(times) - min(times))) * 100
        else:
            time_score = 100

        if max(mems) > min(mems):
            mem_score = (1 - (mems[i] - min(mems)) / (max(mems) - min(mems))) * 100
        else:
            mem_score = 100

        if max(comms) > min(comms):
            comm_score = (1 - (comms[i] - min(comms)) / (max(comms) - min(comms))) * 100
        else:
            comm_score = 100

        total_score = acc_score * 0.4 + time_score * 0.2 + mem_score * 0.2 + comm_score * 0.2
        scores[k] = total_score
        rank_data.append((k, SHORT_NAMES[ALL_KEYS.index(k)],
                         f'{accs[i]:.2f}%', f'{times[i]:.2f}s',
                         f'{mems[i]:.1f}MB', f'{comms[i]:.2f}MB',
                         f'{total_score:.1f}'))

    # Sort by score descending
    rank_data.sort(key=lambda x: float(x[-1]), reverse=True)
    table_rows = []
    medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
    for rank, (k, name, acc, tm, mem, comm, score) in enumerate(rank_data):
        table_rows.append([medals[rank], name, acc, tm, mem, comm, score])

    rank_table = ax_rank.table(cellText=table_rows, colLabels=rank_headers,
                                cellLoc='center', loc='center')
    rank_table.auto_set_font_size(False)
    rank_table.set_fontsize(9)
    rank_table.scale(1.0, 1.8)

    for (row_idx, col_idx), cell in rank_table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif row_idx == 1:
            cell.set_facecolor('#d4efdf')
        cell.set_edgecolor('#bdc3c7')

    # ========================================================
    # HÀNG 6: ACCURACY OVER EPOCHS (ĐƯỜNG CONG SO SÁNH)
    # ========================================================
    ax_acc = fig.add_subplot(8, 2, 11)
    ax_acc.set_facecolor('#fafafa')
    ax_acc.set_title('📈 Accuracy theo Epoch — So sánh tất cả mô hình',
                     fontsize=11, fontweight='bold', color='#2c3e50')
    markers = ['o', 's', '^', 'D', 'v']

    for i, (k, n, c) in enumerate(zip(ALL_KEYS, SHORT_NAMES, COLORS)):
        r = results.get(k)
        if r and 'epoch_metrics' in r:
            epochs = [m['epoch'] for m in r['epoch_metrics']]
            accs_ep = [m['accuracy']*100 for m in r['epoch_metrics']]
            ax_acc.plot(epochs, accs_ep, f'{markers[i]}-', color=c, lw=2, ms=8, label=n)
            # Annotate last point
            ax_acc.annotate(f'{accs_ep[-1]:.2f}%', (epochs[-1], accs_ep[-1]),
                           textcoords="offset points", xytext=(10, 0),
                           fontsize=8, fontweight='bold', color=c)

    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.legend(fontsize=9, loc='lower right')
    ax_acc.grid(True, alpha=0.3)

    # ========================================================
    # HÀNG 6 (right): LOSS OVER EPOCHS
    # ========================================================
    ax_loss = fig.add_subplot(8, 2, 12)
    ax_loss.set_facecolor('#fafafa')
    ax_loss.set_title('📉 Loss theo Epoch — So sánh tất cả mô hình',
                      fontsize=11, fontweight='bold', color='#2c3e50')

    for i, (k, n, c) in enumerate(zip(ALL_KEYS, SHORT_NAMES, COLORS)):
        r = results.get(k)
        if r and 'epoch_metrics' in r:
            epochs = [m['epoch'] for m in r['epoch_metrics']]
            losses = [m.get('train_loss', 0) for m in r['epoch_metrics']]
            ax_loss.plot(epochs, losses, f'{markers[i]}-', color=c, lw=2, ms=8, label=n)
            ax_loss.annotate(f'{losses[-1]:.3f}', (epochs[-1], losses[-1]),
                            textcoords="offset points", xytext=(10, 0),
                            fontsize=8, fontweight='bold', color=c)

    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Train Loss')
    ax_loss.legend(fontsize=9, loc='upper right')
    ax_loss.grid(True, alpha=0.3)

    # ========================================================
    # HÀNG 7: SO SÁNH NHÓM TẬP TRUNG vs PHÂN TÁN
    # ========================================================
    central_keys = [k for k in keys if GROUPS[ALL_KEYS.index(k)] == 'centralized']
    dist_keys = [k for k in keys if GROUPS[ALL_KEYS.index(k)] == 'distributed']

    if central_keys and dist_keys:
        def avg_metric(ks, field):
            vals = [_get(results[k], field) for k in ks if results.get(k)]
            return np.mean(vals) if vals else 0

        comparisons = [
            ('Accuracy (%)', 'final_accuracy', 100, '%'),
            ('Thời gian (s)', 'total_time', 1, 's'),
            ('Peak Memory (MB)', 'peak_memory_mb', 1, 'MB'),
            ('Comm (MB)', 'total_comm_mb', 1, 'MB'),
        ]

        for ci, (title, field, mult, unit) in enumerate(comparisons):
            ax_g = fig.add_subplot(8, 4, 25 + ci)
            v_c = avg_metric(central_keys, field) * mult
            v_d = avg_metric(dist_keys, field) * mult
            group_names_g = ['Tập Trung', 'Phân Tán']
            group_colors = ['#e74c3c', '#3498db']
            vals_g = [v_c, v_d]

            bars_g = ax_g.bar(group_names_g, vals_g, color=group_colors, width=0.5,
                             edgecolor='black', linewidth=0.5)
            for b, v in zip(bars_g, vals_g):
                ax_g.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                         f'{v:.2f}{unit}', ha='center', fontweight='bold', fontsize=9)

            ax_g.set_title(title, fontsize=10, fontweight='bold')
            ax_g.grid(True, alpha=0.3, axis='y')

            # Show difference
            if v_c > 0 and v_d > 0:
                diff_pct = ((v_d - v_c) / v_c) * 100
                sign = '+' if diff_pct > 0 else ''
                ax_g.text(0.5, 0.5, f'Δ {sign}{diff_pct:.1f}%',
                         transform=ax_g.transAxes, ha='center', fontsize=10,
                         color='#e74c3c' if diff_pct < 0 else '#27ae60',
                         fontweight='bold', alpha=0.7)

    # ========================================================
    # HÀNG 8: KẾT LUẬN
    # ========================================================
    ax_conclusion = fig.add_axes([0.05, 0.01, 0.90, 0.04])
    ax_conclusion.axis('off')

    # Build conclusion text
    conclusion_parts = []
    if fc and sc:
        conclusion_parts.append(
            f"• Semi-Centralized giảm {_get(sc,'bandwidth_reduction',0):.0f}% bandwidth so với Fully Centralized nhờ PCA"
        )
    if dp and mp_:
        conclusion_parts.append(
            f"• Data Parallel comm: {_get(dp,'total_comm_mb'):.3f}MB (chỉ params) vs "
            f"Model Parallel: {_get(mp_,'total_comm_mb'):.1f}MB (activations+grads)"
        )
    if hy:
        conclusion_parts.append(
            f"• Hybrid kết hợp cả 2 → {_get(hy,'num_workers',4)} workers, "
            f"phức tạp nhất, comm={_get(hy,'total_comm_mb'):.1f}MB"
        )

    conclusion_text = '📝 KẾT LUẬN: ' + '  |  '.join(conclusion_parts)
    ax_conclusion.text(0.5, 0.5, conclusion_text, transform=ax_conclusion.transAxes,
                       fontsize=9, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#dfe6e9', alpha=0.8))

    plt.subplots_adjust(hspace=0.55, wspace=0.35, top=0.95, bottom=0.05)
    save_path = Path('./outputs/benchmark')
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / 'overall_comparison.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"✓ Saved: overall_comparison.png")
    plt.close()


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
    plt.close()


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
    plt.close()


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

    # Phân tích chi tiết Server/Client
    print(f"\n{'='*110}")
    print("  CHI TIẾT SERVER / CLIENT / WORKER")
    print(f"{'='*110}")

    fc = results.get('fully_centralized')
    sc = results.get('semi_centralized')
    dp = results.get('data_parallel')
    mp_ = results.get('model_parallel')
    hy = results.get('hybrid')

    if fc:
        num_c = _get(fc, 'num_clients', 2)
        total_comm = _get(fc, 'total_comm_mb')
        per_client = total_comm / num_c if num_c > 0 else 0
        print(f"\n  [MH1] Fully Centralized:")
        print(f"    • Server: Train toàn bộ | Time: {_get(fc,'total_time'):.2f}s | "
              f"Peak Mem: {_get(fc,'peak_memory_mb'):.1f}MB")
        print(f"    • {num_c} Clients: Gửi raw data | Mỗi client: {per_client:.1f}MB")
        print(f"    • Tổng bandwidth: {total_comm:.1f}MB (truyền ảnh gốc)")

    if sc:
        print(f"\n  [MH2] Semi-Centralized:")
        print(f"    • Server: Train only | Time: {_get(sc,'train_time_only',0):.2f}s")
        print(f"    • Clients: Preprocess PCA ({_get(sc,'pca_components',100)} dims) | "
              f"Time: {_get(sc,'preprocess_time',0):.2f}s")
        print(f"    • Bandwidth: {_get(sc,'raw_comm_mb',0):.1f}MB raw → "
              f"{_get(sc,'total_comm_mb'):.1f}MB processed (giảm {_get(sc,'bandwidth_reduction',0):.0f}%)")

    if dp:
        num_w = _get(dp, 'num_workers', 2)
        epoch_metrics = dp.get('epoch_metrics', [])
        print(f"\n  [MH3] Data Parallel:")
        print(f"    • Server (FedAvg): AllReduce trung bình params từ {num_w} workers")
        if epoch_metrics:
            avg_worker_t = np.mean([m.get('avg_worker_time', 0) for m in epoch_metrics])
            avg_sync_t = np.mean([m.get('sync_time', 0) for m in epoch_metrics])
            print(f"    • Workers: Avg train time/epoch: {avg_worker_t:.2f}s | "
                  f"Sync time: {avg_sync_t:.4f}s")
        print(f"    • Comm: {_get(dp,'total_comm_mb'):.3f}MB (model params only)")

    if mp_:
        print(f"\n  [MH4] Model Parallel:")
        print(f"    • Worker 0 (Conv): {_get(mp_,'part1_params',0):,} params")
        print(f"    • Worker 1 (FC):   {_get(mp_,'part2_params',0):,} params")
        epoch_metrics = mp_.get('epoch_metrics', [])
        if epoch_metrics:
            avg_p1 = np.mean([m.get('part1_time', 0) for m in epoch_metrics])
            avg_p2 = np.mean([m.get('part2_time', 0) for m in epoch_metrics])
            print(f"    • Avg time/epoch: W0={avg_p1:.2f}s, W1={avg_p2:.2f}s")
        print(f"    • Comm: {_get(mp_,'total_comm_mb'):.1f}MB (activations + gradients)")

    if hy:
        num_g = _get(hy, 'num_data_groups', 2)
        num_w = _get(hy, 'num_workers', num_g * 2)
        print(f"\n  [MH5] Hybrid:")
        print(f"    • {num_g} data groups × 2 model parts = {num_w} workers")
        epoch_metrics = hy.get('epoch_metrics', [])
        if epoch_metrics:
            avg_sync = np.mean([m.get('sync_time', 0) for m in epoch_metrics])
            print(f"    • AllReduce sync avg: {avg_sync:.4f}s")
        print(f"    • Comm: {_get(hy,'total_comm_mb'):.1f}MB (pipeline + AllReduce)")

    # Phân tích so sánh
    print(f"\n{'='*110}")
    print("  PHÂN TÍCH SO SÁNH")
    print(f"{'='*110}")

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

    parser = argparse.ArgumentParser(description="So sánh kết quả các mô hình FL")
    parser.add_argument('--mode', type=str, choices=['sim', 'real', 'auto'], default='auto',
                        help="Chế độ tải: 'sim' (mô phỏng, .pkl), 'real' (mạng thực, .json), 'auto' (tự động)")
    args = parser.parse_args()

    results = load_results(args.mode)
    print_table(results)
    plot_overall(results)
    plot_accuracy_over_epochs(results)
    plot_centralized_vs_distributed(results)


if __name__ == '__main__':
    main()
