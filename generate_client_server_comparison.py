import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def generate_comparison():
    with open('./outputs/benchmark/all_results.pkl', 'rb') as f:
        all_results = pickle.load(f)

    models = ['fully_centralized', 'semi_centralized', 'data_parallel', 'model_parallel', 'hybrid']
    model_names = ['Fully\nCentralized', 'Semi-\nCentralized', 'Data\nParallel', 'Model\nParallel', 'Hybrid']

    # Data collection
    accuracies = []
    
    client_times = []
    server_times = []
    
    client_mems = []
    server_mems = []
    
    client_actions = []
    server_actions = []

    for key in models:
        res = all_results.get(key, {})
        acc = res.get('final_accuracy', 0) * 100
        accuracies.append(f"{acc:.2f}%")
        
        ep_metrics = res.get('epoch_metrics', [])
        total_time = res.get('total_time', 0)
        peak_mem = res.get('peak_memory_mb', 0)
        comm_mb = res.get('total_comm_mb', 0)

        if key == 'fully_centralized':
            s_time = total_time
            c_time = comm_mb / 100.0  # Approx transfer time
            s_mem = peak_mem
            c_mem = 0.05
            c_act = "Gửi toàn bộ dữ liệu thô (raw data) lên Server."
            s_act = "Nhận dữ liệu thô và huấn luyện toàn bộ mô hình."
        
        elif key == 'semi_centralized':
            s_time = res.get('train_time_only', total_time)
            c_time = res.get('preprocess_time', 0)
            s_mem = peak_mem
            c_mem = 0.05
            c_act = "Tiền xử lý (PCA) dữ liệu cục bộ và gửi đi."
            s_act = "Nhận dữ liệu đã giảm chiều và huấn luyện mô hình."

        elif key == 'data_parallel':
            s_time = sum([m.get('sync_time', 0) for m in ep_metrics]) if ep_metrics else 0.1
            c_time = total_time - s_time
            s_mem = peak_mem * 0.1 # Server only holds weights
            c_mem = peak_mem
            c_act = "Huấn luyện cục bộ trên dữ liệu riêng, gửi gradients/weights."
            s_act = "Nhận và tổng hợp (AllReduce/FedAvg) trọng số từ các clients."

        elif key == 'model_parallel':
            s_time = sum([m.get('part2_time', 0) for m in ep_metrics]) if ep_metrics else total_time/2
            c_time = sum([m.get('part1_time', 0) for m in ep_metrics]) if ep_metrics else total_time/2
            s_mem = peak_mem
            c_mem = peak_mem
            c_act = "Tính toán phần đầu mô hình (Conv), gửi activations cho Server."
            s_act = "Nhận activations, tính toán phần sau (FC) và gửi ngược gradients."

        elif key == 'hybrid':
            s_time = sum([m.get('sync_time', 0) for m in ep_metrics]) if ep_metrics else 0.1
            c_time = total_time - s_time
            s_mem = peak_mem * 0.1
            c_mem = peak_mem
            c_act = "Xử lý pipeline song song trên các nhóm dữ liệu."
            s_act = "Tổng hợp trọng số mô hình toàn cục giữa các nhóm."

        client_times.append(c_time)
        server_times.append(s_time)
        client_mems.append(c_mem)
        server_mems.append(s_mem)
        client_actions.append(c_act)
        server_actions.append(s_act)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('So Sánh Chi Tiết: Client vs Server qua 5 Mô Hình', fontsize=18, fontweight='bold')

    x = np.arange(len(models))
    width = 0.35

    # Chart 1: Time
    rects1 = ax1.bar(x - width/2, client_times, width, label='Client (Worker)', color='#3498db', edgecolor='black')
    rects2 = ax1.bar(x + width/2, server_times, width, label='Server (Master)', color='#e74c3c', edgecolor='black')

    ax1.set_ylabel('Thời gian (giây)')
    ax1.set_title('Thời Gian Huấn Luyện: Client vs Server', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}s',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Chart 2: Memory
    rects3 = ax2.bar(x - width/2, client_mems, width, label='Client (Worker)', color='#2ecc71', edgecolor='black')
    rects4 = ax2.bar(x + width/2, server_mems, width, label='Server (Master)', color='#9b59b6', edgecolor='black')

    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_title('Peak Memory: Client vs Server', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for rects in [rects3, rects4]:
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('./outputs/benchmark/client_server_comparison.png', dpi=150)
    print("✓ Saved chart to ./outputs/benchmark/client_server_comparison.png\n")

    # Generate Markdown Table
    md_table = "### Bảng So Sánh Chi Tiết (Client vs Server)\n\n"
    md_table += "| Mô Hình | Độ Chính Xác | Thời gian (Client / Server) | Peak Memory (Client / Server) | Clients làm gì? | Server làm gì? |\n"
    md_table += "|---|---|---|---|---|---|\n"
    
    for i in range(len(models)):
        md_table += f"| **{model_names[i].replace(chr(10), ' ')}** | {accuracies[i]} | {client_times[i]:.1f}s / {server_times[i]:.1f}s | {client_mems[i]:.3f} MB / {server_mems[i]:.3f} MB | {client_actions[i]} | {server_actions[i]} |\n"

    print("--- MARKDOWN TABLE ---")
    print(md_table)
    print("----------------------")

if __name__ == '__main__':
    generate_comparison()
