# 📖 Hướng Dẫn Sử Dụng Toàn Diện: Benchmark 5 Mô Hình Học Máy

Dự án này đánh giá 5 kiến trúc phân bổ dữ liệu/mô hình khác nhau (2 mô hình tập trung, 3 mô hình phân tán). Bạn có thể chạy mô phỏng tất cả trên 1 máy tính để lấy báo cáo, hoặc triển khai thực tế trên 3 máy tính (1 Server, 2 Clients).

---

## PHẦN 1: CHẠY MÔ PHỎNG & SO SÁNH TRÊN 1 MÁY

Phần này dùng để đánh giá và xuất biểu đồ so sánh: Accuracy, Training Time, Bandwidth, Memory.

### Bước 1: Mở môi trường ảo (Virtual Environment)
Cần đảm bảo bạn đang dùng môi trường có cài sẵn PyTorch và thư viện cần thiết:
```powershell
# Trên Windows PowerShell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.\fl_env\Scripts\activate
```

### Bước 2: Chạy Benchmark cho toàn bộ 5 mô hình
Script này sẽ chạy lần lượt 5 file (`bench_1_fully_centralized.py` -> `bench_5_hybrid.py`).
```bash
python benchmark_runner.py
```
*(Quá trình này tốn khoảng 10-15 phút. Kết quả lưu vào `outputs/benchmark/`)*

### Bước 3: Đọc báo cáo và Vẽ biểu đồ
Sau khi chạy xong, lệnh này sẽ phân tích file kết quả, hiển thị bảng text ra màn hình và lưu 3 ảnh biểu đồ vào thư mục `outputs/benchmark/`:
```bash
python compare_results.py
```
**Các biểu đồ thu được:**
1. `overall_comparison.png` (Biểu đồ cột so sánh tổng hợp)
2. `accuracy_over_epochs.png` (Đường cong độ chính xác qua 5 epochs)
3. `centralized_vs_distributed.png` (So sánh trung bình nhóm Tập Trung vs Phân Tán)

---

## PHẦN 2: CHẠY QUA MẠNG LAN THỰC TẾ (1 Server + 2 Clients)

> ⚠️ **Chú ý:**
> - Máy Server phải mở Firewall cho các cổng mạng tương ứng (5001, 5003, 5004).
> - Giả định địa chỉ IP của Máy Server trong mạng LAN là `192.168.1.100`. (Bạn cần đổi sang IP thật khi chạy).

### [Nhóm 1] TẬP TRUNG HOÀN TOÀN (Fully Centralized)
*Gửi dữ liệu thô (Raw Image) từ Client lên Server.*
* **Máy Server:** `python real_1_2_server.py --port 5001 --num_clients 2 --mode fully`
* **Máy Client 1:** `python real_1_2_client.py --server_ip 192.168.1.100 --port 5001 --client_id 0 --num_clients 2 --mode fully`
* **Máy Client 2:** `python real_1_2_client.py --server_ip 192.168.1.100 --port 5001 --client_id 1 --num_clients 2 --mode fully`

### [Nhóm 2] TẬP TRUNG CÓ TIỀN XỬ LÝ (Semi-Centralized)
*Client nén dữ liệu bằng PCA rồi mới gửi.*
* **Máy Server:** `python real_1_2_server.py --port 5001 --num_clients 2 --mode semi`
* **Máy Client 1:** `python real_1_2_client.py --server_ip 192.168.1.100 --port 5001 --client_id 0 --num_clients 2 --mode semi`
* **Máy Client 2:** `python real_1_2_client.py --server_ip 192.168.1.100 --port 5001 --client_id 1 --num_clients 2 --mode semi`

### [Nhóm 3] SONG SONG DỮ LIỆU (Data Parallelism)
*Client giữ dữ liệu cục bộ, chỉ gửi Model Weights (Federated Learning).*
* **Máy Server:** `python real_3_server.py --port 5003 --num_clients 2 --epochs 5`
* **Máy Client 1:** `python real_3_client.py --server_ip 192.168.1.100 --port 5003 --client_id 0 --num_clients 2`
* **Máy Client 2:** `python real_3_client.py --server_ip 192.168.1.100 --port 5003 --client_id 1 --num_clients 2`

### [Nhóm 4] SONG SONG MÔ HÌNH (Model Parallelism)
*Chia cắt mạng Neural: Client giữ Conv Layers, Server giữ FC Layers. Liên tục truyền Activations.* (Chạy với 1 Client).
* **Máy Server:** `python real_4_5_server.py --port 5004 --num_clients 1`
* **Máy Client 1:** `python real_4_5_client.py --server_ip 192.168.1.100 --port 5004 --client_id 0 --num_clients 1`

### [Nhóm 5] KẾT HỢP (Hybrid)
*Kết hợp cả Song song Dữ liệu và Song song Mô hình.*
* **Máy Server:** `python real_4_5_server.py --port 5004 --num_clients 2`
* **Máy Client 1:** `python real_4_5_client.py --server_ip 192.168.1.100 --port 5004 --client_id 0 --num_clients 2`
* **Máy Client 2:** `python real_4_5_client.py --server_ip 192.168.1.100 --port 5004 --client_id 1 --num_clients 2`

---
*Lưu ý: Ở các script mạng thực (nhất là Model Parallelism), số lượng dữ liệu đã được thu nhỏ lại (ví dụ 1000 mẫu/client) để tránh việc đường truyền LAN bị đứt kết nối (timeout) khi luồng Gradient lưu thông quá nhiều.*
