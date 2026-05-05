# 📖 HƯỚNG DẪN SỬ DỤNG: BENCHMARK 5 MÔ HÌNH FEDERATED LEARNING

Dự án này cung cấp mã nguồn để chạy và so sánh 5 mô hình phân bổ dữ liệu/mô hình:
1. Tập Trung Hoàn Toàn (Fully Centralized)
2. Tập Trung Có Tiền Xử Lý (Semi-Centralized)
3. Song Song Dữ Liệu (Data Parallelism)
4. Song Song Mô Hình (Model Parallelism)
5. Kết hợp (Hybrid)

Dưới đây là 2 hướng dẫn chi tiết để bạn có thể (1) chạy mô phỏng trên 1 máy tính để ra báo cáo, và (2) chạy thực tế trên 3 máy tính qua mạng LAN.

---

## 💻 PHẦN 1: HƯỚNG DẪN CHẠY MÔ PHỎNG TRÊN 1 MÁY TÍNH
*Phần này dùng để đánh giá nhanh, so sánh hiệu năng của 5 mô hình và tự động xuất ra các biểu đồ/báo cáo.*

### Bước 1: Kích hoạt môi trường ảo (Virtual Environment)
Mở terminal/PowerShell và chạy:
```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.\fl_env\Scripts\activate
```

### Bước 2: Chạy File Mô Phỏng (Benchmark)
Script này sẽ chạy lần lượt 5 mô hình bằng cách chia luồng ngay trên máy tính của bạn:
```bash
python benchmark_runner.py
```
*(Quá trình này tốn khoảng 10 phút. Kết quả được lưu vào thư mục `outputs/benchmark/`)*

### Bước 3: Đọc Báo Cáo & Vẽ Biểu Đồ So Sánh
Sau khi bước 2 chạy xong, lệnh này sẽ phân tích kết quả và tạo ảnh biểu đồ:
```bash
python compare_results.py
```
**Kết quả bạn sẽ nhận được trong `outputs/benchmark/`:**
1. `overall_comparison.png` (Cột so sánh Accuracy, Time, Bandwidth, Memory)
2. `accuracy_over_epochs.png` (Đường cong accuracy)
3. `centralized_vs_distributed.png` (So sánh trung bình nhóm Tập Trung vs Phân Tán)
4. Bảng số liệu dạng text ngay trên màn hình terminal.

---

## 🌐 PHẦN 2: HƯỚNG DẪN CHẠY TRÊN MẠNG THỰC (3 MÁY TÍNH)
*Phần này triển khai code thực tế qua kết nối TCP Socket giữa 1 máy Server và 2 máy Client.*

> ⚠️ **Chú ý Quan Trọng:**
> 1. Copy thư mục dự án này sang cả 3 máy tính.
> 2. Đảm bảo 3 máy tính cùng kết nối chung một mạng LAN/WiFi.
> 3. Lấy IP của máy Server (dùng lệnh `ipconfig`). Giả sử IP máy Server là `192.168.1.100`.
> 4. Cài đặt các thư viện trên tất cả máy: `pip install torch torchvision numpy scikit-learn psutil`
> 5. **BẮT BUỘC:** Mở Terminal/PowerShell và kích hoạt môi trường ảo (chạy lệnh `.\fl_env\Scripts\activate`) trên cả 3 máy tính TRƯỚC khi gọi lệnh python.
> 6. **Luôn chạy script ở máy Server TRƯỚC, sau đó mới chạy các máy Client.**

### [Mô hình 1] TẬP TRUNG HOÀN TOÀN (Fully Centralized)
*Client gửi dữ liệu ảnh gốc trực tiếp lên Server.*
* **Máy Server:**
  `python real_1_fully_centralized_server.py --port 5001 --num_clients 2`
* **Máy Client 1:**
  `python real_1_fully_centralized_client.py --server_ip 192.168.1.100 --port 5001 --client_id 0 --num_clients 2`
* **Máy Client 2:**
  `python real_1_fully_centralized_client.py --server_ip 192.168.1.100 --port 5001 --client_id 1 --num_clients 2`

### [Mô hình 2] TẬP TRUNG CÓ TIỀN XỬ LÝ (Semi-Centralized)
*Client dùng PCA để nén dữ liệu rồi mới gửi lên Server.*
* **Máy Server:**
  `python real_2_semi_centralized_server.py --port 5002 --num_clients 2`
* **Máy Client 1:**
  `python real_2_semi_centralized_client.py --server_ip 192.168.1.100 --port 5002 --client_id 0 --num_clients 2`
* **Máy Client 2:**
  `python real_2_semi_centralized_client.py --server_ip 192.168.1.100 --port 5002 --client_id 1 --num_clients 2`

### [Mô hình 3] SONG SONG DỮ LIỆU (Data Parallelism)
*Đây là cốt lõi của Federated Learning: Client train cục bộ, chỉ gửi Model Weights (tham số) về Server để trung bình hóa (FedAvg).*
* **Máy Server:**
  `python real_3_data_parallel_server.py --port 5003 --num_clients 2 --epochs 5`
* **Máy Client 1:**
  `python real_3_data_parallel_client.py --server_ip 192.168.1.100 --port 5003 --client_id 0 --num_clients 2`
* **Máy Client 2:**
  `python real_3_data_parallel_client.py --server_ip 192.168.1.100 --port 5003 --client_id 1 --num_clients 2`

### [Mô hình 4] SONG SONG MÔ HÌNH (Model Parallelism)
*Chia cắt mạng Neural Network: Client giữ các lớp đầu (Conv Layers), Server giữ các lớp cuối (FC Layers). Data dạng activations truyền liên tục.*
*(Chạy với 1 Server và 1 Client)*
* **Máy Server:**
  `python real_4_model_parallel_server.py --port 5004`
* **Máy Client 1:**
  `python real_4_model_parallel_client.py --server_ip 192.168.1.100 --port 5004`

### [Mô hình 5] KẾT HỢP (Hybrid)
*Kết hợp cả Song song Dữ liệu (nhiều clients) và Song song Mô hình (cắt đôi neural network).*
* **Máy Server:**
  `python real_5_hybrid_server.py --port 5005 --num_clients 2 --epochs 5`
* **Máy Client 1:**
  `python real_5_hybrid_client.py --server_ip 192.168.1.100 --port 5005 --client_id 0 --num_clients 2`
* **Máy Client 2:**
  `python real_5_hybrid_client.py --server_ip 192.168.1.100 --port 5005 --client_id 1 --num_clients 2`
