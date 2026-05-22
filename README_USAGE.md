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
*Phần này dùng để đánh giá nhanh, so sánh hiệu năng của 5 mô hình. Chạy từng mô hình riêng lẻ và tự động xuất ra các biểu đồ/báo cáo.*

### Bước 1: Kích hoạt môi trường ảo (Virtual Environment)
Mở terminal/PowerShell và chạy:
```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.\fl_env\Scripts\activate
```

### Bước 2: Chạy Từng Mô Hình (Benchmark) — Chạy lần lượt từng cái
> ⚠️ **Chạy từng lệnh một.** Đợi lệnh trước hoàn thành rồi mới chạy lệnh tiếp theo.
> Mỗi mô hình tự lưu kết quả riêng vào thư mục `outputs/benchmark/`.

**[Mô hình 1/5] Tập Trung Hoàn Toàn (Fully Centralized):**
```bash
python bench_1_fully_centralized.py
```
→ Kết quả lưu: `outputs/benchmark/fully_centralized_results.pkl`

**[Mô hình 2/5] Tập Trung Có Tiền Xử Lý (Semi-Centralized):**
```bash
python bench_2_semi_centralized.py
```
→ Kết quả lưu: `outputs/benchmark/semi_centralized_results.pkl`

**[Mô hình 3/5] Song Song Dữ Liệu (Data Parallelism):**
```bash
python bench_3_data_parallel.py
```
→ Kết quả lưu: `outputs/benchmark/data_parallel_results.pkl`

**[Mô hình 4/5] Song Song Mô Hình (Model Parallelism):**
```bash
python bench_4_model_parallel.py
```
→ Kết quả lưu: `outputs/benchmark/model_parallel_results.pkl`

**[Mô hình 5/5] Kết hợp Hybrid (Data + Model Parallelism):**
```bash
python bench_5_hybrid.py
```
→ Kết quả lưu: `outputs/benchmark/hybrid_parallel_results.pkl`

### Bước 3: Đọc Báo Cáo & Vẽ Biểu Đồ So Sánh
Sau khi chạy xong **tất cả 5 mô hình** ở bước 2, chạy lệnh này để phân tích kết quả và tạo biểu đồ:
```bash
python compare_results.py
```
**Kết quả bạn sẽ nhận được trong `outputs/benchmark/`:**
1. `overall_comparison.png` — Bảng số liệu chi tiết + biểu đ�> ⚠️ **Chú ý Quan Trọng:**
> 1. **Môi trường & Thư viện:** Copy thư mục dự án này sang cả 3 máy tính (hoặc chạy thử nghiệm local trên 1 máy bằng cách mở nhiều Terminal). Đảm bảo đã cài đặt đầy đủ thư viện: `pip install torch torchvision numpy scikit-learn psutil matplotlib`.
> 2. **Kích hoạt môi trường ảo:** Trên mỗi terminal (cả Server và Client), kích hoạt môi trường ảo bằng: `.\fl_env\Scripts\activate`.
> 3. **Lưu ý Mã hóa (Windows UTF-8):** Trên Windows, để tránh lỗi Unicode tiếng Việt khi in log, luôn thêm cờ `-X utf8` khi gọi python. Ví dụ: `python -X utf8 script.py ...`
> 4. **IP & Cổng kết nối:** Lấy IP của máy Server (dùng lệnh `ipconfig` trên Windows, ví dụ: `192.168.1.100`). Nếu chạy thử nghiệm trên cùng 1 máy tính, hãy thay `192.168.1.100` bằng `127.0.0.1`.
> 5. **Thứ tự chạy:** **Luôn khởi động Server trước**, sau đó mới mở các Client để kết nối.

---

### 💻 HƯỚNG DẪN CHẠY CHI TIẾT TỪNG MÔ HÌNH TRÊN MẠNG THỰC

Dưới đây là các lệnh chạy cho từng mô hình cụ thể. Bạn có thể thay đổi `--epochs` (mặc định là 5) hoặc các thông số khác. (Ví dụ chạy dưới đây mặc định với số lượng Client là 2, IP Server giả định là `192.168.1.100` hoặc `127.0.0.1` nếu chạy trên cùng 1 máy).

#### [Mô hình 1] TẬP TRUNG HOÀN TOÀN (Fully Centralized)
*Client gửi trực tiếp toàn bộ dữ liệu thô (raw images) lên Server.*
* **Máy Server:**
  ```bash
  python -X utf8 real_1_fully_centralized_server.py --port 5001 --num_clients 2 --epochs 2
  ```
* **Máy Client 1:**
  ```bash
  python -X utf8 real_1_fully_centralized_client.py --server_ip 127.0.0.1 --port 5001 --client_id 0 --num_clients 2
  ```
* **Máy Client 2:**
  ```bash
  python -X utf8 real_1_fully_centralized_client.py --server_ip 127.0.0.1 --port 5001 --client_id 1 --num_clients 2
  ```
→ Kết quả lưu: `outputs/benchmark/real_fully_centralized_results.json`

#### [Mô hình 2] TẬP TRUNG CÓ TIỀN XỬ LÝ (Semi-Centralized)
*Client dùng thuật toán PCA nén giảm chiều dữ liệu trước khi gửi lên Server để tiết kiệm băng thông.*
* **Máy Server:**
  ```bash
  python -X utf8 real_2_semi_centralized_server.py --port 5002 --num_clients 2 --epochs 2
  ```
* **Máy Client 1:**
  ```bash
  python -X utf8 real_2_semi_centralized_client.py --server_ip 127.0.0.1 --port 5002 --client_id 0 --num_clients 2
  ```
* **Máy Client 2:**
  ```bash
  python -X utf8 real_2_semi_centralized_client.py --server_ip 127.0.0.1 --port 5002 --client_id 1 --num_clients 2
  ```
→ Kết quả lưu: `outputs/benchmark/real_semi_centralized_results.json`

#### [Mô hình 3] SONG SONG DỮ LIỆU (Data Parallelism)
*Mô hình lõi của Federated Learning: Phân chia tập dữ liệu cho các Client, các Client tự huấn luyện cục bộ rồi chỉ gửi các tham số trọng số (weights) của mô hình về Server để tổng hợp (FedAvg).*
* **Máy Server:**
  ```bash
  python -X utf8 real_3_data_parallel_server.py --port 5003 --num_clients 2 --epochs 2
  ```
* **Máy Client 1:**
  ```bash
  python -X utf8 real_3_data_parallel_client.py --server_ip 127.0.0.1 --port 5003 --client_id 0 --num_clients 2
  ```
* **Máy Client 2:**
  ```bash
  python -X utf8 real_3_data_parallel_client.py --server_ip 127.0.0.1 --port 5003 --client_id 1 --num_clients 2
  ```
→ Kết quả lưu: `outputs/benchmark/real_data_parallel_results.json`

#### [Mô hình 4] SONG SONG MÔ HÌNH (Model Parallelism)
*Cắt dọc mạng Neural Network: Client giữ các lớp đầu (Feature Extractor - Conv layers), Server giữ các lớp cuối (Classifier - FC layers). Client gửi activations lên Server, nhận gradients về để cập nhật.*
*(Mô hình này chạy với 1 Server và 1 Client)*
* **Máy Server:**
  ```bash
  python -X utf8 real_4_model_parallel_server.py --port 5004 --epochs 2
  ```
* **Máy Client 1:**
  ```bash
  python -X utf8 real_4_model_parallel_client.py --server_ip 127.0.0.1 --port 5004 --epochs 2
  ```
→ Kết quả lưu: `outputs/benchmark/real_model_parallel_results.json`

#### [Mô hình 5] KẾT HỢP (Hybrid Parallelism)
*Kết hợp cả Song song Dữ liệu (nhiều clients) và Song song Mô hình (cắt đôi neural network cho mỗi group, AllReduce sync chéo).*
* **Máy Server:**
  ```bash
  python -X utf8 real_5_hybrid_server.py --port 5005 --num_clients 2 --epochs 2
  ```
* **Máy Client 1:**
  ```bash
  python -X utf8 real_5_hybrid_client.py --server_ip 127.0.0.1 --port 5005 --client_id 0 --num_clients 2 --epochs 2
  ```
* **Máy Client 2:**
  ```bash
  python -X utf8 real_5_hybrid_client.py --server_ip 127.0.0.1 --port 5005 --client_id 1 --num_clients 2 --epochs 2
  ```
→ Kết quả lưu: `outputs/benchmark/real_hybrid_results.json`

---

### 📊 BƯỚC 3: PHÂN TÍCH VÀ VẼ BIỂU ĐỒ SO SÁNH MẠNG THỰC

Sau khi chạy xong các mô hình trên mạng thực tế (các tệp `.json` kết quả đã được lưu đầy đủ vào thư mục `outputs/benchmark/`), hãy chạy lệnh này tại máy Server để tổng hợp:
```bash
python compare_results.py --mode real
```
Lệnh này sẽ quét các tệp `real_*_results.json`, xuất bảng so sánh chi tiết ngay trên màn hình terminal và lưu các biểu đồ phân tích trực quan:
1. **`overall_comparison.png`** — So sánh trực quan và chi tiết tất cả các chỉ số (Accuracy, Total Time, Memory, Băng thông truyền nhận, CPU) với các thành phần cụ thể của Server/Client/Worker.
2. **`accuracy_over_epochs.png`** — Đường cong biến thiên của độ chính xác thực tế qua từng Epoch.
3. **`centralized_vs_distributed.png`** — So sánh hiệu quả tổng quan giữa Nhóm Tập Trung (Centralized) và Nhóm Phân Tán (Distributed) thực tế.

### 📊 Bước 3: Vẽ Biểu Đồ So Sánh Cho Mạng Thực
Sau khi chạy xong các mô hình trên mạng thực tế (máy Server sẽ tự động lưu kết quả `.json` tương ứng vào thư mục `outputs/benchmark/`), hãy chạy lệnh này tại máy Server để vẽ biểu đồ và phân tích:
```bash
python compare_results.py --mode real
```
Lệnh này sẽ tự động tải các tệp kết quả từ mạng thực tế (`real_*_results.json`) và xuất ra các biểu đồ chi tiết:
1. `overall_comparison.png` — So sánh thời gian (Train, Preprocess, AllReduce Sync, ...), Bandwidth, Memory, CPU của 5 mô hình thực tế.
2. `accuracy_over_epochs.png` — So sánh độ chính xác của các mô hình qua từng epoch thực tế.
3. `centralized_vs_distributed.png` — So sánh giữa nhóm Tập Trung và Phân Tán trong mạng thực tế.

