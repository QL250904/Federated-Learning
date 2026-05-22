# ðŸ“– HÆ¯á»šNG DáºªN Sá»¬ Dá»¤NG: BENCHMARK 5 MÃ” HÃŒNH FEDERATED LEARNING

Dá»± Ã¡n nÃ y cung cáº¥p mÃ£ nguá»“n Ä‘á»ƒ cháº¡y vÃ  so sÃ¡nh 5 mÃ´ hÃ¬nh phÃ¢n bá»• dá»¯ liá»‡u/mÃ´ hÃ¬nh:
1. Táº­p Trung HoÃ n ToÃ n (Fully Centralized)
2. Táº­p Trung CÃ³ Tiá»n Xá»­ LÃ½ (Semi-Centralized)
3. Song Song Dá»¯ Liá»‡u (Data Parallelism)
4. Song Song MÃ´ HÃ¬nh (Model Parallelism)
5. Káº¿t há»£p (Hybrid)

DÆ°á»›i Ä‘Ã¢y lÃ  2 hÆ°á»›ng dáº«n chi tiáº¿t Ä‘á»ƒ báº¡n cÃ³ thá»ƒ (1) cháº¡y mÃ´ phá»ng trÃªn 1 mÃ¡y tÃ­nh Ä‘á»ƒ ra bÃ¡o cÃ¡o, vÃ  (2) cháº¡y thá»±c táº¿ trÃªn 3 mÃ¡y tÃ­nh qua máº¡ng LAN.

---

## ðŸ’» PHáº¦N 1: HÆ¯á»šNG DáºªN CHáº Y MÃ” PHá»ŽNG TRÃŠN 1 MÃY TÃNH
*Pháº§n nÃ y dÃ¹ng Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ nhanh, so sÃ¡nh hiá»‡u nÄƒng cá»§a 5 mÃ´ hÃ¬nh. Cháº¡y tá»«ng mÃ´ hÃ¬nh riÃªng láº» vÃ  tá»± Ä‘á»™ng xuáº¥t ra cÃ¡c biá»ƒu Ä‘á»“/bÃ¡o cÃ¡o.*

### BÆ°á»›c 1: KÃ­ch hoáº¡t mÃ´i trÆ°á»ng áº£o (Virtual Environment)
Má»Ÿ terminal/PowerShell vÃ  cháº¡y:
```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.\fl_env\Scripts\activate
```

### BÆ°á»›c 2: Cháº¡y Tá»«ng MÃ´ HÃ¬nh (Benchmark) â€” Cháº¡y láº§n lÆ°á»£t tá»«ng cÃ¡i
> âš ï¸ **Cháº¡y tá»«ng lá»‡nh má»™t.** Äá»£i lá»‡nh trÆ°á»›c hoÃ n thÃ nh rá»“i má»›i cháº¡y lá»‡nh tiáº¿p theo.
> Má»—i mÃ´ hÃ¬nh tá»± lÆ°u káº¿t quáº£ riÃªng vÃ o thÆ° má»¥c `outputs/benchmark/`.

**[MÃ´ hÃ¬nh 1/5] Táº­p Trung HoÃ n ToÃ n (Fully Centralized):**
```bash
python bench_1_fully_centralized.py
```
â†’ Káº¿t quáº£ lÆ°u: `outputs/benchmark/fully_centralized_results.pkl`

**[MÃ´ hÃ¬nh 2/5] Táº­p Trung CÃ³ Tiá»n Xá»­ LÃ½ (Semi-Centralized):**
```bash
python bench_2_semi_centralized.py
```
â†’ Káº¿t quáº£ lÆ°u: `outputs/benchmark/semi_centralized_results.pkl`

**[MÃ´ hÃ¬nh 3/5] Song Song Dá»¯ Liá»‡u (Data Parallelism):**
```bash
python bench_3_data_parallel.py
```
â†’ Káº¿t quáº£ lÆ°u: `outputs/benchmark/data_parallel_results.pkl`

**[MÃ´ hÃ¬nh 4/5] Song Song MÃ´ HÃ¬nh (Model Parallelism):**
```bash
python bench_4_model_parallel.py
```
â†’ Káº¿t quáº£ lÆ°u: `outputs/benchmark/model_parallel_results.pkl`

**[MÃ´ hÃ¬nh 5/5] Káº¿t há»£p Hybrid (Data + Model Parallelism):**
```bash
python bench_5_hybrid.py
```
â†’ Káº¿t quáº£ lÆ°u: `outputs/benchmark/hybrid_parallel_results.pkl`

### BÆ°á»›c 3: Äá»c BÃ¡o CÃ¡o & Váº½ Biá»ƒu Äá»“ So SÃ¡nh
Sau khi cháº¡y xong **táº¥t cáº£ 5 mÃ´ hÃ¬nh** á»Ÿ bÆ°á»›c 2, cháº¡y lá»‡nh nÃ y Ä‘á»ƒ phÃ¢n tÃ­ch káº¿t quáº£ vÃ  táº¡o biá»ƒu Ä‘á»“:
```bash
python compare_results.py
```
**Káº¿t quáº£ báº¡n sáº½ nháº­n Ä‘Æ°á»£c trong `outputs/benchmark/`:**
1. `overall_comparison.png` â€” Báº£ng sá»‘ liá»‡u chi tiáº¿t + biá»ƒu Ä‘á> âš ï¸ **ChÃº Ã½ Quan Trá»ng:**
> 1. **MÃ´i trÆ°á»ng & ThÆ° viá»‡n:** Copy thÆ° má»¥c dá»± Ã¡n nÃ y sang cáº£ 3 mÃ¡y tÃ­nh (hoáº·c cháº¡y thá»­ nghiá»‡m local trÃªn 1 mÃ¡y báº±ng cÃ¡ch má»Ÿ nhiá»u Terminal). Äáº£m báº£o Ä‘Ã£ cÃ i Ä‘áº·t Ä‘áº§y Ä‘á»§ thÆ° viá»‡n: `pip install torch torchvision numpy scikit-learn psutil matplotlib`.
> 2. **KÃ­ch hoáº¡t mÃ´i trÆ°á»ng áº£o:** TrÃªn má»—i terminal (cáº£ Server vÃ  Client), kÃ­ch hoáº¡t mÃ´i trÆ°á»ng áº£o báº±ng: `.\fl_env\Scripts\activate`.
> 3. **LÆ°u Ã½ MÃ£ hÃ³a (Windows UTF-8):** TrÃªn Windows, Ä‘á»ƒ trÃ¡nh lá»—i Unicode tiáº¿ng Viá»‡t khi in log, luÃ´n thÃªm cá» `-X utf8` khi gá»i python. VÃ­ dá»¥: `python -X utf8 script.py ...`
> 4. **IP & Cá»•ng káº¿t ná»‘i:** Láº¥y IP cá»§a mÃ¡y Server (dÃ¹ng lá»‡nh `ipconfig` trÃªn Windows, vÃ­ dá»¥: `192.168.1.100`). Náº¿u cháº¡y thá»­ nghiá»‡m trÃªn cÃ¹ng 1 mÃ¡y tÃ­nh, hÃ£y thay `192.168.1.100` báº±ng `127.0.0.1`.
> 5. **Thá»© tá»± cháº¡y:** **LuÃ´n khá»Ÿi Ä‘á»™ng Server trÆ°á»›c**, sau Ä‘Ã³ má»›i má»Ÿ cÃ¡c Client Ä‘á»ƒ káº¿t ná»‘i.

---

### ðŸ’» HÆ¯á»šNG DáºªN CHáº Y CHI TIáº¾T Tá»ªNG MÃ” HÃŒNH TRÃŠN Máº NG THá»°C

DÆ°á»›i Ä‘Ã¢y lÃ  cÃ¡c lá»‡nh cháº¡y cho tá»«ng mÃ´ hÃ¬nh cá»¥ thá»ƒ. Báº¡n cÃ³ thá»ƒ thay Ä‘á»•i `--epochs` (máº·c Ä‘á»‹nh lÃ  5) hoáº·c cÃ¡c thÃ´ng sá»‘ khÃ¡c. (VÃ­ dá»¥ cháº¡y dÆ°á»›i Ä‘Ã¢y máº·c Ä‘á»‹nh vá»›i sá»‘ lÆ°á»£ng Client lÃ  2, IP Server giáº£ Ä‘á»‹nh lÃ  `192.168.1.100` hoáº·c `127.0.0.1` náº¿u cháº¡y trÃªn cÃ¹ng 1 mÃ¡y).

#### [MÃ´ hÃ¬nh 1] Táº¬P TRUNG HOÃ€N TOÃ€N (Fully Centralized)
*Client gá»­i trá»±c tiáº¿p toÃ n bá»™ dá»¯ liá»‡u thÃ´ (raw images) lÃªn Server.*
* **MÃ¡y Server:**
  ```bash
  python -X utf8 real_1_fully_centralized_server.py --port 5001 --num_clients 2 --epochs 2
  ```
* **MÃ¡y Client 1:**
  ```bash
  python -X utf8 real_1_fully_centralized_client.py --server_ip 172.20.2.30 --port 5001 --client_id 0 --num_clients 2
  ```
* **MÃ¡y Client 2:**
  ```bash
  python -X utf8 real_1_fully_centralized_client.py --server_ip 172.20.2.30 --port 5001 --client_id 1 --num_clients 2
  ```
â†’ Káº¿t quáº£ lÆ°u: `outputs/benchmark/real_fully_centralized_results.json`

#### [MÃ´ hÃ¬nh 2] Táº¬P TRUNG CÃ“ TIá»€N Xá»¬ LÃ (Semi-Centralized)
*Client dÃ¹ng thuáº­t toÃ¡n PCA nÃ©n giáº£m chiá»u dá»¯ liá»‡u trÆ°á»›c khi gá»­i lÃªn Server Ä‘á»ƒ tiáº¿t kiá»‡m bÄƒng thÃ´ng.*
* **MÃ¡y Server:**
  ```bash
  python -X utf8 real_2_semi_centralized_server.py --port 5002 --num_clients 2 --epochs 2
  ```
* **MÃ¡y Client 1:**
  ```bash
  python -X utf8 real_2_semi_centralized_client.py --server_ip 172.20.2.30 --port 5002 --client_id 0 --num_clients 2
  ```
* **MÃ¡y Client 2:**
  ```bash
  python -X utf8 real_2_semi_centralized_client.py --server_ip 172.20.2.30 --port 5002 --client_id 1 --num_clients 2
  ```
â†’ Káº¿t quáº£ lÆ°u: `outputs/benchmark/real_semi_centralized_results.json`

#### [MÃ´ hÃ¬nh 3] SONG SONG Dá»® LIá»†U (Data Parallelism)
*MÃ´ hÃ¬nh lÃµi cá»§a Federated Learning: PhÃ¢n chia táº­p dá»¯ liá»‡u cho cÃ¡c Client, cÃ¡c Client tá»± huáº¥n luyá»‡n cá»¥c bá»™ rá»“i chá»‰ gá»­i cÃ¡c tham sá»‘ trá»ng sá»‘ (weights) cá»§a mÃ´ hÃ¬nh vá» Server Ä‘á»ƒ tá»•ng há»£p (FedAvg).*
* **MÃ¡y Server:**
  ```bash
  python -X utf8 real_3_data_parallel_server.py --port 5003 --num_clients 2 --epochs 2
  ```
* **MÃ¡y Client 1:**
  ```bash
  python -X utf8 real_3_data_parallel_client.py --server_ip 172.20.2.30 --port 5003 --client_id 0 --num_clients 2
  ```
* **MÃ¡y Client 2:**
  ```bash
  python -X utf8 real_3_data_parallel_client.py --server_ip 172.20.2.30 --port 5003 --client_id 1 --num_clients 2
  ```
â†’ Káº¿t quáº£ lÆ°u: `outputs/benchmark/real_data_parallel_results.json`

#### [MÃ´ hÃ¬nh 4] SONG SONG MÃ” HÃŒNH (Model Parallelism)
*Cáº¯t dá»c máº¡ng Neural Network: Client giá»¯ cÃ¡c lá»›p Ä‘áº§u (Feature Extractor - Conv layers), Server giá»¯ cÃ¡c lá»›p cuá»‘i (Classifier - FC layers). Client gá»­i activations lÃªn Server, nháº­n gradients vá» Ä‘á»ƒ cáº­p nháº­t.*
*(MÃ´ hÃ¬nh nÃ y cháº¡y vá»›i 1 Server vÃ  1 Client)*
* **MÃ¡y Server:**
  ```bash
  python -X utf8 real_4_model_parallel_server.py --port 5004 --epochs 2
  ```
* **MÃ¡y Client 1:**
  ```bash
  python -X utf8 real_4_model_parallel_client.py --server_ip 172.20.2.30 --port 5004 --epochs 2
  ```
â†’ Káº¿t quáº£ lÆ°u: `outputs/benchmark/real_model_parallel_results.json`

#### [MÃ´ hÃ¬nh 5] Káº¾T Há»¢P (Hybrid Parallelism)
*Káº¿t há»£p cáº£ Song song Dá»¯ liá»‡u (nhiá»u clients) vÃ  Song song MÃ´ hÃ¬nh (cáº¯t Ä‘Ã´i neural network cho má»—i group, AllReduce sync chÃ©o).*
* **MÃ¡y Server:**
  ```bash
  python -X utf8 real_5_hybrid_server.py --port 5005 --num_clients 2 --epochs 2
  ```
* **MÃ¡y Client 1:**
  ```bash
  python -X utf8 real_5_hybrid_client.py --server_ip 172.20.2.30 --port 5005 --client_id 0 --num_clients 2 --epochs 2
  ```
* **MÃ¡y Client 2:**
  ```bash
  python -X utf8 real_5_hybrid_client.py --server_ip 172.20.2.30 --port 5005 --client_id 1 --num_clients 2 --epochs 2
  ```
â†’ Káº¿t quáº£ lÆ°u: `outputs/benchmark/real_hybrid_results.json`

---

### ðŸ“Š BÆ¯á»šC 3: PHÃ‚N TÃCH VÃ€ Váº¼ BIá»‚U Äá»’ SO SÃNH Máº NG THá»°C

Sau khi cháº¡y xong cÃ¡c mÃ´ hÃ¬nh trÃªn máº¡ng thá»±c táº¿ (cÃ¡c tá»‡p `.json` káº¿t quáº£ Ä‘Ã£ Ä‘Æ°á»£c lÆ°u Ä‘áº§y Ä‘á»§ vÃ o thÆ° má»¥c `outputs/benchmark/`), hÃ£y cháº¡y lá»‡nh nÃ y táº¡i mÃ¡y Server Ä‘á»ƒ tá»•ng há»£p:
```bash
python compare_results.py --mode real
```
Lá»‡nh nÃ y sáº½ quÃ©t cÃ¡c tá»‡p `real_*_results.json`, xuáº¥t báº£ng so sÃ¡nh chi tiáº¿t ngay trÃªn mÃ n hÃ¬nh terminal vÃ  lÆ°u cÃ¡c biá»ƒu Ä‘á»“ phÃ¢n tÃ­ch trá»±c quan:
1. **`overall_comparison.png`** â€” So sÃ¡nh trá»±c quan vÃ  chi tiáº¿t táº¥t cáº£ cÃ¡c chá»‰ sá»‘ (Accuracy, Total Time, Memory, BÄƒng thÃ´ng truyá»n nháº­n, CPU) vá»›i cÃ¡c thÃ nh pháº§n cá»¥ thá»ƒ cá»§a Server/Client/Worker.
2. **`accuracy_over_epochs.png`** â€” ÄÆ°á»ng cong biáº¿n thiÃªn cá»§a Ä‘á»™ chÃ­nh xÃ¡c thá»±c táº¿ qua tá»«ng Epoch.
3. **`centralized_vs_distributed.png`** â€” So sÃ¡nh hiá»‡u quáº£ tá»•ng quan giá»¯a NhÃ³m Táº­p Trung (Centralized) vÃ  NhÃ³m PhÃ¢n TÃ¡n (Distributed) thá»±c táº¿.

### ðŸ“Š BÆ°á»›c 3: Váº½ Biá»ƒu Äá»“ So SÃ¡nh Cho Máº¡ng Thá»±c
Sau khi cháº¡y xong cÃ¡c mÃ´ hÃ¬nh trÃªn máº¡ng thá»±c táº¿ (mÃ¡y Server sáº½ tá»± Ä‘á»™ng lÆ°u káº¿t quáº£ `.json` tÆ°Æ¡ng á»©ng vÃ o thÆ° má»¥c `outputs/benchmark/`), hÃ£y cháº¡y lá»‡nh nÃ y táº¡i mÃ¡y Server Ä‘á»ƒ váº½ biá»ƒu Ä‘á»“ vÃ  phÃ¢n tÃ­ch:
```bash
python compare_results.py --mode real
```
Lá»‡nh nÃ y sáº½ tá»± Ä‘á»™ng táº£i cÃ¡c tá»‡p káº¿t quáº£ tá»« máº¡ng thá»±c táº¿ (`real_*_results.json`) vÃ  xuáº¥t ra cÃ¡c biá»ƒu Ä‘á»“ chi tiáº¿t:
1. `overall_comparison.png` â€” So sÃ¡nh thá»i gian (Train, Preprocess, AllReduce Sync, ...), Bandwidth, Memory, CPU cá»§a 5 mÃ´ hÃ¬nh thá»±c táº¿.
2. `accuracy_over_epochs.png` â€” So sÃ¡nh Ä‘á»™ chÃ­nh xÃ¡c cá»§a cÃ¡c mÃ´ hÃ¬nh qua tá»«ng epoch thá»±c táº¿.
3. `centralized_vs_distributed.png` â€” So sÃ¡nh giá»¯a nhÃ³m Táº­p Trung vÃ  PhÃ¢n TÃ¡n trong máº¡ng thá»±c táº¿.


---

### ðŸš¨ HÆ¯á»šNG DáºªN Káº¾T Ná»I TRÃŠN Máº NG THá»°C (DÃ€NH CHO IP 172.20.2.30)

**QUAN TRá»ŒNG Vá»€ TÆ¯á»œNG Lá»¬A (FIREWALL):**
Lá»—i vÄƒng ngay láº­p tá»©c khi káº¿t ná»‘i vá»›i mÃ¡y khÃ¡c háº§u háº¿t lÃ  do Windows Firewall trÃªn mÃ¡y Server Ä‘ang cháº·n. Äá»ƒ sá»­a lá»—i:
1. TrÃªn mÃ¡y Server, má»Ÿ PowerShell (Run as Administrator).
2. Cháº¡y lá»‡nh sau Ä‘á»ƒ má»Ÿ cá»•ng (hoáº·c táº¡m táº¯t Firewall):
   ``powershell
   New-NetFirewallRule -DisplayName "Open FL Ports" -Direction Inbound -LocalPort 5001-5005 -Protocol TCP -Action Allow
   ``

#### CÁC LỆNH CHẠY CHÍNH XÁC CHO 5 MÔ HÌNH (IP: 172.20.2.30)
Bạn phải chạy lệnh ở máy Server trước, đợi thông báo lắng nghe rồi mới qua máy Client chạy.

---
**[Mô hình 1] Tập trung Hoàn toàn (Fully Centralized)**
* Máy Server:
``bash
python -X utf8 real_1_fully_centralized_server.py --port 5001 --num_clients 2 --epochs 2
``
* Máy Client 1:
``bash
python -X utf8 real_1_fully_centralized_client.py --server_ip 172.20.2.30 --port 5001 --client_id 0 --num_clients 2
``
* Máy Client 2:
``bash
python -X utf8 real_1_fully_centralized_client.py --server_ip 172.20.2.30 --port 5001 --client_id 1 --num_clients 2
``

---
**[Mô hình 2] Tập trung Cải tiến (Semi-Centralized)**
* Máy Server:
``bash
python -X utf8 real_2_semi_centralized_server.py --port 5002 --num_clients 2 --epochs 2
``
* Máy Client 1:
``bash
python -X utf8 real_2_semi_centralized_client.py --server_ip 172.20.2.30 --port 5002 --client_id 0 --num_clients 2
``
* Máy Client 2:
``bash
python -X utf8 real_2_semi_centralized_client.py --server_ip 172.20.2.30 --port 5002 --client_id 1 --num_clients 2
``

---
**[Mô hình 3] Song song Dữ liệu (Data Parallelism)**
* Máy Server:
``bash
python -X utf8 real_3_data_parallel_server.py --port 5003 --num_clients 2 --epochs 2
``
* Máy Client 1:
``bash
python -X utf8 real_3_data_parallel_client.py --server_ip 172.20.2.30 --port 5003 --client_id 0 --num_clients 2
``
* Máy Client 2:
``bash
python -X utf8 real_3_data_parallel_client.py --server_ip 172.20.2.30 --port 5003 --client_id 1 --num_clients 2
``

---
**[Mô hình 4] Song song Mô hình (Model Parallelism)** (Mô hình này chỉ có 1 Client)
* Máy Server:
``bash
python -X utf8 real_4_model_parallel_server.py --port 5004 --epochs 2
``
* Máy Client 1:
``bash
python -X utf8 real_4_model_parallel_client.py --server_ip 172.20.2.30 --port 5004 --epochs 2
``

---
**[Mô hình 5] Kết hợp (Hybrid Parallelism)**
* Máy Server:
``bash
python -X utf8 real_5_hybrid_server.py --port 5005 --num_clients 2 --epochs 2
``
* Máy Client 1:
``bash
python -X utf8 real_5_hybrid_client.py --server_ip 172.20.2.30 --port 5005 --client_id 0 --num_clients 2 --epochs 2
``
* Máy Client 2:
``bash
python -X utf8 real_5_hybrid_client.py --server_ip 172.20.2.30 --port 5005 --client_id 1 --num_clients 2 --epochs 2
``
