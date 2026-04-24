import pickle

try:
    with open('outputs/benchmark/forest_results.pkl', 'rb') as f:
        rf = pickle.load(f)
    print(f"\n[1] RANDOM FOREST (Tập trung dữ liệu truyền thống)")
    print(f"   - Phương pháp: Thu thập toàn bộ hình ảnh về Server để học")
    print(f"   - Độ chính xác: {rf['accuracy']*100:.2f}%")
    print(f"   - Tổng thời gian thi hành: {rf['total_time']:.2f} giây")
except Exception as e:
    print('Lỗi:', e)

try:
    with open('outputs/benchmark/flower_simulation_results.pkl', 'rb') as f:
        fl_c = pickle.load(f)
    print(f"\n[2] FL CENTRALIZED (Mạng Hình Sao có máy chủ - Chạy rập khuôn trên Máy ảo Flower Virtual Engine)")
    print(f"   - Phương pháp: Các thiết bị tự học rồi đưa kinh nghiệm lên Máy chủ tạo Model tổng")
    print(f"   - Độ chính xác sau 8 Vòng lặp: {fl_c['accuracy']*100:.2f}%")
    print(f"   - Tổng thời gian thi hành 8 vòng: {fl_c['total_time']:.2f} giây (~ {fl_c['total_time']/60:.2f} phút)")
except Exception as e:
    print('Lỗi:', e)

try:
    with open('outputs/benchmark/fl_decentralized_results.pkl', 'rb') as f:
        fl_d = pickle.load(f)
    print(f"\n[3] FL DECENTRALIZED (Mạng Ngang Hàng Không Máy Chủ - Decentralized Ring)")
    print(f"   - Phương pháp: Các thiết bị tuần tự truyền kinh nghiệm thành vòng tròn cho nhau")
    print(f"   - Độ chính xác sau 5 Vòng lặp: {fl_d['rounds'][-1]['accuracy']*100:.2f}%")
    print(f"   - Tổng thời gian thi hành (truyền bắc cầu): {fl_d['total_time']:.2f} giây (nếu chạy 8 vòng sẽ lâu hơn bản Centralized kể trên gấp 1.5 lần)")
    print(f"   - Tổng số lượt truyền nhảy tham số (Hops): {sum([r['communication_hops'] for r in fl_d['rounds']])} bước nhảy")
except Exception as e:
    print('Lỗi:', e)
