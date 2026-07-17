# File: plot_results.py
import os, json
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
# Danh sách các mức tải (Rate) mà bạn đã chạy trong mô phỏng
RATES = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Danh sách thuật toán tương ứng
POLICIES = {
    "AllLocal":      {"color": "orange", "label": "All Local"},
    "AllEdge0":      {"color": "gray",   "label": "All Edge 0"},
    "TypeAware":     {"color": "blue",   "label": "Type-Aware Heuristic"},
    "EpsGreedy_Ctx": {"color": "green",  "label": "Epsilon-Greedy"},
    "LinUCB":        {"color": "red",    "label": "LinUCB"}
}

def main():
    print("Đang đọc dữ liệu từ thư mục results...")
    
    latency_data = {p: [] for p in POLICIES}
    pf_data = {p: [] for p in POLICIES}
    valid_rates = {p: [] for p in POLICIES}

    for p in POLICIES.keys():
        for r in RATES:
            # Code quét tự động tên file tương ứng với mỗi mức Rate và Policy
            json_name = f"Baseline_{p}_Rate{r}_summary.json"
            file_path = os.path.join(RESULTS_DIR, json_name)
                
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    mean_lat = data.get("latency_all_ms", {}).get("mean", 0)
                    tot_gen = data.get("total_generated", 0)
                    tot_done = data.get("total_done", 0)
                    
                    # Tính Xác suất thất bại = Số lượng DAG rớt / Tổng DAG tạo ra
                    pf = ((tot_gen - tot_done) / tot_gen * 100) if tot_gen > 0 else 0
                    
                    latency_data[p].append(mean_lat)
                    pf_data[p].append(pf)
                    valid_rates[p].append(r)

    # ---------------------------------------------------------
    # VẼ BIỂU ĐỒ 1: ĐỘ TRỄ (SERVICE TIME)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for p in POLICIES.keys():
        if len(valid_rates[p]) > 0:
            plt.plot(valid_rates[p], latency_data[p], marker='o', 
                     color=POLICIES[p]["color"], label=POLICIES[p]["label"], linewidth=2)
            
    plt.title("So sánh Độ trễ (Service Time) giữa các thuật toán", fontsize=14, fontweight='bold')
    plt.xlabel("Arrival Rate (requests/s)", fontsize=12)
    plt.ylabel("Service Time (ms)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(RESULTS_DIR, "Chart_1_ServiceTime.png"))
    plt.close()

    # ---------------------------------------------------------
    # VẼ BIỂU ĐỒ 2: XÁC SUẤT THẤT BẠI (PF)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for p in POLICIES.keys():
        if len(valid_rates[p]) > 0:
            plt.plot(valid_rates[p], pf_data[p], marker='s', linestyle='--',
                     color=POLICIES[p]["color"], label=POLICIES[p]["label"], linewidth=2)
            
    plt.title("So sánh Xác suất Thất bại (PF) giữa các thuật toán", fontsize=14, fontweight='bold')
    plt.xlabel("Arrival Rate (requests/s)", fontsize=12)
    plt.ylabel("Xác suất thất bại - PF (%)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(RESULTS_DIR, "Chart_2_FailureProb.png"))
    plt.close()

    print("Vẽ xong! Hai biểu đồ đã được lưu vào thư mục 'results/':")
    print(" 1. Chart_1_ServiceTime.png")
    print(" 2. Chart_2_FailureProb.png")

if __name__ == "__main__":
    main()