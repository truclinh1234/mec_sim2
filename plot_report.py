import os
import json
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cài đặt font chữ chuẩn Báo cáo khoa học
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def extract_data_from_json():
    data = []
    json_files = glob.glob(os.path.join("results", "*_summary.json"))
    
    if not json_files:
        print("Lỗi: Không có JSON trong thư mục results/")
        return None

    for file_path in json_files:
        filename = os.path.basename(file_path)
        match = re.search(r'_(.*?)_Rate([\d\.]+)_summary\.json', filename)
        if match:
            policy_name = match.group(1)
            rate = float(match.group(2))
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    content = json.load(f)
                    total_generated = content.get('total_generated', 0)
                    total_done = content.get('total_done', 0)
                    mean_latency = content.get('latency_all_ms', {}).get('mean', 0)
                    
                    # Tính Xác suất thất bại (PF)
                    failure_prob = 1.0 - (total_done / total_generated) if total_generated > 0 else 0.0
                    
                    data.append({
                        "Rate": rate,
                        "Policy": policy_name,
                        "DAG_Finished": total_done,
                        "Total_Generated": total_generated,
                        "Failure_Prob": failure_prob * 100, # Đổi ra phần trăm (%)
                        "Latency": mean_latency
                    })
                except:
                    pass
                    
    df = pd.DataFrame(data)
    df = df.sort_values(by=['Rate', 'Policy'])
    return df

def plot_all_figures():
    df = extract_data_from_json()
    if df is None or df.empty:
        return

    rates = sorted(df['Rate'].unique())
    policies = df['Policy'].unique()

    # Style markers và màu sắc để phân biệt rõ các đường
    markers = ['o', 's', '^', 'x', 'D', '*']
    colors = ['gray', 'orange', 'green', 'purple', 'blue', 'red']
    
    # =======================================================
    # 1. BIỂU ĐỒ ĐỘ TRỄ (Fig2_Latency.png)
    # =======================================================
    # Mở rộng chiều ngang để đồ thị thoáng hơn
    plt.figure(figsize=(10, 6))
    for i, policy in enumerate(policies):
        policy_data = df[df['Policy'] == policy]
        is_linucb = 'LinUCB' in policy
        
        plt.plot(policy_data['Rate'], policy_data['Latency'], 
                 marker=markers[i % len(markers)], 
                 color='red' if is_linucb else colors[i % len(colors)],
                 linestyle='-' if is_linucb else '--',
                 linewidth=2.5 if is_linucb else 1.5,
                 markersize=10 if is_linucb else 6,
                 label=policy)
        
    plt.xlabel('Arrival Rate', fontweight='bold')
    plt.ylabel('Độ trễ TB (ms)', fontweight='bold')
    plt.title('So sánh Độ trễ giữa các thuật toán', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    # Thêm xoay chữ nếu cần
    plt.xticks(rates, rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/Fig2_Latency.png', dpi=300)

    # =======================================================
    # 2. BIỂU ĐỒ DAG HOÀN THÀNH (Fig3_Completed_DAGs.png)
    # =======================================================
    # BƯỚC SỬA 1: Tăng kích thước chiều ngang của ảnh lên 12 để các cột không bị ép
    plt.figure(figsize=(12, 6))
    num_policies = len(policies)
    bar_width = 0.8 / num_policies
    x = np.arange(len(rates))

    for i, policy in enumerate(policies):
        policy_data = df[df['Policy'] == policy]
        is_linucb = 'LinUCB' in policy
        
        y_data = []
        for r in rates:
            val = policy_data[policy_data['Rate'] == r]['DAG_Finished'].values
            y_data.append(val[0] if len(val) > 0 else 0)

        offset = (i - num_policies / 2) * bar_width + bar_width / 2
        plt.bar(x + offset, y_data, width=bar_width, 
                label=policy, color='red' if is_linucb else colors[i % len(colors)])

    plt.xlabel('Arrival Rate', fontweight='bold')
    plt.ylabel('Số lượng hoàn thành (Success)', fontweight='bold')
    plt.title('Đánh giá Số lượng hoàn thành ứng dụng', fontweight='bold')
    
    # BƯỚC SỬA 2: Xoay nghiêng chữ 45 độ, căn phải, giảm cỡ chữ xuống 10
    plt.xticks(x, rates, rotation=45, ha='right', fontsize=10)
    
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/Fig3_Completed_DAGs.png', dpi=300)

    # =======================================================
    # 3. BIỂU ĐỒ XÁC SUẤT THẤT BẠI - PF (Fig4_Failure_Probability.png)
    # =======================================================
    # Mở rộng chiều ngang
    plt.figure(figsize=(10, 6))
    for i, policy in enumerate(policies):
        policy_data = df[df['Policy'] == policy]
        is_linucb = 'LinUCB' in policy
        
        if not policy_data.empty:
            plt.plot(policy_data['Rate'], policy_data['Failure_Prob'], 
                     marker=markers[i % len(markers)], 
                     color='red' if is_linucb else colors[i % len(colors)],
                     linestyle='-' if is_linucb else '--',
                     linewidth=2.5 if is_linucb else 1.5,
                     markersize=10 if is_linucb else 6,
                     label=policy)

    plt.xlabel('Arrival Rate', fontweight='bold')
    plt.ylabel('Xác suất thất bại - PF (%)', fontweight='bold')
    plt.title('Đánh giá Xác suất thất bại', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    # Thêm xoay chữ nếu cần
    plt.xticks(rates, rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/Fig4_Failure_Probability.png', dpi=300)
    print("Tuyệt vời! Đã xuất thành công 3 ảnh, chữ ở trục hoành không còn bị đè lên nhau nữa.")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    plot_all_figures()