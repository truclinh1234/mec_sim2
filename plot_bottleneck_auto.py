import os
import json
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def extract_latency_from_dir(directory):
    data = {"AllLocal": {}, "AllEdge0": {}}
    json_files = glob.glob(os.path.join(directory, "*_summary.json"))
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        match = re.search(r'_(AllLocal|AllEdge0)_Rate([\d\.]+)_summary\.json', filename)
        if match:
            policy = match.group(1)
            rate = float(match.group(2))
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    content = json.load(f)
                    latency = content.get('latency_all_ms', {}).get('mean', 0)
                    data[policy][rate] = latency
                except:
                    pass
                    
    sr_local = pd.Series(data["AllLocal"]).sort_index() if data["AllLocal"] else pd.Series()
    sr_edge  = pd.Series(data["AllEdge0"]).sort_index() if data["AllEdge0"] else pd.Series()
    return sr_local, sr_edge


def plot_auto_bottleneck():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Màu chuẩn IEEE paper
    COLOR_LOCAL = '#1f77b4'   # xanh dương
    COLOR_EDGE  = '#d62728'   # đỏ

    # ── TASK LIGHT ──────────────────────────────────────────
    light_local, light_edge = extract_latency_from_dir("results_light")
    if not light_local.empty and not light_edge.empty:
        axes[0].plot(light_local.index, light_local.values,
                     'o-', color=COLOR_LOCAL, linewidth=1.8,
                     markersize=6, label='AllLocal')
        axes[0].plot(light_edge.index, light_edge.values,
                     's--', color=COLOR_EDGE, linewidth=1.8,
                     markersize=6, label='AllEdge0')
        axes[0].set_xticks(sorted(light_local.index))
    axes[0].set_title('Task Light (1MB, 0.3G Cycles)', fontweight='bold')
    axes[0].set_xlabel('Request Rate (tasks/s)',        fontweight='bold')
    axes[0].set_ylabel('Service Time (ms)',             fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper left')

    # ── TASK MEDIUM ──────────────────────────────────────────
    med_local, med_edge = extract_latency_from_dir("results_medium")
    if not med_local.empty and not med_edge.empty:
        axes[1].plot(med_local.index, med_local.values,
                     'o-', color=COLOR_LOCAL, linewidth=1.8,
                     markersize=6, label='AllLocal')
        axes[1].plot(med_edge.index, med_edge.values,
                     's--', color=COLOR_EDGE, linewidth=1.8,
                     markersize=6, label='AllEdge0')
        axes[1].set_xticks(sorted(med_local.index))
    axes[1].set_title('Task Medium (5MB, 1.0G Cycles)', fontweight='bold')
    axes[1].set_xlabel('Request Rate (tasks/s)',         fontweight='bold')
    axes[1].set_ylabel('Service Time (ms)',              fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper left')

    # ── TASK HEAVY ──────────────────────────────────────────
    heavy_local, heavy_edge = extract_latency_from_dir("results_heavy")
    if not heavy_local.empty and not heavy_edge.empty:
        axes[2].plot(heavy_local.index, heavy_local.values,
                     'o-', color=COLOR_LOCAL, linewidth=1.8,
                     markersize=6, label='AllLocal')
        axes[2].plot(heavy_edge.index, heavy_edge.values,
                     's--', color=COLOR_EDGE, linewidth=1.8,
                     markersize=6, label='AllEdge0')
        axes[2].set_xticks(sorted(heavy_local.index))
    axes[2].set_title('Task Heavy (20MB, 2.0G Cycles)', fontweight='bold')
    axes[2].set_xlabel('Request Rate (tasks/s)',         fontweight='bold')
    axes[2].set_ylabel('Service Time (ms)',              fontweight='bold')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend(loc='upper left')

    plt.tight_layout(pad=2.0)
    plt.savefig('Fig1_Bottleneck_Analysis_Auto.png', dpi=300, bbox_inches='tight')
    print("Xong! Đã tạo ảnh Fig1_Bottleneck_Analysis_Auto.png")


if __name__ == "__main__":
    plot_auto_bottleneck()