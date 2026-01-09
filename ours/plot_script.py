#!/usr/bin/env python3
"""
性能数据可视化脚本（按数据集分组）
为每个数据集前缀生成独立的图表
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# 设置中文字体支持（可选）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(json_file: str) -> dict:
    """加载JSON数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def group_data_by_prefix(data: dict) -> dict:
    """按数据集前缀分组数据"""
    grouped = defaultdict(dict)
    
    for dataset_name, dataset_data in data.items():
        # 提取数据集前缀（如 us_acc, imdb_name, wiki, lineitem）
        prefix = dataset_name.split('.')[0]
        grouped[prefix][dataset_name] = dataset_data
    
    return dict(grouped)


def prepare_plot_data_for_prefix(data: dict):
    """为特定前缀准备绘图数据"""
    labels = []
    avg_times = []
    avg_recalls = []
    
    # 定义kind的顺序和颜色
    kinds = ['base', 'trgm', 'prefix']
    colors = {'base': '#FF6B6B', 'trgm': '#4ECDC4', 'prefix': '#45B7D1'}
    
    # 遍历该前缀下的所有数据集
    for dataset_name in sorted(data.keys()):
        dataset_data = data[dataset_name]
        
        for kind in kinds:
            if kind in dataset_data:
                label = f"{dataset_name}.{kind}"
                labels.append(label)
                avg_times.append(dataset_data[kind]['avg_time'])
                avg_recalls.append(dataset_data[kind]['avg_recall'])
    
    return labels, avg_times, avg_recalls, colors


def create_time_plot(prefix, labels, avg_times, colors, output_file):
    """创建平均时间图表"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 为每个kind分配颜色
    bar_colors = []
    for label in labels:
        if '.base' in label:
            bar_colors.append(colors['base'])
        elif '.trgm' in label:
            bar_colors.append(colors['trgm'])
        elif '.prefix' in label:
            bar_colors.append(colors['prefix'])
        else:
            bar_colors.append('#95A5A6')
    
    x = np.arange(len(labels))
    bars = ax.bar(x, avg_times, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 设置标签和标题
    ax.set_xlabel('Configuration.Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'Average Execution Time - {prefix.upper()} Dataset', fontsize=14, fontweight='bold', pad=20)
    
    # 设置x轴标签
    ax.set_xticks(x)
    # 简化标签，移除前缀
    simplified_labels = [label.replace(f"{prefix}.", "") for label in labels]
    ax.set_xticklabels(simplified_labels, rotation=45, ha='right', fontsize=9)
    
    # 添加网格
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['base'], label='Base', alpha=0.8),
        Patch(facecolor=colors['trgm'], label='Trigram', alpha=0.8),
        Patch(facecolor=colors['prefix'], label='Prefix', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # 在柱状图上添加数值标签
    max_time = max(avg_times) if avg_times else 1
    for i, (bar, time) in enumerate(zip(bars, avg_times)):
        if time > max_time * 0.05:  # 只显示较大值的标签
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.0f}',
                   ha='center', va='bottom', fontsize=7, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Time plot saved to: {output_file}")
    plt.close()


def create_recall_plot(prefix, labels, avg_recalls, colors, output_file):
    """创建平均召回率图表"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 为每个kind分配颜色
    bar_colors = []
    for label in labels:
        if '.base' in label:
            bar_colors.append(colors['base'])
        elif '.trgm' in label:
            bar_colors.append(colors['trgm'])
        elif '.prefix' in label:
            bar_colors.append(colors['prefix'])
        else:
            bar_colors.append('#95A5A6')
    
    x = np.arange(len(labels))
    bars = ax.bar(x, avg_recalls, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 设置标签和标题
    ax.set_xlabel('Configuration.Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Recall', fontsize=12, fontweight='bold')
    ax.set_title(f'Average Recall - {prefix.upper()} Dataset', fontsize=14, fontweight='bold', pad=20)
    
    # 设置x轴标签
    ax.set_xticks(x)
    # 简化标签，移除前缀
    simplified_labels = [label.replace(f"{prefix}.", "") for label in labels]
    ax.set_xticklabels(simplified_labels, rotation=45, ha='right', fontsize=9)
    
    # 设置y轴范围为0-1
    ax.set_ylim([0, 1.05])
    
    # 添加网格
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['base'], label='Base', alpha=0.8),
        Patch(facecolor=colors['trgm'], label='Trigram', alpha=0.8),
        Patch(facecolor=colors['prefix'], label='Prefix', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)
    
    # 在柱状图上添加数值标签
    for i, (bar, recall) in enumerate(zip(bars, avg_recalls)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{recall:.2f}',
               ha='center', va='bottom', fontsize=7, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Recall plot saved to: {output_file}")
    plt.close()


def create_combined_plot(prefix, labels, avg_times, avg_recalls, colors, output_file):
    """创建组合图表（时间和召回率）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 为每个kind分配颜色
    bar_colors = []
    for label in labels:
        if '.base' in label:
            bar_colors.append(colors['base'])
        elif '.trgm' in label:
            bar_colors.append(colors['trgm'])
        elif '.prefix' in label:
            bar_colors.append(colors['prefix'])
        else:
            bar_colors.append('#95A5A6')
    
    x = np.arange(len(labels))
    
    # 简化标签，移除前缀
    simplified_labels = [label.replace(f"{prefix}.", "") for label in labels]
    
    # 第一个子图：平均时间
    bars1 = ax1.bar(x, avg_times, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Average Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Average Execution Time - {prefix.upper()} Dataset', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(simplified_labels, rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # 第二个子图：平均召回率
    bars2 = ax2.bar(x, avg_recalls, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Configuration.Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Recall', fontsize=12, fontweight='bold')
    ax2.set_title(f'Average Recall - {prefix.upper()} Dataset', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(simplified_labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # 添加图例（只在第一个子图）
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['base'], label='Base', alpha=0.8),
        Patch(facecolor=colors['trgm'], label='Trigram', alpha=0.8),
        Patch(facecolor=colors['prefix'], label='Prefix', alpha=0.8)
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Combined plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='绘制性能数据图表（按数据集前缀分组）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python plot_script.py --input analysis_results.json --output_dir ./plots
  python plot_script.py --input analysis_results.json --output_dir ./plots --combined
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入的JSON文件路径'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./plots',
        help='输出图表的目录（默认: ./plots）'
    )
    
    parser.add_argument(
        '--combined',
        action='store_true',
        help='生成组合图表（时间和召回在一张图）'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"Loading data from: {args.input}")
    data = load_data(args.input)
    
    # 按前缀分组数据
    print("Grouping data by dataset prefix...")
    grouped_data = group_data_by_prefix(data)
    
    print(f"Found {len(grouped_data)} dataset prefixes: {', '.join(sorted(grouped_data.keys()))}")
    
    # 为每个前缀生成图表
    for prefix in sorted(grouped_data.keys()):
        print(f"\nProcessing {prefix} dataset...")
        prefix_data = grouped_data[prefix]
        
        # 准备绘图数据
        labels, avg_times, avg_recalls, colors = prepare_plot_data_for_prefix(prefix_data)
        
        print(f"  Found {len(labels)} data points")
        
        # 生成图表
        if args.combined:
            output_file = output_dir / f"{prefix}_performance_combined.png"
            create_combined_plot(prefix, labels, avg_times, avg_recalls, colors, output_file)
        else:
            # 生成时间图表
            time_output = output_dir / f"{prefix}_avg_time.png"
            create_time_plot(prefix, labels, avg_times, colors, time_output)
            
            # 生成召回率图表
            recall_output = output_dir / f"{prefix}_avg_recall.png"
            create_recall_plot(prefix, labels, avg_recalls, colors, recall_output)
    
    print("\n" + "="*60)
    print(f"All plots saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()