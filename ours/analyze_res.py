#!/usr/bin/env python3
"""
正则搜索性能分析脚本
分析不同索引方法（base, trgm, prefix）的执行时间和召回率
"""

import json
import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def extract_filename(input_path: str) -> str:
    """从input路径提取文件名（不含扩展名）"""
    filename = Path(input_path).name
    # 移除.jsonl扩展名
    if filename.endswith('.jsonl'):
        filename = filename[:-6]
    return filename


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """计算统计指标：平均值、最大值、P90、P99"""
    if not values:
        return {
            "avg": 0.0,
            "max": 0.0,
            "p90": 0.0,
            "p99": 0.0
        }
    
    arr = np.array(values)
    return {
        "avg": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99))
    }


def process_file(file_path: str) -> Dict[str, Any]:
    """处理单个JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取文件名
    filename = extract_filename(data.get('input', ''))
    
    # 收集每种kind的数据
    kind_data = {
        'base': {'times': [], 'recalls': []},
        'trgm': {'times': [], 'recalls': []},
        'prefix': {'times': [], 'recalls': []}
    }
    
    # 遍历results
    results = data.get('results', [])
    for result in results:
        if result.get('error'):
            continue
            
        queries = result.get('queries', [])
        
        # 首先找到base的count作为基准
        base_count = None
        for query in queries:
            if query.get('kind') == 'base' and query.get('error') is None:
                base_count = query.get('count', 0)
                break
        
        if base_count is None or base_count == 0:
            continue
        
        # 处理每个query
        for query in queries:
            kind = query.get('kind')
            if kind not in kind_data or query.get('error'):
                continue
            
            exec_time = query.get('execution_time_ms')
            count = query.get('count', 0)
            
            if exec_time is not None:
                kind_data[kind]['times'].append(exec_time)
                
                # 计算召回率
                recall = count / base_count if base_count > 0 else 0.0
                kind_data[kind]['recalls'].append(recall)
    
    # 计算统计数据
    result = {}
    for kind in ['base', 'trgm', 'prefix']:
        times = kind_data[kind]['times']
        recalls = kind_data[kind]['recalls']
        
        time_stats = calculate_statistics(times)
        recall_stats = calculate_statistics(recalls)
        
        result[kind] = {
            'avg_time': time_stats['avg'],
            'max_time': time_stats['max'],
            'p90_time': time_stats['p90'],
            'p99_time': time_stats['p99'],
            'avg_recall': recall_stats['avg'],
            'max_recall': recall_stats['max'],
            'p90_recall': recall_stats['p90'],
            'p99_recall': recall_stats['p99'],
            'count': len(times)
        }
    
    return {filename: result}


def process_folder(input_folder: str, output_json: str):
    """处理文件夹中的所有JSON文件"""
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"错误：输入文件夹不存在: {input_folder}")
        return
    
    # 收集所有JSON文件
    json_files = list(input_path.glob('*.json'))
    
    if not json_files:
        print(f"警告：文件夹中没有找到JSON文件: {input_folder}")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 处理所有文件
    all_results = {}
    for json_file in json_files:
        print(f"处理: {json_file.name}")
        try:
            file_result = process_file(str(json_file))
            all_results.update(file_result)
        except Exception as e:
            print(f"  错误: {e}")
            continue
    
    # 保存结果
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_json}")
    print(f"处理了 {len(all_results)} 个数据集")


def main():
    parser = argparse.ArgumentParser(
        description='分析正则搜索性能数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python script.py --input_folder ./Res --output_json ./analysis_results.json
        """
    )
    
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='输入文件夹路径，包含待处理的JSON文件'
    )
    
    parser.add_argument(
        '--output_json',
        type=str,
        required=True,
        help='输出JSON文件路径'
    )
    
    args = parser.parse_args()
    
    process_folder(args.input_folder, args.output_json)


if __name__ == '__main__':
    main()