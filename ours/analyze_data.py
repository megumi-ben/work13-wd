#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析数据中结构化 token 的比例
帮助判断动态 regex 生成的多样性

使用: python analyze_data.py --data_path ./data/seeds.jsonl
"""

import argparse
import json
import re
from collections import Counter
from typing import List, Dict, Any
from tqdm import tqdm

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
ALPHA_RE = re.compile(r"^[A-Za-z]+$")
DIGIT_RE = re.compile(r"\d")


def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(s or "")


def is_punct(tok: str) -> bool:
    return bool(re.fullmatch(r"[^\w\s]", tok or ""))


def is_structured_token(tok: str) -> bool:
    """结构化 token：包含数字或连字符"""
    if not tok or is_punct(tok):
        return False
    return (DIGIT_RE.search(tok) is not None) or ("-" in tok)


def is_pure_alpha(tok: str) -> bool:
    """纯字母 token"""
    return bool(ALPHA_RE.fullmatch(tok or ""))


def analyze_record(seed: str) -> Dict[str, Any]:
    """分析单条记录"""
    tokens = tokenize(seed)
    non_punct_tokens = [t for t in tokens if not is_punct(t)]
    
    struct_tokens = [t for t in non_punct_tokens if is_structured_token(t)]
    alpha_tokens = [t for t in non_punct_tokens if is_pure_alpha(t)]
    
    # 计算字符暴露长度
    struct_char_len = sum(len(re.findall(r'[A-Za-z0-9_]', t)) for t in struct_tokens)
    alpha_char_len = sum(len(t) for t in alpha_tokens)
    total_char_len = sum(len(re.findall(r'[A-Za-z0-9_]', t)) for t in non_punct_tokens)
    
    return {
        "n_tokens": len(non_punct_tokens),
        "n_struct": len(struct_tokens),
        "n_alpha": len(alpha_tokens),
        "struct_tokens": struct_tokens,
        "struct_ratio": len(struct_tokens) / max(1, len(non_punct_tokens)),
        "struct_char_ratio": struct_char_len / max(1, total_char_len),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--seed_field", type=str, default="seed_string")
    parser.add_argument("--max_samples", type=int, default=-1, help="-1 for all")
    parser.add_argument("--show_examples", type=int, default=10)
    args = parser.parse_args()

    # 加载数据
    records = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                seed = obj.get(args.seed_field, "")
            except json.JSONDecodeError:
                seed = line
            if seed:
                records.append(seed)
    
    if args.max_samples > 0:
        records = records[:args.max_samples]
    
    print(f"\n总记录数: {len(records)}")
    
    # 分析每条记录
    results = []
    struct_token_counter = Counter()
    
    for seed in tqdm(records, desc="Analyzing"):
        r = analyze_record(seed)
        results.append(r)
        for t in r["struct_tokens"]:
            struct_token_counter[t] += 1
    
    # 统计
    n_has_struct = sum(1 for r in results if r["n_struct"] > 0)
    avg_struct_ratio = sum(r["struct_ratio"] for r in results) / len(results)
    avg_struct_char_ratio = sum(r["struct_char_ratio"] for r in results) / len(results)
    
    # 按结构化程度分类
    high_struct = [r for r in results if r["struct_ratio"] > 0.5]  # >50% 结构化
    medium_struct = [r for r in results if 0.1 < r["struct_ratio"] <= 0.5]
    low_struct = [r for r in results if 0 < r["struct_ratio"] <= 0.1]
    no_struct = [r for r in results if r["struct_ratio"] == 0]
    
    print("\n" + "=" * 70)
    print("结构化 Token 分析报告")
    print("=" * 70)
    
    print(f"\n【整体统计】")
    print(f"  总记录数: {len(records)}")
    print(f"  包含结构化token的记录: {n_has_struct} ({n_has_struct/len(records)*100:.1f}%)")
    print(f"  平均结构化token比例: {avg_struct_ratio*100:.1f}%")
    print(f"  平均结构化字符比例: {avg_struct_char_ratio*100:.1f}%")
    
    print(f"\n【分布情况】")
    print(f"  高结构化 (>50%):  {len(high_struct):6d} 条 ({len(high_struct)/len(records)*100:.1f}%)")
    print(f"  中结构化 (10-50%): {len(medium_struct):6d} 条 ({len(medium_struct)/len(records)*100:.1f}%)")
    print(f"  低结构化 (1-10%):  {len(low_struct):6d} 条 ({len(low_struct)/len(records)*100:.1f}%)")
    print(f"  无结构化 (0%):    {len(no_struct):6d} 条 ({len(no_struct)/len(records)*100:.1f}%)")
    
    print(f"\n【多样性预估】")
    # 无结构化 → 100% diversity
    # 低结构化 → 80-100%
    # 中结构化 → 50-80%
    # 高结构化 → 20-50%
    estimated_diversity = (
        len(no_struct) * 1.0 +
        len(low_struct) * 0.9 +
        len(medium_struct) * 0.65 +
        len(high_struct) * 0.35
    ) / len(records)
    print(f"  预估整体多样性: {estimated_diversity*100:.1f}%")
    
    if estimated_diversity > 0.8:
        print("  → 效果良好，无需特殊处理")
    elif estimated_diversity > 0.6:
        print("  → 效果中等，可以接受")
    else:
        print("  → 效果较差，建议优化结构化token处理逻辑")
    
    print(f"\n【最常见的结构化 Token (Top 20)】")
    for tok, cnt in struct_token_counter.most_common(20):
        print(f"  {tok:30s} 出现 {cnt:6d} 次")
    
    # 示例
    if args.show_examples > 0:
        print(f"\n【高结构化记录示例】")
        for i, r in enumerate(high_struct[:args.show_examples]):
            idx = results.index(r)
            print(f"  [{i+1}] struct_ratio={r['struct_ratio']:.0%}")
            print(f"      seed: {records[idx][:80]}...")
            print(f"      struct_tokens: {r['struct_tokens'][:5]}")
        
        print(f"\n【纯文本记录示例】")
        for i, r in enumerate(no_struct[:args.show_examples]):
            idx = results.index(r)
            print(f"  [{i+1}] {records[idx][:80]}...")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()