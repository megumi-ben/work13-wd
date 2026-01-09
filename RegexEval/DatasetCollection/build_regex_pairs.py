#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用法示例：
    python build_regex_pos_pairs_from_regexeval.py \
        --input RegexEval.json \
        --output regex_pos_pairs.jsonl \
        --regex-field expression \
        --matches-field matches \
        --max-pos-per-regex 50
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_pairs_for_sample(
    sample: Dict[str, Any],
    regex_field: str,
    matches_field: str,
    max_pos_per_regex: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """从单条样本构造若干 (regex, positive_example) 对。"""
    if regex_field not in sample:
        print(
            f"[WARN] 样本缺少正则字段 '{regex_field}', keys={list(sample.keys())}",
            file=sys.stderr,
        )
        return []

    if matches_field not in sample:
        print(
            f"[WARN] 样本缺少正样例字段 '{matches_field}', keys={list(sample.keys())}",
            file=sys.stderr,
        )
        return []

    regex = sample[regex_field]
    matches = sample[matches_field]

    if not isinstance(matches, list) or not matches:
        return []

    # 可选：每个 regex 最多保留多少个正样例
    if max_pos_per_regex is not None and len(matches) > max_pos_per_regex:
        matches = random.sample(matches, k=max_pos_per_regex)

    sid = sample.get("id", None)

    pairs = []
    for m in matches:
        if not isinstance(m, str):
            continue
        pairs.append(
            {
                "id": sid,
                "regex": regex,
                "positive_example": m,
            }
        )
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="从 RegexEval JSON 构造 (regex, positive_example) 数据集"
    )
    parser.add_argument("--input", type=str, required=True, help="RegexEval.json 路径")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 JSONL 路径（每行一个 (regex, positive_example)）",
    )
    parser.add_argument(
        "--regex-field",
        type=str,
        default="expression",
        help="JSON 中存放正则的字段名（例如 'expression'）",
    )
    parser.add_argument(
        "--matches-field",
        type=str,
        default="matches",
        help="JSON 中存放正样例列表的字段名（例如 'matches'）",
    )
    parser.add_argument(
        "--max-pos-per-regex",
        type=int,
        default=None,
        help="每个 regex 最多保留多少个 positive_example（默认不过滤）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.is_file():
        print(f"[ERROR] 输入文件不存在: {in_path}", file=sys.stderr)
        sys.exit(1)

    # 这里假设 RegexEval.json 是一个大 list
    with in_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"[ERROR] 读取/解析 JSON 失败: {e}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(data, list):
        print(f"[ERROR] 期望 JSON 顶层是 list，但实际是 {type(data)}", file=sys.stderr)
        sys.exit(1)

    num_samples = 0
    num_pairs = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for sample in data:
            num_samples += 1
            pairs = build_pairs_for_sample(
                sample,
                regex_field=args.regex_field,
                matches_field=args.matches_field,
                max_pos_per_regex=args.max_pos_per_regex,
            )
            for p in pairs:
                fout.write(json.dumps(p, ensure_ascii=False) + "\n")
                num_pairs += 1

    print(
        f"[INFO] 处理完成：原始样本 {num_samples} 条，输出 (regex, positive_example) 对 {num_pairs} 条。"
    )


if __name__ == "__main__":
    main()
