#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import Counter
from typing import List, Iterable

META_CHARS = set(".^$*+?{}()|")


# ------------------------------
# 1) Regex literal 抽取（保持你原来的启发式）
# ------------------------------
def extract_literals(regex: str, min_len: int = 2) -> List[str]:
    r"""
    保守、可用的 regex literal 抽取器（启发式）：
    - 处理转义 "\x" 作为字面字符（使用 raw docstring 避免 Python 源码层 \x 解析报错）
    - 跳过字符类 [...]（里面不当作 literal 串）
    - 遇到元字符就断开当前 literal
    目标是 baseline 级别“足够像 REI”的统计口径。

    注意：这是“字符串层”的保守抽取，不试图完整解析正则语义。
    """
    literals: List[str] = []
    buf: List[str] = []
    i = 0
    n = len(regex)
    in_class = False

    while i < n:
        ch = regex[i]

        # 进入字符类
        if not in_class and ch == "[":
            if buf:
                literals.append("".join(buf))
                buf = []
            in_class = True
            i += 1
            continue

        # 在字符类中：跳过内容
        if in_class:
            # 跳过转义
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "]":
                in_class = False
            i += 1
            continue

        # 转义字符 -> 作为字面字符（保持原启发式）
        if ch == "\\":
            if i + 1 < n:
                nxt = regex[i + 1]
                buf.append(nxt)
                i += 2
            else:
                # 末尾孤立反斜杠：保守处理为字面字符
                buf.append(ch)
                i += 1
            continue

        # 元字符 -> 断开
        if ch in META_CHARS:
            if buf:
                literals.append("".join(buf))
                buf = []
            i += 1
            continue

        # 普通字符
        buf.append(ch)
        i += 1

    if buf:
        literals.append("".join(buf))

    # bigram 至少需要长度 2 的 literal
    min_len = max(2, min_len)
    return [s for s in literals if len(s) >= min_len]


def bigrams_of(s: str) -> List[str]:
    return [s[i:i + 2] for i in range(len(s) - 1)]


# ------------------------------
# 2) 读取 regex 的统一入口
# ------------------------------
def iter_regex_from_jsonl(file_path: str, field: str = "regex") -> Iterable[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 允许文件里偶尔混入非 json 行
                continue
            r = obj.get(field)
            if isinstance(r, str) and r:
                yield r


def iter_regex_from_txt(file_path: str) -> Iterable[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            r = line.strip()
            if r:
                yield r


def collect_input_files(input_path: str, exts=(".jsonl", ".txt")) -> List[str]:
    """
    - 如果 input_path 是文件：直接返回
    - 如果是目录：递归收集指定后缀文件
    """
    if os.path.isfile(input_path):
        return [input_path]

    files: List[str] = []
    for root, _, names in os.walk(input_path):
        for name in names:
            if name.endswith(exts):
                files.append(os.path.join(root, name))

    # 排序保证可复现
    files.sort()
    return files


def iter_regex_from_path(path: str, regex_field: str = "regex") -> Iterable[str]:
    files = collect_input_files(path)

    for fp in files:
        if fp.endswith(".jsonl"):
            yield from iter_regex_from_jsonl(fp, field=regex_field)
        elif fp.endswith(".txt"):
            yield from iter_regex_from_txt(fp)
        else:
            continue


# ------------------------------
# 3) 主逻辑：按 “query-frequency” 统计 top-k bigrams
# ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Extract top-k bigrams from regex literals (query-frequency)."
    )
    ap.add_argument(
        "--input",
        required=True,
        help=(
            "Workload file or directory. "
            "If directory, will scan *.jsonl/*.txt under it."
        ),
    )
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument(
        "--regex_field",
        type=str,
        default="regex",
        help="Field name for regex in JSONL lines.",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output json mapping bigram->bitpos",
    )
    ap.add_argument(
        "--min_literal_len",
        type=int,
        default=2,
        help="Min literal length to be considered (default 2).",
    )
    args = ap.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be a positive integer")

    # bigram 至少需要长度 2 的 literal
    if args.min_literal_len < 2:
        args.min_literal_len = 2

    qfreq = Counter()
    total_queries = 0

    for r in iter_regex_from_path(args.input, regex_field=args.regex_field):
        total_queries += 1
        lits = extract_literals(r, min_len=args.min_literal_len)

        # query-frequency 口径：每条 query 内去重
        bg_set = set()
        for lit in lits:
            for bg in bigrams_of(lit):
                bg_set.add(bg)

        for bg in bg_set:
            qfreq[bg] += 1

    top = [bg for bg, _ in qfreq.most_common(args.k)]
    mapping = {bg: i for i, bg in enumerate(top)}

    out_obj = {
        "k": args.k,
        "mapping": mapping,
        "stats": {
            "input": args.input,
            "regex_field": args.regex_field,
            "min_literal_len": args.min_literal_len,
            "total_queries_counted": total_queries,
            "unique_bigrams_in_workload": len(qfreq),
            "topk_query_frequency": {bg: qfreq[bg] for bg in top},
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote top-{args.k} bigram mapping to {args.output}")
    print(f"[INFO] total queries counted: {total_queries}")
    print(f"[INFO] unique bigrams in workload: {len(qfreq)}")


if __name__ == "__main__":
    main()
