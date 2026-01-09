#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import Counter
from typing import List, Iterable

# 正则元字符，用来“断开 literal”的分隔符
META_CHARS = set(".^$*+?{}()|")


# ------------------------------
# 1) Regex literal 抽取（改进版）
# ------------------------------
def extract_literals(regex: str, min_len: int = 2) -> List[str]:
    r"""
    保守、可用的 regex literal 抽取器（启发式）：

    - 处理转义 "\\x" 作为字面字符（例如 "\-" -> "-"、"\." -> "."）
    - 跳过字符类 [...]（里面不当作 literal 串）
    - 跳过量词 {m,n} 里的数字和逗号（避免把 "10,10" 这类当成 literal）
    - 跳过 \d/\w/\s 这种“语义类”转义，不把 d/w/s 当成字面
    - 其它元字符 . ^ $ * + ? ( ) | 仅作为“分隔符”，用来断开 literal

    目标是：bigram 统计尽量贴近“真正的文本片段”，
    又不过度复杂化解析。
    """
    literals: List[str] = []
    buf: List[str] = []
    i = 0
    n = len(regex)
    in_class = False  # 是否在字符类 [...] 里面

    while i < n:
        ch = regex[i]

        # 1) 进入字符类 [...]
        if not in_class and ch == "[":
            if buf:
                literals.append("".join(buf))
                buf = []
            in_class = True
            i += 1
            continue

        # 2) 字符类内部：全部跳过，不当作 literal
        if in_class:
            # 跳过转义 "\x"
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            # 遇到 ] 结束字符类
            if ch == "]":
                in_class = False
            i += 1
            continue

        # 3) 量词 {m,n}：整个块跳过（包括里面的数字/逗号）
        if ch == "{":
            if buf:
                literals.append("".join(buf))
                buf = []
            i += 1
            while i < n:
                c2 = regex[i]
                # 量词内部的转义，整体跳过
                if c2 == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c2 == "}":
                    i += 1
                    break
                i += 1
            continue

        # 4) 转义字符：区分“语义类转义”和“转义字面量”
        if ch == "\\":
            if i + 1 < n:
                nxt = regex[i + 1]

                # 4.1 \d \D \w \W \s \S 等：视为“模式”，不进入 literal
                if nxt in ("d", "D", "w", "W", "s", "S"):
                    i += 2
                    continue

                # 4.2 一些位置类/锚点转义，也可以直接跳过
                if nxt in ("b", "B", "A", "Z", "z"):
                    i += 2
                    continue

                # 4.3 其它情况：当作“转义字面字符”
                #    例如 "\." -> "."，"\-" -> "-"，"\(" -> "("
                buf.append(nxt)
                i += 2
            else:
                # 末尾孤立 '\'，极少见，保守地当作字面
                buf.append(ch)
                i += 1
            continue

        # 5) 普通元字符：作为分隔符，断开 literal
        if ch in META_CHARS:
            if buf:
                literals.append("".join(buf))
                buf = []
            i += 1
            continue

        # 6) 其它普通字符：进入当前 literal 缓冲
        buf.append(ch)
        i += 1

    # 循环结束，收尾
    if buf:
        literals.append("".join(buf))

    # bigram 至少需要长度 2 的 literal
    min_len = max(2, min_len)
    return [s for s in literals if len(s) >= min_len]


def bigrams_of(s: str) -> List[str]:
    """把字符串切成所有长度=2 的重叠 bigram。"""
    return [s[i:i + 2] for i in range(len(s) - 1)]


# ------------------------------
# 2) 读取 regex 的统一入口
# ------------------------------
def iter_regex_from_jsonl(file_path: str, field: str = "regex") -> Iterable[str]:
    """从 JSONL 文件中按给定字段名提取 regex 字符串。"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 允许偶尔混入非 json 行
                continue
            r = obj.get(field)
            if isinstance(r, str) and r:
                yield r


def iter_regex_from_txt(file_path: str) -> Iterable[str]:
    """从纯文本文件中，一行一个 regex。"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            r = line.strip()
            if r:
                yield r


def collect_input_files(input_path: str, exts=(".jsonl", ".txt")) -> List[str]:
    """
    - 如果 input_path 是文件：直接返回 [path]
    - 如果是目录：递归收集指定后缀的所有文件
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
    """统一入口：既支持一个文件，也支持一个目录。"""
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

    # 遍历 workload（可以是单文件，也可以是目录）
    for r in iter_regex_from_path(args.input, regex_field=args.regex_field):
        total_queries += 1
        lits = extract_literals(r, min_len=args.min_literal_len)

        # query-frequency 口径：同一条 query 内对 bigram 去重
        bg_set = set()
        for lit in lits:
            for bg in bigrams_of(lit):
                bg_set.add(bg)

        for bg in bg_set:
            qfreq[bg] += 1

    # 取 top-k bigram，并赋予 bit 位置
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

