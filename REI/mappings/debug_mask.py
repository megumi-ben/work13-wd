#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from typing import Dict, List, Tuple

META_CHARS = set(".^$*+?{}()|")

def extract_literals(regex: str) -> List[str]:
    literals: List[str] = []
    buf: List[str] = []
    i = 0
    n = len(regex)
    in_class = False

    while i < n:
        ch = regex[i]

        # --- 字符类 ---
        if not in_class and ch == "[":
            if buf:
                literals.append("".join(buf))
                buf = []
            in_class = True
            i += 1
            continue

        if in_class:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "]":
                in_class = False
            i += 1
            continue

        # --- 量词 {m}/{m,n}：跳过 ---
        if ch == "{":
            if buf:
                literals.append("".join(buf))
                buf = []
            i += 1
            depth = 1
            while i < n and depth > 0:
                c = regex[i]
                if c == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c == "{":
                    depth += 1
                    i += 1
                    continue
                if c == "}":
                    depth -= 1
                    i += 1
                    continue
                i += 1
            continue

        # --- 转义序列 ---
        if ch == "\\":
            if i + 1 < n:
                nxt = regex[i + 1]

                # 模式类
                if nxt in ("d","D","w","W","s","S","b","B","t","n","r","f","v"):
                    if buf:
                        literals.append("".join(buf))
                        buf = []
                    i += 2
                    continue

                # backref \1..\9
                if nxt.isdigit():
                    if buf:
                        literals.append("".join(buf))
                        buf = []
                    i += 2
                    continue

                # 其它：当作字面字符
                buf.append(nxt)
                i += 2
                continue
            else:
                buf.append(ch)
                i += 1
                continue

        # --- 结构元字符：分隔 ---
        if ch in META_CHARS:
            if buf:
                literals.append("".join(buf))
                buf = []
            i += 1
            continue

        # --- 普通字面 ---
        buf.append(ch)
        i += 1

    if buf:
        literals.append("".join(buf))

    return [s for s in literals if len(s) >= 2]

def bigrams_of(s: str) -> List[str]:
    return [s[i:i+2] for i in range(len(s)-1)]

def load_mapping(path: str) -> Tuple[int, Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return int(obj["k"]), obj["mapping"]

def build_mask_from_positions(k: int, positions: List[int], reverse: bool) -> str:
    bits = ["0"] * k
    for p in positions:
        if not (0 <= p < k):
            continue
        idx = (k - 1 - p) if reverse else p
        bits[idx] = "1"
    return "".join(bits)

def pg_escape_e_literal(s: str) -> str:
    """
    用于拼接到 Postgres 的 E'...' 字符串字面量中：
    - 反斜杠必须翻倍（E'' 里 \ 是转义前缀）
    - 单引号用 '' 表示
    """
    return s.replace("\\", "\\\\").replace("'", "''")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", required=True, help="mapping json: *_topk_64.json")
    ap.add_argument("--regex", required=True, help="regex string (same as you put into PG)")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--sig_col", default="sig_rei_a_64", help="for SQL snippet printing")
    ap.add_argument("--table", default="public.us_accidents_rei_64", help="for SQL snippet printing")
    ap.add_argument("--case_insensitive", action="store_true",
                    help="use ~* instead of ~ (recommended if you used ~* in PG)")
    args = ap.parse_args()

    k_m, mapping = load_mapping(args.mapping)
    if k_m != args.k:
        raise ValueError(f"k mismatch: mapping k={k_m}, args.k={args.k}")

    lits = extract_literals(args.regex)
    bg_set = set()
    for lit in lits:
        bg_set.update(bigrams_of(lit))

    hit = []
    miss = []
    for bg in sorted(bg_set):
        if bg in mapping:
            hit.append((bg, int(mapping[bg])))
        else:
            miss.append(bg)

    positions = [pos for _, pos in hit]

    mask_direct = build_mask_from_positions(args.k, positions, reverse=False)
    mask_reverse = build_mask_from_positions(args.k, positions, reverse=True)

    op = "~*" if args.case_insensitive else "~"
    regex_sql = pg_escape_e_literal(args.regex)

    print("=== literals ===")
    print(lits)
    print(f"literal_count={len(lits)}")

    print("\n=== unique bigrams from literals ===")
    print(f"unique_bigrams={len(bg_set)}")

    print("\n=== hit in mapping (bigram -> pos) ===")
    print(f"hit={len(hit)}")
    for bg, pos in hit:
        print(f"{bg!r}\t{pos}")

    print("\n=== masks ===")
    print(f"[direct ] ones={mask_direct.count('1')} mask={mask_direct}")
    print(f"[reverse] ones={mask_reverse.count('1')} mask={mask_reverse}")

    print("\n=== SQL snippets (copy/paste) ===")
    print("-- 1) candidate count (direct)")
    print(
        f"SELECT count(*) AS candidates_direct FROM {args.table} "
        f"WHERE ({args.sig_col} & B'{mask_direct}') = B'{mask_direct}';"
    )

    print("-- 2) match count (direct)")
    print(
        f"SELECT count(*) AS matched_direct FROM {args.table} "
        f"WHERE ({args.sig_col} & B'{mask_direct}') = B'{mask_direct}' "
        f"AND text_content {op} E'{regex_sql}';"
    )

    print("-- 3) candidate count (reverse)  (用于验证 bit 顺序是否反了)")
    print(
        f"SELECT count(*) AS candidates_reverse FROM {args.table} "
        f"WHERE ({args.sig_col} & B'{mask_reverse}') = B'{mask_reverse}';"
    )

    print("-- 4) match count (reverse)")
    print(
        f"SELECT count(*) AS matched_reverse FROM {args.table} "
        f"WHERE ({args.sig_col} & B'{mask_reverse}') = B'{mask_reverse}' "
        f"AND text_content {op} E'{regex_sql}';"
    )

if __name__ == "__main__":
    main()
