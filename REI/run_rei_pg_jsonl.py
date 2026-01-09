#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from typing import List, Dict, Tuple

import psycopg2


# =========================
# 0) 固定你的数据库参数（你可以按需改）
# =========================
DEFAULT_DSN = "host=127.0.0.1 port=5000 dbname=regex_db user=lyz password=123456"


# =========================
# 1) Regex literal 抽取（与 top-k 统计保持一致的“更干净”版本）
# =========================
META_CHARS = set(".^$*+?{}()|")


def extract_literals(regex: str) -> List[str]:
    r"""
    Baseline 级 literal 抽取器，用于 REI-style bigram 统计 / mask 构造。

    设计要点：
    - 跳过字符类 [...] 内的内容；
    - 跳过量词 {m} / {m,n} 内的数字、逗号等（整体视为模式，不算字面）；
    - 对 \d \w \s \b \t \n 等“类/空白/边界”视为模式，不算字面；
    - 对 \1...\9 这类回溯引用视为模式，不算字面；
    - 对 \. \- \( \) 这类被转义的符号，视为对应的字面字符；
    - 其它元字符 . ^ $ * + ? ( ) | 仅作为分隔符，不进入 literal；
    - 最终只保留长度 >= 2 的 literal，用于 bigram。
    """
    literals: List[str] = []
    buf: List[str] = []
    i = 0
    n = len(regex)
    in_class = False  # 是否在 [...] 里

    while i < n:
        ch = regex[i]

        # ---------- 字符类 [ ... ] ----------
        if not in_class and ch == "[":
            if buf:
                literals.append("".join(buf))
                buf = []
            in_class = True
            i += 1
            continue

        if in_class:
            # 字符类内部，统统跳过（包括转义）
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "]":
                in_class = False
            i += 1
            continue

        # ---------- 量词 {m} / {m,n}：整体视为模式，跳过 ----------
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

        # ---------- 转义序列 ----------
        if ch == "\\":
            if i + 1 < n:
                nxt = regex[i + 1]

                # \d \D \w \W \s \S \b \B \t \n \r \f \v 等：都当作“模式”，不计入 literal
                if nxt in ("d", "D", "w", "W", "s", "S", "b", "B", "t", "n", "r", "f", "v"):
                    if buf:
                        literals.append("".join(buf))
                        buf = []
                    i += 2
                    continue

                # 回溯引用 \1..\9：也当作模式
                if nxt.isdigit():
                    if buf:
                        literals.append("".join(buf))
                        buf = []
                    i += 2
                    continue

                # 其它情况：视为“转义后的字面字符”，例如 \. \- \( 等
                buf.append(nxt)
                i += 2
                continue
            else:
                # 末尾孤立反斜杠，保守当作字面
                buf.append(ch)
                i += 1
                continue

        # ---------- 普通元字符：当作分隔符，终止当前 literal ----------
        if ch in META_CHARS:
            if buf:
                literals.append("".join(buf))
                buf = []
            i += 1
            continue

        # ---------- 普通字面字符 ----------
        buf.append(ch)
        i += 1

    if buf:
        literals.append("".join(buf))

    return [s for s in literals if len(s) >= 2]


def ngrams_of(s: str, n: int) -> List[str]:
    if not s or len(s) < n:
        return []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def build_required_mask(regex: str, mapping: Dict[str, int], k: int, n: int = 2) -> str:
    bits = ["0"] * k

    lits = extract_literals(regex)
    required = set()

    for lit in lits:
        for g in ngrams_of(lit, n):
            if g in mapping:
                required.add(g)

    for g in required:
        bits[mapping[g]] = "1"

    return "".join(bits)


def load_regexes(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = line.strip()
            if r:
                out.append(r)
    return out


def load_mapping(mapping_path: str) -> Tuple[int, Dict[str, int]]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    k = int(obj["k"])
    mapping = obj["mapping"]
    return k, mapping


def infer_table_name(dataset_key: str, k: int, schema: str = "public") -> str:
    # 注意：未加引号的标识符在 PG 会折叠为小写
    return f"{schema}.{dataset_key}_rei_{k}"


def infer_sig_col(workload: str, k: int) -> str:
    # 明确生成小写，避免歧义
    return f"sig_rei_{workload.lower()}_{k}"


def infer_mapping_path(mapping_dir: str, dataset_key: str, workload: str, k: int) -> str:
    return os.path.join(mapping_dir, f"{dataset_key}_{workload}_topk_{k}.json")


def infer_regex_path(regex_dir: str, dataset_key: str, workload: str) -> str:
    return os.path.join(regex_dir, f"{dataset_key}_{workload}.txt")


def normalize_regex_for_pg_param(regex: str, unescape_double_backslash: bool = True) -> str:
    """
    很多 workload 文件里的 regex 是按 SQL E'' 字面量写法输出的（例如 \\s、\\d、\\.）。
    但 psycopg2 参数传值不经过 SQL 字符串转义，因此应把 \\ 还原为 \，否则会失配。

    注意：如果你的 regex 确实需要匹配“字面反斜杠”，可以通过 --no_unescape_regex 关闭该行为。
    """
    if unescape_double_backslash and "\\\\" in regex:
        return regex.replace("\\\\", "\\")
    return regex


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset_key",
        required=True,
        choices=["wiki", "reddit", "lineitem", "imdb_name", "US_Accidents"],
    )
    ap.add_argument("--workload", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--k", type=int, default=64)

    ap.add_argument(
        "--regex_dir",
        default="/home/lyz/wd/LLM_RegularSearch/REI/workload/imdb_name",
    )
    ap.add_argument(
        "--mapping_dir",
        default="/home/lyz/wd/LLM_RegularSearch/REI/mappings",
    )

    ap.add_argument("--schema", default="public")
    ap.add_argument("--table", default="")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--text_col", default="text_content")
    ap.add_argument("--sig_col", default="")

    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--limit", type=int, default=10)

    ap.add_argument("--output_json", required=True)
    ap.add_argument(
        "--calc_candidates",
        action="store_true",
        help="额外统计候选数（多一次过滤计数扫描）",
    )

    # 新增：更贴近你当前实验的执行习惯
    ap.add_argument(
        "--case_insensitive",
        action="store_true",
        help="使用 ~*（大小写不敏感）而不是 ~",
    )
    ap.add_argument(
        "--work_mem",
        default="4MB",
        help="会话级 work_mem（例如 64MB/128MB/256MB），尽量避免 CTE 物化落盘",
    )
    ap.add_argument(
        "--jit_off",
        action="store_true",
        default=True,
        help="默认关闭 JIT，减少每条查询的固定开销（可用 --no-jit_off 关闭）",
    )
    ap.add_argument(
        "--no_unescape_regex",
        action="store_true",
        help="默认会把 regex 中的 \\\\ 还原成 \\；如需保留双反斜杠用于匹配字面反斜杠，可开启此选项",
    )

    args = ap.parse_args()

    # ---------- 推断文件路径 ----------
    mapping_path = infer_mapping_path(args.mapping_dir, args.dataset_key, args.workload, args.k)
    regex_path = infer_regex_path(args.regex_dir, args.dataset_key, args.workload)

    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"mapping not found: {mapping_path}")
    if not os.path.exists(regex_path):
        raise FileNotFoundError(f"regex list not found: {regex_path}")

    # ---------- 载入 mapping ----------
    k_m, mapping = load_mapping(mapping_path)
    if k_m != args.k:
        raise ValueError(f"k mismatch: args.k={args.k} but mapping file k={k_m}")

    # ---------- 载入 regex ----------
    regexes = load_regexes(regex_path)
    if args.limit > 0:
        regexes = regexes[:args.limit]

    # ---------- 推断表/列 ----------
    table = args.table.strip() or infer_table_name(args.dataset_key, args.k, args.schema)
    sig_col = args.sig_col.strip() or infer_sig_col(args.workload, args.k)

    # ---------- SQL：两阶段（MATERIALIZED）过滤 + regex 精配 ----------
    # 关键：强制先执行位过滤并物化候选，再对候选做 regex，避免 planner 重排导致全表先跑 regex。
    regex_op = "~*" if args.case_insensitive else "~"

    sql_match = f"""
    WITH candidates AS MATERIALIZED (
      SELECT {args.text_col} AS txt
      FROM {table}
      WHERE ({sig_col} & %s::bit({args.k})) = %s::bit({args.k})
    )
    SELECT count(*)
    FROM candidates
    WHERE candidates.txt {regex_op} %s;
    """

    sql_candidates = f"""
    SELECT count(*)
    FROM {table}
    WHERE ({sig_col} & %s::bit({args.k})) = %s::bit({args.k});
    """

    # ---------- 连接 PG ----------
    conn = psycopg2.connect(args.dsn)
    conn.autocommit = True
    cur = conn.cursor()

    # 会话参数（建议）
    # 1) 关闭 JIT：大量查询时能省下稳定固定开销
    if args.jit_off:
        cur.execute("SET jit = off;")

    # 2) 提高 work_mem：减少 candidates 物化时落盘（temp written）
    if args.work_mem:
        cur.execute(f"SET work_mem = '{args.work_mem}';")

    # 全表行数
    cur.execute(f"SELECT count(*) FROM {table};")
    total_rows = cur.fetchone()[0]

    results = []
    for idx, r0 in enumerate(regexes):
        # mask 由 literal bigram 构造，长度为 k 的 01 字符串
        mask = build_required_mask(r0, mapping, args.k, n=2)

        # 参数传值时，通常需要把 \\ 还原成 \
        r = normalize_regex_for_pg_param(r0, unescape_double_backslash=(not args.no_unescape_regex))

        t0 = time.perf_counter()
        cur.execute(sql_match, (mask, mask, r))
        matched = cur.fetchone()[0]
        t1 = time.perf_counter()

        row = {
            "idx": idx,
            "dataset": args.dataset_key,
            "workload": args.workload,
            "table": table,
            "sig_col": sig_col,
            "k": args.k,
            "ngram_n": 2,
            "regex": r0,  # 记录原始 regex（文件中的样子）
            "regex_sent": r,  # 记录实际送进 PG 的 regex（便于排查 \\ 问题）
            "mask_bits_set": mask.count("1"),
            "total_rows": total_rows,
            "matched": matched,
            "time_sec": (t1 - t0)*1000.0,  # 毫秒
            "regex_op": regex_op,
        }

        if args.calc_candidates:
            cur.execute(sql_candidates, (mask, mask))
            cand = cur.fetchone()[0]
            row["candidates"] = cand
            row["candidates_ratio"] = (cand / total_rows) if total_rows else 0.0

        results.append(row)

        if (idx + 1) % 50 == 0:
            print(f"[PROGRESS] {idx + 1}/{len(regexes)}")

    cur.close()
    conn.close()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset_key,
                "workload": args.workload,
                "table": table,
                "sig_col": sig_col,
                "k": args.k,
                "ngram_n": 2,
                "regex_list": regex_path,
                "mapping_file": mapping_path,
                "results": results,
                "session": {
                    "jit": "off" if args.jit_off else "on",
                    "work_mem": args.work_mem,
                    "regex_op": regex_op,
                    "unescape_double_backslash": (not args.no_unescape_regex),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] wrote results to {args.output_json}")


if __name__ == "__main__":
    main()

# python run_rei_pg_jsonl.py --dataset_key US_Accidents --workload A --k 64 --output_json ./result/us_accidents_A.json --calc_candidates