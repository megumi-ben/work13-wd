#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import psycopg2
except Exception as e:
    raise RuntimeError("Please install psycopg2 or psycopg2-binary: pip install psycopg2-binary") from e


# =============================================================================
# Tokenization (用于 seed_string 前 N 个 token 前缀)
# 与你之前 workload 代码一致：word / hyphen-word / 单个标点
# =============================================================================
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)


# =============================================================================
# JSONL sampling (reservoir sampling，适合大文件)
# =============================================================================
def reservoir_sample_jsonl(path: str, k: int, seed: int) -> Tuple[List[Dict[str, Any]], int]:
    """
    从 jsonl 中随机抽取 k 条（单遍扫描，内存 O(k)）
    返回: (samples, n_total)
    """
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    n_total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error at line {line_no}: {e}") from e

            n_total += 1
            if k <= 0:
                continue

            if len(reservoir) < k:
                reservoir.append(obj)
            else:
                j = rng.randrange(0, n_total)
                if j < k:
                    reservoir[j] = obj

    return reservoir, n_total


# =============================================================================
# SQL string helpers
# =============================================================================
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_ident(ident: str) -> str:
    """
    安全引用标识符，支持 schema-qualified: public.us_accidents
    仅允许 [A-Za-z_][A-Za-z0-9_]* 的片段，避免注入风险。
    """
    parts = ident.split(".")
    for p in parts:
        if not _IDENT_RE.fullmatch(p):
            raise ValueError(f"Invalid identifier part: {p!r} (from {ident!r})")
    return ".".join(f'"{p}"' for p in parts)


def normalize_backslashes(s: str) -> str:
    """
    归一化 workload 里可能存在的“多重反斜杠”：
    - '\\\\' 折叠为 '\\'，直到稳定
    """
    prev = None
    while prev != s:
        prev = s
        s = s.replace("\\\\", "\\")
    return s


def escape_for_pg_e_literal(s: str) -> str:
    """
    构造 E'...' 所需文本：
    - 单引号：'' 
    - 反斜杠：\\
    """
    s = s.replace("'", "''")
    s = s.replace("\\", "\\\\")
    return s


def escape_for_pg_standard_literal(s: str) -> str:
    """构造普通 '...' 字面量所需文本（只处理单引号）。"""
    return s.replace("'", "''")


def escape_like_pattern(s: str, escape_char: str = "\\") -> str:
    """
    将文本转换为 LIKE pattern 的“字面匹配”部分：
    - 转义 escape_char / % / _
    """
    out = []
    for ch in s:
        if ch == escape_char or ch == "%" or ch == "_":
            out.append(escape_char)
        out.append(ch)
    return "".join(out)


# =============================================================================
# prefix extraction: seed_string 前 N 个 token 对应的原始前缀子串
# =============================================================================
def seed_prefix_by_tokens(seed_string: str, n_tokens: int) -> str:
    """
    用 TOKEN_RE 在原字符串上定位 token span，然后返回前 n_tokens 个 token 覆盖的原始前缀子串。
    这样可以最大程度保留原始空格/标点布局，避免 join 后不一致。
    """
    if n_tokens <= 0:
        return ""
    matches = list(TOKEN_RE.finditer(seed_string))
    if not matches:
        return ""

    idx = min(n_tokens, len(matches)) - 1
    end_pos = matches[idx].end()
    prefix = seed_string[:end_pos]
    return prefix.strip()


# =============================================================================
# DB worker (thread-local connection)
# =============================================================================
_thread_local = threading.local()


def get_conn(args) -> "psycopg2.extensions.connection":
    conn = getattr(_thread_local, "conn", None)
    if conn is None or conn.closed != 0:
        conn = psycopg2.connect(
            dbname=args.dbname,
            user=args.user,
            password=args.password,
            host=args.host,
            port=args.port,
        )
        conn.autocommit = True
        _thread_local.conn = conn

        with conn.cursor() as cur:
            if args.statement_timeout_ms > 0:
                cur.execute("SET statement_timeout = %s;", (int(args.statement_timeout_ms),))
            if args.jit_off:
                cur.execute("SET jit = off;")
    return conn


def exec_count_sql(conn, sql: str) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    """
    执行 count(*) SQL，返回 (count, execution_time_ms, error)
    """
    try:
        with conn.cursor() as cur:
            t0 = time.perf_counter()
            cur.execute(sql)
            cnt = cur.fetchone()[0]
            t1 = time.perf_counter()
        return int(cnt), (t1 - t0) * 1000.0, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def run_one_item(args, item: Dict[str, Any]) -> Dict[str, Any]:
    raw_regex = item.get("regex")
    seed_string = item.get("seed_string")

    base = {
        "dataset": item.get("dataset"),
        "seed_id": item.get("seed_id"),
        "seed_string": seed_string,
        "regex": raw_regex,
        "regex_norm": None,
        "prefix_tokens": int(args.prefix_tokens),
        "prefix_string": None,
        "queries": [],
        "error": None,  # item-level error（字段缺失等）
    }

    if not isinstance(raw_regex, str):
        base["error"] = "missing_or_invalid_regex_field"
        return base
    if not isinstance(seed_string, str):
        base["error"] = "missing_or_invalid_seed_string_field"
        return base

    # 1) normalize regex, then embed as E'...'
    norm_regex = normalize_backslashes(raw_regex)
    base["regex_norm"] = norm_regex
    patt_sql = escape_for_pg_e_literal(norm_regex)

    # 2) seed prefix (first N tokens)
    prefix = seed_prefix_by_tokens(seed_string, int(args.prefix_tokens))
    base["prefix_string"] = prefix

    # 3) identifiers
    tbl = quote_ident(args.table)
    col = quote_ident(args.col)

    # derived tables
    tbl_trgm_name = args.table_trgm if args.table_trgm else f"{args.table}_trgm"
    tbl_prefix_name = args.table_prefix if args.table_prefix else f"{args.table}_prefix"
    tbl_trgm = quote_ident(tbl_trgm_name)
    tbl_prefix = quote_ident(tbl_prefix_name)

    # 4) build 3 SQLs
    sql_base = f"SELECT count(*) FROM {tbl} WHERE {col} ~ E'{patt_sql}';"
    sql_trgm = f"SELECT count(*) FROM {tbl_trgm} WHERE {col} ~ E'{patt_sql}';"

    # prefix LIKE 'prefix%' ESCAPE '\'
    like_lit = escape_like_pattern(prefix, "\\") + "%"
    like_lit_sql = escape_for_pg_standard_literal(like_lit)
    sql_prefix = (
        f"SELECT count(*) FROM {tbl_prefix} "
        f"WHERE {col} LIKE '{like_lit_sql}' ESCAPE '\\' "
        f"AND {col} ~ E'{patt_sql}';"
    )

    # 5) execute
    conn = get_conn(args)

    for kind, sql in [("base", sql_base), ("trgm", sql_trgm), ("prefix", sql_prefix)]:
        cnt, ms, err = exec_count_sql(conn, sql)
        base["queries"].append(
            {
                "kind": kind,
                "sql": sql,
                "execution_time_ms": ms,
                "count": cnt,
                "error": err,
            }
        )

    return base


# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to workload jsonl")
    ap.add_argument("--output", required=True, help="Path to output json")

    ap.add_argument("-k", "--k", type=int, default=90, help="Number of samples")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed")
    ap.add_argument("-w", "--workers", type=int, default=30, help="Number of threads/workers")

    ap.add_argument("--dbname", default="regex_db")
    ap.add_argument("--user", default="lyz")
    ap.add_argument("--password", default="123456")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default="5000")

    # as you asked: --table --col --id (id 目前不参与 count(*) SQL，但保留参数位以兼容你现有调用)
    ap.add_argument("--table", required=True, help="Base table name, e.g. us_accidents")
    ap.add_argument("--col", required=True, help="Text column name, e.g. description")
    ap.add_argument("--id", default="id", help="ID column name (kept for interface compatibility)")

    # optional overrides
    ap.add_argument("--table_trgm", default=None, help="Override trgm table name (default: <table>_trgm)")
    ap.add_argument("--table_prefix", default=None, help="Override prefix table name (default: <table>_prefix)")
    ap.add_argument("--prefix_tokens", type=int, default=6, help="Use first N tokens from seed_string for LIKE prefix")

    ap.add_argument("--statement_timeout_ms", type=int, default=0, help="Per-statement timeout in ms (0=off)")
    ap.add_argument("--jit_off", action="store_true", help="SET jit=off in each session")

    args = ap.parse_args()

    # 1) sample
    samples, n_total = reservoir_sample_jsonl(args.input, args.k, args.seed)
    if not samples:
        raise RuntimeError(f"No rows sampled from input jsonl: {args.input}")

    # 2) run in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Dict[str, Any]] = []
    t_all0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one_item, args, it) for it in samples]
        for fut in as_completed(futs):
            results.append(fut.result())

    t_all1 = time.perf_counter()

    # stable order: seed_id first, else keep
    results.sort(key=lambda r: (r.get("seed_id") is None, r.get("seed_id", 0)))

    out = {
        "input": args.input,
        "output": args.output,
        "n_total": int(n_total),
        "k_sampled": int(len(samples)),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "dbname": args.dbname,
        "table": args.table,
        "col": args.col,
        "id": args.id,
        "table_trgm": args.table_trgm if args.table_trgm else f"{args.table}_trgm",
        "table_prefix": args.table_prefix if args.table_prefix else f"{args.table}_prefix",
        "prefix_tokens": int(args.prefix_tokens),
        "statement_timeout_ms": int(args.statement_timeout_ms),
        "jit_off": bool(args.jit_off),
        "wall_time_ms_total": (t_all1 - t_all0) * 1000.0,
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 3) quick summary
    # 以 query 维度统计成功率与延迟分位数
    all_q = []
    for r in results:
        for q in r.get("queries", []):
            if q.get("error") is None and isinstance(q.get("execution_time_ms"), (int, float)):
                all_q.append(q["execution_time_ms"])
    if all_q:
        all_q.sort()
        p50 = all_q[int(0.50 * (len(all_q) - 1))]
        p90 = all_q[int(0.90 * (len(all_q) - 1))]
        avg = sum(all_q) / len(all_q)
        print(f"Done. queries_ok={len(all_q)}  avg_ms={avg:.3f}  p50_ms={p50:.3f}  p90_ms={p90:.3f}")
        print(f"Output written to: {args.output}")
    else:
        print("Done. No successful queries; check output JSON for errors.")
        print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
