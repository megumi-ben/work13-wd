#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
from psycopg2 import sql
from tqdm import tqdm


# =========================
# Fixed DB connection params
# =========================
DB_PARAMS = {
    "host": "localhost",
    "port": 5000,
    "dbname": "regex_db",
    "user": "lyz",
    "password": "123456",
}


# =========================
# Schema helpers
# =========================

def table_exists(table_name: str, connection_params: dict, schema: str = "public") -> bool:
    try:
        with psycopg2.connect(**connection_params) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                    LIMIT 1;
                    """,
                    (schema, table_name)
                )
                return cur.fetchone() is not None
    except Exception:
        return False


def table_has_column(table_name: str, column_name: str, connection_params: dict, schema: str = "public") -> bool:
    try:
        with psycopg2.connect(**connection_params) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema=%s AND table_name=%s AND column_name=%s
                    LIMIT 1;
                    """,
                    (schema, table_name, column_name),
                )
                return cur.fetchone() is not None
    except Exception:
        return False


def get_text_column_name(table_name: str, connection_params: dict) -> Optional[str]:
    """
    Identify the main text column. Excludes id/tokens.
    Prefers common names if present.
    """
    preferred = {"text_content", "content", "body", "comment", "name", "text", "title", "primaryname"}
    try:
        with psycopg2.connect(**connection_params) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = %s
                      AND column_name NOT IN ('id', 'ID', 'tokens')
                    ORDER BY ordinal_position;
                    """,
                    (table_name,)
                )
                cols = [r[0] for r in cur.fetchall()]
        if not cols:
            return None
        for c in cols:
            if c.lower() in preferred:
                return c
        return cols[0]
    except Exception as e:
        print(f"[get_text_column_name] Error for table={table_name}: {e}")
        return None


def get_table_rowcount(table_name: str, connection_params: dict,
                       timeout_ms: Optional[int]) -> Optional[int]:
    """
    Return total rows of a table.
    """
    tbl = sql.Identifier(table_name)
    query = sql.SQL("SELECT count(*) FROM {tbl};").format(tbl=tbl)
    try:
        with psycopg2.connect(**connection_params) as conn:
            maybe_set_statement_timeout(conn, timeout_ms)
            with conn.cursor() as cur:
                cur.execute(query)
                return int(cur.fetchone()[0])
    except Exception as e:
        print(f"[get_table_rowcount] Error table={table_name}: {e}")
        return None


# =========================
# Misc helpers
# =========================

def maybe_set_statement_timeout(conn, timeout_ms: Optional[int]):
    if not timeout_ms or timeout_ms <= 0:
        return
    with conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout = %s;", (timeout_ms,))


def rows_to_values(rows: List[Tuple[Any]]) -> List[Any]:
    return [r[0] for r in rows]


def compute_recall(gold: List[Any], pred: List[Any]) -> Optional[float]:
    """
    Set-based recall.
    If gold empty, define recall=1.0 to avoid divide-by-zero.
    """
    if gold is None:
        return None
    if len(gold) == 0:
        return 1.0
    gset = set(gold)
    pset = set(pred) if pred is not None else set()
    tp = len(gset.intersection(pset))
    return round(tp / len(gset), 6)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def workload_tag_from_file(fname: str) -> str:
    if fname.endswith(".jsonl"):
        return fname[:-6]
    if fname.endswith(".json"):
        return fname[:-5]
    return fname


def strip_values(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k != "values"}


def latency_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_items = [x for x in items if x.get("ok")]
    durs = [x.get("t_total_ms") for x in ok_items if x.get("t_total_ms") is not None]
    if not durs:
        return {"n_all": len(items), "n_ok": len(ok_items), "p50_ms": None, "p90_ms": None, "avg_ms": None}
    durs_sorted = sorted(durs)
    p50 = durs_sorted[int(0.50 * (len(durs_sorted) - 1))]
    p90 = durs_sorted[int(0.90 * (len(durs_sorted) - 1))]
    avg = round(sum(durs_sorted) / len(durs_sorted), 2)
    return {"n_all": len(items), "n_ok": len(ok_items), "p50_ms": p50, "p90_ms": p90, "avg_ms": avg}


def build_prefix_from_seed(seed: Optional[str], k: int) -> str:
    if not seed:
        return ""
    seed = str(seed)
    if k is None or k <= 0:
        return seed
    return seed if len(seed) <= k else seed[:k]


# =========================
# Pattern helpers: LIKE escape
# =========================

def clean_like_pattern(s: str) -> str:
    """
    Escape LIKE meta-chars for literal substring matching.
    We will use: LIKE %s ESCAPE '\\'
    """
    if s is None:
        return ""
    s = s.replace("\\", "\\\\")  # escape backslash first for LIKE
    s = s.replace("%", r"\%")
    s = s.replace("_", r"\_")
    return s


# =========================
# Regex literal extraction (used by Tokgist)
# =========================

META_CHARS = set(".^$*+?{}()|")

def extract_literals(regex: str, min_len: int = 3) -> List[str]:
    r"""
    Conservative regex literal extractor (heuristic):
    - Treat escaped "\\x" as literal character (e.g., "\-" -> "-", "\." -> ".")
    - Skip character classes [...] entirely
    - Skip quantifiers {m,n} entirely
    - Skip semantic escapes like \d \w \s (and uppercase) as separators
    - Meta chars . ^ $ * + ? ( ) | and braces are separators
    """
    literals: List[str] = []
    buf: List[str] = []
    i = 0
    n = len(regex)
    in_class = False

    def flush_buf():
        nonlocal buf
        if buf:
            literals.append("".join(buf))
            buf = []

    while i < n:
        ch = regex[i]

        # Enter char class
        if not in_class and ch == "[":
            flush_buf()
            in_class = True
            i += 1
            continue

        # Inside char class: skip
        if in_class:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "]":
                in_class = False
            i += 1
            continue

        # Quantifier {m,n}: skip block
        if ch == "{":
            flush_buf()
            i += 1
            while i < n:
                c2 = regex[i]
                if c2 == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c2 == "}":
                    i += 1
                    break
                i += 1
            continue

        # Escapes
        if ch == "\\":
            if i + 1 < n:
                nxt = regex[i + 1]

                # semantic escapes: separator
                if nxt in ("d", "D", "w", "W", "s", "S"):
                    flush_buf()
                    i += 2
                    continue

                # anchors/boundaries: separator
                if nxt in ("b", "B", "A", "Z", "z"):
                    flush_buf()
                    i += 2
                    continue

                # treat as escaped literal
                buf.append(nxt)
                i += 2
                continue

            # trailing backslash
            buf.append(ch)
            i += 1
            continue

        # Meta chars as separators
        if ch in META_CHARS:
            flush_buf()
            i += 1
            continue

        # ordinary char
        buf.append(ch)
        i += 1

    flush_buf()

    min_len = max(1, int(min_len))
    out = [s for s in literals if len(s) >= min_len and s.strip() != ""]
    return out


# =========================
# Tokgist tokenization
# =========================

TOKEN_RE = re.compile(r"[a-z0-9_-]+")

def tokens_from_literals(literals: List[str], min_token_len: int = 3) -> List[str]:
    """
    Convert extracted literals -> tokens:
      lower + keep [a-z0-9_-]+
    """
    toks: List[str] = []
    for lit in literals:
        for t in TOKEN_RE.findall(lit.lower()):
            if len(t) >= min_token_len:
                toks.append(t)
    # stable dedup then sort by length desc
    seen = set()
    uniq = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    uniq.sort(key=lambda x: (-len(x), x))
    return uniq


# =========================
# Token table build helpers
# =========================

def ensure_tok_table(base: str,
                     src_table: str,
                     tok_table: str,
                     text_col: str,
                     conn_params: dict,
                     timeout_ms: Optional[int],
                     rebuild: bool = False) -> bool:
    """
    Create tok table (id, text_col, tokens text[]) from src_table.
    Does not modify src_table.

    tokens computed as:
      array_remove(
        regexp_split_to_array(
          regexp_replace(lower(text_col), '[^a-z0-9_-]+', ' ', 'g'),
          '\s+'
        ),
        ''
      )

    Creates gist(tokens) index and ANALYZE.
    """
    if rebuild and table_exists(tok_table, conn_params):
        try:
            with psycopg2.connect(**conn_params) as conn:
                maybe_set_statement_timeout(conn, timeout_ms)
                with conn.cursor() as cur:
                    cur.execute(sql.SQL("DROP TABLE IF EXISTS {t};").format(t=sql.Identifier(tok_table)))
            print(f"[TokTable] Dropped existing: {tok_table}")
        except Exception as e:
            print(f"[TokTable] Drop failed for {tok_table}: {e}")
            return False

    if table_exists(tok_table, conn_params):
        if table_has_column(tok_table, "tokens", conn_params):
            return True
        print(f"[TokTable] Table exists but missing tokens column: {tok_table}")
        return False

    try:
        with psycopg2.connect(**conn_params) as conn:
            maybe_set_statement_timeout(conn, timeout_ms)
            with conn.cursor() as cur:
                create_sql = sql.SQL("""
                    CREATE TABLE {tok} AS
                    SELECT
                      id,
                      {col} AS {col_alias},
                      array_remove(
                        regexp_split_to_array(
                          regexp_replace(lower({col}), '[^a-z0-9_-]+', ' ', 'g'),
                          '\\s+'
                        ),
                        ''
                      ) AS tokens
                    FROM {src};
                """).format(
                    tok=sql.Identifier(tok_table),
                    src=sql.Identifier(src_table),
                    col=sql.Identifier(text_col),
                    col_alias=sql.Identifier(text_col),
                )
                cur.execute(create_sql)

                idx_name = f"{tok_table}_tokens_gist"
                idx_sql = sql.SQL("CREATE INDEX {idx} ON {tok} USING gist (tokens);").format(
                    idx=sql.Identifier(idx_name),
                    tok=sql.Identifier(tok_table)
                )
                cur.execute(idx_sql)

                cur.execute(sql.SQL("ANALYZE {tok};").format(tok=sql.Identifier(tok_table)))

        print(f"[TokTable] Created: {tok_table} + gist(tokens) index")
        return True
    except Exception as e:
        print(f"[TokTable] Create failed for {tok_table}: {e}")
        return False


# =========================
# Query runners
# =========================

def _regex_op_sql() -> sql.SQL:
    """Force case-sensitive regex operator ONLY."""
    return sql.SQL("~")


def run_seqscan(regex_pattern: str, table: str, col: str, conn_params: dict,
                timeout_ms: Optional[int],
                table_nrows: Optional[int],
                t_infer_ms: float = 0.0) -> Dict[str, Any]:
    tbl = sql.Identifier(table)
    column = sql.Identifier(col)
    op = _regex_op_sql()

    query = sql.SQL(
        "SELECT {col} FROM {tbl} WHERE {col} {op} %s;"
    ).format(col=column, tbl=tbl, op=op)

    start = time.time()
    try:
        with psycopg2.connect(**conn_params) as conn:
            maybe_set_statement_timeout(conn, timeout_ms)
            with conn.cursor() as cur:
                cur.execute(query, (regex_pattern,))
                rows = cur.fetchall()
        end = time.time()
        dur_ms = round((end - start) * 1000, 2)
        return {
            "ok": True,
            "t_infer_ms": round(float(t_infer_ms), 2),
            "t_probe_ms": 0.0,
            "t_total_ms": dur_ms,
            "t_verify_ms": dur_ms,
            "t_verify_is_derived": False,
            "n_candidates": int(table_nrows) if table_nrows is not None else None,
            "n_matches": len(rows),
            "duration_ms": dur_ms,
            "values": rows_to_values(rows),
        }
    except Exception as e:
        end = time.time()
        dur_ms = round((end - start) * 1000, 2)
        return {
            "ok": False,
            "t_infer_ms": round(float(t_infer_ms), 2),
            "t_probe_ms": 0.0,
            "t_total_ms": dur_ms,
            "t_verify_ms": dur_ms,
            "t_verify_is_derived": False,
            "n_candidates": int(table_nrows) if table_nrows is not None else None,
            "n_matches": 0,
            "duration_ms": dur_ms,
            "error": str(e),
            "values": [],
        }


def run_trgm(regex_pattern: str, table: str, col: str, conn_params: dict,
             limit_k: int,
             timeout_ms: Optional[int],
             table_nrows: Optional[int],
             t_infer_ms: float = 0.0) -> Dict[str, Any]:
    """
    TRGM baseline: WHERE col ~ %s LIMIT k
    """
    tbl = sql.Identifier(table)
    column = sql.Identifier(col)
    op = _regex_op_sql()

    query = sql.SQL(
        "SELECT {col} FROM {tbl} WHERE {col} {op} %s LIMIT {lim};"
    ).format(col=column, tbl=tbl, op=op, lim=sql.Literal(limit_k))

    start = time.time()
    try:
        with psycopg2.connect(**conn_params) as conn:
            maybe_set_statement_timeout(conn, timeout_ms)
            with conn.cursor() as cur:
                cur.execute(query, (regex_pattern,))
                rows = cur.fetchall()
        end = time.time()
        dur_ms = round((end - start) * 1000, 2)
        return {
            "ok": True,
            "t_infer_ms": round(float(t_infer_ms), 2),
            "t_probe_ms": 0.0,
            "t_total_ms": dur_ms,
            "t_verify_ms": dur_ms,
            "t_verify_is_derived": False,
            "n_candidates": int(table_nrows) if table_nrows is not None else None,
            "n_matches": len(rows),
            "limited": True,
            "limit_k": int(limit_k),
            "duration_ms": dur_ms,
            "values": rows_to_values(rows),
        }
    except Exception as e:
        end = time.time()
        dur_ms = round((end - start) * 1000, 2)
        return {
            "ok": False,
            "t_infer_ms": round(float(t_infer_ms), 2),
            "t_probe_ms": 0.0,
            "t_total_ms": dur_ms,
            "t_verify_ms": dur_ms,
            "t_verify_is_derived": False,
            "n_candidates": int(table_nrows) if table_nrows is not None else None,
            "n_matches": 0,
            "limited": True,
            "limit_k": int(limit_k),
            "duration_ms": dur_ms,
            "error": str(e),
            "values": [],
        }


def run_gist(regex_pattern: str, table: str, col: str, conn_params: dict,
            limit_k: int,
            timeout_ms: Optional[int],
            table_nrows: Optional[int],
            t_infer_ms: float = 0.0) -> Dict[str, Any]:
    """
    gist+pg_trgm baseline table (e.g., *_gist).
    Query same as TRGM.
    """
    return run_trgm(
        regex_pattern=regex_pattern,
        table=table,
        col=col,
        conn_params=conn_params,
        limit_k=limit_k,
        timeout_ms=timeout_ms,
        table_nrows=table_nrows,
        t_infer_ms=t_infer_ms,
    )


def run_prefix(regex_pattern: str, prefix_string: str,
               table: str, col: str, conn_params: dict,
               timeout_ms: Optional[int],
               table_nrows: Optional[int],
               t_infer_ms: float = 0.0) -> Dict[str, Any]:
    """
    Prefix method (case-sensitive only):
    - probe: COUNT(*) with LIKE 'prefix%'
    - total: SELECT col with LIKE + regex
    - verify derived: max(0, total - probe)
    """
    tbl = sql.Identifier(table)
    column = sql.Identifier(col)
    op = _regex_op_sql()

    safe_prefix = clean_like_pattern(prefix_string or "")
    like_pattern = f"{safe_prefix}%"

    probe_query = sql.SQL(
        "SELECT count(*) FROM {tbl} WHERE {col} LIKE %s ESCAPE '\\';"
    ).format(tbl=tbl, col=column)

    total_query = sql.SQL(
        "SELECT {col} FROM {tbl} "
        "WHERE {col} LIKE %s ESCAPE '\\' "
        "AND {col} {op} %s;"
    ).format(tbl=tbl, col=column, op=op)

    try:
        with psycopg2.connect(**conn_params) as conn:
            maybe_set_statement_timeout(conn, timeout_ms)
            with conn.cursor() as cur:
                t0 = time.time()
                cur.execute(probe_query, (like_pattern,))
                n_cand = int(cur.fetchone()[0])
                t1 = time.time()
                t_probe_ms = round((t1 - t0) * 1000, 2)

                t2 = time.time()
                cur.execute(total_query, (like_pattern, regex_pattern))
                rows = cur.fetchall()
                t3 = time.time()
                t_total_ms = round((t3 - t2) * 1000, 2)

        t_verify_raw = round(t_total_ms - t_probe_ms, 2)
        t_verify_ms = round(max(0.0, t_verify_raw), 2)

        return {
            "ok": True,
            "t_infer_ms": round(float(t_infer_ms), 2),
            "t_probe_ms": t_probe_ms,
            "t_total_ms": t_total_ms,
            "t_verify_ms": t_verify_ms,
            "t_verify_ms_raw": t_verify_raw,
            "t_verify_is_derived": True,
            "n_candidates": n_cand,
            "n_matches": len(rows),
            "like_pattern_used": like_pattern,
            "duration_ms": t_total_ms,
            "values": rows_to_values(rows),
            "table_nrows": int(table_nrows) if table_nrows is not None else None,
        }
    except Exception as e:
        return {
            "ok": False,
            "t_infer_ms": round(float(t_infer_ms), 2),
            "t_probe_ms": 0.0,
            "t_total_ms": 0.0,
            "t_verify_ms": 0.0,
            "t_verify_is_derived": True,
            "n_candidates": None,
            "n_matches": 0,
            "like_pattern_used": like_pattern,
            "error": str(e),
            "values": [],
            "table_nrows": int(table_nrows) if table_nrows is not None else None,
        }


def run_tokgist(regex_pattern: str, tokens: List[str],
               table: str, text_col: str, conn_params: dict,
               limit_k: Optional[int],
               timeout_ms: Optional[int],
               table_nrows: Optional[int],
               t_infer_ms: float = 0.0) -> Dict[str, Any]:
    """
    Token-array baseline:
      probe: COUNT(*) WHERE tokens @> %s::text[]
      total: SELECT text_col WHERE tokens @> %s::text[] AND text_col ~ regex (optional LIMIT)
      verify: derived = total - probe
    """
    if not tokens:
        return run_seqscan(regex_pattern, table, text_col, conn_params, timeout_ms, table_nrows, t_infer_ms)

    tbl = sql.Identifier(table)
    col = sql.Identifier(text_col)
    op = _regex_op_sql()

    probe_query = sql.SQL(
        "SELECT count(*) FROM {tbl} WHERE tokens @> %s::text[];"
    ).format(tbl=tbl)

    total_query = sql.SQL(
        "SELECT {col} FROM {tbl} WHERE tokens @> %s::text[] AND {col} {op} %s"
    ).format(col=col, tbl=tbl, op=op)

    if limit_k and limit_k > 0:
        total_query += sql.SQL(" LIMIT {lim}").format(lim=sql.Literal(int(limit_k)))

    try:
        with psycopg2.connect(**conn_params) as conn:
            maybe_set_statement_timeout(conn, timeout_ms)
            with conn.cursor() as cur:
                t0 = time.time()
                cur.execute(probe_query, (tokens,))
                n_cand = int(cur.fetchone()[0])
                t1 = time.time()
                t_probe_ms = round((t1 - t0) * 1000, 2)

                t2 = time.time()
                cur.execute(total_query, (tokens, regex_pattern))
                rows = cur.fetchall()
                t3 = time.time()
                t_total_ms = round((t3 - t2) * 1000, 2)

        t_verify_raw = round(t_total_ms - t_probe_ms, 2)
        t_verify_ms = round(max(0.0, t_verify_raw), 2)

        return {
            "ok": True,
            "t_infer_ms": round(float(t_infer_ms), 2),
            "t_probe_ms": t_probe_ms,
            "t_total_ms": t_total_ms,
            "t_verify_ms": t_verify_ms,
            "t_verify_ms_raw": t_verify_raw,
            "t_verify_is_derived": True,
            "n_candidates": n_cand,
            "n_matches": len(rows),
            "tok_k": len(tokens),
            "tok_tokens": tokens,
            "limited": bool(limit_k and limit_k > 0),
            "limit_k": int(limit_k) if (limit_k and limit_k > 0) else None,
            "duration_ms": t_total_ms,
            "values": rows_to_values(rows),
            "table_nrows": int(table_nrows) if table_nrows is not None else None,
        }
    except Exception as e:
        return {
            "ok": False,
            "t_infer_ms": round(float(t_infer_ms), 2),
            "t_probe_ms": 0.0,
            "t_total_ms": 0.0,
            "t_verify_ms": 0.0,
            "t_verify_is_derived": True,
            "n_candidates": None,
            "n_matches": 0,
            "tok_k": len(tokens),
            "tok_tokens": tokens,
            "limited": bool(limit_k and limit_k > 0),
            "limit_k": int(limit_k) if (limit_k and limit_k > 0) else None,
            "duration_ms": 0.0,
            "error": str(e),
            "values": [],
            "table_nrows": int(table_nrows) if table_nrows is not None else None,
        }


# =========================
# Workload IO
# =========================

def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if limit is not None and len(data) >= limit:
                break
    return data


def discover_workload_files(root: str) -> List[Tuple[str, str]]:
    """
    Supports BOTH layouts:

    Layout 1 (single dataset):
      root/
        workload_A.jsonl
        workload_B.jsonl
        ...

      -> dataset_name = basename(root)

    Layout 2 (multi dataset):
      root/
        imdb_name/
          workload_A.jsonl ...
        wiki/
          workload_A.jsonl ...

      -> dataset_name = subdir name
    """
    pairs: List[Tuple[str, str]] = []

    # Layout 1: root directly contains workload files
    direct = []
    for w in ["A", "B", "C", "D"]:
        fp = os.path.join(root, f"workload_{w}.jsonl")
        if os.path.isfile(fp):
            direct.append(fp)
    if direct:
        ds = os.path.basename(os.path.normpath(root))
        for fp in direct:
            pairs.append((ds, fp))
        return pairs

    # Layout 2: root contains dataset subdirs
    for name in sorted(os.listdir(root)):
        ds_path = os.path.join(root, name)
        if not os.path.isdir(ds_path):
            continue
        for w in ["A", "B", "C", "D"]:
            fp = os.path.join(ds_path, f"workload_{w}.jsonl")
            if os.path.isfile(fp):
                pairs.append((name, fp))
    return pairs


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Run seqscan/trgm/prefix/gist + Tokgist baselines with recall + candidates + timing fields. "
                    "This version forces case-sensitive regex operator '~' only."
    )
    parser.add_argument("--workload_root", type=str, default="/home/lyz/wd/LLM_RegularSearch/v2_data/workload/")
    parser.add_argument("--output_root", type=str, default="/home/lyz/wd/LLM_RegularSearch/v2_data/Res/run_workload/")
    parser.add_argument("--limit", type=int, default=None, help="Limit records per workload file.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout_ms", type=int, default=0)
    parser.add_argument("--limit_k", type=int, default=1024, help="LIMIT k for TRGM/gist baselines")

    # Prefix: seed_string -> prefix_string
    parser.add_argument(
        "--prefix_lit_k",
        type=int,
        default=0,
        help="Construct prefix_string from seed_string[:k]. 0 means use full seed_string."
    )

    # Tokgist
    parser.add_argument("--enable_tokgist", action="store_true", help="Enable Token-array baseline (tokens @> ...)")
    parser.add_argument("--tok_k", type=int, default=3, help="Top-K tokens for Tokgist")
    parser.add_argument("--tok_min_token_len", type=int, default=3, help="Min token length for Tokgist tokenization")
    parser.add_argument("--tok_limit_k", type=int, default=0, help="Optional LIMIT k for Tokgist total query (0=unlimited)")
    parser.add_argument("--tok_table_suffix", type=str, default="_tok_gist", help="Suffix for tok tables, default: _tok_gist")
    parser.add_argument("--build_tok_tables", action="store_true",
                        help="Auto-create tok tables + gist(tokens) index if missing (does NOT modify existing tables)")
    parser.add_argument("--rebuild_tok_tables", action="store_true",
                        help="Drop + rebuild tok tables before running (DANGEROUS but still does NOT touch base tables)")

    args = parser.parse_args()

    conn_params = DB_PARAMS
    timeout_ms = args.timeout_ms if args.timeout_ms and args.timeout_ms > 0 else None

    ensure_dir(args.output_root)

    workload_pairs = discover_workload_files(args.workload_root)
    if not workload_pairs:
        print(f"No workload_*.jsonl found under {args.workload_root}")
        return

    regex_op_str = "~"  # forced

    for dataset_name, workload_fp in workload_pairs:
        base = dataset_name.lower()

        workload_file = os.path.basename(workload_fp)          # workload_A.jsonl
        workload_json = workload_file.replace(".jsonl", ".json")
        workload_tag = workload_tag_from_file(workload_file)

        # Physical tables
        tbl_base = base
        tbl_trgm = f"{base}_trgm"
        tbl_pref = f"{base}_prefix"
        tbl_gist = f"{base}_gist"  # optional
        tbl_tok = f"{base}{args.tok_table_suffix}"  # e.g., imdb_name_tok_gist

        # Verify base table exists
        if not table_exists(tbl_base, conn_params):
            print(f"[Skip] base table not found: {tbl_base}")
            continue

        # Infer text col (prefer from base)
        col_base = get_text_column_name(tbl_base, conn_params)
        if not col_base:
            print(f"[Skip] Cannot infer text column for base table: {tbl_base}")
            continue

        # Derived tables existence
        trgm_available = table_exists(tbl_trgm, conn_params)
        pref_available = table_exists(tbl_pref, conn_params)
        gist_available = table_exists(tbl_gist, conn_params)

        # Tok table: may be built
        tok_available = table_exists(tbl_tok, conn_params) and table_has_column(tbl_tok, "tokens", conn_params)

        # If requested, build tok table if missing (or rebuild)
        if args.enable_tokgist and args.build_tok_tables:
            if args.rebuild_tok_tables or (not tok_available):
                ok = ensure_tok_table(
                    base=base,
                    src_table=tbl_base,
                    tok_table=tbl_tok,
                    text_col=col_base,
                    conn_params=conn_params,
                    timeout_ms=timeout_ms,
                    rebuild=args.rebuild_tok_tables
                )
                tok_available = ok and table_exists(tbl_tok, conn_params) and table_has_column(tbl_tok, "tokens", conn_params)

        # Tok table text col (should match base col name, but still infer safely)
        col_tok = None
        if args.enable_tokgist and tok_available:
            col_tok = get_text_column_name(tbl_tok, conn_params)
            if not col_tok:
                col_tok = col_base

        records = load_jsonl(workload_fp, limit=args.limit)
        if not records:
            print(f"[Skip] Empty workload file: {workload_fp}")
            continue

        print(f"\n=== Dataset={dataset_name}  Workload={workload_file}  "
              f"BaseCol={col_base}  N={len(records)}  RegexOp={regex_op_str} ===")
        if trgm_available:
            print(f"    table: {tbl_trgm} (trgm)")
        if gist_available:
            print(f"    table: {tbl_gist} (gist_trgm)")
        if pref_available:
            print(f"    table: {tbl_pref} (prefix)")
        if args.enable_tokgist:
            print(f"    table: {tbl_tok} (tok_gist) available={tok_available}")

        # Prepare output dirs
        dir_base = os.path.join(args.output_root, tbl_base)
        dir_trgm = os.path.join(args.output_root, tbl_trgm)
        dir_pref = os.path.join(args.output_root, tbl_pref)
        dir_gist = os.path.join(args.output_root, tbl_gist)
        dir_tok = os.path.join(args.output_root, tbl_tok)

        ensure_dir(dir_base)
        if trgm_available:
            ensure_dir(dir_trgm)
        if pref_available:
            ensure_dir(dir_pref)
        if gist_available:
            ensure_dir(dir_gist)
        if args.enable_tokgist and tok_available:
            ensure_dir(dir_tok)

        # Cache table sizes
        nrows_cache: Dict[str, Optional[int]] = {}
        for tname in [tbl_base, tbl_trgm, tbl_pref, tbl_gist, tbl_tok]:
            if table_exists(tname, conn_params):
                nrows_cache[tname] = get_table_rowcount(tname, conn_params, timeout_ms)
            else:
                nrows_cache[tname] = None

        # -------------------------
        # Run groups
        # -------------------------

        def run_group_seqscan() -> Dict[int, Dict[str, Any]]:
            results: Dict[int, Dict[str, Any]] = {}

            def task(idx: int, rec: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
                t_infer = float(rec.get("t_infer_ms", 0.0) or 0.0)
                return idx, run_seqscan(
                    rec["regex"], tbl_base, col_base, conn_params, timeout_ms,
                    table_nrows=nrows_cache.get(tbl_base),
                    t_infer_ms=t_infer
                )

            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = [ex.submit(task, i, rec) for i, rec in enumerate(records)]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Group=seqscan({tbl_base})"):
                    idx, r = fut.result()
                    results[idx] = r
            return results

        def run_group_trgm() -> Dict[int, Dict[str, Any]]:
            if not trgm_available:
                return {}
            results: Dict[int, Dict[str, Any]] = {}

            def task(idx: int, rec: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
                t_infer = float(rec.get("t_infer_ms", 0.0) or 0.0)
                return idx, run_trgm(
                    rec["regex"], tbl_trgm, col_base, conn_params, args.limit_k, timeout_ms,
                    table_nrows=nrows_cache.get(tbl_trgm),
                    t_infer_ms=t_infer
                )

            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = [ex.submit(task, i, rec) for i, rec in enumerate(records)]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Group=trgm({tbl_trgm})"):
                    idx, r = fut.result()
                    results[idx] = r
            return results

        def run_group_gist() -> Dict[int, Dict[str, Any]]:
            if not gist_available:
                return {}
            results: Dict[int, Dict[str, Any]] = {}

            def task(idx: int, rec: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
                t_infer = float(rec.get("t_infer_ms", 0.0) or 0.0)
                return idx, run_gist(
                    rec["regex"], tbl_gist, col_base, conn_params, args.limit_k, timeout_ms,
                    table_nrows=nrows_cache.get(tbl_gist),
                    t_infer_ms=t_infer
                )

            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = [ex.submit(task, i, rec) for i, rec in enumerate(records)]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Group=gist({tbl_gist})"):
                    idx, r = fut.result()
                    results[idx] = r
            return results

        def run_group_prefix() -> Dict[int, Dict[str, Any]]:
            if not pref_available:
                return {}
            results: Dict[int, Dict[str, Any]] = {}

            def task(idx: int, rec: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
                t_infer = float(rec.get("t_infer_ms", 0.0) or 0.0)
                seed = rec.get("seed_string", "")
                prefix = build_prefix_from_seed(seed, args.prefix_lit_k)
                return idx, run_prefix(
                    rec["regex"], prefix,
                    tbl_pref, col_base, conn_params, timeout_ms,
                    table_nrows=nrows_cache.get(tbl_pref),
                    t_infer_ms=t_infer
                )

            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = [ex.submit(task, i, rec) for i, rec in enumerate(records)]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Group=prefix({tbl_pref})"):
                    idx, r = fut.result()
                    results[idx] = r
            return results

        def run_group_tokgist() -> Dict[int, Dict[str, Any]]:
            if not (args.enable_tokgist and tok_available and col_tok):
                return {}
            results: Dict[int, Dict[str, Any]] = {}

            tok_limit = args.tok_limit_k if args.tok_limit_k and args.tok_limit_k > 0 else None

            def task(idx: int, rec: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
                t_infer = float(rec.get("t_infer_ms", 0.0) or 0.0)
                lits = extract_literals(rec["regex"], min_len=3)
                toks = tokens_from_literals(lits, min_token_len=args.tok_min_token_len)
                top_toks = toks[:max(0, args.tok_k)]
                return idx, run_tokgist(
                    regex_pattern=rec["regex"],
                    tokens=top_toks,
                    table=tbl_tok,
                    text_col=col_tok,
                    conn_params=conn_params,
                    limit_k=tok_limit,
                    timeout_ms=timeout_ms,
                    table_nrows=nrows_cache.get(tbl_tok),
                    t_infer_ms=t_infer
                )

            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = [ex.submit(task, i, rec) for i, rec in enumerate(records)]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Group=tokgist({tbl_tok})"):
                    idx, r = fut.result()
                    results[idx] = r
            return results

        # Run all groups
        seq_res = run_group_seqscan()
        trgm_res = run_group_trgm()
        gist_res = run_group_gist()
        pref_res = run_group_prefix()
        tok_res = run_group_tokgist()

        # Outputs per method
        out_base_list: List[Dict[str, Any]] = []
        out_trgm_list: List[Dict[str, Any]] = []
        out_gist_list: List[Dict[str, Any]] = []
        out_pref_list: List[Dict[str, Any]] = []
        out_tok_list: List[Dict[str, Any]] = []

        micro_gold = 0
        micro_tp_trgm = 0
        micro_tp_gist = 0
        micro_tp_pref = 0
        micro_tp_tok = 0

        for i, rec in enumerate(records):
            s = seq_res.get(i, {})
            t = trgm_res.get(i, {})
            g = gist_res.get(i, {})
            p = pref_res.get(i, {})
            tk = tok_res.get(i, {})

            gold_vals = s.get("values", []) if s.get("ok") else []
            trgm_vals = t.get("values", []) if t.get("ok") else []
            gist_vals = g.get("values", []) if g.get("ok") else []
            pref_vals = p.get("values", []) if p.get("ok") else []
            tok_vals = tk.get("values", []) if tk.get("ok") else []

            recall_trgm = compute_recall(gold_vals, trgm_vals) if trgm_res else None
            recall_gist = compute_recall(gold_vals, gist_vals) if gist_res else None
            recall_pref = compute_recall(gold_vals, pref_vals) if pref_res else None
            recall_tok = compute_recall(gold_vals, tok_vals) if tok_res else None

            gset = set(gold_vals)
            micro_gold += len(gset)

            if trgm_res:
                micro_tp_trgm += len(gset.intersection(set(trgm_vals)))
            if gist_res:
                micro_tp_gist += len(gset.intersection(set(gist_vals)))
            if pref_res:
                micro_tp_pref += len(gset.intersection(set(pref_vals)))
            if tok_res:
                micro_tp_tok += len(gset.intersection(set(tok_vals)))

            common_fields = {
                "idx": i,
                "regex": rec.get("regex"),
                "workload": rec.get("workload"),
                "regime": rec.get("regime"),
                "first_abstract_idx": rec.get("first_abstract_idx"),
            }

            # Base
            out_base_list.append({
                **common_fields,
                "method": "seqscan",
                **strip_values(s),
            })

            # TRGM
            if trgm_res:
                out_trgm_list.append({
                    **common_fields,
                    "method": "trgm_limit",
                    "recall": recall_trgm,
                    **strip_values(t),
                })

            # gist
            if gist_res:
                out_gist_list.append({
                    **common_fields,
                    "method": "gist_trgm_limit",
                    "recall": recall_gist,
                    **strip_values(g),
                })

            # PREFIX (seed_string -> prefix_string)
            if pref_res:
                seed = rec.get("seed_string", "")
                prefix = build_prefix_from_seed(seed, args.prefix_lit_k)
                out_pref_list.append({
                    **common_fields,
                    "seed_string": seed,
                    "prefix_lit_k": args.prefix_lit_k,
                    "prefix_string": prefix,
                    "method": "prefix_like_then_regex",
                    "recall": recall_pref,
                    **strip_values(p),
                })

            # Tokgist
            if tok_res:
                out_tok_list.append({
                    **common_fields,
                    "method": f"tokgist_k{args.tok_k}",
                    "recall": recall_tok,
                    **strip_values(tk),
                })

        # Macro recall
        def macro_of(lst: List[Dict[str, Any]], key: str = "recall") -> Optional[float]:
            vals = [x[key] for x in lst if x.get(key) is not None]
            if not vals:
                return None
            return round(sum(vals) / len(vals), 6)

        macro_trgm = macro_of(out_trgm_list) if out_trgm_list else None
        macro_gist = macro_of(out_gist_list) if out_gist_list else None
        macro_pref = macro_of(out_pref_list) if out_pref_list else None
        macro_tok = macro_of(out_tok_list) if out_tok_list else None

        # Micro recall
        micro_trgm = round(micro_tp_trgm / micro_gold, 6) if (micro_gold > 0 and out_trgm_list) else None
        micro_gist = round(micro_tp_gist / micro_gold, 6) if (micro_gold > 0 and out_gist_list) else None
        micro_pref = round(micro_tp_pref / micro_gold, 6) if (micro_gold > 0 and out_pref_list) else None
        micro_tok = round(micro_tp_tok / micro_gold, 6) if (micro_gold > 0 and out_tok_list) else None

        # Write outputs
        with open(os.path.join(dir_base, workload_json), "w", encoding="utf-8") as f:
            json.dump(out_base_list, f, ensure_ascii=False, indent=2)

        if out_trgm_list:
            with open(os.path.join(dir_trgm, workload_json), "w", encoding="utf-8") as f:
                json.dump(out_trgm_list, f, ensure_ascii=False, indent=2)

        if out_gist_list:
            with open(os.path.join(dir_gist, workload_json), "w", encoding="utf-8") as f:
                json.dump(out_gist_list, f, ensure_ascii=False, indent=2)

        if out_pref_list:
            with open(os.path.join(dir_pref, workload_json), "w", encoding="utf-8") as f:
                json.dump(out_pref_list, f, ensure_ascii=False, indent=2)

        if out_tok_list:
            with open(os.path.join(dir_tok, workload_json), "w", encoding="utf-8") as f:
                json.dump(out_tok_list, f, ensure_ascii=False, indent=2)

        # Summary
        summary = {
            "dataset": dataset_name,
            "workload_file": workload_file,
            "workload_tag": workload_tag,
            "column_base": col_base,
            "column_tok": col_tok if col_tok else None,
            "n_records": len(records),
            "regex_op": regex_op_str,
            "limit_k_trgm_gist": args.limit_k,
            "prefix": {
                "prefix_lit_k": args.prefix_lit_k,
                "seed_key": "seed_string",
                "prefix_rule": "prefix_string = seed_string[:k] (or full if len<k)",
            },
            "tokgist": {
                "enabled": bool(args.enable_tokgist),
                "tok_available": bool(tok_available),
                "tok_table": tbl_tok,
                "tok_k": args.tok_k,
                "tok_min_token_len": args.tok_min_token_len,
                "tok_limit_k": args.tok_limit_k,
                "build_tok_tables": bool(args.build_tok_tables),
                "rebuild_tok_tables": bool(args.rebuild_tok_tables),
            },
            "table_nrows": {
                tbl_base: nrows_cache.get(tbl_base),
                tbl_trgm: nrows_cache.get(tbl_trgm),
                tbl_gist: nrows_cache.get(tbl_gist),
                tbl_pref: nrows_cache.get(tbl_pref),
                tbl_tok: nrows_cache.get(tbl_tok),
            },
            "latency_t_total_ms": {
                tbl_base: latency_summary(out_base_list),
                tbl_trgm: latency_summary(out_trgm_list) if out_trgm_list else None,
                tbl_gist: latency_summary(out_gist_list) if out_gist_list else None,
                tbl_pref: latency_summary(out_pref_list) if out_pref_list else None,
                tbl_tok: latency_summary(out_tok_list) if out_tok_list else None,
            },
            "recall": {
                "macro_trgm_limit": macro_trgm,
                "macro_gist_limit": macro_gist,
                "macro_prefix": macro_pref,
                "macro_tokgist": macro_tok,
                "micro_trgm_limit": micro_trgm,
                "micro_gist_limit": micro_gist,
                "micro_prefix": micro_pref,
                "micro_tokgist": micro_tok,
            },
        }

        summary_path = os.path.join(dir_base, f"{workload_tag}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[Saved] {dir_base}/{workload_json}")
        if out_trgm_list:
            print(f"[Saved] {dir_trgm}/{workload_json}")
        if out_gist_list:
            print(f"[Saved] {dir_gist}/{workload_json}")
        if out_pref_list:
            print(f"[Saved] {dir_pref}/{workload_json}")
        if out_tok_list:
            print(f"[Saved] {dir_tok}/{workload_json}")
        print(f"[Saved] {summary_path}")

        print(f"[Recall] macro(trgm={macro_trgm}, gist={macro_gist}, prefix={macro_pref}, tok={macro_tok}) "
              f"micro(trgm={micro_trgm}, gist={micro_gist}, prefix={micro_pref}, tok={micro_tok})")

    print("\nAll done.")


if __name__ == "__main__":
    main()

# Example:
# python run_workload_v2.py \
#   --workload_root /home/lyz/wd/LLM_RegularSearch/v2_data/workload/imdb_name \
#   --output_root /home/lyz/wd/LLM_RegularSearch/v2_data/Res/run_workload \
#   --limit 16 --concurrency 8 --limit_k 1024 \
#   --prefix_lit_k 8 \
#   --enable_tokgist --tok_k 3 --tok_min_token_len 3 --build_tok_tables
