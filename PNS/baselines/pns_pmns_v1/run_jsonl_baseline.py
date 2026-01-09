"""
Batch runner for regex_span-backed baseline on a JSONL dataset.

Each line of the input should contain at least:
  - "seed_string": the text to match against
  - "regex": the pattern (PostgreSQL flavor)

The script adds "matches_pg" to each record (bool, same语义 as txt ~ regex)
and writes results to the output JSONL.
"""
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2

# Ensure project root is on sys.path when run from arbitrary cwd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.pns_pmns_v1.factor_extractor import extract_factors
from baselines.pns_pmns_v1.pns_pmns import PNSPMNSVerifier
from baselines.pns_pmns_v1.prefilter_sql import (
    build_where_for_necessary_literals,
    merge_with_user_where,
)
from baselines.pns_pmns_v1.verify_pg import find_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline verifier over JSONL dataset.")
    parser.add_argument("--input", required=True, help="Input JSONL file path.")
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument(
        "--dsn",
        default=os.getenv("PG_DSN", "dbname=postgres"),
        help="PostgreSQL DSN string (defaults to PG_DSN env or 'dbname=postgres').",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of records to process.")
    parser.add_argument(
        "--keep-all-fields",
        action="store_true",
        help="If set, copy all input fields; otherwise only seed_string/regex/matches_pg/elapsed_ms.",
    )
    parser.add_argument(
        "--with-count",
        action="store_true",
        help="If set, also compute match_count via regex_find_all (overlap, allow_empty=false).",
    )
    parser.add_argument(
        "--mode",
        choices=["pns", "pmns-lite", "pns-n", "pmns-lite-n"],
        default="pns",
        help="Verification mode (pmns-lite adds M prune; suffix -n enables N-factor-lite).",
    )
    parser.add_argument(
        "--table-mode",
        action="store_true",
        help="Enable table mode: prefilter rows in a table then verify each row. Otherwise treat input JSONL seed_string as the text.",
    )
    parser.add_argument("--table", help="Table name when --table-mode is enabled.")
    parser.add_argument(
        "--column",
        default="name",
        help="Column to run regex against when --table-mode is enabled (default: name).",
    )
    parser.add_argument(
        "--where",
        default=None,
        help="Optional SQL WHERE clause (without 'WHERE') to constrain rows when scanning a table.",
    )
    parser.add_argument(
        "--prefilter",
        choices=["none", "necessary_like", "necessary_ilike"],
        default="none",
        help="Prefilter strategy in table mode (only uses proven necessary_literals).",
    )
    parser.add_argument(
        "--prefilter-k",
        type=int,
        default=2,
        help="Max number of necessary literals to use in prefilter (longest first).",
    )
    parser.add_argument(
        "--no-prefilter-count",
        dest="prefilter_count",
        action="store_false",
        help="Disable counting candidate rows before verification.",
    )
    parser.set_defaults(prefilter_count=True)
    parser.add_argument(
        "--fetch-size",
        type=int,
        default=1000,
        help="Fetch size for server-side cursor in table mode.",
    )
    parser.add_argument(
        "--emit-sql",
        action="store_true",
        help="Emit the generated prefilter SQL fragment in output stats.",
    )
    parser.add_argument(
        "--compare-pg-count",
        action="store_true",
        help="Also run SELECT count(*) ... column ~ pattern for sanity (extra cost).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.table_mode and not args.table:
        print("--table is required when --table-mode is enabled", file=sys.stderr)
        sys.exit(1)

    try:
        conn = psycopg2.connect(args.dsn)
    except Exception as exc:  # pragma: no cover - operational path
        print(f"Failed to connect to Postgres with DSN={args.dsn}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Named cursor in table-mode requires an open transaction; disable autocommit there.
    conn.autocommit = not args.table_mode
    verifier = PNSPMNSVerifier(conn, mode=args.mode)

    processed = 0
    matched = 0
    total_ms = 0.0

    # If table-mode is enabled, pre-compute base row count (respecting user WHERE) once.
    base_rows = None
    if args.table_mode and args.prefilter_count:
        base_sql = f"SELECT count(*) FROM {args.table}"
        if args.where:
            base_sql += f" WHERE {args.where}"
        with conn.cursor() as cur:
            cur.execute(base_sql)
            base_rows = cur.fetchone()[0]

    def table_count(pattern: str) -> Optional[int]:
        if not args.table:
            return None
        sql = f"SELECT count(*) FROM {args.table}"
        if args.where:
            sql += f" WHERE {args.where} AND {args.column} ~ %s"
        else:
            sql += f" WHERE {args.column} ~ %s"
        with conn.cursor() as cur:
            cur.execute(sql, (pattern,))
            return cur.fetchone()[0]

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if args.limit is not None and processed >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pattern = rec["regex"]

            if not args.table_mode:
                txt = rec["seed_string"]
                t0 = time.perf_counter()
                ok = verifier.matches(txt, pattern, lmin=1)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                total_ms += elapsed_ms

                match_count = None
                if args.with_count:
                    spans = find_all(conn, txt, pattern, start_pos=1, overlap=True, allow_empty=False)
                    match_count = len(spans)
                table_matches = table_count(pattern)

                if args.keep_all_fields:
                    out = dict(rec)
                    out["matches_pg"] = bool(ok)
                    out["elapsed_ms"] = elapsed_ms
                    if match_count is not None:
                        out["match_count"] = match_count
                    if table_matches is not None:
                        out["table_match_count"] = table_matches
                else:
                    out = {
                        "seed_string": txt,
                        "regex": pattern,
                        "matches_pg": bool(ok),
                        "elapsed_ms": elapsed_ms,
                    }
                    if match_count is not None:
                        out["match_count"] = match_count
                    if table_matches is not None:
                        out["table_match_count"] = table_matches

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                processed += 1
                matched += 1 if ok else 0
                continue

            # table-mode path: prefilter + cursor fetch + per-row verification
            t_total_start = time.perf_counter()
            factors = extract_factors(pattern)
            necessary_literals: List[str] = []
            if factors:
                for isl in factors.islands:
                    necessary_literals.extend([lit for lit in isl.necessary_literals if len(lit) >= 2])
            necessary_literals = sorted(set(necessary_literals), key=lambda x: (-len(x), x))

            prefilter_enabled = False
            prefilter_kind = args.prefilter
            prefilter_reason = None
            prefilter_lits_used: List[str] = []
            prefilter_sql_fragment = ""
            prefilter_params: List[str] = []

            if args.prefilter == "none":
                prefilter_reason = "disabled_by_flag"
            elif not factors:
                prefilter_reason = "parse_fail"
            elif not necessary_literals:
                prefilter_reason = "no_necessary"
            else:
                for lit in necessary_literals:
                    if lit in prefilter_lits_used:
                        continue
                    prefilter_lits_used.append(lit)
                    if len(prefilter_lits_used) >= args.prefilter_k:
                        break
                use_ilike = args.prefilter == "necessary_ilike"
                frag, params = build_where_for_necessary_literals(
                    args.column, prefilter_lits_used, use_ilike=use_ilike, k=args.prefilter_k
                )
                if frag:
                    prefilter_sql_fragment = frag
                    prefilter_params = params
                    prefilter_enabled = True
                else:
                    prefilter_reason = "no_prefilter_fragment"

            full_where = args.where
            if prefilter_enabled:
                full_where = merge_with_user_where(args.where, prefilter_sql_fragment)
            else:
                full_where = args.where or ""

            candidate_rows = None
            prefilter_count_ms = None
            if args.prefilter_count:
                count_sql = f"SELECT count(*) FROM {args.table}"
                if full_where:
                    count_sql += f" WHERE {full_where}"
                t_cnt = time.perf_counter()
                with conn.cursor() as cur:
                    cur.execute(count_sql, prefilter_params)
                    candidate_rows = cur.fetchone()[0]
                prefilter_count_ms = (time.perf_counter() - t_cnt) * 1000.0
            elif prefilter_enabled:
                candidate_rows = None
            else:
                candidate_rows = base_rows

            fetch_ms = 0.0
            verify_ms = 0.0
            verified_rows = 0
            matched_rows = 0

            select_sql = f"SELECT {args.column} FROM {args.table}"
            if full_where:
                select_sql += f" WHERE {full_where}"
            cursor_name = f"pref_{uuid.uuid4().hex[:12]}"
            t_verify_start = time.perf_counter()
            with conn.cursor(name=cursor_name) as cur:
                cur.execute(select_sql, prefilter_params)
                while True:
                    t_fetch0 = time.perf_counter()
                    batch = cur.fetchmany(args.fetch_size)
                    fetch_ms += (time.perf_counter() - t_fetch0) * 1000.0
                    if not batch:
                        break
                    for (txt_val,) in batch:
                        verified_rows += 1
                        t_v0 = time.perf_counter()
                        ok = verifier.matches(txt_val, pattern, lmin=1)
                        verify_ms += (time.perf_counter() - t_v0) * 1000.0
                        if ok:
                            matched_rows += 1
            pg_match_count = None
            pg_count_ms = None
            if args.compare_pg_count:
                sql_pg = f"SELECT count(*) FROM {args.table}"
                if args.where:
                    sql_pg += f" WHERE {args.where} AND {args.column} ~ %s"
                else:
                    sql_pg += f" WHERE {args.column} ~ %s"
                t_pg = time.perf_counter()
                with conn.cursor() as cur:
                    cur.execute(sql_pg, (pattern,))
                    pg_match_count = cur.fetchone()[0]
                pg_count_ms = (time.perf_counter() - t_pg) * 1000.0

            total_query_ms = (time.perf_counter() - t_total_start) * 1000.0
            total_ms += total_query_ms

            candidate_ratio = None
            if candidate_rows is not None and base_rows not in (None, 0):
                candidate_ratio = candidate_rows / base_rows

            out_stats: Dict[str, Optional[object]] = {
                "table": args.table,
                "column": args.column,
                "user_where": args.where,
                "mode": args.mode,
                "prefilter_kind": prefilter_kind,
                "prefilter_enabled": int(prefilter_enabled),
                "prefilter_disabled_reason": prefilter_reason,
                "prefilter_lits": prefilter_lits_used,
                "prefilter_k": args.prefilter_k,
                "candidate_rows": candidate_rows,
                "candidate_ratio": candidate_ratio,
                "verified_rows": verified_rows,
                "matched_rows": matched_rows,
                "prefilter_count_ms": prefilter_count_ms,
                "fetch_ms": fetch_ms,
                "verify_ms": verify_ms,
                "total_ms": total_query_ms,
                "pg_count_ms": pg_count_ms,
                "pg_match_count": pg_match_count,
            }
            if args.emit_sql and prefilter_enabled:
                out_stats["prefilter_sql"] = prefilter_sql_fragment
                out_stats["prefilter_params"] = prefilter_params
            if hasattr(verifier, "last_stats"):
                out_stats["verifier_last_stats"] = getattr(verifier, "last_stats", None)

            out = dict(rec)
            out["regex"] = pattern
            out["matches_pg"] = matched_rows > 0
            out["stats"] = out_stats

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            processed += 1
            matched += 1 if matched_rows > 0 else 0
            if args.table_mode:
                conn.commit()

    conn.close()
    avg_ms = (total_ms / processed) if processed else 0.0
    print(
        f"Processed {processed} records; matched={matched}; "
        f"total_ms={total_ms:.3f}; avg_ms={avg_ms:.3f}. Output -> {args.output}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
