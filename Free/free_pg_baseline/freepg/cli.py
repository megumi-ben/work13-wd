import argparse
import json
import os
import sys
import time
from typing import Optional

from . import config
from .db import (
    get_conn,
    get_index_meta,
    latest_index_id,
    run_schema,
)
from .exec_postings import execute_query
from .ingest import ingest_csv
from .key_discovery import discover_keys
from .plan import build_plan, plan_to_str
from .plan_substitute import substitute_plan
from .regex_literals import extract_literals
from .util import log


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="freepg", description="FREE baseline on PostgreSQL")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest CSV into docs table")
    p_ingest.add_argument("--dsn", required=True, help="PostgreSQL DSN")
    p_ingest.add_argument("--csv", required=True, help="CSV file path")
    p_ingest.add_argument("--text-col", required=True, help="Text column name")
    p_ingest.add_argument("--limit", type=int, default=None, help="Optional row limit")
    p_ingest.add_argument("--docs-table", dest="docs_table", default="docs", help="Target docs table name")

    p_keys = sub.add_parser("build-keys", help="Discover minimal useful grams")
    p_keys.add_argument("--dsn", required=True)
    p_keys.add_argument("--c", type=float, default=config.DEFAULT_C)
    p_keys.add_argument("--lmax", type=int, default=config.DEFAULT_LMAX)
    p_keys.add_argument("--sample", type=int, default=config.DEFAULT_SAMPLE_ROWS)
    p_keys.add_argument("--use-shell", dest="use_shell", action="store_true", default=True)
    p_keys.add_argument("--no-shell", dest="use_shell", action="store_false")
    p_keys.add_argument("--docs-table", dest="docs_table", default="docs", help="Docs table to read from")

    p_post = sub.add_parser("build-postings", help="Build postings for key set")
    p_post.add_argument("--dsn", required=True)
    p_post.add_argument("--index-id", dest="index_id", help="Index id to build postings for (default latest)")
    p_post.add_argument("--backend", choices=["auto", "enum_ngrams", "ac_chunk", "bucket"], default="auto")
    p_post.add_argument("--shards", type=int, default=config.DEFAULT_POSTINGS_SHARDS, help="Shard count for enum_ngrams backend")
    p_post.add_argument("--tmpdir", default=config.DEFAULT_TMPDIR, help="Temporary directory for enum_ngrams backend")
    p_post.add_argument("--resume", action="store_true", help="Resume aggregation; skip completed shards")
    p_post.add_argument("--logged", dest="logged", action="store_true", default=config.DEFAULT_LOGGED, help="Store free_index as LOGGED table")
    p_post.add_argument("--unlogged", dest="logged", action="store_false")
    p_post.add_argument("--sort-mem", dest="sort_mem", default=config.DEFAULT_SORT_MEM, help="External sort memory, e.g., 2G")
    p_post.add_argument("--sort-cmd", dest="sort_cmd", default=config.DEFAULT_SORT_CMD, help="External sort command path")

    p_q = sub.add_parser("query", help="Run regex query")
    p_q.add_argument("--dsn", required=True)
    p_q.add_argument("--regex", required=True)
    p_q.add_argument("--limit", type=int, default=None, help="Limit printed final ids")
    p_q.add_argument("--show-plan", action="store_true")
    p_q.add_argument("--index-id", dest="index_id", help="Index id to use (default latest)")
    p_q.add_argument("--docs-table", dest="docs_table", help="Docs table override (default uses index meta)")

    p_qf = sub.add_parser("query-file", help="Run regex queries from a JSON lines file and store results as JSON")
    p_qf.add_argument("--dsn", required=True)
    p_qf.add_argument("--file", required=True, help="Input JSONL file, one JSON object per line with a 'regex' field")
    p_qf.add_argument("--output", required=True, help="Output JSON file to write results list")
    p_qf.add_argument("--index-id", dest="index_id", help="Index id to use (default latest)")
    p_qf.add_argument("--show-plan", action="store_true")
    p_qf.add_argument("--docs-table", dest="docs_table", help="Docs table override (default uses index meta)")

    return parser.parse_args(argv)


def cmd_ingest(args):
    ingest_csv(args.dsn, args.csv, args.text_col, args.limit, docs_table=args.docs_table)


def cmd_build_keys(args):
    index_id, keys, meta = discover_keys(
        dsn=args.dsn,
        c=args.c,
        lmax=args.lmax,
        sample_rows=args.sample,
        use_shell=args.use_shell,
        docs_table=args.docs_table,
    )
    log(f"Index created: {index_id}, keys={meta['key_count']}")


def cmd_build_postings(args):
    from .postings_build import build_postings

    index_id = args.index_id
    conn = get_conn(args.dsn)
    if not index_id:
        index_id = latest_index_id(conn)
    if not index_id:
        raise RuntimeError("No index_id found. Run build-keys first.")
    log(f"Building postings for index_id={index_id}")
    conn.close()
    build_postings(
        args.dsn,
        index_id=index_id,
        backend=args.backend,
        shards=args.shards,
        tmpdir=args.tmpdir,
        resume=args.resume,
        logged=args.logged,
        sort_mem=args.sort_mem,
        sort_cmd=args.sort_cmd,
    )


def cmd_query(args):
    conn = get_conn(args.dsn)
    index_id = args.index_id or latest_index_id(conn)
    if not index_id:
        conn.close()
        raise RuntimeError("No index_id found. Run build-keys/build-postings first.")
    meta = get_index_meta(conn, index_id)
    lmax = meta["lmax"] if meta else config.DEFAULT_LMAX
    docs_table = args.docs_table or (meta["docs_table"] if meta and "docs_table" in meta else "docs")
    t0 = time.perf_counter()
    ext = extract_literals(args.regex)
    if not ext.safe or not ext.literals:
        logical = build_plan([])
        physical = logical
    else:
        logical = build_plan(ext.literals)
        physical = substitute_plan(conn, logical, index_id=index_id, lmax=lmax)
    plan_time = (time.perf_counter() - t0) * 1000
    conn.close()
    if args.show_plan:
        log(f"Logical plan: {plan_to_str(logical)}")
        log(f"Physical plan: {plan_to_str(physical)}")
        if not ext.safe:
            log(f"Extractor unsafe: {ext.reason_if_unsafe}")
        log(f"Extractor used_template={ext.used_template}, literals={len(ext.literals)}")
    else:
        log(f"Extractor safe={ext.safe}, used_template={ext.used_template}, literals={len(ext.literals)}")
    res = execute_query(args.dsn, index_id, physical, args.regex, docs_table=docs_table)
    res["timings"]["plan_build_ms"] = plan_time
    log(f"Candidate count: {'all docs' if res['candidates'] is None else len(res['candidates'])}")
    log(f"Final matches: {len(res['final_ids'])}")
    if res.get("candidate_ratio") is not None:
        log(f"Candidate ratio: {res['candidate_ratio']}")
    for k, v in res["timings"].items():
        log(f"Timing {k}: {v:0.1f}ms")
    if args.limit:
        log(f"First {args.limit} ids: {res['final_ids'][:args.limit]}")


def cmd_query_file(args):
    conn = get_conn(args.dsn)
    index_id = args.index_id or latest_index_id(conn)
    if not index_id:
        conn.close()
        raise RuntimeError("No index_id found. Run build-keys/build-postings first.")
    meta = get_index_meta(conn, index_id)
    lmax = meta["lmax"] if meta else config.DEFAULT_LMAX
    docs_table = args.docs_table or (meta["docs_table"] if meta and "docs_table" in meta else "docs")
    conn.close()
    queries = _load_queries(args.file)
    results = []
    for idx, obj in enumerate(queries, start=1):
        regex = obj.get("regex")
        if not regex:
            log(f"Skipping item {idx}: missing 'regex'")
            continue
        ext = extract_literals(regex)
        if not ext.safe or not ext.literals:
            logical = build_plan([])
            physical = logical
        else:
            logical = build_plan(ext.literals)
            qconn = get_conn(args.dsn)
            physical = substitute_plan(qconn, logical, index_id=index_id, lmax=lmax)
            qconn.close()
        res = execute_query(args.dsn, index_id, physical, regex, docs_table=docs_table)
        if args.show_plan:
            res["logical_plan"] = plan_to_str(logical)
            res["physical_plan"] = plan_to_str(physical)
        res["regex"] = regex
        res["safe"] = ext.safe
        res["used_template"] = getattr(ext, "used_template", False)
        res["input"] = obj
        res["idx"] = idx
        # drop heavy fields before storing
        res.pop("candidates", None)
        res.pop("final_ids", None)
        results.append(res)
    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)
    log(f"Wrote results for {len(results)} queries to {args.output}")


def _load_queries(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if content.lstrip().startswith("["):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            else:
                log("Top-level JSON is not a list; treating as single object")
                return [data]
        except json.JSONDecodeError as e:
            log(f"Failed to parse JSON array: {e}; falling back to JSONL")
    # JSONL fallback
    queries = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            queries.append(obj)
        except json.JSONDecodeError as e:
            log(f"Skipping line {line_no}: invalid JSON ({e})")
    return queries


def main(argv=None):
    args = parse_args(argv)
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "build-keys":
        cmd_build_keys(args)
    elif args.command == "build-postings":
        cmd_build_postings(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "query-file":
        cmd_query_file(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
