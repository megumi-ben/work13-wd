"""Show how SQL is constructed for direct/3way/multiway given a regex."""
import argparse
from typing import List, Tuple

import psycopg2

from blare_pg.config import BlareConfig, PgConfig
from blare_pg.pg_io import _build_literal_clauses
from blare_pg.split_matcher import select_literals_for_mode
from blare_pg.splitter import build_split_info
from blare_pg.types import Mode


def _build_query(pg_conf: PgConfig,
                 literals: List[str],
                 regex: str,
                 case_mode: str,
                 workload_mode: str,
                 limit_k: int,
                 inline_like: bool) -> Tuple[str, List]:
    clauses, params, _ = _build_literal_clauses(pg_conf, literals, case_mode, inline=inline_like)
    all_clauses = list(clauses)
    query_params: List = list(params)
    all_clauses.append(
        psycopg2.sql.SQL("{col} ~ %s").format(col=psycopg2.sql.Identifier(pg_conf.column))
    )
    query_params.append(regex)

    if workload_mode == "count":
        base = psycopg2.sql.SQL("SELECT COUNT(*) FROM {tbl}").format(tbl=psycopg2.sql.Identifier(pg_conf.table))
        order_limit = psycopg2.sql.SQL("")
    else:
        base = psycopg2.sql.SQL("SELECT {id_col} FROM {tbl}").format(
            id_col=psycopg2.sql.Identifier(pg_conf.id_column),
            tbl=psycopg2.sql.Identifier(pg_conf.table),
        )
        order_limit = psycopg2.sql.SQL(" ORDER BY {id_col} LIMIT %s").format(
            id_col=psycopg2.sql.Identifier(pg_conf.id_column)
        )
        query_params.append(limit_k)

    where_sql = psycopg2.sql.SQL("")
    if all_clauses:
        where_sql = psycopg2.sql.SQL(" WHERE ") + psycopg2.sql.SQL(" AND ").join(all_clauses)

    query_sql = base + where_sql + order_limit
    return query_sql.as_string(psycopg2.connect(pg_conf.dsn)), query_params


def main() -> int:
    ap = argparse.ArgumentParser(description="Show constructed SQL for BLARE-PG arms.")
    ap.add_argument("--dsn", required=True, help="PostgreSQL DSN")
    ap.add_argument("--table", required=True, help="Base table (for direct)")
    ap.add_argument("--trgm_table", help="Trgm table for prefilter arms (default: <table>_trgm)")
    ap.add_argument("--column", required=True, help="Text column name")
    ap.add_argument("--id_column", default="id", help="ID column")
    ap.add_argument("--regex", required=True, help="Regex to inspect")
    ap.add_argument("--workload_mode", choices=["count", "topk"], default="count")
    ap.add_argument("--limit_k", type=int, default=100)
    args = ap.parse_args()

    pg_conf_base = PgConfig(dsn=args.dsn,
                            table=args.table,
                            column=args.column,
                            id_column=args.id_column,
                            trgm_table=args.trgm_table or f"{args.table}_trgm")
    pg_conf_trgm = PgConfig(dsn=args.dsn,
                            table=pg_conf_base.trgm_table,
                            column=args.column,
                            id_column=args.id_column,
                            trgm_table=pg_conf_base.trgm_table)
    blare_conf = BlareConfig()

    info = build_split_info(args.regex)
    print(f"regex={args.regex}")
    print(f"  splittable={info.splittable} reason_code={info.reason_code} case_mode={info.case_mode}")
    print(f"  required_literals={info.required_literals}")

    for mode in (Mode.DIRECT, Mode.THREE_WAY, Mode.MULTI_WAY):
        literals, lit_reason = select_literals_for_mode(info, mode, min_len=blare_conf.min_literal_len)
        active_conf = pg_conf_base if mode == Mode.DIRECT else pg_conf_trgm
        try:
            sql_str, params = _build_query(
                active_conf,
                literals,
                args.regex,
                info.case_mode,
                args.workload_mode,
                args.limit_k,
                blare_conf.inline_like_literals,
            )
        except Exception as exc:
            print(f"[{mode.value}] error building SQL: {exc}")
            continue
        print(f"[{mode.value}] literals={literals} lit_reason={lit_reason}")
        print(f"[{mode.value}] sql={sql_str}")
        print(f"[{mode.value}] params={params}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
