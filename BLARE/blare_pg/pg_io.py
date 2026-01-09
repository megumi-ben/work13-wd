"""PostgreSQL IO helpers for BLARE-PG (server-side filtering only)."""
import hashlib
import random
from typing import Dict, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

from .config import BlareConfig, PgConfig
from .types import ExplainStats, GucSettings


# ---------------------- Connection / GUCs ---------------------- #
def get_connection(pg_conf: PgConfig) -> psycopg2.extensions.connection:
    """Create and return a psycopg2 connection (GUCs applied separately)."""
    return psycopg2.connect(pg_conf.dsn)


def apply_session_gucs(conn, blare_conf: BlareConfig) -> GucSettings:
    """Apply deterministic GUCs (jit off, no parallel, work_mem) and return actual settings."""
    with conn.cursor() as cur:
        if blare_conf.enforce_gucs:
            if blare_conf.disable_jit:
                cur.execute("SET jit = off;")
            if blare_conf.disable_parallel:
                cur.execute("SET max_parallel_workers_per_gather = 0;")
            if blare_conf.work_mem:
                cur.execute("SET work_mem = %s;", (blare_conf.work_mem,))
        cur.execute("SHOW jit;")
        jit_val = cur.fetchone()[0]
        cur.execute("SHOW max_parallel_workers_per_gather;")
        par_val = cur.fetchone()[0]
        cur.execute("SHOW work_mem;")
        wm_val = cur.fetchone()[0]
    return GucSettings(jit=jit_val, max_parallel_workers_per_gather=par_val, work_mem=wm_val)


# ---------------------- SQL helpers ---------------------- #
def escape_like(literal: str) -> str:
    """Escape %, _, and backslash for safe LIKE patterns using ESCAPE '\\'."""
    return (
        literal.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _build_literal_clauses(pg_conf: PgConfig,
                           literals: Sequence[str],
                           case_mode: str,
                           inline: bool = True) -> Tuple[List[sql.SQL], List, str]:
    """
    Build SQL clauses and parameters for literal AND filters using pg_trgm-friendly LIKE.

    Returns (clauses, params, index_hint). When inline is True, LIKE patterns are inlined
    as SQL literals (planner sees constants) and params will exclude those patterns.
    """
    clauses: List[sql.SQL] = []
    params: List = []
    insensitive = case_mode in ("insensitive", "unknown")

    escape_expr = sql.Literal("\\")
    for lit in literals:
        pat = f"%{escape_like(lit.lower() if insensitive else lit)}%"
        # When inlining, double % so psycopg2's pyformat rendering doesn't treat them as placeholders.
        pat_for_sql = pat.replace("%", "%%") if inline else pat
        if inline:
            pattern_literal = sql.Literal(pat_for_sql)
        else:
            pattern_literal = sql.SQL("%s")
            params.append(pat)
        column_expr = sql.SQL("lower({col})").format(col=sql.Identifier(pg_conf.column)) if insensitive else sql.Identifier(pg_conf.column)
        clauses.append(
            sql.SQL("{col_expr} LIKE {pat} ESCAPE {esc}").format(
                col_expr=column_expr,
                pat=pattern_literal,
                esc=escape_expr,
            )
        )
    index_hint = (
        f"lower({pg_conf.column}) gin_trgm_ops"
        if insensitive else f"{pg_conf.column} gin_trgm_ops"
    )
    return clauses, params, index_hint


def explain_json(conn, query_sql: sql.SQL, params: Sequence, analyze: bool = False) -> Dict:
    """Run EXPLAIN (FORMAT JSON) or EXPLAIN ANALYZE (when analyze=True) and return the parsed JSON object."""
    prefix = "EXPLAIN (FORMAT JSON)" if not analyze else "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)"
    with conn.cursor() as cur:
        exec_params = None if not params else params
        cur.execute(sql.SQL(prefix) + sql.SQL(" ") + query_sql, exec_params)
        row = cur.fetchone()
    if not row:
        return {}
    data = row[0]
    if isinstance(data, list):
        if data:
            return data[0]
        return {}
    return data or {}


def _collect_node_types(plan: Dict, acc: List[str]) -> None:
    node_type = plan.get("Node Type")
    if node_type:
        acc.append(node_type)
    for child in plan.get("Plans", []) or []:
        _collect_node_types(child, acc)


def _collect_buffers(plan: Dict) -> Tuple[int, int]:
    hit = plan.get("Shared Hit Blocks", 0)
    read = plan.get("Shared Read Blocks", 0)
    for child in plan.get("Plans", []) or []:
        c_hit, c_read = _collect_buffers(child)
        hit += c_hit
        read += c_read
    return hit, read


def extract_explain_stats(explain: Dict) -> ExplainStats:
    """Extract execution time, node summary, and buffer stats from EXPLAIN JSON."""
    if not explain:
        return ExplainStats(exec_ms=0.0, plan_node_summary="", buffers_hit=0, buffers_read=0)
    exec_ms = explain.get("Execution Time", 0.0)
    if exec_ms == 0.0 and "Plan" in explain:
        exec_ms = explain["Plan"].get("Actual Total Time", 0.0)
    nodes: List[str] = []
    plan = explain.get("Plan", {})
    _collect_node_types(plan, nodes)
    unique_nodes = sorted(set(nodes))
    summary = "/".join(unique_nodes)
    hit, read = _collect_buffers(plan)
    return ExplainStats(exec_ms=exec_ms, plan_node_summary=summary, buffers_hit=hit, buffers_read=read)


def _first_non_aggregate(plan: Dict) -> Dict:
    """
    Drill down past Aggregate nodes; if no children exist, return the plan itself.
    """
    while plan and plan.get("Node Type") == "Aggregate":
        plans = plan.get("Plans") or []
        if not plans:
            break
        plan = plans[0]
    return plan


def _extract_rows(plan: Dict) -> Tuple[int, int]:
    if not isinstance(plan, dict):
        return 0, 0
    node = _first_non_aggregate(plan)
    if not isinstance(node, dict):
        return 0, 0
    actual = int(node.get("Actual Rows", 0) or 0)
    estimated = int(node.get("Plan Rows", 0) or 0)
    return actual, estimated


def _has_index_node(nodes: List[str]) -> bool:
    return any("Index" in n or "Bitmap" in n for n in nodes)


def _extract_nodes(plan: Dict) -> List[str]:
    nodes: List[str] = []
    _collect_node_types(plan, nodes)
    return nodes


def _run_query_fetch_one(conn, query_sql: sql.SQL, params: Sequence) -> int:
    with conn.cursor() as cur:
        cur.execute(query_sql, params if params else None)
        (val,) = cur.fetchone()
    return int(val)


# ---------------------- Counting / Retrieval ---------------------- #
def estimate_candidates(conn,
                        pg_conf: PgConfig,
                        clauses: List[sql.SQL],
                        params: Sequence) -> Tuple[int, Dict]:
    """
    Use EXPLAIN (no ANALYZE) on prefilter-only query to obtain estimated rows.
    """
    base = sql.SQL("SELECT 1 FROM {tbl}").format(tbl=sql.Identifier(pg_conf.table))
    where_sql = sql.SQL("")
    if clauses:
        where_sql = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses)
    query_sql = base + where_sql
    exec_params = tuple(params) if params else None
    explain = explain_json(conn, query_sql, exec_params, analyze=False)
    plan = explain.get("Plan", {}) if explain else {}
    _, est_rows = _extract_rows(plan)
    return est_rows, explain


def run_prefilter_verify(pg_conf: PgConfig,
                         blare_conf: BlareConfig,
                         literals: Sequence[str],
                         regex: str,
                         case_mode: str,
                         workload_mode: str = "count",
                         limit_k: int = 100,
                         sample_ids: Optional[Sequence[int]] = None,
                         debug_counts: bool = False) -> Tuple[int, int, bool, str, ExplainStats, GucSettings, str, bool]:
    """
    Run prefilter + regex in PostgreSQL and return stats.

    Returns:
        n_matches (actual rows)
        candidates_count
        candidates_is_estimate
        candidates_estimate_source
        explain_stats (for main query)
        gucs used
        index_hint
        plan_guard_triggered
    """
    conn = get_connection(pg_conf)
    gucs = apply_session_gucs(conn, blare_conf)
    try:
        clauses, params, index_hint = _build_literal_clauses(
            pg_conf, literals, case_mode, inline=blare_conf.inline_like_literals
        )
        pf_params: List = list(params)
        all_clauses: List[sql.SQL] = list(clauses)

        if sample_ids:
            temp_table_name = "tmp_blare_sample_ids"
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {temp_table_name};")
                cur.execute(f"CREATE TEMP TABLE {temp_table_name} (id bigint PRIMARY KEY) ON COMMIT DROP;")
                execute_values(cur, f"INSERT INTO {temp_table_name} (id) VALUES %s", [(int(x),) for x in sample_ids])
            all_clauses.append(
                sql.SQL("{id_col} IN (SELECT id FROM {tmp})").format(
                    id_col=sql.Identifier(pg_conf.id_column),
                    tmp=sql.Identifier(temp_table_name),
                )
            )

        # Candidates estimate (no execution)
        candidates_count, _ = estimate_candidates(conn, pg_conf, all_clauses, pf_params)
        candidates_is_estimate = True
        candidates_source = "EXPLAIN_ROWS"
        if debug_counts:
            where_sql = sql.SQL("")
            if all_clauses:
                where_sql = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(all_clauses)
            pf_query = sql.SQL("SELECT COUNT(*) FROM {tbl}").format(
                tbl=sql.Identifier(pg_conf.table)
            ) + where_sql
            candidates_count = _run_query_fetch_one(conn, pf_query, tuple(pf_params) if pf_params else None)
            candidates_is_estimate = False
            candidates_source = "REAL_COUNT"

        # Main query
        clauses_with_regex = list(all_clauses)
        regex_param = regex
        clauses_with_regex.append(
            sql.SQL("{col} ~ %s").format(col=sql.Identifier(pg_conf.column))
        )
        main_params: List = list(pf_params) + [regex_param]

        base_select = sql.SQL("SELECT COUNT(*)") if workload_mode == "count" else sql.SQL("SELECT {id_col}").format(
            id_col=sql.Identifier(pg_conf.id_column)
        )
        base = base_select + sql.SQL(" FROM {tbl}").format(tbl=sql.Identifier(pg_conf.table))
        where_sql = sql.SQL("")
        if clauses_with_regex:
            where_sql = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses_with_regex)
        order_limit = sql.SQL("")
        if workload_mode == "topk":
            order_limit = sql.SQL(" ORDER BY {id_col} LIMIT %s").format(
                id_col=sql.Identifier(pg_conf.id_column)
            )
            main_params.append(limit_k)
        query_sql = base + where_sql + order_limit

        explain = explain_json(conn, query_sql, tuple(main_params) if main_params else None, analyze=True)
        stats = extract_explain_stats(explain)
        plan = explain.get("Plan", {}) if explain else {}
        actual_rows, _ = _extract_rows(plan)
        n_matches = actual_rows

        nodes = _extract_nodes(plan)
        plan_guard_triggered = False
        # Only enforce guard when running against full table (no sample_ids), otherwise
        # small sampled queries may choose Seq Scan even if the real query would use an index.
        if sample_ids is None and blare_conf.plan_guard_enabled and literals:
            long_lit = any(len(l) >= blare_conf.min_literal_len for l in literals)
            if long_lit and "Seq Scan" in nodes and not _has_index_node(nodes):
                plan_guard_triggered = True

        return n_matches, candidates_count, candidates_is_estimate, candidates_source, stats, gucs, index_hint, plan_guard_triggered
    finally:
        conn.close()


def count_direct(pg_conf: PgConfig,
                 blare_conf: BlareConfig,
                 regex: str) -> int:
    """Count matches with direct regex (no prefilter), used for correctness checks."""
    conn = get_connection(pg_conf)
    apply_session_gucs(conn, blare_conf)
    try:
        query_sql = sql.SQL(
            "SELECT COUNT(*) FROM {tbl} WHERE {col} ~ %s"
        ).format(
            tbl=sql.Identifier(pg_conf.table),
            col=sql.Identifier(pg_conf.column),
        )
        return _run_query_fetch_one(conn, query_sql, (regex,))
    finally:
        conn.close()


def estimate_rows(pg_conf: PgConfig) -> int:
    """Estimate row count using pg_class.reltuples. Returns 0 on failure."""
    conn = get_connection(pg_conf)
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT reltuples::bigint FROM pg_class WHERE oid = {table}::regclass").format(
                    table=sql.Literal(pg_conf.table)
                )
            )
            row = cur.fetchone()
            if not row or row[0] is None:
                return 0
            return int(row[0])
    finally:
        conn.close()


# ---------------------- Sampling ---------------------- #
def get_id_bounds(pg_conf: PgConfig) -> Optional[Tuple[int, int]]:
    """Return (min_id, max_id) for integer id_column. None when table is empty or unsupported."""
    conn = get_connection(pg_conf)
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT MIN({id_col})::bigint, MAX({id_col})::bigint FROM {tbl}").format(
                    id_col=sql.Identifier(pg_conf.id_column),
                    tbl=sql.Identifier(pg_conf.table),
                )
            )
            row = cur.fetchone()
            if not row or row[0] is None or row[1] is None:
                return None
            return int(row[0]), int(row[1])
    finally:
        conn.close()


def _fetch_sample_tablesample(pg_conf: PgConfig, blare_conf: BlareConfig, limit: int) -> Tuple[List[str], List[int]]:
    """Fallback sampling using TABLESAMPLE SYSTEM. This may scan more data and is unstable."""
    sample_pct = min(100.0, max(0.01, blare_conf.tablesample_pct))
    conn = get_connection(pg_conf)
    apply_session_gucs(conn, blare_conf)
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT {id_col}, {col} FROM {tbl} TABLESAMPLE SYSTEM (%s) LIMIT %s"
                ).format(
                    id_col=sql.Identifier(pg_conf.id_column),
                    col=sql.Identifier(pg_conf.column),
                    tbl=sql.Identifier(pg_conf.table),
                ),
                (sample_pct, limit),
            )
            rows = cur.fetchall()
            ids = [int(r[0]) for r in rows]
            lines = [r[1] for r in rows]
            return lines, ids
    finally:
        conn.close()


def sample_rows_by_index_jump(pg_conf: PgConfig,
                              blare_conf: BlareConfig,
                              target_rows: int,
                              max_attempts: int = 5) -> Tuple[List[str], List[int]]:
    """
    Sample rows by random index jump: pick random id within bounds, jump to next available id.
    Returns lines and ids (may be fewer than target_rows if table sparse).
    """
    bounds = get_id_bounds(pg_conf)
    if not bounds:
        return [], []
    min_id, max_id = bounds
    if min_id > max_id:
        return [], []

    conn = get_connection(pg_conf)
    apply_session_gucs(conn, blare_conf)
    seen: set[int] = set()
    lines: List[str] = []
    ids: List[int] = []
    try:
        with conn.cursor() as cur:
            attempts = 0
            while len(lines) < target_rows and attempts < max_attempts * target_rows:
                probe = random.randint(min_id, max_id)
                cur.execute(
                    sql.SQL(
                        "SELECT {id_col}, {col} FROM {tbl} WHERE {id_col} >= %s ORDER BY {id_col} LIMIT 1"
                    ).format(
                        id_col=sql.Identifier(pg_conf.id_column),
                        col=sql.Identifier(pg_conf.column),
                        tbl=sql.Identifier(pg_conf.table),
                    ),
                    (probe,),
                )
                row = cur.fetchone()
                attempts += 1
                if not row:
                    continue
                row_id = int(row[0])
                if row_id in seen:
                    continue
                seen.add(row_id)
                ids.append(row_id)
                lines.append(row[1])
            return lines, ids
    finally:
        conn.close()


def sample_rows_by_bucket(pg_conf: PgConfig,
                          blare_conf: BlareConfig,
                          target_rows: int,
                          bucket_count: int = 20) -> Tuple[List[str], List[int]]:
    """
    Bucketed index-jump sampling for more even coverage across ID range.
    """
    bounds = get_id_bounds(pg_conf)
    if not bounds:
        return [], []
    min_id, max_id = bounds
    if min_id > max_id:
        return [], []
    span = max_id - min_id + 1
    bucket_size = max(1, span // bucket_count)
    per_bucket = max(1, target_rows // bucket_count)

    conn = get_connection(pg_conf)
    apply_session_gucs(conn, blare_conf)
    seen: set[int] = set()
    lines: List[str] = []
    ids: List[int] = []
    try:
        with conn.cursor() as cur:
            for b in range(bucket_count):
                start = min_id + b * bucket_size
                end = min(start + bucket_size - 1, max_id)
                for _ in range(per_bucket):
                    probe = random.randint(start, end)
                    cur.execute(
                        sql.SQL(
                            "SELECT {id_col}, {col} FROM {tbl} WHERE {id_col} >= %s ORDER BY {id_col} LIMIT 1"
                        ).format(
                            id_col=sql.Identifier(pg_conf.id_column),
                            col=sql.Identifier(pg_conf.column),
                            tbl=sql.Identifier(pg_conf.table),
                        ),
                        (probe,),
                    )
                    row = cur.fetchone()
                    if not row:
                        continue
                    row_id = int(row[0])
                    if row_id in seen:
                        continue
                    seen.add(row_id)
                    ids.append(row_id)
                    lines.append(row[1])
                    if len(lines) >= target_rows:
                        return lines, ids
            return lines, ids
    finally:
        conn.close()


def fetch_sample_lines(pg_conf: PgConfig,
                       blare_conf: BlareConfig) -> Tuple[List[str], List[int], bool, str, str, int]:
    """
    Sample lines for learner using configured method; ensure enough samples or report reason.
    Returns (lines, ids, ok, reason, sampling_method, n_sample).
    """
    target = min(blare_conf.max_sample_rows, max(blare_conf.min_sample_rows, 1))
    method = blare_conf.sampling_method
    if method == "index_jump":
        lines, ids = sample_rows_by_index_jump(pg_conf, blare_conf, target_rows=target)
    elif method == "bucket_index_jump":
        lines, ids = sample_rows_by_bucket(pg_conf, blare_conf, target_rows=target, bucket_count=blare_conf.bucket_count)
    elif method == "tablesample":
        lines, ids = _fetch_sample_tablesample(pg_conf, blare_conf, limit=target)
    else:
        lines, ids = sample_rows_by_bucket(pg_conf, blare_conf, target_rows=target, bucket_count=blare_conf.bucket_count)
    ok = len(lines) >= blare_conf.min_sample_rows
    reason = "" if ok else "insufficient_samples"
    return lines, ids, ok, reason, method, len(lines)


# ---------------------- Utility ---------------------- #
def hash_dsn(dsn: str) -> str:
    """Hash DSN for policy cache key."""
    return hashlib.sha256(dsn.encode("utf-8")).hexdigest()
