"""Command-line runner for BLARE-PG prototype."""
import argparse
import json
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from .config import BlareConfig, PgConfig
from .learner import choose_best_mode_greedy
from .pg_io import (
    count_direct,
    estimate_rows,
    fetch_sample_lines,
    hash_dsn,
    run_prefilter_verify,
)
from .splitter import build_split_info
from .split_matcher import select_literals_for_mode
from .types import EvalStats, Mode


def _load_regex_list(regex: str = None,
                     workload_jsonl: str = None) -> List[str]:
    """Load regex strings from a single regex or a JSONL workload file."""
    regexes: List[str] = []
    if regex:
        regexes.append(regex)
    if workload_jsonl:
        with open(workload_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pattern = obj.get("regex")
                if pattern:
                    regexes.append(pattern)
    return regexes


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BLARE-PG runner. Prepare indexes: CREATE EXTENSION pg_trgm; "
        "CREATE INDEX ... gin_trgm_ops; CREATE INDEX ... lower(col) gin_trgm_ops;"
    )
    parser.add_argument("--dsn", required=True, help="PostgreSQL DSN string")
    parser.add_argument("--table", required=True, help="Logs table name")
    parser.add_argument("--column", required=True, help="Log text column name")
    parser.add_argument("--id_column", default="id",
                        help="Identifier column name (primary key); use 'ctid' if the table has no ID column")
    parser.add_argument("--trgm_table", help="Optional trgm-optimized table name (default: <table>_trgm)")
    parser.add_argument("--regex", help="Single regex to run")
    parser.add_argument("--workload_jsonl", help="JSONL workload file path")
    parser.add_argument("--sample_ratio", type=float,
                        help="Sampling ratio override")
    parser.add_argument("--min_sample_rows", type=int,
                        help="Minimum sample rows override")
    parser.add_argument("--output_json", help="Path to save EvalStats JSON")
    parser.add_argument("--debug_counts", action="store_true",
                        help="When set, run extra prefilter COUNT(*) for true candidate cardinality")
    parser.add_argument("--check_correctness", action="store_true",
                        help="When set, compare direct regex count vs BLARE plan to ensure correctness")
    parser.add_argument("--workload_mode", choices=["count", "topk"], default="count",
                        help="Workload mode: count(*) or top-k retrieval")
    parser.add_argument("--limit_k", type=int, default=100,
                        help="K for topk retrieval (default 100)")
    parser.add_argument("--policy_cache", default=".blare_policy_cache.json",
                        help="Path to policy cache file")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable policy cache")
    parser.add_argument("--sampling_method", choices=["index_jump", "bucket_index_jump", "tablesample"],
                        help="Sampling method override")
    return parser.parse_args()


def _create_configs(args: argparse.Namespace) -> tuple[PgConfig, BlareConfig]:
    """Build configuration objects from CLI arguments."""
    pg_conf = PgConfig(
        dsn=args.dsn,
        table=args.table,
        column=args.column,
        id_column=args.id_column,
        trgm_table=args.trgm_table or f"{args.table}_trgm",
    )
    blare_conf = BlareConfig()
    if args.sample_ratio is not None:
        blare_conf.sample_ratio = args.sample_ratio
    if args.min_sample_rows is not None:
        blare_conf.min_sample_rows = args.min_sample_rows
    if args.sampling_method:
        blare_conf.sampling_method = args.sampling_method
    return pg_conf, blare_conf


def _load_cache(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _evaluate_regex(regex: str,
                    pg_conf: PgConfig,
                    blare_conf: BlareConfig,
                    debug_counts: bool,
                    workload_mode: str,
                    limit_k: int,
                    cache: Dict,
                    cache_path: Path,
                    use_cache: bool) -> EvalStats:
    """Evaluate a single regex with server-side prefilter + verify stats."""
    split_info = build_split_info(regex)
    _, sample_ids, sample_ok, sample_reason, sampling_method, n_sample = fetch_sample_lines(pg_conf, blare_conf)
    sample_subset = sample_ids if sample_ids else None

    cache_key = None
    policy_cache_hit = False
    best_mode = None
    cached_literals: Optional[List[str]] = None
    if use_cache:
        cache_key = hashlib.sha256(
            f"{hash_dsn(pg_conf.dsn)}::{pg_conf.table}::{pg_conf.trgm_table}::{pg_conf.column}::{regex}::{workload_mode}".encode("utf-8")
        ).hexdigest()
        entry = cache.get(cache_key)
        if entry:
            policy_cache_hit = True
            best_mode = Mode(entry["mode"])
            cached_literals = entry.get("literals", [])

    learner_reason = None
    learner_choice = None
    learner_results = {}
    if best_mode is None:
        best_mode, learner_results, learner_reason, learner_choice = choose_best_mode_greedy(
            split_info, regex, pg_conf, blare_conf, sample_subset, sample_ok, sample_reason
        )
    # Immediate log of learner decision or cache reuse
    if policy_cache_hit:
        print(
            f"[BLARE-PG][learner] regex={regex!r} cache_hit=True best={best_mode.value} literals={cached_literals}"
        )
    else:
        arms_debug = "; ".join(
            f"{m.value}:{learner_results.get(m, {}).get('median_ms')}/err={learner_results.get(m, {}).get('error')}"
            for m in learner_results.keys()
        ) if learner_results else "no_learner_results"
        print(
            f"[BLARE-PG][learner] regex={regex!r} cache_hit=False choice={learner_choice} "
            f"best={best_mode.value} arms={arms_debug}"
        )

    literals, lit_reason = select_literals_for_mode(split_info, best_mode, min_len=blare_conf.min_literal_len)
    if cached_literals is not None:
        literals = [lit for lit in cached_literals if len(lit) >= blare_conf.min_literal_len]
    reason_code = split_info.reason_code
    if learner_reason and reason_code == "OK":
        reason_code = learner_reason
    if lit_reason:
        reason_code = lit_reason
    if not literals and best_mode != Mode.DIRECT:
        best_mode = Mode.DIRECT
        reason_code = lit_reason or reason_code

    active_conf = pg_conf

    n_total = estimate_rows(pg_conf)
    n_matches, candidates_count, candidates_is_estimate, cand_source, verify_stats, gucs, index_hint, plan_guard = run_prefilter_verify(
        pg_conf,
        blare_conf,
        literals,
        regex,
        split_info.case_mode,
        workload_mode=workload_mode,
        limit_k=limit_k,
        debug_counts=debug_counts,
    )
    # If plan guard triggers on non-direct, force direct retry.
    if plan_guard and best_mode != Mode.DIRECT:
        best_mode = Mode.DIRECT
        literals = []
        reason_code = "PLAN_GUARD"
        n_matches, candidates_count, candidates_is_estimate, cand_source, verify_stats, gucs, index_hint, plan_guard = run_prefilter_verify(
            pg_conf,
            blare_conf,
            literals,
            regex,
            split_info.case_mode,
            workload_mode=workload_mode,
            limit_k=limit_k,
            debug_counts=debug_counts,
        )

    t_total_ms = verify_stats.exec_ms

    print(
        f"[BLARE-PG] regex={regex!r}, mode={best_mode.value}, split_ok={split_info.splittable}, reason={reason_code}, "
        f"case_mode={split_info.case_mode}, req_lits={literals}, est_total={n_total}, candidates={candidates_count} "
        f"(estimate={candidates_is_estimate}, source={cand_source}), matches={n_matches}, "
        f"T_verify_ms={verify_stats.exec_ms:.2f}, plan={verify_stats.plan_node_summary}, index_hint={index_hint}, "
        f"plan_guard={plan_guard}, cache_hit={policy_cache_hit}, sampling={sampling_method}({n_sample}), "
        f"learner_choice={learner_choice}, "
        f"gucs={{jit:{gucs.jit}, parallel:{gucs.max_parallel_workers_per_gather}, work_mem:{gucs.work_mem}}}"
    )

    if cache_key and not policy_cache_hit:
        cache[cache_key] = {
            "mode": best_mode.value,
            "literals": literals,
            "case_mode": split_info.case_mode,
            "k": len(literals),
            "timestamp": time.time(),
            "plan_summary": verify_stats.plan_node_summary,
        }
        _save_cache(cache_path, cache)

    return EvalStats(
        regex=regex,
        mode=best_mode,
        required_literals_used=literals,
        case_mode=split_info.case_mode,
        reason_code=reason_code,
        n_total=n_total,
        candidates_count=candidates_count,
        candidates_is_estimate=candidates_is_estimate,
        candidates_estimate_source=cand_source,
        workload_mode=workload_mode,
        limit_k=limit_k,
        plan_guard_triggered=plan_guard,
        policy_cache_hit=policy_cache_hit,
        sampling_method=sampling_method,
        n_matches=n_matches,
        t_verify_ms=verify_stats.exec_ms,
        t_total_ms=t_total_ms,
        plan_verify=verify_stats,
        prefilter_index_hint=index_hint,
        guc_settings=gucs,
    )


def main():
    """Main entry point for the BLARE-PG runner."""
    args = _parse_args()
    regexes = _load_regex_list(args.regex, args.workload_jsonl)
    if not regexes:
        raise SystemExit("At least one of --regex or --workload_jsonl is required")
    pg_conf, blare_conf = _create_configs(args)
    workload_mode = args.workload_mode
    limit_k = args.limit_k
    cache_path = Path(args.policy_cache)
    cache = {} if args.no_cache else _load_cache(cache_path)

    results: List[EvalStats] = []
    summary_reason: Dict[str, int] = {}
    case_mode_dist: Dict[str, int] = {}
    for pattern in regexes:
        result = _evaluate_regex(
            pattern,
            pg_conf,
            blare_conf,
            debug_counts=args.debug_counts,
            workload_mode=workload_mode,
            limit_k=limit_k,
            cache=cache,
            cache_path=cache_path,
            use_cache=not args.no_cache,
        )
        summary_reason[result.reason_code] = summary_reason.get(result.reason_code, 0) + 1
        case_mode_dist[result.case_mode] = case_mode_dist.get(result.case_mode, 0) + 1
        if args.check_correctness and workload_mode == "count":
            direct_count = count_direct(pg_conf, blare_conf, pattern)
            if direct_count != result.n_matches:
                print(f"[ERROR] correctness check failed for {pattern!r}: direct={direct_count}, blare={result.n_matches}, reason={result.case_mode}", file=sys.stderr)
                sys.exit(1)
        results.append(result)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "regex": r.regex,
                        "mode": r.mode.value,
                        "required_literals_used": r.required_literals_used,
                        "k": len(r.required_literals_used),
                        "case_mode": r.case_mode,
                        "reason_code": r.reason_code,
                        "n_total": r.n_total,
                        "candidates_count": r.candidates_count,
                        "candidates_is_estimate": r.candidates_is_estimate,
                        "candidates_estimate_source": r.candidates_estimate_source,
                        "n_matches": r.n_matches,
                        "workload_mode": r.workload_mode,
                        "limit_k": r.limit_k,
                        "plan_guard_triggered": r.plan_guard_triggered,
                        "policy_cache_hit": r.policy_cache_hit,
                        "sampling_method": r.sampling_method,
                        "t_verify_ms": r.t_verify_ms,
                        "t_total_ms": r.t_total_ms,
                        "plan_verify": {
                            "exec_ms": r.plan_verify.exec_ms,
                            "plan_node_summary": r.plan_verify.plan_node_summary,
                            "buffers_hit": r.plan_verify.buffers_hit,
                            "buffers_read": r.plan_verify.buffers_read,
                        },
                        "prefilter_index_hint": r.prefilter_index_hint,
                        "guc_settings": {
                            "jit": r.guc_settings.jit,
                            "max_parallel_workers_per_gather": r.guc_settings.max_parallel_workers_per_gather,
                            "work_mem": r.guc_settings.work_mem,
                        },
                    }
                    for r in results
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

    total = len(regexes) or 1
    print("[BLARE-PG] Summary:")
    print(f"  splittable OK: {summary_reason.get('OK',0)} / {total}")
    print(f"  reason_code distribution: {summary_reason}")
    print(f"  case_mode distribution: {case_mode_dist}")

    return results


if __name__ == "__main__":
    main()
