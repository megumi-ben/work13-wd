"""Learner that chooses the best execution arm using server-side timing."""
from statistics import median
from typing import Dict, List, Optional, Tuple

from .config import BlareConfig, PgConfig
from .pg_io import run_prefilter_verify
from .split_matcher import select_literals_for_mode
from .types import Mode, SplitInfo

ALL_MODES = [Mode.DIRECT, Mode.THREE_WAY, Mode.MULTI_WAY]


def _measure_arm(mode: Mode,
                 split_info: SplitInfo,
                 regex: str,
                 pg_conf: PgConfig,
                 blare_conf: BlareConfig,
                 sample_ids: Optional[List[int]],
                 case_mode: str) -> Tuple[Optional[float], List[str], Optional[str], Optional[str]]:
    """
    Measure one arm using EXPLAIN server time on a sample subset.

    Returns:
        (median_ms, literals_used, error_reason, reason_code_override)
    """
    literals, lit_reason = select_literals_for_mode(split_info, mode, min_len=blare_conf.min_literal_len)
    if mode != Mode.DIRECT and (not split_info.splittable or not literals):
        reason = lit_reason or "unsplittable_or_no_literals"
        return None, [], reason, lit_reason

    try:
        effective_ids = sample_ids
        # Warmup runs (ignored timings).
        for _ in range(blare_conf.warmup_runs):
            _, _, _, _, _, _, _, guard = run_prefilter_verify(
                pg_conf,
                blare_conf,
                literals,
                regex,
                case_mode,
                sample_ids=effective_ids,
            )
            if guard:
                return None, literals, "plan_guard_triggered", "PLAN_GUARD"

        times: List[float] = []
        for _ in range(blare_conf.measure_runs):
            _, _, _, _, stats, _, _, guard = run_prefilter_verify(
                pg_conf,
                blare_conf,
                literals,
                regex,
                case_mode,
                sample_ids=effective_ids,
            )
            if guard:
                return None, literals, "plan_guard_triggered", "PLAN_GUARD"
            times.append(stats.exec_ms)
        med = median(times) if times else None
        return med, literals, None, lit_reason
    except Exception as exc:
        import traceback
        tb = traceback.format_exc(limit=6)
        return None, literals, f"{type(exc).__name__}: {exc}; traceback={tb}", lit_reason


def choose_best_mode_greedy(split_info: SplitInfo,
                            regex: str,
                            pg_conf: PgConfig,
                            blare_conf: BlareConfig,
                            sample_ids: Optional[List[int]],
                            sample_ok: bool,
                            sample_reason: str) -> Tuple[Mode, Dict[Mode, Dict], Optional[str], Optional[str]]:
    """
    Benchmark all arms using server-side EXPLAIN timing and return the best.

    If the best arm is within direct_prefer_threshold of DIRECT, choose DIRECT.
    """
    if not sample_ok:
        # Fall back to direct when sampling is insufficient.
        return Mode.DIRECT, {Mode.DIRECT: {"median_ms": None, "literals": [], "error": sample_reason}}, sample_reason, "sample_insufficient"

    results: Dict[Mode, Dict] = {}
    best_mode = Mode.DIRECT
    best_time = float("inf")
    direct_time: Optional[float] = None

    for mode in ALL_MODES:
        med, literals, error, reason_override = _measure_arm(
            mode,
            split_info,
            regex,
            pg_conf,
            blare_conf,
            sample_ids,
            split_info.case_mode,
        )
        results[mode] = {"median_ms": med, "literals": literals, "error": error, "reason_override": reason_override}
        if med is None:
            continue
        if mode == Mode.DIRECT:
            direct_time = med
        if med < best_time:
            best_time = med
            best_mode = mode

    choice_reason = f"best={best_mode.value}"
    if direct_time is not None and best_time < float("inf"):
        # Prefer direct only when it is within the threshold of the best arm.
        # i.e., best is not significantly faster than direct.
        if best_time >= direct_time * (1 - blare_conf.direct_prefer_threshold):
            best_mode = Mode.DIRECT
            choice_reason = "direct_within_threshold"
    per_arm_entries = []
    for m in ALL_MODES:
        res = results.get(m, {})
        med = res.get("median_ms")
        err = res.get("error")
        if isinstance(med, (int, float)):
            per_arm_entries.append(f"{m.value}:{med:.4f}")
        else:
            per_arm_entries.append(f"{m.value}:err={err}")
    per_arm = ", ".join(per_arm_entries)
    if not choice_reason:
        choice_reason = f"best={best_mode.value}"
    choice_reason = f"{choice_reason}; per_arm={per_arm}"

    return best_mode, results, None, choice_reason
