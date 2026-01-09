"""Common types and enums used across BLARE-PG."""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Mode(str, Enum):
    """Execution modes (arms) for regex matching."""

    DIRECT = "direct"
    THREE_WAY = "3way"
    MULTI_WAY = "multiway"


@dataclass
class SplitInfo:
    """
    Conservative split information derived from a regex string.

    Attributes:
        raw: Original regex string.
        required_literals: Literals that must occur in any true match; safe for prefiltering.
        splittable: Whether required_literals is proven sound (otherwise fall back to direct).
        reason_code: Machine-friendly reason for split decision.
        reason_detail: Human-readable reason when unsplittable or degraded.
        case_mode: "sensitive" | "insensitive" | "unknown" derived from regex flags.
    """

    raw: str
    required_literals: List[str]
    splittable: bool
    reason_code: str = "OK"
    reason_detail: str = ""
    case_mode: str = "sensitive"


@dataclass
class ExplainStats:
    """Execution stats extracted from EXPLAIN (ANALYZE, FORMAT JSON)."""

    exec_ms: float
    plan_node_summary: str
    buffers_hit: int
    buffers_read: int


@dataclass
class GucSettings:
    """Session-level GUCs used for measurements."""

    jit: Optional[str] = None
    max_parallel_workers_per_gather: Optional[str] = None
    work_mem: Optional[str] = None


@dataclass
class EvalStats:
    """
    Evaluation statistics for one regex under a specific arm.

    All timings are server-side (from EXPLAIN ANALYZE) to avoid client noise.
    """

    regex: str
    mode: Mode
    required_literals_used: List[str]
    case_mode: str
    reason_code: str
    n_total: int
    candidates_count: int
    candidates_is_estimate: bool
    candidates_estimate_source: str
    workload_mode: str
    limit_k: int
    plan_guard_triggered: bool
    policy_cache_hit: bool
    sampling_method: str
    n_matches: int
    t_verify_ms: float
    t_total_ms: float
    plan_verify: ExplainStats
    prefilter_index_hint: str
    guc_settings: GucSettings

    @property
    def k(self) -> int:
        """Number of required literals used in the prefilter."""
        return len(self.required_literals_used)
