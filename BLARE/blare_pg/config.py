"""Configuration objects for PostgreSQL and BLARE-PG."""
from dataclasses import dataclass


@dataclass
class PgConfig:
    """PostgreSQL connection and table configuration."""

    dsn: str
    table: str
    column: str
    id_column: str = "id"
    trgm_table: str = ""


@dataclass
class BlareConfig:
    """
    BLARE sampling and learner configuration.

    Attributes:
        sample_ratio: Ratio of ID span to sample for mode selection.
        min_sample_rows: Minimum rows to sample for benchmarking.
        max_sample_rows: Upper bound on sampled rows to avoid huge IN lists.
        warmup_runs: Warmup runs per arm before measurement.
        measure_runs: Measurement runs per arm (median taken).
        direct_prefer_threshold: If best arm is within this fractional gap of direct, choose direct.
        enforce_gucs: Whether to set deterministic GUCs (jit off, no parallel).
        work_mem: work_mem setting applied when enforce_gucs is True.
        disable_jit: When True, `SET jit = off`.
        disable_parallel: When True, `SET max_parallel_workers_per_gather = 0`.
        inline_like_literals: Inline LIKE patterns as SQL Literal (planner sees constants).
        sampling_method: index_jump | bucket_index_jump | tablesample.
        bucket_count: number of buckets for bucket_index_jump.
        tablesample_pct: percentage for TABLESAMPLE fallback/option.
        min_literal_len: literals shorter than this are ignored for prefilter.
    """

    sample_ratio: float = 0.05
    min_sample_rows: int = 200
    max_sample_rows: int = 500000
    warmup_runs: int = 1
    measure_runs: int = 5
    direct_prefer_threshold: float = 0.03
    enforce_gucs: bool = True
    work_mem: str = "4MB"
    disable_jit: bool = True
    disable_parallel: bool = True
    inline_like_literals: bool = True
    sampling_method: str = "tablesample"
    bucket_count: int = 20
    tablesample_pct: float = 0.5
    min_literal_len: int = 2
    plan_guard_enabled: bool = False
