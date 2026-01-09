"""Prefilter literal selection utilities (no client-side matching)."""
from typing import List, Tuple

from .types import Mode, SplitInfo


def select_literals_for_mode(split_info: SplitInfo,
                             mode: Mode,
                             k_multi: int = 3,
                             min_len: int = 2) -> Tuple[List[str], str]:
    """
    Choose required literals to use in PostgreSQL prefilter for a given arm.

    Returns (literals, reason_code_override). Falls back to empty when unsplittable
    or no literals available.
    """
    if not split_info.splittable:
        return [], split_info.reason_code
    if mode == Mode.DIRECT:
        return [], None

    # Prefer longer literals for better selectivity while keeping K small.
    sorted_lits = [lit for lit in split_info.required_literals if len(lit) >= min_len]
    sorted_lits = sorted(sorted_lits, key=lambda s: len(s), reverse=True)
    if not sorted_lits:
        return [], "LITS_TOO_SHORT_FOR_TRGM"

    if mode == Mode.THREE_WAY:
        return (sorted_lits[:2] if len(sorted_lits) >= 2 else sorted_lits[:1], None)
    if mode == Mode.MULTI_WAY:
        return (sorted_lits[:k_multi], None)

    raise ValueError(f"Unsupported mode: {mode}")
