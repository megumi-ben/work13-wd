"""Lightweight literal search helpers (Unicode-safe, 1-based positions)."""
from bisect import bisect_left
from typing import Dict, Iterable, List


def find_all_literals(text: str, lit: str, overlap: bool = True) -> List[int]:
    """Return 1-based start positions of literal occurrences."""
    if not lit:
        return []
    positions: List[int] = []
    start = 0
    step = 1 if overlap else len(lit)
    while True:
        idx = text.find(lit, start)
        if idx == -1:
            break
        positions.append(idx + 1)  # 1-based
        start = idx + step
    return positions


def occurs_in_window(pos_list: List[int], s: int, e: int, lit_len: int) -> bool:
    """Return True if there exists p in pos_list with s <= p <= e - lit_len + 1."""
    if not pos_list or lit_len <= 0:
        return False
    upper = e - lit_len + 1
    if upper < s:
        return False
    idx = bisect_left(pos_list, s)
    if idx >= len(pos_list):
        return False
    return pos_list[idx] <= upper


def find_all_many(text: str, lits: List[str], overlap: bool = True) -> Dict[str, List[int]]:
    """Find occurrences for many literals, returning dict literal->positions."""
    return {lit: find_all_literals(text, lit, overlap=overlap) for lit in lits if lit}


class OccurrenceCache:
    """Cache literal occurrences per text to avoid repeated scans."""

    def __init__(self, text: str):
        self.text = text
        self.cache: Dict[tuple, List[int]] = {}

    def find(self, lit: str, overlap: bool = True) -> List[int]:
        if not lit:
            return []
        key = (lit, overlap)
        if key in self.cache:
            return self.cache[key]
        positions = find_all_literals(self.text, lit, overlap=overlap)
        self.cache[key] = positions
        return positions

    def find_many(self, lits: Iterable[str], overlap: bool = True) -> Dict[str, List[int]]:
        result: Dict[str, List[int]] = {}
        for lit in lits:
            if not lit:
                continue
            result[lit] = self.find(lit, overlap=overlap)
        return result
