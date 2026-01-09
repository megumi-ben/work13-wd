"""Lightweight anchor extraction for workload-specific regexes."""
import re
from typing import Dict, List, Optional, Tuple


def _extract_islands(pattern: str) -> List[str]:
    """Find segments wrapped by \y ... \y in order."""
    islands: List[str] = []
    idx = 0
    while True:
        start = pattern.find(r"\y", idx)
        if start == -1:
            break
        end = pattern.find(r"\y", start + 2)
        if end == -1:
            break
        content = pattern[start + 2 : end]
        if content:
            islands.append(content)
        idx = end + 2
    return islands


def _choose_anchor(island: str) -> Tuple[str, int]:
    """
    Pick a stable anchor inside an island and return (pattern, minlen).
    Prefer a literal run to shorten the anchor; otherwise fall back to the full island with minlen=1.
    """
    literal = re.search(r"([A-Za-z0-9_]{2,})", island)
    if literal:
        frag = literal.group(1)
        return r"\y" + re.escape(frag) + r"\y", len(frag)
    return r"\y" + island + r"\y", 1


def anchor_patterns(pattern: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Return (prefix_anchor, suffix_anchor, fallback_used).

    Anchors are non-empty strings usable directly as regex patterns. If extraction
    fails, both anchors are None and fallback_used is True.
    """
    islands = _extract_islands(pattern)
    if not islands:
        return None, None, True

    prefix = _choose_anchor(islands[0])
    suffix = _choose_anchor(islands[-1])
    return prefix, suffix, False


def parse_structured_regex(regex_text: str) -> Optional[Dict]:
    """
    Parse limited generator-style regex with optional leading gap and \y islands.

    Returns dict with:
      - first_island_pattern
      - last_island_pattern
      - leading_gap_max (int)
      - trailing_gap_max (int, best-effort; 0 if none/unknown)
    """
    islands = []
    positions = []
    idx = 0
    while True:
        start = regex_text.find(r"\y", idx)
        if start == -1:
            break
        end = regex_text.find(r"\y", start + 2)
        if end == -1:
            break
        content = regex_text[start + 2 : end]
        if content:
            islands.append(content)
            positions.append((start, end + 2))
        idx = end + 2

    if not islands:
        return None

    leading_part = regex_text[: positions[0][0]]
    leading_gap_max = 0
    if leading_part:
        m = re.fullmatch(r"\(\?:\.\|\\n\)\{0,(\d+)\}\?", leading_part)
        if m:
            leading_gap_max = int(m.group(1))
        else:
            return None  # unknown leading construct; fallback

    trailing_part = regex_text[positions[-1][1] :]
    trailing_gap_max = 0
    if trailing_part:
        m = re.fullmatch(r"\(\?:\.\|\\n\)\{0,(\d+)\}\?", trailing_part)
        if m:
            trailing_gap_max = int(m.group(1))
        else:
            trailing_gap_max = 0  # ignore unknown trailing; not used yet

    first_island_pattern, first_minlen = _choose_anchor(islands[0])
    last_island_pattern, last_minlen = _choose_anchor(islands[-1])

    return {
        "first_island_pattern": first_island_pattern,
        "last_island_pattern": last_island_pattern,
        "leading_gap_max": leading_gap_max,
        "trailing_gap_max": trailing_gap_max,
        "first_anchor_minlen": first_minlen,
        "last_anchor_minlen": last_minlen,
    }
