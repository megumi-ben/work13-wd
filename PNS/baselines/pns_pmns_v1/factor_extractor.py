"""
Factor extractor for the workload-specific regex subset (gaps + \y islands).

Conservative: on any uncertainty, returns None so callers can fallback.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import re


@dataclass
class Island:
    raw: str
    necessary_literals: List[str] = field(default_factory=list)
    anchor_literals: List[str] = field(default_factory=list)
    min_len: int = 0
    max_len: Optional[int] = None


@dataclass
class Factors:
    lmin: int
    min_len: int
    max_len: Optional[int]
    leading_gap_max: int
    trailing_gap_max: int
    islands: List[Island]
    parse_ok: bool = True
    fallback_reason: Optional[str] = None
    num_M_literals: int = 0


_GAP_RE = re.compile(r"\(\?:\.\|\\n\)\{0,(\d+)\}\?")


def _scan_literal_runs(s: str) -> List[str]:
    runs: List[str] = []
    buf: List[str] = []
    in_class = False
    escaped = False
    for ch in s:
        if escaped:
            escaped = False
            if buf:
                if len(buf) >= 2:
                    runs.append("".join(buf))
                buf = []
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "[":
            if buf and len(buf) >= 2:
                runs.append("".join(buf))
            buf = []
            in_class = True
            continue
        if ch == "]" and in_class:
            in_class = False
            continue
        if in_class:
            continue
        if ch in "(){}|*+?.^$" or ch.isspace():
            if buf and len(buf) >= 2:
                runs.append("".join(buf))
            buf = []
            continue
        if ch.isalnum() or ch == "_":
            buf.append(ch)
        else:
            if buf and len(buf) >= 2:
                runs.append("".join(buf))
            buf = []
    if buf and len(buf) >= 2:
        runs.append("".join(buf))
    return runs


def _has_top_level_or(s: str) -> bool:
    depth = 0
    in_class = False
    escaped = False
    for ch in s:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "[":
            in_class = True
            continue
        if ch == "]" and in_class:
            in_class = False
            continue
        if in_class:
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            continue
        if ch == "|" and depth == 0:
            return True
    return False


def _split_top_level_or(body: str) -> Optional[List[str]]:
    depth = 0
    in_class = False
    escaped = False
    parts: List[str] = []
    buf: List[str] = []
    for ch in body:
        if escaped:
            escaped = False
            buf.append(ch)
            continue
        if ch == "\\":
            escaped = True
            buf.append(ch)
            continue
        if ch == "[":
            in_class = True
            buf.append(ch)
            continue
        if ch == "]" and in_class:
            in_class = False
            buf.append(ch)
            continue
        if in_class:
            buf.append(ch)
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth -= 1
            if depth < 0:
                return None
            buf.append(ch)
            continue
        if ch == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if depth != 0 or escaped or in_class:
        return None
    parts.append("".join(buf))
    return parts


_PURE_RE = re.compile(r"^[A-Za-z0-9_\-\'\. ]+$")


def _is_pure_literal_branch(branch: str) -> bool:
    if "\\" in branch:
        return False
    if any(ch in "()[]{}|*+?.^$" for ch in branch):
        return False
    return bool(_PURE_RE.match(branch))


def _common_substrings(branches: List[str], min_len: int = 2, limit: int = 256) -> List[str]:
    if not branches:
        return []
    shortest = min(branches, key=len)
    commons = set()
    for i in range(len(shortest)):
        for j in range(i + min_len, len(shortest) + 1):
            commons.add(shortest[i:j])
            if len(commons) >= limit:
                break
        if len(commons) >= limit:
            break
    commons = [c for c in commons if all(c in b for b in branches)]
    commons.sort(key=lambda x: (-len(x), x))
    return commons


def _split_islands(pattern: str) -> Tuple[List[str], int, int, List[int]]:
    islands = []
    leading_gap_max = 0
    trailing_gap_max = 0
    gap_max_list: List[int] = []
    m = _GAP_RE.match(pattern)
    if m:
        leading_gap_max = int(m.group(1))
    for m_all in _GAP_RE.finditer(pattern):
        gap_max_list.append(int(m_all.group(1)))
    m_tail = _GAP_RE.search(pattern[::-1])
    if pattern.endswith("}?") and _GAP_RE.search(pattern):
        m2 = _GAP_RE.search(pattern)
        if m2:
            trailing_gap_max = int(m2.group(1))
    idx = 0
    while True:
        start = pattern.find(r"\y", idx)
        if start == -1:
            break
        end = pattern.find(r"\y", start + 2)
        if end == -1:
            break
        islands.append(pattern[start + 2 : end])
        idx = end + 2
    return islands, leading_gap_max, trailing_gap_max, gap_max_list


def _has_top_level_or_body(body: str) -> bool:
    return _has_top_level_or(body)


def _estimate_island_len(raw: str) -> Tuple[int, Optional[int]]:
    i = 0
    n = len(raw)
    min_len = 0
    max_len: Optional[int] = 0
    while i < n:
        ch = raw[i]
        if ch == "[":
            j = raw.find("]", i + 1)
            if j == -1:
                return min_len, None
            token_min = token_max = 1
            k = j + 1
            if k < n and raw[k] == "{":
                k += 1
                num1 = ""
                while k < n and raw[k].isdigit():
                    num1 += raw[k]
                    k += 1
                if k < n and raw[k] == ",":
                    k += 1
                    num2 = ""
                    while k < n and raw[k].isdigit():
                        num2 += raw[k]
                        k += 1
                    if k < n and raw[k] == "}":
                        token_min = int(num1) if num1 else 0
                        token_max = int(num2) if num2 else token_min
                        k += 1
                    else:
                        return min_len, None
                elif k < n and raw[k] == "}":
                    if not num1:
                        return min_len, None
                    token_min = token_max = int(num1)
                    k += 1
                else:
                    return min_len, None
            min_len += token_min
            if max_len is not None:
                max_len += token_max
            i = k
            continue
        if ch.isalnum() or ch == "_":
            j = i
            while j < n and (raw[j].isalnum() or raw[j] == "_"):
                j += 1
            token = raw[i:j]
            token_len = len(token)
            i = j
            # optional quantifier {a,b}
            if i < n and raw[i] == "{":
                i += 1
                num1 = ""
                while i < n and raw[i].isdigit():
                    num1 += raw[i]
                    i += 1
                if i < n and raw[i] == ",":
                    i += 1
                    num2 = ""
                    while i < n and raw[i].isdigit():
                        num2 += raw[i]
                        i += 1
                    if i < n and raw[i] == "}":
                        a = int(num1) if num1 else 0
                        b = int(num2) if num2 else a
                        i += 1
                    else:
                        return min_len, None
                elif i < n and raw[i] == "}":
                    if not num1:
                        return min_len, None
                    a = b = int(num1)
                    i += 1
                else:
                    return min_len, None
                min_len += token_len * a
                if max_len is not None:
                    max_len += token_len * b
            else:
                min_len += token_len
                if max_len is not None:
                    max_len += token_len
            continue
        # unknown/complex char => max unknown
        max_len = None
        i += 1
    return min_len, max_len


def extract_factors(pattern: str) -> Optional[Factors]:
    islands_raw, leading_gap_max, trailing_gap_max, gap_max_list = _split_islands(pattern)
    if not islands_raw:
        return None

    islands: List[Island] = []
    total_min = 0
    total_max_sum = 0
    total_max_unknown = False
    for raw in islands_raw:
        runs = _scan_literal_runs(raw)
        necessary_literals: List[str] = []
        anchor_literals: List[str] = []

        # OR handling: only when raw is exactly a non-capturing group with top-level OR and pure literal branches
        if raw.startswith("?:") and raw.endswith(")"):
            body = raw[2:-1]
        elif raw.startswith("(?:") and raw.endswith(")"):
            body = raw[3:-1]
        else:
            body = None
        has_or = False
        if body and _has_top_level_or_body(body):
            has_or = True
            branches = _split_top_level_or(body)
            if branches and all(_is_pure_literal_branch(b) for b in branches):
                commons = _common_substrings(branches, min_len=2, limit=256)
                if commons:
                    necessary_literals = commons[:2]
        if not necessary_literals and not has_or and not _has_top_level_or(raw):
            if runs:
                necessary_literals = [sorted(runs, key=lambda x: (-len(x), x))[0]]

        if necessary_literals:
            anchor_literals = sorted(necessary_literals, key=lambda x: (-len(x), x))[:2]
        elif runs:
            anchor_literals = sorted(runs, key=lambda x: (-len(x), x))[:2]

        # Length bounds per island
        min_len, max_len = _estimate_island_len(raw)
        if max_len is None:
            total_max_unknown = True
        else:
            total_max_sum += max_len
        total_min += min_len

        islands.append(
            Island(
                raw=raw,
                necessary_literals=necessary_literals,
                anchor_literals=anchor_literals,
                min_len=min_len,
                max_len=max_len,
            )
        )

    gap_max_total = sum(gap_max_list) if gap_max_list else 0
    total_max = None if total_max_unknown else total_max_sum + gap_max_total
    factors = Factors(
        lmin=total_min,
        min_len=total_min,
        max_len=total_max,
        leading_gap_max=leading_gap_max,
        trailing_gap_max=trailing_gap_max,
        islands=islands,
        num_M_literals=sum(len(i.necessary_literals) for i in islands),
    )
    return factors
