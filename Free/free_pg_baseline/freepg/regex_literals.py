from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Union

from . import config

LiteralGroup = Union[str, List[str]]


@dataclass
class ExtractResult:
    safe: bool
    literals: List[LiteralGroup]
    reason_if_unsafe: str = ""
    used_template: bool = False


def extract_literals(pattern: str, min_len: int = config.DEFAULT_LITERAL_MIN_LEN) -> ExtractResult:
    if re.search(r"\\[1-9]", pattern):
        return ExtractResult(False, [], "backreference", False)
    # Fast-path template for generator-style patterns
    tpl = template_extract(pattern, min_len)
    if tpl:
        return tpl
    literals: List[LiteralGroup] = []
    safe = True
    reason = ""
    i = 0
    current = []
    pattern_no_boundary = pattern.replace(r"\y", "")
    n = len(pattern_no_boundary)

    def flush_current():
        nonlocal literals, current
        if current:
            lit = "".join(current)
            if len(lit) >= min_len:
                literals.append(lit)
            current = []

    while i < n:
        ch = pattern_no_boundary[i]
        nxt = pattern_no_boundary[i + 1] if i + 1 < n else ""
        if ch == "\\":
            if i + 1 >= n:
                break
            esc = pattern_no_boundary[i + 1]
            current.append(escape_char(esc))
            i += 2
            continue
        if (
            pattern_no_boundary.startswith("(?:.|\\n){", i)
            or pattern_no_boundary.startswith("(?:.|\\\\n){", i)
            or pattern_no_boundary.startswith("(?:.|\n){", i)
        ):
            # gap
            flush_current()
            quantifier_start = pattern_no_boundary.find("{", i)
            gap_end = pattern_no_boundary.find("}", i)
            if quantifier_start == -1 or gap_end == -1 or gap_end < quantifier_start:
                safe, reason = False, "unterminated gap quantifier"
                break
            quantifier = pattern_no_boundary[quantifier_start + 1 : gap_end]
            if not quantifier.startswith("0,") or not quantifier[2:].isdigit():
                safe, reason = False, "unsupported gap quantifier"
                break
            # optional trailing '?'
            i = gap_end + 1
            if i < n and pattern_no_boundary[i] == "?":
                i += 1
            continue
        if ch == "(":
            # only allow non-capturing groups
            if not pattern_no_boundary.startswith("(?:", i):
                safe, reason = False, "capturing or unsupported group"
                break
            end = find_group_end(pattern_no_boundary, i + 3)
            if end == -1:
                safe, reason = False, "unterminated group"
                break
            flush_current()
            content = pattern_no_boundary[i + 3 : end]
            group_safe, token_or_reason = parse_non_capturing(content, min_len)
            if not group_safe:
                safe, reason = False, token_or_reason
                break
            if token_or_reason:
                literals.append(token_or_reason)
            i = end + 1
            continue
        if ch == "[":
            # character class => separator
            end = find_class_end(pattern_no_boundary, i + 1)
            if end == -1:
                safe, reason = False, "unterminated character class"
                break
            flush_current()
            i = end + 1
            continue
        if ch in "*+?{}|":
            safe, reason = False, f"unsupported quantifier or alternation near {ch}"
            break
        current.append(ch)
        i += 1
    flush_current()
    if not safe:
        return ExtractResult(False, [], reason, False)
    return ExtractResult(True, literals, "", False)


def escape_char(ch: str) -> str:
    if ch == "y":
        return ""
    return {
        "n": "\n",
        "t": "\t",
    }.get(ch, ch)


def find_group_end(pattern: str, start: int) -> int:
    depth = 1
    i = start
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def find_class_end(pattern: str, start: int) -> int:
    i = start
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "]":
            return i
        i += 1
    return -1


def parse_non_capturing(content: str, min_len: int):
    # support literal OR of pure literals only
    if "|" in content:
        parts = split_unescaped(content, "|")
        opts: List[str] = []
        for p in parts:
            if not is_pure_literal(p):
                return False, "non-literal alternation"
            if len(p) >= min_len:
                opts.append(unescape_literal(p))
        if not opts:
            return True, None
        return True, opts
    if is_pure_literal(content):
        lit = unescape_literal(content)
        if len(lit) >= min_len:
            return True, lit
        return True, None
    return False, "unsupported group content"


def is_pure_literal(text: str) -> bool:
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch in "[](){}*+?|":
            return False
        i += 1
    return True


def unescape_literal(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            out.append(escape_char(text[i + 1]))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def split_unescaped(text: str, sep: str) -> List[str]:
    out: List[str] = []
    current = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            if i + 1 < len(text):
                current.append(text[i])
                current.append(text[i + 1])
                i += 2
                continue
        if ch == sep:
            out.append("".join(current))
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    out.append("".join(current))
    return out


def template_extract(pattern: str, min_len: int) -> Optional[ExtractResult]:
    """
    Template fast-path for workload patterns:
    - \y token \y (gap)? token \y ...
    - token may include literal runs and character classes with quantifiers.
    """
    safe = True
    reason = ""
    literals: List[LiteralGroup] = []
    used = False
    i = 0
    n = len(pattern)
    current = []

    def flush():
        nonlocal literals, current
        if current:
            lit = "".join(current)
            if len(lit) >= min_len:
                literals.append(lit)
            current = []

    while i < n:
        ch = pattern[i]
        if ch == "\\":
            if i + 1 >= n:
                break
            nxt = pattern[i + 1]
            if nxt == "y":
                flush()
                i += 2
                used = True
                continue
            current.append(escape_char(nxt))
            i += 2
            continue
        if pattern.startswith("(?:.|\\n){", i) or pattern.startswith("(?:.|\\\\n){", i) or pattern.startswith("(?:.|\n){", i):
            flush()
            used = True
            quantifier_start = pattern.find("{", i)
            quantifier_end = pattern.find("}", i)
            if quantifier_start == -1 or quantifier_end == -1 or quantifier_end < quantifier_start:
                safe, reason = False, "unterminated gap quantifier"
                break
            q = pattern[quantifier_start + 1 : quantifier_end]
            if not q.startswith("0,"):
                safe, reason = False, "unsupported gap quantifier"
                break
            i = quantifier_end + 1
            if i < n and pattern[i] == "?":
                i += 1
            continue
        if pattern.startswith("(?:", i):
            end = find_group_end(pattern, i + 3)
            if end == -1:
                safe, reason = False, "unterminated group"
                break
            content = pattern[i + 3 : end]
            group_safe, token_or_reason = parse_non_capturing(content, min_len)
            flush()
            if not group_safe:
                safe, reason = False, token_or_reason
                break
            if token_or_reason:
                literals.append(token_or_reason)
            used = True
            i = end + 1
            continue
        if ch == "[":
            end = find_class_end(pattern, i + 1)
            if end == -1:
                safe, reason = False, "unterminated class"
                break
            flush()
            used = True
            i = end + 1
            # optional quantifier after class
            if i < n and pattern[i] == "{":
                close = pattern.find("}", i)
                if close == -1:
                    safe, reason = False, "unterminated quantifier"
                    break
                i = close + 1
            continue
        if ch in "*+?|":
            safe, reason = False, f"unsupported quantifier near {ch}"
            break
        current.append(ch)
        i += 1
    flush()
    if not used:
        return None
    if not safe:
        return ExtractResult(False, [], reason, True)
    return ExtractResult(True, literals, "", True)
