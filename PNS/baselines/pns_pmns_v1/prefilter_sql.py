"""Helpers to build safe prefilter SQL for necessary literals."""
from typing import Iterable, List, Tuple


def escape_like_literal(lit: str, escape_char: str = "\\") -> str:
    if not lit:
        return lit
    return (
        lit.replace(escape_char, escape_char + escape_char)
        .replace("%", escape_char + "%")
        .replace("_", escape_char + "_")
    )


def build_where_for_necessary_literals(col_sql: str, lits: List[str], use_ilike: bool = False, k: int = 2) -> Tuple[str, List[str]]:
    uniq = []
    seen = set()
    for lit in sorted(lits, key=lambda x: (-len(x), x)):
        if len(lit) < 2:
            continue
        if lit in seen:
            continue
        uniq.append(lit)
        seen.add(lit)
        if len(uniq) >= k:
            break
    if not uniq:
        return "", []
    op = "ILIKE" if use_ilike else "LIKE"
    frags = []
    params: List[str] = []
    for lit in uniq:
        esc = escape_like_literal(lit)
        frags.append(f"({col_sql} {op} %s ESCAPE E'\\\\')")
        params.append(f"%{esc}%")
    return " AND ".join(frags), params


def merge_with_user_where(user_where: str, extra_fragment: str) -> str:
    if not extra_fragment:
        return user_where or ""
    if user_where:
        return f"({user_where}) AND ({extra_fragment})"
    return extra_fragment
