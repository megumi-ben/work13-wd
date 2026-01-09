"""Thin wrappers around the regex_span PostgreSQL extension."""
from typing import List, Optional, Sequence, Tuple


def _scalar(conn, sql: str, params: Sequence) -> Optional[object]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return row[0] if row else None


def match_span(conn, txt: str, pattern: str, start_pos: int, end_pos: int) -> bool:
    """Return True when a match spans exactly [start_pos, end_pos] (inclusive)."""
    result = _scalar(
        conn,
        "SELECT regex_match_span(%s, %s, %s, %s)",
        (txt, pattern, start_pos, end_pos),
    )
    return bool(result)


def match_from_upto(conn, txt: str, pattern: str, start_pos: int, end_pos: int) -> bool:
    """Return True when a match starts at start_pos and ends no later than end_pos."""
    result = _scalar(
        conn,
        "SELECT regex_match_from_upto(%s, %s, %s, %s)",
        (txt, pattern, start_pos, end_pos),
    )
    return bool(result)


def find_from(conn, txt: str, pattern: str, start_pos: int, allow_empty: bool = False) -> Optional[Tuple[int, int]]:
    """First match at or after start_pos; returns (mstart, mend) inclusive or None."""
    res = _scalar(
        conn,
        "SELECT regex_find_from(%s, %s, %s, %s)",
        (txt, pattern, start_pos, allow_empty),
    )
    if res is None:
        return None
    return tuple(res)


def find_at_upto(conn, txt: str, pattern: str, start_pos: int, end_pos: int, allow_empty: bool = False) -> Optional[Tuple[int, int]]:
    """First match that starts at start_pos and ends <= end_pos; returns (mstart, mend) or None."""
    res = _scalar(
        conn,
        "SELECT regex_find_at_upto(%s, %s, %s, %s, %s)",
        (txt, pattern, start_pos, end_pos, allow_empty),
    )
    if res is None:
        return None
    return tuple(res)


def find_all(conn, txt: str, pattern: str, start_pos: int = 1, overlap: bool = True, allow_empty: bool = False) -> List[Tuple[int, int]]:
    """All match spans from start_pos forward; overlap controls whether overlapping matches are returned."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT regex_find_all(%s, %s, %s, %s, %s)",
            (txt, pattern, start_pos, overlap, allow_empty),
        )
        rows = cur.fetchall()
    return [tuple(row[0]) for row in rows]


def verify_windows(conn, txt: str, pattern: str, windows: List[int]) -> int:
    """Batch verify windows; windows is flattened [s1,e1,s2,e2,...]. Returns first hit window_id or -1."""
    result = _scalar(
        conn,
        "SELECT regex_verify_windows(%s, %s, %s)",
        (txt, pattern, windows),
    )
    return int(result) if result is not None else -1
