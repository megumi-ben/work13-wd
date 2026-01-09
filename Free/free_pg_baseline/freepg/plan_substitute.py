from typing import Dict, Iterable, List

from . import config
from .db import load_keys
from .plan import PlanNode, simplify


def load_key_set(conn, index_id: str) -> set:
    return set(load_keys(conn, index_id).keys())


def substitute_plan(
    conn,
    node: PlanNode,
    index_id: str,
    lmax: int = config.DEFAULT_LMAX,
    max_subs: int = config.DEFAULT_SUBSTRING_CHOICES,
) -> PlanNode:
    key_set = load_key_set(conn, index_id)
    df_cache: Dict[str, int] = {}
    substituted = _substitute(node, key_set, df_cache, conn, index_id, lmax, max_subs)
    return simplify(substituted)


def _substitute(
    node: PlanNode,
    key_set: set,
    df_cache: Dict[str, int],
    conn,
    index_id: str,
    lmax: int,
    max_subs: int,
) -> PlanNode:
    if node.node_type == "LEAF":
        lit = node.value or ""
        if lit in key_set:
            return node
        candidates = [(s, pos) for s, pos in substrings(lit, lmax) if s in key_set]
        if not candidates:
            return PlanNode("TRUE")
        grams_only = [g for g, _ in candidates]
        df_map = fetch_df(conn, index_id, grams_only, df_cache)
        ordered = pick_substrings(candidates, df_map, max_subs)
        children = [PlanNode("LEAF", g) for g in ordered]
        if len(children) == 1:
            return children[0]
        return PlanNode("AND", children=children)
    if node.node_type in ("AND", "OR"):
        kids = [_substitute(c, key_set, df_cache, conn, index_id, lmax, max_subs) for c in node.children]
        return PlanNode(node.node_type, children=kids)
    return node


def substrings(literal: str, lmax: int) -> List[tuple]:
    out: List[tuple] = []
    n = len(literal)
    for i in range(n):
        for l in range(2, min(lmax, n - i) + 1):
            out.append((literal[i : i + l], i))
    return out


def fetch_df(conn, index_id: str, grams: Iterable[str], cache: Dict[str, int]) -> Dict[str, int]:
    missing = [g for g in grams if g not in cache]
    if not missing:
        return cache
    with conn.cursor() as cur:
        cur.execute(
            "SELECT gram, df_full FROM free_index WHERE index_id=%s AND gram = ANY(%s)",
            (index_id, missing),
        )
        for gram, df in cur.fetchall():
            cache[gram] = int(df)
    return cache


def pick_substrings(candidates: List[tuple], df_map: Dict[str, int], max_subs: int) -> List[str]:
    # candidates: list of (substr, pos)
    scored = sorted(
        candidates,
        key=lambda t: (df_map.get(t[0], 1_000_000_000), len(t[0]), t[1]),
    )
    chosen: List[tuple] = []
    for substr, pos in scored:
        if len(chosen) >= max_subs:
            break
        redundant = False
        for s2, pos2 in chosen:
            if substr in s2 and df_map.get(substr, 0) >= df_map.get(s2, 0):
                redundant = True
                break
            if s2 in substr and df_map.get(s2, 0) <= df_map.get(substr, 0):
                redundant = True
                break
            if abs(pos - pos2) <= 1:
                # too close; keep more selective one
                if df_map.get(substr, 0) >= df_map.get(s2, 0):
                    redundant = True
                    break
        if not redundant:
            chosen.append((substr, pos))
    return [s for s, _ in chosen[:max_subs]]
