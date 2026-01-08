#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
import random
import re
import time
import traceback
import zlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================================================================================
# Tokenization
# =====================================================================================
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
ALPHA_RE = re.compile(r"^[A-Za-z]+$")

def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(s)

def is_punct(tok: str) -> bool:
    return bool(re.fullmatch(r"[^\w\s]", tok))

# =====================================================================================
# Error logging (per-worker, JSONL)
# =====================================================================================
def _is_file_like_path(p: str) -> bool:
    p = (p or "").strip()
    if not p:
        return False
    base = os.path.basename(p)
    return ("." in base)

def resolve_err_log_path(err_log_dir: str, worker_id: int) -> str:
    err_log_dir = (err_log_dir or "").strip()
    if not err_log_dir:
        err_log_dir = "./log/gen_workload/ngram.log"

    if _is_file_like_path(err_log_dir):
        parent = os.path.dirname(err_log_dir) or "."
        base = os.path.basename(err_log_dir)
        return os.path.join(parent, f"{base}.worker_{worker_id}.jsonl")
    return os.path.join(err_log_dir, f"worker_{worker_id}.jsonl")

def log_db_error(path: str, payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except Exception:
        pass
    rec = dict(payload)
    rec["ts"] = time.time()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def rollback_quiet(conn) -> None:
    try:
        conn.rollback()
    except Exception:
        pass

# =====================================================================================
# Stable hash for de-dup
# =====================================================================================
def _hash64_str(s: str) -> int:
    b = s.encode("utf-8", errors="ignore")
    h1 = zlib.crc32(b) & 0xFFFFFFFF
    h2 = zlib.adler32(b) & 0xFFFFFFFF
    return (h1 << 32) | h2

def pair_key_u128(seed_id: int, regex: str) -> int:
    return (int(seed_id) << 64) | _hash64_str(regex)

def hash64_regex(regex: str) -> int:
    return _hash64_str(regex)

# =====================================================================================
# SQL E'...' helpers (for exporting runnable SQL)
# =====================================================================================
def sql_escape_E_literal(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "''")

def make_where_regex_sql(select_expr: str, table: str, col: str, regex: str, limit: Optional[int] = None) -> str:
    rx = sql_escape_E_literal(regex)
    q = f"SELECT {select_expr} FROM {table} WHERE {col} ~ E'{rx}'"
    if limit is not None and int(limit) > 0:
        q += f" LIMIT {int(limit)}"
    q += ";"
    return q

# =====================================================================================
# PG-safe regex building blocks
# =====================================================================================
WORD_CH = r"[A-Za-z0-9_]"  # keep consistent with PG \w-ish

def anychar_gap_pattern(N: int) -> str:
    N = max(0, int(N))
    return rf"(?:.|\n){{0,{N}}}?"

def gap_pattern_char(dist_tokens: int, base: int, per_token: int, cap: int) -> str:
    N = base + per_token * max(0, int(dist_tokens))
    N = min(int(cap), int(N))
    return anychar_gap_pattern(N)

def token_pattern_soft(tok: str) -> str:
    if re.fullmatch(r"[A-Za-z]+", tok):
        return r"\y[A-Za-z]+\y"
    return rf"\y{WORD_CH}+\y"

# =====================================================================================
# Pools
# =====================================================================================
@dataclass
class Pools:
    N: int
    df_low: int
    df_high: int
    df_low_pct: float
    df_high_pct: float
    token_stop_pct: float
    token_df_min_len: int
    stop_alpha_tokens: Set[str]
    mid_anch_2: Set[str]
    mid_anch_3: Set[str]
    df2_mid: Dict[str, int]
    df3_mid: Dict[str, int]

def _row_unique_alpha_ngrams(tokens: List[str], n: int) -> Set[str]:
    out: Set[str] = set()
    for tok in tokens:
        if not ALPHA_RE.fullmatch(tok):
            continue
        t = tok.lower()
        if len(t) < n:
            continue
        for i in range(len(t) - n + 1):
            out.add(t[i:i+n])
    return out

def _row_unique_alpha_tokens(tokens: List[str], min_len: int) -> Set[str]:
    out: Set[str] = set()
    for tok in tokens:
        if not ALPHA_RE.fullmatch(tok):
            continue
        if len(tok) < int(min_len):
            continue
        out.add(tok.lower())
    return out

def _dump_df_map_as_pairs(df_map: Dict[str, int], keys: Set[str]) -> List[List[Any]]:
    items = []
    for g in keys:
        items.append([g, int(df_map.get(g, 0))])
    items.sort(key=lambda x: x[0])
    return items

def build_pools_from_csv(
    csv_path: str,
    csv_col: str,
    out_json: str,
    df_low_pct: float = 0.005,
    df_high_pct: float = 0.15,
    *,
    token_stop_pct: float = 0.20,
    token_df_min_len: int = 4,
    max_rows: Optional[int] = None,
) -> None:
    df2: Dict[str, int] = defaultdict(int)
    df3: Dict[str, int] = defaultdict(int)
    df_tok: Dict[str, int] = defaultdict(int)
    N = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in tqdm(r, desc="Pools(DF) pass [CSV]"):
            s = row.get(csv_col, "")
            toks = tokenize(s)

            row2 = _row_unique_alpha_ngrams(toks, 2)
            row3 = _row_unique_alpha_ngrams(toks, 3)
            row_tok = _row_unique_alpha_tokens(toks, token_df_min_len)

            for g in row2:
                df2[g] += 1
            for g in row3:
                df3[g] += 1
            for t in row_tok:
                df_tok[t] += 1

            N += 1
            if max_rows and N >= max_rows:
                break

    low = max(1, int(math.floor(df_low_pct * N)))
    high = max(low, int(math.floor(df_high_pct * N)))

    mid2 = {g for g, c in df2.items() if (low <= c <= high)}
    mid3 = {g for g, c in df3.items() if (low <= c <= high)}

    tok_stop_df = max(1, int(math.floor(float(token_stop_pct) * N)))
    stop_tokens = {t for t, c in df_tok.items() if c >= tok_stop_df}

    obj = {
        "N": int(N),
        "df_low_pct": float(df_low_pct),
        "df_high_pct": float(df_high_pct),
        "df_low": int(low),
        "df_high": int(high),

        "token_stop_pct": float(token_stop_pct),
        "token_df_min_len": int(token_df_min_len),
        "token_stop_df": int(tok_stop_df),
        "stop_alpha_tokens_size": int(len(stop_tokens)),
        "stop_alpha_tokens": sorted(stop_tokens),

        "mid_anch_2_size": int(len(mid2)),
        "mid_anch_3_size": int(len(mid3)),
        "mid_anch_2": sorted(mid2),
        "mid_anch_3": sorted(mid3),

        "df2_mid_pairs": _dump_df_map_as_pairs(df2, mid2),
        "df3_mid_pairs": _dump_df_map_as_pairs(df3, mid3),
    }
    with open(out_json, "w", encoding="utf-8") as w:
        json.dump(obj, w, ensure_ascii=False)

    print(f"[OK] pools saved: {out_json}")
    print(f"  N={N}  df_low={low}  df_high={high}  mid2={len(mid2)}  mid3={len(mid3)}")
    print(f"  stop_tokens(alpha,len>={token_df_min_len}) DF>={tok_stop_df} -> {len(stop_tokens)} tokens")

def build_pools_from_pg(
    pg_dsn: str,
    table: str,
    col: str,
    id_col: str,
    out_json: str,
    df_low_pct: float = 0.005,
    df_high_pct: float = 0.15,
    *,
    token_stop_pct: float = 0.20,
    token_df_min_len: int = 4,
    sample_rows: int = 500000,
    seed: int = 42,
) -> None:
    rnd = random.Random(int(seed))
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute(f"SELECT min({id_col}), max({id_col}) FROM {table};")
        mn, mx = cur.fetchone()
    if mn is None or mx is None:
        conn.close()
        raise RuntimeError("Could not get id bounds for pools.")

    mn = int(mn); mx = int(mx)
    span = max(1, mx - mn + 1)

    if span == 1:
        stride = 0
        cur_off = 0
    else:
        while True:
            stride = rnd.randrange(1, span)
            if math.gcd(stride, span) == 1:
                break
        cur_off = rnd.randrange(span)

    df2: Dict[str, int] = defaultdict(int)
    df3: Dict[str, int] = defaultdict(int)
    df_tok: Dict[str, int] = defaultdict(int)
    N = 0

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        for _ in tqdm(range(int(sample_rows)), desc="Pools(DF) pass [PG sample]"):
            rid = mn + cur_off
            if span > 1:
                cur_off = (cur_off + stride) % span

            cur.execute(f"SELECT {col} AS text_col FROM {table} WHERE {id_col} = %s;", (rid,))
            row = cur.fetchone()
            if not row:
                continue
            s = row.get("text_col", "") or ""
            toks = tokenize(s)

            row2 = _row_unique_alpha_ngrams(toks, 2)
            row3 = _row_unique_alpha_ngrams(toks, 3)
            row_tok = _row_unique_alpha_tokens(toks, token_df_min_len)

            for g in row2:
                df2[g] += 1
            for g in row3:
                df3[g] += 1
            for t in row_tok:
                df_tok[t] += 1

            N += 1

    conn.close()

    low = max(1, int(math.floor(df_low_pct * N)))
    high = max(low, int(math.floor(df_high_pct * N)))

    mid2 = {g for g, c in df2.items() if (low <= c <= high)}
    mid3 = {g for g, c in df3.items() if (low <= c <= high)}

    tok_stop_df = max(1, int(math.floor(float(token_stop_pct) * N)))
    stop_tokens = {t for t, c in df_tok.items() if c >= tok_stop_df}

    obj = {
        "N": int(N),
        "df_low_pct": float(df_low_pct),
        "df_high_pct": float(df_high_pct),
        "df_low": int(low),
        "df_high": int(high),

        "token_stop_pct": float(token_stop_pct),
        "token_df_min_len": int(token_df_min_len),
        "token_stop_df": int(tok_stop_df),
        "stop_alpha_tokens_size": int(len(stop_tokens)),
        "stop_alpha_tokens": sorted(stop_tokens),

        "mid_anch_2_size": int(len(mid2)),
        "mid_anch_3_size": int(len(mid3)),
        "mid_anch_2": sorted(mid2),
        "mid_anch_3": sorted(mid3),

        "df2_mid_pairs": _dump_df_map_as_pairs(df2, mid2),
        "df3_mid_pairs": _dump_df_map_as_pairs(df3, mid3),

        "pg_sample_rows": int(sample_rows),
        "pg_table": table,
        "pg_col": col,
        "pg_id_col": id_col,
    }
    with open(out_json, "w", encoding="utf-8") as w:
        json.dump(obj, w, ensure_ascii=False)

    print(f"[OK] pools saved: {out_json}")
    print(f"  N={N}  df_low={low}  df_high={high}  mid2={len(mid2)}  mid3={len(mid3)}")
    print(f"  stop_tokens(alpha,len>={token_df_min_len}) DF>={tok_stop_df} -> {len(stop_tokens)} tokens")

def load_pools(pools_json: str) -> Pools:
    with open(pools_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    df2_mid: Dict[str, int] = {}
    df3_mid: Dict[str, int] = {}
    for p in obj.get("df2_mid_pairs", []):
        if isinstance(p, list) and len(p) == 2 and isinstance(p[0], str):
            df2_mid[p[0]] = int(p[1])
    for p in obj.get("df3_mid_pairs", []):
        if isinstance(p, list) and len(p) == 2 and isinstance(p[0], str):
            df3_mid[p[0]] = int(p[1])

    return Pools(
        N=int(obj.get("N", 0)),
        df_low=int(obj.get("df_low", 1)),
        df_high=int(obj.get("df_high", 1)),
        df_low_pct=float(obj.get("df_low_pct", 0.0)),
        df_high_pct=float(obj.get("df_high_pct", 1.0)),
        token_stop_pct=float(obj.get("token_stop_pct", 0.20)),
        token_df_min_len=int(obj.get("token_df_min_len", 4)),
        stop_alpha_tokens=set(obj.get("stop_alpha_tokens", [])),
        mid_anch_2=set(obj.get("mid_anch_2", [])),
        mid_anch_3=set(obj.get("mid_anch_3", [])),
        df2_mid=df2_mid,
        df3_mid=df3_mid,
    )

# =====================================================================================
# Anchor selection: LENGTH-RATIO VERSION
#   expose_ratio means: (n * k) / len(seed_string) ≈ expose_ratio
# =====================================================================================
@dataclass(frozen=True)
class AnchorPick:
    pos: int
    tok: str
    start: int
    gram: str

def _weighted_choice_idx(weights: List[float], rnd: random.Random) -> int:
    s = 0.0
    for w in weights:
        s += float(w)
    if s <= 0:
        return rnd.randrange(len(weights))
    r = rnd.random() * s
    acc = 0.0
    for i, w in enumerate(weights):
        acc += float(w)
        if acc >= r:
            return i
    return len(weights) - 1

def _token_ngram_candidates_scored(
    tok: str,
    n: int,
    mid_set: Set[str],
    df_mid: Dict[str, int],
) -> List[Tuple[int, str, float]]:
    if not ALPHA_RE.fullmatch(tok):
        return []
    t_low = tok.lower()
    L = len(t_low)
    if L < n:
        return []
    out: List[Tuple[int, str, float]] = []
    for i in range(L - n + 1):
        g_low = t_low[i:i+n]
        if g_low not in mid_set:
            continue
        df = int(df_mid.get(g_low, 0))
        score = (1.0 / (df + 1.0)) * (1.0 + 0.05 * min(L, 30))
        out.append((i, tok[i:i+n], float(score)))
    if not out:
        return []
    best: Dict[str, Tuple[int, str, float]] = {}
    for st, gr, sc in out:
        cur = best.get(gr)
        if cur is None or sc > cur[2]:
            best[gr] = (st, gr, sc)
    return list(best.values())

def _clip_int(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def _k_from_expose_ratio(seed_len_chars: int, n: int, expose_ratio: float, k_min: int, k_max: int) -> Tuple[int, int]:
    L = max(1, int(seed_len_chars))
    r = float(expose_ratio)
    if r < 0.0:
        r = 0.0
    if r > 1.0:
        r = 1.0
    budget_chars = int(math.floor(r * L))  # do not exceed target budget
    k = budget_chars // max(1, int(n))
    if budget_chars > 0 and k == 0:
        k = 1
    if k == 0:
        k = 1  # must have at least 1 anchor to generate a regex
    k = _clip_int(k, int(k_min), int(k_max))
    return k, budget_chars

def pick_ngram_anchors_by_expose_ratio(
    tokens: List[str],
    pools: Pools,
    *,
    n: int,
    seed_len_chars: int,
    expose_ratio: float,
    k_min: int,
    k_max: int,
    rnd: random.Random,
) -> Tuple[Optional[List[AnchorPick]], Dict[str, Any]]:
    if n == 2:
        mid_set = pools.mid_anch_2
        df_mid = pools.df2_mid
    elif n == 3:
        mid_set = pools.mid_anch_3
        df_mid = pools.df3_mid
    else:
        return None, {"reason": "bad_n"}

    # compute k target from character budget
    k_target, budget_chars = _k_from_expose_ratio(seed_len_chars, n, expose_ratio, k_min, k_max)

    # collect candidates by token position (we keep: max 1 anchor per token position)
    cand_by_pos: Dict[int, List[Tuple[int, str, float]]] = {}
    pos_score: Dict[int, float] = {}
    for i, tok in enumerate(tokens):
        if is_punct(tok):
            continue
        if not ALPHA_RE.fullmatch(tok):
            continue
        t_low = tok.lower()
        if t_low in pools.stop_alpha_tokens:
            continue
        cands = _token_ngram_candidates_scored(tok, n, mid_set, df_mid)
        if cands:
            cand_by_pos[i] = cands
            pos_score[i] = max(sc for _, _, sc in cands)

    positions = sorted(cand_by_pos.keys())
    M = len(positions)
    if M == 0:
        return None, {"reason": "no_eligible_positions", "k_target": k_target, "budget_chars": budget_chars}

    if M < k_target:
        # strict: do not silently under-shoot the leakage budget
        return None, {
            "reason": "insufficient_positions_for_budget",
            "eligible_positions": M,
            "k_target": k_target,
            "budget_chars": budget_chars,
            "seed_len_chars": int(seed_len_chars),
            "expose_ratio": float(expose_ratio),
        }

    T = max(1, len(tokens))
    early = [p for p in positions if p <= int(0.2 * T)]
    mid = [p for p in positions if int(0.2 * T) < p < int(0.8 * T)]
    late = [p for p in positions if p >= int(0.8 * T)]

    def choose_pos(bucket: List[int], used_pos: Set[int]) -> Optional[int]:
        cand = [p for p in bucket if p not in used_pos]
        if not cand:
            return None
        w = [max(1e-12, float(pos_score.get(p, 1e-12))) for p in cand]
        idx = _weighted_choice_idx(w, rnd)
        return cand[idx]

    chosen_pos: List[int] = []
    used_pos: Set[int] = set()

    # segment coverage preference
    for bucket in (mid, late, early):
        if len(chosen_pos) >= k_target:
            break
        p = choose_pos(bucket, used_pos)
        if p is not None:
            chosen_pos.append(p)
            used_pos.add(p)

    while len(chosen_pos) < k_target:
        p = choose_pos(positions, used_pos)
        if p is None:
            break
        chosen_pos.append(p)
        used_pos.add(p)

    if len(chosen_pos) < k_target:
        return None, {
            "reason": "not_enough_positions_after_sampling",
            "eligible_positions": M,
            "k_target": k_target,
            "picked": len(chosen_pos),
        }

    chosen_pos.sort()

    # pick one gram per chosen position with gram diversity
    used_grams_low: Set[str] = set()
    out: List[AnchorPick] = []
    for p in chosen_pos:
        tok = tokens[p]
        cands = cand_by_pos.get(p, [])
        if not cands:
            continue

        cands_sorted = sorted(cands, key=lambda x: x[2], reverse=True)
        filt = []
        for st, gr, sc in cands_sorted:
            if gr.lower() in used_grams_low:
                continue
            filt.append((st, gr, sc))
        use = filt if filt else cands_sorted

        Mtop = min(len(use), 8)
        use = use[:Mtop]
        weights = [max(1e-12, u[2]) for u in use]
        idx = _weighted_choice_idx(weights, rnd)
        st, gr, _sc = use[idx]

        out.append(AnchorPick(pos=int(p), tok=str(tok), start=int(st), gram=str(gr)))
        used_grams_low.add(gr.lower())

    if len(out) < k_target:
        return None, {
            "reason": "gram_diversity_exhausted",
            "k_target": k_target,
            "picked": len(out),
            "eligible_positions": M,
        }

    leak_chars = int(len(out) * int(n))
    seed_len = max(1, int(seed_len_chars))
    leak_ratio = float(leak_chars) / float(seed_len)

    stats = {
        "eligible_positions": M,
        "k_target": int(k_target),
        "k_used": int(len(out)),
        "budget_chars": int(budget_chars),
        "leak_chars": int(leak_chars),
        "seed_len_chars": int(seed_len),
        "expose_ratio_target": float(expose_ratio),
        "expose_ratio_effective": float(leak_ratio),
        "chosen_positions": [a.pos for a in out],
    }
    return out, stats

# =====================================================================================
# Anchor token rendering
# =====================================================================================
def _q_range(n: int, slack: int) -> Tuple[int, int]:
    if slack <= 0:
        return (n, n)
    lo = max(0, n - slack)
    hi = n + slack
    return (lo, hi)

def token_pattern_from_ngram_anchor(
    tok: str,
    gram: str,
    start: int,
    n: int,
    *,
    slack: int = 1,
    cap_side: int = 12,
) -> str:
    if not tok or not gram or n <= 0:
        return token_pattern_soft(tok)
    if not ALPHA_RE.fullmatch(tok):
        return token_pattern_soft(tok)

    L = len(tok)
    start = max(0, min(int(start), L))
    if start + n > L:
        return token_pattern_soft(tok)

    pre_len = start
    suf_len = L - (start + n)

    lo_pre, hi_pre = _q_range(pre_len, slack)
    lo_suf, hi_suf = _q_range(suf_len, slack)

    hi_pre = min(hi_pre, cap_side)
    hi_suf = min(hi_suf, cap_side)
    lo_pre = min(lo_pre, hi_pre)
    lo_suf = min(lo_suf, hi_suf)

    a = re.escape(gram)
    return rf"\y{WORD_CH}{{{lo_pre},{hi_pre}}}{a}{WORD_CH}{{{lo_suf},{hi_suf}}}\y"

# =====================================================================================
# Assemble regex
# =====================================================================================
def assemble_regex_from_anchors(
    tokens: List[str],
    anchors: List[AnchorPick],
    *,
    n: int,
    gap_base: int = 50,
    gap_per_token: int = 10,
    gap_cap: int = 120,
    anchor_slack: int = 1,
    anchor_cap_side: int = 12,
    prefix_any_g: int = 0,
    suffix_any_g: int = 0,
    case_insensitive: bool = False,
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not anchors:
        return None, {"reason": "no_anchors"}

    anchors_sorted = sorted(anchors, key=lambda a: a.pos)
    kept_pos = [a.pos for a in anchors_sorted]

    parts: List[str] = []
    if case_insensitive:
        parts.append("(?i)")
    if prefix_any_g > 0:
        parts.append(anychar_gap_pattern(prefix_any_g))

    a0 = anchors_sorted[0]
    parts.append(token_pattern_from_ngram_anchor(
        a0.tok, a0.gram, a0.start, n,
        slack=anchor_slack, cap_side=anchor_cap_side
    ))

    prev_pos = a0.pos
    for a in anchors_sorted[1:]:
        dist = a.pos - prev_pos - 1
        parts.append(gap_pattern_char(dist, gap_base, gap_per_token, gap_cap))
        parts.append(token_pattern_from_ngram_anchor(
            a.tok, a.gram, a.start, n,
            slack=anchor_slack, cap_side=anchor_cap_side
        ))
        prev_pos = a.pos

    if suffix_any_g > 0:
        parts.append(anychar_gap_pattern(suffix_any_g))

    regex = "".join(parts)
    stats = {
        "kept_positions": kept_pos,
        "n_kept": len(kept_pos),
        "anchors": [{"pos": a.pos, "gram": a.gram, "start": a.start, "tok": a.tok} for a in anchors_sorted],
    }
    return regex, stats

# =====================================================================================
# Seed verification
# =====================================================================================
def approx_verify_seed_py(seed: str, pg_regex: str) -> bool:
    try:
        rx = pg_regex.replace(r"\y", r"\b").replace(r"\m", r"\b").replace(r"\M", r"\b")
        return re.search(rx, seed, flags=re.DOTALL) is not None
    except Exception:
        return False

# =====================================================================================
# DB helpers
# =====================================================================================
def _get_id_bounds(conn, table: str, id_col: str) -> Optional[Tuple[int, int]]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT min({id_col}), max({id_col}) FROM {table};")
        mn, mx = cur.fetchone()
    if mn is None or mx is None:
        return None
    return int(mn), int(mx)

def _split_id_ranges(mn: int, mx: int, workers: int) -> List[Tuple[int, int]]:
    W = max(1, int(workers))
    if mn > mx:
        return [(mn, mx)] * W
    span = mx - mn + 1
    step = max(1, span // W)
    ranges: List[Tuple[int, int]] = []
    cur = mn
    for i in range(W):
        lo = cur
        hi = mx if i == W - 1 else min(mx, lo + step - 1)
        ranges.append((lo, hi))
        cur = hi + 1
        if cur > mx and i < W - 1:
            ranges.extend([(mx + 1, mx)] * (W - i - 1))
            break
    return ranges

class IdSampler:
    def next_id(self) -> int:
        raise NotImplementedError

class RandomIdSampler(IdSampler):
    def __init__(self, lo: int, hi: int, rnd: random.Random):
        self.lo = lo; self.hi = hi; self.rnd = rnd
    def next_id(self) -> int:
        return self.rnd.randint(self.lo, self.hi)

class StrideIdSampler(IdSampler):
    def __init__(self, lo: int, hi: int, rnd: random.Random):
        self.lo = lo; self.hi = hi
        self.span = max(1, (hi - lo + 1))
        self.cur = rnd.randrange(self.span)
        if self.span == 1:
            self.stride = 0
        else:
            while True:
                stride = rnd.randrange(1, self.span)
                if math.gcd(stride, self.span) == 1:
                    self.stride = stride
                    break
    def next_id(self) -> int:
        rid = self.lo + self.cur
        if self.span > 1:
            self.cur = (self.cur + self.stride) % self.span
        return rid

def make_id_sampler(kind: str, lo: int, hi: int, base_seed: int, worker_id: int) -> IdSampler:
    rr = random.Random((int(base_seed) * 1000003) ^ (worker_id * 9176) ^ 0x9E3779B97F4A7C15)
    if hi < lo:
        return RandomIdSampler(lo, lo, rr)
    if kind == "random":
        return RandomIdSampler(lo, hi, rr)
    return StrideIdSampler(lo, hi, rr)

def fetch_random_seed(
    conn,
    table: str,
    col: str,
    id_col: str,
    min_tokens: int,
    id_lo: int,
    id_hi: int,
    sampler: Optional[IdSampler],
) -> Tuple[int, str, List[str]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        while True:
            row = None
            if id_lo <= id_hi:
                rid = sampler.next_id() if sampler is not None else random.randint(id_lo, id_hi)

                cur.execute(f"SELECT {id_col} AS sid, {col} AS text_col FROM {table} WHERE {id_col} = %s;", (rid,))
                row = cur.fetchone()

                if not row:
                    cur.execute(
                        f"SELECT {id_col} AS sid, {col} AS text_col FROM {table} "
                        f"WHERE {id_col} > %s AND {id_col} <= %s ORDER BY {id_col} LIMIT 1;",
                        (rid, id_hi),
                    )
                    row = cur.fetchone()

                    if not row:
                        cur.execute(
                            f"SELECT {id_col} AS sid, {col} AS text_col FROM {table} "
                            f"WHERE {id_col} >= %s AND {id_col} <= %s ORDER BY {id_col} LIMIT 1;",
                            (id_lo, id_hi),
                        )
                        row = cur.fetchone()

            if not row:
                cur.execute(f"SELECT {id_col} AS sid, {col} AS text_col FROM {table} ORDER BY {id_col} LIMIT 1;")
                row = cur.fetchone()
                if not row:
                    raise RuntimeError("No rows available in table.")

            s = (row.get("text_col", "") or "")
            toks = tokenize(s)
            if len(toks) >= min_tokens:
                return int(row["sid"]), s, toks

# =====================================================================================
# Worker
# =====================================================================================
def worker_generate(
    worker_id: int,
    num_target: int,
    pools_json: str,
    pg_dsn: str,
    dataset: str,
    table: str,
    col: str,
    id_col: str,
    id_lo: int,
    id_hi: int,
    seed_min_tokens: int,
    max_attempts: int,
    out_tmp_dir: str,
    rng_seed: int,
    dedup_in_worker: bool,
    max_per_seed: int,
    err_log_dir: str,
    seed_sampler: str,
    # knobs
    n_gram: int,
    expose_ratio: float,
    k_min: int,
    k_max: int,
    # fixed policies
    gap_base: int,
    gap_per_token: int,
    gap_cap: int,
    anchor_slack: int,
    anchor_cap_side: int,
    prefix_any_g: int,
    suffix_any_g: int,
    case_insensitive: bool,
    # hygiene
    dedup_regex_only_in_worker: bool,
    max_per_regex: int,
    # seed verify
    seed_verify: str,
) -> Dict[str, Any]:
    random.seed((int(rng_seed) * 1000003) ^ (worker_id * 9176) ^ (os.getpid() << 8))
    rnd = random.Random((int(rng_seed) * 1000003) ^ (worker_id * 9176) ^ 0xC0FFEE)

    pools = load_pools(pools_json)
    err_log_path = resolve_err_log_path(err_log_dir, worker_id)

    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = True
    sampler = make_id_sampler(seed_sampler, id_lo, id_hi, rng_seed, worker_id)

    reasons = defaultdict(int)
    ok = 0
    os.makedirs(out_tmp_dir, exist_ok=True)
    tmp_path = os.path.join(out_tmp_dir, f"tmp_worker_{worker_id}.jsonl")

    seen_pairs: Set[int] = set()
    seen_regex: Set[int] = set()
    per_seed_cnt: Dict[int, int] = {}
    per_regex_cnt: Dict[int, int] = {}

    with open(tmp_path, "w", encoding="utf-8") as w:
        attempts = 0
        while ok < num_target and attempts < max_attempts:
            attempts += 1

            try:
                seed_id, seed, toks = fetch_random_seed(
                    conn=conn, table=table, col=col, id_col=id_col,
                    min_tokens=seed_min_tokens, id_lo=id_lo, id_hi=id_hi, sampler=sampler
                )
            except psycopg2.Error as e:
                diag = getattr(e, "diag", None)
                log_db_error(err_log_path, {
                    "stage": "fetch_random_seed",
                    "worker_id": worker_id,
                    "pgcode": getattr(e, "pgcode", None),
                    "pgerror": str(e),
                    "diag_message_primary": getattr(diag, "message_primary", None) if diag else None,
                    "diag_message_detail": getattr(diag, "message_detail", None) if diag else None,
                    "diag_context": getattr(diag, "context", None) if diag else None,
                    "range": [int(id_lo), int(id_hi)],
                })
                rollback_quiet(conn)
                reasons["db_error_fetch_seed"] += 1
                continue
            except Exception as e:
                log_db_error(err_log_path, {
                    "stage": "fetch_random_seed_non_pg",
                    "worker_id": worker_id,
                    "error": repr(e),
                    "trace": traceback.format_exc(limit=8),
                    "range": [int(id_lo), int(id_hi)],
                })
                rollback_quiet(conn)
                reasons["db_error_fetch_seed_non_pg"] += 1
                continue

            if max_per_seed > 0:
                c = per_seed_cnt.get(seed_id, 0)
                if c >= max_per_seed:
                    reasons["seed_over_cap"] += 1
                    continue

            anchors, pick_stats = pick_ngram_anchors_by_expose_ratio(
                toks, pools,
                n=n_gram,
                seed_len_chars=len(seed),
                expose_ratio=expose_ratio,
                k_min=k_min,
                k_max=k_max,
                rnd=rnd,
            )
            if anchors is None:
                reasons[pick_stats.get("reason", "anchor_pick_fail")] += 1
                continue

            regex, gen_stats = assemble_regex_from_anchors(
                tokens=toks,
                anchors=anchors,
                n=n_gram,
                gap_base=gap_base,
                gap_per_token=gap_per_token,
                gap_cap=gap_cap,
                anchor_slack=anchor_slack,
                anchor_cap_side=anchor_cap_side,
                prefix_any_g=prefix_any_g,
                suffix_any_g=suffix_any_g,
                case_insensitive=case_insensitive,
            )
            if not regex:
                reasons[gen_stats.get("reason", "gen_fail")] += 1
                continue

            rh = hash64_regex(regex)
            if dedup_regex_only_in_worker:
                if rh in seen_regex:
                    reasons["dup_regex_worker"] += 1
                    continue
                seen_regex.add(rh)

            if max_per_regex > 0:
                c = per_regex_cnt.get(rh, 0)
                if c >= int(max_per_regex):
                    reasons["regex_over_cap"] += 1
                    continue

            if seed_verify == "py":
                if not approx_verify_seed_py(seed, regex):
                    reasons["seed_verify_fail_py"] += 1
                    continue
            elif seed_verify == "off":
                pass
            elif seed_verify == "pg":
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT %s::text ~ %s::text;", (seed, regex))
                        ok_seed = bool(cur.fetchone()[0])
                    if not ok_seed:
                        reasons["seed_verify_fail_pg"] += 1
                        continue
                except Exception:
                    reasons["seed_verify_fail_pg_err"] += 1
                    continue
            else:
                reasons["seed_verify_bad_mode"] += 1
                continue

            if dedup_in_worker:
                kkey = pair_key_u128(seed_id, regex)
                if kkey in seen_pairs:
                    reasons["dup_pair_worker"] += 1
                    continue
                seen_pairs.add(kkey)

            if max_per_seed > 0:
                per_seed_cnt[seed_id] = per_seed_cnt.get(seed_id, 0) + 1
            if max_per_regex > 0:
                per_regex_cnt[rh] = per_regex_cnt.get(rh, 0) + 1

            rec = {
                "dataset": dataset,
                "seed_id": int(seed_id),
                "seed_string": seed,
                "regex": regex,

                "anchor_n": int(n_gram),
                "expose_ratio": float(expose_ratio),

                "anchors": gen_stats.get("anchors", []),
                "kept_positions": gen_stats.get("kept_positions", []),

                "pick_stats": pick_stats,
                "stats": {
                    "seed_len_tokens": int(len(toks)),
                    "seed_len_chars": int(len(seed)),
                    "n_kept": int(gen_stats.get("n_kept", 0)),
                },
                "sql_E": make_where_regex_sql(
                    select_expr=id_col,
                    table=table,
                    col=col,
                    regex=regex,
                    limit=None,
                ),
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1

    try:
        conn.close()
    except Exception:
        pass

    return {
        "tmp_path": tmp_path,
        "n_ok": ok,
        "attempts": attempts,
        "reasons": dict(reasons),
        "range": (int(id_lo), int(id_hi)),
        "err_log_path": err_log_path,
    }

# =====================================================================================
# Main
# =====================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prep_csv", "prep_pg", "gen"], required=True)

    # Pools
    ap.add_argument("--pools", default="pools_ngram.json")
    ap.add_argument("--df_low_pct", type=float, default=0.005)
    ap.add_argument("--df_high_pct", type=float, default=0.15)

    # (Fixed policy) stop-token suppression
    ap.add_argument("--token_stop_pct", type=float, default=0.20)
    ap.add_argument("--token_df_min_len", type=int, default=4)

    # prep_csv
    ap.add_argument("--csv", default="")
    ap.add_argument("--csv_col", default="Description")
    ap.add_argument("--max_rows_df", type=int, default=0)

    # prep_pg / gen: DB
    ap.add_argument("--pg_dsn", default="dbname=regex_db user=lyz host=localhost port=5000")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--table", default="table_name")
    ap.add_argument("--col", default="text_col")
    ap.add_argument("--id_col", default="id")

    ap.add_argument("--pg_pools_sample_rows", type=int, default=500000)
    ap.add_argument("--pg_pools_seed", type=int, default=42)

    # gen output
    ap.add_argument("--out", default="workload/out.jsonl")
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tmp_dir", default="")

    # Error logs
    ap.add_argument("--err_log_dir", default="./log/gen_workload/ngram.log")

    # sampling / anti-dup
    ap.add_argument("--split_id_ranges", dest="split_id_ranges", action="store_true", default=True)
    ap.add_argument("--no_split_id_ranges", dest="split_id_ranges", action="store_false")
    ap.add_argument("--dedup_in_worker", dest="dedup_in_worker", action="store_true", default=True)
    ap.add_argument("--no_dedup_in_worker", dest="dedup_in_worker", action="store_false")
    ap.add_argument("--max_per_seed", type=int, default=4)
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--seed_sampler", choices=["stride", "random"], default="stride")
    ap.add_argument("--max_attempts_per_worker", type=int, default=2_000_000)
    ap.add_argument("--seed_min_tokens", type=int, default=12)

    # =======================
    # CORE KNOBS (RATIO VERSION)
    # =======================
    ap.add_argument("--n_gram", type=int, choices=[2, 3], default=3)
    ap.add_argument("--expose_ratio", type=float, default=0.20,
                    help="Target leakage ratio: (n_gram * k) / len(seed_string). k is derived from expose_ratio.")

    # safety caps (fixed policy)
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=12)

    # fixed policies
    ap.add_argument("--gap_base", type=int, default=50)
    ap.add_argument("--gap_per_token", type=int, default=10)
    ap.add_argument("--gap_cap", type=int, default=120)
    ap.add_argument("--anchor_slack", type=int, default=1)
    ap.add_argument("--anchor_cap_side", type=int, default=12)
    ap.add_argument("--prefix_any_g", type=int, default=0)
    ap.add_argument("--suffix_any_g", type=int, default=0)
    ap.add_argument("--case_insensitive", action="store_true", default=False)

    # hygiene
    ap.add_argument("--dedup_regex_only_in_worker", action="store_true", default=True)
    ap.add_argument("--no_dedup_regex_only_in_worker", dest="dedup_regex_only_in_worker", action="store_false")
    ap.add_argument("--max_per_regex", type=int, default=0)

    ap.add_argument("--seed_verify", choices=["py", "off", "pg"], default="pg",
                    help="Use pg to ensure exact Postgres semantics (recommended).")

    args = ap.parse_args()

    if args.mode == "prep_csv":
        if not args.csv:
            raise SystemExit("[Error] --csv is required for mode=prep_csv")
        build_pools_from_csv(
            csv_path=args.csv,
            csv_col=args.csv_col,
            out_json=args.pools,
            df_low_pct=args.df_low_pct,
            df_high_pct=args.df_high_pct,
            token_stop_pct=args.token_stop_pct,
            token_df_min_len=args.token_df_min_len,
            max_rows=(args.max_rows_df or None),
        )
        return

    if args.mode == "prep_pg":
        build_pools_from_pg(
            pg_dsn=args.pg_dsn,
            table=args.table,
            col=args.col,
            id_col=args.id_col,
            out_json=args.pools,
            df_low_pct=args.df_low_pct,
            df_high_pct=args.df_high_pct,
            token_stop_pct=args.token_stop_pct,
            token_df_min_len=args.token_df_min_len,
            sample_rows=args.pg_pools_sample_rows,
            seed=args.pg_pools_seed,
        )
        return

    # gen
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_dir = args.tmp_dir or os.path.join(out_dir if out_dir else ".", "tmp_workers")
    os.makedirs(tmp_dir, exist_ok=True)

    conn0 = psycopg2.connect(args.pg_dsn)
    conn0.autocommit = True
    bounds = _get_id_bounds(conn0, args.table, args.id_col)
    conn0.close()
    if bounds is None:
        raise SystemExit("[Error] Could not get id bounds from table.")
    mn, mx = bounds

    W = max(1, int(args.workers))
    per = args.num // W
    rem = args.num % W
    targets = [per + (1 if i < rem else 0) for i in range(W)]
    ranges = _split_id_ranges(mn, mx, W) if args.split_id_ranges else [(mn, mx)] * W

    futures = []
    all_reasons = defaultdict(int)
    tmp_paths: List[str] = []
    err_paths: List[str] = []

    with ProcessPoolExecutor(max_workers=W) as ex:
        for wid, n_target in enumerate(targets):
            if n_target <= 0:
                continue
            lo, hi = ranges[wid]
            futures.append(ex.submit(
                worker_generate,
                wid,
                n_target,
                args.pools,
                args.pg_dsn,
                args.dataset,
                args.table,
                args.col,
                args.id_col,
                lo,
                hi,
                args.seed_min_tokens,
                args.max_attempts_per_worker,
                tmp_dir,
                args.rng_seed,
                args.dedup_in_worker,
                args.max_per_seed,
                args.err_log_dir,
                args.seed_sampler,
                # knobs
                args.n_gram,
                args.expose_ratio,
                args.k_min,
                args.k_max,
                # fixed policies
                args.gap_base,
                args.gap_per_token,
                args.gap_cap,
                args.anchor_slack,
                args.anchor_cap_side,
                args.prefix_any_g,
                args.suffix_any_g,
                args.case_insensitive,
                # hygiene
                args.dedup_regex_only_in_worker,
                args.max_per_regex,
                # verify
                args.seed_verify,
            ))

        for fu in tqdm(as_completed(futures), total=len(futures), desc="Workers"):
            res = fu.result()
            tmp_paths.append(res["tmp_path"])
            err_paths.append(res.get("err_log_path", ""))
            for k, v in res["reasons"].items():
                all_reasons[k] += int(v)

    # Merge
    n_written = 0
    if not args.dedup:
        with open(args.out, "w", encoding="utf-8") as out_f:
            for p in tmp_paths:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out_f.write(line)
                            n_written += 1
    else:
        seen: Set[int] = set()
        with open(args.out, "w", encoding="utf-8") as out_f:
            for p in tmp_paths:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            sid = int(obj.get("seed_id"))
                            rx = obj.get("regex", "")
                            if not isinstance(rx, str):
                                continue
                            kkey = pair_key_u128(sid, rx)
                        except Exception:
                            continue
                        if kkey in seen:
                            continue
                        seen.add(kkey)
                        out_f.write(line)
                        n_written += 1

    print(f"[OK] wrote {n_written} lines -> {args.out}")
    print("[REASONS]", dict(all_reasons))
    print(f"[TMP] worker files in: {tmp_dir}")

    uniq_err = sorted(set(p for p in err_paths if p))
    if uniq_err:
        print("[ERRLOG] per-worker DB error logs:")
        for p in uniq_err:
            print("  ", p)

    print("[INFO] Core knobs (length-ratio version):")
    print("  - --n_gram (2/3)")
    print("  - --expose_ratio (target: (n_gram*k)/len(seed_string))")
    print("  - k is derived per sample; safety caps: --k_min/--k_max (fixed policy)")

if __name__ == "__main__":
    main()
