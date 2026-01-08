#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
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
WORDCHAR_RE = re.compile(r"[A-Za-z0-9_]")
DIGIT_RE = re.compile(r"\d")


def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(s or "")


def is_punct(tok: str) -> bool:
    return bool(re.fullmatch(r"[^\w\s]", tok or ""))


def wordchar_len(s: str) -> int:
    return len(WORDCHAR_RE.findall(s or ""))


# =====================================================================================
# Error logging (per-worker, JSONL)
# =====================================================================================
def _is_file_like_path(p: str) -> bool:
    p = (p or "").strip()
    if not p:
        return False
    base = os.path.basename(p)
    return "." in base


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
    b = (s or "").encode("utf-8", errors="ignore")
    h1 = zlib.crc32(b) & 0xFFFFFFFF
    h2 = zlib.adler32(b) & 0xFFFFFFFF
    return (h1 << 32) | h2


def pair_key_u128(seed_id: int, regex: str) -> int:
    return (int(seed_id) << 64) | _hash64_str(regex)


def hash64_regex(regex: str) -> int:
    return _hash64_str(regex)


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
    # fallback: generic word token
    if re.fullmatch(r"[A-Za-z]+", tok or ""):
        return r"\y[A-Za-z]+\y"
    return rf"\y{WORD_CH}+\y"


# =====================================================================================
# Pools: be tolerant to older/newer pool json formats
# =====================================================================================
@dataclass
class Pools:
    N: int
    df_low: int
    df_high: int
    mid_anch_2: Set[str]
    mid_anch_3: Set[str]
    df2_mid: Dict[str, int]
    df3_mid: Dict[str, int]
    stop_alpha_tokens: Set[str]


def load_pools(pools_json: str) -> Pools:
    with open(pools_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    mid2 = set(obj.get("mid_anch_2", [])) if isinstance(obj.get("mid_anch_2", []), list) else set()
    mid3 = set(obj.get("mid_anch_3", [])) if isinstance(obj.get("mid_anch_3", []), list) else set()

    df2_mid: Dict[str, int] = {}
    df3_mid: Dict[str, int] = {}

    # v6 stores df2_mid_pairs / df3_mid_pairs (list of [gram, df])
    for p in obj.get("df2_mid_pairs", []):
        if isinstance(p, list) and len(p) == 2 and isinstance(p[0], str):
            df2_mid[p[0]] = int(p[1])
    for p in obj.get("df3_mid_pairs", []):
        if isinstance(p, list) and len(p) == 2 and isinstance(p[0], str):
            df3_mid[p[0]] = int(p[1])

    stop_tokens = set(obj.get("stop_alpha_tokens", [])) if isinstance(obj.get("stop_alpha_tokens", []), list) else set()

    return Pools(
        N=int(obj.get("N", 0) or 0),
        df_low=int(obj.get("df_low", 1) or 1),
        df_high=int(obj.get("df_high", 1) or 1),
        mid_anch_2=mid2,
        mid_anch_3=mid3,
        df2_mid=df2_mid,
        df3_mid=df3_mid,
        stop_alpha_tokens=stop_tokens,
    )


# =====================================================================================
# Candidate islands
# =====================================================================================
@dataclass(frozen=True)
class Island:
    kind: str              # "struct" or "ngram"
    pos: int               # token index
    tok: str               # token text
    expose_len: int        # contributes to exposure budget
    score: float           # for weighted choice
    pattern: str           # regex fragment for this island
    # optional debug fields
    n: int = 0             # n for ngram island
    gram: str = ""         # gram for ngram island
    start: int = 0         # start offset for ngram inside tok


def _weighted_choice(items: List[int], weights: List[float], rnd: random.Random) -> int:
    if not items:
        raise ValueError("empty items")
    s = 0.0
    for w in weights:
        s += float(w)
    if s <= 0:
        return rnd.choice(items)
    r = rnd.random() * s
    acc = 0.0
    for it, w in zip(items, weights):
        acc += float(w)
        if acc >= r:
            return it
    return items[-1]


def _is_structured_token(tok: str) -> bool:
    # structured = contains digit OR contains hyphen (common in road IDs)
    if not tok:
        return False
    if is_punct(tok):
        return False
    return (DIGIT_RE.search(tok) is not None) or ("-" in tok)


def _render_structured_token(tok: str) -> str:
    """
    Render structured token with digit runs generalized to [0-9]{k},
    keeping hyphens literal (escaped). This reduces fixed trigrams.
    Example: "CA-84" -> "CA\\-[0-9]{2}"
             "I-680" -> "I\\-[0-9]{3}"
             "680"   -> "[0-9]{3}"
    Wrap with \\y ... \\y.
    """
    if not tok:
        return token_pattern_soft(tok)

    out_parts: List[str] = []
    i = 0
    L = len(tok)
    while i < L:
        ch = tok[i]
        if ch.isdigit():
            j = i + 1
            while j < L and tok[j].isdigit():
                j += 1
            k = j - i
            out_parts.append(rf"[0-9]{{{k}}}")
            i = j
            continue
        out_parts.append(re.escape(ch))
        i += 1

    body = "".join(out_parts)
    return rf"\y{body}\y"


def _q_range(n: int, slack: int) -> Tuple[int, int]:
    if slack <= 0:
        return (n, n)
    lo = max(0, n - slack)
    hi = n + slack
    return (lo, hi)


def _anchor_alt(gram: str, n: int) -> str:
    """
    Light alternation to make trgm extraction less certain, while still matching the seed.
      n=3: (?:abc|ab[WORD_CH])
      n=2: (?:ab|a[WORD_CH])
    """
    gram = gram or ""
    if len(gram) < n or n <= 0:
        return re.escape(gram)
    if n == 3:
        a = re.escape(gram)
        b = re.escape(gram[:2])
        return rf"(?:{a}|{b}{WORD_CH})"
    if n == 2:
        a = re.escape(gram)
        b = re.escape(gram[:1])
        return rf"(?:{a}|{b}{WORD_CH})"
    return re.escape(gram)


def _token_pattern_from_ngram_anchor(
    tok: str,
    gram: str,
    start: int,
    n: int,
    *,
    slack: int = 1,
    cap_side: int = 10,
) -> str:
    """
    Build a token-level regex:
      \\y WORD_CH{lo_pre,hi_pre} <ALT(gram)> WORD_CH{lo_suf,hi_suf} \\y
    """
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

    alt = _anchor_alt(gram, n)
    return rf"\y{WORD_CH}{{{lo_pre},{hi_pre}}}{alt}{WORD_CH}{{{lo_suf},{hi_suf}}}\y"


def _token_ngram_candidates(tok: str, n: int, mid_set: Set[str], df_mid: Dict[str, int]) -> List[Tuple[int, str, float]]:
    if not tok or not ALPHA_RE.fullmatch(tok):
        return []
    t_low = tok.lower()
    if len(t_low) < n:
        return []
    out: List[Tuple[int, str, float]] = []
    for i in range(len(t_low) - n + 1):
        g_low = t_low[i:i+n]
        if g_low not in mid_set:
            continue
        df = int(df_mid.get(g_low, 0))
        score = (1.0 / (df + 1.0)) * (1.0 + 0.03 * min(len(tok), 30))
        out.append((i, tok[i:i+n], float(score)))
    best: Dict[str, Tuple[int, str, float]] = {}
    for st, gr, sc in out:
        cur = best.get(gr)
        if cur is None or sc > cur[2]:
            best[gr] = (st, gr, sc)
    return list(best.values())


def pick_islands_by_exposure_budget(
    seed: str,
    tokens: List[str],
    pools: Pools,
    *,
    n_gram: int,
    expose_ratio: float,
    k_max: int,
    rnd: random.Random,
    anchor_slack: int,
    anchor_cap_side: int,
) -> Optional[Tuple[List[Island], Dict[str, Any]]]:
    """
    Select islands to satisfy exposure budget:
      budget = ceil(expose_ratio * wordchar_len(seed))

    Priority:
      1) structured token islands (digits/hyphen), rendered with digit-run generalization
      2) alpha n-gram islands (n=2/3), rendered with light alternation
    """
    if not tokens:
        return None

    W = max(0, wordchar_len(seed))
    if W <= 0:
        return None

    expose_ratio = float(expose_ratio)
    if expose_ratio <= 0:
        return None

    k_max = max(1, int(k_max))

    if n_gram == 2:
        mid_set = pools.mid_anch_2
        df_mid = pools.df2_mid
    elif n_gram == 3:
        mid_set = pools.mid_anch_3
        df_mid = pools.df3_mid
    else:
        return None

    best_expose_by_pos: Dict[int, int] = {}

    # Structured candidates
    struct_cand: Dict[int, Island] = {}
    for i, tok in enumerate(tokens):
        if is_punct(tok):
            continue
        if not _is_structured_token(tok):
            continue
        e = max(0, wordchar_len(tok))
        if e <= 0:
            continue
        has_digit = 1.0 if DIGIT_RE.search(tok) else 0.0
        has_hyphen = 1.0 if "-" in tok else 0.0
        score = 1.0 + 0.15 * e + 0.8 * has_digit + 0.4 * has_hyphen
        pat = _render_structured_token(tok)
        struct_cand[i] = Island(kind="struct", pos=i, tok=tok, expose_len=e, score=float(score), pattern=pat)
        best_expose_by_pos[i] = max(best_expose_by_pos.get(i, 0), e)

    # N-gram candidates
    ngram_cand: Dict[int, List[Island]] = {}
    for i, tok in enumerate(tokens):
        if is_punct(tok):
            continue
        if not ALPHA_RE.fullmatch(tok):
            continue
        if tok.lower() in pools.stop_alpha_tokens:
            continue

        cands = _token_ngram_candidates(tok, n_gram, mid_set, df_mid)
        if not cands:
            continue

        lst: List[Island] = []
        for st, gr, sc in cands:
            e = int(n_gram)
            pat = _token_pattern_from_ngram_anchor(
                tok=tok,
                gram=gr,
                start=st,
                n=n_gram,
                slack=anchor_slack,
                cap_side=anchor_cap_side,
            )
            lst.append(Island(
                kind="ngram",
                pos=i,
                tok=tok,
                expose_len=e,
                score=float(sc),
                pattern=pat,
                n=int(n_gram),
                gram=str(gr),
                start=int(st),
            ))
        lst.sort(key=lambda x: x.score, reverse=True)
        lst = lst[:12]
        ngram_cand[i] = lst
        best_expose_by_pos[i] = max(best_expose_by_pos.get(i, 0), int(n_gram))

    if not best_expose_by_pos:
        return None

    raw_budget = int(math.ceil(expose_ratio * W))
    raw_budget = max(1, raw_budget)

    exposures = sorted(best_expose_by_pos.values(), reverse=True)
    max_possible = sum(exposures[:k_max]) if exposures else 0
    if max_possible <= 0:
        return None
    budget = min(raw_budget, max_possible)

    chosen: Dict[int, Island] = {}
    used_tok_low: Set[str] = set()
    used_gram_low: Set[str] = set()
    exposure_sum = 0

    struct_positions = list(struct_cand.keys())

    def can_take(island: Island) -> bool:
        if island.pos in chosen:
            return False
        tl = (island.tok or "").lower()
        if island.kind == "struct":
            if tl in used_tok_low:
                return False
            return True
        if tl in used_tok_low:
            return False
        gl = (island.gram or "").lower()
        if gl and gl in used_gram_low:
            return False
        return True

    # 1) structured first
    while struct_positions and len(chosen) < k_max and exposure_sum < budget:
        ws = [max(1e-12, struct_cand[p].score) for p in struct_positions]
        p = _weighted_choice(struct_positions, ws, rnd)
        isl = struct_cand[p]
        struct_positions.remove(p)
        if not can_take(isl):
            continue
        chosen[p] = isl
        used_tok_low.add(isl.tok.lower())
        exposure_sum += isl.expose_len

    # 2) fill with n-grams
    ngram_positions = list(ngram_cand.keys())
    while ngram_positions and len(chosen) < k_max and exposure_sum < budget:
        pos_best = []
        for p in ngram_positions:
            if p in chosen:
                continue
            lst = ngram_cand.get(p, [])
            if not lst:
                continue
            pos_best.append((p, lst[0].score))
        if not pos_best:
            break
        ps = [x[0] for x in pos_best]
        ws = [max(1e-12, x[1]) for x in pos_best]
        p = _weighted_choice(ps, ws, rnd)

        lst = ngram_cand[p]
        filt = [c for c in lst if (c.gram.lower() not in used_gram_low) and (c.tok.lower() not in used_tok_low)]
        pool = filt if filt else lst
        ws2 = [max(1e-12, c.score) for c in pool]
        idx = _weighted_choice(list(range(len(pool))), ws2, rnd)
        isl = pool[idx]

        if p in ngram_positions:
            ngram_positions.remove(p)
        if not can_take(isl):
            continue
        chosen[p] = isl
        used_tok_low.add(isl.tok.lower())
        used_gram_low.add(isl.gram.lower())
        exposure_sum += isl.expose_len

    # 3) ensure at least one ngram if possible (make n_gram knob visible)
    if len(chosen) < k_max:
        has_ngram = any(v.kind == "ngram" for v in chosen.values())
        if (not has_ngram) and ngram_cand:
            cand_pos = [p for p in ngram_cand.keys() if p not in chosen]
            if cand_pos:
                cand_pos.sort(key=lambda p: ngram_cand[p][0].score, reverse=True)
                for p in cand_pos[:20]:
                    for isl in ngram_cand[p][:6]:
                        if can_take(isl):
                            chosen[p] = isl
                            used_tok_low.add(isl.tok.lower())
                            used_gram_low.add(isl.gram.lower())
                            exposure_sum += isl.expose_len
                            break
                    if any(v.kind == "ngram" for v in chosen.values()):
                        break

    if not chosen:
        return None

    islands = sorted(chosen.values(), key=lambda x: x.pos)
    actual_ratio = float(exposure_sum) / float(W) if W > 0 else 0.0

    meta = {
        "word_chars_total": int(W),
        "budget_raw": int(raw_budget),
        "budget_capped": int(budget),
        "max_possible": int(max_possible),
        "exposed_len": int(exposure_sum),
        "expose_ratio_target": float(expose_ratio),
        "expose_ratio_actual": float(actual_ratio),
        "n_islands": int(len(islands)),
        "n_struct": int(sum(1 for x in islands if x.kind == "struct")),
        "n_ngram": int(sum(1 for x in islands if x.kind == "ngram")),
    }
    return islands, meta


# =====================================================================================
# Assemble full regex from islands
# =====================================================================================
def assemble_regex_from_islands(
    tokens: List[str],
    islands: List[Island],
    *,
    gap_base: int,
    gap_per_token: int,
    gap_cap: int,
    prefix_any_g: int,
    suffix_any_g: int,
    case_insensitive: bool,
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not islands:
        return None, {"reason": "no_islands"}

    parts: List[str] = []
    if case_insensitive:
        parts.append("(?i)")
    if prefix_any_g > 0:
        parts.append(anychar_gap_pattern(prefix_any_g))

    parts.append(islands[0].pattern)
    prev_pos = islands[0].pos

    for isl in islands[1:]:
        dist = isl.pos - prev_pos - 1
        parts.append(gap_pattern_char(dist, gap_base, gap_per_token, gap_cap))
        parts.append(isl.pattern)
        prev_pos = isl.pos

    if suffix_any_g > 0:
        parts.append(anychar_gap_pattern(suffix_any_g))

    regex = "".join(parts)
    stats = {
        "kept_positions": [x.pos for x in islands],
        "islands": [
            {
                "kind": x.kind,
                "pos": x.pos,
                "tok": x.tok,
                "expose_len": x.expose_len,
                "n": x.n,
                "gram": x.gram,
                "start": x.start,
            }
            for x in islands
        ],
    }
    return regex, stats


# =====================================================================================
# Seed verification
# =====================================================================================
def approx_verify_seed_py(seed: str, pg_regex: str) -> bool:
    try:
        rx = pg_regex
        rx = rx.replace(r"\y", r"\b")
        rx = rx.replace(r"\m", r"\b")
        rx = rx.replace(r"\M", r"\b")
        return re.search(rx, seed, flags=re.DOTALL) is not None
    except Exception:
        return False


# =====================================================================================
# DB helpers: id bounds + split ranges + id sampling
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
        self.lo = lo
        self.hi = hi
        self.rnd = rnd

    def next_id(self) -> int:
        return self.rnd.randint(self.lo, self.hi)


class StrideIdSampler(IdSampler):
    def __init__(self, lo: int, hi: int, rnd: random.Random):
        self.lo = lo
        self.hi = hi
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
    """
    Fetch (seed_id, seed_string, tokens) by sampling id range.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        while True:
            row = None
            if id_lo <= id_hi:
                rid = sampler.next_id() if sampler is not None else random.randint(id_lo, id_hi)

                # 1) exact id
                cur.execute(f"SELECT {id_col} AS sid, {col} AS text_col FROM {table} WHERE {id_col} = %s;", (rid,))
                row = cur.fetchone()

                if not row:
                    # 2) next existing id in range
                    cur.execute(
                        f"SELECT {id_col} AS sid, {col} AS text_col FROM {table} "
                        f"WHERE {id_col} > %s AND {id_col} <= %s ORDER BY {id_col} LIMIT 1;",
                        (rid, id_hi),
                    )
                    row = cur.fetchone()

                    if not row:
                        # 3) wrap
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
    # verify
    seed_verify: str,
    # progress
    progress_every: int,
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

    # dedup controls
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
                    conn=conn,
                    table=table,
                    col=col,
                    id_col=id_col,
                    min_tokens=seed_min_tokens,
                    id_lo=id_lo,
                    id_hi=id_hi,
                    sampler=sampler,
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

            picked = pick_islands_by_exposure_budget(
                seed=seed,
                tokens=toks,
                pools=pools,
                n_gram=int(n_gram),
                expose_ratio=float(expose_ratio),
                k_max=int(k_max),
                rnd=rnd,
                anchor_slack=int(anchor_slack),
                anchor_cap_side=int(anchor_cap_side),
            )
            if picked is None:
                reasons["no_islands_or_budget_fail"] += 1
                continue
            islands, budget_meta = picked

            regex, gen_stats = assemble_regex_from_islands(
                tokens=toks,
                islands=islands,
                gap_base=int(gap_base),
                gap_per_token=int(gap_per_token),
                gap_cap=int(gap_cap),
                prefix_any_g=int(prefix_any_g),
                suffix_any_g=int(suffix_any_g),
                case_insensitive=bool(case_insensitive),
            )
            if not regex:
                reasons[gen_stats.get("reason", "gen_fail")] += 1
                continue

            # hygiene: regex-only dedup/cap
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

            # seed verify
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

            # NOTE: sql_E 已按你的要求彻底移除
            rec = {
                "dataset": dataset,
                "seed_id": int(seed_id),
                "seed_string": seed,
                "regex": regex,

                # knobs
                "anchor_n": int(n_gram),
                "expose_ratio_target": float(expose_ratio),
                "k_max": int(k_max),

                # what was actually exposed (length-budget semantics)
                "exposure": budget_meta,

                # debug islands
                "islands": gen_stats.get("islands", []),
                "kept_positions": gen_stats.get("kept_positions", []),

                "stats": {
                    "seed_len_tokens": int(len(toks)),
                    "n_islands": int(len(islands)),
                },
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1

            if progress_every > 0 and (ok % int(progress_every) == 0):
                print(
                    f"[worker {worker_id}] ok={ok}/{num_target} attempts={attempts} "
                    f"last_ratio={budget_meta.get('expose_ratio_actual'):.4f}",
                    flush=True,
                )

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

    ap.add_argument("--mode", choices=["gen"], required=True)

    # Pools
    ap.add_argument("--pools", required=True)

    # DB
    ap.add_argument("--pg_dsn", default="dbname=regex_db user=lyz host=localhost port=5000")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--col", required=True)
    ap.add_argument("--id_col", default="id")

    # output
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tmp_dir", default="", help="Default: <out_dir>/tmp_workers")

    # error logs
    ap.add_argument("--err_log_dir", default="./log/gen_workload/ngram.log")

    # sampling / anti-dup
    ap.add_argument("--split_id_ranges", dest="split_id_ranges", action="store_true", default=True)
    ap.add_argument("--no_split_id_ranges", dest="split_id_ranges", action="store_false")
    ap.add_argument("--dedup_in_worker", dest="dedup_in_worker", action="store_true", default=True)
    ap.add_argument("--no_dedup_in_worker", dest="dedup_in_worker", action="store_false")
    ap.add_argument("--max_per_seed", type=int, default=4)
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--dedup", action="store_true", help="Dedup on merge by (seed_id, regex) hash key.")
    ap.add_argument("--seed_sampler", choices=["stride", "random"], default="stride")
    ap.add_argument("--max_attempts_per_worker", type=int, default=2_000_000)
    ap.add_argument("--seed_min_tokens", type=int, default=6)

    # =======================
    # TWO KNOBS (your story)
    # =======================
    ap.add_argument("--n_gram", type=int, choices=[2, 3], required=True)
    ap.add_argument(
        "--expose_ratio",
        type=float,
        required=True,
        help="Exposure budget semantics: exposed_len(word-chars) / total_word_chars(seed).",
    )
    ap.add_argument("--k_max", type=int, default=12, help="Hard cap of number of islands in one regex (fixed for fairness).")

    # fixed policies (not part of the story)
    ap.add_argument("--gap_base", type=int, default=90)
    ap.add_argument("--gap_per_token", type=int, default=20)
    ap.add_argument("--gap_cap", type=int, default=260)
    ap.add_argument("--anchor_slack", type=int, default=1)
    ap.add_argument("--anchor_cap_side", type=int, default=10)
    ap.add_argument("--prefix_any_g", type=int, default=0)
    ap.add_argument("--suffix_any_g", type=int, default=0)
    ap.add_argument("--case_insensitive", action="store_true", default=False)

    # hygiene
    ap.add_argument("--dedup_regex_only_in_worker", action="store_true", default=True)
    ap.add_argument("--no_dedup_regex_only_in_worker", dest="dedup_regex_only_in_worker", action="store_false")
    ap.add_argument("--max_per_regex", type=int, default=0)

    # verify
    ap.add_argument("--seed_verify", choices=["py", "off", "pg"], default="py")

    # progress
    ap.add_argument("--progress_every", type=int, default=200, help="Print per-worker progress every N accepted samples (0=off).")

    args = ap.parse_args()

    # gen
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_dir = args.tmp_dir or os.path.join(out_dir if out_dir else ".", "tmp_workers")
    os.makedirs(tmp_dir, exist_ok=True)

    # Resolve global id bounds
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
                args.k_max,
                # fixed
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
                # progress
                args.progress_every,
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

    print("[INFO] knobs: --n_gram (2/3) and --expose_ratio (length-budget semantics).")
    print("[INFO] fixed policies: structured-token islands prioritized; digits generalized; ngram uses light alternation; long gaps by default.")
    print("[INFO] seed_verify: use 'py' for speed; 'pg' is much slower (one SQL per attempt).")


if __name__ == "__main__":
    main()
