#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 gen_workload_v8.py 的训练数据生成脚本

功能：
1. 完全复用 gen_workload_v8.py 的 regex 生成逻辑（pools, islands, gaps 等）
2. 为每个生成的 regex 查询数据库，找到多个匹配
3. 从匹配文本中提取固定长度的 prefix
4. 输出格式：{"regex": "...", "prefixes": ["p1", "p2", ...], ...}

使用示例：
    python generate_training_data_v8.py \
        --pools ./pools/pools_lineitem_v6.json \
        --pg_dsn "dbname=sql_data_test user=postgres host=localhost port=5432" \
        --table lineitem \
        --col text_content \
        --id_col id \
        --out ./training_data.jsonl \
        --num_regex 20000 \
        --target_prefixes 10 \
        --fixed_prefix_tokens 6 \
        --workers 8 \
        --n_gram 3 \
        --expose_ratio 0.3 \
        --k_max 12 \
        --gap_base 30 \
        --gap_per_token 6 \
        --gap_cap 120 \
        --gap_kind anychar \
        --anchor_slack 0 \
        --anchor_cap_side 8 \
        --struct_ratio 0.5 \
        --struct_digit_mode literal \
        --ngram_alt_mode off
"""

import argparse
import json
import math
import os
import random
import re
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm


# =====================================================================================
# Tokenization (与 gen_workload_v8.py 完全一致)
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


def count_non_punct_tokens(s: str) -> int:
    return sum(1 for t in tokenize(s) if not is_punct(t))


def extract_prefix(s: str, k: int) -> Optional[str]:
    """提取前 k 个非标点 token 的前缀"""
    if k <= 0 or not s:
        return None
    matches = list(TOKEN_RE.finditer(s))
    cnt, end, idx = 0, None, -1
    for i, m in enumerate(matches):
        if not is_punct(m.group(0)):
            cnt += 1
            if cnt == k:
                end, idx = m.end(), i
                break
    if end is None:
        return None
    # 包含尾随标点
    j = idx + 1
    while j < len(matches) and is_punct(matches[j].group(0)):
        end = matches[j].end()
        j += 1
    return s[:end].rstrip() or None


# =====================================================================================
# PG-safe regex building blocks (与 gen_workload_v8.py 完全一致)
# =====================================================================================
WORD_CH = r"[A-Za-z0-9_]"
GapKind = Literal["anychar", "space", "nonword"]


def _norm_gap_kind(kind: str) -> GapKind:
    k = (kind or "anychar").strip().lower()
    if k in ("space", "nonword", "anychar"):
        return k  # type: ignore
    return "anychar"


def gap_pattern(N: int, kind: GapKind) -> str:
    N = max(0, int(N))
    if kind == "space":
        return rf"(?:[[:space:]]){{0,{N}}}?"
    if kind == "nonword":
        return rf"(?:[^A-Za-z0-9_]){{0,{N}}}?"
    return rf"(?:.|\n){{0,{N}}}?"


def gap_pattern_char(dist_tokens: int, base: int, per_token: int, cap: int, kind: GapKind) -> str:
    N = base + per_token * max(0, int(dist_tokens))
    N = min(int(cap), int(N))
    return gap_pattern(N, kind)


def token_pattern_soft(tok: str) -> str:
    if re.fullmatch(r"[A-Za-z]+", tok or ""):
        return r"\y[A-Za-z]+\y"
    return rf"\y{WORD_CH}+\y"


# =====================================================================================
# Pools (与 gen_workload_v8.py 完全一致)
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
# Island 结构 (与 gen_workload_v8.py 完全一致)
# =====================================================================================
@dataclass(frozen=True)
class Island:
    kind: str
    pos: int
    tok: str
    expose_len: int
    score: float
    pattern: str
    n: int = 0
    gram: str = ""
    start: int = 0


def _weighted_choice(items: List[int], weights: List[float], rnd: random.Random) -> int:
    if not items:
        raise ValueError("empty items")
    s = sum(float(w) for w in weights)
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
    if not tok or is_punct(tok):
        return False
    return (DIGIT_RE.search(tok) is not None) or ("-" in tok)


def _render_structured_token(tok: str, *, digit_mode: str, rnd: random.Random, generalize_prob: float) -> str:
    if not tok:
        return token_pattern_soft(tok)

    mode = (digit_mode or "generalize").strip().lower()
    if mode == "mixed":
        mode = "generalize" if (rnd.random() < float(generalize_prob)) else "literal"

    if mode == "literal":
        return rf"\y{re.escape(tok)}\y"

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


def _render_repeat(cls: str, lo: int, hi: int) -> str:
    lo, hi = int(lo), int(hi)
    if hi <= 0:
        return ""
    if lo <= 0 and hi == 0:
        return ""
    if lo == hi:
        if lo <= 0:
            return ""
        if lo == 1:
            return cls
        return f"{cls}{{{lo}}}"
    lo = max(0, lo)
    hi = max(lo, hi)
    return f"{cls}{{{lo},{hi}}}"


def _anchor_alt(gram: str, n: int, *, alt_mode: str) -> str:
    gram = gram or ""
    if len(gram) < n or n <= 0:
        return re.escape(gram)
    mode = (alt_mode or "light").strip().lower()
    if mode == "off":
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
    tok: str, gram: str, start: int, n: int, *,
    slack: int = 1, cap_side: int = 10, alt_mode: str = "light",
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

    alt = _anchor_alt(gram, n, alt_mode=alt_mode)
    pre = _render_repeat(WORD_CH, lo_pre, hi_pre)
    suf = _render_repeat(WORD_CH, lo_suf, hi_suf)
    return rf"\y{pre}{alt}{suf}\y"


def _token_ngram_candidates(tok: str, n: int, mid_set: Set[str], df_mid: Dict[str, int]) -> List[Tuple[int, str, float]]:
    if not tok or not ALPHA_RE.fullmatch(tok):
        return []
    t_low = tok.lower()
    if len(t_low) < n:
        return []
    out: List[Tuple[int, str, float]] = []
    for i in range(len(t_low) - n + 1):
        g_low = t_low[i : i + n]
        if g_low not in mid_set:
            continue
        df = int(df_mid.get(g_low, 0))
        score = (1.0 / (df + 1.0)) * (1.0 + 0.03 * min(len(tok), 30))
        out.append((i, tok[i : i + n], float(score)))
    best: Dict[str, Tuple[int, str, float]] = {}
    for st, gr, sc in out:
        cur = best.get(gr)
        if cur is None or sc > cur[2]:
            best[gr] = (st, gr, sc)
    return list(best.values())


# =====================================================================================
# pick_islands_by_exposure_budget (与 gen_workload_v8.py 完全一致)
# =====================================================================================
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
    struct_ratio: float,
    struct_digit_mode: str,
    struct_generalize_prob: float,
    ngram_alt_mode: str,
) -> Optional[Tuple[List[Island], Dict[str, Any]]]:
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
        pat = _render_structured_token(tok, digit_mode=struct_digit_mode, rnd=rnd, generalize_prob=struct_generalize_prob)
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
                tok=tok, gram=gr, start=st, n=n_gram,
                slack=anchor_slack, cap_side=anchor_cap_side, alt_mode=ngram_alt_mode,
            )
            lst.append(Island(
                kind="ngram", pos=i, tok=tok, expose_len=e, score=float(sc),
                pattern=pat, n=int(n_gram), gram=str(gr), start=int(st),
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

    def can_take(island: Island) -> bool:
        if island.pos in chosen:
            return False
        tl = (island.tok or "").lower()
        if tl in used_tok_low:
            return False
        if island.kind == "ngram":
            gl = (island.gram or "").lower()
            if gl and gl in used_gram_low:
                return False
        return True

    enforce_mix = (0.0 <= float(struct_ratio) <= 1.0)
    
    if not enforce_mix:
        # old behavior: structured first
        struct_positions = list(struct_cand.keys())
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

        ngram_positions = list(ngram_cand.keys())
        while ngram_positions and len(chosen) < k_max and exposure_sum < budget:
            pos_best = [(p, ngram_cand[p][0].score) for p in ngram_positions if p not in chosen and ngram_cand.get(p)]
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
    else:
        # mix behavior
        struct_positions = list(struct_cand.keys())
        ngram_positions = list(ngram_cand.keys())

        slots = min(int(k_max), len(struct_positions) + len(ngram_positions))
        target_struct = min(len(struct_positions), max(0, int(round(float(struct_ratio) * slots))))
        target_ngram = min(len(ngram_positions), max(0, slots - target_struct))

        def pick_struct_once() -> bool:
            nonlocal exposure_sum
            if not struct_positions:
                return False
            ws = [max(1e-12, struct_cand[p].score) for p in struct_positions]
            p = _weighted_choice(struct_positions, ws, rnd)
            isl = struct_cand[p]
            struct_positions.remove(p)
            if not can_take(isl):
                return False
            chosen[p] = isl
            used_tok_low.add(isl.tok.lower())
            exposure_sum += isl.expose_len
            return True

        def pick_ngram_once() -> bool:
            nonlocal exposure_sum
            if not ngram_positions:
                return False
            pos_best = [(p, ngram_cand[p][0].score) for p in ngram_positions if p not in chosen and ngram_cand.get(p)]
            if not pos_best:
                return False
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
                return False
            chosen[p] = isl
            used_tok_low.add(isl.tok.lower())
            used_gram_low.add(isl.gram.lower())
            exposure_sum += isl.expose_len
            return True

        while (struct_positions or ngram_positions) and len(chosen) < k_max and exposure_sum < budget:
            n_struct_now = sum(1 for v in chosen.values() if v.kind == "struct")
            n_ngram_now = sum(1 for v in chosen.values() if v.kind == "ngram")

            need_struct = (n_struct_now < target_struct) and bool(struct_positions)
            need_ngram = (n_ngram_now < target_ngram) and bool(ngram_positions)

            if need_struct and need_ngram:
                rem_s = target_struct - n_struct_now
                rem_n = target_ngram - n_ngram_now
                if rnd.random() < (rem_s / max(1e-9, (rem_s + rem_n))):
                    pick_struct_once() or pick_ngram_once()
                else:
                    pick_ngram_once() or pick_struct_once()
            elif need_struct:
                pick_struct_once() or pick_ngram_once()
            elif need_ngram:
                pick_ngram_once() or pick_struct_once()
            else:
                best_s = max((struct_cand[p].score for p in struct_positions), default=-1.0)
                best_n = max((ngram_cand[p][0].score for p in ngram_positions if ngram_cand.get(p)), default=-1.0)
                if best_s >= best_n:
                    pick_struct_once() or pick_ngram_once()
                else:
                    pick_ngram_once() or pick_struct_once()
                break

    # ensure at least one ngram
    if chosen and len(chosen) < k_max:
        has_ngram = any(v.kind == "ngram" for v in chosen.values())
        if (not has_ngram) and ngram_cand and (not (enforce_mix and float(struct_ratio) >= 0.999)):
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
        "exposed_len": int(exposure_sum),
        "expose_ratio_actual": float(actual_ratio),
        "n_islands": int(len(islands)),
        "n_struct": int(sum(1 for x in islands if x.kind == "struct")),
        "n_ngram": int(sum(1 for x in islands if x.kind == "ngram")),
    }
    return islands, meta


# =====================================================================================
# assemble_regex_from_islands (与 gen_workload_v8.py 完全一致)
# =====================================================================================
def assemble_regex_from_islands(
    tokens: List[str],
    islands: List[Island],
    *,
    gap_base: int,
    gap_per_token: int,
    gap_cap: int,
    gap_kind: str,
    case_insensitive: bool = True,
) -> Optional[str]:
    if not islands:
        return None

    gk = _norm_gap_kind(gap_kind)

    parts: List[str] = []
    if case_insensitive:
        parts.append("(?i)")

    parts.append(islands[0].pattern)
    prev_pos = islands[0].pos

    for isl in islands[1:]:
        dist = isl.pos - prev_pos - 1
        parts.append(gap_pattern_char(dist, gap_base, gap_per_token, gap_cap, gk))
        parts.append(isl.pattern)
        prev_pos = isl.pos

    return "".join(parts)


# =====================================================================================
# Seed 验证 (与 gen_workload_v8.py 完全一致)
# =====================================================================================
def verify_seed_py(seed: str, pg_regex: str) -> bool:
    try:
        rx = pg_regex.replace(r"\y", r"\b").replace(r"\m", r"\b").replace(r"\M", r"\b")
        return re.search(rx, seed, flags=re.DOTALL) is not None
    except Exception:
        return False


# =====================================================================================
# Worker 逻辑
# =====================================================================================
def worker_generate(
    worker_id: int,
    num_target: int,
    pools_json: str,
    pg_dsn: str,
    table: str,
    col: str,
    id_col: str,
    id_lo: int,
    id_hi: int,
    seed_min_tokens: int,
    max_attempts: int,
    rng_seed: int,
    # regex 生成参数
    n_gram: int,
    expose_ratio: float,
    k_max: int,
    gap_base: int,
    gap_per_token: int,
    gap_cap: int,
    gap_kind: str,
    anchor_slack: int,
    anchor_cap_side: int,
    struct_ratio: float,
    struct_digit_mode: str,
    struct_generalize_prob: float,
    ngram_alt_mode: str,
    # prefix 参数
    target_prefixes: int,
    fixed_prefix_tokens: int,
    max_trg_len: int,
    query_timeout: str,
) -> Dict[str, Any]:
    
    rnd = random.Random((int(rng_seed) * 1000003) ^ (worker_id * 9176))
    pools = load_pools(pools_json)

    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = True
    
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = '{query_timeout}';")

    results = []
    stats = {"attempts": 0, "success": 0, "no_islands": 0, "verify_fail": 0, "no_prefixes": 0}
    
    seen_regex: Set[str] = set()

    while len(results) < num_target and stats["attempts"] < max_attempts:
        stats["attempts"] += 1

        # 1. 随机采样一个 seed
        try:
            rid = rnd.randint(id_lo, id_hi)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT {id_col} AS sid, {col} AS text_col 
                    FROM {table} 
                    WHERE {id_col} >= %s 
                    ORDER BY {id_col} 
                    LIMIT 1;
                """, (rid,))
                row = cur.fetchone()
                
            if not row:
                continue
                
            seed = row.get("text_col", "") or ""
            seed_id = row.get("sid")
            toks = tokenize(seed)
            
            if len(toks) < seed_min_tokens:
                continue
                
        except Exception:
            continue

        # 2. 生成 regex
        picked = pick_islands_by_exposure_budget(
            seed=seed,
            tokens=toks,
            pools=pools,
            n_gram=n_gram,
            expose_ratio=expose_ratio,
            k_max=k_max,
            rnd=rnd,
            anchor_slack=anchor_slack,
            anchor_cap_side=anchor_cap_side,
            struct_ratio=struct_ratio,
            struct_digit_mode=struct_digit_mode,
            struct_generalize_prob=struct_generalize_prob,
            ngram_alt_mode=ngram_alt_mode,
        )
        
        if picked is None:
            stats["no_islands"] += 1
            continue
            
        islands, meta = picked

        regex = assemble_regex_from_islands(
            tokens=toks,
            islands=islands,
            gap_base=gap_base,
            gap_per_token=gap_per_token,
            gap_cap=gap_cap,
            gap_kind=gap_kind,
            case_insensitive=True,
        )
        
        if not regex:
            stats["no_islands"] += 1
            continue

        # 3. 去重
        if regex in seen_regex:
            continue
        seen_regex.add(regex)

        # 4. 验证
        if not verify_seed_py(seed, regex):
            stats["verify_fail"] += 1
            continue

        # 5. 查询数据库找更多匹配
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT {col}
                    FROM {table}
                    WHERE {col} ~* %s
                    LIMIT %s;
                """, (regex, target_prefixes * 3))
                matches = [r[0] for r in cur.fetchall() if r[0]]
        except Exception:
            matches = []

        # 6. 提取固定长度的 prefix
        prefixes = set()
        for m in matches:
            if count_non_punct_tokens(m) >= fixed_prefix_tokens:
                p = extract_prefix(m, fixed_prefix_tokens)
                if p and len(p) <= max_trg_len:
                    prefixes.add(p)
                    if len(prefixes) >= target_prefixes:
                        break

        # 确保原 seed 的 prefix 在里面
        orig_p = extract_prefix(seed, fixed_prefix_tokens)
        if orig_p and len(orig_p) <= max_trg_len:
            prefixes.add(orig_p)

        if not prefixes:
            stats["no_prefixes"] += 1
            continue

        # 7. 保存结果
        results.append({
            "regex": regex,
            "prefixes": list(prefixes)[:target_prefixes],
            "seed_id": seed_id,
            "n_islands": meta["n_islands"],
            "expose_ratio_actual": meta["expose_ratio_actual"],
        })
        stats["success"] += 1

        if len(results) % 100 == 0:
            print(f"[Worker {worker_id}] {len(results)}/{num_target} done, "
                  f"attempts={stats['attempts']}", flush=True)

    conn.close()
    return {"results": results, "stats": stats, "worker_id": worker_id}


# =====================================================================================
# Main
# =====================================================================================
def main():
    p = argparse.ArgumentParser(description="Generate training data based on gen_workload_v8.py logic")
    
    # 数据库
    p.add_argument("--pg_dsn", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--col", required=True)
    p.add_argument("--id_col", default="id")
    p.add_argument("--pools", required=True, help="Path to pools JSON file")
    
    # 输出
    p.add_argument("--out", required=True)
    
    # 数量
    p.add_argument("--num_regex", type=int, default=20000)
    p.add_argument("--target_prefixes", type=int, default=10)
    p.add_argument("--fixed_prefix_tokens", type=int, default=6)
    p.add_argument("--max_trg_len", type=int, default=160)
    
    # 并行
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_attempts_per_worker", type=int, default=500000)
    p.add_argument("--query_timeout", type=str, default="5s")
    p.add_argument("--seed_min_tokens", type=int, default=6)
    p.add_argument("--rng_seed", type=int, default=42)
    
    # Regex 生成参数 (与 gen_workload_v8.py 一致)
    p.add_argument("--n_gram", type=int, choices=[2, 3], default=3)
    p.add_argument("--expose_ratio", type=float, default=0.3)
    p.add_argument("--k_max", type=int, default=12)
    p.add_argument("--gap_base", type=int, default=30)
    p.add_argument("--gap_per_token", type=int, default=6)
    p.add_argument("--gap_cap", type=int, default=120)
    p.add_argument("--gap_kind", choices=["anychar", "space", "nonword"], default="anychar")
    p.add_argument("--anchor_slack", type=int, default=0)
    p.add_argument("--anchor_cap_side", type=int, default=8)
    p.add_argument("--struct_ratio", type=float, default=0.5)
    p.add_argument("--struct_digit_mode", choices=["generalize", "literal", "mixed"], default="literal")
    p.add_argument("--struct_generalize_prob", type=float, default=0.7)
    p.add_argument("--ngram_alt_mode", choices=["light", "off"], default="off")
    
    args = p.parse_args()

    print(f"\n{'='*70}")
    print("Training Data Generator (based on gen_workload_v8.py)")
    print(f"{'='*70}")
    print(f"Target: {args.num_regex} regex × up to {args.target_prefixes} prefixes")
    print(f"Fixed prefix length: {args.fixed_prefix_tokens} tokens")
    print(f"Workers: {args.workers}")
    print(f"{'='*70}\n")

    t0 = time.time()

    # 获取 ID 范围
    conn = psycopg2.connect(args.pg_dsn)
    with conn.cursor() as cur:
        cur.execute(f"SELECT min({args.id_col}), max({args.id_col}) FROM {args.table};")
        mn, mx = cur.fetchone()
    conn.close()

    if mn is None or mx is None:
        raise SystemExit("Could not get ID bounds")

    print(f"[DB] ID range: {mn} - {mx}")

    # 分配任务
    W = args.workers
    per = args.num_regex // W
    rem = args.num_regex % W
    targets = [per + (1 if i < rem else 0) for i in range(W)]
    
    # 分配 ID 范围
    span = mx - mn + 1
    step = max(1, span // W)
    ranges = []
    cur = mn
    for i in range(W):
        lo = cur
        hi = mx if i == W - 1 else min(mx, lo + step - 1)
        ranges.append((lo, hi))
        cur = hi + 1

    # 并行执行
    all_results = []
    all_stats = defaultdict(int)

    with ProcessPoolExecutor(max_workers=W) as ex:
        futures = []
        for wid in range(W):
            if targets[wid] <= 0:
                continue
            lo, hi = ranges[wid]
            futures.append(ex.submit(
                worker_generate,
                wid, targets[wid], args.pools, args.pg_dsn,
                args.table, args.col, args.id_col, lo, hi,
                args.seed_min_tokens, args.max_attempts_per_worker, args.rng_seed,
                args.n_gram, args.expose_ratio, args.k_max,
                args.gap_base, args.gap_per_token, args.gap_cap, args.gap_kind,
                args.anchor_slack, args.anchor_cap_side,
                args.struct_ratio, args.struct_digit_mode, args.struct_generalize_prob,
                args.ngram_alt_mode,
                args.target_prefixes, args.fixed_prefix_tokens, args.max_trg_len,
                args.query_timeout,
            ))

        for fu in tqdm(as_completed(futures), total=len(futures), desc="Workers"):
            res = fu.result()
            all_results.extend(res["results"])
            for k, v in res["stats"].items():
                all_stats[k] += v

    # 保存结果
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    prefix_counts = [len(r["prefixes"]) for r in all_results]
    
    print(f"\n{'='*70}")
    print(f"Results: {len(all_results)} groups")
    print(f"{'='*70}")
    print(f"Stats: {dict(all_stats)}")
    
    if prefix_counts:
        print(f"\nPrefix distribution:")
        print(f"  Min: {min(prefix_counts)}")
        print(f"  Max: {max(prefix_counts)}")
        print(f"  Avg: {sum(prefix_counts)/len(prefix_counts):.2f}")
        
        buckets = {">=10": 0, "5-9": 0, "2-4": 0, "1": 0}
        for c in prefix_counts:
            if c >= 10: buckets[">=10"] += 1
            elif c >= 5: buckets["5-9"] += 1
            elif c >= 2: buckets["2-4"] += 1
            else: buckets["1"] += 1
        
        for k, v in buckets.items():
            pct = 100 * v / len(prefix_counts)
            print(f"  {k:>5}: {v:>5} ({pct:>5.1f}%)")

    print(f"\nOutput: {args.out}")
    print(f"File size: {os.path.getsize(args.out)/1024/1024:.1f} MB")
    print(f"Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()