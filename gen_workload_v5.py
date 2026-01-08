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
from typing import List, Tuple, Optional, Dict, Any, Set

import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================================================================================
# Tokenization
# =====================================================================================
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)


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
    """
    If err_log_dir looks like a file path (e.g., ./log/gen_workload/v5.log),
    create per-worker files:
      ./log/gen_workload/v5.log.worker_0.jsonl
    Else treat it as a directory and write:
      <dir>/worker_0.jsonl
    """
    err_log_dir = (err_log_dir or "").strip()
    if not err_log_dir:
        err_log_dir = "./log/gen_workload/v5.log"

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
# Regex helpers: max literal run (proxy)
# =====================================================================================
def max_literal_run_len(regex: str) -> int:
    """
    Approx max consecutive literal alnum run outside charclass and escapes.
    (Proxy only; not exact PG behavior.)
    """
    in_class = False
    run = best = 0
    i = 0
    while i < len(regex):
        c = regex[i]
        if in_class:
            if c == "]":
                in_class = False
            i += 1
            continue
        if c == "[":
            in_class = True
            run = 0
            i += 1
            continue
        if c == "\\":
            run = 0
            i += 2
            continue
        if c.isalnum():
            run += 1
            best = max(best, run)
        else:
            run = 0
        i += 1
    return best


# =====================================================================================
# trgm_budget proxy (static)
# =====================================================================================
ABSTRACT_ESCAPES = set("dDsSwWymM")  # \d \D \s \S \w \W \y \m \M


def _is_literal_char(c: str) -> bool:
    return c.isalnum() or c in "-_"


def extract_literal_trigrams(regex: str) -> Set[str]:
    """
    Extract unique 3-grams from contiguous literal segments outside [...] and outside abstract escapes.
    Proxy of pg_trgm extraction; used to bucket difficulty.
    """
    grams: Set[str] = set()
    in_class = False
    seg: List[str] = []

    def flush_seg():
        nonlocal seg, grams
        if len(seg) >= 3:
            s = "".join(seg).lower()
            for j in range(len(s) - 2):
                grams.add(s[j:j + 3])
        seg = []

    i = 0
    n = len(regex)
    while i < n:
        c = regex[i]
        if in_class:
            if c == "]":
                in_class = False
            i += 1
            continue

        if c == "[":
            flush_seg()
            in_class = True
            i += 1
            continue

        if c == "\\":
            if i + 1 >= n:
                flush_seg()
                break
            esc = regex[i + 1]
            if esc in ABSTRACT_ESCAPES:
                flush_seg()
                i += 2
                continue
            if _is_literal_char(esc):
                seg.append(esc)
            else:
                flush_seg()
            i += 2
            continue

        if c in "^$(){}|+*?.":  # unescaped metachar
            flush_seg()
            i += 1
            continue

        if _is_literal_char(c):
            seg.append(c)
        else:
            flush_seg()

        i += 1

    flush_seg()
    return grams


def trgm_budget(regex: str) -> int:
    return len(extract_literal_trigrams(regex))


# =====================================================================================
# Stable hash for (seed_id, regex) de-dup
# =====================================================================================
def _hash64_str(s: str) -> int:
    b = s.encode("utf-8", errors="ignore")
    h1 = zlib.crc32(b) & 0xFFFFFFFF
    h2 = zlib.adler32(b) & 0xFFFFFFFF
    return (h1 << 32) | h2


def pair_key_u128(seed_id: int, regex: str) -> int:
    return (int(seed_id) << 64) | _hash64_str(regex)


# =====================================================================================
# SQL E'...' helpers (for exporting runnable SQL)
# =====================================================================================
def sql_escape_E_literal(s: str) -> str:
    # For E'...': escape backslash and single quote.
    # IMPORTANT: this already produces a runnable E'...' literal.
    # Do NOT add extra backslashes manually when you copy-paste sql_E into psql.
    return s.replace("\\", "\\\\").replace("'", "''")


def make_where_regex_sql(select_expr: str, table: str, col: str, regex: str, limit: Optional[int] = None) -> str:
    rx = sql_escape_E_literal(regex)
    q = f"SELECT {select_expr} FROM {table} WHERE {col} ~ E'{rx}'"
    if limit is not None and int(limit) > 0:
        q += f" LIMIT {int(limit)}"
    q += ";"
    return q


# =====================================================================================
# Basic token utilities
# =====================================================================================
def keep_numeric_literal(tok: str) -> bool:
    return any(ch.isdigit() for ch in tok)


# Road / traffic domain lexicon (you can extend)
ROAD_TYPES = {
    "blvd": ("Blvd", "Boulevard"),
    "boulevard": ("Blvd", "Boulevard"),
    "ave": ("Ave", "Avenue"),
    "avenue": ("Ave", "Avenue"),
    "rd": ("Rd", "Road"),
    "road": ("Rd", "Road"),
    "st": ("St", "Street"),
    "street": ("St", "Street"),
    "hwy": ("Hwy", "Highway"),
    "highway": ("Hwy", "Highway"),
    "fwy": ("Fwy", "Freeway"),
    "freeway": ("Fwy", "Freeway"),
    "expwy": ("Expwy", "Expressway"),
    "expressway": ("Expwy", "Expressway"),
}


def is_road_type(tok: str) -> bool:
    t = tok.lower().rstrip(".")
    return t in ROAD_TYPES


def road_type_pattern(tok: str) -> str:
    t = tok.lower().rstrip(".")
    a, b = ROAD_TYPES.get(t, (tok, tok))
    # Use Postgres word-boundary \y
    return rf"(?:{re.escape(a)}|{re.escape(b)})\.?\y"


# Road IDs like I-10, US-101, SR-99, CA-1, WA-167 etc.
ROAD_ID_RE = re.compile(
    r"^(I|US|SR|CA|WA|OR|TX|FL|GA|IL|NY|NJ|PA|AZ|NV|UT|CO|NM|NC|SC|VA|MD|MA|MI|OH|OK|LA|AL|TN|IN|WI|MN|IA|MO|KS|NE|SD|ND|ID|MT|WY|AK|HI)\-[0-9]{1,4}$",
    re.IGNORECASE
)


def is_road_id(tok: str) -> bool:
    return bool(ROAD_ID_RE.fullmatch(tok))


def road_id_pattern(tok: str) -> str:
    return re.escape(tok)


# Exit patterns
def is_exit_word(tok: str) -> bool:
    return tok.lower() == "exit"


def seed_has_road_or_exit_tokens(toks: List[str]) -> bool:
    """
    True if token list contains a road-id like I-10 / US-40
    OR contains 'Exit' followed by a numeric-ish token.
    """
    for i, tok in enumerate(toks):
        if is_road_id(tok):
            return True
        if is_exit_word(tok) and (i + 1) < len(toks) and keep_numeric_literal(toks[i + 1]):
            return True
    return False


# =====================================================================================
# Pools (DF mid-frequency anchors)
# =====================================================================================
@dataclass
class Pools:
    anchor_len: int
    mid_anch: Set[str]


def build_pools(
    csv_path: str,
    out_json: str,
    anchor_len: int = 3,
    df_low_pct: float = 0.005,
    df_high_pct: float = 0.15,
    max_rows: Optional[int] = None,
):
    """
    pools 文件仍然只保存 mid_anch（避免巨大的 DF 表）。
    """
    df = defaultdict(int)
    N = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in tqdm(r, desc="DF pass"):
            s = row["Description"]
            toks = tokenize(s)
            seen = set(t for t in toks if (not is_punct(t)) and t.strip())
            for t in seen:
                df[t] += 1
            N += 1
            if max_rows and N >= max_rows:
                break

    low = int(df_low_pct * N)
    high = int(df_high_pct * N)

    mid_anch: Set[str] = set()
    L = int(anchor_len)

    for t, c in df.items():
        if c < low or c > high:
            continue
        if len(t) < L:
            continue
        if keep_numeric_literal(t):
            continue

        pre = t[:L]
        suf = t[-L:]

        for a in (pre, suf):
            if not re.fullmatch(r"[A-Za-z]{%d}" % L, a):
                continue
            mid_anch.add(a)

    obj = {
        "N": N,
        "df_low": low,
        "df_high": high,
        "anchor_len": L,
        "mid_anch_size": len(mid_anch),
        "mid_anch": sorted(mid_anch),
    }
    with open(out_json, "w", encoding="utf-8") as w:
        json.dump(obj, w, ensure_ascii=False)
    print(f"[OK] pools saved: {out_json}  mid_anch={len(mid_anch)}  N={N}  L={L}")


def load_pools(pools_json: str) -> Pools:
    with open(pools_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return Pools(anchor_len=int(obj["anchor_len"]), mid_anch=set(obj["mid_anch"]))


# =====================================================================================
# Anchor selection (增强：禁用形态后缀锚 + 禁用过泛 3-gram)
# =====================================================================================
LOW_VALUE_ANCH = {"the", "and", "for", "wit", "are", "you", "acc", "on", "at"}

# 这些 3-gram 非常容易导致匹配爆炸（形态后缀/常见词根），不需要 PG 就能显著降风险
BAD_SUFFIX_3GRAMS = {
    "ive", "ing", "ion", "ent", "ers", "est", "ity", "ily", "ous", "ial", "ate", "ize", "ise",
    "ment"[0:3], "tion"[0:3],  # "men", "tio"
}
BAD_GENERIC_3GRAMS = {
    "inc", "acc", "cra", "slo", "que", "veh", "mul",  # 常见头词根
    "due", "bef", "aft", "nea", "bet", "fro", "to",   # 过泛
}


def pick_df_anchors(
    tokens: List[str],
    pools: Pools,
    k: int = 2,
    prefer_pre_ratio: float = 0.75,
) -> Optional[List[Tuple[int, str, str]]]:
    """
    Pick k anchors (pos, anchor, mode) using mid-frequency L-grams from pools.
    增强点：
      - 过滤 BAD_SUFFIX_3GRAMS / BAD_GENERIC_3GRAMS
      - 更偏好 pre-anchor（一般更具辨识度）
    """
    L = pools.anchor_len
    cand_pre: List[Tuple[int, str, str]] = []
    cand_suf: List[Tuple[int, str, str]] = []
    T = len(tokens)

    for i, tok in enumerate(tokens):
        if is_punct(tok):
            continue
        if keep_numeric_literal(tok) or is_road_id(tok) or is_road_type(tok) or is_exit_word(tok):
            continue
        if len(tok) < L:
            continue

        pre = tok[:L]
        suf = tok[-L:]
        pre_l = pre.lower()
        suf_l = suf.lower()

        if pre in pools.mid_anch and pre_l not in LOW_VALUE_ANCH and pre_l not in BAD_GENERIC_3GRAMS:
            cand_pre.append((i, pre, "pre"))

        # suffix anchor 更容易触发形态学爆炸，额外过滤
        if (
            suf in pools.mid_anch
            and suf != pre
            and suf_l not in LOW_VALUE_ANCH
            and suf_l not in BAD_SUFFIX_3GRAMS
            and suf_l not in BAD_GENERIC_3GRAMS
        ):
            cand_suf.append((i, suf, "suf"))

    cand = cand_pre + cand_suf
    if len(cand) < k:
        return None

    early = [c for c in cand if c[0] <= int(0.2 * T)]
    mid = [c for c in cand if int(0.2 * T) < c[0] < int(0.8 * T)]
    late = [c for c in cand if c[0] >= int(0.8 * T)]

    picked: List[Tuple[int, str, str]] = []

    def pick_one(pool_list: List[Tuple[int, str, str]]) -> Optional[Tuple[int, str, str]]:
        if not pool_list:
            return None
        # 偏好 pre-anchor
        pre_list = [x for x in pool_list if x[2] == "pre"]
        suf_list = [x for x in pool_list if x[2] == "suf"]
        if pre_list and (random.random() < prefer_pre_ratio or not suf_list):
            return random.choice(pre_list)
        if suf_list:
            return random.choice(suf_list)
        return random.choice(pool_list)

    for bucket in (mid, late, early):
        if len(picked) >= k:
            break
        x = pick_one(bucket)
        if x is not None:
            picked.append(x)

    # 补齐
    while len(picked) < k:
        x = pick_one(cand)
        if x is None:
            break
        picked.append(x)

    # 去重（按 anchor 文本）
    uniq: Dict[str, Tuple[int, str, str]] = {}
    for x in picked + random.sample(cand, min(len(cand), 12)):
        uniq.setdefault(x[1], x)
        if len(uniq) >= k:
            break
    if len(uniq) < k:
        return None

    out = list(uniq.values())[:k]
    out.sort(key=lambda x: x[0])
    return out


# =====================================================================================
# Gap + token-pattern assembly (Postgres-compatible)
# =====================================================================================
WORD_CH = r"[A-Za-z0-9_]"


def anychar_gap_pattern(N: int) -> str:
    """
    Match any char including newline, bounded and non-greedy.
    Use (?:.|\n) rather than [\s\S] to avoid invalid \S in Postgres.
    """
    N = max(0, int(N))
    return rf"(?:.|\n){{0,{N}}}?"


def gap_pattern_char(dist_tokens: int, base: int, per_token: int, cap: int) -> str:
    N = base + per_token * max(0, dist_tokens)
    N = min(int(cap), int(N))
    return anychar_gap_pattern(N)


def _q_range(n: int, slack: int) -> Tuple[int, int]:
    if slack <= 0:
        return (n, n)
    lo = max(0, n - slack)
    hi = n + slack
    return (lo, hi)


def token_pattern_soft(tok: str) -> str:
    """
    Soft token class pattern, PG-safe:
      - alphabetic -> \y[A-Za-z]+\y
      - other word-ish -> \y[A-Za-z0-9_]+\y
    """
    if re.fullmatch(r"[A-Za-z]+", tok):
        return r"\y[A-Za-z]+\y"
    return rf"\y{WORD_CH}+\y"


def head_token_pattern_pg(
    tok: str,
    *,
    L: int = 3,
    slack: int = 1,
    prefer: str = "mix",
    strong_common: bool = True,
    strong_suf_len: int = 2,
) -> str:
    """
    Head token anchoring (no ^):
      - pre:  \y <tok[:L]> + [a-z]{lo,hi} \y
      - suf:  \y [A-Z]{lo,hi} + <tok[-L:]> \y
    增强：
      - 对常见头词根（Acc/Inc/Que/...）使用 “prefix + suffix” 双侧约束，降低匹配爆炸概率
      - 支持形如 Multi-vehicle 的连字符头词：两侧分别做 prefix 约束
    """
    if not tok:
        return r"\y[A-Za-z]+\y"

    # 连字符头词：Multi-vehicle
    m = re.fullmatch(r"([A-Za-z]+)\-([A-Za-z]+)", tok)
    if m:
        a, b = m.group(1), m.group(2)
        if len(a) >= 3 and len(b) >= 3:
            a_pre = a[:3]
            b_pre = b[:3]
            a_rest = len(a) - 3
            b_rest = len(b) - 3
            a_lo, a_hi = _q_range(a_rest, slack)
            b_lo, b_hi = _q_range(b_rest, slack)
            pat = rf"\y{re.escape(a_pre)}[a-z]{{{a_lo},{a_hi}}}\-{re.escape(b_pre)}[a-z]{{{b_lo},{b_hi}}}\y"
            return pat
        return token_pattern_soft(tok)

    if keep_numeric_literal(tok) or is_road_id(tok) or is_road_type(tok) or is_exit_word(tok):
        return token_pattern_soft(tok)

    if not re.fullmatch(r"[A-Za-z]+", tok) or len(tok) < (L + 1):
        return token_pattern_soft(tok)

    # 变量部分 char-class：尽量保持“自然感”
    if tok.islower():
        cls = r"[a-z]"
    elif tok.isupper():
        cls = r"[A-Z]"
    else:
        cls = r"[a-z]"

    # 对常见头词根使用 “prefix + suffix” 双侧约束
    if strong_common:
        root3 = tok[:3].lower()
        if root3 in BAD_GENERIC_3GRAMS and len(tok) >= (L + strong_suf_len + 1):
            pre = tok[:L]
            suf = tok[-strong_suf_len:]
            mid_len = len(tok) - L - strong_suf_len
            lo, hi = _q_range(mid_len, slack)
            return rf"\y{re.escape(pre)}{cls}{{{lo},{hi}}}{re.escape(suf)}\y"

    if prefer == "mix":
        prefer = "pre" if random.random() < 0.7 else "suf"

    if prefer == "suf":
        suf = tok[-L:]
        pre_len = len(tok) - L
        lo, hi = _q_range(pre_len, slack)
        pre_cls = r"[A-Z]" if tok[0].isupper() else cls
        pat = rf"{pre_cls}{{{lo},{hi}}}{re.escape(suf)}"
        return rf"\y{pat}\y"

    pre = tok[:L]
    rest_len = len(tok) - L
    lo, hi = _q_range(rest_len, slack)
    pat = rf"{re.escape(pre)}{cls}{{{lo},{hi}}}"
    return rf"\y{pat}\y"


def token_pattern_from_anchor(
    tok: str,
    anchor: str,
    mode: str,
    use_vehicle_synonyms: bool,
    rest_min_floor: int,
    rest_slack: int,
    rest_max_cap: int,
) -> str:
    """
    DF-anchor token pattern（增强：有上界，避免匹配爆炸）：
      - vehicle anchor: \y(?:vehicle|veh)[A-Za-z0-9_]*\y
      - else:
          pre: \y <anchor> [A-Za-z0-9_]{lo,hi} \y
          suf: \y [A-Za-z0-9_]{lo,hi} <anchor> \y
    """
    a = re.escape(anchor)

    if use_vehicle_synonyms and anchor.lower() == "veh":
        return rf"\y(?:vehicle|veh){WORD_CH}*\y"

    L = len(anchor)
    rest_len = max(0, len(tok) - L)

    if rest_len <= 0:
        return rf"\y{a}\y"

    lo = max(0, rest_len - int(rest_slack))
    hi = min(int(rest_len + int(rest_slack)), int(rest_max_cap))
    lo = max(int(rest_min_floor), lo)
    if hi < lo:
        hi = lo

    if mode == "pre":
        return rf"\y{a}{WORD_CH}{{{lo},{hi}}}\y"
    return rf"\y{WORD_CH}{{{lo},{hi}}}{a}\y"


# =====================================================================================
# Kept positions (增强：head_tokens=1 固定；可强制包含 road_id/exit_num)
# =====================================================================================
def build_kept_positions(
    tokens: List[str],
    df_anchors: Optional[List[Tuple[int, str, str]]],
    keep_road_id: bool,
    keep_road_type: bool,
    keep_exit_num: bool,
    extra_mid: int,
    keep_head_token: bool,
) -> Tuple[List[int], Dict[int, Dict[str, Any]], Dict[str, bool]]:
    meta: Dict[int, Dict[str, Any]] = {}
    kept: Set[int] = set()

    flags = {
        "has_head": False,
        "has_road_id": False,
        "has_exit_num": False,   # Exit + number
        "has_road_type": False,
    }

    # head_tokens=1 固定
    if keep_head_token:
        head_pos = None
        for i, tok in enumerate(tokens):
            if is_punct(tok):
                continue
            head_pos = i
            break
        if head_pos is not None:
            kept.add(head_pos)
            meta[head_pos] = {"kind": "head"}
            flags["has_head"] = True

    if df_anchors:
        for pos, anch, mode in df_anchors:
            kept.add(pos)
            meta[pos] = {"kind": "df_anchor", "anchor": anch, "mode": mode}

    if keep_road_id:
        for i, tok in enumerate(tokens):
            if is_road_id(tok):
                kept.add(i)
                meta[i] = {"kind": "road_id"}
                flags["has_road_id"] = True
                break

    if keep_road_type:
        for i in range(len(tokens) - 1, -1, -1):
            if is_road_type(tokens[i]):
                kept.add(i)
                meta[i] = {"kind": "road_type"}
                flags["has_road_type"] = True
                break

    if keep_exit_num:
        for i, tok in enumerate(tokens):
            if is_exit_word(tok):
                kept.add(i)
                meta[i] = {"kind": "exit"}
                # Exit number
                if i + 1 < len(tokens) and keep_numeric_literal(tokens[i + 1]):
                    kept.add(i + 1)
                    meta[i + 1] = {"kind": "exit_num"}
                    flags["has_exit_num"] = True
                break

    if extra_mid > 0:
        T = len(tokens)
        Lm = int(0.3 * T)
        Rm = int(0.7 * T)
        cand = [
            i for i in range(Lm, Rm + 1)
            if i not in kept
            and (not is_punct(tokens[i]))
            and (not is_road_type(tokens[i]))
            and (not is_road_id(tokens[i]))
        ]
        random.shuffle(cand)
        for i in cand[:extra_mid]:
            kept.add(i)
            meta[i] = {"kind": "soft"}

    kept_positions = sorted(kept)
    return kept_positions, meta, flags


# =====================================================================================
# Assemble regex
# =====================================================================================
def assemble_regex_human(
    tokens: List[str],
    pools: Pools,
    anchors_k: int,
    keep_road_id: bool,
    keep_road_type: bool,
    keep_exit_num: bool,
    extra_mid: int,
    # gaps
    gap_base: int,
    gap_per_token: int,
    gap_cap: int,
    head_gap_base: int,
    head_gap_per_token: int,
    head_gap_cap: int,
    # anchor rendering
    use_vehicle_synonyms: bool,
    rest_min_floor: int,
    rest_slack: int,
    rest_max_cap: int,
    # head rendering
    head_L: int,
    head_slack: int,
    head_strong_common: bool,
    head_strong_suf_len: int,
    # prefix
    prefix_any_g: int,
    case_insensitive: bool,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Build a human-like regex using Postgres-compatible constructs.

    关键增强（不靠 PG）：
      - head_tokens=1 固定（不加 ^），并对常见头词根用 prefix+suffix 双侧约束
      - DF anchors 的词内扩展变成 {lo,hi} 有上界
      - 首段 gap 更紧（head -> next）
    """
    df_anchors = pick_df_anchors(tokens, pools, k=anchors_k) if anchors_k > 0 else None

    kept_pos, meta, flags = build_kept_positions(
        tokens=tokens,
        df_anchors=df_anchors,
        keep_road_id=keep_road_id,
        keep_road_type=keep_road_type,
        keep_exit_num=keep_exit_num,
        extra_mid=extra_mid,
        keep_head_token=True,  # 固定
    )

    if len(kept_pos) < 2:
        return None, {"reason": "too_few_kept"}

    parts: List[str] = []
    if case_insensitive:
        parts.append("(?i)")

    if prefix_any_g > 0:
        parts.append(anychar_gap_pattern(int(prefix_any_g)))

    def render(i: int) -> str:
        tok = tokens[i]
        info = meta.get(i, {"kind": "soft"})
        kind = info.get("kind", "soft")

        if kind == "head":
            return head_token_pattern_pg(
                tok,
                L=head_L,
                slack=head_slack,
                prefer="mix",
                strong_common=head_strong_common,
                strong_suf_len=head_strong_suf_len,
            )

        if kind == "road_id":
            return road_id_pattern(tok)  # road id 不包 \y（包含 '-'）
        if kind == "road_type":
            return road_type_pattern(tok)
        if kind == "exit":
            return r"\yExit\y"
        if kind == "exit_num":
            return re.escape(tok)

        if kind == "df_anchor":
            anch = info["anchor"]
            mode = info["mode"]
            return token_pattern_from_anchor(
                tok, anch, mode,
                use_vehicle_synonyms=use_vehicle_synonyms,
                rest_min_floor=rest_min_floor,
                rest_slack=rest_slack,
                rest_max_cap=rest_max_cap,
            )

        if keep_numeric_literal(tok):
            return re.escape(tok)

        if is_punct(tok):
            return r"\." if tok == "." else re.escape(tok)

        return token_pattern_soft(tok)

    prev = kept_pos[0]
    parts.append(render(prev))

    for cur in kept_pos[1:]:
        dist = cur - prev - 1

        prev_kind = meta.get(prev, {}).get("kind", "")
        if prev_kind == "head":
            parts.append(gap_pattern_char(dist, head_gap_base, head_gap_per_token, head_gap_cap))
        else:
            parts.append(gap_pattern_char(dist, gap_base, gap_per_token, gap_cap))

        parts.append(render(cur))
        prev = cur

    regex = "".join(parts)

    stats = {
        "kept_positions": kept_pos,
        "n_kept": len(kept_pos),
        "df_anchors": [a for _, a, _ in df_anchors] if df_anchors else [],
        **flags,
        "has_road_or_exit": bool(flags.get("has_road_id") or flags.get("has_exit_num")),
    }
    return regex, stats


# =====================================================================================
# Fast seed verification (no PG): approximate PG regex in Python
# =====================================================================================
def approx_verify_seed_py(seed: str, pg_regex: str) -> bool:
    """
    不走 PG 的轻量近似验证：
      - \y/\m/\M -> \b
      - (?:.|\n) 直接可用
    注意：这只是近似（PG 的 \y 与 Python 的 \b 略有差异），但足以过滤明显不匹配样本。
    """
    try:
        rx = pg_regex
        rx = rx.replace(r"\y", r"\b")
        rx = rx.replace(r"\m", r"\b")
        rx = rx.replace(r"\M", r"\b")
        return re.search(rx, seed, flags=re.DOTALL) is not None
    except Exception:
        return False


# =====================================================================================
# DB helpers (fast sampling with per-worker disjoint id ranges)
# =====================================================================================
def _get_id_bounds(conn, table: str) -> Optional[Tuple[int, int]]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT min(id), max(id) FROM {table};")
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


# =====================================================================================
# ID Samplers (reduce duplicate seeds)
# =====================================================================================
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
    """
    Pseudo-random permutation over [lo, hi] using modular stride:
      x_{t+1} = (x_t + stride) mod span
    If gcd(stride, span) == 1, it cycles through all residues before repeating.
    This dramatically reduces repeated sampling compared to randint().
    """

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
    min_tokens: int,
    id_lo: int,
    id_hi: int,
    sampler: Optional[IdSampler] = None,
    require_road_or_exit: bool = False,
    prefilter_road_or_exit: bool = False,
) -> Tuple[int, str, List[str]]:
    """
    Fetch a seed row (id, description), tokenized, with optional prefilter:
      - if prefilter_road_or_exit and require_road_or_exit: require seed itself to contain road_id OR Exit+num
    Sampling:
      - prefer exact id lookup, then "next id" in range, then wrap.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        while True:
            row = None
            if id_lo <= id_hi:
                rid = sampler.next_id() if sampler is not None else random.randint(id_lo, id_hi)

                # 1) exact id lookup
                cur.execute(f"SELECT id, description FROM {table} WHERE id = %s;", (rid,))
                row = cur.fetchone()

                if not row:
                    # 2) next existing id in range
                    cur.execute(
                        f"SELECT id, description FROM {table} "
                        f"WHERE id > %s AND id <= %s ORDER BY id LIMIT 1;",
                        (rid, id_hi),
                    )
                    row = cur.fetchone()

                    if not row:
                        # 3) wrap
                        cur.execute(
                            f"SELECT id, description FROM {table} "
                            f"WHERE id >= %s AND id <= %s ORDER BY id LIMIT 1;",
                            (id_lo, id_hi),
                        )
                        row = cur.fetchone()

            if not row:
                # fallback
                cur.execute(f"SELECT id, description FROM {table} WHERE id IS NOT NULL ORDER BY id LIMIT 1;")
                row = cur.fetchone()
                if not row:
                    raise RuntimeError("No rows available in table.")

            s = row["description"]
            toks = tokenize(s)
            if len(toks) >= min_tokens:
                if prefilter_road_or_exit and require_road_or_exit:
                    if not seed_has_road_or_exit_tokens(toks):
                        continue
                return int(row["id"]), s, toks


# =====================================================================================
# Worker
# =====================================================================================
def worker_generate(
    worker_id: int,
    num_target: int,
    pools_json: str,
    pg_dsn: str,
    table: str,
    col: str,
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
    seed_prefilter_road_or_exit: bool,
    # generation knobs
    anchors_k: int,
    keep_road_id: bool,
    keep_road_type: bool,
    keep_exit_num: bool,
    extra_mid: int,
    # gaps
    gap_base: int,
    gap_per_token: int,
    gap_cap: int,
    head_gap_base: int,
    head_gap_per_token: int,
    head_gap_cap: int,
    prefix_any_g: int,
    case_insensitive: bool,
    # anchor rendering
    use_vehicle_synonyms: bool,
    rest_min_floor: int,
    rest_slack: int,
    rest_max_cap: int,
    # head rendering
    head_L: int,
    head_slack: int,
    head_strong_common: bool,
    head_strong_suf_len: int,
    # filters
    trgm_min: int,
    trgm_max: int,
    max_lit_run: int,
    # requirement (no PG)
    require_road_or_exit: bool,
    # seed verify mode
    seed_verify: str,
) -> Dict[str, Any]:
    random.seed((int(rng_seed) * 1000003) ^ (worker_id * 9176) ^ (os.getpid() << 8))

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
    per_seed_cnt: Dict[int, int] = {}

    with open(tmp_path, "w", encoding="utf-8") as w:
        attempts = 0
        while ok < num_target and attempts < max_attempts:
            attempts += 1

            try:
                seed_id, seed, toks = fetch_random_seed(
                    conn,
                    table=table,
                    min_tokens=seed_min_tokens,
                    id_lo=id_lo,
                    id_hi=id_hi,
                    sampler=sampler,
                    require_road_or_exit=require_road_or_exit,
                    prefilter_road_or_exit=seed_prefilter_road_or_exit,
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

            regex, gen_stats = assemble_regex_human(
                tokens=toks,
                pools=pools,
                anchors_k=anchors_k,
                keep_road_id=keep_road_id,
                keep_road_type=keep_road_type,
                keep_exit_num=keep_exit_num,
                extra_mid=extra_mid,
                gap_base=gap_base,
                gap_per_token=gap_per_token,
                gap_cap=gap_cap,
                head_gap_base=head_gap_base,
                head_gap_per_token=head_gap_per_token,
                head_gap_cap=head_gap_cap,
                use_vehicle_synonyms=use_vehicle_synonyms,
                rest_min_floor=rest_min_floor,
                rest_slack=rest_slack,
                rest_max_cap=rest_max_cap,
                head_L=head_L,
                head_slack=head_slack,
                head_strong_common=head_strong_common,
                head_strong_suf_len=head_strong_suf_len,
                prefix_any_g=prefix_any_g,
                case_insensitive=case_insensitive,
            )
            if not regex:
                reasons[gen_stats.get("reason", "gen_fail")] += 1
                continue

            # No-PG selectivity safeguard (still keep this; prefilter only reduces wasted sampling)
            if require_road_or_exit and (not gen_stats.get("has_road_or_exit", False)):
                reasons["no_road_or_exit"] += 1
                continue

            ml = max_literal_run_len(regex)
            if ml > max_lit_run:
                reasons["max_lit_run_too_big"] += 1
                continue

            tb = trgm_budget(regex)
            if tb < trgm_min:
                reasons["trgm_budget_too_low"] += 1
                continue
            if tb > trgm_max:
                reasons["trgm_budget_too_high"] += 1
                continue

            # seed verify
            if seed_verify == "py":
                if not approx_verify_seed_py(seed, regex):
                    reasons["seed_verify_fail_py"] += 1
                    continue
            elif seed_verify == "off":
                pass
            elif seed_verify == "pg":
                # 你明确说暂时不走 PG；这里保留开关以备后续需要
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

            rec = {
                "dataset": "us_acc",
                "seed_id": int(seed_id),
                "seed_string": seed,
                "regex": regex,
                "bucket": "hard",
                "stats": {
                    "max_literal_run_len": int(ml),
                    "trgm_budget": int(tb),
                    "seed_len_tokens": int(len(toks)),
                    "n_kept": int(gen_stats.get("n_kept", 0)),
                    "has_road_id": bool(gen_stats.get("has_road_id", False)),
                    "has_exit_num": bool(gen_stats.get("has_exit_num", False)),
                    "has_road_type": bool(gen_stats.get("has_road_type", False)),
                },
                "kept_positions": gen_stats.get("kept_positions", []),
                "anchors": gen_stats.get("df_anchors", []),
                "anchor_len": int(pools.anchor_len),
                "sql_E": make_where_regex_sql(select_expr="id", table=table, col=col, regex=regex, limit=None),
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
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default="/home/lyz/wd/LLM_RegularSearch/v1_data/gen_workload/data/old_data/US_Accidents_7m.csv")
    ap.add_argument("--mode", choices=["prep", "gen"], required=True)
    ap.add_argument("--pools", default="us_acc_pools_L3.json")
    ap.add_argument("--out", default="workload/us_acc_hard_10k.jsonl")
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tmp_dir", default="", help="Default: <out_dir>/tmp_workers")

    # Error logs
    ap.add_argument("--err_log_dir", default="./log/gen_workload/v6.log",
                    help="If looks like a file path, create per-worker files as <path>.worker_<id>.jsonl; "
                         "otherwise treat as directory and write <dir>/worker_<id>.jsonl.")

    # pool / DF knobs
    ap.add_argument("--anchor_len", type=int, default=3)
    ap.add_argument("--df_low_pct", type=float, default=0.005)
    ap.add_argument("--df_high_pct", type=float, default=0.15)
    ap.add_argument("--max_rows_df", type=int, default=0)

    # DB
    ap.add_argument("--pg_dsn", default="dbname=regex_db user=lyz host=localhost port=5000")
    ap.add_argument("--table", default="us_accidents")
    ap.add_argument("--col", default="description")

    # sampling / anti-dup
    ap.add_argument("--split_id_ranges", dest="split_id_ranges", action="store_true", default=True)
    ap.add_argument("--no_split_id_ranges", dest="split_id_ranges", action="store_false")
    ap.add_argument("--dedup_in_worker", dest="dedup_in_worker", action="store_true", default=True)
    ap.add_argument("--no_dedup_in_worker", dest="dedup_in_worker", action="store_false")
    ap.add_argument("--max_per_seed", type=int, default=4)
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--dedup", action="store_true", help="Dedup on merge by (seed_id, regex) hash key.")
    ap.add_argument("--seed_sampler", choices=["stride", "random"], default="stride",
                    help="How to sample ids inside each worker range. stride reduces duplicates dramatically.")
    ap.add_argument("--seed_prefilter_road_or_exit", action="store_true", default=True,
                    help="If require_road_or_exit, prefilter seeds at fetch time to avoid wasted attempts.")
    ap.add_argument("--no_seed_prefilter_road_or_exit", dest="seed_prefilter_road_or_exit", action="store_false")

    # safety
    ap.add_argument("--max_attempts_per_worker", type=int, default=2_000_000)
    ap.add_argument("--seed_min_tokens", type=int, default=12)

    # generation knobs
    ap.add_argument("--anchors", type=int, default=2, help="How many DF anchors to include (0/1/2). Recommend 2.")
    ap.add_argument("--keep_road_id", action="store_true", default=True)
    ap.add_argument("--no_keep_road_id", dest="keep_road_id", action="store_false")
    ap.add_argument("--keep_road_type", action="store_true", default=True)
    ap.add_argument("--no_keep_road_type", dest="keep_road_type", action="store_false")
    ap.add_argument("--keep_exit_num", action="store_true", default=True)
    ap.add_argument("--no_keep_exit_num", dest="keep_exit_num", action="store_false")
    ap.add_argument("--extra_mid", type=int, default=0, help="Extra mid soft tokens to keep (0..2).")

    # gaps (global)
    ap.add_argument("--gap_base", type=int, default=50)
    ap.add_argument("--gap_per_token", type=int, default=10)
    ap.add_argument("--gap_cap", type=int, default=120)

    # gaps (head->next 更紧)
    ap.add_argument("--head_gap_base", type=int, default=28)
    ap.add_argument("--head_gap_per_token", type=int, default=6)
    ap.add_argument("--head_gap_cap", type=int, default=80)

    ap.add_argument("--prefix_any_g", type=int, default=0, help="Optional leading anychar gap {0,g}?")
    ap.add_argument("--case_insensitive", action="store_true", default=False)

    # anchor rendering (bounded)
    ap.add_argument("--use_vehicle_synonyms", action="store_true", default=True,
                    help="If DF anchor is 'veh', render as \\y(?:vehicle|veh)[A-Za-z0-9_]*\\y")
    ap.add_argument("--no_use_vehicle_synonyms", dest="use_vehicle_synonyms", action="store_false")
    ap.add_argument("--rest_min_floor", type=int, default=0, help="Minimum rest length floor for DF anchor tokens.")
    ap.add_argument("--rest_slack", type=int, default=1, help="Length slack around token rest (lo/hi).")
    ap.add_argument("--rest_max_cap", type=int, default=10, help="Max cap for DF anchor rest length.")

    # head rendering
    ap.add_argument("--head_L", type=int, default=3)
    ap.add_argument("--head_slack", type=int, default=1)
    ap.add_argument("--head_strong_common", action="store_true", default=True,
                    help="Use prefix+suffix constraint for common head roots (Acc/Inc/etc) to reduce explosion.")
    ap.add_argument("--no_head_strong_common", dest="head_strong_common", action="store_false")
    ap.add_argument("--head_strong_suf_len", type=int, default=2)

    # filters
    ap.add_argument("--trgm_min", type=int, default=6)
    ap.add_argument("--trgm_max", type=int, default=14)
    ap.add_argument("--max_lit_run", type=int, default=12)

    # requirement (no PG)
    ap.add_argument("--require_road_or_exit", action="store_true", default=True,
                    help="Require each regex to include road_id OR Exit number (recommended).")
    ap.add_argument("--no_require_road_or_exit", dest="require_road_or_exit", action="store_false")

    # seed verify mode (no PG by default)
    ap.add_argument("--seed_verify", choices=["py", "off", "pg"], default="py",
                    help="Seed verification mode: py (default, no PG), off, pg (slow).")

    args = ap.parse_args()

    if args.mode == "prep":
        build_pools(
            csv_path=args.csv,
            out_json=args.pools,
            anchor_len=args.anchor_len,
            df_low_pct=args.df_low_pct,
            df_high_pct=args.df_high_pct,
            max_rows=(args.max_rows_df or None),
        )
        return

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_dir = args.tmp_dir or os.path.join(out_dir if out_dir else ".", "tmp_workers")
    os.makedirs(tmp_dir, exist_ok=True)

    # Resolve global id bounds
    conn0 = psycopg2.connect(args.pg_dsn)
    conn0.autocommit = True
    bounds = _get_id_bounds(conn0, args.table)
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
                args.table,
                args.col,
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
                args.seed_prefilter_road_or_exit,
                # knobs
                args.anchors,
                args.keep_road_id,
                args.keep_road_type,
                args.keep_exit_num,
                args.extra_mid,
                # gaps
                args.gap_base,
                args.gap_per_token,
                args.gap_cap,
                args.head_gap_base,
                args.head_gap_per_token,
                args.head_gap_cap,
                args.prefix_any_g,
                args.case_insensitive,
                # anchor rendering
                args.use_vehicle_synonyms,
                args.rest_min_floor,
                args.rest_slack,
                args.rest_max_cap,
                # head rendering
                args.head_L,
                args.head_slack,
                args.head_strong_common,
                args.head_strong_suf_len,
                # filters
                args.trgm_min,
                args.trgm_max,
                args.max_lit_run,
                # requirement + verify
                args.require_road_or_exit,
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

    print("[INFO] No PG full-table validation by default.")
    print("[INFO] PG regex compatibility: avoided \\S and \\b; use (?:.|\\n) gaps and \\y word boundaries.")
    print("[INFO] Head token anchoring is ON (head_tokens=1), without adding ^.")
    print("[INFO] DF-anchor expansion is bounded {lo,hi} to reduce match explosion.")
    print("[INFO] seed_verify default is 'py' (no PG).")
    print(f"[INFO] seed_sampler={args.seed_sampler}, seed_prefilter_road_or_exit={args.seed_prefilter_road_or_exit}, "
          f"max_per_seed={args.max_per_seed}")


if __name__ == "__main__":
    main()
