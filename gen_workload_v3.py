#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import errors as pg_errors
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------
# Tokenization
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[^\w\s]")

def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(s)

def is_punct(tok: str) -> bool:
    return bool(re.fullmatch(r"[^\w\s]", tok))

# -----------------------------
# Structured token detection (high-signal tokens)
# -----------------------------
# Highway token like I-405, US-101, CA-138, NM-317, etc.
HWY_RE = re.compile(
    r"^(?:I|US|SR|CA|TX|NM|AZ|UT|NV|CO|FL|GA|VA|PA|OH|WI|IL|IN|KY|LA|NC|SC|AL|MO|IA|CR)-[0-9A-Za-z]+$"
)

DIR_SET = {"northbound", "southbound", "eastbound", "westbound", "nb", "sb", "eb", "wb"}
EXIT_KW = {"exit", "exits", "ramp"}  # optional

def structured_type(tok: str) -> Optional[str]:
    tl = tok.lower()
    if HWY_RE.fullmatch(tok):
        return "hwy"
    if tl in DIR_SET:
        return "dir"
    if tl in EXIT_KW:
        return "exitkw"
    return None

# -----------------------------
# Regex building blocks
# -----------------------------
def _q(n: int, slack: int) -> str:
    if slack <= 0:
        return f"{{{n},{n}}}"
    lo = max(1, n - slack)
    hi = n + slack
    return f"{{{lo},{hi}}}"

def shape_block(tok: str, slack: int = 0) -> str:
    """
    Keep your original shaping logic. This is still used for most tokens.
    The important-info tokens will be handled separately (safe literal encoding).
    """
    if tok == "":
        return ""

    if tok.isdigit():
        return rf"[0-9]{_q(len(tok), slack)}"

    if re.fullmatch(r"[0-9]+(st|nd|rd|th)", tok):
        m = re.match(r"([0-9]+)", tok)
        n = m.group(1) if m else ""
        return rf"[0-9]{_q(len(n), slack)}(st|nd|rd|th)"

    if re.fullmatch(r"[A-Z]+", tok):
        return rf"[A-Z]{_q(len(tok), slack)}"

    if re.fullmatch(r"[a-z]+", tok):
        return rf"[a-z]{_q(len(tok), slack)}"

    if re.fullmatch(r"[A-Za-z]+", tok):
        if tok[0].isupper() and tok[1:].islower():
            if len(tok) == 1:
                return r"[A-Z]"
            return rf"[A-Z][a-z]{_q(len(tok)-1, slack)}"
        return rf"[A-Za-z]{_q(len(tok), slack)}"

    if "-" in tok and re.fullmatch(r"[A-Za-z0-9-]+", tok):
        parts = tok.split("-")
        return "-".join(shape_block(p, slack=slack) for p in parts)

    return rf"[A-Za-z0-9]{_q(len(tok), slack)}"

def max_literal_run_len(regex: str) -> int:
    """
    Count maximal consecutive [A-Za-z0-9] outside of char classes.
    With safe literal encoding ([Aa][Bb]...), this stays very small.
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

# -----------------------------
# Safe literal encoding (information-preserving AND max_literal_run-safe)
# -----------------------------
_META = set(r"\.^$|?*+()[]{}")

def literal_safe_ci(tok: str) -> str:
    """
    Encode token as case-insensitive per-char class to keep info
    while keeping max_literal_run_len very small.

    Examples:
      "Dixie" -> "[Dd][Ii][Xx][Ii][Ee]"
      "I-405" -> "[Ii]\-[4][0][5]"
      "11A-11B" -> "[1][1][Aa]\-[1][1][Bb]"
    """
    out = []
    for ch in tok:
        if ch.isalpha():
            out.append(f"[{ch.lower()}{ch.upper()}]")
        elif ch.isdigit():
            out.append(f"[{ch}]")
        elif ch in _META:
            out.append("\\" + ch)
        else:
            out.append(re.escape(ch))
    return "".join(out)

def a2_ci_pat(a2: str) -> str:
    """
    2-char anchor pattern, case-insensitive for letters.
    "So" -> "[Ss][Oo]", "11" -> "[1][1]"
    """
    if len(a2) != 2:
        return re.escape(a2)
    return literal_safe_ci(a2)

def anchor_token_pattern(tok: str, anchor2: str, mode: str, slack: int = 0) -> str:
    """
    Keep your anchor-on-2char idea, but make the 2-char anchor itself case-insensitive.
    """
    a = a2_ci_pat(anchor2)
    if len(tok) == 2 and tok.lower() == anchor2.lower():
        return a

    if mode == "pre":
        rest = tok[2:]
        rest_pat = shape_block(rest, slack=slack) if rest else ""
        return a + (rest_pat if rest_pat else "")
    else:
        rest = tok[:-2]
        rest_pat = shape_block(rest, slack=slack) if rest else ""
        return (rest_pat if rest_pat else "") + a

# -----------------------------
# Pools
# -----------------------------
@dataclass
class Pools:
    # lower-case 2ch anchor set
    mid_2ch: set
    # token df map (lower-case) for "mid-frequency" words used as islands candidates
    df_keep: Dict[str, int]
    N: int
    df_low: int
    df_high: int

def build_pools(
    csv_path: str,
    out_json: str,
    df_low_pct=0.005,
    df_high_pct=0.15,
    # for islands: keep tokens in a narrower mid-frequency band by default
    island_low_pct=0.0005,
    island_high_pct=0.05,
    max_rows: Optional[int] = None,
):
    """
    Prep pass over CSV to compute df and build:
      1) mid_2ch anchors
      2) df_keep (token->count) for mid-frequency islands selection
    """
    df = defaultdict(int)
    N = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in tqdm(r, desc="DF pass"):
            s = row["Description"]
            toks = tokenize(s)
            seen = set(t.lower() for t in toks if (not is_punct(t)) and t.strip())
            for t in seen:
                df[t] += 1
            N += 1
            if max_rows and N >= max_rows:
                break

    low = int(df_low_pct * N)
    high = int(df_high_pct * N)

    island_low = int(island_low_pct * N)
    island_high = int(island_high_pct * N)

    mid_2ch = set()
    df_keep = {}

    for t, c in df.items():
        # df_keep for islands: avoid too-common and too-rare, and avoid very short tokens
        if island_low <= c <= island_high and len(t) >= 3:
            df_keep[t] = int(c)

        # 2ch anchor pool: similar to your original, but based on lower tokens
        if c < low or c > high:
            continue
        if len(t) >= 2:
            for a in (t[:2], t[-2:]):
                if any(ch.isspace() for ch in a):
                    continue
                if re.fullmatch(r"[^\w\s]{2}", a):
                    continue
                mid_2ch.add(a)

    obj = {
        "N": N,
        "df_low": low,
        "df_high": high,
        "mid_2ch_size": len(mid_2ch),
        "mid_2ch": sorted(mid_2ch),
        "island_low": island_low,
        "island_high": island_high,
        "df_keep_size": len(df_keep),
        # store as list of pairs to keep json stable and avoid huge dict overhead in some parsers
        "df_keep": [[k, v] for k, v in df_keep.items()],
    }
    with open(out_json, "w", encoding="utf-8") as w:
        json.dump(obj, w, ensure_ascii=False)
    print(f"[OK] pools saved: {out_json}")
    print(f"     N={N} mid_2ch={len(mid_2ch)} df_keep={len(df_keep)}")
    print(f"     df_low/high(anchor)={low}/{high}  island_low/high={island_low}/{island_high}")

def load_pools(pools_json: str) -> Pools:
    with open(pools_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    df_keep = {}
    for k, v in obj.get("df_keep", []):
        df_keep[k] = int(v)
    return Pools(
        mid_2ch=set(obj["mid_2ch"]),
        df_keep=df_keep,
        N=int(obj["N"]),
        df_low=int(obj["df_low"]),
        df_high=int(obj["df_high"]),
    )

# -----------------------------
# Anchor / middle-block selection
# -----------------------------
LOW_VALUE_2CH = {
    "nd", "st", "rd", "th",
    "on", "at", "to", "of", "in", "an", "or", "by", "as", "is",
}

def pick_anchors(tokens: List[str], pools: Pools, k=2) -> Optional[List[Tuple[int, str, str]]]:
    """
    Modified:
      - operate on lower-case 2ch pool
      - strong preference: at least one anchor in early 30% tokens (better for prefix learning)
      - avoid using structured tokens as anchors (they will be handled by literal keep/islands)
    """
    cand = []
    T = len(tokens)
    for i, tok in enumerate(tokens):
        if is_punct(tok):
            continue
        if structured_type(tok) is not None:
            continue
        tl = tok.lower()
        if len(tl) >= 2:
            pre = tl[:2]
            suf = tl[-2:]
            if pre in pools.mid_2ch:
                cand.append((i, pre, "pre"))
            if suf in pools.mid_2ch and suf != pre:
                cand.append((i, suf, "suf"))

    if len(cand) < k:
        return None

    def _prefer(lst: List[Tuple[int, str, str]]) -> List[Tuple[int, str, str]]:
        good = [x for x in lst if x[1] not in LOW_VALUE_2CH]
        return good if len(good) >= max(1, min(k, len(lst))) else lst

    early_cut = max(1, int(0.30 * T))
    early = _prefer([c for c in cand if c[0] <= early_cut])
    mid = _prefer([c for c in cand if early_cut < c[0] < int(0.80 * T)])
    late = _prefer([c for c in cand if c[0] >= int(0.80 * T)])
    cand2 = _prefer(cand)

    picked = []

    # force one early if possible
    if early:
        picked.append(random.choice(early))

    # then fill with mid/late
    pools_order = [mid, late, cand2]
    for pool in pools_order:
        while pool and len(picked) < k:
            picked.append(random.choice(pool))
        if len(picked) >= k:
            break

    # de-dup by anchor string first
    uniq_by_anchor: Dict[str, Tuple[int, str, str]] = {}
    extra = random.sample(cand2, min(len(cand2), 12))
    for x in picked + extra:
        uniq_by_anchor.setdefault(x[1], x)
        if len(uniq_by_anchor) >= k:
            break

    if len(uniq_by_anchor) >= k:
        picked = list(uniq_by_anchor.values())[:k]
    else:
        # fallback: de-dup by position
        uniq_by_idx = {}
        for x in picked:
            uniq_by_idx[x[0]] = x
        picked = list(uniq_by_idx.values())
        if len(picked) < k:
            return None
        picked = random.sample(picked, k)

    picked.sort(key=lambda x: x[0])
    return picked

def pick_middle_blocks_segmented(
    tokens: List[str],
    reserved_pos: set,
    min_blocks=2,
) -> Optional[List[int]]:
    """
    Modified:
      - instead of forcing middle 0.3~0.7 only, use segmented coverage:
          early-mid (15%~35%), mid (35%~65%), late-mid (65%~85%)
      - improves that early discriminative tokens can be shaped (or kept), helping prefix learnability.
    """
    T = len(tokens)
    if T <= 0:
        return None

    def valid(i: int) -> bool:
        if i < 0 or i >= T:
            return False
        if i in reserved_pos:
            return False
        if is_punct(tokens[i]):
            return False
        return True

    segs = [
        (int(0.15 * T), int(0.35 * T)),
        (int(0.35 * T), int(0.65 * T)),
        (int(0.65 * T), int(0.85 * T)),
    ]

    picks = []
    # try to pick one from each segment first
    for L, R in segs:
        idx = [i for i in range(L, min(R + 1, T)) if valid(i)]
        if idx:
            picks.append(random.choice(idx))
        if len(picks) >= min_blocks:
            break

    # fill remaining from union of all candidate positions
    if len(picks) < min_blocks:
        all_idx = []
        for L, R in segs:
            all_idx.extend([i for i in range(L, min(R + 1, T)) if valid(i)])
        all_idx = list(set(all_idx))
        if len(all_idx) < (min_blocks - len(picks)):
            return None
        remain = random.sample(all_idx, min_blocks - len(picks))
        picks.extend(remain)

    picks = sorted(set(picks))
    if len(picks) < min_blocks:
        return None
    return picks[:min_blocks]

# -----------------------------
# Islands selection (information-preserving tokens)
# -----------------------------
LOW_VALUE_TOK = {
    "on","at","to","of","in","an","or","by","as","is","and","the","a","due","for","from","before","after",
}

def pick_keep_positions(
    tokens: List[str],
    pools: Pools,
    islands_k: int,
    max_hwy_keep: int,
    max_dir_keep: int,
    max_exit_follow: int,
    force_early: bool = True,
) -> Tuple[set, List[Dict[str, Any]]]:
    """
    Decide which token positions are kept as safe-literal (literal_safe_ci) to preserve key info.

    Strategy:
      1) keep up to max_hwy_keep highway tokens (prefer early and then by length)
      2) keep up to max_dir_keep direction tokens
      3) if "Exit/Exits/Ramp" appears, keep that keyword and up to max_exit_follow following tokens (numbers like 209, 11A, 11A-11B)
      4) pick up to islands_k extra "mid-frequency" tokens from df_keep (length>=3), avoiding stopwords,
         and (optionally) force at least one early token.

    Returns:
      keep_pos: set[int]
      keep_meta: list of {"pos":..., "tok":..., "kind":...}
    """
    T = len(tokens)
    keep_pos = set()
    meta = []

    def add_pos(i: int, kind: str):
        if i in keep_pos:
            return
        keep_pos.add(i)
        meta.append({"pos": int(i), "tok": tokens[i], "kind": kind})

    # 1) Highway tokens
    hwy_pos = [i for i, tok in enumerate(tokens) if structured_type(tok) == "hwy"]
    # prefer early then longer token
    hwy_pos.sort(key=lambda i: (i > int(0.5 * T), i, -len(tokens[i])))
    for i in hwy_pos[:max(0, int(max_hwy_keep))]:
        add_pos(i, "hwy")

    # 2) Direction tokens
    dir_pos = [i for i, tok in enumerate(tokens) if structured_type(tok) == "dir"]
    dir_pos.sort(key=lambda i: (i > int(0.5 * T), i))
    for i in dir_pos[:max(0, int(max_dir_keep))]:
        add_pos(i, "dir")

    # 3) Exit keyword + following tokens
    for i, tok in enumerate(tokens):
        if structured_type(tok) == "exitkw":
            add_pos(i, "exitkw")
            # keep a few following tokens if they look like exit ids
            j = i + 1
            kept = 0
            while j < T and kept < max_exit_follow:
                tj = tokens[j]
                if is_punct(tj):
                    break
                # allow digits / alnum-with-dash like 11A-11B
                if re.fullmatch(r"[0-9A-Za-z]+(?:-[0-9A-Za-z]+)*", tj):
                    add_pos(j, "exitid")
                    kept += 1
                    j += 1
                else:
                    break

    # 4) Mid-frequency islands (df-based)
    # Candidates: not punctuation, not already kept, length>=3, not stopwords
    cand = []
    for i, tok in enumerate(tokens):
        if i in keep_pos:
            continue
        if is_punct(tok):
            continue
        tl = tok.lower()
        if tl in LOW_VALUE_TOK:
            continue
        if len(tl) < 3:
            continue
        dfc = pools.df_keep.get(tl)
        if dfc is None:
            continue
        # smaller df = more selective; also prefer earlier and longer slightly
        cand.append((dfc, i, -len(tok)))

    if cand:
        # force one early if requested
        if force_early:
            early_cut = max(1, int(0.30 * T))
            early_cand = [x for x in cand if x[1] <= early_cut]
            if early_cand:
                early_cand.sort(key=lambda x: (x[0], x[1], x[2]))
                _, i0, _ = early_cand[0]
                add_pos(i0, "island_df")

        # pick remaining by df ascending
        cand.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, i, _ in cand:
            if len([m for m in meta if m["kind"] == "island_df"]) >= islands_k:
                break
            if i in keep_pos:
                continue
            add_pos(i, "island_df")

    return keep_pos, meta

# -----------------------------
# Joining logic: punctuation-aware spacing
# -----------------------------
RIGHT_ATTACH = {".", ",", ":", ";", "!", "?", ")", "]", "}"}
LEFT_ATTACH = {"(", "[", "{"}
GLUE = {"#", "'", "/", "-", "&"}

def _sep(prev_tok: str, cur_tok: str) -> str:
    if cur_tok in RIGHT_ATTACH:
        return r"\s*"
    if prev_tok in LEFT_ATTACH:
        return r"\s*"
    if prev_tok in GLUE or cur_tok in GLUE:
        return r"\s*"
    return r"\s+"

def join_token_patterns(tokens: List[str], pats: List[str]) -> str:
    if not tokens:
        return ""
    out = [pats[0]]
    for i in range(1, len(tokens)):
        out.append(_sep(tokens[i - 1], tokens[i]))
        out.append(pats[i])
    return "".join(out)

# -----------------------------
# Assemble regex
# -----------------------------
def assemble_regex(
    tokens: List[str],
    anchors: List[Tuple[int, str, str]],
    middle_blocks: List[int],
    keep_pos: set,
    replace_ratio=0.6,
    g=0,
    len_slack: int = 0,
) -> str:
    """
    Modified:
      - positions in keep_pos are encoded with literal_safe_ci(tok) (info-preserving)
      - anchors are still applied elsewhere
      - middle_blocks shaped
      - other tokens replaced by shape with probability replace_ratio, else 2ch anchor fallback
    """
    anchor_pos = {i for i, _, _ in anchors}
    mid_pos = set(middle_blocks)

    pats: List[str] = []
    for i, tok in enumerate(tokens):
        if is_punct(tok):
            if tok == ".":
                pats.append(r"\.")
            else:
                pats.append(re.escape(tok))
            continue

        # 1) keep important-info tokens
        if i in keep_pos:
            pats.append(literal_safe_ci(tok))
            continue

        # 2) anchor tokens
        if i in anchor_pos:
            _, a2, mode = next(x for x in anchors if x[0] == i)
            # a2 stored in lower-case; use original token but anchor2 is lower
            pats.append(anchor_token_pattern(tok, a2, mode, slack=len_slack))
        elif i in mid_pos:
            pats.append(shape_block(tok, slack=len_slack))
        else:
            if random.random() < replace_ratio:
                pats.append(shape_block(tok, slack=len_slack))
            else:
                # fallback: weak keep of first 2 chars
                if len(tok) >= 2:
                    a2 = tok[:2]
                    pats.append(anchor_token_pattern(tok, a2, "pre", slack=len_slack))
                else:
                    pats.append(re.escape(tok))

    body = join_token_patterns(tokens, pats)
    if g > 0:
        return rf".{{0,{g}}}" + body
    return body

# -----------------------------
# DB helpers
# -----------------------------
def _get_id_bounds(conn) -> Optional[Tuple[int, int]]:
    with conn.cursor() as cur:
        cur.execute("SELECT min(id), max(id) FROM us_accidents;")
        mn, mx = cur.fetchone()
    if mn is None or mx is None:
        return None
    return int(mn), int(mx)

def fetch_random_seed(
    conn,
    min_tokens=6,
    id_bounds: Optional[Tuple[int, int]] = None,
) -> Tuple[int, str, List[str]]:
    if id_bounds is None:
        id_bounds = _get_id_bounds(conn)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        while True:
            row = None
            try:
                cur.execute("SELECT id, description FROM us_accidents TABLESAMPLE SYSTEM (0.01) LIMIT 1;")
                row = cur.fetchone()
            except Exception:
                row = None

            if not row and id_bounds is not None:
                mn, mx = id_bounds
                rid = random.randint(mn, mx)
                cur.execute(
                    "SELECT id, description FROM us_accidents WHERE id >= %s ORDER BY id LIMIT 1;",
                    (rid,),
                )
                row = cur.fetchone()
                if not row:
                    cur.execute(
                        "SELECT id, description FROM us_accidents WHERE id < %s ORDER BY id DESC LIMIT 1;",
                        (rid,),
                    )
                    row = cur.fetchone()

            if not row:
                cur.execute("SELECT id, description FROM us_accidents ORDER BY random() LIMIT 1;")
                row = cur.fetchone()

            s = row["description"]
            toks = tokenize(s)
            if len(toks) >= min_tokens:
                return int(row["id"]), s, toks

def verify_seed_pg(conn, seed: str, regex: str) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT %s::text ~ %s::text;", (seed, regex))
            return bool(cur.fetchone()[0])
    except (pg_errors.InvalidRegularExpression, pg_errors.SyntaxError, pg_errors.DataException):
        return False

def hits_probe_pg_multi_terms(
    conn,
    terms: List[str],
    regex: str,
    cand_cap: int,
    probe_pct: float,
    probe_rounds: int,
) -> Tuple[int, int]:
    """
    Modified:
      - allow using 2 terms from keep/islands (more selective than 2-char anchors),
        falling back to anchors when needed.
      - still returns (cand_est, hits_est)
    """
    pct = max(0.0001, min(100.0, float(probe_pct)))
    rounds = max(1, int(probe_rounds))

    # pick up to 2 non-trivial terms
    clean = []
    for t in terms:
        if not t:
            continue
        tl = t.lower()
        if tl in LOW_VALUE_TOK:
            continue
        if len(tl) < 2:
            continue
        clean.append(t)
    clean = clean[:2]
    if len(clean) < 2:
        # not enough terms; degrade to 1-term probe
        clean = clean[:1]

    if not clean:
        return 0, 0

    likes = [f"%{t}%" for t in clean]
    # Compose SQL with 1 or 2 ILIKE filters
    if len(likes) == 1:
        sql = f"""
        WITH samp AS (
          SELECT description
          FROM us_accidents TABLESAMPLE SYSTEM ({pct})
        ),
        cand AS (
          SELECT description
          FROM samp
          WHERE description ILIKE %s
          LIMIT %s
        )
        SELECT
          (SELECT count(*) FROM cand) AS cand_est,
          EXISTS(SELECT 1 FROM cand WHERE description ~ %s) AS has_hit;
        """
        args = (likes[0], cand_cap, regex)
    else:
        sql = f"""
        WITH samp AS (
          SELECT description
          FROM us_accidents TABLESAMPLE SYSTEM ({pct})
        ),
        cand AS (
          SELECT description
          FROM samp
          WHERE description ILIKE %s
            AND description ILIKE %s
          LIMIT %s
        )
        SELECT
          (SELECT count(*) FROM cand) AS cand_est,
          EXISTS(SELECT 1 FROM cand WHERE description ~ %s) AS has_hit;
        """
        args = (likes[0], likes[1], cand_cap, regex)

    last_cand = 0
    try:
        with conn.cursor() as cur:
            for _ in range(rounds):
                cur.execute(sql, args)
                cand_est, has_hit = cur.fetchone()
                last_cand = int(cand_est)
                if has_hit:
                    return last_cand, 1
        return last_cand, 0
    except (pg_errors.InvalidRegularExpression, pg_errors.SyntaxError, pg_errors.DataException):
        return 0, 0

# -----------------------------
# Worker: generate a chunk
# -----------------------------
def worker_generate(
    worker_id: int,
    num_target: int,
    pools_json: str,
    pg_dsn: str,
    anchors_k: int,
    min_mid_blocks: int,
    replace_ratio: float,
    g: int,
    len_slack: int,
    cand_cap: int,
    probe_pct: float,
    probe_rounds: int,
    min_hits_est: int,
    seed_min_tokens: int,
    max_attempts: int,
    out_tmp_dir: str,
    # new knobs
    islands_k: int,
    max_hwy_keep: int,
    max_dir_keep: int,
    max_exit_follow: int,
    skip_probe: bool,
) -> Dict[str, Any]:
    """
    Generate num_target records and write to a worker tmp file.
    Returns dict: {"tmp_path": ..., "n_ok": ..., "reasons": {...}}
    """
    random.seed((os.getpid() << 16) ^ int.from_bytes(os.urandom(2), "little") ^ worker_id)

    pools = load_pools(pools_json)
    conn = psycopg2.connect(pg_dsn)
    id_bounds = _get_id_bounds(conn)

    reasons = defaultdict(int)
    ok = 0

    os.makedirs(out_tmp_dir, exist_ok=True)
    tmp_path = os.path.join(out_tmp_dir, f"tmp_worker_{worker_id}.jsonl")

    with open(tmp_path, "w", encoding="utf-8") as w:
        attempts = 0
        while ok < num_target and attempts < max_attempts:
            attempts += 1

            seed_id, seed, toks = fetch_random_seed(conn, min_tokens=seed_min_tokens, id_bounds=id_bounds)

            anchors = pick_anchors(toks, pools, k=anchors_k)
            if not anchors:
                reasons["no_anchor"] += 1
                continue

            anchor_pos = {i for i, _, _ in anchors}

            # pick keep positions (structured + df-islands)
            keep_pos, keep_meta = pick_keep_positions(
                toks,
                pools=pools,
                islands_k=islands_k,
                max_hwy_keep=max_hwy_keep,
                max_dir_keep=max_dir_keep,
                max_exit_follow=max_exit_follow,
                force_early=True,
            )

            reserved = set(anchor_pos) | set(keep_pos)
            mid_blocks = pick_middle_blocks_segmented(toks, reserved, min_blocks=min_mid_blocks)
            if not mid_blocks:
                reasons["no_mid_blocks"] += 1
                continue

            regex = assemble_regex(
                toks, anchors, mid_blocks,
                keep_pos=keep_pos,
                replace_ratio=replace_ratio,
                g=g,
                len_slack=len_slack,
            )

            ml = max_literal_run_len(regex)
            if ml > 2:
                reasons["ml_gt_2"] += 1
                continue

            if not verify_seed_pg(conn, seed, regex):
                reasons["seed_verify_fail"] += 1
                continue

            a_list = [a for _, a, _ in anchors]

            # probe terms priority:
            #   - use kept tokens first (original surface form), then anchors 2ch strings
            # This makes probe more aligned with selectivity and reduces "hits_full too high" cases.
            if skip_probe:
                cand_est, hits_est = (0, 1)
            else:
                keep_terms = []
                # prefer highway/dir/island_df terms
                # note: keep_meta already has surface token; add in that order
                for m in keep_meta:
                    t = m["tok"]
                    if is_punct(t):
                        continue
                    keep_terms.append(t)

                # fallback anchors (2ch)
                # anchors stored in lower; use them as terms for ILIKE
                probe_terms = keep_terms + a_list

                cand_est, hits_est = hits_probe_pg_multi_terms(
                    conn,
                    terms=probe_terms,
                    regex=regex,
                    cand_cap=cand_cap,
                    probe_pct=probe_pct,
                    probe_rounds=probe_rounds,
                )

                if hits_est < min_hits_est:
                    reasons["hits_est_too_low"] += 1
                    continue

            rec = {
                "dataset": "us_acc",
                "seed_id": int(seed_id),
                "seed_string": seed,
                "regex": regex,
                "anchors": a_list,
                "g": int(g),
                "middle_blocks": sorted(mid_blocks),
                "keep_meta": keep_meta,   # <--- NEW: what we preserved
                "stats": {
                    "max_literal_run_len": int(ml),
                    "cand_est": (None if skip_probe else int(cand_est)),
                    "hits_est": int(hits_est),
                },
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1

    conn.close()
    return {"tmp_path": tmp_path, "n_ok": ok, "reasons": dict(reasons)}

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/lyz/wd/LLM_RegularSearch/v1_data/gen_workload/data/old_data/US_Accidents.csv")
    ap.add_argument("--mode", choices=["prep", "gen"], required=True)
    ap.add_argument("--pools", default="us_acc_pools_v2.json")
    ap.add_argument("--out", default="us_acc_stage1_v2.jsonl")
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers (processes).")
    ap.add_argument("--tmp_dir", default="", help="Directory for worker tmp jsonl files. Default: <out_dir>/tmp_workers")

    ap.add_argument("--anchors", type=int, default=3)
    ap.add_argument("--min_mid_blocks", type=int, default=3)
    ap.add_argument("--replace_ratio", type=float, default=0.55)
    ap.add_argument("--g", type=int, default=0)
    ap.add_argument("--cand_cap", type=int, default=3000)
    ap.add_argument("--max_rows_df", type=int, default=0)
    ap.add_argument("--pg_dsn", default="dbname=regex_db user=lyz host=localhost port=5000")
    ap.add_argument("--len_slack", type=int, default=0)

    ap.add_argument("--min_hits_est", type=int, default=1)
    ap.add_argument("--probe_pct", type=float, default=0.03)
    ap.add_argument("--probe_rounds", type=int, default=2)

    ap.add_argument("--seed_min_tokens", type=int, default=8)
    ap.add_argument("--max_attempts_per_worker", type=int, default=2_000_000,
                    help="Hard cap to avoid infinite loops if acceptance rate is too low.")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate by (seed_id, regex) before writing final output.")

    # prep knobs
    ap.add_argument("--df_low_pct", type=float, default=0.005)
    ap.add_argument("--df_high_pct", type=float, default=0.15)
    ap.add_argument("--island_low_pct", type=float, default=0.0005)
    ap.add_argument("--island_high_pct", type=float, default=0.05)

    # NEW: keep/islands knobs
    ap.add_argument("--islands_k", type=int, default=2, help="How many df-islands to keep as safe literal (in addition to structured keep).")
    ap.add_argument("--max_hwy_keep", type=int, default=2, help="Max highway tokens (I-405/US-101/...) to keep as safe literal.")
    ap.add_argument("--max_dir_keep", type=int, default=1, help="Max direction tokens (Northbound/Westbound/...) to keep.")
    ap.add_argument("--max_exit_follow", type=int, default=2, help="If Exit/Exits appears, keep up to N following tokens as exit ids.")
    ap.add_argument("--skip_probe", action="store_true", help="Skip PG sampling probe (cand_est/hits_est), only verify seed matches regex.")

    args = ap.parse_args()

    if args.mode == "prep":
        build_pools(
            args.csv,
            args.pools,
            df_low_pct=args.df_low_pct,
            df_high_pct=args.df_high_pct,
            island_low_pct=args.island_low_pct,
            island_high_pct=args.island_high_pct,
            max_rows=(args.max_rows_df or None),
        )
        return

    # output dirs
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_dir = args.tmp_dir or os.path.join(out_dir if out_dir else ".", "tmp_workers")
    os.makedirs(tmp_dir, exist_ok=True)

    # split target across workers
    W = max(1, int(args.workers))
    per = args.num // W
    rem = args.num % W
    targets = [per + (1 if i < rem else 0) for i in range(W)]

    futures = []
    all_reasons = defaultdict(int)
    tmp_paths: List[str] = []

    with ProcessPoolExecutor(max_workers=W) as ex:
        for wid, n_target in enumerate(targets):
            if n_target <= 0:
                continue
            futures.append(ex.submit(
                worker_generate,
                wid,
                n_target,
                args.pools,
                args.pg_dsn,
                args.anchors,
                args.min_mid_blocks,
                args.replace_ratio,
                args.g,
                args.len_slack,
                args.cand_cap,
                args.probe_pct,
                args.probe_rounds,
                args.min_hits_est,
                args.seed_min_tokens,
                args.max_attempts_per_worker,
                tmp_dir,
                # new knobs
                args.islands_k,
                args.max_hwy_keep,
                args.max_dir_keep,
                args.max_exit_follow,
                args.skip_probe,
            ))

        for fu in tqdm(as_completed(futures), total=len(futures), desc="Workers"):
            res = fu.result()
            tmp_paths.append(res["tmp_path"])
            for k, v in res["reasons"].items():
                all_reasons[k] += int(v)

    # merge to final out (optional dedup)
    seen = set()
    n_written = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for p in tmp_paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    if args.dedup:
                        try:
                            obj = json.loads(line)
                            key = (obj.get("seed_id"), obj.get("regex"))
                        except Exception:
                            key = None
                        if key is not None:
                            if key in seen:
                                continue
                            seen.add(key)
                    out_f.write(line)
                    n_written += 1

    print(f"[OK] wrote {n_written} lines -> {args.out}")
    print("[REASONS]", dict(all_reasons))
    print(f"[TMP] worker files in: {tmp_dir}")
    print(f"[CFG] anchors={args.anchors} min_mid_blocks={args.min_mid_blocks} replace_ratio={args.replace_ratio} "
          f"islands_k={args.islands_k} max_hwy_keep={args.max_hwy_keep} max_dir_keep={args.max_dir_keep} "
          f"skip_probe={args.skip_probe} probe_pct={args.probe_pct} probe_rounds={args.probe_rounds}")

if __name__ == "__main__":
    main()
