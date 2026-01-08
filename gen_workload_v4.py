#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
import re
import zlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Set

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import errors as pg_errors
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
# Regex helpers: max literal run
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
ABSTRACT_ESCAPES = set("dDsSwW")  # \d \D \s \S \w \W


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
                grams.add(s[j : j + 3])
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
# Stable (fast) 64-bit hash for (seed_id, regex) de-dup
# =====================================================================================
def _hash64_str(s: str) -> int:
    """
    Fast-ish 64-bit hash using two independent 32-bit checksums.
    Collision probability is negligible for our scale.
    """
    b = s.encode("utf-8", errors="ignore")
    h1 = zlib.crc32(b) & 0xFFFFFFFF
    h2 = zlib.adler32(b) & 0xFFFFFFFF
    return (h1 << 32) | h2


def pair_key_u128(seed_id: int, regex: str) -> int:
    """
    Pack into a single Python int:
      key = (seed_id << 64) | hash64(regex)
    """
    return (int(seed_id) << 64) | _hash64_str(regex)


# =====================================================================================
# Regex building blocks
# =====================================================================================
def _q(n: int, slack: int) -> str:
    if slack <= 0:
        return f"{{{n},{n}}}"
    lo = max(1, n - slack)
    hi = n + slack
    return f"{{{lo},{hi}}}"


def keep_numeric_literal(tok: str) -> bool:
    # any token containing a digit: keep literal (e.g., I-710, GA-400, 11A-11B, Exit 6)
    return any(ch.isdigit() for ch in tok)


def shape_block(tok: str, slack: int = 0, keep_digits_literal: bool = True) -> str:
    """
    Convert a token to a regex fragment.
    Hard-only generator:
      - If keep_digits_literal and token contains digits => keep it literally (escaped).
      - Else fall back to shape-based abstraction.
    """
    if tok == "":
        return ""

    if keep_digits_literal and keep_numeric_literal(tok):
        return re.escape(tok)

    if re.fullmatch(r"[0-9]+(st|nd|rd|th)", tok):
        return re.escape(tok)

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

    return rf"[A-Za-z0-9]{_q(len(tok), slack)}"


def anchor_token_pattern(
    tok: str,
    anchor: str,
    mode: str,
    slack: int = 0,
    keep_digits_literal: bool = True,
) -> str:
    """
    Use a length-L anchor (literal) inside a token, abstract the rest.
    Example: Right with anchor="Rig" pre => Rig[a-z]{2,2}
    """
    if keep_digits_literal and keep_numeric_literal(tok):
        return re.escape(tok)

    a = re.escape(anchor)
    L = len(anchor)

    if len(tok) == L and tok == anchor:
        return a

    if mode == "pre":
        rest = tok[L:]
        rest_pat = shape_block(rest, slack=slack, keep_digits_literal=keep_digits_literal) if rest else ""
        return a + (rest_pat if rest_pat else "")
    else:
        rest = tok[:-L]
        rest_pat = shape_block(rest, slack=slack, keep_digits_literal=keep_digits_literal) if rest else ""
        return (rest_pat if rest_pat else "") + a


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
    Build a pool of mid-frequency L-grams from token prefixes/suffixes, restricted to alphabetic grams.
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
# Anchor / middle-block selection
# =====================================================================================
LOW_VALUE_ANCH = {
    "the", "and", "for", "wit", "are", "you", "acc", "on", "at"
}


def pick_anchors(tokens: List[str], pools: Pools, k: int = 2) -> Optional[List[Tuple[int, str, str]]]:
    """
    Pick k anchors (pos, anchor, mode) using mid-frequency L-grams from pools.
    Restrict to alphabetic tokens and anchor grams in pool.
    """
    L = pools.anchor_len
    cand: List[Tuple[int, str, str]] = []

    T = len(tokens)
    for i, tok in enumerate(tokens):
        if is_punct(tok):
            continue
        if keep_numeric_literal(tok):
            continue
        if len(tok) < L:
            continue

        pre = tok[:L]
        suf = tok[-L:]
        if pre in pools.mid_anch and pre.lower() not in LOW_VALUE_ANCH:
            cand.append((i, pre, "pre"))
        if suf in pools.mid_anch and suf != pre and suf.lower() not in LOW_VALUE_ANCH:
            cand.append((i, suf, "suf"))

    if len(cand) < k:
        return None

    early = [c for c in cand if c[0] <= int(0.2 * T)]
    mid = [c for c in cand if int(0.2 * T) < c[0] < int(0.8 * T)]
    late = [c for c in cand if c[0] >= int(0.8 * T)]

    picked: List[Tuple[int, str, str]] = []
    if mid:
        picked.append(random.choice(mid))
    if late and len(picked) < k:
        picked.append(random.choice(late))
    if early and len(picked) < k:
        picked.append(random.choice(early))
    while len(picked) < k:
        picked.append(random.choice(cand))

    uniq: Dict[str, Tuple[int, str, str]] = {}
    for x in picked + random.sample(cand, min(len(cand), 8)):
        uniq.setdefault(x[1], x)
        if len(uniq) >= k:
            break
    if len(uniq) < k:
        return None

    out = list(uniq.values())[:k]
    out.sort(key=lambda x: x[0])
    return out


def pick_middle_blocks(tokens: List[str], anchor_pos: set, min_blocks: int = 2) -> Optional[List[int]]:
    T = len(tokens)
    L = int(0.3 * T)
    R = int(0.7 * T)
    mid_idx = [
        i for i in range(L, R + 1)
        if i not in anchor_pos and (not is_punct(tokens[i]))
    ]
    if len(mid_idx) < min_blocks:
        return None
    return random.sample(mid_idx, min_blocks)


# =====================================================================================
# Joining logic: punctuation-aware spacing
# =====================================================================================
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


# =====================================================================================
# Assemble regex (Hard-only controlled 3-grams)
# =====================================================================================
def token_prefix_in_pool(tok: str, pools: Pools) -> Optional[str]:
    L = pools.anchor_len
    if keep_numeric_literal(tok) or len(tok) < L:
        return None
    pre = tok[:L]
    if pre in pools.mid_anch and pre.lower() not in LOW_VALUE_ANCH:
        return pre
    return None


def assemble_regex(
    tokens: List[str],
    pools: Pools,
    anchors: List[Tuple[int, str, str]],
    middle_blocks: List[int],
    replace_ratio: float = 0.75,
    g: int = 0,
    len_slack: int = 0,
    keep_digits_literal: bool = True,
    keep3_prob: float = 0.10,
    mid_keep3_prob: float = 0.35,
) -> Tuple[str, int]:
    """
    Return (regex, kept3_count).
    kept3_count: number of non-anchor tokens where we kept a 3-letter literal prefix.
    """
    anchor_pos = {i for i, _, _ in anchors}
    mid_pos = set(middle_blocks)

    pats: List[str] = []
    kept3 = 0

    for i, tok in enumerate(tokens):
        if is_punct(tok):
            pats.append(r"\." if tok == "." else re.escape(tok))
            continue

        if keep_digits_literal and keep_numeric_literal(tok):
            pats.append(re.escape(tok))
            continue

        if i in anchor_pos:
            _, a, mode = next(x for x in anchors if x[0] == i)
            pats.append(
                anchor_token_pattern(tok, a, mode, slack=len_slack, keep_digits_literal=keep_digits_literal)
            )
            continue

        if i in mid_pos:
            if random.random() < mid_keep3_prob:
                pre = token_prefix_in_pool(tok, pools)
                if pre is not None:
                    rest = tok[len(pre) :]
                    rest_pat = shape_block(rest, slack=len_slack, keep_digits_literal=keep_digits_literal) if rest else ""
                    pats.append(re.escape(pre) + rest_pat)
                    kept3 += 1
                else:
                    pats.append(shape_block(tok, slack=len_slack, keep_digits_literal=keep_digits_literal))
            else:
                pats.append(shape_block(tok, slack=len_slack, keep_digits_literal=keep_digits_literal))
            continue

        if random.random() < replace_ratio:
            if random.random() < keep3_prob:
                pre = token_prefix_in_pool(tok, pools)
                if pre is not None:
                    rest = tok[len(pre) :]
                    rest_pat = shape_block(rest, slack=len_slack, keep_digits_literal=keep_digits_literal) if rest else ""
                    pats.append(re.escape(pre) + rest_pat)
                    kept3 += 1
                else:
                    pats.append(shape_block(tok, slack=len_slack, keep_digits_literal=keep_digits_literal))
            else:
                pats.append(shape_block(tok, slack=len_slack, keep_digits_literal=keep_digits_literal))
        else:
            pre = token_prefix_in_pool(tok, pools)
            if pre is not None:
                rest = tok[len(pre) :]
                rest_pat = shape_block(rest, slack=len_slack, keep_digits_literal=keep_digits_literal) if rest else ""
                pats.append(re.escape(pre) + rest_pat)
                kept3 += 1
            else:
                pats.append(shape_block(tok, slack=len_slack, keep_digits_literal=keep_digits_literal))

    body = join_token_patterns(tokens, pats)
    if g > 0:
        return rf".{{0,{g}}}" + body, kept3
    return body, kept3


# =====================================================================================
# DB helpers (key changes: range-partition sampling; TABLESAMPLE optional and default OFF)
# =====================================================================================
def _get_id_bounds(conn, table: str) -> Optional[Tuple[int, int]]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT min(id), max(id) FROM {table};")
        mn, mx = cur.fetchone()
    if mn is None or mx is None:
        return None
    return int(mn), int(mx)


def _split_id_ranges(mn: int, mx: int, workers: int) -> List[Tuple[int, int]]:
    """
    Split [mn, mx] into W disjoint contiguous ranges by id (approx equal width).
    This prevents cross-worker seed collisions (major source of duplicates before).
    """
    W = max(1, int(workers))
    if mn > mx:
        return [(mn, mx)] * W
    span = mx - mn + 1
    step = max(1, span // W)
    ranges: List[Tuple[int, int]] = []
    cur = mn
    for i in range(W):
        lo = cur
        if i == W - 1:
            hi = mx
        else:
            hi = min(mx, lo + step - 1)
        ranges.append((lo, hi))
        cur = hi + 1
        if cur > mx and i < W - 1:
            # if span < W, pad remaining with empty-ish ranges
            ranges.extend([(mx + 1, mx)] * (W - i - 1))
            break
    return ranges


def fetch_random_seed(
    conn,
    table: str,
    min_tokens: int,
    id_lo: int,
    id_hi: int,
    use_tablesample: bool = False,
    tablesample_pct: float = 0.01,
) -> Tuple[int, str, List[str]]:
    """
    Sample a seed row.
    Default path (recommended): random id within [id_lo, id_hi] and seek forward using PK.
    This avoids TABLESAMPLE cross-process page collisions which caused extreme seed duplication.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        while True:
            row = None

            if use_tablesample:
                try:
                    pct = max(0.0001, min(100.0, float(tablesample_pct)))
                    cur.execute(f"SELECT id, description FROM {table} TABLESAMPLE SYSTEM ({pct}) LIMIT 1;")
                    row = cur.fetchone()
                except Exception:
                    row = None

            # Range-seek sampling (fast on PK)
            if (not row) and id_lo <= id_hi:
                rid = random.randint(id_lo, id_hi)

                cur.execute(
                    f"SELECT id, description FROM {table} WHERE id >= %s AND id <= %s ORDER BY id LIMIT 1;",
                    (rid, id_hi),
                )
                row = cur.fetchone()

                # wrap-around within range
                if not row:
                    cur.execute(
                        f"SELECT id, description FROM {table} WHERE id >= %s AND id <= %s ORDER BY id LIMIT 1;",
                        (id_lo, id_hi),
                    )
                    row = cur.fetchone()

            # absolute fallback (should be rare; but keep to avoid infinite loop)
            if not row:
                cur.execute(f"SELECT id, description FROM {table} WHERE id IS NOT NULL ORDER BY id LIMIT 1;")
                row = cur.fetchone()
                if not row:
                    raise RuntimeError("No rows available in table.")

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
    except Exception:
        return False


# =====================================================================================
# Worker
# =====================================================================================
def worker_generate(
    worker_id: int,
    num_target: int,
    pools_json: str,
    pg_dsn: str,
    table: str,
    id_lo: int,
    id_hi: int,
    use_tablesample: bool,
    tablesample_pct: float,
    anchors_k: int,
    min_mid_blocks: int,
    replace_ratio: float,
    g: int,
    len_slack: int,
    seed_min_tokens: int,
    max_attempts: int,
    out_tmp_dir: str,
    keep_digits_literal: bool,
    keep3_prob: float,
    mid_keep3_prob: float,
    trgm_min: int,
    trgm_max: int,
    max_lit_run: int,
    # new knobs
    dedup_in_worker: bool,
    max_per_seed: int,
    rng_seed: int,
) -> Dict[str, Any]:
    """
    Key changes vs your version:
      1) Per-worker disjoint id range => almost eliminates cross-worker seed collisions.
      2) Optional in-worker de-dup of (seed_id, regex) via 128-bit packed hash key (memory-light).
      3) Optional cap for how many queries to emit per seed_id (max_per_seed).
    """
    # deterministic per-worker seed (reproducible; avoids accidental correlations)
    random.seed((int(rng_seed) * 1000003) ^ (worker_id * 9176) ^ (os.getpid() << 8))

    pools = load_pools(pools_json)
    conn = psycopg2.connect(pg_dsn)

    reasons = defaultdict(int)
    ok = 0

    os.makedirs(out_tmp_dir, exist_ok=True)
    tmp_path = os.path.join(out_tmp_dir, f"tmp_worker_{worker_id}.jsonl")

    seen_pairs: Set[int] = set() if dedup_in_worker else set()
    per_seed_cnt: Dict[int, int] = {} if max_per_seed > 0 else {}

    with open(tmp_path, "w", encoding="utf-8") as w:
        attempts = 0
        while ok < num_target and attempts < max_attempts:
            attempts += 1

            seed_id, seed, toks = fetch_random_seed(
                conn,
                table=table,
                min_tokens=seed_min_tokens,
                id_lo=id_lo,
                id_hi=id_hi,
                use_tablesample=use_tablesample,
                tablesample_pct=tablesample_pct,
            )

            if max_per_seed > 0:
                c = per_seed_cnt.get(seed_id, 0)
                if c >= max_per_seed:
                    reasons["seed_over_cap"] += 1
                    continue

            anchors = pick_anchors(toks, pools, k=anchors_k)
            if not anchors:
                reasons["no_anchor"] += 1
                continue

            anchor_pos = {i for i, _, _ in anchors}
            mid_blocks = pick_middle_blocks(toks, anchor_pos, min_blocks=min_mid_blocks)
            if not mid_blocks:
                reasons["no_mid_blocks"] += 1
                continue

            regex, kept3 = assemble_regex(
                toks,
                pools=pools,
                anchors=anchors,
                middle_blocks=mid_blocks,
                replace_ratio=replace_ratio,
                g=g,
                len_slack=len_slack,
                keep_digits_literal=keep_digits_literal,
                keep3_prob=keep3_prob,
                mid_keep3_prob=mid_keep3_prob,
            )

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

            if not verify_seed_pg(conn, seed, regex):
                reasons["seed_verify_fail"] += 1
                continue

            if dedup_in_worker:
                k = pair_key_u128(seed_id, regex)
                if k in seen_pairs:
                    reasons["dup_pair_worker"] += 1
                    continue
                seen_pairs.add(k)

            if max_per_seed > 0:
                per_seed_cnt[seed_id] = per_seed_cnt.get(seed_id, 0) + 1

            a_list = [a for _, a, _ in anchors]
            rec = {
                "dataset": "us_acc",
                "seed_id": int(seed_id),
                "seed_string": seed,
                "regex": regex,
                "anchors": a_list,
                "anchor_len": int(pools.anchor_len),
                "g": int(g),
                "middle_blocks": sorted(mid_blocks),
                "stats": {
                    "max_literal_run_len": int(ml),
                    "trgm_budget": int(tb),
                    "kept3_count": int(kept3),
                },
                "bucket": "hard",
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1

    conn.close()
    return {
        "tmp_path": tmp_path,
        "n_ok": ok,
        "attempts": attempts,
        "reasons": dict(reasons),
        "range": (int(id_lo), int(id_hi)),
    }


# =====================================================================================
# Main
# =====================================================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default="/home/lyz/wd/LLM_RegularSearch/v1_data/gen_workload/data/old_data/US_Accidents_7m.csv")
    ap.add_argument("--mode", choices=["prep", "gen"], required=True)
    ap.add_argument("--pools", default="us_acc_pools_L3.json")
    ap.add_argument("--out", default="workload/us_acc_hard.jsonl")
    ap.add_argument("--num", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tmp_dir", default="", help="Default: <out_dir>/tmp_workers")

    # pool / DF knobs
    ap.add_argument("--anchor_len", type=int, default=3)
    ap.add_argument("--df_low_pct", type=float, default=0.005)
    ap.add_argument("--df_high_pct", type=float, default=0.15)
    ap.add_argument("--max_rows_df", type=int, default=0)

    # regex structure knobs
    ap.add_argument("--anchors", type=int, default=2)
    ap.add_argument("--min_mid_blocks", type=int, default=2)
    ap.add_argument("--replace_ratio", type=float, default=0.75)
    ap.add_argument("--g", type=int, default=0)
    ap.add_argument("--len_slack", type=int, default=1)

    # info knobs
    ap.add_argument("--keep_digits_literal", dest="keep_digits_literal", action="store_true", default=True)
    ap.add_argument("--no_keep_digits_literal", dest="keep_digits_literal", action="store_false")
    ap.add_argument("--keep3_prob", type=float, default=0.10)
    ap.add_argument("--mid_keep3_prob", type=float, default=0.35)

    # Hard-only bucket by trgm_budget
    ap.add_argument("--trgm_min", type=int, default=3)
    ap.add_argument("--trgm_max", type=int, default=12)

    # misc filters
    ap.add_argument("--max_lit_run", type=int, default=12)
    ap.add_argument("--seed_min_tokens", type=int, default=8)

    # DB
    ap.add_argument("--pg_dsn", default="dbname=regex_db user=lyz host=localhost port=5000")
    ap.add_argument("--table", default="us_accidents")

    # Seed sampling strategy (key changes)
    ap.add_argument("--split_id_ranges", dest="split_id_ranges", action="store_true", default=True,
                    help="Split id range across workers to avoid cross-worker seed collisions (recommended).")
    ap.add_argument("--no_split_id_ranges", dest="split_id_ranges", action="store_false")
    ap.add_argument("--use_tablesample", action="store_true", default=False,
                    help="Optional: try TABLESAMPLE first. Default OFF (it caused heavy duplication in multi-proc).")
    ap.add_argument("--tablesample_pct", type=float, default=0.01)

    # new anti-dup knobs
    ap.add_argument("--dedup_in_worker", dest="dedup_in_worker", action="store_true", default=True,
                    help="Dedup exact (seed_id, regex) within each worker before writing tmp file (recommended).")
    ap.add_argument("--no_dedup_in_worker", dest="dedup_in_worker", action="store_false")
    ap.add_argument("--max_per_seed", type=int, default=0,
                    help="Cap emitted queries per seed_id inside each worker. 0 = unlimited. "
                         "Useful to reduce seed domination; may reduce max throughput if too small.")
    ap.add_argument("--rng_seed", type=int, default=42,
                    help="Global RNG seed for reproducibility (per-worker derived).")

    # safety
    ap.add_argument("--max_attempts_per_worker", type=int, default=2_000_000)
    ap.add_argument("--dedup", action="store_true", help="Dedup by (seed_id, regex) on merge (hash-based, memory-light).")

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

    # Resolve global id bounds once in main
    conn0 = psycopg2.connect(args.pg_dsn)
    bounds = _get_id_bounds(conn0, args.table)
    conn0.close()
    if bounds is None:
        raise SystemExit("[Error] Could not get id bounds from table.")
    mn, mx = bounds

    W = max(1, int(args.workers))

    # Split targets across workers
    per = args.num // W
    rem = args.num % W
    targets = [per + (1 if i < rem else 0) for i in range(W)]

    # Split id ranges across workers (recommended)
    if args.split_id_ranges:
        ranges = _split_id_ranges(mn, mx, W)
    else:
        ranges = [(mn, mx)] * W

    futures = []
    all_reasons = defaultdict(int)
    tmp_paths: List[str] = []

    with ProcessPoolExecutor(max_workers=W) as ex:
        for wid, n_target in enumerate(targets):
            if n_target <= 0:
                continue
            lo, hi = ranges[wid]
            futures.append(
                ex.submit(
                    worker_generate,
                    wid,
                    n_target,
                    args.pools,
                    args.pg_dsn,
                    args.table,
                    lo,
                    hi,
                    args.use_tablesample,
                    args.tablesample_pct,
                    args.anchors,
                    args.min_mid_blocks,
                    args.replace_ratio,
                    args.g,
                    args.len_slack,
                    args.seed_min_tokens,
                    args.max_attempts_per_worker,
                    tmp_dir,
                    args.keep_digits_literal,
                    args.keep3_prob,
                    args.mid_keep3_prob,
                    args.trgm_min,
                    args.trgm_max,
                    args.max_lit_run,
                    args.dedup_in_worker,
                    args.max_per_seed,
                    args.rng_seed,
                )
            )

        for fu in tqdm(as_completed(futures), total=len(futures), desc="Workers"):
            res = fu.result()
            tmp_paths.append(res["tmp_path"])
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
        # Hash-based merge dedup: keeps memory reasonable even for multi-million lines
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
                            k = pair_key_u128(sid, rx)
                        except Exception:
                            continue
                        if k in seen:
                            continue
                        seen.add(k)
                        out_f.write(line)
                        n_written += 1

    print(f"[OK] wrote {n_written} lines -> {args.out}")
    print("[REASONS]", dict(all_reasons))
    print(f"[TMP] worker files in: {tmp_dir}")
    if args.split_id_ranges:
        print("[INFO] split_id_ranges=ON (cross-worker seed collisions largely eliminated)")
    else:
        print("[INFO] split_id_ranges=OFF (expect more duplicates across workers)")

    # Quick operational hints (printed, not enforced)
    if args.use_tablesample:
        print("[WARN] use_tablesample=ON: may increase seed collisions under multi-process. Prefer OFF if duplicates rise.")


if __name__ == "__main__":
    main()
