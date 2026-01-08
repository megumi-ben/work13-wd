#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Any

# Optional PG support
try:
    import psycopg2
except Exception:
    psycopg2 = None

# -----------------------------
# Tokenization (for prefix boundary)
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)

def token_spans(s: str) -> List[Tuple[int, int, str]]:
    """Return [(start, end, tok)] in original string order."""
    return [(m.start(), m.end(), m.group(0)) for m in TOKEN_RE.finditer(s)]

def count_tokens(s: str) -> int:
    return len(TOKEN_RE.findall(s))

# -----------------------------
# Regex normalization
# -----------------------------
_WS_PAT = re.compile(r"\\s(\*|\+)")
_QUANT_PAT = re.compile(r"\{(\d+),(\d+)\}")

def normalize_regex(regex: str, ws_to_plus: bool = True, drop_quant_numbers: bool = False) -> str:
    """
    Pragmatic normalization to reduce "string-different but semantically-close" variants.

    - ws_to_plus: normalize all \s* and \s+ => \s+
    - drop_quant_numbers: replace {m,n} => {#,#} to further merge variants (use with care)
    """
    r = regex

    if ws_to_plus:
        r = _WS_PAT.sub(r"\\s+", r)

    if drop_quant_numbers:
        r = _QUANT_PAT.sub("{#,#}", r)

    # collapse repeated \s+ occurrences
    r = re.sub(r"(\\s\+){2,}", r"\\s+", r)

    return r

# -----------------------------
# LIKE escaping for prefix%
# -----------------------------
def like_escape_literal(prefix: str) -> str:
    """Escape % and _ for LIKE/ILIKE, using backslash as ESCAPE."""
    return prefix.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")

# -----------------------------
# Optional PG probing: estimate candidate size for prefix%
# -----------------------------
@dataclass
class PgProbeConfig:
    table: str
    probe_pct: float
    probe_rounds: int

class PgProber:
    def __init__(self, dsn: str, cfg: PgProbeConfig):
        if psycopg2 is None:
            raise RuntimeError("psycopg2 not available, but --pg_dsn was provided.")
        self.conn = psycopg2.connect(dsn)
        self.cfg = cfg

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def estimate_prefix_candidates(self, prefix: str) -> Optional[int]:
        """
        Use TABLESAMPLE SYSTEM(pct) and count how many rows match prefix% in the sample.
        Return a scaled estimate: sample_count / (pct/100), using median across rounds.
        """
        pct = max(0.0001, min(100.0, float(self.cfg.probe_pct)))
        rounds = max(1, int(self.cfg.probe_rounds))

        pat = like_escape_literal(prefix) + "%"

        sql = f"""
        WITH samp AS (
          SELECT description
          FROM {self.cfg.table} TABLESAMPLE SYSTEM ({pct})
        )
        SELECT count(*)::bigint
        FROM samp
        WHERE description ILIKE %s ESCAPE '\\\\';
        """

        counts: List[int] = []
        try:
            with self.conn.cursor() as cur:
                for _ in range(rounds):
                    cur.execute(sql, (pat,))
                    counts.append(int(cur.fetchone()[0]))
        except Exception:
            return None

        if not counts:
            return None

        med = int(statistics.median(counts))
        scale = 100.0 / pct
        est = int(med * scale)
        return est

# -----------------------------
# Prefix candidate generation & selection
# -----------------------------
def gen_prefix_candidates_from_seed(
    seed: str,
    min_tokens: int,
    max_tokens: int,
    include_trailing_space: bool,
) -> List[str]:
    """
    Generate prefix strings at token boundaries (based on TOKEN_RE spans),
    preserving exact original characters (incl. punctuation and spacing).
    """
    spans = token_spans(seed)
    if len(spans) < min_tokens:
        return []

    out: List[str] = []
    upto = min(len(spans), max_tokens)
    for t in range(min_tokens, upto + 1):
        end_pos = spans[t - 1][1]  # end of t-th token
        if include_trailing_space:
            j = end_pos
            while j < len(seed) and seed[j].isspace():
                j += 1
            end_pos = j
        pref = seed[:end_pos]
        if pref and pref.strip():
            out.append(pref)
    return out

def startswith_ci(s: str, pref: str) -> bool:
    return s.lower().startswith(pref.lower())

def select_prefixes_for_group(
    seeds: List[str],
    rng: random.Random,
    target_k: int,
    min_tokens: int,
    max_tokens: int,
    include_trailing_space: bool,
    min_support: int,
    min_support_ratio: float,
    max_candidates: int,
    pg: Optional[PgProber],
    bucket_tokens: int,
    prefer_longer_prefixes: bool,
) -> List[Dict[str, Any]]:
    """
    Build a candidate pool from group seeds, filter by support, score and select top-K.

    Returns:
      [{"prefix": "...", "support": n, "support_ratio": r, "cand_est": int|None, ...}, ...]
    """
    n = len(seeds)
    if n == 0:
        return []

    # Make min_support feasible for small groups to avoid pathological fallback.
    min_support_eff = min(max(1, int(min_support)), n)

    # 1) candidate pool
    cand_set: set = set()
    for s in seeds:
        for p in gen_prefix_candidates_from_seed(s, min_tokens, max_tokens, include_trailing_space):
            cand_set.add(p)

    if not cand_set:
        return []

    # 2) support filtering (within group)
    cand: List[Tuple[str, int, float]] = []
    for p in cand_set:
        sup = sum(1 for s in seeds if startswith_ci(s, p))
        ratio = sup / n
        if sup >= min_support_eff and ratio >= float(min_support_ratio):
            cand.append((p, sup, ratio))

    # If nothing passes, relax (but keep all candidates with their support stats).
    if not cand:
        for p in cand_set:
            sup = sum(1 for s in seeds if startswith_ci(s, p))
            ratio = sup / n
            cand.append((p, sup, ratio))

    # 3) pre-trim (avoid exploding DB probes)
    # Prefer longer + higher support before probing.
    cand.sort(key=lambda x: (-len(x[0]), -x[1], -x[2]))
    cand = cand[:max(1, int(max_candidates))]

    # 4) optional PG estimation + token count
    enriched: List[Tuple[str, int, float, Optional[int], int]] = []
    for p, sup, ratio in cand:
        est = pg.estimate_prefix_candidates(p) if pg is not None else None
        tok_cnt = count_tokens(p)
        enriched.append((p, sup, ratio, est, tok_cnt))

    # 5) scoring
    # Primary: cand_est (smaller is better) if available.
    # Secondary: longer prefix, higher support.
    BIG = 10**18

    def score(item: Tuple[str, int, float, Optional[int], int]) -> Tuple[int, int, int, float]:
        p, sup, ratio, est, _tok_cnt = item
        e = int(est) if est is not None else BIG
        return (e, -len(p), -sup, -ratio)

    # 6) bucket-based selection to get K non-trivial prefixes of different lengths
    bucket_size = max(1, int(bucket_tokens))
    # bucket index based on token count (min_tokens..max_tokens)
    def bucket_idx(tok_cnt: int) -> int:
        return max(0, (tok_cnt - min_tokens) // bucket_size)

    # pick best candidate per bucket (by score)
    best_by_bucket: Dict[int, Tuple[str, int, float, Optional[int], int]] = {}
    for item in enriched:
        b = bucket_idx(item[4])
        cur = best_by_bucket.get(b)
        if cur is None or score(item) < score(cur):
            best_by_bucket[b] = item

    bucket_items = list(best_by_bucket.values())

    # order buckets: prefer longer-token prefixes first (usually tighter), else shorter first
    if prefer_longer_prefixes:
        bucket_items.sort(key=lambda x: (-x[4], score(x)))
    else:
        bucket_items.sort(key=lambda x: (x[4], score(x)))

    selected: List[Dict[str, Any]] = []
    selected_set = set()
    selected_buckets = set()

    def add_item(item: Tuple[str, int, float, Optional[int], int]):
        p, sup, ratio, est, tok_cnt = item
        key = p.lower()
        if key in selected_set:
            return False
        selected_set.add(key)
        selected_buckets.add(bucket_idx(tok_cnt))
        selected.append({
            "prefix": p,
            "support": int(sup),
            "support_ratio": float(ratio),
            "cand_est": int(est) if est is not None else None,
            "len_chars": len(p),
            "len_tokens": int(tok_cnt),
            "bucket": int(bucket_idx(tok_cnt)),
        })
        return True

    # first pass: 1 per bucket
    for item in bucket_items:
        add_item(item)
        if len(selected) >= target_k:
            break

    # second pass: fill remaining slots, preferring new buckets if possible
    if len(selected) < target_k:
        # Sort all by score and try to add unseen buckets first.
        enriched_sorted = sorted(enriched, key=score)

        # Pass A: unseen buckets
        for item in enriched_sorted:
            if len(selected) >= target_k:
                break
            b = bucket_idx(item[4])
            if b in selected_buckets:
                continue
            add_item(item)

        # Pass B: any remaining (only de-dup exact prefix string)
        for item in enriched_sorted:
            if len(selected) >= target_k:
                break
            add_item(item)

    # guarantee at least 1
    if not selected:
        enriched_sorted = sorted(enriched, key=score)
        add_item(enriched_sorted[0])

    return selected

# -----------------------------
# IO helpers
# -----------------------------
def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Input per-seed workload jsonl (generator output).")
    ap.add_argument("--out_jsonl", required=True, help="Output grouped jsonl (regex_norm -> prefixes).")

    # regex_norm options (default ON, can disable)
    ap.add_argument("--norm_ws_to_plus", dest="norm_ws_to_plus", action="store_true", default=True,
                    help="Normalize \\s* and \\s+ into \\s+ (default: on).")
    ap.add_argument("--no_norm_ws_to_plus", dest="norm_ws_to_plus", action="store_false",
                    help="Disable whitespace normalization.")
    ap.add_argument("--drop_quant_numbers", action="store_true", default=False,
                    help="Replace {m,n} => {#,#} to merge more variants (use with care).")

    # grouping / sampling
    ap.add_argument("--min_group_size", type=int, default=1,
                    help="Minimum group size to keep. Default=1 keeps singleton groups.")
    ap.add_argument("--seed_sample_per_group", type=int, default=200,
                    help="Max seeds used for prefix selection per group.")
    ap.add_argument("--target_prefixes", type=int, default=5,
                    help="Top-K prefixes to keep per group.")
    ap.add_argument("--random_seed", type=int, default=1234)

    # prefix candidate controls
    ap.add_argument("--prefix_min_tokens", type=int, default=2)
    ap.add_argument("--prefix_max_tokens", type=int, default=10)

    ap.add_argument("--include_trailing_space", dest="include_trailing_space",
                    action="store_true", default=True,
                    help="Include following spaces after token boundary (default: on).")
    ap.add_argument("--no_include_trailing_space", dest="include_trailing_space",
                    action="store_false",
                    help="Disable including trailing spaces.")

    ap.add_argument("--min_support", type=int, default=2, help="Min #seeds in group sharing the prefix.")
    ap.add_argument("--min_support_ratio", type=float, default=0.05, help="Min fraction of group covered by prefix.")
    ap.add_argument("--max_candidates", type=int, default=300, help="Cap candidate pool per group before scoring.")

    # multi-prefix selection controls
    ap.add_argument("--prefix_bucket_tokens", type=int, default=2,
                    help="Bucket size by token count when selecting multiple prefixes. 2 => [2-3],[4-5],...")
    ap.add_argument("--prefer_longer_prefixes", action="store_true", default=True,
                    help="Prefer longer-token prefixes first when selecting across buckets (default: on).")
    ap.add_argument("--prefer_shorter_prefixes", dest="prefer_longer_prefixes",
                    action="store_false",
                    help="Prefer shorter-token prefixes first.")

    # optional PG probing
    ap.add_argument("--pg_dsn", default="", help="If set, estimate prefix candidate size using TABLESAMPLE.")
    ap.add_argument("--pg_table", default="us_accidents")
    ap.add_argument("--probe_pct", type=float, default=0.05,
                    help="TABLESAMPLE percent per round (e.g., 0.05 = 0.05%%).")
    ap.add_argument("--probe_rounds", type=int, default=3)

    args = ap.parse_args()
    rng = random.Random(args.random_seed)

    # 1) group by regex_norm
    groups: Dict[str, List[Dict]] = defaultdict(list)
    n_in = 0
    for rec in read_jsonl(args.in_jsonl):
        n_in += 1
        regex = rec.get("regex", "")
        rnorm = normalize_regex(regex, ws_to_plus=args.norm_ws_to_plus, drop_quant_numbers=args.drop_quant_numbers)
        rec["_regex_norm"] = rnorm
        groups[rnorm].append(rec)

    # 2) optional PG prober
    pg = None
    if args.pg_dsn:
        pg = PgProber(args.pg_dsn, PgProbeConfig(
            table=args.pg_table,
            probe_pct=args.probe_pct,
            probe_rounds=args.probe_rounds,
        ))

    out_rows: List[Dict] = []
    dropped_small = 0

    # 3) build grouped records
    for rnorm, items in groups.items():
        if len(items) < int(args.min_group_size):
            dropped_small += 1
            continue

        # canonical regex = most common raw regex string in the group
        raw_regex_counts = Counter(x.get("regex", "") for x in items)
        canon_regex, canon_cnt = raw_regex_counts.most_common(1)[0]

        # sample seeds for prefix selection
        seeds_all = [x["seed_string"] for x in items
                     if "seed_string" in x and isinstance(x["seed_string"], str)]
        if not seeds_all:
            continue

        if len(seeds_all) > int(args.seed_sample_per_group):
            seeds = rng.sample(seeds_all, int(args.seed_sample_per_group))
        else:
            seeds = seeds_all

        prefixes = select_prefixes_for_group(
            seeds=seeds,
            rng=rng,
            target_k=int(args.target_prefixes),
            min_tokens=int(args.prefix_min_tokens),
            max_tokens=int(args.prefix_max_tokens),
            include_trailing_space=bool(args.include_trailing_space),
            min_support=int(args.min_support),
            min_support_ratio=float(args.min_support_ratio),
            max_candidates=int(args.max_candidates),
            pg=pg,
            bucket_tokens=int(args.prefix_bucket_tokens),
            prefer_longer_prefixes=bool(args.prefer_longer_prefixes),
        )

        # group-level stats from input items
        cand_est_list: List[int] = []
        hits_est_list: List[int] = []
        for x in items:
            st = x.get("stats") or {}
            if isinstance(st, dict):
                ce = st.get("cand_est")
                he = st.get("hits_est")
                if isinstance(ce, int):
                    cand_est_list.append(ce)
                if isinstance(he, int):
                    hits_est_list.append(he)

        row = {
            "dataset": items[0].get("dataset", "unknown"),
            "regex_norm": rnorm,
            "regex_canon": canon_regex,
            "group_size": len(items),
            "canon_freq_in_group": int(canon_cnt),
            # sample a few seed_ids (for debugging)
            "seed_ids_sampled": [int(x["seed_id"]) for x in items[:min(len(items), 50)] if "seed_id" in x],
            "target_prefixes": prefixes,
            "stats": {
                "input_cand_est_median": int(statistics.median(cand_est_list)) if cand_est_list else None,
                "input_hits_est_sum": int(sum(hits_est_list)) if hits_est_list else None,
            },
        }
        out_rows.append(row)

    # 4) output
    write_jsonl(args.out_jsonl, out_rows)

    if pg is not None:
        pg.close()

    # 5) summary
    sizes = [r["group_size"] for r in out_rows]
    print(f"[INPUT] lines={n_in}  unique_regex_norm={len(groups)}")
    print(f"[OK] groups_written={len(out_rows)}  groups_dropped_small={dropped_small}  min_group_size={args.min_group_size}")
    if sizes:
        sizes_sorted = sorted(sizes)
        def q(p):
            idx = int(round((p/100.0) * (len(sizes_sorted)-1)))
            return sizes_sorted[max(0, min(len(sizes_sorted)-1, idx))]
        print(f"[GROUP_SIZE] p50={q(50)} p90={q(90)} p95={q(95)} max={max(sizes_sorted)}")
    with_prefix = sum(1 for r in out_rows if r.get("target_prefixes"))
    print(f"[PREFIX] groups_with_prefix={with_prefix}/{len(out_rows)}  target_prefixes={args.target_prefixes}  bucket_tokens={args.prefix_bucket_tokens}")

if __name__ == "__main__":
    main()
