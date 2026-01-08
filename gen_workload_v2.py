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
# Regex building blocks
# -----------------------------
def _q(n: int, slack: int) -> str:
    if slack <= 0:
        return f"{{{n},{n}}}"
    lo = max(1, n - slack)
    hi = n + slack
    return f"{{{lo},{hi}}}"

def shape_block(tok: str, slack: int = 0) -> str:
    if tok == "":
        return ""  # avoid {0,0}

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

def anchor_token_pattern(tok: str, anchor2: str, mode: str, slack: int = 0) -> str:
    a = re.escape(anchor2)
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

def max_literal_run_len(regex: str) -> int:
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
# Pools
# -----------------------------
@dataclass
class Pools:
    mid_2ch: set

def build_pools(
    csv_path: str,
    out_json: str,
    df_low_pct=0.005,
    df_high_pct=0.15,
    max_rows: Optional[int] = None,
):
    df = defaultdict(int)
    N = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in tqdm(r, desc="DF pass"):
            s = row["Description"]
            toks = tokenize(s)
            seen = set(t for t in toks if not is_punct(t))
            for t in seen:
                df[t] += 1
            N += 1
            if max_rows and N >= max_rows:
                break

    low = int(df_low_pct * N)
    high = int(df_high_pct * N)
    mid_2ch = set()

    for t, c in df.items():
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
    }
    with open(out_json, "w", encoding="utf-8") as w:
        json.dump(obj, w, ensure_ascii=False)
    print(f"[OK] pools saved: {out_json}  mid_2ch={len(mid_2ch)}  N={N}")

def load_pools(pools_json: str) -> Pools:
    with open(pools_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return Pools(mid_2ch=set(obj["mid_2ch"]))

# -----------------------------
# Anchor / middle-block selection
# -----------------------------
LOW_VALUE_2CH = {
    "nd", "st", "rd", "th",
    "on", "at", "to", "of", "in", "an", "or", "by", "as", "is",
}

def pick_anchors(tokens: List[str], pools: Pools, k=2) -> Optional[List[Tuple[int, str, str]]]:
    cand = []
    T = len(tokens)
    for i, tok in enumerate(tokens):
        if is_punct(tok):
            continue
        if len(tok) >= 2:
            pre = tok[:2]
            suf = tok[-2:]
            if pre in pools.mid_2ch:
                cand.append((i, pre, "pre"))
            if suf in pools.mid_2ch and suf != pre:
                cand.append((i, suf, "suf"))

    if len(cand) < k:
        return None

    early = [c for c in cand if c[0] <= int(0.2 * T)]
    mid = [c for c in cand if int(0.2 * T) < c[0] < int(0.8 * T)]
    late = [c for c in cand if c[0] >= int(0.8 * T)]

    def _prefer(lst: List[Tuple[int, str, str]]) -> List[Tuple[int, str, str]]:
        good = [x for x in lst if x[1].lower() not in LOW_VALUE_2CH]
        return good if len(good) >= k else lst

    early = _prefer(early)
    mid = _prefer(mid)
    late = _prefer(late)
    cand2 = _prefer(cand)

    picked = []
    if mid:
        picked.append(random.choice(mid))
    if late and len(picked) < k:
        picked.append(random.choice(late))
    if early and len(picked) < k:
        picked.append(random.choice(early))
    while len(picked) < k:
        picked.append(random.choice(cand2))

    uniq_by_anchor: Dict[str, Tuple[int, str, str]] = {}
    extra = random.sample(cand2, min(len(cand2), 8))
    for x in picked + extra:
        uniq_by_anchor.setdefault(x[1], x)
        if len(uniq_by_anchor) >= k:
            break

    if len(uniq_by_anchor) >= k:
        picked = list(uniq_by_anchor.values())[:k]
    else:
        uniq_by_idx = {}
        for x in picked:
            uniq_by_idx[x[0]] = x
        picked = list(uniq_by_idx.values())
        if len(picked) < k:
            return None
        picked = random.sample(picked, k)

    picked.sort(key=lambda x: x[0])
    return picked

def pick_middle_blocks(tokens: List[str], anchor_pos: set, min_blocks=2) -> Optional[List[int]]:
    T = len(tokens)
    L = int(0.3 * T)
    R = int(0.7 * T)
    mid_idx = [i for i in range(L, R + 1) if i not in anchor_pos and (not is_punct(tokens[i]))]
    if len(mid_idx) < min_blocks:
        return None
    return random.sample(mid_idx, min_blocks)

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
    replace_ratio=0.7,
    g=0,
    len_slack: int = 0,
) -> str:
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

        if i in anchor_pos:
            _, a2, mode = next(x for x in anchors if x[0] == i)
            pats.append(anchor_token_pattern(tok, a2, mode, slack=len_slack))
        elif i in mid_pos:
            pats.append(shape_block(tok, slack=len_slack))
        else:
            if random.random() < replace_ratio:
                pats.append(shape_block(tok, slack=len_slack))
            else:
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

def hits_probe_pg(
    conn,
    a1: str,
    a2: str,
    regex: str,
    cand_cap: int,
    probe_pct: float,
    probe_rounds: int,
) -> Tuple[int, int]:
    p1 = f"%{a1}%"
    p2 = f"%{a2}%"
    pct = max(0.0001, min(100.0, float(probe_pct)))
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
    last_cand = 0
    try:
        with conn.cursor() as cur:
            for _ in range(max(1, int(probe_rounds))):
                cur.execute(sql, (p1, p2, cand_cap, regex))
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
) -> Dict[str, Any]:
    """
    Generate num_target records and write to a worker tmp file.
    Returns dict: {"tmp_path": ..., "n_ok": ..., "reasons": {...}}
    """
    # Per-process randomness
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
            mid_blocks = pick_middle_blocks(toks, anchor_pos, min_blocks=min_mid_blocks)
            if not mid_blocks:
                reasons["no_mid_blocks"] += 1
                continue

            regex = assemble_regex(
                toks, anchors, mid_blocks,
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
            cand_est, hits_est = hits_probe_pg(
                conn, a_list[0], a_list[1], regex,
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
                "stats": {
                    "max_literal_run_len": int(ml),
                    "cand_est": int(cand_est),
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
    ap.add_argument("--pools", default="us_acc_pools.json")
    ap.add_argument("--out", default="us_acc_stage1.jsonl")
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers (processes).")
    ap.add_argument("--tmp_dir", default="", help="Directory for worker tmp jsonl files. Default: <out_dir>/tmp_workers")

    ap.add_argument("--anchors", type=int, default=2)
    ap.add_argument("--min_mid_blocks", type=int, default=2)
    ap.add_argument("--replace_ratio", type=float, default=0.7)
    ap.add_argument("--g", type=int, default=0)
    ap.add_argument("--cand_cap", type=int, default=2000)
    ap.add_argument("--max_rows_df", type=int, default=0)
    ap.add_argument("--pg_dsn", default="dbname=regex_db user=lyz host=localhost port=5000")
    ap.add_argument("--len_slack", type=int, default=0)

    ap.add_argument("--min_hits_est", type=int, default=1)
    ap.add_argument("--probe_pct", type=float, default=0.05)
    ap.add_argument("--probe_rounds", type=int, default=3)

    ap.add_argument("--seed_min_tokens", type=int, default=6)
    ap.add_argument("--max_attempts_per_worker", type=int, default=2_000_000,
                    help="Hard cap to avoid infinite loops if acceptance rate is too low.")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate by (seed_id, regex) before writing final output.")

    args = ap.parse_args()

    if args.mode == "prep":
        build_pools(args.csv, args.pools, max_rows=(args.max_rows_df or None))
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

    # run
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

if __name__ == "__main__":
    main()
