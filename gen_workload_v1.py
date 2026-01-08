import argparse
import csv
import json
import os
import random
import re
import time
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import errors as pg_errors
from tqdm import tqdm

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[^\w\s]")


def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(s)


def is_punct(tok: str) -> bool:
    return bool(re.fullmatch(r"[^\w\s]", tok))


def _q(n: int, slack: int) -> str:
    """
    quantifier for length with slack:
      slack=0 => {n,n}
      slack=2 => {max(1,n-2), n+2}
    """
    if slack <= 0:
        return f"{{{n},{n}}}"
    lo = max(1, n - slack)
    hi = n + slack
    return f"{{{lo},{hi}}}"


def shape_block(tok: str, slack: int = 0) -> str:
    """
    纯形状块：尽量不引入长字面量。必要时保留 '-' 这种单字符分隔。

    修复点：
      - tok == "" 时直接返回 ""，避免生成 {0,0}（例如 "I-" split("-") 会产生空串段）。
    """
    if tok == "":
        return ""  # ✅ 修复 {0,0}

    if tok.isdigit():
        return rf"[0-9]{_q(len(tok), slack)}"

    if re.fullmatch(r"[0-9]+(st|nd|rd|th)", tok):
        m = re.match(r"([0-9]+)", tok)
        n = m.group(1) if m else ""
        # 数字部分可 slack，后缀保持原样（如需更抽象可改成 [a-z]{2,2}）
        return rf"[0-9]{_q(len(n), slack)}(st|nd|rd|th)"

    if re.fullmatch(r"[A-Z]+", tok):
        return rf"[A-Z]{_q(len(tok), slack)}"

    if re.fullmatch(r"[a-z]+", tok):
        return rf"[a-z]{_q(len(tok), slack)}"

    if re.fullmatch(r"[A-Za-z]+", tok):
        # 首字母大写常见词
        if tok[0].isupper() and tok[1:].islower():
            if len(tok) == 1:
                return r"[A-Z]"
            return rf"[A-Z][a-z]{_q(len(tok) - 1, slack)}"
        return rf"[A-Za-z]{_q(len(tok), slack)}"

    if "-" in tok and re.fullmatch(r"[A-Za-z0-9-]+", tok):
        parts = tok.split("-")
        return "-".join(shape_block(p, slack=slack) for p in parts)

    # fallback
    return rf"[A-Za-z0-9]{_q(len(tok), slack)}"


def anchor_token_pattern(tok: str, anchor2: str, mode: str, slack: int = 0) -> str:
    """
    在一个 token 内保留 2 字符字面量，剩余用形状补齐。
    mode: "pre" or "suf"
    """
    a = re.escape(anchor2)

    if len(tok) == 2 and tok.lower() == anchor2.lower():
        core = a
    else:
        if mode == "pre":
            rest = tok[2:]
            rest_pat = shape_block(rest, slack=slack) if rest else ""
            core = a + (rest_pat if rest_pat else "")
        else:
            rest = tok[:-2]
            rest_pat = shape_block(rest, slack=slack) if rest else ""
            core = (rest_pat if rest_pat else "") + a
    return core


def max_literal_run_len(regex: str) -> int:
    """
    近似计算：统计 regex 中“连续字面量字符”的最长长度。
    忽略：字符类[...]、转义序列 \s \d \b 等、量词 {m,n}、分组符号等。
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


def pick_anchors(tokens: List[str], pools: Pools, k=2) -> Optional[List[Tuple[int, str, str]]]:
    """
    返回 [(idx, anchor2, mode)], mode in {"pre","suf"}
    关键修改：最终 anchors 按 anchor2（忽略大小写）去重，避免 ["nd","nd"]。
    """
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

    # 位置分散：尽量不要都在前 20%
    early = [c for c in cand if c[0] <= int(0.2 * T)]
    mid = [c for c in cand if int(0.2 * T) < c[0] < int(0.8 * T)]
    late = [c for c in cand if c[0] >= int(0.8 * T)]

    picked = []
    if mid:
        picked.append(random.choice(mid))
    if late and len(picked) < k:
        picked.append(random.choice(late))
    if early and len(picked) < k:
        picked.append(random.choice(early))
    while len(picked) < k:
        picked.append(random.choice(cand))

    # 去重 idx
    uniq_by_idx = {}
    for x in picked:
        uniq_by_idx[x[0]] = x
    picked = list(uniq_by_idx.values())
    if len(picked) < k:
        return None

    picked = random.sample(picked, k)
    picked.sort(key=lambda x: x[0])

    # ✅ 按 anchor2 去重（忽略大小写）
    seen_a = set()
    uniq2 = []
    for idx, a2, mode in picked:
        key = a2.lower()
        if key in seen_a:
            continue
        seen_a.add(key)
        uniq2.append((idx, a2, mode))

    if len(uniq2) < k:
        return None

    return uniq2


def pick_middle_blocks(tokens: List[str], anchor_pos: set, min_blocks=2) -> Optional[List[int]]:
    T = len(tokens)
    L = int(0.3 * T)
    R = int(0.7 * T)
    mid_idx = [i for i in range(L, R + 1) if i not in anchor_pos and (not is_punct(tokens[i]))]
    if len(mid_idx) < min_blocks:
        return None
    return random.sample(mid_idx, min_blocks)


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

    parts = []
    for i, tok in enumerate(tokens):
        if is_punct(tok):
            if tok == ".":
                parts.append(r"\.")
            else:
                parts.append(re.escape(tok))
            continue

        if i in anchor_pos:
            _, a2, mode = next(x for x in anchors if x[0] == i)
            parts.append(anchor_token_pattern(tok, a2, mode, slack=len_slack))
        elif i in mid_pos:
            parts.append(shape_block(tok, slack=len_slack))
        else:
            if random.random() < replace_ratio:
                parts.append(shape_block(tok, slack=len_slack))
            else:
                # “半替换”：保留 1-2 字符 + 形状补齐（仍不产生长字面量）
                if len(tok) >= 2:
                    a2 = tok[:2]
                    parts.append(anchor_token_pattern(tok, a2, "pre", slack=len_slack))
                else:
                    parts.append(re.escape(tok))

    body = r"\s+".join(parts)
    if g > 0:
        return rf".{{0,{g}}}" + body
    return body


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
    """
    随机抽取 seed：优先 TABLESAMPLE；否则用随机 id 跳转；最后兜底 random().
    """
    if id_bounds is None:
        id_bounds = _get_id_bounds(conn)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        while True:
            row = None

            # 1) TABLESAMPLE（快）
            try:
                cur.execute("SELECT id, description FROM us_accidents TABLESAMPLE SYSTEM (0.01) LIMIT 1;")
                row = cur.fetchone()
            except Exception:
                row = None

            # 2) 随机 id 跳转（一般比 ORDER BY random() 快很多）
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

            # 3) 最后兜底
            if not row:
                cur.execute("SELECT id, description FROM us_accidents ORDER BY random() LIMIT 1;")
                row = cur.fetchone()

            s = row["description"]
            toks = tokenize(s)
            if len(toks) >= min_tokens:
                return int(row["id"]), s, toks


def verify_seed_pg(conn, seed: str, regex: str) -> bool:
    """
    必须匹配 seed 本身（PG 语义）。
    psycopg2 参数占位符必须用 %s，而不是 $1/$2。
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT %s::text ~ %s::text;", (seed, regex))
            return bool(cur.fetchone()[0])
    except (pg_errors.InvalidRegularExpression, pg_errors.SyntaxError, pg_errors.DataException):
        return False


def hits_estimate_pg(conn, a1: str, a2: str, regex: str, seed_id: int, cand_cap=2000) -> Tuple[int, int]:
    """
    anchors 用 ILIKE 粗过滤，cap 限制候选规模，再用 ~ 做 regex 复核。

    ✅ 关键修复：cand 必须包含 seed 行（否则 hits_est 可能误报 0）。
       使用 ORDER BY (id = seed_id) DESC 把 seed 排到 cand 的最前面。
    """
    sql = """
    WITH cand AS (
      SELECT id, description
      FROM us_accidents
      WHERE description ILIKE %s
        AND description ILIKE %s
      ORDER BY (id = %s) DESC, id
      LIMIT %s
    )
    SELECT
      (SELECT count(*) FROM cand) AS cand_est,
      (SELECT count(*) FROM cand WHERE description ~ %s) AS hits_est;
    """
    p1 = f"%{a1}%"
    p2 = f"%{a2}%"
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (p1, p2, seed_id, cand_cap, regex))
            cand_est, hits_est = cur.fetchone()
        return int(cand_est), int(hits_est)
    except (pg_errors.InvalidRegularExpression, pg_errors.SyntaxError, pg_errors.DataException):
        return 0, 0


def _merge_parts(final_out: str, part_paths: List[str], keep_parts: bool):
    out_dir = os.path.dirname(final_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(final_out, "w", encoding="utf-8") as w:
        for p in part_paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as r:
                for line in r:
                    w.write(line)

    if not keep_parts:
        for p in part_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _run_worker(args_dict: dict, worker_id: int, out_path: str, num_to_gen: int):
    """
    每个 worker 独立建立 PG 连接，独立生成 num_to_gen 条到 out_path。
    """
    args = argparse.Namespace(**args_dict)

    # 每个 worker 独立随机种子，降低重复
    base_seed = args.seed if args.seed != 0 else int(time.time())
    worker_seed = base_seed + worker_id * 1000003
    random.seed(worker_seed)

    pools = load_pools(args.pools)
    conn = psycopg2.connect(args.pg_dsn)
    id_bounds = _get_id_bounds(conn)

    reasons = defaultdict(int)

    # 多进程下 tqdm 会比较乱；给每个 worker 单独一条进度条（position=worker_id）
    pbar = tqdm(total=num_to_gen, desc=f"W{worker_id}", position=worker_id, leave=False)

    n_ok = 0
    with open(out_path, "w", encoding="utf-8") as w:
        while n_ok < num_to_gen:
            seed_id, seed, toks = fetch_random_seed(conn, min_tokens=6, id_bounds=id_bounds)

            anchors = pick_anchors(toks, pools, k=args.anchors)
            if not anchors:
                reasons["no_anchor"] += 1
                continue

            anchor_pos = {i for i, _, _ in anchors}
            mid_blocks = pick_middle_blocks(toks, anchor_pos, min_blocks=args.min_mid_blocks)
            if not mid_blocks:
                reasons["no_mid_blocks"] += 1
                continue

            slack_try_list = [0, 1, 2] if args.len_slack == 0 else [args.len_slack]
            best = None

            for slack in slack_try_list:
                regex = assemble_regex(
                    toks,
                    anchors,
                    mid_blocks,
                    replace_ratio=args.replace_ratio,
                    g=args.g,
                    len_slack=slack,
                )

                ml = max_literal_run_len(regex)
                if ml > 2:
                    reasons["ml_gt_2"] += 1
                    continue

                if not verify_seed_pg(conn, seed, regex):
                    reasons["seed_verify_fail"] += 1
                    continue

                a_list = [a for _, a, _ in anchors]
                cand_est, hits_est = hits_estimate_pg(
                    conn, a_list[0], a_list[1], regex, seed_id=seed_id, cand_cap=args.cand_cap
                )

                # ✅ 你要求的最低命中门槛
                if hits_est < 1:
                    reasons["hits_lt_1"] += 1
                    continue

                best = (regex, ml, cand_est, hits_est, slack)
                break

            if best is None:
                continue

            regex, ml, cand_est, hits_est, slack_used = best
            a_list = [a for _, a, _ in anchors]

            rec = {
                "dataset": "us_acc",
                "seed_id": int(seed_id),
                "seed_string": seed,
                "regex": regex,
                "anchors": a_list,
                "g": int(args.g),
                "middle_blocks": sorted(mid_blocks),
                "stats": {
                    "max_literal_run_len": int(ml),
                    "cand_est": int(cand_est),
                    "hits_est": int(hits_est),
                    "len_slack_used": int(slack_used),
                    "worker_id": int(worker_id),
                },
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            pbar.update(1)

    pbar.close()
    conn.close()

    # 每个 worker 写一个 reasons sidecar，方便你调参
    try:
        with open(out_path + ".reasons.json", "w", encoding="utf-8") as rf:
            json.dump(dict(reasons), rf, ensure_ascii=False, indent=2)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/lyz/wd/LLM_RegularSearch/v1_data/gen_workload/data/old_data/US_Accidents.csv")
    ap.add_argument("--mode", choices=["prep", "gen"], required=True)
    ap.add_argument("--pools", default="us_acc_pools.json")
    ap.add_argument("--out", default="us_acc_stage1.jsonl")
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--anchors", type=int, default=2)
    ap.add_argument("--min_mid_blocks", type=int, default=2)
    ap.add_argument("--replace_ratio", type=float, default=0.7)
    ap.add_argument("--g", type=int, default=0)
    ap.add_argument("--cand_cap", type=int, default=2000)
    ap.add_argument("--max_rows_df", type=int, default=0)  # 0=全量
    ap.add_argument("--pg_dsn", default="dbname=regex_db user=lyz host=localhost port=5000")

    ap.add_argument(
        "--len_slack",
        type=int,
        default=0,
        help="Length slack for shape blocks. 0 => exact {n,n}; if 0, generator adaptively tries slack=0/1/2 until hits_est>=1.",
    )

    # ✅ 并行参数
    ap.add_argument("--workers", type=int, default=1, help="Number of worker processes for --mode gen.")
    ap.add_argument("--seed", type=int, default=0, help="Base random seed (0 => use time-based seed).")
    ap.add_argument("--keep_parts", action="store_true", help="Keep shard part files after merging.")

    args = ap.parse_args()

    if args.mode == "prep":
        build_pools(args.csv, args.pools, max_rows=(args.max_rows_df or None))
        return

    # ---- gen mode ----
    if args.workers <= 1:
        # 单进程生成：复用 worker 逻辑
        args_dict = vars(args).copy()
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        _run_worker(args_dict, worker_id=0, out_path=args.out, num_to_gen=args.num)
        print(f"[OK] wrote {args.num} lines -> {args.out}")
        return

    # 多进程：主进程分片并合并
    workers = args.workers
    total = args.num
    base = total // workers
    rem = total % workers

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    part_paths = []
    procs = []
    args_dict = vars(args).copy()  # 可 picklable dict

    for wid in range(workers):
        n_i = base + (1 if wid < rem else 0)
        part_path = f"{args.out}.part{wid}.jsonl"
        part_paths.append(part_path)

        p = mp.Process(target=_run_worker, args=(args_dict, wid, part_path, n_i))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # 合并 shards
    _merge_parts(args.out, part_paths, keep_parts=args.keep_parts)
    print(f"[OK] merged {workers} parts -> {args.out}")
    if args.keep_parts:
        print("[Info] kept part files:", ", ".join(part_paths))


if __name__ == "__main__":
    main()
