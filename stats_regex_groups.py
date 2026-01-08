#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from collections import Counter

# -----------------------------
# regex_norm: 仅做“轻量归一化”
# 目的：判断是不是“归一化过猛”导致过度归并
# 你可以通过 --norm_mode 选择不同强度
# -----------------------------
_WS_STAR = re.compile(r"\\s\*")   # matches literal \s*
_WS_PLUS = re.compile(r"\\s\+")   # matches literal \s+
_WS_RUN  = re.compile(r"(\\s\+)+")  # collapse repeated \s+

def regex_norm(rx: str, mode: str = "ws_plus") -> str:
    """
    mode:
      - "none"    : 不做任何归一化（等价于 raw）
      - "ws_plus" : 把 \s* 和 \s+ 都归一成 \s+，并压缩重复的 \s+
      - "ws_star" : 把 \s* 和 \s+ 都归一成 \s*，并压缩重复的 \s*
    """
    if mode == "none":
        return rx

    if mode == "ws_plus":
        rx = _WS_STAR.sub(r"\\s+", rx)
        rx = _WS_PLUS.sub(r"\\s+", rx)
        rx = _WS_RUN.sub(r"\\s+", rx)
        return rx

    if mode == "ws_star":
        # 将 \s+ 统一成 \s*（注意：这会更“粗”，通常归并会更猛）
        rx = _WS_PLUS.sub(r"\\s*", rx)
        rx = _WS_STAR.sub(r"\\s*", rx)
        # 压缩重复 \s*
        rx = re.sub(r"(\\s\*)+", r"\\s*", rx)
        return rx

    raise ValueError(f"Unknown norm_mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input jsonl, e.g. ./workload/us_acc_stage1_10000.jsonl")
    ap.add_argument("--norm_mode", choices=["none", "ws_plus", "ws_star"], default="ws_plus",
                    help="how to compute regex_norm")
    ap.add_argument("--topk", type=int, default=0, help="optional: show top-k most frequent regex_norm groups")
    args = ap.parse_args()

    n = 0

    uniq_raw = set()
    uniq_norm = set()
    uniq_norm_anchors = set()
    uniq_norm_anchors_mid = set()

    # 可选：看归并后分布（哪个 regex_norm 占比最高）
    cnt_norm = Counter()

    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            rx = rec["regex"]
            anchors = tuple(rec.get("anchors", []))
            mid = rec.get("middle_blocks", [])
            # 你生成时 middle_blocks 已经是 sorted(mid_blocks) 了，但这里再保险一次
            mid = tuple(sorted(mid))

            rxn = regex_norm(rx, mode=args.norm_mode)

            n += 1
            uniq_raw.add(rx)
            uniq_norm.add(rxn)
            uniq_norm_anchors.add((rxn, anchors))
            uniq_norm_anchors_mid.add((rxn, anchors, mid))

            cnt_norm[rxn] += 1

    def ratio(a, b):
        return (a / b) if b else 0.0

    print(f"[INPUT] lines = {n}")
    print(f"[1] #unique raw regex                     = {len(uniq_raw)}   (ratio={ratio(len(uniq_raw), n):.4f})")
    print(f"[2] #unique regex_norm (mode={args.norm_mode})   = {len(uniq_norm)}  (ratio={ratio(len(uniq_norm), n):.4f})")
    print(f"[3] #unique (regex_norm, anchors)          = {len(uniq_norm_anchors)}  (ratio={ratio(len(uniq_norm_anchors), n):.4f})")
    print(f"[4] #unique (regex_norm, anchors, mid)     = {len(uniq_norm_anchors_mid)}  (ratio={ratio(len(uniq_norm_anchors_mid), n):.4f})")

    if args.topk and args.topk > 0:
        print("\n[TOP regex_norm groups]")
        for rxn, c in cnt_norm.most_common(args.topk):
            print(f"count={c:6d}  frac={c/n:.4%}  regex_norm={rxn}")

if __name__ == "__main__":
    main()

