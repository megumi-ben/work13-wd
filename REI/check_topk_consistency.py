#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set


FNAME_RE = re.compile(r"^(?P<dataset>.+)_(?P<wl>[ABCD])_topk_(?P<k>\d+)\.json$")


def load_mapping(fp: str) -> Dict[str, int]:
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)
    mapping = obj.get("mapping", {})
    if not isinstance(mapping, dict):
        raise ValueError(f"Bad mapping format in {fp}")
    # keys are bigrams, values are bit positions
    return mapping


def parse_filename(name: str):
    m = FNAME_RE.match(name)
    if not m:
        return None
    return m.group("dataset"), m.group("wl"), int(m.group("k"))


def compare_workloads(files_by_wl: Dict[str, str], check_order: bool):
    """
    Returns:
      same_set: bool
      same_order: bool (only meaningful if check_order=True)
      details: dict with diffs
    """
    wls = sorted(files_by_wl.keys())
    if not wls:
        return True, True, {}

    base_wl = wls[0]
    base_fp = files_by_wl[base_wl]
    base_map = load_mapping(base_fp)
    base_set = set(base_map.keys())

    same_set = True
    same_order = True

    diffs = {}

    for wl in wls[1:]:
        fp = files_by_wl[wl]
        mp = load_mapping(fp)
        s = set(mp.keys())

        only_in_base = sorted(base_set - s)
        only_in_curr = sorted(s - base_set)

        if only_in_base or only_in_curr:
            same_set = False
            diffs[wl] = {
                "only_in_base": only_in_base,
                "only_in_" + wl: only_in_curr,
            }

        if check_order:
            # order equivalence means identical bigram->bitpos mapping
            if mp != base_map:
                same_order = False
                # also provide a small hint of mismatched items
                mismatch = []
                # collect up to 20 mismatches
                for bg in sorted(base_set & s):
                    if base_map.get(bg) != mp.get(bg):
                        mismatch.append((bg, base_map.get(bg), mp.get(bg)))
                        if len(mismatch) >= 20:
                            break
                diffs.setdefault(wl, {})
                diffs[wl]["order_mismatch_sample(bg, base_pos, wl_pos)"] = mismatch

    return same_set, same_order, diffs


def main():
    ap = argparse.ArgumentParser(
        description="Check whether top-k bigram mappings are identical across workloads per dataset."
    )
    ap.add_argument(
        "--dir",
        default="./data/mappings",
        help="Directory containing *_A_topk_*.json etc.",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=64,
        help="Only check files with this k (default 64).",
    )
    ap.add_argument(
        "--check_order",
        action="store_true",
        help="Additionally require identical bigram->bitpos order across workloads.",
    )
    ap.add_argument(
        "--strict_abcd",
        action="store_true",
        help="Require all A/B/C/D to exist for each dataset; otherwise mark as incomplete.",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError(f"Directory not found: {args.dir}")

    groups: Dict[str, Dict[str, str]] = defaultdict(dict)

    for name in os.listdir(args.dir):
        parsed = parse_filename(name)
        if not parsed:
            continue
        dataset, wl, k = parsed
        if k != args.k:
            continue
        fp = os.path.join(args.dir, name)
        groups[dataset][wl] = fp

    if not groups:
        print(f"[WARN] No files matched pattern in {args.dir} with k={args.k}")
        return

    all_wls = ["A", "B", "C", "D"]

    print(f"[INFO] Scanning dir: {args.dir}")
    print(f"[INFO] Target k: {args.k}")
    print(f"[INFO] Check order: {bool(args.check_order)}")
    print("")

    any_diff = False

    for dataset in sorted(groups.keys()):
        files_by_wl = groups[dataset]
        present = sorted(files_by_wl.keys())

        incomplete = False
        if args.strict_abcd and any(wl not in files_by_wl for wl in all_wls):
            incomplete = True

        same_set, same_order, diffs = compare_workloads(files_by_wl, args.check_order)

        status_parts = []
        status_parts.append("SET=OK" if same_set else "SET=DIFF")
        if args.check_order:
            status_parts.append("ORDER=OK" if same_order else "ORDER=DIFF")
        if incomplete:
            status_parts.append("INCOMPLETE")

        status = ", ".join(status_parts)

        print(f"== Dataset: {dataset} ==")
        print(f"Workloads present: {present}")
        print(f"Result: {status}")

        if not same_set or (args.check_order and not same_order) or incomplete:
            any_diff = True

        # Print concise diffs
        if diffs:
            for wl, info in diffs.items():
                print(f"  - Diff vs base for workload {wl}:")
                if "only_in_base" in info:
                    ob = info["only_in_base"]
                    oc = info.get("only_in_" + wl, [])
                    print(f"      only_in_base_count: {len(ob)}")
                    print(f"      only_in_{wl}_count: {len(oc)}")
                    # show a small sample for readability
                    if ob:
                        print(f"      only_in_base_sample: {ob[:10]}")
                    if oc:
                        print(f"      only_in_{wl}_sample: {oc[:10]}")
                if "order_mismatch_sample(bg, base_pos, wl_pos)" in info:
                    mm = info["order_mismatch_sample(bg, base_pos, wl_pos)"]
                    print(f"      order_mismatch_sample_count(shown): {len(mm)}")
                    if mm:
                        print(f"      order_mismatch_sample: {mm}")
        print("")

    if not any_diff:
        print("[SUMMARY] All datasets have identical top-k bigram sets across workloads.")
        if args.check_order:
            print("[SUMMARY] And their bigram->bitpos orders are also identical.")
        print("[SUGGESTION] You can safely reuse a single mapping per dataset for A/B/C/D.")
    else:
        print("[SUMMARY] Some datasets/workloads differ.")
        print("[SUGGESTION] Consider keeping separate mappings for those that show SET=DIFF or ORDER=DIFF.")


if __name__ == "__main__":
    main()
