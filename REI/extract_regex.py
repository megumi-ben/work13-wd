#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        default="/home/lyz/wd/LLM_RegularSearch/v2_data/workload/imdb_name",
        help="*.jsonl 所在目录"
    )
    ap.add_argument(
        "--output_dir",
        default="/home/lyz/wd/LLM_RegularSearch/REI/workload/imdb_name",
        help="导出的 txt 保存目录"
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pattern = os.path.join(args.input_dir, "*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] no jsonl found under {args.input_dir}")
        return

    total_files = 0
    total_lines = 0
    total_regex = 0

    for in_path in files:
        base = os.path.basename(in_path)               # xxx.jsonl
        stem, _ = os.path.splitext(base)              # xxx
        out_path = os.path.join(args.output_dir, f"{stem}.txt")

        print(f"[INFO] {in_path} -> {out_path}")

        written = 0
        bad = 0

        with open(in_path, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue

                regex = obj.get("regex")
                if not regex:
                    continue

                fout.write(regex + "\n")
                written += 1

        print(f"[OK] wrote {written} regex (bad_json={bad}) to {out_path}")
        total_files += 1
        total_regex += written

    print(f"[DONE] files={total_files}, total_lines={total_lines}, total_regex={total_regex}")


if __name__ == "__main__":
    main()
