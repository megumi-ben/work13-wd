#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple


def text_bigrams(s: str):
    if not s:
        return []
    return [s[i:i + 2] for i in range(len(s) - 1)]


def build_bitstring_for_text(text: str, mapping: Dict[str, int], k: int):
    # k 一般是 64
    bits = ["0"] * k
    present = set(text_bigrams(text))
    # mapping: bigram -> pos
    for bg, pos in mapping.items():
        if bg in present:
            bits[pos] = "1"
    return "".join(bits)


def load_mapping(path: str) -> Tuple[int, Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    k = int(obj["k"])
    mapping = obj["mapping"]
    return k, mapping


def detect_header(first_row: List[str]) -> bool:
    # 非严格启发：如果第一行包含明显字段名，就认为有 header
    joined = ",".join(first_row).lower()
    return ("id" in joined and "text" in joined) or ("text_content" in joined) or ("author" in joined) or ("primaryname" in joined)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--mapping_dir", required=True)
    ap.add_argument("--dataset_key", required=True,
                    help="prefix of mapping files, e.g., wiki / reddit / lineitem / imdb_name")

    ap.add_argument("--workloads", default="ABCD",
                    help="which workload letters to load, default ABCD")
    ap.add_argument("--k", type=int, default=64)

    ap.add_argument("--output_csv", required=True)

    ap.add_argument("--id_col", default="id")
    ap.add_argument("--text_col", default="text_content")

    ap.add_argument("--header", choices=["auto", "yes", "no"], default="auto",
                    help="whether input has header")
    ap.add_argument("--gen_id_if_missing", action="store_true",
                    help="if id_col not found, generate id by row_number starting from 1")

    args = ap.parse_args()

    # 1) 载入所有 workload 的 mapping
    workload_list = list(args.workloads.strip())
    maps = {}  # w -> (k, mapping)
    for w in workload_list:
        mp = os.path.join(args.mapping_dir, f"{args.dataset_key}_{w}_topk_{args.k}.json")
        if not os.path.exists(mp):
            raise FileNotFoundError(f"mapping not found: {mp}")
        k_m, mapping = load_mapping(mp)
        if k_m != args.k:
            raise ValueError(f"k mismatch: args.k={args.k} but {mp} has k={k_m}")
        maps[w] = mapping

    # 2) 读 input
    with open(args.input_csv, "r", encoding="utf-8") as fin:
        reader_raw = csv.reader(fin)
        first = next(reader_raw, None)
        if first is None:
            raise ValueError("empty input_csv")

        has_header = False
        if args.header == "yes":
            has_header = True
        elif args.header == "no":
            has_header = False
        else:
            has_header = detect_header(first)

        # 重新构造 DictReader
        fin.seek(0)
        if has_header:
            reader = csv.DictReader(fin)
        else:
            # 无 header 情况：
            # 你默认 dump 应该是 id,text_content，
            # 如果不是，你可以用 --gen_id_if_missing 来兜底
            fieldnames = [args.id_col, args.text_col]
            reader = csv.DictReader(fin, fieldnames=fieldnames)

        # 3) 写 output（宽表：A/B/C/D 多签名列）
        out_fields = [args.id_col, args.text_col] + [
            f"sig_rei_{w}_{args.k}" for w in workload_list
        ]

        with open(args.output_csv, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields)
            writer.writeheader()

            cnt = 0
            auto_id = 1

            for row in reader:
                # 取 id
                _id = row.get(args.id_col, "")
                if (not _id) and args.gen_id_if_missing:
                    _id = str(auto_id)

                # 取 text
                txt = row.get(args.text_col, "")

                # 如果 header=auto 且误判导致第一行是 header 字段名，也能在这里被自然跳过/覆盖
                if (not txt) and (not _id) and not args.gen_id_if_missing:
                    continue

                present_out = {
                    args.id_col: _id if _id is not None else "",
                    args.text_col: txt if txt is not None else "",
                }

                # 4) 一次性为 A/B/C/D 生成多个签名
                for w in workload_list:
                    bitstring = build_bitstring_for_text(txt, maps[w], args.k)
                    present_out[f"sig_rei_{w}_{args.k}"] = bitstring

                writer.writerow(present_out)

                cnt += 1
                auto_id += 1

    print(f"[OK] wrote {cnt} rows to {args.output_csv}")
    print(f"[INFO] workloads={workload_list}, k={args.k}, header={has_header}")


if __name__ == "__main__":
    main()
