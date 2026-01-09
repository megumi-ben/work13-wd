#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始 US_Accidents_March23.csv 路径")
    parser.add_argument("--all_output", required=True, help="输出：所有 Description 的 CSV")
    parser.add_argument("--sample_output", required=True, help="输出：随机抽样的 Description CSV")
    parser.add_argument("--sample_size", type=int, default=10000, help="抽样数量，默认 10000")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    args = parser.parse_args()

    random.seed(args.seed)

    # 打开输入和“全部描述”输出文件
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.all_output, "w", encoding="utf-8", newline="") as fall:

        reader = csv.DictReader(fin)
        all_writer = csv.DictWriter(fall, fieldnames=["Description"])
        all_writer.writeheader()

        # 蓄水池采样容器
        k = args.sample_size
        reservoir = []

        for i, row in enumerate(reader):
            desc = row.get("Description", "")

            # 写入“所有 Description”文件
            all_writer.writerow({"Description": desc})

            # 蓄水池采样逻辑（单遍 O(N)）
            if i < k:
                reservoir.append(desc)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = desc

    # 循环结束后，把 reservoir 写入“抽样文件”
    with open(args.sample_output, "w", encoding="utf-8", newline="") as fsample:
        writer = csv.DictWriter(fsample, fieldnames=["Description"])
        writer.writeheader()
        for desc in reservoir:
            writer.writerow({"Description": desc})


if __name__ == "__main__":
    main()
