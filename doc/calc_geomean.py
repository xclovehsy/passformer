#!/usr/bin/env python3
"""读取 doc 目录下所有 benchmark_results CSV 文件，计算 reward 列的几何均值并写入最后一行。"""

import csv
import math
import glob
import os

def calc_geomean(rewards):
    log_sum = sum(math.log(r) for r in rewards)
    return math.exp(log_sum / len(rewards))

def process_csv(filepath):
    rows = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or not row[0] or row[0].startswith("geomean"):
                continue
            rows.append(row)

    rewards = [float(row[1]) for row in rows]
    geomean = calc_geomean(rewards)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
        writer.writerow(["geomean", geomean, "", ""])

    print(f"{os.path.basename(filepath)}: geomean = {geomean}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = sorted(glob.glob(os.path.join(script_dir, "benchmark_results*.csv")))

    if not csv_files:
        print("未找到 benchmark_results*.csv 文件")
        return

    for f in csv_files:
        process_csv(f)

if __name__ == "__main__":
    main()
