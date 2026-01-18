"""检查 GA 搜索结果 CSV 的统计信息"""
import pandas as pd
from pathlib import Path
from collections import Counter

input_path = Path("/home/xucong24/CompilerOptimizationByTransformer/data")

df = pd.read_csv(input_path / 'results.csv', index_col=0)
print(f"总行数: {len(df)}")
print(f"列名: {df.columns.tolist()}")

# 检查 commandline 的唯一性
commandlines = df['commandline'].tolist()
unique_commandlines = set(commandlines)
print(f"\n唯一 commandline 数: {len(unique_commandlines)}")
print(f"重复率: {100 * (1 - len(unique_commandlines) / len(commandlines)):.1f}%")

# 每个唯一 commandline 出现的次数
cmdline_counts = Counter(commandlines)
print(f"\n最常见的 5 个 commandline 及其出现次数:")
for cmdline, count in cmdline_counts.most_common(5):
    pct = 100 * count / len(commandlines)
    print(f"  出现 {count} 次 ({pct:.1f}%): {cmdline[:80]}...")

# 检查 benchmark 的唯一性
benchmarks = df['benchmark'].tolist()
unique_benchmarks = set(benchmarks)
print(f"\n唯一 benchmark 数: {len(unique_benchmarks)}")

# 每个 benchmark 有多少条记录
benchmark_counts = Counter(benchmarks)
print(f"每个 benchmark 的记录数统计:")
print(f"  平均: {len(df) / len(unique_benchmarks):.1f}")
print(f"  最多: {max(benchmark_counts.values())}")
print(f"  最少: {min(benchmark_counts.values())}")

