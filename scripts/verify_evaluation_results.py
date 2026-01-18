#!/usr/bin/env python3
"""
使用 CompilerGym 验证评估结果的脚本

验证内容：
1. IrInstructionCount (原始IR指令数)
2. IrInstructionCountO3 (O3优化后的IR指令数)
3. IrInstructionCountOz (Oz优化后的IR指令数)
4. 执行predicted_passes后的优化结果
5. relative_to_o3 计算是否正确
"""

import argparse
import csv
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

import compiler_gym


@dataclass
class VerificationResult:
    """单个benchmark的验证结果"""
    benchmark: str
    # CSV中的值
    csv_original_ir: int
    csv_optimized_ir: int
    csv_o3_ir: int
    csv_relative_to_o3: float
    # CompilerGym验证值
    cg_original_ir: int
    cg_o3_ir: int
    cg_oz_ir: int
    cg_optimized_ir: int
    cg_relative_to_o3: float
    # 验证结果
    original_match: bool
    o3_match: bool
    optimized_match: bool
    relative_match: bool
    success: bool
    error_message: str = ""


def parse_benchmark_uri(benchmark_name: str) -> str:
    """
    将CSV中的benchmark名称转换为CompilerGym的benchmark URI
    
    例如: benchmark_cbench-v1_adpcm -> benchmark://cbench-v1/adpcm
    """
    # 移除 "benchmark_" 前缀（如果存在）
    if benchmark_name.startswith("benchmark_"):
        benchmark_name = benchmark_name[len("benchmark_"):]
    
    # 格式: dataset_name -> benchmark://dataset/name
    parts = benchmark_name.split("_", 1)
    if len(parts) == 2:
        dataset, name = parts
        return f"benchmark://{dataset}/{name}"
    else:
        # 假设是完整的benchmark名称
        return benchmark_name


def parse_passes(passes_str: str) -> List[str]:
    """解析优化pass序列字符串"""
    if not passes_str or passes_str.strip() == "":
        return []
    return passes_str.strip().split()


def verify_single_benchmark(
    env,
    benchmark_uri: str,
    predicted_passes: List[str],
    csv_original_ir: int,
    csv_optimized_ir: int,
    csv_o3_ir: int,
    csv_relative_to_o3: float,
    tolerance: float = 0.001
) -> VerificationResult:
    """
    验证单个benchmark的结果
    
    Args:
        env: CompilerGym环境
        benchmark_uri: benchmark的URI
        predicted_passes: 预测的优化pass列表
        csv_original_ir: CSV中的原始IR指令数
        csv_optimized_ir: CSV中的优化后IR指令数
        csv_o3_ir: CSV中的O3优化后IR指令数
        csv_relative_to_o3: CSV中的relative_to_o3值
        tolerance: 浮点数比较的容差
    
    Returns:
        VerificationResult
    """
    benchmark_name = benchmark_uri.split("/")[-1] if "/" in benchmark_uri else benchmark_uri
    
    try:
        # Step 1: 重置环境到指定benchmark
        env.reset(benchmark=benchmark_uri)
        
        # Step 2: 获取初始观测值
        cg_original_ir = env.observation["IrInstructionCount"]
        cg_o3_ir = env.observation["IrInstructionCountO3"]
        cg_oz_ir = env.observation["IrInstructionCountOz"]
        
        # Step 3: 执行predicted_passes
        for pass_name in predicted_passes:
            try:
                action_idx = env.action_space.flags.index(pass_name)
                env.step(action_idx)
            except ValueError:
                # 如果pass不存在，跳过
                print(f"  警告: Pass '{pass_name}' 不在action space中，跳过")
                continue
            except Exception as e:
                print(f"  警告: 执行pass '{pass_name}'时出错: {e}")
                continue
        
        # Step 4: 获取优化后的IR指令数
        cg_optimized_ir = env.observation["IrInstructionCount"]
        
        # Step 5: 计算relative_to_o3
        # 公式: (original - optimized) / (original - o3)
        o3_reduction = cg_original_ir - cg_o3_ir
        if o3_reduction > 0:
            cg_relative_to_o3 = (cg_original_ir - cg_optimized_ir) / o3_reduction
        else:
            # 如果O3没有带来改进，设为1.0
            cg_relative_to_o3 = 1.0
        
        # Step 6: 比较结果
        original_match = (csv_original_ir == cg_original_ir)
        o3_match = (csv_o3_ir == cg_o3_ir)
        optimized_match = (csv_optimized_ir == cg_optimized_ir)
        
        # 对于relative_to_o3，使用容差比较
        relative_match = math.isclose(csv_relative_to_o3, cg_relative_to_o3, rel_tol=tolerance, abs_tol=tolerance)
        
        return VerificationResult(
            benchmark=benchmark_name,
            csv_original_ir=csv_original_ir,
            csv_optimized_ir=csv_optimized_ir,
            csv_o3_ir=csv_o3_ir,
            csv_relative_to_o3=csv_relative_to_o3,
            cg_original_ir=cg_original_ir,
            cg_o3_ir=cg_o3_ir,
            cg_oz_ir=cg_oz_ir,
            cg_optimized_ir=cg_optimized_ir,
            cg_relative_to_o3=cg_relative_to_o3,
            original_match=original_match,
            o3_match=o3_match,
            optimized_match=optimized_match,
            relative_match=relative_match,
            success=True
        )
        
    except Exception as e:
        return VerificationResult(
            benchmark=benchmark_name,
            csv_original_ir=csv_original_ir,
            csv_optimized_ir=csv_optimized_ir,
            csv_o3_ir=csv_o3_ir,
            csv_relative_to_o3=csv_relative_to_o3,
            cg_original_ir=0,
            cg_o3_ir=0,
            cg_oz_ir=0,
            cg_optimized_ir=0,
            cg_relative_to_o3=0.0,
            original_match=False,
            o3_match=False,
            optimized_match=False,
            relative_match=False,
            success=False,
            error_message=str(e)
        )


def read_csv_results(csv_path: str) -> List[dict]:
    """读取CSV评估结果"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('benchmark'):
                continue
            results.append({
                'benchmark': row['benchmark'],
                'original_ir': int(row['original_ir']),
                'optimized_ir': int(row['optimized_ir']),
                'o3_ir': int(row['o3_ir']),
                'relative_to_o3': float(row['relative_to_o3']),
                'predicted_passes': row.get('predicted_passes', '')
            })
    return results


def print_verification_result(result: VerificationResult, verbose: bool = True):
    """打印单个验证结果"""
    status = "✓" if result.success else "✗"
    print(f"\n{status} Benchmark: {result.benchmark}")
    
    if not result.success:
        print(f"  错误: {result.error_message}")
        return
    
    if verbose:
        # 原始IR指令数
        match_str = "✓" if result.original_match else "✗"
        print(f"  {match_str} Original IR: CSV={result.csv_original_ir}, CG={result.cg_original_ir}")
        
        # O3优化后IR指令数
        match_str = "✓" if result.o3_match else "✗"
        print(f"  {match_str} O3 IR: CSV={result.csv_o3_ir}, CG={result.cg_o3_ir}")
        
        # Oz优化后IR指令数 (仅显示CG值，CSV中可能没有)
        print(f"  - Oz IR: CG={result.cg_oz_ir}")
        
        # 优化后IR指令数
        match_str = "✓" if result.optimized_match else "✗"
        print(f"  {match_str} Optimized IR: CSV={result.csv_optimized_ir}, CG={result.cg_optimized_ir}")
        
        # Relative to O3
        match_str = "✓" if result.relative_match else "✗"
        print(f"  {match_str} Relative to O3: CSV={result.csv_relative_to_o3:.4f}, CG={result.cg_relative_to_o3:.4f}")
    else:
        # 简洁模式：只显示不匹配的项
        mismatches = []
        if not result.original_match:
            mismatches.append(f"Original(CSV={result.csv_original_ir}, CG={result.cg_original_ir})")
        if not result.o3_match:
            mismatches.append(f"O3(CSV={result.csv_o3_ir}, CG={result.cg_o3_ir})")
        if not result.optimized_match:
            mismatches.append(f"Optimized(CSV={result.csv_optimized_ir}, CG={result.cg_optimized_ir})")
        if not result.relative_match:
            mismatches.append(f"Relative(CSV={result.csv_relative_to_o3:.4f}, CG={result.cg_relative_to_o3:.4f})")
        
        if mismatches:
            print(f"  不匹配: {', '.join(mismatches)}")
        else:
            print(f"  全部匹配")


def save_verification_results(results: List[VerificationResult], output_path: str):
    """保存验证结果到CSV"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'benchmark',
            'csv_original_ir', 'cg_original_ir', 'original_match',
            'csv_o3_ir', 'cg_o3_ir', 'o3_match',
            'cg_oz_ir',
            'csv_optimized_ir', 'cg_optimized_ir', 'optimized_match',
            'csv_relative_to_o3', 'cg_relative_to_o3', 'relative_match',
            'success', 'error_message'
        ])
        for r in results:
            writer.writerow([
                r.benchmark,
                r.csv_original_ir, r.cg_original_ir, r.original_match,
                r.csv_o3_ir, r.cg_o3_ir, r.o3_match,
                r.cg_oz_ir,
                r.csv_optimized_ir, r.cg_optimized_ir, r.optimized_match,
                f"{r.csv_relative_to_o3:.4f}", f"{r.cg_relative_to_o3:.4f}", r.relative_match,
                r.success, r.error_message
            ])
    print(f"\n验证结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="使用CompilerGym验证评估结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python verify_evaluation_results.py --csv evaluation_results.csv
  python verify_evaluation_results.py --csv evaluation_results.csv --output verification_results.csv
  python verify_evaluation_results.py --csv evaluation_results.csv --verbose
""")
    parser.add_argument(
        "--csv", "-c", type=str, required=True,
        help="评估结果CSV文件路径"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="验证结果输出CSV文件路径（可选）"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="显示详细信息"
    )
    parser.add_argument(
        "--tolerance", "-t", type=float, default=0.01,
        help="浮点数比较容差 (默认: 0.01)"
    )
    parser.add_argument(
        "--benchmarks", "-b", type=str, nargs="*", default=None,
        help="指定要验证的benchmark名称列表（可选，不指定则验证全部）"
    )
    args = parser.parse_args()
    
    # 检查CSV文件存在
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"错误: CSV文件不存在: {csv_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("CompilerGym 评估结果验证")
    print("=" * 70)
    print(f"CSV文件: {csv_path}")
    print(f"容差: {args.tolerance}")
    
    # 读取CSV结果
    print("\n正在读取CSV文件...")
    csv_results = read_csv_results(str(csv_path))
    print(f"读取到 {len(csv_results)} 条记录")
    
    # 过滤benchmark（如果指定）
    if args.benchmarks:
        filtered = []
        for r in csv_results:
            name = r['benchmark']
            short_name = name.split("_")[-1] if "_" in name else name
            if name in args.benchmarks or short_name in args.benchmarks:
                filtered.append(r)
        csv_results = filtered
        print(f"过滤后剩余 {len(csv_results)} 条记录")
    
    if not csv_results:
        print("没有要验证的记录！")
        sys.exit(0)
    
    # 创建CompilerGym环境
    print("\n正在初始化CompilerGym环境...")
    env = compiler_gym.make("llvm-v0")
    
    # 验证每个benchmark
    print("\n开始验证...")
    verification_results = []
    
    for i, csv_row in enumerate(csv_results, 1):
        benchmark_name = csv_row['benchmark']
        benchmark_uri = parse_benchmark_uri(benchmark_name)
        predicted_passes = parse_passes(csv_row['predicted_passes'])
        
        print(f"\n[{i}/{len(csv_results)}] 验证: {benchmark_name}")
        print(f"  URI: {benchmark_uri}")
        print(f"  Passes数量: {len(predicted_passes)}")
        
        result = verify_single_benchmark(
            env=env,
            benchmark_uri=benchmark_uri,
            predicted_passes=predicted_passes,
            csv_original_ir=csv_row['original_ir'],
            csv_optimized_ir=csv_row['optimized_ir'],
            csv_o3_ir=csv_row['o3_ir'],
            csv_relative_to_o3=csv_row['relative_to_o3'],
            tolerance=args.tolerance
        )
        
        verification_results.append(result)
        print_verification_result(result, verbose=args.verbose)
    
    # 关闭环境
    env.close()
    
    # 统计
    print("\n" + "=" * 70)
    print("验证汇总")
    print("=" * 70)
    
    total = len(verification_results)
    success = sum(1 for r in verification_results if r.success)
    original_match = sum(1 for r in verification_results if r.success and r.original_match)
    o3_match = sum(1 for r in verification_results if r.success and r.o3_match)
    optimized_match = sum(1 for r in verification_results if r.success and r.optimized_match)
    relative_match = sum(1 for r in verification_results if r.success and r.relative_match)
    all_match = sum(1 for r in verification_results if r.success and r.original_match and r.o3_match and r.optimized_match and r.relative_match)
    
    print(f"总数: {total}")
    print(f"成功执行: {success}/{total} ({100*success/total:.1f}%)")
    if success > 0:
        print(f"Original IR 匹配: {original_match}/{success} ({100*original_match/success:.1f}%)")
        print(f"O3 IR 匹配: {o3_match}/{success} ({100*o3_match/success:.1f}%)")
        print(f"Optimized IR 匹配: {optimized_match}/{success} ({100*optimized_match/success:.1f}%)")
        print(f"Relative to O3 匹配: {relative_match}/{success} ({100*relative_match/success:.1f}%)")
        print(f"全部匹配: {all_match}/{success} ({100*all_match/success:.1f}%)")
    
    # 保存结果
    if args.output:
        save_verification_results(verification_results, args.output)
    else:
        # 默认保存到CSV文件同目录
        output_path = csv_path.parent / f"{csv_path.stem}_verification.csv"
        save_verification_results(verification_results, str(output_path))
    
    print("\n验证完成!")


if __name__ == "__main__":
    main()


