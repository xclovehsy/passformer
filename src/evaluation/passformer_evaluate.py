"""Passformer 模型评估脚本

评估流程：
1. 加载 bc 文件并转换为 LLVM IR
2. 使用模型推理生成优化 pass 序列
3. 应用优化 pass 序列并计算 IR 指令缩减比例
4. 与 -O3 基线对比
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
from tqdm import tqdm

from src.config import load_config
from src.utils.llvm import (
    bitcode_to_llvm_ir,
    count_llvm_ir_instructions,
    optimize_llvm_ir,
    get_llvm_utils
)
from src.inference.optseq_gen_inference import OptSeqGenInference


@dataclass
class BenchmarkResult:
    """单个 benchmark 的评估结果"""
    benchmark_name: str
    original_ir_count: int
    optimized_ir_count: int
    o3_ir_count: int
    predicted_passes: str
    reduction_ratio: float          # (original - optimized) / original
    o3_reduction_ratio: float       # (original - o3) / original
    relative_improvement: float     # (original - optimized) / (original - o3)，>1 表示比 O3 好
    success: bool
    error_message: str = ""


@dataclass
class EvaluationResult:
    """整体评估结果"""
    benchmark_results: List[BenchmarkResult]
    avg_reduction_ratio: float
    avg_o3_reduction_ratio: float
    avg_relative_improvement: float
    success_rate: float
    model_path: str
    llvm_path: str
    benchmark_dir: str
    timestamp: str


def parse_args():
    parser = argparse.ArgumentParser(description="Passformer 模型评估")
    parser.add_argument(
        "--config", type=str, default=None,
        help="评估配置文件路径 (可选，也可通过命令行参数指定)"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="训练好的模型路径 (final_model 目录)"
    )
    parser.add_argument(
        "--benchmark_dir", type=str, default=None,
        help="benchmark 目录路径，包含 .bc 文件"
    )
    parser.add_argument(
        "--benchmarks", type=str, nargs="*", default=None,
        help="指定要评估的 benchmark 名称列表，不指定则评估全部"
    )
    parser.add_argument(
        "--llvm_path", type=str, 
        default="/home/xucong24/.local/share/compiler_gym/llvm-v0/bin",
        help="LLVM 工具链路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="评估结果输出目录"
    )
    parser.add_argument(
        "--max_input_length", type=int, default=1024,
        help="模型输入最大长度"
    )
    parser.add_argument(
        "--max_output_length", type=int, default=256,
        help="模型输出最大长度"
    )
    parser.add_argument(
        "--num_beams", type=int, default=4,
        help="beam search 的 beam 数量"
    )
    parser.add_argument(
        "--device", type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备"
    )
    parser.add_argument(
        "--encoder_tokenizer_type", type=str, default="inst2vec",
        choices=["auto", "inst2vec"],
        help="encoder tokenizer 类型"
    )
    return parser.parse_args()


def discover_benchmarks(benchmark_dir: str, benchmarks: Optional[List[str]] = None) -> List[Path]:
    """发现 benchmark 目录中的 .bc 文件
    
    Args:
        benchmark_dir: benchmark 目录路径
        benchmarks: 指定的 benchmark 名称列表，None 表示全部
        
    Returns:
        .bc 文件路径列表
    """
    benchmark_path = Path(benchmark_dir)
    bc_files = sorted(benchmark_path.glob("*.bc"))
    
    if benchmarks:
        # 过滤指定的 benchmarks
        filtered = []
        for bc_file in bc_files:
            name = bc_file.stem
            # 处理形如 benchmark_cbench-v1_xxx 的命名
            short_name = name.split("_")[-1] if "_" in name else name
            if name in benchmarks or short_name in benchmarks:
                filtered.append(bc_file)
        bc_files = filtered
    
    return bc_files


def evaluate_single_benchmark(
    bc_path: Path,
    inferencer: OptSeqGenInference,
    llvm_path: str,
    max_input_length: int = 1024,
    max_output_length: int = 256,
    num_beams: int = 4
) -> BenchmarkResult:
    """评估单个 benchmark
    
    Args:
        bc_path: .bc 文件路径
        inferencer: 推理器
        llvm_path: LLVM 工具链路径
        max_input_length: 最大输入长度
        max_output_length: 最大输出长度
        num_beams: beam search 数量
        
    Returns:
        BenchmarkResult
    """
    benchmark_name = bc_path.stem
    llvm_dis_path, _, _, _ = get_llvm_utils(llvm_path)
    
    try:
        # Step 1: 将 bc 转换为 LLVM IR
        llvm_ir = bitcode_to_llvm_ir(str(bc_path), llvm_dis_path)
        
        # Step 2: 计算原始 IR 指令数
        original_ir_count = count_llvm_ir_instructions(llvm_ir)
        
        # Step 3: 使用模型推理生成优化序列
        predicted_passes = inferencer.generate(
            llvm_ir,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            num_beams=num_beams
        )
        
        # 解析优化 passes (格式: "-pass1 -pass2 -pass3")
        if isinstance(predicted_passes, str):
            opt_flags = predicted_passes.strip().split()
        else:
            opt_flags = predicted_passes
        
        # Step 4: 应用模型预测的优化
        optimized_ir = optimize_llvm_ir(llvm_ir, opt_flags=opt_flags, llvm_path=llvm_path)
        
        if not optimized_ir:
            # 优化失败，使用原始 IR 数量
            optimized_ir_count = original_ir_count
        else:
            optimized_ir_count = count_llvm_ir_instructions(optimized_ir)
        
        # Step 5: 计算 O3 基线
        o3_ir = optimize_llvm_ir(llvm_ir, opt_flags=["-O3"], llvm_path=llvm_path)
        
        if not o3_ir:
            o3_ir_count = original_ir_count
        else:
            o3_ir_count = count_llvm_ir_instructions(o3_ir)
        
        # Step 6: 计算指标
        reduction_ratio = (original_ir_count - optimized_ir_count) / original_ir_count if original_ir_count > 0 else 0
        o3_reduction_ratio = (original_ir_count - o3_ir_count) / original_ir_count if original_ir_count > 0 else 0
        
        # relative_improvement: 采用 CompilerGym IrInstructionCountO3 标准
        # R(s_t) = (C(s_initial) - C(s_optimized)) / (C(s_initial) - C(s_O3))
        # > 1 表示模型比 O3 更好
        # = 1 表示与 O3 相同
        # < 1 表示模型比 O3 差
        o3_reduction = original_ir_count - o3_ir_count
        relative_improvement = (original_ir_count - optimized_ir_count) / o3_reduction if o3_reduction > 0 else 1.0
        
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            original_ir_count=original_ir_count,
            optimized_ir_count=optimized_ir_count,
            o3_ir_count=o3_ir_count,
            predicted_passes=predicted_passes if isinstance(predicted_passes, str) else " ".join(predicted_passes),
            reduction_ratio=reduction_ratio,
            o3_reduction_ratio=o3_reduction_ratio,
            relative_improvement=relative_improvement,
            success=True
        )
        
    except Exception as e:
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            original_ir_count=0,
            optimized_ir_count=0,
            o3_ir_count=0,
            predicted_passes="",
            reduction_ratio=0.0,
            o3_reduction_ratio=0.0,
            relative_improvement=1.0,
            success=False,
            error_message=str(e)
        )


def evaluate_benchmarks(
    bc_files: List[Path],
    inferencer: OptSeqGenInference,
    llvm_path: str,
    max_input_length: int = 1024,
    max_output_length: int = 256,
    num_beams: int = 4
) -> List[BenchmarkResult]:
    """评估多个 benchmarks
    
    Args:
        bc_files: .bc 文件路径列表
        inferencer: 推理器
        llvm_path: LLVM 工具链路径
        max_input_length: 最大输入长度
        max_output_length: 最大输出长度
        num_beams: beam search 数量
        
    Returns:
        BenchmarkResult 列表
    """
    results = []
    
    for bc_path in tqdm(bc_files, desc="Evaluating benchmarks"):
        result = evaluate_single_benchmark(
            bc_path=bc_path,
            inferencer=inferencer,
            llvm_path=llvm_path,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            num_beams=num_beams
        )
        results.append(result)
        
        # 打印单个结果
        if result.success:
            print(f"\n{result.benchmark_name}:")
            print(f"  Original IR: {result.original_ir_count}")
            print(f"  Optimized IR: {result.optimized_ir_count} (reduction: {result.reduction_ratio:.2%})")
            print(f"  O3 IR: {result.o3_ir_count} (reduction: {result.o3_reduction_ratio:.2%})")
            print(f"  Relative to O3: {result.relative_improvement:.4f}")
            print(f"  Predicted passes: {result.predicted_passes[:100]}...")
        else:
            print(f"\n{result.benchmark_name}: FAILED - {result.error_message}")
    
    return results


def compute_summary_metrics(results: List[BenchmarkResult]) -> Dict:
    """计算汇总指标"""
    successful = [r for r in results if r.success]
    
    if not successful:
        return {
            "avg_reduction_ratio": 0.0,
            "avg_o3_reduction_ratio": 0.0,
            "avg_relative_improvement": 1.0,
            "success_rate": 0.0,
            "total_benchmarks": len(results),
            "successful_benchmarks": 0
        }
    
    avg_reduction = sum(r.reduction_ratio for r in successful) / len(successful)
    avg_o3_reduction = sum(r.o3_reduction_ratio for r in successful) / len(successful)
    avg_relative = sum(r.relative_improvement for r in successful) / len(successful)
    success_rate = len(successful) / len(results)
    
    return {
        "avg_reduction_ratio": avg_reduction,
        "avg_o3_reduction_ratio": avg_o3_reduction,
        "avg_relative_improvement": avg_relative,
        "success_rate": success_rate,
        "total_benchmarks": len(results),
        "successful_benchmarks": len(successful)
    }


def save_results(
    results: List[BenchmarkResult],
    metrics: Dict,
    output_dir: str,
    model_path: str,
    llvm_path: str,
    benchmark_dir: str
):
    """保存评估结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    detailed_results = {
        "metadata": {
            "model_path": model_path,
            "llvm_path": llvm_path,
            "benchmark_dir": benchmark_dir,
            "timestamp": timestamp
        },
        "summary": metrics,
        "benchmarks": [asdict(r) for r in results]
    }
    
    results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")
    
    # 保存详细 CSV（包含优化序列）
    csv_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        # CSV 表头
        f.write("benchmark,original_ir,optimized_ir,o3_ir,reduction_ratio,o3_reduction,relative_to_o3,success,error_message,predicted_passes\n")
        for r in results:
            # 处理 predicted_passes 中可能包含的特殊字符（用双引号括起来，内部双引号转义）
            passes_escaped = r.predicted_passes.replace('"', '""')
            error_escaped = r.error_message.replace('"', '""')
            f.write(f'{r.benchmark_name},{r.original_ir_count},{r.optimized_ir_count},'
                    f'{r.o3_ir_count},{r.reduction_ratio:.4f},{r.o3_reduction_ratio:.4f},'
                    f'{r.relative_improvement:.4f},{r.success},"{error_escaped}","{passes_escaped}"\n')
    
    print(f"CSV 结果已保存到: {csv_file}")
    
    return results_file


def main():
    args = parse_args()
    
    # 加载配置 (如果提供)
    if args.config:
        cfg = load_config(args.config)
        model_path = cfg.get("evaluation", {}).get("model_path", args.model_path)
        benchmark_dir = cfg.get("evaluation", {}).get("benchmark_dir", args.benchmark_dir)
        llvm_path = cfg.get("evaluation", {}).get("llvm_path", args.llvm_path)
        output_dir = cfg.get("evaluation", {}).get("output_dir", args.output_dir)
        max_input_length = cfg.get("evaluation", {}).get("max_input_length", args.max_input_length)
        max_output_length = cfg.get("evaluation", {}).get("max_output_length", args.max_output_length)
        num_beams = cfg.get("evaluation", {}).get("num_beams", args.num_beams)
        benchmarks = cfg.get("evaluation", {}).get("benchmarks", args.benchmarks)
    else:
        model_path = args.model_path
        benchmark_dir = args.benchmark_dir
        llvm_path = args.llvm_path
        output_dir = args.output_dir
        max_input_length = args.max_input_length
        max_output_length = args.max_output_length
        num_beams = args.num_beams
        benchmarks = args.benchmarks
    
    # 验证必需参数
    if not model_path:
        raise ValueError("必须指定 --model_path 或在配置文件中提供")
    if not benchmark_dir:
        raise ValueError("必须指定 --benchmark_dir 或在配置文件中提供")
    
    # 设置默认输出目录
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(model_path), "evaluation_results")
    
    print("=" * 60)
    print("Passformer 模型评估")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"Benchmark 目录: {benchmark_dir}")
    print(f"LLVM 路径: {llvm_path}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {args.device}")
    print("=" * 60)
    
    # 发现 benchmarks
    bc_files = discover_benchmarks(benchmark_dir, benchmarks)
    print(f"\n发现 {len(bc_files)} 个 benchmark 文件:")
    for f in bc_files:
        print(f"  - {f.name}")
    
    if not bc_files:
        print("未找到任何 .bc 文件！")
        return
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    inferencer = OptSeqGenInference.from_pretrained(
        model_path,
        encoder_tokenizer_type=args.encoder_tokenizer_type,
        decoder_tokenizer_type="optiseq",
        device=args.device
    )
    print("模型加载完成！")
    
    # 评估
    print("\n开始评估...")
    results = evaluate_benchmarks(
        bc_files=bc_files,
        inferencer=inferencer,
        llvm_path=llvm_path,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        num_beams=num_beams
    )
    
    # 计算汇总指标
    metrics = compute_summary_metrics(results)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"总 benchmarks: {metrics['total_benchmarks']}")
    print(f"成功: {metrics['successful_benchmarks']}")
    print(f"成功率: {metrics['success_rate']:.2%}")
    print(f"平均 IR 缩减比例: {metrics['avg_reduction_ratio']:.2%}")
    print(f"O3 平均 IR 缩减比例: {metrics['avg_o3_reduction_ratio']:.2%}")
    print(f"相对 O3 性能: {metrics['avg_relative_improvement']:.4f}")
    print("  (> 1.0 表示优于 O3, = 1.0 表示相当, < 1.0 表示劣于 O3)")
    print("=" * 60)
    
    # 保存结果
    save_results(
        results=results,
        metrics=metrics,
        output_dir=output_dir,
        model_path=model_path,
        llvm_path=llvm_path,
        benchmark_dir=benchmark_dir
    )


if __name__ == "__main__":
    main()

