"""Passformer 模型评估脚本

在给定的 benchmark 数据集上评估 Passformer 模型的优化效果。
使用 CompilerGym 环境来应用优化序列并测量性能指标。
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
import compiler_gym
from compiler_gym.envs.llvm import FileBenchmark
from compiler_gym.util.runfiles_path import runfiles_path

from src.inference.passformer_inference import PassformerInference
from src.config.config import load_config
from src.utils.llvm import bitcode_to_llvm_ir


@dataclass
class BenchmarkResult:
    """单个 benchmark 的评估结果"""
    benchmark: str
    benchmark_path: str
    original_ir_count: int
    optimized_ir_count: int
    o3_ir_count: Optional[int] = None
    oz_ir_count: Optional[int] = None
    pass_sequence: str = ""
    num_passes: int = 0
    rewards: List[float] = []
    returns: float = 0.0
    inference_time: float = 0.0
    optimization_time: float = 0.0
    total_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class EvaluationSummary:
    """评估结果汇总"""
    total_benchmarks: int
    successful_benchmarks: int
    failed_benchmarks: int
    avg_returns: float
    avg_inference_time: float
    avg_optimization_time: float
    avg_total_time: float


def get_dataset_benchmarks(dataset_name: str, env=None) -> List[str]:
    should_close = False
    if env is None:
        env = compiler_gym.make("llvm-v0")
        should_close = True
    
    try:
        # 获取数据集
        dataset = env.datasets[dataset_name]
        # 获取所有 benchmark 的 URI
        benchmarks = [str(bm.uri) for bm in dataset.benchmarks()]
        return benchmarks
    finally:
        if should_close:
            env.close()


def find_bc_files(benchmark_dir: str, benchmarks: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    benchmark_dir = Path(benchmark_dir)
    if not benchmark_dir.exists():
        raise ValueError(f"Benchmark directory does not exist: {benchmark_dir}")
    
    bc_files = list(benchmark_dir.glob("*.bc"))
    
    if benchmarks is not None:
        # 过滤指定的 benchmarks
        filtered_files = []
        for bc_file in bc_files:
            # 支持完整名称或简短名称匹配
            name = bc_file.stem
            for bm in benchmarks:
                if bm in name or name.endswith(f"_{bm}") or name.startswith(f"{bm}_"):
                    filtered_files.append((name, str(bc_file)))
                    break
        return filtered_files
    
    return [(bc_file.stem, str(bc_file)) for bc_file in bc_files]


def create_benchmark(benchmark_uri_or_path: str):
    # 检查是否是 URI 格式
    if benchmark_uri_or_path.startswith("benchmark://") or benchmark_uri_or_path.startswith("generator://"):
        # 直接返回 URI，CompilerGym 可以直接使用
        return benchmark_uri_or_path
    else:
        # 从文件路径创建 FileBenchmark
        return FileBenchmark(benchmark_uri_or_path)


def apply_pass_sequence(env, pass_sequence: str) -> Tuple[bool, Optional[str], List[float], float]:
    print(f"apply_pass_sequence, step:begin, benchmark: {env.benchmark}, pass_sequence: {pass_sequence}")
    
    if not pass_sequence or not pass_sequence.strip():
        return True, None, [], 0.0
    
    passes = pass_sequence.strip().split()
    rewards = []
    total_reward = 0.0
    
    try:
        for pass_name in passes:
            try:
                action_idx = env.action_space.flags.index(pass_name)
            except ValueError:
                # Pass 不在 action space 中，跳过
                raise ValueError(f"Pass {pass_name} not in action space")
            
            print(f"apply_pass_sequence, step:iter_begin, pass_name: {pass_name}")
            
            observation, reward, done, info = env.step(action_idx)
            total_reward += reward
            rewards.append(reward)        
            print(f"apply_pass_sequence, step:iter_end, pass_name: {pass_name}, reward: {reward}")

        print(f"apply_pass_sequence, step:end, benchmark: {env.benchmark}, total_reward: {total_reward}")
        return True, None, rewards, total_reward
    except Exception as e:
        return False, str(e), [], 0.0


def evaluate_single_benchmark(
    inference: PassformerInference,
    env,
    benchmark_name: str,
    benchmark_uri_or_path: str,
    llvm_path: Optional[str] = None,
    max_input_length: Optional[int] = None,
    max_output_length: Optional[int] = None,
    num_beams: int = 1,
    do_sample: bool = False,
) -> BenchmarkResult:

    start_time = time.time()
    
    try:
        # 创建 benchmark
        benchmark = create_benchmark(benchmark_uri_or_path)
        
        # 重置环境
        env.reset(benchmark=benchmark)
        
        # 获取初始指标
        original_ir_count = env.observation["IrInstructionCount"]
        o3_ir_count = env.observation["IrInstructionCountO3"]
        oz_ir_count = env.observation["IrInstructionCountOz"]
        
        # 获取 LLVM IR 和 Autophase
        llvm_ir = env.observation["Ir"]
        autophase = env.observation["Autophase"]
    
        # 模型推理生成优化序列
        inference_start = time.time()
        pass_sequence = inference.generate(
            llvm=llvm_ir,
            autophase=autophase,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            num_beams=num_beams,
            do_sample=do_sample,
        )
        inference_time = time.time() - inference_start
        
        # 应用优化序列
        opt_start = time.time()
        success, error_msg, rewards, total_reward = apply_pass_sequence(env, pass_sequence)
        optimization_time = time.time() - opt_start
        
        if not success:
            return BenchmarkResult(
                benchmark=env.benchmark,
                pass_sequence=benchmark_uri_or_path,
                num_passes=len(pass_sequence.split()) if pass_sequence else 0,
                success=False,
                error_message=error_msg,
            )
        
        # 获取优化后的指标
        optimized_ir_count = env.observation["IrInstructionCount"]
        
        return BenchmarkResult(
            benchmark=env.benchmark,
            benchmark_path=benchmark_uri_or_path,
            original_ir_count=original_ir_count,
            optimized_ir_count=optimized_ir_count,
            o3_ir_count=o3_ir_count,
            oz_ir_count=oz_ir_count,
            pass_sequence=pass_sequence,
            num_passes=len(pass_sequence.split()) if pass_sequence else 0,
            rewards=rewards,
            returns=total_reward,
            inference_time=inference_time,
            optimization_time=optimization_time,
            total_time=time.time() - start_time,
            success=True,
            error_message=None,
        )
    
    except Exception as e:
        return BenchmarkResult(
            benchmark=benchmark_name,
            benchmark_path=benchmark_uri_or_path,
            success=False,
            error_message=str(e),
        )


def compute_summary(results: List[BenchmarkResult]) -> EvaluationSummary:
    """计算评估结果汇总"""
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    
    if not successful_results:
        raise ValueError("No successful results")
    
    def geometric_mean(values):
        """计算几何平均数，过滤掉 None 和非正数"""
        positive_values = [v for v in values if v is not None and v > 0]
        if not positive_values:
            return None
        return np.exp(np.mean(np.log(positive_values)))

    return EvaluationSummary(
        total_benchmarks=len(results),
        successful_benchmarks=len(successful_results),
        failed_benchmarks=len(failed_results),
        avg_returns=geometric_mean([r.returns for r in successful_results]),
        avg_inference_time=np.mean([r.inference_time for r in successful_results]),
        avg_optimization_time=np.mean([r.optimization_time for r in successful_results]),
        avg_total_time=np.mean([r.total_time for r in successful_results]),
    )


def save_results(
    results: List[BenchmarkResult],
    summary: EvaluationSummary,
    output_dir: str,
    format: str = "both"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format in ("json", "both"):
        # 保存 JSON 格式
        json_path = output_dir / f"evaluation_results_{timestamp}.json"
        json_data = {
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {json_path}")
    
    if format in ("csv", "both"):
        # 保存 CSV 格式
        import csv
        csv_path = output_dir / f"evaluation_results_{timestamp}.csv"
        fieldnames = [
            "benchmark", "benchmark_path", "original_ir_count", "optimized_ir_count",
            "o3_ir_count", "oz_ir_count", "pass_sequence", "num_passes",
            "returns", "rewards",
            "inference_time", "optimization_time", "total_time", "success", "error_message"
        ]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        print(f"结果已保存到: {csv_path}")
    
    # 保存汇总信息
    summary_path = output_dir / f"evaluation_summary_{timestamp}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Passformer 模型评估汇总\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总 benchmark 数: {summary.total_benchmarks}\n")
        f.write(f"成功评估: {summary.successful_benchmarks}\n")
        f.write(f"失败评估: {summary.failed_benchmarks}\n\n")
        f.write("性能指标:\n")
        f.write(f"  平均返回值: {summary.avg_returns:.4f}\n")
        f.write("\n时间统计:\n")
        f.write(f"  平均推理时间: {summary.avg_inference_time:.4f} 秒\n")
        f.write(f"  平均优化时间: {summary.avg_optimization_time:.4f} 秒\n")
        f.write(f"  平均总时间: {summary.avg_total_time:.4f} 秒\n")
    print(f"汇总已保存到: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="评估 Passformer 模型在给定 benchmark 上的效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 配置文件或直接参数
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (YAML)"
    )
    
    # 模型路径
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型路径"
    )
    parser.add_argument(
        "--encoder_tokenizer_path",
        type=str,
        default=None,
        help="Encoder tokenizer 路径"
    )
    parser.add_argument(
        "--decoder_tokenizer_path",
        type=str,
        default=None,
        help="Decoder tokenizer 路径"
    )
    
    # Benchmark 配置
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default=None,
        help="Benchmark 目录（包含 .bc 文件），与 --dataset 互斥"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="CompilerGym 数据集名称（如 cbench-v1, anghabench-v1），与 --benchmark_dir 互斥。参考: https://compilergym.com/llvm/index.html#datasets"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="要评估的 benchmark 名称列表（可选，不指定则评估全部）。对于数据集，可以是 benchmark URI 或索引"
    )
    parser.add_argument(
        "--max_benchmarks",
        type=int,
        default=None,
        help="限制评估的 benchmark 数量（用于大数据集采样）"
    )
    
    # LLVM 配置
    parser.add_argument(
        "--llvm_path",
        type=str,
        default=None,
        help="LLVM 工具链路径（用于转换 bitcode）"
    )
    
    # 输出配置
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="结果输出目录"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["json", "csv", "both"],
        default="both",
        help="输出格式"
    )
    
    # 推理参数
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=None,
        help="最大输入长度"
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=None,
        help="最大输出长度"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Beam search 的 beam 数量"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="是否使用采样"
    )
    
    # 设备配置
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备 (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
        eval_config = config.get("evaluation", {})
        
        # 从配置文件读取参数（如果命令行未指定）
        if args.model_path is None:
            args.model_path = eval_config.get("model_path")
        if args.benchmark_dir is None:
            args.benchmark_dir = eval_config.get("benchmark_dir")
        if args.benchmarks is None:
            args.benchmarks = eval_config.get("benchmarks")
        if args.llvm_path is None:
            args.llvm_path = eval_config.get("llvm_path")
        if args.output_dir is None:
            args.output_dir = eval_config.get("output_dir")
        if args.max_input_length is None:
            args.max_input_length = eval_config.get("max_input_length")
        if args.max_output_length is None:
            args.max_output_length = eval_config.get("max_output_length")
        if args.num_beams == 1:
            args.num_beams = eval_config.get("num_beams", 1)
    
    # 验证必需参数
    if args.model_path is None:
        parser.error("必须指定 --model_path 或通过 --config 配置文件指定")
    if args.benchmark_dir is None:
        parser.error("必须指定 --benchmark_dir 或通过 --config 配置文件指定")
    
    # 设置默认输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "evaluation_results")
    
    print("=" * 60)
    print("Passformer 模型评估")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"Benchmark 目录: {args.benchmark_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {args.device}")
    print()
    
    # 加载模型和 tokenizer
    print("正在加载模型和 tokenizer...")
    inference = PassformerInference.from_pretrained(
        model_path=args.model_path,
        encoder_tokenizer_path=args.encoder_tokenizer_path,
        decoder_tokenizer_path=args.decoder_tokenizer_path,
        device=args.device,
    )
    print("模型加载完成")
    
    # 获取 benchmark 列表
    print(f"\n正在获取 benchmark 列表...")
    benchmark_list = []
    
    if args.dataset:
        # 从 CompilerGym 数据集获取
        # 需要先创建临时环境来获取数据集信息
        print(f"正在从数据集 '{args.dataset}' 获取 benchmarks...")
        temp_env = compiler_gym.make("llvm-v0")
        try:
            benchmark_uris = get_dataset_benchmarks(args.dataset, temp_env)
            print(f"数据集 '{args.dataset}' 包含 {len(benchmark_uris)} 个 benchmark")
            
            # 如果指定了 benchmarks，进行过滤
            if args.benchmarks:
                filtered_uris = []
                for bm in args.benchmarks:
                    # 支持完整 URI 或索引
                    if bm.startswith("benchmark://") or bm.startswith("generator://"):
                        if bm in benchmark_uris:
                            filtered_uris.append(bm)
                    else:
                        # 尝试作为索引
                        try:
                            idx = int(bm)
                            if 0 <= idx < len(benchmark_uris):
                                filtered_uris.append(benchmark_uris[idx])
                        except ValueError:
                            # 尝试名称匹配
                            for uri in benchmark_uris:
                                if bm in uri:
                                    filtered_uris.append(uri)
                                    break
                benchmark_uris = filtered_uris
                print(f"过滤后剩余 {len(benchmark_uris)} 个 benchmark")
            
            # 限制数量（如果指定）
            if args.max_benchmarks and args.max_benchmarks > 0:
                benchmark_uris = benchmark_uris[:args.max_benchmarks]
                print(f"限制为前 {len(benchmark_uris)} 个 benchmark")
            
            # 转换为 (name, uri) 元组列表
            benchmark_list = [(uri.split("/")[-1] if "/" in uri else uri, uri) for uri in benchmark_uris]
        except Exception as e:
            print(f"错误: 无法从数据集 '{args.dataset}' 获取 benchmarks: {e}")
            temp_env.close()
            return
        finally:
            temp_env.close()
    else:
        # 从文件目录获取
        benchmark_list = find_bc_files(args.benchmark_dir, args.benchmarks)
    
    print(f"共找到 {len(benchmark_list)} 个 benchmark")
    
    if not benchmark_list:
        print("错误: 未找到任何 benchmark")
        return
    
    # 创建 CompilerGym 环境
    print("\n正在创建 CompilerGym 环境...")
    # 检查模型是否需要 autophase
    requires_autophase = (
        hasattr(inference.model.config, 'fusion_method') and 
        inference.model.config.fusion_method is not None and 
        inference.model.config.fusion_method != "none"
    )
    
    if requires_autophase:
        # 如果模型需要 autophase，尝试使用包含 Autophase 的环境
        try:
            env = compiler_gym.make("llvm-autophase-ic-v0")
        except:
            env = compiler_gym.make("llvm-v0")
    else:
        env = compiler_gym.make("llvm-v0")
    
    # 设置观察空间
    env.observation_space = "IrInstructionCount"
    
    # 评估每个 benchmark
    print("\n开始评估...")
    results = []
    
    for benchmark_name, benchmark_uri_or_path in tqdm(benchmark_list, desc="评估进度"):
        result = evaluate_single_benchmark(
            inference=inference,
            env=env,
            benchmark_name=benchmark_name,
            benchmark_uri_or_path=benchmark_uri_or_path,
            llvm_path=args.llvm_path,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
        )
        results.append(result)
        
        if result.success:
            print(f"\n{benchmark_name}:")
            print(f"  原始 IR 指令数: {result.original_ir_count}")
            print(f"  优化后 IR 指令数: {result.optimized_ir_count}")
            print(f"  改进比例: {result.improvement_ratio:.4f}")
            if result.relative_to_o3 is not None:
                print(f"  相对 O3: {result.relative_to_o3:.4f}")
            print(f"  优化序列: {result.pass_sequence}")
        else:
            print(f"\n{benchmark_name}: 失败 - {result.error_message}")
    
    # 关闭环境
    env.close()
    
    # 计算汇总
    print("\n正在计算汇总...")
    summary = compute_summary(results)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"总 benchmark 数: {summary.total_benchmarks}")
    print(f"成功评估: {summary.successful_benchmarks}")
    print(f"失败评估: {summary.failed_benchmarks}")
    print(f"\n平均改进比例: {summary.avg_improvement_ratio:.4f}")
    if summary.avg_relative_to_o3 is not None:
        print(f"平均相对 O3: {summary.avg_relative_to_o3:.4f}")
    if summary.avg_relative_to_oz is not None:
        print(f"平均相对 Oz: {summary.avg_relative_to_oz:.4f}")
    print(f"\n平均推理时间: {summary.avg_inference_time:.4f} 秒")
    print(f"平均优化时间: {summary.avg_optimization_time:.4f} 秒")
    print(f"平均总时间: {summary.avg_total_time:.4f} 秒")
    
    # 保存结果
    print("\n正在保存结果...")
    save_results(results, summary, args.output_dir, args.output_format)
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()
