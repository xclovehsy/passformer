"""
Evaluate a trained Passformer (PPO) model on the cBench-v1 dataset.

Supports three generation strategies:
  - Greedy decoding (temperature → 0)
  - Beam search
  - Multi-temperature sampling (same as training rollout)

For each benchmark the script records:
  - Per-step and total reward from the LLVM environment
  - Original / optimized / O3 / Oz instruction counts
  - Best pass sequence found
  - Timing information

Results are saved to CSV + JSON and a summary is printed.

Usage:
    python -m src.reinforce.test_cbench --config configs/reinforce_test_cbench.yaml
    python -m src.reinforce.test_cbench --config configs/reinforce_test_cbench.yaml --model_path /path/to/checkpoint
"""

import os
import csv
import json
import math
import time
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

import torch
import numpy as np
from tqdm import tqdm

from src.grpo.llvm_wrapper import llvm_wrapper
from src.model import PassformerModel, Inst2VecTokenizer, OptiSeqTokenizer
from src.config import load_config
from src.utils.utils import get_logger


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class BenchmarkTestResult:
    benchmark: str
    bc_path: str
    original_ic: int = 0
    optimized_ic: int = 0
    o3_ic: int = 0
    oz_ic: int = 0
    reward_total: float = 0.0
    reward_mean: float = 0.0
    reward_max: float = 0.0
    ic_reduction: float = 0.0
    ic_reduction_vs_o3: float = 0.0
    best_pass_sequence: str = ""
    num_passes: int = 0
    strategy: str = ""
    num_rollouts: int = 0
    inference_time: float = 0.0
    eval_time: float = 0.0
    success: bool = True
    error_message: str = ""


@dataclass
class TestSummary:
    total_benchmarks: int = 0
    successful: int = 0
    failed: int = 0
    reward_mean: float = 0.0
    reward_geomean: float = 0.0
    reward_max: float = 0.0
    ic_reduction_mean: float = 0.0
    ic_reduction_geomean: float = 0.0
    strategy: str = ""
    timestamp: str = ""


# ======================================================================
# Helpers
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Passformer on cBench-v1",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override model path from config")
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=None,
                        help="Generation strategies: greedy / beam / sampling (default: all enabled in config)")
    return parser.parse_args()


def compute_geomean(values: List[float]) -> float:
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def build_single_sample(bc_path: str):
    env = llvm_wrapper([bc_path], is_from_bc=True)
    obs = env.reset()
    sample = {
        "llvm_ir": [obs.llvm_ir],
        "autophase": torch.tensor(obs.autophase).unsqueeze(0),
        "bc_path": [bc_path],
    }
    env.close()
    return sample


def get_ic_counts(bc_path: str) -> Dict[str, int]:
    """Get original / O3 / Oz instruction counts via CompilerGym."""
    import compiler_gym
    from compiler_gym.envs.llvm import make_benchmark

    env = compiler_gym.make("llvm-ic-v0")
    try:
        benchmark = make_benchmark(bc_path)
        env.reset(benchmark=benchmark)
        original = env.observation["IrInstructionCount"]
        o3 = env.observation["IrInstructionCountO3"]
        oz = env.observation["IrInstructionCountOz"]
        return {"original": original, "o3": o3, "oz": oz}
    finally:
        env.close()


def apply_passes_and_get_ic(bc_path: str, pass_sequence: str) -> int:
    """Apply a pass sequence and return the optimized instruction count."""
    import compiler_gym
    from compiler_gym.envs.llvm import make_benchmark

    env = compiler_gym.make("llvm-ic-v0")
    try:
        benchmark = make_benchmark(bc_path)
        env.reset(benchmark=benchmark)
        passes = pass_sequence.strip().split()
        for p in passes:
            try:
                idx = env.action_space.flags.index(p)
                _, _, done, _ = env.step(idx)
                if done:
                    break
            except ValueError:
                continue
        return env.observation["IrInstructionCount"]
    finally:
        env.close()


# ======================================================================
# Generation strategies
# ======================================================================

@torch.no_grad()
def generate_greedy(model, enc_inputs, dec_tok, max_gen_length, device):
    """Greedy decoding (argmax at each step)."""
    sequences = model.generate(
        input_ids=enc_inputs["input_ids"].to(device),
        attention_mask=enc_inputs["attention_mask"].to(device),
        autophase=enc_inputs["autophase"].to(device),
        max_length=max_gen_length,
        do_sample=False,
        pad_token_id=dec_tok.pad_token_id,
        eos_token_id=dec_tok.eos_token_id,
    )
    return sequences


@torch.no_grad()
def generate_beam(model, enc_inputs, dec_tok, max_gen_length, num_beams, device):
    """Beam search decoding."""
    sequences = model.generate(
        input_ids=enc_inputs["input_ids"].to(device),
        attention_mask=enc_inputs["attention_mask"].to(device),
        autophase=enc_inputs["autophase"].to(device),
        max_length=max_gen_length,
        num_beams=num_beams,
        do_sample=False,
        pad_token_id=dec_tok.pad_token_id,
        eos_token_id=dec_tok.eos_token_id,
    )
    return sequences


@torch.no_grad()
def generate_sampling(model, enc_inputs, dec_tok, max_gen_length,
                      num_rollouts, temperatures, device):
    """Multi-temperature sampling (same as training rollout)."""
    n_temps = len(temperatures)
    base, remainder = divmod(num_rollouts, n_temps)
    rollout_splits = [base + (1 if i < remainder else 0) for i in range(n_temps)]

    seq_groups = []
    for temp, count in zip(temperatures, rollout_splits):
        if count == 0:
            continue
        ids = enc_inputs["input_ids"].to(device).repeat_interleave(count, dim=0)
        mask = enc_inputs["attention_mask"].to(device).repeat_interleave(count, dim=0)
        auto = enc_inputs["autophase"].to(device).repeat_interleave(count, dim=0)
        seqs = model.generate(
            input_ids=ids,
            attention_mask=mask,
            autophase=auto,
            max_length=max_gen_length,
            do_sample=True,
            temperature=temp,
            pad_token_id=dec_tok.pad_token_id,
            eos_token_id=dec_tok.eos_token_id,
        )
        seq_groups.append(seqs)

    max_len = max(s.shape[1] for s in seq_groups)
    padded = []
    for s in seq_groups:
        if s.shape[1] < max_len:
            pad = torch.full(
                (s.shape[0], max_len - s.shape[1]),
                dec_tok.pad_token_id, dtype=s.dtype, device=s.device,
            )
            s = torch.cat([s, pad], dim=1)
        padded.append(s)

    return torch.cat(padded, dim=0)


# ======================================================================
# Reward evaluation
# ======================================================================

def eval_pass_sequence(dec_tok, token_ids: List[int], bc_path: str) -> float:
    """Evaluate a single pass sequence in the LLVM environment, return total reward."""
    special_ids = {dec_tok.pad_token_id, dec_tok.eos_token_id, dec_tok.bos_token_id}
    env = None
    total_reward = 0.0
    try:
        env = llvm_wrapper([bc_path], is_from_bc=True)
        env.reset()
        flag_to_id = {f: i for i, f in enumerate(env.action_space.flags)}

        for tid in token_ids:
            if tid in special_ids and tid != 126:
                break
            pass_flag = dec_tok.ids_to_tokens.get(tid)
            if pass_flag is None or pass_flag not in flag_to_id:
                continue
            _, reward, done, _ = env.env.step(flag_to_id[pass_flag])
            total_reward += float(reward)
            if done:
                break
    except Exception:
        pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return total_reward


def evaluate_sequences(dec_tok, sequences, bc_path: str):
    """Evaluate all generated sequences and return (rewards, best_idx, best_pass_str)."""
    gen_tokens = sequences[:, 1:]
    rewards = []
    for i in range(gen_tokens.shape[0]):
        token_ids = gen_tokens[i].tolist()
        r = eval_pass_sequence(dec_tok, token_ids, bc_path)
        rewards.append(r)

    rewards_arr = np.array(rewards)
    best_idx = int(rewards_arr.argmax())
    best_seq = sequences[best_idx]
    best_pass_str = dec_tok.decode(best_seq, skip_special_tokens=True)
    return rewards_arr, best_idx, best_pass_str


# ======================================================================
# Single benchmark evaluation
# ======================================================================

def evaluate_benchmark(
    model, enc_tok, dec_tok, bc_path: str, strategy: str,
    cfg_test: dict, cfg_data: dict, device, logger,
) -> BenchmarkTestResult:
    bm_name = os.path.basename(bc_path).replace(".bc", "")
    result = BenchmarkTestResult(benchmark=bm_name, bc_path=bc_path, strategy=strategy)

    try:
        batch = build_single_sample(bc_path)
    except Exception as e:
        logger.warning(f"[{bm_name}] Failed to build sample: {e}")
        result.success = False
        result.error_message = str(e)
        return result

    llvm_irs = batch["llvm_ir"]
    autophases = batch["autophase"].to(device)
    inputs = enc_tok(
        llvm_irs, padding=True, truncation=True,
        max_length=cfg_data.get("max_length", 512), return_tensors="pt",
    )
    enc_inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "autophase": autophases,
    }

    max_gen_length = int(cfg_test.get("max_gen_length", 32))

    # --- generate ---
    t0 = time.time()
    model.eval()

    if strategy == "greedy":
        sequences = generate_greedy(model, enc_inputs, dec_tok, max_gen_length, device)
        result.num_rollouts = 1
    elif strategy == "beam":
        num_beams = int(cfg_test.get("num_beams", 4))
        sequences = generate_beam(model, enc_inputs, dec_tok, max_gen_length, num_beams, device)
        result.num_rollouts = 1
    elif strategy == "sampling":
        num_rollouts = int(cfg_test.get("num_rollouts", 16))
        temps = cfg_test.get("temperatures", [0.3, 0.7])
        if not isinstance(temps, list):
            temps = [float(temps)]
        sequences = generate_sampling(
            model, enc_inputs, dec_tok, max_gen_length,
            num_rollouts, [float(t) for t in temps], device,
        )
        result.num_rollouts = num_rollouts
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    result.inference_time = time.time() - t0

    # --- evaluate in LLVM env ---
    t1 = time.time()
    rewards_arr, best_idx, best_pass_str = evaluate_sequences(dec_tok, sequences, bc_path)
    result.eval_time = time.time() - t1

    result.reward_total = float(rewards_arr.sum())
    result.reward_mean = float(rewards_arr.mean())
    result.reward_max = float(rewards_arr.max())
    result.best_pass_sequence = best_pass_str
    result.num_passes = len(best_pass_str.split()) if best_pass_str.strip() else 0

    # --- instruction count metrics ---
    try:
        ic = get_ic_counts(bc_path)
        result.original_ic = ic["original"]
        result.o3_ic = ic["o3"]
        result.oz_ic = ic["oz"]

        if best_pass_str.strip():
            result.optimized_ic = apply_passes_and_get_ic(bc_path, best_pass_str)
        else:
            result.optimized_ic = result.original_ic

        if result.original_ic > 0:
            result.ic_reduction = 1.0 - result.optimized_ic / result.original_ic
        if ic["o3"] > 0:
            result.ic_reduction_vs_o3 = 1.0 - result.optimized_ic / ic["o3"]
    except Exception as e:
        logger.warning(f"[{bm_name}] IC count failed: {e}")

    return result


# ======================================================================
# Main
# ======================================================================

def print_results_table(results: List[BenchmarkTestResult], logger):
    header = (
        f"{'Benchmark':<35} {'Reward':>8} {'Orig IC':>8} {'Opt IC':>8} "
        f"{'O3 IC':>8} {'IC Red%':>8} {'vs O3%':>8} {'#Pass':>6} {'Time':>7}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        if not r.success:
            logger.info(f"{r.benchmark:<35} FAILED: {r.error_message}")
            continue
        logger.info(
            f"{r.benchmark:<35} {r.reward_max:>8.2f} {r.original_ic:>8d} "
            f"{r.optimized_ic:>8d} {r.o3_ic:>8d} "
            f"{r.ic_reduction * 100:>7.2f}% {r.ic_reduction_vs_o3 * 100:>7.2f}% "
            f"{r.num_passes:>6d} {r.inference_time + r.eval_time:>6.1f}s"
        )


def compute_test_summary(results: List[BenchmarkTestResult], strategy: str) -> TestSummary:
    successful = [r for r in results if r.success]
    rewards = [r.reward_max for r in successful]
    ic_reds = [r.ic_reduction for r in successful if r.ic_reduction > 0]
    return TestSummary(
        total_benchmarks=len(results),
        successful=len(successful),
        failed=len(results) - len(successful),
        reward_mean=float(np.mean(rewards)) if rewards else 0.0,
        reward_geomean=compute_geomean([max(r, 1e-10) for r in rewards]),
        reward_max=float(np.max(rewards)) if rewards else 0.0,
        ic_reduction_mean=float(np.mean(ic_reds)) if ic_reds else 0.0,
        ic_reduction_geomean=compute_geomean(ic_reds) if ic_reds else 0.0,
        strategy=strategy,
        timestamp=datetime.now().isoformat(),
    )


def save_results(results: List[BenchmarkTestResult], summary: TestSummary,
                 work_dir: str, strategy: str):
    csv_path = os.path.join(work_dir, f"results_{strategy}.csv")
    fieldnames = [
        "benchmark", "bc_path", "original_ic", "optimized_ic", "o3_ic", "oz_ic",
        "reward_total", "reward_mean", "reward_max",
        "ic_reduction", "ic_reduction_vs_o3",
        "best_pass_sequence", "num_passes", "strategy", "num_rollouts",
        "inference_time", "eval_time", "success", "error_message",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    json_path = os.path.join(work_dir, f"results_{strategy}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def main():
    args = parse_args()
    cfg = load_config(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(cfg["output"]["base_work_dir"], timestamp)
    os.makedirs(work_dir, exist_ok=True)
    logger = get_logger(work_dir, logging_name="test.log")

    logger.info(f"cBench-v1 evaluation  work_dir={work_dir}")
    logger.info(f"Config: {args.config}")

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- model & tokenizers ----
    model_path = args.model_path or cfg["model"]["model_id"]
    logger.info(f"Loading model from: {model_path}")

    enc_tok = Inst2VecTokenizer.from_pretrained(cfg["model"]["encoder_tokenizer_id"])
    dec_tok = OptiSeqTokenizer.from_pretrained(cfg["model"]["decoder_tokenizer_id"])
    model = PassformerModel.from_pretrained(model_path).to(device).eval()
    logger.info("Model and tokenizers loaded")

    # ---- benchmark list ----
    bc_files = cfg["data"]["bc_files"]
    logger.info(f"Benchmarks to evaluate: {len(bc_files)}")

    # ---- determine strategies ----
    cfg_test = cfg.get("test", {})
    if args.strategies:
        strategies = args.strategies
    else:
        strategies = []
        if cfg_test.get("greedy", True):
            strategies.append("greedy")
        if cfg_test.get("beam_search", True):
            strategies.append("beam")
        if cfg_test.get("temperatures"):
            strategies.append("sampling")
    logger.info(f"Strategies: {strategies}")

    # ---- run evaluation for each strategy ----
    all_summaries = []

    for strategy in strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"{'='*60}")

        results: List[BenchmarkTestResult] = []

        for bc_path in tqdm(bc_files, desc=f"Eval [{strategy}]"):
            bm_name = os.path.basename(bc_path).replace(".bc", "")
            logger.info(f"\n--- {bm_name} ({strategy}) ---")

            result = evaluate_benchmark(
                model, enc_tok, dec_tok, bc_path, strategy,
                cfg_test, cfg["data"], device, logger,
            )
            results.append(result)

            if result.success:
                logger.info(
                    f"  reward_max={result.reward_max:.4f}  "
                    f"IC: {result.original_ic} -> {result.optimized_ic} "
                    f"(O3={result.o3_ic})  "
                    f"reduction={result.ic_reduction*100:.2f}%  "
                    f"passes={result.num_passes}"
                )
                if result.best_pass_sequence:
                    logger.info(f"  best_seq: {result.best_pass_sequence}")
            else:
                logger.warning(f"  FAILED: {result.error_message}")

        # ---- summary ----
        summary = compute_test_summary(results, strategy)
        all_summaries.append(summary)

        logger.info(f"\n{'='*60}")
        logger.info(f"Summary [{strategy}]")
        logger.info(f"{'='*60}")
        print_results_table(results, logger)

        logger.info(f"\nBenchmarks: {summary.total_benchmarks} "
                     f"(success={summary.successful}, failed={summary.failed})")
        logger.info(f"Reward  mean={summary.reward_mean:.4f}  "
                     f"geomean={summary.reward_geomean:.4f}  "
                     f"max={summary.reward_max:.4f}")
        logger.info(f"IC Reduction  mean={summary.ic_reduction_mean*100:.2f}%  "
                     f"geomean={summary.ic_reduction_geomean*100:.2f}%")

        csv_path, json_path = save_results(results, summary, work_dir, strategy)
        logger.info(f"Results saved: {csv_path}")

    # ---- cross-strategy comparison ----
    if len(all_summaries) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("Strategy Comparison")
        logger.info(f"{'='*60}")
        header = f"{'Strategy':<12} {'Reward Mean':>12} {'Reward Geo':>12} {'IC Red Mean':>12} {'IC Red Geo':>12}"
        logger.info(header)
        logger.info("-" * len(header))
        for s in all_summaries:
            logger.info(
                f"{s.strategy:<12} {s.reward_mean:>12.4f} {s.reward_geomean:>12.4f} "
                f"{s.ic_reduction_mean*100:>11.2f}% {s.ic_reduction_geomean*100:>11.2f}%"
            )

    # ---- save overall summary ----
    overall_path = os.path.join(work_dir, "overall_summary.json")
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": model_path,
            "config": args.config,
            "strategies": [asdict(s) for s in all_summaries],
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"\nOverall summary saved: {overall_path}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
