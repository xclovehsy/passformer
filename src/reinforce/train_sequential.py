import os
import random
import csv
import time
import math
import torch
import argparse
from datetime import datetime
from tqdm import tqdm

from src.reinforce.trainer import REINFORCETrainer
from src.rl.llvm_wrapper import llvm_wrapper
from src.model import PassformerModel, Inst2VecTokenizer, OptiSeqTokenizer
from src.config import load_config
from src.utils.utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential REINFORCE: train each benchmark to convergence before moving on",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def bc_path_to_benchmark_uri(bc_path: str) -> str:
    """Convert bc file path to benchmark URI, e.g. benchmark://cbench-v1/adpcm."""
    stem = os.path.basename(bc_path).replace(".bc", "")
    parts = stem.split("_")
    if len(parts) >= 3:
        return f"benchmark://{parts[1]}/{parts[2]}"
    return f"benchmark://unknown/{stem}"


def compute_geomean(rewards: list[float]) -> float:
    """Geometric mean of rewards. Uses log for numerical stability."""
    if not rewards:
        return 0.0
    log_sum = sum(math.log(max(r, 1e-10)) for r in rewards)
    return math.exp(log_sum / len(rewards))


def build_single_sample(bc_path):
    """Build observation dict for a single bc file."""
    env = llvm_wrapper([bc_path], is_from_bc=True)
    obs = env.reset()
    sample = {
        "llvm_ir": [obs.llvm_ir],
        "autophase": torch.tensor(obs.autophase).unsqueeze(0),
        "bc_path": [bc_path],
    }
    env.close()
    return sample


def main():
    args = parse_args()
    cfg = load_config(args.config)

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(cfg["output"]["base_work_dir"], time_str)
    os.makedirs(work_dir, exist_ok=True)
    logger = get_logger(work_dir)
    logger.info(f"Sequential REINFORCE training  work_dir={work_dir}")

    # ---- model & tokenizers ----
    enc_tok = Inst2VecTokenizer.from_pretrained(cfg["model"]["encoder_tokenizer_id"])
    dec_tok = OptiSeqTokenizer.from_pretrained(cfg["model"]["decoder_tokenizer_id"])
    model = PassformerModel.from_pretrained(cfg["model"]["model_id"])
    logger.info("Model and tokenizers loaded")

    # ---- trainer ----
    trainer = REINFORCETrainer(cfg, model, enc_tok, dec_tok, logger, work_dir)

    # ---- sequential training params ----
    bc_files = cfg["data"]["bc_files"]
    train_cfg = cfg["training"]
    epochs_per_bm = int(train_cfg.get("epochs_per_benchmark", 30))
    reward_threshold = float(train_cfg.get("reward_threshold", 10))
    patience = int(train_cfg.get("patience", 8))
    replay_ratio = float(train_cfg.get("replay_ratio", 0.3))
    replay_epochs = int(train_cfg.get("replay_epochs", 2))
    save_steps = int(train_cfg.get("save_steps", 50))
    log_steps = int(train_cfg.get("log_steps", 1))

    global_step = 0
    trained_bc_files = []
    benchmark_results: list[dict] = []  # {benchmark, reward, walltime, commandline}

    logger.info(f"Benchmarks: {len(bc_files)}, epochs_per_bm={epochs_per_bm}, "
                f"reward_threshold={reward_threshold}, patience={patience}")

    for bm_idx, bc_path in enumerate(bc_files):
        bm_name = os.path.basename(bc_path).replace(".bc", "")
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmark [{bm_idx+1}/{len(bc_files)}]: {bm_name}")
        logger.info(f"{'='*60}")

        batch = build_single_sample(bc_path)

        best_reward = -float("inf")
        best_commandline = ""
        no_improve_count = 0
        time_start = time.perf_counter()

        pbar = tqdm(range(epochs_per_bm), desc=f"[{bm_idx+1}] {bm_name}")
        for epoch in pbar:
            metrics = trainer.train_step(batch)
            global_step += 1
            mean_reward = metrics["reward_mean"]
            max_reward = metrics["reward_max"]

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "rew": f"{mean_reward:.4f}",
                "best": f"{best_reward:.4f}",
            })

            if global_step % log_steps == 0:
                num_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                parts = " | ".join(f"{k}={v:.4f}" for k, v in num_metrics.items())
                logger.info(f"Step {global_step} | bm={bm_name} | {parts}")

            if global_step % save_steps == 0:
                ckpt = os.path.join(work_dir, f"step_{global_step}")
                trainer.model.save_pretrained(ckpt)
                logger.info(f"Checkpoint saved: {ckpt}")

            if max_reward > best_reward:
                best_reward = max_reward
                best_commandline = metrics.get("best_commandline", "")
                no_improve_count = 0
            else:
                no_improve_count += 1

            if mean_reward >= reward_threshold:
                logger.info(f"  -> reward {mean_reward:.4f} >= threshold {reward_threshold}, "
                            f"moving to next benchmark")
                break

            if no_improve_count >= patience:
                logger.info(f"  -> no improvement for {patience} epochs "
                            f"(best={best_reward:.4f}), moving to next benchmark")
                break

        walltime = time.perf_counter() - time_start
        benchmark_uri = bc_path_to_benchmark_uri(bc_path)
        benchmark_results.append({
            "benchmark": benchmark_uri,
            "reward": best_reward,
            "walltime": walltime,
            "commandline": best_commandline,
        })
        logger.info(f"Benchmark {bm_name} done: best_reward={best_reward:.4f}, "
                    f"epochs={epoch+1}, walltime={walltime:.2f}s")

        trained_bc_files.append(bc_path)

        # ---- replay on previously trained benchmarks ----
        if trained_bc_files and replay_ratio > 0 and len(trained_bc_files) > 1:
            num_replay = max(1, int(len(trained_bc_files) * replay_ratio))
            replay_bcs = random.sample(trained_bc_files[:-1],
                                       min(num_replay, len(trained_bc_files) - 1))
            logger.info(f"  Replay on {len(replay_bcs)} previous benchmark(s)")

            for rbc in replay_bcs:
                rbc_name = os.path.basename(rbc).replace(".bc", "")
                replay_batch = build_single_sample(rbc)
                for r_epoch in range(replay_epochs):
                    metrics = trainer.train_step(replay_batch)
                    global_step += 1
                    if global_step % log_steps == 0:
                        num_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                        parts = " | ".join(f"{k}={v:.4f}" for k, v in num_metrics.items())
                        logger.info(f"Step {global_step} | replay={rbc_name} | {parts}")

    # ---- save final model ----
    final = os.path.join(work_dir, "final_model")
    trainer.model.save_pretrained(final)
    dec_tok.save_pretrained(final)
    logger.info(f"Training finished. Final model: {final}")

    # ---- save benchmark results table & geomean ----
    if benchmark_results:
        results_csv = os.path.join(work_dir, "benchmark_results.csv")
        rewards = [r["reward"] for r in benchmark_results]
        geomean = compute_geomean(rewards)
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["benchmark", "reward", "walltime", "commandline"])
            writer.writeheader()
            writer.writerows(benchmark_results)
            writer.writerow({"benchmark": "geomean", "reward": geomean, "walltime": "", "commandline": ""})
        logger.info(f"Benchmark results saved to {results_csv}")
        logger.info(f"Reward geometric mean (geomean) over {len(rewards)} benchmarks: {geomean:.6f}")


if __name__ == "__main__":
    main()
