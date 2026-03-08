"""
Large-scale PPO training on shuffled dataset with TensorBoard logging.

Usage:
    python -m src.reinforce.train_largescale --config configs/reinforce_codecontest.yaml
"""

import os
import glob
import random
import math
import time
import torch
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.reinforce.trainer import PPOTrainer
from src.grpo.llvm_wrapper import llvm_wrapper
from src.model import PassformerModel, Inst2VecTokenizer, OptiSeqTokenizer
from src.config import load_config
from src.utils.utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Large-scale PPO training on shuffled dataset",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    return parser.parse_args()


def collect_bc_files(data_dir: str) -> list[str]:
    """Collect all .bc files from the given directory."""
    pattern = os.path.join(data_dir, "*.bc")
    files = sorted(glob.glob(pattern))
    return files


def split_train_val(bc_files: list[str], val_ratio: float, seed: int):
    """Split bc files into train and val sets."""
    rng = random.Random(seed)
    files = list(bc_files)
    rng.shuffle(files)
    val_size = max(1, int(len(files) * val_ratio))
    val_files = files[:val_size]
    train_files = files[val_size:]
    return train_files, val_files


def build_single_sample(bc_path: str):
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


def compute_geomean(rewards: list[float]) -> float:
    if not rewards:
        return 0.0
    log_sum = sum(math.log(max(r, 1e-10)) for r in rewards)
    return math.exp(log_sum / len(rewards))


@torch.no_grad()
def evaluate(trainer, val_files: list[str], num_samples: int, logger):
    """Evaluate on a subset of validation files.

    Returns dict with aggregated metrics.
    """
    sample_files = random.sample(val_files, min(num_samples, len(val_files)))
    total_rewards = []
    total_max_rewards = []

    for bc_path in sample_files:
        try:
            batch = build_single_sample(bc_path)
        except Exception as e:
            logger.warning(f"Skipping val sample {bc_path}: {e}")
            continue
        sequences, enc_inputs = trainer.rollout(batch)
        _, rewards = trainer.compute_step_rewards(sequences, batch["bc_path"])
        total_rewards.append(rewards.mean().item())
        total_max_rewards.append(rewards.max().item())

    if not total_rewards:
        return {"val_reward_mean": 0.0, "val_reward_max": 0.0, "val_reward_geomean": 0.0}

    return {
        "val_reward_mean": sum(total_rewards) / len(total_rewards),
        "val_reward_max": max(total_max_rewards),
        "val_reward_geomean": compute_geomean(total_rewards),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(cfg["output"]["base_work_dir"], time_str)
    os.makedirs(work_dir, exist_ok=True)
    logger = get_logger(work_dir)
    logger.info(f"Large-scale PPO training  work_dir={work_dir}")

    tb_dir = os.path.join(work_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)
    logger.info(f"TensorBoard logs: {tb_dir}")

    # ---- model & tokenizers ----
    enc_tok = Inst2VecTokenizer.from_pretrained(cfg["model"]["encoder_tokenizer_id"])
    dec_tok = OptiSeqTokenizer.from_pretrained(cfg["model"]["decoder_tokenizer_id"])

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = PassformerModel.from_pretrained(args.resume)
    else:
        model = PassformerModel.from_pretrained(cfg["model"]["model_id"])
    logger.info("Model and tokenizers loaded")

    # ---- trainer ----
    trainer = PPOTrainer(cfg, model, enc_tok, dec_tok, logger, work_dir)

    # ---- data split ----
    data_cfg = cfg["data"]
    data_dir = data_cfg["data_dir"]
    val_ratio = float(data_cfg.get("val_ratio", 0.05))
    seed = int(data_cfg.get("seed", 42))

    all_bc_files = collect_bc_files(data_dir)
    if not all_bc_files:
        raise FileNotFoundError(f"No .bc files found in {data_dir}")

    train_files, val_files = split_train_val(all_bc_files, val_ratio, seed)
    logger.info(f"Dataset: {len(all_bc_files)} total, "
                f"{len(train_files)} train, {len(val_files)} val")

    # ---- training params ----
    train_cfg = cfg["training"]
    num_epochs = int(train_cfg.get("num_epochs", 5))
    save_steps = int(train_cfg.get("save_steps", 500))
    log_steps = int(train_cfg.get("log_steps", 10))
    eval_steps = int(train_cfg.get("eval_steps", 200))
    eval_samples = int(train_cfg.get("eval_samples", 50))

    global_step = 0
    best_val_reward = -float("inf")
    total_train_samples = len(train_files) * num_epochs
    logger.info(f"Training: {num_epochs} epochs, {len(train_files)} samples/epoch, "
                f"{total_train_samples} total steps")

    # ---- metrics accumulators for logging ----
    acc_loss = 0.0
    acc_reward_mean = 0.0
    acc_reward_max = 0.0
    acc_entropy = 0.0
    acc_policy_loss = 0.0
    acc_kl = 0.0
    acc_clip_frac = 0.0
    acc_count = 0
    skipped = 0

    for epoch in range(num_epochs):
        random.shuffle(train_files)
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*60}")

        epoch_rewards = []
        pbar = tqdm(train_files, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for bc_path in pbar:
            bm_name = os.path.basename(bc_path).replace(".bc", "")
            try:
                batch = build_single_sample(bc_path)
            except Exception as e:
                logger.warning(f"Skipping {bm_name}: {e}")
                skipped += 1
                continue

            try:
                metrics = trainer.train_step(batch)
            except Exception as e:
                logger.warning(f"Train step failed for {bm_name}: {e}")
                skipped += 1
                continue

            global_step += 1
            mean_reward = metrics["reward_mean"]
            max_reward = metrics["reward_max"]
            epoch_rewards.append(mean_reward)

            acc_loss += metrics["loss"]
            acc_reward_mean += mean_reward
            acc_reward_max += max_reward
            acc_entropy += metrics.get("entropy", 0.0)
            acc_policy_loss += metrics.get("policy_loss", 0.0)
            acc_kl += metrics.get("kl", 0.0)
            acc_clip_frac += metrics.get("clip_frac", 0.0)
            acc_count += 1

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "rew": f"{mean_reward:.4f}",
                "step": global_step,
            })

            # ---- periodic logging ----
            if global_step % log_steps == 0 and acc_count > 0:
                avg_loss = acc_loss / acc_count
                avg_reward_mean = acc_reward_mean / acc_count
                avg_reward_max = acc_reward_max / acc_count
                avg_entropy = acc_entropy / acc_count
                avg_policy_loss = acc_policy_loss / acc_count
                avg_kl = acc_kl / acc_count
                avg_clip_frac = acc_clip_frac / acc_count

                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/policy_loss", avg_policy_loss, global_step)
                writer.add_scalar("train/reward_mean", avg_reward_mean, global_step)
                writer.add_scalar("train/reward_max", avg_reward_max, global_step)
                writer.add_scalar("train/entropy", avg_entropy, global_step)
                writer.add_scalar("train/kl", avg_kl, global_step)
                writer.add_scalar("train/clip_frac", avg_clip_frac, global_step)
                writer.add_scalar("train/epoch", epoch + 1, global_step)
                writer.add_scalar("train/skipped_samples", skipped, global_step)

                logger.info(
                    f"Step {global_step} | "
                    f"loss={avg_loss:.4f} | "
                    f"reward_mean={avg_reward_mean:.4f} | "
                    f"reward_max={avg_reward_max:.4f} | "
                    f"entropy={avg_entropy:.4f} | "
                    f"clip_frac={avg_clip_frac:.4f}"
                )

                acc_loss = 0.0
                acc_reward_mean = 0.0
                acc_reward_max = 0.0
                acc_entropy = 0.0
                acc_policy_loss = 0.0
                acc_kl = 0.0
                acc_clip_frac = 0.0
                acc_count = 0

            # ---- periodic evaluation ----
            if global_step % eval_steps == 0:
                logger.info(f"Running evaluation at step {global_step}...")
                val_metrics = evaluate(trainer, val_files, eval_samples, logger)

                writer.add_scalar("val/reward_mean", val_metrics["val_reward_mean"], global_step)
                writer.add_scalar("val/reward_max", val_metrics["val_reward_max"], global_step)
                writer.add_scalar("val/reward_geomean", val_metrics["val_reward_geomean"], global_step)

                logger.info(
                    f"Eval step {global_step} | "
                    f"val_reward_mean={val_metrics['val_reward_mean']:.4f} | "
                    f"val_reward_max={val_metrics['val_reward_max']:.4f} | "
                    f"val_reward_geomean={val_metrics['val_reward_geomean']:.4f}"
                )

                if val_metrics["val_reward_mean"] > best_val_reward:
                    best_val_reward = val_metrics["val_reward_mean"]
                    best_dir = os.path.join(work_dir, "best_model")
                    trainer.model.save_pretrained(best_dir)
                    dec_tok.save_pretrained(best_dir)
                    logger.info(f"New best model saved: val_reward_mean={best_val_reward:.4f}")

            # ---- periodic checkpoint ----
            if global_step % save_steps == 0:
                ckpt_dir = os.path.join(work_dir, f"checkpoint_step_{global_step}")
                trainer.model.save_pretrained(ckpt_dir)
                dec_tok.save_pretrained(ckpt_dir)
                logger.info(f"Checkpoint saved: {ckpt_dir}")

        # ---- epoch summary ----
        if epoch_rewards:
            epoch_mean = sum(epoch_rewards) / len(epoch_rewards)
            epoch_geomean = compute_geomean(epoch_rewards)
            writer.add_scalar("epoch/reward_mean", epoch_mean, epoch + 1)
            writer.add_scalar("epoch/reward_geomean", epoch_geomean, epoch + 1)
            logger.info(
                f"Epoch {epoch + 1} summary | "
                f"samples={len(epoch_rewards)} | "
                f"reward_mean={epoch_mean:.4f} | "
                f"reward_geomean={epoch_geomean:.4f} | "
                f"skipped={skipped}"
            )

    # ---- save final model ----
    final_dir = os.path.join(work_dir, "final_model")
    trainer.model.save_pretrained(final_dir)
    dec_tok.save_pretrained(final_dir)
    logger.info(f"Training finished. Final model: {final_dir}")

    writer.close()
    logger.info("TensorBoard writer closed.")


if __name__ == "__main__":
    main()
