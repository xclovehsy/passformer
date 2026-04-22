"""
Large-scale PPO training on shuffled dataset with TensorBoard logging.

Usage:
    python -m src.reinforce.train_largescale --config configs/reinforce_codecontest.yaml
"""

import os
import random
import math
import time
import json
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
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit training samples per epoch (for debugging).",
    )
    return parser.parse_args()


def _resolve_bc_paths(data_dir: str, entries: list[str], split_name: str, logger) -> list[str]:
    """Resolve metadata entries to absolute bc file paths."""
    resolved = []
    missing = []
    for entry in tqdm(entries, desc=f"Resolving bc paths for {split_name}"):
        bc_path = entry if os.path.isabs(entry) else os.path.join(data_dir, entry)
        bc_path = os.path.normpath(bc_path)
        if os.path.exists(bc_path):
            resolved.append(bc_path)
        else:
            missing.append(entry)

    if missing:
        preview = ", ".join(missing[:5])
        logger.info(f"Missing bc files: {missing}")
        # raise FileNotFoundError(
        #     f"{split_name}: {len(missing)} files from metadata not found under data_dir={data_dir}. "
        #     f"Examples: {preview}"
        # )
    return resolved


def load_datasets_from_metadata(data_dir: str, logger):
    """Load train/test/valid datasets from metadata_small_*.json files."""
    metadata_files = {
        "train": "metadata_small_train.json",
        "test": "metadata_small_test.json",
        "valid": "metadata_small_valid.json",
    }

    splits = {}
    for split_name, file_name in metadata_files.items():
        metadata_path = os.path.join(data_dir, file_name)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError(f"Metadata must be a list: {metadata_path}")
        logger.info(f"Loaded {len(entries)} entries for {split_name}")
        splits[split_name] = _resolve_bc_paths(data_dir, entries, split_name, logger)
        logger.info(f"Loaded {len(splits[split_name])} {split_name} samples")

    return splits["train"], splits["test"], splits["valid"]


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


def compute_percentile(values: list[float], percentile: float) -> float:
    """Compute percentile with linear interpolation on sorted values."""
    if not values:
        return 0.0
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    rank = (n - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_vals[lower]
    weight = rank - lower
    return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight


@torch.no_grad()
def rollout_greedy_once(trainer, batch):
    """Generate one greedy sequence for each sample in batch."""
    trainer.model.eval()

    llvm_irs = batch["llvm_ir"]
    autophases = batch["autophase"].to(trainer.device)
    inputs = trainer.enc_tok(
        llvm_irs,
        padding=True,
        truncation=True,
        max_length=trainer.cfg["data"]["max_length"],
        return_tensors="pt",
    )
    inputs = {k: v.to(trainer.device) for k, v in inputs.items()}

    sequences = trainer.model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        autophase=autophases,
        max_length=trainer.max_gen_length,
        do_sample=False,
        num_beams=1,
        pad_token_id=trainer.pad_id,
        eos_token_id=trainer.eos_id,
    )
    return sequences


@torch.no_grad()
def evaluate(trainer, val_files: list[str], logger, global_step: int):
    """Evaluate on full validation split using single greedy rollout."""
    total_rewards = []
    pass_lengths = []
    sample_records = []

    for bc_path in val_files:
        try:
            batch = build_single_sample(bc_path)
        except Exception as e:
            logger.warning(f"Skipping val sample {bc_path}: {e}")
            continue
        sequences = rollout_greedy_once(trainer, batch)
        _, rewards = trainer.compute_step_rewards(sequences, batch["bc_path"])
        reward_value = rewards[0].item()
        total_rewards.append(reward_value)
        pass_sequence = trainer.dec_tok.decode(
            sequences[0].detach().cpu(),
            skip_special_tokens=True,
        )
        pass_length = len(pass_sequence.split()) if pass_sequence.strip() else 0
        pass_lengths.append(pass_length)
        sample_records.append(
            {
                "bc_filename": os.path.basename(bc_path),
                "bc_path": bc_path,
                "reward": reward_value,
                "pass_sequence": pass_sequence,
                "pass_length": pass_length,
            }
        )

    eval_records_dir = os.path.join(trainer.work_dir, "eval_records")
    os.makedirs(eval_records_dir, exist_ok=True)
    eval_records_path = os.path.join(eval_records_dir, f"eval_step_{global_step}.jsonl")
    if not total_rewards:
        val_metrics = {
            "val_num_samples": 0,
            "val_reward_mean": 0.0,
            "val_reward_max": 0.0,
            "val_reward_min": 0.0,
            "val_reward_std": 0.0,
            "val_reward_p25": 0.0,
            "val_reward_p50": 0.0,
            "val_reward_p75": 0.0,
            "val_reward_p90": 0.0,
            "val_reward_p95": 0.0,
            "val_reward_geomean": 0.0,
            "val_improve_rate": 0.0,
            "val_degrade_rate": 0.0,
            "val_pass_length_mean": 0.0,
            "val_pass_length_p50": 0.0,
            "val_pass_length_p90": 0.0,
            "val_pass_length_max": 0.0,
            "val_records_path": eval_records_path,
        }
    else:
        reward_mean = sum(total_rewards) / len(total_rewards)
        reward_std = math.sqrt(sum((r - reward_mean) ** 2 for r in total_rewards) / len(total_rewards))
        improve_rate = sum(1 for r in total_rewards if r > 1.0) / len(total_rewards)
        degrade_rate = sum(1 for r in total_rewards if r < 1.0) / len(total_rewards)
        pass_length_mean = sum(pass_lengths) / len(pass_lengths)

        val_metrics = {
            "val_num_samples": len(total_rewards),
            "val_reward_mean": reward_mean,
            "val_reward_max": max(total_rewards),
            "val_reward_min": min(total_rewards),
            "val_reward_std": reward_std,
            "val_reward_p25": compute_percentile(total_rewards, 25),
            "val_reward_p50": compute_percentile(total_rewards, 50),
            "val_reward_p75": compute_percentile(total_rewards, 75),
            "val_reward_p90": compute_percentile(total_rewards, 90),
            "val_reward_p95": compute_percentile(total_rewards, 95),
            "val_reward_geomean": compute_geomean(total_rewards),
            "val_improve_rate": improve_rate,
            "val_degrade_rate": degrade_rate,
            "val_pass_length_mean": pass_length_mean,
            "val_pass_length_p50": compute_percentile(pass_lengths, 50),
            "val_pass_length_p90": compute_percentile(pass_lengths, 90),
            "val_pass_length_max": max(pass_lengths),
            "val_records_path": eval_records_path,
        }

    summary_record = {
        "record_type": "summary",
        "global_step": global_step,
        "num_samples": val_metrics["val_num_samples"],
        "reward_mean": val_metrics["val_reward_mean"],
        "reward_max": val_metrics["val_reward_max"],
        "reward_min": val_metrics["val_reward_min"],
        "reward_std": val_metrics["val_reward_std"],
        "reward_p25": val_metrics["val_reward_p25"],
        "reward_p50": val_metrics["val_reward_p50"],
        "reward_p75": val_metrics["val_reward_p75"],
        "reward_p90": val_metrics["val_reward_p90"],
        "reward_p95": val_metrics["val_reward_p95"],
        "reward_geomean": val_metrics["val_reward_geomean"],
        "improve_rate": val_metrics["val_improve_rate"],
        "degrade_rate": val_metrics["val_degrade_rate"],
        "pass_length_mean": val_metrics["val_pass_length_mean"],
        "pass_length_p50": val_metrics["val_pass_length_p50"],
        "pass_length_p90": val_metrics["val_pass_length_p90"],
        "pass_length_max": val_metrics["val_pass_length_max"],
    }

    with open(eval_records_path, "w", encoding="utf-8") as f:
        for record in sample_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")
    logger.info(f"Saved eval sample records to {eval_records_path}")

    return val_metrics


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
    logger.info(f"Trainer initialized with model: {trainer.model}")

    # ---- data split ----
    data_cfg = cfg["data"]
    data_dir = data_cfg["data_dir"]
    train_files, test_files, val_files = load_datasets_from_metadata(data_dir, logger)
    if not train_files:
        raise ValueError(f"Train split is empty in metadata under {data_dir}")
    if not val_files:
        raise ValueError(f"Valid split is empty in metadata under {data_dir}")

    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max_train_samples must be > 0 when provided")
        train_files = train_files[: args.max_train_samples]
        logger.info(f"Debug mode enabled: using first {len(train_files)} train samples per epoch")

    total_files = len(train_files) + len(test_files) + len(val_files)
    logger.info(
        f"Dataset from metadata: {total_files} total, "
        f"{len(train_files)} train, {len(test_files)} test, {len(val_files)} valid"
    )

    # ---- training params ----
    train_cfg = cfg["training"]
    num_epochs = int(train_cfg.get("num_epochs", 5))
    save_steps = int(train_cfg.get("save_steps", 500))
    log_steps = int(train_cfg.get("log_steps", 10))
    eval_steps = int(train_cfg.get("eval_steps", 200))

    rl_cfg = cfg["rl"]
    ref_update_steps = int(rl_cfg.get("ref_update_steps", 0))

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
    acc_reward_mean_values = []
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
            acc_reward_mean_values.append(mean_reward)

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
                avg_reward_geomean = compute_geomean(acc_reward_mean_values)

                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/policy_loss", avg_policy_loss, global_step)
                writer.add_scalar("train/reward_mean", avg_reward_mean, global_step)
                writer.add_scalar("train/reward_max", avg_reward_max, global_step)
                writer.add_scalar("train/reward_geomean", avg_reward_geomean, global_step)
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
                    f"reward_geomean={avg_reward_geomean:.4f} | "
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
                acc_reward_mean_values = []

            # ---- periodic evaluation ----
            if global_step % eval_steps == 0:
                logger.info(f"Running evaluation at step {global_step}...")
                val_metrics = evaluate(trainer, val_files, logger, global_step)

                writer.add_scalar("val/reward_mean", val_metrics["val_reward_mean"], global_step)
                writer.add_scalar("val/reward_max", val_metrics["val_reward_max"], global_step)
                writer.add_scalar("val/reward_min", val_metrics["val_reward_min"], global_step)
                writer.add_scalar("val/reward_std", val_metrics["val_reward_std"], global_step)
                writer.add_scalar("val/reward_p25", val_metrics["val_reward_p25"], global_step)
                writer.add_scalar("val/reward_p50", val_metrics["val_reward_p50"], global_step)
                writer.add_scalar("val/reward_p75", val_metrics["val_reward_p75"], global_step)
                writer.add_scalar("val/reward_p90", val_metrics["val_reward_p90"], global_step)
                writer.add_scalar("val/reward_p95", val_metrics["val_reward_p95"], global_step)
                writer.add_scalar("val/reward_geomean", val_metrics["val_reward_geomean"], global_step)
                writer.add_scalar("val/improve_rate", val_metrics["val_improve_rate"], global_step)
                writer.add_scalar("val/degrade_rate", val_metrics["val_degrade_rate"], global_step)
                writer.add_scalar("val/pass_length_mean", val_metrics["val_pass_length_mean"], global_step)
                writer.add_scalar("val/pass_length_p50", val_metrics["val_pass_length_p50"], global_step)
                writer.add_scalar("val/pass_length_p90", val_metrics["val_pass_length_p90"], global_step)
                writer.add_scalar("val/pass_length_max", val_metrics["val_pass_length_max"], global_step)
                writer.add_scalar("val/num_samples", val_metrics["val_num_samples"], global_step)

                logger.info(
                    f"Eval step {global_step} | "
                    f"n={val_metrics['val_num_samples']} | "
                    f"val_reward_mean={val_metrics['val_reward_mean']:.4f} | "
                    f"val_reward_max={val_metrics['val_reward_max']:.4f} | "
                    f"val_reward_min={val_metrics['val_reward_min']:.4f} | "
                    f"val_reward_std={val_metrics['val_reward_std']:.4f} | "
                    f"val_reward_p50={val_metrics['val_reward_p50']:.4f} | "
                    f"val_reward_p90={val_metrics['val_reward_p90']:.4f} | "
                    f"val_reward_geomean={val_metrics['val_reward_geomean']:.4f} | "
                    f"improve_rate={val_metrics['val_improve_rate']:.2%} | "
                    f"degrade_rate={val_metrics['val_degrade_rate']:.2%} | "
                    f"pass_len_mean={val_metrics['val_pass_length_mean']:.2f} | "
                    f"pass_len_p50={val_metrics['val_pass_length_p50']:.2f} | "
                    f"records={val_metrics['val_records_path']}"
                )

                if val_metrics["val_reward_geomean"] > best_val_reward:
                    best_val_reward = val_metrics["val_reward_geomean"]
                    best_dir = os.path.join(work_dir, "best_model")
                    trainer.model.save_pretrained(best_dir)
                    dec_tok.save_pretrained(best_dir)
                    logger.info(f"New best model saved: val_reward_geomean={best_val_reward:.4f}")

            # ---- periodic checkpoint ----
            if global_step % save_steps == 0:
                ckpt_dir = os.path.join(work_dir, f"checkpoint_step_{global_step}")
                trainer.model.save_pretrained(ckpt_dir)
                dec_tok.save_pretrained(ckpt_dir)
                logger.info(f"Checkpoint saved: {ckpt_dir}")

            # ---- periodic reference model update ----
            if ref_update_steps > 0 and global_step % ref_update_steps == 0:
                trainer.update_ref_model()

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
