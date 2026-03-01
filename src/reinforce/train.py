import os
import torch
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset

from src.reinforce.trainer import REINFORCETrainer
from src.rl.llvm_wrapper import llvm_wrapper
from src.model import PassformerModel, Inst2VecTokenizer, OptiSeqTokenizer
from src.config import load_config
from src.utils.utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="REINFORCE Training for Compiler Pass Optimization")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def build_dataset(cfg):
    """Collect initial observations from bc files."""
    all_data = {"llvm_ir": [], "autophase": [], "bc_path": []}
    for bc in cfg["data"]["bc_files"]:
        env = llvm_wrapper([bc], is_from_bc=True)
        obs = env.reset()
        all_data["llvm_ir"].append(obs.llvm_ir)
        all_data["autophase"].append(obs.autophase)
        all_data["bc_path"].append(bc)
        env.close()
    return Dataset.from_dict(all_data)


def collate_fn(batch):
    return {
        "llvm_ir": [x["llvm_ir"] for x in batch],
        "autophase": torch.stack([torch.tensor(x["autophase"]) for x in batch]),
        "bc_path": [x["bc_path"] for x in batch],
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(cfg["output"]["base_work_dir"], time_str)
    os.makedirs(work_dir, exist_ok=True)
    logger = get_logger(work_dir)
    logger.info(f"REINFORCE training  work_dir={work_dir}")

    # ---- model & tokenizers ----
    enc_tok = Inst2VecTokenizer.from_pretrained(cfg["model"]["encoder_tokenizer_id"])
    dec_tok = OptiSeqTokenizer.from_pretrained(cfg["model"]["decoder_tokenizer_id"])
    model = PassformerModel.from_pretrained(cfg["model"]["model_id"])
    logger.info("Model and tokenizers loaded")

    # ---- dataset ----
    dataset = build_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collate_fn,
    )
    logger.info(f"Dataset: {len(dataset)} benchmarks, batch_size={cfg['training']['batch_size']}")

    # ---- trainer ----
    trainer = REINFORCETrainer(cfg, model, enc_tok, dec_tok, logger, work_dir)

    # ---- training loop ----
    global_step = 0
    epochs = int(cfg["training"]["epochs"])
    save_steps = int(cfg["training"].get("save_steps", 50))
    log_steps = int(cfg["training"].get("log_steps", 1))

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            metrics = trainer.train_step(batch)
            global_step += 1

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "rew": f"{metrics['reward_mean']:.4f}",
            })

            if global_step % log_steps == 0:
                parts = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                logger.info(f"Step {global_step} | {parts}")

            if global_step % save_steps == 0:
                ckpt = os.path.join(work_dir, f"step_{global_step}")
                trainer.model.save_pretrained(ckpt)
                logger.info(f"Checkpoint saved: {ckpt}")

    # ---- save final model ----
    final = os.path.join(work_dir, "final_model")
    trainer.model.save_pretrained(final)
    dec_tok.save_pretrained(final)
    logger.info(f"Training finished. Final model: {final}")


if __name__ == "__main__":
    main()
