import os
import torch
import argparse
import yaml
import copy
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.rl.llvm_wrapper import llvm_wrapper
from src.model import PassformerModel, Inst2VecTokenizer, OptiSeqTokenizer
from src.config import load_config
from src.utils.utils import get_logger
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training for Compiler Pass Optimization")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return parser.parse_args()

def compiler_gym_reward_fn(completions, bc_paths):
    """Reward function connecting to CompilerGym."""
    rewards = []
    for completion, path in zip(completions, bc_paths):
        try:
            env = llvm_wrapper([path], is_from_bc=True)
            env.reset()
            # 执行 Pass 序列并获取 reward (代码体积变化率)
            _, reward, _, _ = env.multistep_by_action_flags(completion)
            rewards.append(float(reward))
            env.close()
        except Exception:
            rewards.append(-1.0)  # 编译失败惩罚
    return rewards

class GRPOTrainer:
    def __init__(self, cfg, model, enc_tok, dec_tok, logger, work_dir):
        self.cfg = cfg
        self.logger = logger
        self.work_dir = work_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        self.ref_model = copy.deepcopy(model).to(self.device).eval()
        self.enc_tok = enc_tok
        self.dec_tok = dec_tok
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=float(cfg["training"]["learning_rate"])
        )

    def train_step(self, batch):
        llvm_irs = batch['llvm_ir']
        autophases = batch['autophase'].to(self.device)
        bc_paths = batch['bc_path']
        num_gens = self.cfg["rl"]["num_generations"]

        # --- 1. Rollout ---
        self.model.eval()
        with torch.no_grad():
            inputs = self.enc_tok(llvm_irs, padding=True, truncation=True, 
                                  max_length=self.cfg["data"]["max_length"], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            exp_input_ids = inputs['input_ids'].repeat_interleave(num_gens, dim=0)
            exp_attn_mask = inputs['attention_mask'].repeat_interleave(num_gens, dim=0)
            exp_autophase = autophases.repeat_interleave(num_gens, dim=0)

            output_ids = self.model.generate(
                input_ids=exp_input_ids,
                attention_mask=exp_attn_mask,
                autophase=exp_autophase,
                max_length=self.cfg["rl"]["max_gen_length"],
                do_sample=True,
                temperature=self.cfg["rl"]["temperature"]
            )
            completion_texts = self.dec_tok.batch_decode(output_ids, skip_special_tokens=True)
            # print(output_ids)
            print('--------------------------------')

        # --- 2. Reward & Advantage ---
        exp_bc_paths = [p for p in bc_paths for _ in range(num_gens)]
        rewards = torch.tensor(compiler_gym_reward_fn(completion_texts, exp_bc_paths), device=self.device)
        print('rewards:', rewards)
        rewards_reshaped = rewards.view(-1, num_gens)
        advantages = ((rewards_reshaped - rewards_reshaped.mean(dim=1, keepdim=True)) / 
                      (rewards_reshaped.std(dim=1, keepdim=True) + 1e-8)).view(-1)

        # --- 3. Optimization ---
        self.model.train()
        outputs = self.model(
            input_ids=exp_input_ids, attention_mask=exp_attn_mask, autophase=exp_autophase,
            decoder_input_ids=output_ids, labels=output_ids
        )
        
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=exp_input_ids, attention_mask=exp_attn_mask, autophase=exp_autophase,
                decoder_input_ids=output_ids, labels=output_ids
            )

        def get_logps(logits, ids):
            log_probs = torch.log_softmax(logits, dim=-1)
            return torch.gather(log_probs, 2, ids.unsqueeze(-1)).squeeze(-1)

        curr_logps = get_logps(outputs.logits, output_ids)
        ref_logps = get_logps(ref_outputs.logits, output_ids)
        
        mask = (output_ids != self.dec_tok.pad_token_id).float()
        curr_logp_sum = (curr_logps * mask).sum(dim=-1)
        old_logp_sum = curr_logp_sum.detach()

        # Ratio & PPO Loss
        ratio = torch.exp(curr_logp_sum - old_logp_sum)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.cfg["rl"]["eps"], 1 + self.cfg["rl"]["eps"]) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL Loss
        kl = torch.exp(ref_logps - curr_logps) - (ref_logps - curr_logps) - 1
        kl_loss = self.cfg["rl"]["beta"] * (kl * mask).sum(dim=-1).mean()

        total_loss = policy_loss + kl_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item(), rewards.mean().item()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(cfg["output"]["base_work_dir"], time_str)
    os.makedirs(work_dir, exist_ok=True)
    logger = get_logger(work_dir)
    logger.info(f"Initialized GRPO training at {work_dir}")

    # 加载 Tokenizers 和 Model
    enc_tok = Inst2VecTokenizer.from_pretrained(cfg["model"]["encoder_tokenizer_id"])
    dec_tok = OptiSeqTokenizer.from_pretrained(cfg["model"]["decoder_tokenizer_id"])
    model = PassformerModel.from_pretrained(cfg["model"]["model_id"])
    logger.info("Model and Tokenizers loaded successfully")

    # 准备 Dataset (复用你之前的逻辑)
    all_data = {"llvm_ir": [], "autophase": [], "bc_path": []}
    for bc in cfg["data"]["bc_files"]:
        env = llvm_wrapper([bc], is_from_bc=True)
        obs = env.reset()
        all_data["llvm_ir"].append(obs.llvm_ir)
        all_data["autophase"].append(obs.autophase)
        all_data["bc_path"].append(bc)
        env.close()
    
    dataset = Dataset.from_dict(all_data)
    
    def collate_fn(batch):
        return {
            'llvm_ir': [x['llvm_ir'] for x in batch],
            'autophase': torch.stack([torch.tensor(x['autophase']) for x in batch]),
            'bc_path': [x['bc_path'] for x in batch]
        }

    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collate_fn)
    trainer = GRPOTrainer(cfg, model, enc_tok, dec_tok, logger, work_dir)

    # 训练循环
    global_step = 0
    for epoch in range(cfg["training"]["epochs"]):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            loss, avg_rew = trainer.train_step(batch)
            global_step += 1
            
            if global_step % 10 == 0:
                logger.info(f"Step {global_step} | Loss: {loss:.4f} | Reward: {avg_rew:.4f}")
            
            pbar.set_postfix({"loss": f"{loss:.4f}", "reward": f"{avg_rew:.4f}"})

            if global_step % cfg["training"]["save_steps"] == 0:
                save_path = os.path.join(work_dir, f"step_{global_step}")
                trainer.model.save_pretrained(save_path)
                logger.info(f"Checkpoint saved at step {global_step}")

    # 保存最终模型
    final_path = os.path.join(work_dir, "final_model")
    trainer.model.save_pretrained(final_path)
    dec_tok.save_pretrained(final_path)
    logger.info(f"Training finished. Final model saved at {final_path}")

if __name__ == "__main__":
    main()