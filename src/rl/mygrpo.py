import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

class GRPOTrainerManual:
    def __init__(self, model, encoder_tokenizer, decoder_tokenizer, reward_fn, 
                 beta=0.01, eps=0.2, lr=1e-5, num_generations=8):
        self.model = model
        # 参考模型，用于计算 KL 散度，保持冻结
        self.ref_model = copy.deepcopy(model).eval()
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.reward_fn = reward_fn
        
        self.beta = beta           # KL 惩罚系数
        self.eps = eps             # PPO 剪切阈值
        self.num_gens = num_generations
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def get_log_probs(self, logits, tokens):
        """计算生成的 token 的 log 概率"""
        log_probs = F.log_softmax(logits, dim=-1)
        # 选取实际生成的 token 对应的概率
        per_token_logps = torch.gather(log_probs, dim=2, index=tokens.unsqueeze(-1)).squeeze(-1)
        return per_token_logps

    def train_step(self, batch):
        """
        batch: 包含 'prompt' (IR文本), 'autophase', 'bc_path'
        """
        device = self.model.device
        prompts = batch['prompt']
        autophases = batch['autophase'].to(device)
        bc_paths = batch['bc_path']

        # --- 1. 采样阶段 (Rollout) ---
        self.model.eval()
        with torch.no_grad():
            # 为每个 prompt 生成多组序列
            # 编码输入
            inputs = self.encoder_tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            
            # 生成结果
            # 注意：这里需要重复输入以实现 Batch 生成
            expanded_input_ids = inputs.input_ids.repeat_interleave(self.num_gens, dim=0)
            expanded_attention_mask = inputs.attention_mask.repeat_interleave(self.num_gens, dim=0)
            expanded_autophase = autophases.repeat_interleave(self.num_gens, dim=0)
            
            output_ids = self.model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                autophase=expanded_autophase,
                max_length=20,
                do_sample=True # 必须开启采样以保证多样性
            )
            # 提取生成的部分 (假设 output_ids 包含 prompt，需切片；若模型直接返回 completion 则不用)
            completions = output_ids 
            
            # 解码为文本以获取 Reward
            completion_texts = self.decoder_tokenizer.batch_decode(completions, skip_special_tokens=True)

        # --- 2. 奖励计算 (Reward) ---
        # 扩展 bc_paths 以匹配生成数量
        expanded_bc_paths = [p for p in bc_paths for _ in range(self.num_gens)]
        rewards = self.reward_fn(completion_texts, expanded_bc_paths)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        # --- 3. 计算优势 (Advantage) ---
        # 按组归一化：(r - mean) / std
        rewards_reshaped = rewards.view(-1, self.num_gens)
        mean_r = rewards_reshaped.mean(dim=1, keepdim=True)
        std_r = rewards_reshaped.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_reshaped - mean_r) / std_r).view(-1)

        # --- 4. 优化阶段 (Update) ---
        self.model.train()
        # 计算当前模型的 log_probs
        outputs = self.model(
            input_ids=expanded_input_ids,
            attention_mask=expanded_attention_mask,
            autophase=expanded_autophase,
            labels=completions # 强制模型计算这些 token 的 logits
        )
        current_log_probs = self.get_log_probs(outputs.logits, completions)
        
        # 计算参考模型的 log_probs (用于 KL)
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                autophase=expanded_autophase
            )
            ref_log_probs = self.get_log_probs(ref_outputs.logits, completions)

        # 重要性采样 Ratio (这里简化处理，假设采样时的 log_probs 与 current 近似)
        # 在标准的 GRPO 中，Ratio 是 current / old_sampling
        ratio = torch.exp(current_log_probs.sum(dim=-1) - current_log_probs.detach().sum(dim=-1)) 
        
        # PPO Clip Loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL Loss (避免偏离参考模型太远)
        kl = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1
        kl_loss = self.beta * kl.mean()

        total_loss = policy_loss + kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), rewards.mean().item()

if __name__ == "__main__":
    
    
    
    # 1. 准备 DataLoader
    def collate_fn(batch):
        return {
            'prompt': [x['prompt'] for x in batch],
            'autophase': torch.stack([torch.tensor(x['autophase']) for x in batch]),
            'bc_path': [x['bc_path'] for x in batch]
        }

    loader = DataLoader(datasets, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 2. 初始化手动 Trainer
    my_trainer = GRPOTrainerManual(
        model=model,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        reward_fn=compiler_gym_reward_fn,
        num_generations=8
    )

    # 3. 训练循环
    for epoch in range(3):
        pbar = tqdm(loader)
        for batch in pbar:
            loss, avg_reward = my_trainer.train_step(batch)
        pbar.set_description(f"Loss: {loss:.4f} | Reward: {avg_reward:.4f}")