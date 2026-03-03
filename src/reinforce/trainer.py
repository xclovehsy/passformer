import os
import copy
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

from src.rl.llvm_wrapper import llvm_wrapper


class REINFORCETrainer:
    """
    REINFORCE with per-step reward for Passformer.

    Key differences from GRPO:
    1. Per-step (per-pass) reward via env.step(), enabling fine-grained credit assignment.
    2. Discounted returns for each timestep, not a single sequence-level reward.
    3. Mean return across rollouts as baseline for variance reduction.
    4. Optional entropy bonus and KL penalty against reference model.

    Reference: CompilerDream (REINFORCE actor gradient on imagined trajectories).
    """

    def __init__(self, cfg, model, enc_tok, dec_tok, logger, work_dir):
        self.cfg = cfg
        self.logger = logger
        self.work_dir = work_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.ref_model = copy.deepcopy(model).to(self.device).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.enc_tok = enc_tok
        self.dec_tok = dec_tok

        rl = cfg["rl"]
        self.num_rollouts = int(rl.get("num_rollouts", 8))
        self.max_gen_length = int(rl.get("max_gen_length", 20))
        self.temperature = float(rl.get("temperature", 0.7))
        self.gamma = float(rl.get("gamma", 0.99))
        self.entropy_coeff = float(rl.get("entropy_coeff", 0.01))
        self.kl_coeff = float(rl.get("kl_coeff", 0.0))
        self.max_grad_norm = float(rl.get("max_grad_norm", 1.0))
        self.reward_norm = rl.get("reward_norm", True)

        self.pad_id = dec_tok.pad_token_id
        self.eos_id = dec_tok.eos_token_id
        self.bos_id = dec_tok.bos_token_id
        self.special_ids = {self.pad_id, self.eos_id, self.bos_id}

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg["training"]["learning_rate"]),
        )

    # ------------------------------------------------------------------
    # 1. Rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(self, batch):
        """Sample sequences from the current policy.

        Returns
        -------
        sequences : LongTensor [B*K, L]   (K = num_rollouts)
        enc_inputs : dict  expanded encoder inputs for later forward pass
        """
        self.model.eval()

        llvm_irs = batch["llvm_ir"]
        autophases = batch["autophase"].to(self.device)

        inputs = self.enc_tok(
            llvm_irs, padding=True, truncation=True,
            max_length=self.cfg["data"]["max_length"], return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        K = self.num_rollouts
        exp_ids = inputs["input_ids"].repeat_interleave(K, dim=0)
        exp_mask = inputs["attention_mask"].repeat_interleave(K, dim=0)
        exp_auto = autophases.repeat_interleave(K, dim=0)

        sequences = self.model.generate(
            input_ids=exp_ids,
            attention_mask=exp_mask,
            autophase=exp_auto,
            max_length=self.max_gen_length,
            do_sample=True,
            temperature=self.temperature,
            pad_token_id=self.pad_id,
            eos_token_id=self.eos_id,
        )

        enc_inputs = {
            "input_ids": exp_ids,
            "attention_mask": exp_mask,
            "autophase": exp_auto,
        }
        return sequences, enc_inputs

    # ------------------------------------------------------------------
    # 2. Per-step reward
    # ------------------------------------------------------------------
    def compute_step_rewards(self, sequences, bc_paths):
        """Execute each generated pass one-by-one and record per-step reward.

        Parameters
        ----------
        sequences : LongTensor [N, L]  (N = batch * num_rollouts)
        bc_paths  : list[str]  length = batch (NOT expanded)

        Returns
        -------
        step_rewards  : FloatTensor [N, gen_len]
        total_rewards : FloatTensor [N]
        """
        gen_tokens = sequences[:, 1:]  # drop decoder_start_token
        N, gen_len = gen_tokens.shape

        all_rewards: List[List[float]] = []

        for i in range(N):
            bc_idx = i // self.num_rollouts
            bc_path = bc_paths[bc_idx]

            seq_rewards: List[float] = []
            try:
                env = llvm_wrapper([bc_path], is_from_bc=True)
                env.reset()
                flags = list(env.action_space.flags)

                for token_id in gen_tokens[i].tolist():
                    if token_id in self.special_ids and token_id != 126:
                        break
                    pass_flag = self.dec_tok.ids_to_tokens.get(token_id)
                    if pass_flag is None or pass_flag not in flags:
                        seq_rewards.append(0.0)
                        continue
                    action_id = flags.index(pass_flag)
                    _, reward, done, _ = env.env.step(action_id)
                    seq_rewards.append(float(reward))
                    if done:
                        break
                env.close()
            except Exception:
                if not seq_rewards:
                    seq_rewards.append(0.0)

            # pad / truncate to gen_len
            seq_rewards = seq_rewards[:gen_len]
            seq_rewards += [0.0] * (gen_len - len(seq_rewards))
            all_rewards.append(seq_rewards)

        step_rewards = torch.tensor(all_rewards, device=self.device, dtype=torch.float32)
        total_rewards = step_rewards.sum(dim=-1)
        return step_rewards, total_rewards

    # ------------------------------------------------------------------
    # 3. Discounted returns
    # ------------------------------------------------------------------
    def compute_returns(self, step_rewards, masks):
        """R_t = r_t + γ · R_{t+1}   (only within valid positions)."""
        B, T = step_rewards.shape
        returns = torch.zeros_like(step_rewards)
        running = torch.zeros(B, device=self.device)
        for t in reversed(range(T)):
            running = step_rewards[:, t] + self.gamma * running * masks[:, t]
            returns[:, t] = running
        return returns

    # ------------------------------------------------------------------
    # 4. Baseline (mean return across rollouts of the same input)
    # ------------------------------------------------------------------
    def compute_advantages(self, returns, masks, total_rewards):
        """Advantages = returns − per-sample baseline, then normalise."""
        N = returns.shape[0]
        K = self.num_rollouts
        B = N // K

        # baseline: mean total reward across K rollouts for each input
        mean_ret = total_rewards.view(B, K).mean(dim=1, keepdim=True)  # [B,1]
        baseline = mean_ret.expand(B, K).reshape(N, 1).expand_as(returns)

        advantages = (returns - baseline) * masks

        # normalise
        valid = masks.sum().clamp(min=1)
        adv_mean = (advantages * masks).sum() / valid
        adv_var = ((advantages - adv_mean).pow(2) * masks).sum() / valid
        advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)
        return advantages

    # ------------------------------------------------------------------
    # 5. Full training step
    # ------------------------------------------------------------------
    def train_step(self, batch) -> Dict[str, float]:
        """One REINFORCE update.

        Flow: rollout → per-step rewards → returns → advantages → policy loss.
        """
        bc_paths = batch["bc_path"]

        # ---- rollout ----
        sequences, enc_inputs = self.rollout(batch)

        # ---- masks & tokens ----
        gen_tokens = sequences[:, 1:]
        masks = (gen_tokens != self.pad_id).float()

        # ---- per-step rewards ----
        step_rewards, total_rewards = self.compute_step_rewards(sequences, bc_paths)

        self.logger.info(
            f"  rewards  mean={total_rewards.mean().item():.4f}  "
            f"max={total_rewards.max().item():.4f}  "
            f"min={total_rewards.min().item():.4f}"
        )

        # ---- returns & advantages ----
        returns = self.compute_returns(step_rewards, masks)
        advantages = self.compute_advantages(returns, masks, total_rewards)

        # ---- forward pass with gradient ----
        self.model.train()
        decoder_input_ids = sequences[:, :-1]
        decoder_attention_mask = (decoder_input_ids != self.pad_id).long()

        outputs = self.model(
            input_ids=enc_inputs["input_ids"],
            attention_mask=enc_inputs["attention_mask"],
            autophase=enc_inputs["autophase"],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        logits = outputs.logits  # [N, gen_len, V]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)  # [N, gen_len]

        valid = masks.sum().clamp(min=1)

        # REINFORCE policy loss
        policy_loss = -(token_lp * advantages.detach() * masks).sum() / valid

        # entropy bonus (encourage exploration)
        ent = -(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1)
        entropy_bonus = (ent * masks).sum() / valid
        entropy_loss = -self.entropy_coeff * entropy_bonus

        # KL penalty against reference model
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.kl_coeff > 0:
            with torch.no_grad():
                ref_out = self.ref_model(
                    input_ids=enc_inputs["input_ids"],
                    attention_mask=enc_inputs["attention_mask"],
                    autophase=enc_inputs["autophase"],
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    return_dict=True,
                )
                ref_lp = F.log_softmax(ref_out.logits, dim=-1)
                ref_token_lp = ref_lp.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)
            kl = torch.exp(ref_token_lp - token_lp) - (ref_token_lp - token_lp) - 1
            kl_loss = self.kl_coeff * (kl * masks).sum() / valid

        total_loss = policy_loss + entropy_loss + kl_loss

        # ---- update ----
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # best commandline: decode the sequence with max total reward
        best_idx = total_rewards.argmax().item()
        best_seq = sequences[best_idx : best_idx + 1]
        best_commandline = self.dec_tok.decode(best_seq.squeeze(0), skip_special_tokens=True)

        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy_bonus.item(),
            "kl": kl_loss.item(),
            "reward_mean": total_rewards.mean().item(),
            "reward_max": total_rewards.max().item(),
            "seq_len_mean": masks.sum(dim=-1).mean().item(),
            "best_commandline": best_commandline,
        }
