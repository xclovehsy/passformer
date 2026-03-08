import os
import copy
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.grpo.llvm_wrapper import llvm_wrapper


class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) with per-step reward for Passformer.

    Improvements over REINFORCE:
    1. Clipped surrogate objective prevents destructively large policy updates.
    2. Multiple optimization epochs per rollout for better sample efficiency.
    3. Approximate KL-based early stopping within PPO epochs.
    4. Per-step (per-pass) reward via env.step() for fine-grained credit assignment.
    5. Discounted returns with mean-rollout baseline for variance reduction.
    6. Optional entropy bonus and KL penalty against reference model.
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
        self.gamma = float(rl.get("gamma", 0.99))

        temps = rl.get("temperatures", None)
        if temps and isinstance(temps, list) and len(temps) > 1:
            self.temperatures = [float(t) for t in temps]
        else:
            self.temperatures = [float(rl.get("temperature", 0.7))]
        self._assign_rollout_splits()
        self.entropy_coeff = float(rl.get("entropy_coeff", 0.01))
        self.kl_coeff = float(rl.get("kl_coeff", 0.0))
        self.max_grad_norm = float(rl.get("max_grad_norm", 1.0))
        self.reward_norm = rl.get("reward_norm", True)
        self.num_reward_workers = int(rl.get("num_reward_workers", self.num_rollouts))

        self.ppo_epochs = int(rl.get("ppo_epochs", 4))
        self.ppo_clip_eps = float(rl.get("ppo_clip_eps", 0.2))
        self.target_kl = float(rl.get("target_kl", 0.02))

        self.pad_id = dec_tok.pad_token_id
        self.eos_id = dec_tok.eos_token_id
        self.bos_id = dec_tok.bos_token_id
        self.special_ids = {self.pad_id, self.eos_id, self.bos_id}

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg["training"]["learning_rate"]),
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _assign_rollout_splits(self):
        """Evenly distribute num_rollouts across temperatures."""
        n_temps = len(self.temperatures)
        base, remainder = divmod(self.num_rollouts, n_temps)
        self.rollout_splits = [base + (1 if i < remainder else 0)
                               for i in range(n_temps)]

    # ------------------------------------------------------------------
    # 1. Rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(self, batch):
        """Sample sequences with mixed temperatures.

        Each temperature group generates a subset of rollouts.
        Results are concatenated so downstream code sees [B*K, L] as before.

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

        if len(self.temperatures) == 1:
            sequences = self.model.generate(
                input_ids=exp_ids,
                attention_mask=exp_mask,
                autophase=exp_auto,
                max_length=self.max_gen_length,
                do_sample=True,
                temperature=self.temperatures[0],
                pad_token_id=self.pad_id,
                eos_token_id=self.eos_id,
            )
        else:
            B = inputs["input_ids"].shape[0]
            seq_groups = []
            for temp, count in zip(self.temperatures, self.rollout_splits):
                if count == 0:
                    continue
                grp_ids = inputs["input_ids"].repeat_interleave(count, dim=0)
                grp_mask = inputs["attention_mask"].repeat_interleave(count, dim=0)
                grp_auto = autophases.repeat_interleave(count, dim=0)
                grp_seq = self.model.generate(
                    input_ids=grp_ids,
                    attention_mask=grp_mask,
                    autophase=grp_auto,
                    max_length=self.max_gen_length,
                    do_sample=True,
                    temperature=temp,
                    pad_token_id=self.pad_id,
                    eos_token_id=self.eos_id,
                )
                seq_groups.append(grp_seq)

            max_len = max(s.shape[1] for s in seq_groups)
            padded = []
            for s in seq_groups:
                if s.shape[1] < max_len:
                    pad = torch.full(
                        (s.shape[0], max_len - s.shape[1]),
                        self.pad_id, dtype=s.dtype, device=s.device,
                    )
                    s = torch.cat([s, pad], dim=1)
                padded.append(s)

            if B == 1:
                sequences = torch.cat(padded, dim=0)
            else:
                interleaved = []
                for b in range(B):
                    offset = 0
                    for grp_idx, count in enumerate(self.rollout_splits):
                        interleaved.append(padded[grp_idx][b * count : (b + 1) * count])
                        offset += count
                sequences = torch.cat(interleaved, dim=0)

        enc_inputs = {
            "input_ids": exp_ids,
            "attention_mask": exp_mask,
            "autophase": exp_auto,
        }
        return sequences, enc_inputs

    # ------------------------------------------------------------------
    # 2. Per-step reward
    # ------------------------------------------------------------------
    def _eval_single_rollout(self, token_ids: List[int], bc_path: str,
                             gen_len: int) -> List[float]:
        """Evaluate one rollout sequence against the LLVM environment.

        Thread-safe: each call creates its own env instance.
        """
        seq_rewards: List[float] = []
        env = None
        try:
            env = llvm_wrapper([bc_path], is_from_bc=True)
            env.reset()
            flag_to_id = {f: i for i, f in enumerate(env.action_space.flags)}

            for token_id in token_ids:
                if token_id in self.special_ids and token_id != 126:
                    break
                pass_flag = self.dec_tok.ids_to_tokens.get(token_id)
                if pass_flag is None or pass_flag not in flag_to_id:
                    seq_rewards.append(0.0)
                    continue
                _, reward, done, _ = env.env.step(flag_to_id[pass_flag])
                seq_rewards.append(float(reward))
                if done:
                    break
        except Exception as e:
            self.logger.warning(f"Rollout eval failed for {bc_path}: {e}")
            if not seq_rewards:
                seq_rewards.append(0.0)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        seq_rewards = seq_rewards[:gen_len]
        seq_rewards += [0.0] * (gen_len - len(seq_rewards))
        return seq_rewards

    def compute_step_rewards(self, sequences, bc_paths):
        """Execute each generated pass one-by-one and record per-step reward.

        Uses a thread pool to evaluate rollouts in parallel.

        Parameters
        ----------
        sequences : LongTensor [N, L]  (N = batch * num_rollouts)
        bc_paths  : list[str]  length = batch (NOT expanded)

        Returns
        -------
        step_rewards  : FloatTensor [N, gen_len]
        total_rewards : FloatTensor [N]
        """
        gen_tokens = sequences[:, 1:]
        N, gen_len = gen_tokens.shape
        token_lists = gen_tokens.tolist()

        all_rewards: List[Optional[List[float]]] = [None] * N

        with ThreadPoolExecutor(max_workers=self.num_reward_workers) as pool:
            futures = {}
            for i in range(N):
                bc_path = bc_paths[i // self.num_rollouts]
                fut = pool.submit(self._eval_single_rollout,
                                  token_lists[i], bc_path, gen_len)
                futures[fut] = i

            for fut in as_completed(futures):
                all_rewards[futures[fut]] = fut.result()

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
    # 5. Full training step (PPO)
    # ------------------------------------------------------------------
    def train_step(self, batch) -> Dict[str, float]:
        """One PPO update.

        Flow: rollout → per-step rewards → returns → advantages
              → compute old log-probs → multi-epoch clipped PPO updates.
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

        # ---- decoder inputs (shared across PPO epochs) ----
        decoder_input_ids = sequences[:, :-1]
        decoder_attention_mask = (decoder_input_ids != self.pad_id).long()
        valid = masks.sum().clamp(min=1)

        # ---- old log-probs from rollout policy (frozen) ----
        with torch.no_grad():
            self.model.eval()
            old_outputs = self.model(
                input_ids=enc_inputs["input_ids"],
                attention_mask=enc_inputs["attention_mask"],
                autophase=enc_inputs["autophase"],
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=True,
            )
            old_log_probs = F.log_softmax(old_outputs.logits, dim=-1)
            old_token_lp = old_log_probs.gather(
                2, gen_tokens.unsqueeze(-1)
            ).squeeze(-1)  # [N, gen_len]

        # ---- ref model log-probs (if KL penalty is used) ----
        ref_token_lp = None
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
                ref_token_lp = ref_lp.gather(
                    2, gen_tokens.unsqueeze(-1)
                ).squeeze(-1)

        # ---- PPO multi-epoch updates ----
        last_metrics: Dict[str, float] = {}
        adv_detached = advantages.detach()

        for ppo_ep in range(self.ppo_epochs):
            self.model.train()
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
            token_lp = log_probs.gather(
                2, gen_tokens.unsqueeze(-1)
            ).squeeze(-1)

            # PPO clipped surrogate objective
            ratio = torch.exp(token_lp - old_token_lp.detach())
            surr1 = ratio * adv_detached
            surr2 = torch.clamp(
                ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps
            ) * adv_detached
            policy_loss = -(torch.min(surr1, surr2) * masks).sum() / valid

            # entropy bonus
            ent = -(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1)
            entropy_bonus = (ent * masks).sum() / valid
            entropy_loss = -self.entropy_coeff * entropy_bonus

            # KL penalty against reference model
            kl_loss = torch.tensor(0.0, device=self.device)
            if self.kl_coeff > 0 and ref_token_lp is not None:
                kl = (torch.exp(ref_token_lp - token_lp)
                      - (ref_token_lp - token_lp) - 1)
                kl_loss = self.kl_coeff * (kl * masks).sum() / valid

            total_loss = policy_loss + entropy_loss + kl_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            # PPO diagnostics
            with torch.no_grad():
                clip_frac = (
                    (ratio - 1.0).abs() > self.ppo_clip_eps
                ).float().mean().item()
                approx_kl = (
                    (ratio - 1.0) - torch.log(ratio)
                ).mean().item()

            last_metrics = {
                "loss": total_loss.item(),
                "policy_loss": policy_loss.item(),
                "entropy": entropy_bonus.item(),
                "kl": kl_loss.item(),
                "clip_frac": clip_frac,
                "approx_kl": approx_kl,
                "ppo_epochs_actual": ppo_ep + 1,
            }

            if approx_kl > self.target_kl:
                self.logger.info(
                    f"  PPO early stop at epoch {ppo_ep + 1}/{self.ppo_epochs} "
                    f"(approx_kl={approx_kl:.4f} > target={self.target_kl})"
                )
                break

        # best commandline
        best_idx = total_rewards.argmax().item()
        best_seq = sequences[best_idx : best_idx + 1]
        best_commandline = self.dec_tok.decode(
            best_seq.squeeze(0), skip_special_tokens=True
        )

        last_metrics.update({
            "reward_mean": total_rewards.mean().item(),
            "reward_max": total_rewards.max().item(),
            "seq_len_mean": masks.sum(dim=-1).mean().item(),
            "best_commandline": best_commandline,
        })

        if len(self.temperatures) > 1:
            B = len(bc_paths)
            offset = 0
            for temp, count in zip(self.temperatures, self.rollout_splits):
                grp_rewards = []
                for b in range(B):
                    start = b * self.num_rollouts + offset
                    end = start + count
                    grp_rewards.append(total_rewards[start:end])
                grp = torch.cat(grp_rewards)
                tag = f"t{temp:.1f}"
                last_metrics[f"reward_mean_{tag}"] = grp.mean().item()
                last_metrics[f"reward_max_{tag}"] = grp.max().item()
                offset += count

        return last_metrics


REINFORCETrainer = PPOTrainer
