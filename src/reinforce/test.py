"""
使用 eval_llvm_instcount_policy 对 Passformer 进行 Leaderboard 评测。

策略：多温度采样 (Multi-temperature sampling)
  - 对每个 benchmark，通过多个温度生成 N 个 pass sequence 候选
  - 为每个候选序列 fork 一个独立的 env 并逐步执行，记录累计 reward
  - 将 reward 最高的候选序列应用到原始 env

用法:
    python -m src.reinforce.test \\
        --model_path /path/to/model \\
        --encoder_tokenizer_path /path/to/enc_tok \\
        --decoder_tokenizer_path /path/to/dec_tok \\
        [--num_rollouts 16] \\
        [--temperatures 0.3,0.7] \\
        [--max_gen_length 32] \\
        [--max_input_length 512] \\
        [--leaderboard_results passformer_results.csv]
"""

import os
import sys
import logging
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from absl import app, flags

from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy

from src.model import PassformerModel, Inst2VecTokenizer, OptiSeqTokenizer


# ======================================================================
# Flags
# ======================================================================

flags.DEFINE_string(
    "model_path",
    None,
    "Path to the pretrained Passformer model directory.",
)
flags.DEFINE_string(
    "encoder_tokenizer_path",
    None,
    "Path to the Inst2VecTokenizer directory. "
    "Defaults to <model_path>/encoder_tokenizer.",
)
flags.DEFINE_string(
    "decoder_tokenizer_path",
    None,
    "Path to the OptiSeqTokenizer directory. "
    "Defaults to <model_path>/decoder_tokenizer.",
)
flags.DEFINE_integer(
    "num_rollouts",
    16,
    "Number of candidate pass sequences to sample per benchmark.",
)
flags.DEFINE_list(
    "temperatures",
    ["0.3", "0.7"],
    "Comma-separated list of sampling temperatures for multi-temperature rollout.",
)
flags.DEFINE_integer(
    "max_gen_length",
    32,
    "Maximum number of tokens (passes) to generate per sequence.",
)
flags.DEFINE_integer(
    "max_input_length",
    512,
    "Maximum token length for encoding the LLVM IR.",
)
flags.DEFINE_integer(
    "num_eval_workers",
    16,
    "Number of parallel workers for evaluating candidate sequences.",
)

FLAGS = flags.FLAGS

logger = logging.getLogger("passformer_test")


# ======================================================================
# Multi-temperature sampling
# ======================================================================

@torch.no_grad()
def sample_sequences(
    model: PassformerModel,
    enc_inputs: dict,
    dec_tok: OptiSeqTokenizer,
    num_rollouts: int,
    temperatures: List[float],
    max_gen_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Multi-temperature sampling：按温度均分 rollout，拼接所有候选序列。

    Returns
    -------
    sequences : LongTensor [num_rollouts, seq_len]
    """
    model.eval()
    n_temps = len(temperatures)
    base, remainder = divmod(num_rollouts, n_temps)
    rollout_splits = [base + (1 if i < remainder else 0) for i in range(n_temps)]

    seq_groups: List[torch.Tensor] = []
    for temp, count in zip(temperatures, rollout_splits):
        if count == 0:
            continue
        ids = enc_inputs["input_ids"].repeat_interleave(count, dim=0)
        mask = enc_inputs["attention_mask"].repeat_interleave(count, dim=0)
        kwargs = dict(
            input_ids=ids,
            attention_mask=mask,
            max_length=max_gen_length,
            do_sample=True,
            temperature=temp,
            pad_token_id=dec_tok.pad_token_id,
            eos_token_id=dec_tok.eos_token_id,
        )
        if "autophase" in enc_inputs:
            kwargs["autophase"] = enc_inputs["autophase"].repeat_interleave(count, dim=0)
        seqs = model.generate(**kwargs)
        seq_groups.append(seqs)

    max_len = max(s.shape[1] for s in seq_groups)
    padded: List[torch.Tensor] = []
    for s in seq_groups:
        if s.shape[1] < max_len:
            pad = torch.full(
                (s.shape[0], max_len - s.shape[1]),
                dec_tok.pad_token_id,
                dtype=s.dtype,
                device=s.device,
            )
            s = torch.cat([s, pad], dim=1)
        padded.append(s)

    return torch.cat(padded, dim=0)  # [num_rollouts, seq_len]


# ======================================================================
# Evaluate a single candidate sequence using a forked env
# ======================================================================

def _write_log(msg: str):
    with open("test_debug.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def eval_sequence_in_forked_env(
    env: LlvmEnv,
    token_ids: List[int],
    dec_tok: OptiSeqTokenizer,
    flag_to_idx: dict,
    special_ids: set,
) -> Tuple[float, List[int]]:
    """Fork env 后逐步执行 token_ids 中的 pass，返回 (累计 reward, 执行的 action 列表)。"""
    forked = env.fork()
    total_reward = 0.0
    applied_actions: List[int] = []
    skipped = 0
    try:
        for step_i, tid in enumerate(token_ids):
            if tid in special_ids:
                # _write_log(f"    step {step_i}: hit special token {tid}, stop")
                continue
            flag = dec_tok.ids_to_tokens.get(tid)
            if flag is None or flag not in flag_to_idx:
                skipped += 1
                continue
            action_idx = flag_to_idx[flag]
            _, reward, done, _ = forked.step(action_idx)
            total_reward += float(reward)
            applied_actions.append(action_idx)
            # _write_log(
            #     f"    step {step_i}: pass={flag}, reward={reward:.4f}, "
            #     f"cumulative={total_reward:.4f}, done={done}"
            # )
            if done:
                break
    except Exception as e:
        # _write_log(f"    ERROR: Forked env evaluation failed: {e}")
        logger.warning(f"Forked env evaluation failed: {e}")
    finally:
        # _write_log(
        #     f"    Summary: total_reward={total_reward:.4f}, "
        #     f"applied={len(applied_actions)}, skipped={skipped}"
        # )
        try:
            forked.close()
        except Exception:
            pass
    return total_reward, applied_actions


# ======================================================================
# Passformer policy
# ======================================================================

class PassformerPolicy:
    """Stateful policy class，持有模型和 tokenizer，供 eval_llvm_instcount_policy 调用。"""

    def __init__(
        self,
        model: PassformerModel,
        enc_tok: Inst2VecTokenizer,
        dec_tok: OptiSeqTokenizer,
        device: torch.device,
        num_rollouts: int,
        temperatures: List[float],
        max_gen_length: int,
        max_input_length: int,
        num_eval_workers: int = 16,
    ):
        self.model = model
        self.enc_tok = enc_tok
        self.dec_tok = dec_tok
        self.device = device
        self.num_rollouts = num_rollouts
        self.temperatures = temperatures
        self.max_gen_length = max_gen_length
        self.max_input_length = max_input_length
        self.num_eval_workers = num_eval_workers

        self.special_ids = {
            dec_tok.pad_token_id,
            dec_tok.eos_token_id,
            dec_tok.bos_token_id,
        } - {None}

    def _log(self, msg: str):
        _write_log(msg)

    def __call__(self, env: LlvmEnv) -> None:
        """对单个 benchmark 执行 Passformer 多温度采样，并应用 reward 最高的序列。"""
        benchmark_name = str(env.benchmark)
        self._log(f"[{benchmark_name}] === Start evaluation ===")

        # ---- 1. 获取观测 ----
        try:
            llvm_ir: str = env.observation["Ir"]
            autophase: np.ndarray = np.array(env.observation["Autophase"], dtype=np.float32)
            self._log(
                f"[{benchmark_name}] Observation: IR length={len(llvm_ir)}, "
                f"autophase shape={autophase.shape}, "
                f"autophase sum={autophase.sum():.0f}"
            )
        except Exception as e:
            logger.warning(f"[{benchmark_name}] Failed to get observation: {e}")
            return

        # ---- 2. 准备 encoder 输入 ----
        try:
            encoded = self.enc_tok(
                [llvm_ir],
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
                return_tensors="pt",
            )
            enc_inputs = {
                "input_ids": encoded["input_ids"].to(self.device),
                "attention_mask": encoded["attention_mask"].to(self.device),
                "autophase": torch.tensor(autophase, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device),
            }
            self._log(
                f"[{benchmark_name}] Encoder input: "
                f"input_ids shape={enc_inputs['input_ids'].shape}, "
                f"attention_mask non-pad={enc_inputs['attention_mask'].sum().item()}"
            )
        except Exception as e:
            logger.warning(f"[{benchmark_name}] Failed to encode observation: {e}")
            return

        # ---- 3. 多温度采样，生成 num_rollouts 个候选序列 ----
        try:
            sequences = sample_sequences(
                model=self.model,
                enc_inputs=enc_inputs,
                dec_tok=self.dec_tok,
                num_rollouts=self.num_rollouts,
                temperatures=self.temperatures,
                max_gen_length=self.max_gen_length,
                device=self.device,
            )
            self._log(
                f"[{benchmark_name}] Sampled {sequences.shape[0]} sequences, "
                f"seq_len={sequences.shape[1]}"
            )
        except Exception as e:
            logger.warning(f"[{benchmark_name}] Sampling failed: {e}")
            return

        # ---- 4. 构建 pass flag → action index 映射 ----
        flag_to_idx = {f: i for i, f in enumerate(env.action_space.flags)}

        # sequences: [num_rollouts, seq_len]，第 0 列是 BOS，从第 1 列开始是生成的 token
        gen_tokens = sequences[:, 1:]

        # ---- 5. 对每个候选序列在 forked env 中并行评估 reward ----
        best_reward = float("-inf")
        best_actions: List[int] = []
        all_rewards = [None] * gen_tokens.shape[0]
        all_actions = [None] * gen_tokens.shape[0]

        num_workers = min(gen_tokens.shape[0], self.num_eval_workers)
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {}
            for i in range(gen_tokens.shape[0]):
                token_ids: List[int] = gen_tokens[i].tolist()
                fut = pool.submit(
                    eval_sequence_in_forked_env,
                    env=env,
                    token_ids=token_ids,
                    dec_tok=self.dec_tok,
                    flag_to_idx=flag_to_idx,
                    special_ids=self.special_ids,
                )
                futures[fut] = i

            for fut in as_completed(futures):
                i = futures[fut]
                reward, actions = fut.result()
                all_rewards[i] = reward
                all_actions[i] = actions
                decoded_flags = [
                    self.dec_tok.ids_to_tokens.get(t, "<unk>")
                    for t in gen_tokens[i].tolist()
                    if t not in self.special_ids
                ]
                self._log(
                    f"[{benchmark_name}] Rollout {i}: reward={reward:.4f}, "
                    f"actions={len(actions)}, passes={decoded_flags}"
                )
                if reward > best_reward:
                    best_reward = reward
                    best_actions = actions

        self._log(
            f"[{benchmark_name}] Rollout rewards: "
            f"mean={np.mean(all_rewards):.4f}, max={np.max(all_rewards):.4f}, "
            f"min={np.min(all_rewards):.4f}, std={np.std(all_rewards):.4f}"
        )

        # ---- 6. 将 reward 最高的 pass sequence 应用到原始 env ----
        if not best_actions:
            self._log(f"[{benchmark_name}] No valid actions found, skipping.")
            return

        best_flags = [env.action_space.flags[a] for a in best_actions]
        self._log(
            f"[{benchmark_name}] Best reward={best_reward:.4f}, "
            f"num_passes={len(best_actions)}, "
            f"passes={best_flags}"
        )
        logger.info(
            f"[{benchmark_name}] Best reward={best_reward:.4f}, "
            f"num_passes={len(best_actions)}"
        )
        for action_idx in best_actions:
            _, _, done, _ = env.step(action_idx)
            if done:
                self._log(f"[{benchmark_name}] Env done after applying actions.")
                break


# ======================================================================
# Entry point
# ======================================================================

def build_policy(argv) -> PassformerPolicy:
    """解析 flags，加载模型和 tokenizer，构建 policy 对象。"""
    assert len(argv) == 1, f"Unknown args: {argv[1:]}"

    model_path = FLAGS.model_path
    if model_path is None:
        raise ValueError("--model_path is required.")

    enc_tok_path = FLAGS.encoder_tokenizer_path or os.path.join(
        model_path, "encoder_tokenizer"
    )
    dec_tok_path = FLAGS.decoder_tokenizer_path or os.path.join(
        model_path, "decoder_tokenizer"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Loading model from: {model_path}")

    enc_tok = Inst2VecTokenizer.from_pretrained(enc_tok_path)
    dec_tok = OptiSeqTokenizer.from_pretrained(dec_tok_path)
    model = PassformerModel.from_pretrained(model_path).to(device).eval()

    temperatures = [float(t) for t in FLAGS.temperatures]

    logger.info(
        f"Policy config: num_rollouts={FLAGS.num_rollouts}, "
        f"temperatures={temperatures}, "
        f"max_gen_length={FLAGS.max_gen_length}, "
        f"num_eval_workers={FLAGS.num_eval_workers}"
    )

    return PassformerPolicy(
        model=model,
        enc_tok=enc_tok,
        dec_tok=dec_tok,
        device=device,
        num_rollouts=FLAGS.num_rollouts,
        temperatures=temperatures,
        max_gen_length=FLAGS.max_gen_length,
        max_input_length=FLAGS.max_input_length,
        num_eval_workers=FLAGS.num_eval_workers,
    )


def main(argv):
    logging.basicConfig(level=logging.INFO)
    policy = build_policy(argv)
    eval_llvm_instcount_policy(policy)


if __name__ == "__main__":
    app.run(main)
