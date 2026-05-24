"""Evaluate Passformer policy on LLVM instcount leaderboard."""
import json
import logging
import os
import time
import atexit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

import compiler_gym
import numpy as np
import torch
from absl import app, flags
from compiler_gym.datasets import Benchmark
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.envs import LlvmEnv

from eval_llvm_instcount import eval_llvm_instcount_policy
from src.model import Inst2VecTokenizer, OptiSeqTokenizer, PassformerModel

flags.DEFINE_string(
    "model_path",
    None,
    "Path to pretrained Passformer model directory.",
)
flags.DEFINE_string(
    "encoder_tokenizer_path",
    None,
    "Path to Inst2VecTokenizer; defaults to <model_path>/encoder_tokenizer.",
)
flags.DEFINE_string(
    "decoder_tokenizer_path",
    None,
    "Path to OptiSeqTokenizer; defaults to <model_path>/decoder_tokenizer.",
)
flags.DEFINE_enum(
    "decode_method",
    "sampling",
    ["greedy", "beam", "sampling", "sampling_topp"],
    "Decoding method used to generate candidate pass sequences.",
)
flags.DEFINE_integer(
    "num_samples",
    16,
    "Number of candidate samples per benchmark (greedy always returns 1).",
)
flags.DEFINE_integer(
    "max_input_length",
    512,
    "Maximum input token length for LLVM IR.",
)
flags.DEFINE_integer(
    "max_ir_lines",
    0,
    "If > 0, keep only first N IR lines before tokenizer preprocessing.",
)
flags.DEFINE_integer(
    "max_gen_length",
    32,
    "Maximum generated sequence length.",
)
flags.DEFINE_integer(
    "num_beams",
    8,
    "Beam size used when decode_method=beam.",
)
flags.DEFINE_float(
    "temperature",
    0.7,
    "Sampling temperature for sampling-based decoding.",
)
flags.DEFINE_float(
    "top_p",
    0.95,
    "Top-p for sampling_topp mode; must be in (0, 1].",
)
flags.DEFINE_integer(
    "num_eval_workers",
    8,
    "Max worker threads used to score candidate samples.",
)
flags.DEFINE_float(
    "soft_eval_time_limit_s",
    0.0,
    "Soft time budget (seconds) for repeated generate+score rounds per benchmark; 0 disables.",
)
flags.DEFINE_integer(
    "warmup_generate_rounds",
    1,
    "Number of warmup generate rounds before the first benchmark; 0 disables warmup.",
)
flags.DEFINE_integer(
    "seed",
    0,
    "Random seed; 0 disables explicit seeding.",
)
flags.DEFINE_string(
    "sample_metrics_path",
    "",
    "Optional JSONL output path for per-sample metrics. "
    "If empty, writes to <eval_output_dir>/passformer_sample_metrics.jsonl.",
)
flags.DEFINE_string(
    "benchmark_timing_path",
    "",
    "Optional JSONL output path for per-benchmark timing breakdown. "
    "If empty, writes to <eval_output_dir>/passformer_benchmark_timing.jsonl.",
)
FLAGS = flags.FLAGS

logger = logging.getLogger("passformer_eval")
_PROCESS_ENV = None


def _close_process_env() -> None:
    global _PROCESS_ENV
    if _PROCESS_ENV is not None:
        try:
            _PROCESS_ENV.close()
        except Exception:
            pass
        _PROCESS_ENV = None


def _init_process_worker() -> None:
    """Create one persistent env per worker process."""
    global _PROCESS_ENV
    _PROCESS_ENV = compiler_gym.make(
        "llvm-ic-v0",
        observation_space="Ir",
        reward_space="IrInstructionCountOz",
    )
    atexit.register(_close_process_env)


def _resolve_benchmark_for_worker(benchmark_spec: Dict[str, str]):
    uri = benchmark_spec.get("uri", "")
    path = benchmark_spec.get("path", "")
    if path:
        p = Path(path).expanduser().resolve()
        if p.is_file():
            return Benchmark.from_file(uri=uri or f"benchmark://user-v0{p}", path=p)
    return uri


def _serialize_benchmark_for_worker(benchmark) -> Dict[str, str]:
    uri = ""
    try:
        uri = str(benchmark.uri)
    except Exception:
        uri = str(benchmark)
    path = ""
    try:
        parsed = BenchmarkUri.from_string(uri)
        if parsed.dataset in ("user-v0", "file-v0") and parsed.path:
            path = parsed.path
    except Exception:
        path = ""
    return {"uri": uri, "path": path}


def _score_sequence_in_process(
    benchmark_spec: Dict[str, str],
    token_ids: List[int],
    token_to_flag: Dict[int, str],
    flag_to_action: Dict[str, int],
    special_ids: set,
) -> Tuple[float, List[int], Dict[str, object]]:
    """Evaluate one generated token sequence in a process-local env."""
    if _PROCESS_ENV is None:
        _init_process_worker()
    worker_env = _PROCESS_ENV
    t0 = time.perf_counter()
    total_reward = 0.0
    actions: List[int] = []
    metrics: Dict[str, object] = {}
    skipped_special = 0
    skipped_unknown_token = 0
    skipped_unmapped_flag = 0
    benchmark = _resolve_benchmark_for_worker(benchmark_spec)
    worker_env.reset(benchmark=benchmark)
    try:
        for token_id in token_ids:
            if token_id in special_ids:
                skipped_special += 1
                continue
            flag = token_to_flag.get(token_id)
            if not flag:
                skipped_unknown_token += 1
                continue
            action = flag_to_action.get(flag)
            if action is None:
                skipped_unmapped_flag += 1
                continue
            _, reward, done, _ = worker_env.step(action)
            total_reward += float(reward)
            actions.append(action)
            if done:
                break
        try:
            metrics["commandline"] = worker_env.commandline()
        except Exception:
            metrics["commandline"] = None
        for key in (
            "IrInstructionCount",
            "IrInstructionCountO0",
            "IrInstructionCountO3",
            "IrInstructionCountOz",
        ):
            try:
                value = worker_env.observation[key]
                if hasattr(value, "item"):
                    value = value.item()
                metrics[key] = float(value) if value is not None else None
            except Exception:
                metrics[key] = None
        metrics["sample_eval_wall_s"] = time.perf_counter() - t0
        metrics["input_token_count"] = len(token_ids)
        metrics["skipped_special_tokens"] = skipped_special
        metrics["skipped_unknown_tokens"] = skipped_unknown_token
        metrics["skipped_unmapped_flags"] = skipped_unmapped_flag
    except Exception:
        # Prevent reusing potentially unhealthy env in this worker process.
        _close_process_env()
        raise
    return total_reward, actions, metrics


class PassformerPolicy:
    """Generate candidate pass sequences and apply best one per benchmark."""

    def __init__(
        self,
        model: PassformerModel,
        enc_tok: Inst2VecTokenizer,
        dec_tok: OptiSeqTokenizer,
        device: torch.device,
    ):
        self.model = model
        self.enc_tok = enc_tok
        self.dec_tok = dec_tok
        self.device = device
        self.special_ids = {
            dec_tok.pad_token_id,
            dec_tok.eos_token_id,
            dec_tok.bos_token_id,
        } - {None}
        self.token_to_flag = dict(dec_tok.ids_to_tokens)
        self.sample_metrics_path = self._resolve_sample_metrics_path()
        self.benchmark_timing_path = self._resolve_benchmark_timing_path()
        self._warmed_up = False
        self.max_eval_workers = max(1, int(FLAGS.num_eval_workers))
        self._pool = None
        if self.max_eval_workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=self.max_eval_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_process_worker,
            )
            atexit.register(self.close)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None

    def _resolve_sample_metrics_path(self) -> str:
        if FLAGS.sample_metrics_path:
            path = os.path.abspath(FLAGS.sample_metrics_path)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            return path

        out_dir = getattr(FLAGS, "eval_output_dir", None)
        if not out_dir:
            leaderboard_results = getattr(FLAGS, "leaderboard_results", "")
            out_dir = os.path.dirname(os.path.abspath(leaderboard_results)) or "."
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, "passformer_sample_metrics.jsonl")

    def _append_sample_metrics(self, rows: List[Dict[str, object]]) -> None:
        with open(self.sample_metrics_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _resolve_benchmark_timing_path(self) -> str:
        if FLAGS.benchmark_timing_path:
            path = os.path.abspath(FLAGS.benchmark_timing_path)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            return path
        out_dir = os.path.dirname(self.sample_metrics_path)
        return os.path.join(out_dir, "passformer_benchmark_timing.jsonl")

    def _append_benchmark_timing(self, row: Dict[str, object]) -> None:
        with open(self.benchmark_timing_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _generate_candidates(
        self, llvm_ir: str, autophase: np.ndarray
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        ir_lines = llvm_ir.splitlines()
        ir_lines_total = len(ir_lines)
        if FLAGS.max_ir_lines > 0 and ir_lines_total > FLAGS.max_ir_lines:
            llvm_ir = "\n".join(ir_lines[: FLAGS.max_ir_lines])
            ir_lines_used = FLAGS.max_ir_lines
        else:
            ir_lines_used = ir_lines_total

        t_encode0 = time.perf_counter()
        encoded = self.enc_tok(
            [llvm_ir],
            padding=True,
            truncation=True,
            max_length=FLAGS.max_input_length,
            return_tensors="pt",
        )
        enc_inputs = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "autophase": torch.tensor(autophase, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device),
        }
        encode_s = time.perf_counter() - t_encode0

        method = FLAGS.decode_method
        t_gen0 = time.perf_counter()
        if method == "greedy":
            # Greedy decoding is deterministic; returns a single sample.
            seq = self.model.generate(
                input_ids=enc_inputs["input_ids"],
                attention_mask=enc_inputs["attention_mask"],
                autophase=enc_inputs["autophase"],
                max_length=FLAGS.max_gen_length,
                do_sample=False,
                pad_token_id=self.dec_tok.pad_token_id,
                eos_token_id=self.dec_tok.eos_token_id,
            )
            if FLAGS.num_samples > 1:
                seq = seq.repeat_interleave(FLAGS.num_samples, dim=0)
            return seq, {
                "encode_s": encode_s,
                "generate_s": time.perf_counter() - t_gen0,
                "ir_lines_total": ir_lines_total,
                "ir_lines_used": ir_lines_used,
                "ir_truncated": ir_lines_used < ir_lines_total,
            }

        if method == "beam":
            sample_count = max(1, FLAGS.num_samples)
            beam_size = max(FLAGS.num_beams, sample_count)
            seq = self.model.generate(
                input_ids=enc_inputs["input_ids"],
                attention_mask=enc_inputs["attention_mask"],
                autophase=enc_inputs["autophase"],
                max_length=FLAGS.max_gen_length,
                do_sample=False,
                num_beams=beam_size,
                num_return_sequences=sample_count,
                pad_token_id=self.dec_tok.pad_token_id,
                eos_token_id=self.dec_tok.eos_token_id,
            )
            return seq, {
                "encode_s": encode_s,
                "generate_s": time.perf_counter() - t_gen0,
                "ir_lines_total": ir_lines_total,
                "ir_lines_used": ir_lines_used,
                "ir_truncated": ir_lines_used < ir_lines_total,
            }

        sample_count = max(1, FLAGS.num_samples)
        ids = enc_inputs["input_ids"].repeat_interleave(sample_count, dim=0)
        mask = enc_inputs["attention_mask"].repeat_interleave(sample_count, dim=0)
        auto = enc_inputs["autophase"].repeat_interleave(sample_count, dim=0)
        kwargs = dict(
            input_ids=ids,
            attention_mask=mask,
            autophase=auto,
            max_length=FLAGS.max_gen_length,
            do_sample=True,
            temperature=float(FLAGS.temperature),
            pad_token_id=self.dec_tok.pad_token_id,
            eos_token_id=self.dec_tok.eos_token_id,
        )
        if method == "sampling_topp":
            kwargs["top_p"] = float(FLAGS.top_p)
        seq = self.model.generate(**kwargs)
        return seq, {
            "encode_s": encode_s,
            "generate_s": time.perf_counter() - t_gen0,
            "ir_lines_total": ir_lines_total,
            "ir_lines_used": ir_lines_used,
            "ir_truncated": ir_lines_used < ir_lines_total,
        }

    @torch.no_grad()
    def __call__(self, env: LlvmEnv) -> None:
        if not self._warmed_up and FLAGS.warmup_generate_rounds > 0:
            warm_llvm_ir = env.observation["Ir"]
            warm_autophase = np.array(env.observation["Autophase"], dtype=np.float32)
            logger.info(
                "[%s] Running warmup generate rounds: %d",
                env.benchmark,
                int(FLAGS.warmup_generate_rounds),
            )
            for _ in range(int(FLAGS.warmup_generate_rounds)):
                self._generate_candidates(warm_llvm_ir, warm_autophase)
            self._warmed_up = True

        t_benchmark0 = time.perf_counter()
        t_obs0 = time.perf_counter()
        llvm_ir = env.observation["Ir"]
        autophase = np.array(env.observation["Autophase"], dtype=np.float32)
        observe_s = time.perf_counter() - t_obs0
        flag_to_action = {f: i for i, f in enumerate(env.action_space.flags)}
        best_reward = -float("inf")
        best_actions: List[int] = []
        sample_rows: List[Dict[str, object]] = []
        generation_timing: Dict[str, object] = {}
        soft_limit_s = float(FLAGS.soft_eval_time_limit_s)
        round_idx = 0
        total_samples_evaluated = 0
        global_sample_index = 0
        total_encode_s = 0.0
        total_generate_s = 0.0
        total_score_all_samples_s = 0.0
        worker_count = 1
        benchmark_spec = _serialize_benchmark_for_worker(env.benchmark)
        while True:
            sequences, generation_timing = self._generate_candidates(llvm_ir, autophase)
            if sequences.dim() == 1:
                sequences = sequences.unsqueeze(0)
            if round_idx == 0 and generation_timing.get("ir_truncated"):
                logger.info(
                    "[%s] IR truncated before encode: %d -> %d lines",
                    env.benchmark,
                    int(generation_timing["ir_lines_total"]),
                    int(generation_timing["ir_lines_used"]),
                )

            # Model output includes BOS at position 0.
            generated_tokens = sequences[:, 1:]
            worker_count = max(1, min(self.max_eval_workers, int(generated_tokens.shape[0])))
            t_score_round0 = time.perf_counter()
            if self._pool is None or worker_count <= 1:
                for idx in range(generated_tokens.shape[0]):
                    token_ids = generated_tokens[idx].tolist()
                    reward, actions, metrics = _score_sequence_in_process(
                        benchmark_spec,
                        token_ids,
                        self.token_to_flag,
                        flag_to_action,
                        self.special_ids,
                    )
                    if reward > best_reward:
                        best_reward = reward
                        best_actions = actions
                    pass_sequence = self.dec_tok.decode(
                        sequences[idx], skip_special_tokens=True
                    )
                    sample_rows.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "benchmark": str(env.benchmark),
                            "decode_method": FLAGS.decode_method,
                            "round_index": int(round_idx),
                            "sample_index": int(global_sample_index),
                            "sample_index_in_round": int(idx),
                            "reward": float(reward),
                            "num_actions": len(actions),
                            "pass_sequence": pass_sequence,
                            "actions": actions,
                            "metrics": metrics,
                        }
                    )
                    global_sample_index += 1
            else:
                futures = {}
                for idx in range(generated_tokens.shape[0]):
                    token_ids = generated_tokens[idx].tolist()
                    future = self._pool.submit(
                        _score_sequence_in_process,
                        benchmark_spec,
                        token_ids,
                        self.token_to_flag,
                        flag_to_action,
                        self.special_ids,
                    )
                    futures[future] = idx
                for future in as_completed(futures):
                    idx = futures[future]
                    reward, actions, metrics = future.result()
                    if reward > best_reward:
                        best_reward = reward
                        best_actions = actions
                    pass_sequence = self.dec_tok.decode(
                        sequences[idx], skip_special_tokens=True
                    )
                    sample_rows.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "benchmark": str(env.benchmark),
                            "decode_method": FLAGS.decode_method,
                            "round_index": int(round_idx),
                            "sample_index": int(global_sample_index),
                            "sample_index_in_round": int(idx),
                            "reward": float(reward),
                            "num_actions": len(actions),
                            "pass_sequence": pass_sequence,
                            "actions": actions,
                            "metrics": metrics,
                        }
                    )
                    global_sample_index += 1

            round_samples = int(generated_tokens.shape[0])
            total_samples_evaluated += round_samples
            total_encode_s += float(generation_timing["encode_s"])
            total_generate_s += float(generation_timing["generate_s"])
            total_score_all_samples_s += time.perf_counter() - t_score_round0
            round_idx += 1

            if soft_limit_s <= 0:
                break
            if (time.perf_counter() - t_benchmark0) >= soft_limit_s:
                break

        logger.info(
            "[%s] decode=%s rounds=%d total_samples=%d best_reward=%.6f best_len=%d",
            env.benchmark,
            FLAGS.decode_method,
            round_idx,
            total_samples_evaluated,
            best_reward,
            len(best_actions),
        )
        if sample_rows:
            max_reward = max(row["reward"] for row in sample_rows)
            for row in sample_rows:
                row["is_best"] = bool(row["reward"] >= max_reward)
            self._append_sample_metrics(sample_rows)

        t_replay0 = time.perf_counter()
        for action in best_actions:
            _, _, done, _ = env.step(action)
            if done:
                break
        replay_s = time.perf_counter() - t_replay0
        total_s = time.perf_counter() - t_benchmark0

        phase_times = {
            "observe_s": observe_s,
            "encode_s": total_encode_s,
            "generate_s": total_generate_s,
            "score_all_samples_s": total_score_all_samples_s,
            "replay_s": replay_s,
        }
        bottleneck_phase, bottleneck_s = max(
            phase_times.items(), key=lambda kv: kv[1]
        )
        sample_eval_times = [
            float(row.get("metrics", {}).get("sample_eval_wall_s", 0.0))
            for row in sample_rows
        ]
        self._append_benchmark_timing(
            {
                "timestamp": datetime.now().isoformat(),
                "benchmark": str(env.benchmark),
                "decode_method": FLAGS.decode_method,
                "num_samples": total_samples_evaluated,
                "num_samples_per_round": int(max(1, FLAGS.num_samples)),
                "rounds_executed": int(round_idx),
                "num_eval_workers": worker_count,
                "soft_eval_time_limit_s": soft_limit_s,
                "max_ir_lines": int(FLAGS.max_ir_lines),
                "ir_lines_total": int(generation_timing["ir_lines_total"]),
                "ir_lines_used": int(generation_timing["ir_lines_used"]),
                "ir_truncated": bool(generation_timing["ir_truncated"]),
                "phase_times_s": phase_times,
                "sample_eval_wall_s_mean": (
                    float(np.mean(sample_eval_times)) if sample_eval_times else 0.0
                ),
                "sample_eval_wall_s_max": (
                    float(np.max(sample_eval_times)) if sample_eval_times else 0.0
                ),
                "best_reward": float(best_reward),
                "best_num_actions": len(best_actions),
                "total_s": total_s,
                "bottleneck_phase": bottleneck_phase,
                "bottleneck_s": float(bottleneck_s),
            }
        )


def build_policy() -> PassformerPolicy:
    if not FLAGS.model_path:
        raise app.UsageError("--model_path is required.")
    if FLAGS.num_samples <= 0:
        raise app.UsageError("--num_samples must be > 0.")
    if FLAGS.top_p <= 0 or FLAGS.top_p > 1:
        raise app.UsageError("--top_p must be in (0, 1].")
    if FLAGS.max_ir_lines < 0:
        raise app.UsageError("--max_ir_lines must be >= 0.")
    if FLAGS.soft_eval_time_limit_s < 0:
        raise app.UsageError("--soft_eval_time_limit_s must be >= 0.")
    if FLAGS.warmup_generate_rounds < 0:
        raise app.UsageError("--warmup_generate_rounds must be >= 0.")
    if FLAGS.seed and FLAGS.seed > 0:
        torch.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)

    model_path = FLAGS.model_path
    enc_path = FLAGS.encoder_tokenizer_path or os.path.join(model_path, "encoder_tokenizer")
    dec_path = FLAGS.decoder_tokenizer_path or os.path.join(model_path, "decoder_tokenizer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model from %s", model_path)
    enc_tok = Inst2VecTokenizer.from_pretrained(enc_path)
    dec_tok = OptiSeqTokenizer.from_pretrained(dec_path)
    model = PassformerModel.from_pretrained(model_path).to(device).eval()
    policy = PassformerPolicy(model=model, enc_tok=enc_tok, dec_tok=dec_tok, device=device)
    # New run starts with a fresh per-sample metrics file.
    with open(policy.sample_metrics_path, "w", encoding="utf-8"):
        pass
    with open(policy.benchmark_timing_path, "w", encoding="utf-8"):
        pass
    logger.info("Per-sample metrics will be written to %s", policy.sample_metrics_path)
    logger.info("Per-benchmark timing will be written to %s", policy.benchmark_timing_path)
    return policy


def main(argv):
    del argv
    logging.basicConfig(level=logging.INFO)
    policy = build_policy()
    try:
        eval_llvm_instcount_policy(policy)
    finally:
        policy.close()


if __name__ == "__main__":
    app.run(main)
