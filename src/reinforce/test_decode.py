"""
对比多种解码方式对本地 .bc 基准的优化效果（reward、指令数）。

在固定模型与输入的前提下，可依次跑 greedy / beam / 多温度采样 / nucleus (top_p) 等配置，
并输出可横向对比的 CSV/JSON 汇总、以及 `run_meta.json`（命令行、主要 FLAGS、
依赖版本、git 提交、会话耗时与可选 GPU 峰值显存），便于与
`doc/Passformer_解码方式实验设计.md` 中的实验设计对齐。

依赖与 `test_cbench` 相同：需要可用的 llvm_wrapper 与 CompilerGym。

用法示例:
    python -m src.reinforce.test_decode \\
        --model_path /path/to/checkpoint \\
        --bc path/to/a.bc,path/to/b.bc \\
        --modes greedy,beam,sampling,sampling_topp

    # 可指定随机种子（用于采样/多候选可复现）
    python -m src.reinforce.test_decode --model_path ... --bc a.bc --seed 42

    # 整目录下所有 .bc（默认递归子目录；仅一层可加 --nobc_recursive）
    python -m src.reinforce.test_decode --model_path ... --bc_dir /path/to/bc_folder
    python -m src.reinforce.test_decode --model_path ... --bc /path/to/some.bc,/path/to/folder
"""

import csv
import importlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

from src.model import Inst2VecTokenizer, OptiSeqTokenizer, PassformerModel
from src.reinforce.test_cbench import (
    BenchmarkTestResult,
    apply_passes_and_get_ic,
    build_single_sample,
    compute_geomean,
    compute_summaries_by_sampling_key,
    compute_test_summary,
    evaluate_benchmark,
    evaluate_sequences,
    get_ic_counts,
    save_rollouts_jsonl,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", None, "Passformer 模型目录。")
flags.DEFINE_string(
    "encoder_tokenizer_path",
    None,
    "Inst2VecTokenizer 目录，默认 <model_path>/encoder_tokenizer。",
)
flags.DEFINE_string(
    "decoder_tokenizer_path",
    None,
    "OptiSeqTokenizer 目录，默认 <model_path>/decoder_tokenizer。",
)
flags.DEFINE_list(
    "bc",
    [],
    "逗号分隔的 .bc 文件路径，或其中可包含**目录**（会展开为该目录下所有 .bc，见 bc_recursive）。"
    "可与 --bc_dir 联用。",
)
flags.DEFINE_string(
    "bc_dir",
    None,
    "若设置，会收集该目录下所有 .bc 并与 --bc 合并（去重后评测）。与 --bc 二选一或同时使用均可，但至少一处有输入。",
)
flags.DEFINE_bool(
    "bc_recursive",
    True,
    "对 --bc_dir 及 --bc 中的目录，是否递归子目录搜索 *.bc。",
)
flags.DEFINE_list(
    "modes",
    ["greedy", "beam", "sampling"],
    "要运行的解码方式，逗号分隔: greedy, beam, sampling, sampling_topp。",
)
flags.DEFINE_integer("max_input_length", 512, "IR 编码最大长度。")
flags.DEFINE_integer("max_gen_length", 32, "解码生成最大长度 (max_length)。")
flags.DEFINE_integer("num_beams", 4, "beam 搜索宽度。")
flags.DEFINE_integer(
    "num_rollouts",
    16,
    "sampling：每个温度各生成的候选条数；sampling_topp：多候选总条数。",
)
flags.DEFINE_list(
    "temperatures",
    ["0.3", "0.7"],
    "sampling：温度列表；每个温度各跑 num_rollouts 条、分别统计/落表。",
)
flags.DEFINE_float("sampling_temp", 0.7, "sampling_topp 单温度。")
flags.DEFINE_float("top_p", 0.95, "sampling_topp: nucleus 采样 (需 <1 才启用) 。")
flags.DEFINE_integer("seed", 0, ">0 时设置 torch / numpy 随机种子。")
flags.DEFINE_string(
    "output_dir",
    None,
    "结果输出目录；默认为 ./work_dirs/decode_eval_<时间戳>。",
)

logger = logging.getLogger("test_decode")

_VALID_MODES = ("greedy", "beam", "sampling", "sampling_topp")


def _glob_bc_in_dir(d: Path, recursive: bool) -> List[Path]:
    d = d.expanduser().resolve()
    if not d.is_dir():
        raise app.UsageError(f"不是有效目录: {d}")
    if recursive:
        return sorted(d.rglob("*.bc"))
    return sorted(d.glob("*.bc"))


def _collect_bc_paths(
    bc_entries: List[str],
    bc_dir: Optional[str],
    recursive: bool,
) -> List[str]:
    """从 --bc 中的文件/目录，以及 --bc_dir，收集 .bc 绝对路径，去重，排序。"""
    seen: Set[str] = set()
    out: List[str] = []
    for raw in bc_entries:
        s = raw.strip()
        if not s:
            continue
        p = Path(s).expanduser()
        if not p.exists():
            raise app.UsageError(f"路径不存在: {s}")
        p = p.resolve()
        if p.is_dir():
            for f in _glob_bc_in_dir(p, recursive):
                key = str(f)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        elif p.is_file():
            if p.suffix.lower() != ".bc":
                logger.warning("跳过非 .bc 文件: %s", p)
                continue
            key = str(p)
            if key not in seen:
                seen.add(key)
                out.append(key)
        else:
            raise app.UsageError(f"不是文件或目录: {s}")
    if bc_dir:
        for f in _glob_bc_in_dir(Path(bc_dir), recursive):
            key = str(f)
            if key not in seen:
                seen.add(key)
                out.append(key)
    out.sort()
    return out


def _compute_best_temp_summary(
    results_mode: List[BenchmarkTestResult],
) -> Tuple[Dict, Dict[str, int]]:
    """sampling 模式跨温度聚合：每个 benchmark 在所有温度的 reward_max 里取 max,
    再做平均/几何均 —— 用于和 greedy/beam 这种"每基准 1 行"的口径横向对比。

    返回:
        summary: 仿 TestSummary 的精简 dict
        win_count: 每个 sampling_key 被选中作为该 bench 最佳温度的次数 (含并列均计)
    """
    by_bench: Dict[str, List[BenchmarkTestResult]] = {}
    for r in results_mode:
        if not r.success:
            continue
        by_bench.setdefault(r.benchmark, []).append(r)

    rewards: List[float] = []
    ic_reds: List[float] = []
    win_count: Dict[str, int] = {}
    for rs in by_bench.values():
        best_reward = max(r.reward_max for r in rs)
        rewards.append(best_reward)
        ic_reds.append(max(r.ic_reduction for r in rs))
        for r in rs:
            if r.reward_max >= best_reward and r.sampling_key:
                win_count[r.sampling_key] = win_count.get(r.sampling_key, 0) + 1

    ic_reds_pos = [v for v in ic_reds if v > 0]
    summary: Dict = {
        "total_benchmarks": len(by_bench),
        "successful": len(by_bench),
        "failed": 0,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_geomean": (
            compute_geomean([max(v, 1e-10) for v in rewards]) if rewards else 0.0
        ),
        "reward_max": float(np.max(rewards)) if rewards else 0.0,
        "ic_reduction_mean": float(np.mean(ic_reds)) if ic_reds else 0.0,
        "ic_reduction_geomean": (
            compute_geomean(ic_reds_pos) if ic_reds_pos else 0.0
        ),
        "strategy": "sampling@best_temp",
    }
    return summary, dict(sorted(win_count.items()))


# ----------------------------------------------------------------------
# Nucleus (top_p) 多候选 — test_cbench 中未提供，故在此实现
# ----------------------------------------------------------------------


@torch.no_grad()
def _generate_sampling_top_p(
    model: PassformerModel,
    enc_inputs: dict,
    dec_tok: OptiSeqTokenizer,
    max_gen_length: int,
    num_rollouts: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> torch.Tensor:
    n = int(num_rollouts)
    ids = enc_inputs["input_ids"].to(device).repeat_interleave(n, dim=0)
    mask = enc_inputs["attention_mask"].to(device).repeat_interleave(n, dim=0)
    ap = enc_inputs["autophase"].to(device).repeat_interleave(n, dim=0)
    return model.generate(
        input_ids=ids,
        attention_mask=mask,
        autophase=ap,
        max_length=max_gen_length,
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        pad_token_id=dec_tok.pad_token_id,
        eos_token_id=dec_tok.eos_token_id,
    )


def _evaluate_sampling_topp(
    model: PassformerModel,
    enc_tok: Inst2VecTokenizer,
    dec_tok: OptiSeqTokenizer,
    bc_path: str,
    cfg_test: dict,
    cfg_data: dict,
    device: torch.device,
) -> BenchmarkTestResult:
    """与 test_cbench.evaluate_benchmark 中 sampling 分支同结构，换用 top_p 生成。"""
    bm_name = os.path.basename(bc_path).replace(".bc", "")
    result = BenchmarkTestResult(
        benchmark=bm_name, bc_path=bc_path, strategy="sampling_topp"
    )
    t_wall0 = time.perf_counter()
    try:
        t0 = time.perf_counter()
        batch = build_single_sample(bc_path)
        result.load_obs_time = time.perf_counter() - t0
    except Exception as e:
        logger.warning("[%s] build sample failed: %s", bm_name, e)
        result.success = False
        result.error_message = str(e)
        result.wall_time_total = time.perf_counter() - t_wall0
        return result

    llvm_irs = batch["llvm_ir"]
    autophases = batch["autophase"].to(device)
    t0 = time.perf_counter()
    inputs = enc_tok(
        llvm_irs,
        padding=True,
        truncation=True,
        max_length=cfg_data.get("max_length", 512),
        return_tensors="pt",
    )
    enc_inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "autophase": autophases,
    }
    result.encode_time = time.perf_counter() - t0
    max_gen = int(cfg_test.get("max_gen_length", 32))
    n_roll = int(cfg_test.get("num_rollouts", 16))
    temp = float(cfg_test.get("sampling_temp", 0.7))
    top_p = float(cfg_test.get("top_p", 0.95))

    t0 = time.perf_counter()
    model.eval()
    sequences = _generate_sampling_top_p(
        model, enc_inputs, dec_tok, max_gen, n_roll, temp, top_p, device
    )
    result.inference_time = time.perf_counter() - t0
    result.num_rollouts = n_roll

    t1 = time.perf_counter()
    rewards_arr, _, best_pass_str, rollout_details = evaluate_sequences(
        dec_tok, sequences, bc_path
    )
    result.eval_time = time.perf_counter() - t1
    result.rollout_details = rollout_details
    for d in result.rollout_details:
        d["temperature"] = float(temp)

    result.reward_total = float(rewards_arr.sum())
    result.reward_mean = float(rewards_arr.mean())
    result.reward_max = float(rewards_arr.max())
    result.best_pass_sequence = best_pass_str
    result.num_passes = (
        len(best_pass_str.split()) if best_pass_str and best_pass_str.strip() else 0
    )
    result.temperature = float(temp)
    result.sampling_key = "sampling_topp"

    t_ic0 = time.perf_counter()
    try:
        ic = get_ic_counts(bc_path)
        result.original_ic = ic["original"]
        result.o3_ic = ic["o3"]
        result.oz_ic = ic["oz"]
        if best_pass_str.strip():
            result.optimized_ic = apply_passes_and_get_ic(bc_path, best_pass_str)
        else:
            result.optimized_ic = result.original_ic
        if result.original_ic > 0:
            result.ic_reduction = 1.0 - result.optimized_ic / result.original_ic
        if ic["o3"] > 0:
            result.ic_reduction_vs_o3 = 1.0 - result.optimized_ic / ic["o3"]
    except Exception as e:
        logger.warning("[%s] IC metrics failed: %s", bm_name, e)
    finally:
        result.ic_count_time = time.perf_counter() - t_ic0
        result.wall_time_total = time.perf_counter() - t_wall0
    return result


def _build_cfg() -> Tuple[dict, dict]:
    cfg_data = {"max_length": FLAGS.max_input_length}
    cfg_test: Dict = {
        "max_gen_length": FLAGS.max_gen_length,
        "num_beams": FLAGS.num_beams,
        "num_rollouts": FLAGS.num_rollouts,
        "temperatures": [float(t) for t in FLAGS.temperatures],
        "sampling_temp": float(FLAGS.sampling_temp),
        "top_p": float(FLAGS.top_p),
    }
    return cfg_data, cfg_test


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _package_version(name: str):
    try:
        return getattr(importlib.import_module(name), "__version__", None)
    except Exception:
        return None


def _git_commit(root: Path) -> Optional[str]:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if p.returncode == 0 and p.stdout:
            return p.stdout.strip()[:40]
    except Exception:
        return None
    return None


def _flags_to_serializable() -> dict:
    keys = [
        "model_path",
        "encoder_tokenizer_path",
        "decoder_tokenizer_path",
        "bc",
        "bc_dir",
        "bc_recursive",
        "modes",
        "max_input_length",
        "max_gen_length",
        "num_beams",
        "num_rollouts",
        "temperatures",
        "sampling_temp",
        "top_p",
        "seed",
        "output_dir",
    ]
    out = {}
    for k in keys:
        try:
            v = getattr(FLAGS, k)
        except (AttributeError, KeyError):
            v = None
        if isinstance(v, (list, tuple)):
            out[k] = [str(x) for x in v]
        else:
            out[k] = v
    return out


def _write_run_meta(
    out_dir: str,
    t_session0: float,
    summaries: List[dict],
    n_benchmarks: int,
) -> None:
    root = _project_root()
    meta: Dict = {
        "finished_at": datetime.now().isoformat(),
        "argv": list(sys.argv),
        "output_dir": os.path.abspath(out_dir),
        "project_root": str(root),
        "git_commit": _git_commit(root),
        "flags": _flags_to_serializable(),
        "session_wall_s": time.perf_counter() - t_session0,
        "n_bc_files": n_benchmarks,
        "per_mode_decode_summary": summaries,
        "python_version": sys.version,
        "versions": {
            "torch": _package_version("torch"),
            "numpy": _package_version("numpy"),
            "transformers": _package_version("transformers"),
            "compiler_gym": _package_version("compiler_gym"),
        },
    }
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            meta["torch_cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            meta["torch_cuda_device"] = None
        try:
            meta["torch_cuda_max_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated()
            )
        except Exception:
            pass
    path = os.path.join(out_dir, "run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", path)


def _serialize_row(
    r: BenchmarkTestResult, fieldnames: List[str]
) -> Dict:
    d = asdict(r)
    row: Dict = {}
    for k in fieldnames:
        v = d.get(k, "")
        if k == "best_pass_sequence" and v is not None and "\n" in str(v):
            v = str(v).replace("\n", " ")[:2000]
        row[k] = v
    return row


def _save_combined(
    all_results: List[BenchmarkTestResult], summaries: List[dict], out_dir: str
) -> None:
    path = os.path.join(out_dir, "decode_all_results.csv")
    fieldnames = [
        "benchmark",
        "bc_path",
        "strategy",
        "original_ic",
        "optimized_ic",
        "o3_ic",
        "oz_ic",
        "reward_total",
        "reward_max",
        "reward_mean",
        "ic_reduction",
        "ic_reduction_vs_o3",
        "best_pass_sequence",
        "num_passes",
        "num_rollouts",
        "temperature",
        "sampling_key",
        "load_obs_time",
        "encode_time",
        "inference_time",
        "eval_time",
        "ic_count_time",
        "wall_time_total",
        "success",
        "error_message",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in all_results:
            w.writerow(_serialize_row(r, fieldnames))
    meta = os.path.join(out_dir, "decode_summary_by_mode.json")
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": FLAGS.model_path,
                "modes": [s["mode"] for s in summaries],
                "per_mode": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    roll_path = os.path.join(out_dir, "decode_all_rollouts.jsonl")
    save_rollouts_jsonl(all_results, roll_path)
    logger.info("Wrote %s, %s, %s", path, meta, roll_path)


def main(argv):
    del argv
    if FLAGS.model_path is None:
        raise app.UsageError("--model_path 为必填。")
    if not FLAGS.bc and not FLAGS.bc_dir:
        raise app.UsageError("请通过 --bc 和/或 --bc_dir 指定 .bc 文件，或含 .bc 的目录。")

    modes = [m.strip().lower() for m in FLAGS.modes if m.strip()]
    for m in modes:
        if m not in _VALID_MODES:
            raise app.UsageError(
                f"未知 mode: {m}，允许: {', '.join(_VALID_MODES)}"
            )

    if FLAGS.seed and FLAGS.seed > 0:
        torch.manual_seed(int(FLAGS.seed))
        np.random.seed(int(FLAGS.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(FLAGS.seed))

    enc_path = FLAGS.encoder_tokenizer_path or os.path.join(
        FLAGS.model_path, "encoder_tokenizer"
    )
    dec_path = FLAGS.decoder_tokenizer_path or os.path.join(
        FLAGS.model_path, "decoder_tokenizer"
    )
    out_dir = FLAGS.output_dir
    if not out_dir:
        out_dir = os.path.join(
            os.getcwd(), "work_dirs", f"decode_eval_{datetime.now():%Y%m%d_%H%M%S}"
        )
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(out_dir, "test_decode.log"), encoding="utf-8"
            ),
            logging.StreamHandler(),
        ],
    )

    if "sampling_topp" in modes and FLAGS.top_p >= 1.0:
        logger.warning(
            "sampling_topp 需要 --top_p < 1，当前为 %s，已忽略 sampling_topp。",
            FLAGS.top_p,
        )
        modes = [m for m in modes if m != "sampling_topp"]

    if not modes:
        raise app.UsageError("无可用解码模式；使用 sampling_topp 时请将 --top_p 设为小于 1。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Output dir: %s", out_dir)
    logger.info("Modes: %s", modes)

    enc_tok = Inst2VecTokenizer.from_pretrained(enc_path)
    dec_tok = OptiSeqTokenizer.from_pretrained(dec_path)
    model = PassformerModel.from_pretrained(FLAGS.model_path).to(device).eval()

    cfg_data, cfg_test = _build_cfg()
    try:
        bc_list = _collect_bc_paths(
            [str(x) for x in FLAGS.bc if x and str(x).strip()],
            FLAGS.bc_dir,
            bool(FLAGS.bc_recursive),
        )
    except app.UsageError:
        raise
    except (OSError, ValueError) as e:
        raise app.UsageError(str(e)) from e
    if not bc_list:
        raise app.UsageError(
            "未找到任何 .bc 文件。请检查目录非空、扩展名为 .bc，或尝试开启递归 (默认开启)。"
        )
    logger.info("共 %d 个 .bc 基准 (来自 --bc / --bc_dir)", len(bc_list))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t_session0 = time.perf_counter()

    all_results: List[BenchmarkTestResult] = []
    per_mode_summary: List[dict] = []

    for mode in modes:
        results_mode: List[BenchmarkTestResult] = []
        for bc_path in tqdm(bc_list, desc=mode):
            if mode == "sampling_topp":
                r = _evaluate_sampling_topp(
                    model, enc_tok, dec_tok, bc_path, cfg_test, cfg_data, device
                )
                results_mode.append(r)
                all_results.append(r)
            else:
                for r in evaluate_benchmark(
                    model,
                    enc_tok,
                    dec_tok,
                    bc_path,
                    mode,
                    cfg_test,
                    cfg_data,
                    device,
                    logger,
                ):
                    results_mode.append(r)
                    all_results.append(r)

        summ = compute_test_summary(results_mode, mode)
        out_entry: Dict = {
            "mode": mode,
            "summary": asdict(summ),
            "reward_geomean": summ.reward_geomean,
            "ic_reduction_geomean": summ.ic_reduction_geomean,
        }
        if mode == "sampling":
            by_k = compute_summaries_by_sampling_key(results_mode)
            out_entry["summary_by_sampling_key"] = {
                k: asdict(s) for k, s in by_k.items()
            }
            best_temp_summary, win_count = _compute_best_temp_summary(results_mode)
            out_entry["summary_best_temp"] = best_temp_summary
            out_entry["best_temp_win_count"] = win_count
        per_mode_summary.append(out_entry)
        logger.info(
            "[%s] reward mean=%.4f geomean=%.4f | IC red geomean=%.2f%%",
            mode,
            summ.reward_mean,
            summ.reward_geomean,
            summ.ic_reduction_geomean * 100,
        )
        if mode == "sampling" and "summary_by_sampling_key" in out_entry:
            for k, s in out_entry["summary_by_sampling_key"].items():
                sm = s
                logger.info(
                    "  [按温度] %s:  n=%s  reward_mean=%.4f  geomean=%.4f  ic_red_geomean=%.2f%%",
                    k,
                    sm["total_benchmarks"],
                    sm["reward_mean"],
                    sm["reward_geomean"],
                    sm["ic_reduction_geomean"] * 100,
                )
            bt = out_entry.get("summary_best_temp") or {}
            if bt:
                logger.info(
                    "  [按基准取最优温度] n=%s  reward_mean=%.4f  geomean=%.4f  ic_red_geomean=%.2f%%",
                    bt["total_benchmarks"],
                    bt["reward_mean"],
                    bt["reward_geomean"],
                    bt["ic_reduction_geomean"] * 100,
                )
                wc = out_entry.get("best_temp_win_count") or {}
                if wc:
                    logger.info(
                        "  [最优温度分布] %s",
                        ", ".join(f"{k}:{v}" for k, v in wc.items()),
                    )

    _save_combined(all_results, per_mode_summary, out_dir)
    _write_run_meta(out_dir, t_session0, per_mode_summary, len(bc_list))

    if len(per_mode_summary) > 1:
        logger.info("--- 横向对比 (reward mean / geomean, IC reduction geomean) ---")
        logger.info(
            "  %-22s  %6s  %12s  %12s  %12s",
            "mode",
            "n",
            "reward_mean",
            "reward_gmean",
            "ic_red_gmean",
        )
        for s in per_mode_summary:
            sm = s["summary"]
            logger.info(
                "  %-22s  %6d  %12.4f  %12.4f  %11.2f%%",
                s["mode"],
                sm["total_benchmarks"],
                sm["reward_mean"],
                sm["reward_geomean"],
                sm["ic_reduction_geomean"] * 100,
            )
            bt = s.get("summary_best_temp") or {}
            if bt:
                logger.info(
                    "    %-20s  %6d  %12.4f  %12.4f  %11.2f%%",
                    "└ @best_temp",
                    bt["total_benchmarks"],
                    bt["reward_mean"],
                    bt["reward_geomean"],
                    bt["ic_reduction_geomean"] * 100,
                )
            by_key = s.get("summary_by_sampling_key") or {}
            for k in sorted(by_key.keys()):
                sk = by_key[k]
                logger.info(
                    "    %-20s  %6d  %12.4f  %12.4f  %11.2f%%",
                    f"└ {k}",
                    sk["total_benchmarks"],
                    sk["reward_mean"],
                    sk["reward_geomean"],
                    sk["ic_reduction_geomean"] * 100,
                )


if __name__ == "__main__":
    app.run(main)
