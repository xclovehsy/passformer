# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""本地实现的 LLVM instcount 评估入口，扩展自 compiler_gym.leaderboard.llvm_instcount。

在保留原有 --leaderboard_results、--n、--test_dataset 等行为的前提下，支持：

- 主结果 CSV 在标准四列外增加 IR 相关列：相对 O0 的策略结果
  ``(original - optimized) / original``、``-O3``/``-Oz`` 基线指令数
  及相对 O0 的 O3/Oz 缩减比例、策略相对 O3/Oz 的缩减比例。
- 从 ``--bc`` / ``--bc_dir`` 构建评测集：为每个 ``.bc`` 调用
  :func:`compiler_gym.datasets.Benchmark.from_file` 嵌入 bitcode，再 ``reset`` 传入
  :class:`Benchmark`（不依赖在 ``env.datasets`` 中注册 ``file-v0``，与仅传 URI 字符串不同）。
- 在评测结束后额外写出 ``run_meta.json`` 与 ``eval_detail.json``（每轮奖励、墙钟时间、
  动作数、以及可选的 IR 指令数观测，便于与 ``src/reinforce/test_decode.py`` 的实验记录对齐）。

若未提供 ``--bc`` 且未设置 ``--bc_dir``，则与上游 ``eval_llvm_instcount_policy`` 一致，
仍使用 ``--test_dataset`` 中的基准列表。

``compiler_gym.bin.validate`` 时会对结果 CSV 取前四列写入临时文件，避免与扩展列不兼容。
"""
import csv
import json
import logging
import os
import tempfile
import subprocess
import sys
import time
from datetime import datetime
# from importlib import importlib
from itertools import islice
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

# 注册与上游一致的 leaderboard 相关 FLAGS（leaderboard_results、n、test_dataset 等）
import compiler_gym.envs  # noqa: F401
import compiler_gym.leaderboard.llvm_instcount  # noqa: F401
import gym
import humanize
from absl import app, flags
from compiler_gym.bin.validate import main as validate
from compiler_gym.compiler_env_state import CompilerEnvState
from compiler_gym.datasets import Benchmark
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import FLAGS
from compiler_gym.util.statistics import arithmetic_mean, geometric_mean
from compiler_gym.util.timer import Timer, humanize_duration_hms

flags.DEFINE_list(
    "bc",
    [],
    "逗号分隔的 .bc 文件路径，或包含 .bc 的目录（见 --bc_recursive）。"
    "若与 --bc_dir 任一有值，则启用自定义 .bc 评测集，不再使用 --test_dataset。",
)
flags.DEFINE_string(
    "bc_dir",
    None,
    "若设置，会收集该目录下所有 .bc 并与 --bc 合并。与自定义评测集联用时覆盖 --test_dataset。",
)
flags.DEFINE_bool(
    "bc_recursive",
    True,
    "对 --bc_dir 及 --bc 中的目录，是否递归子目录搜索 *.bc。",
)
flags.DEFINE_string(
    "eval_output_dir",
    None,
    "若设置，将 run_meta.json 与 eval_detail.json 写入此目录；"
    "否则写在 leaderboard 结果文件所在目录。",
)
flags.DEFINE_bool(
    "eval_record_ir_ic",
    True,
    "在 eval_detail 中尝试记录 IrInstructionCount / O0 / Oz（需 CompilerGym 服务支持）。",
)

Policy = Callable[[LlvmEnv], None]


# def _package_version(name: str):
#     try:
#         return getattr(importlib.import_module(name), "__version__", None)
#     except Exception:
#         return None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


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
        pass
    return None


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


def _bc_path_to_benchmark(p: str) -> Benchmark:
    """从 .bc 路径构建带程序内容的 Benchmark（与仅传 URI 不同，不查 datasets 表）。"""
    path = Path(p).expanduser().resolve()
    if not path.is_file():
        raise app.UsageError(f"不是有效的 .bc 文件: {p}")
    # 与 compiler_gym.util.flags.benchmark_from_flags 中 user-v0 形式一致
    uri = f"benchmark://user-v0{path}"
    return Benchmark.from_file(uri=uri, path=path)


def _benchmark_uri_str(benchmark: Union[str, Benchmark]) -> str:
    if isinstance(benchmark, Benchmark):
        return str(benchmark.uri)
    return benchmark


def _bc_path_from_uri_str(benchmark: str) -> str:
    try:
        u = BenchmarkUri.from_string(benchmark)
    except Exception:
        return ""
    if u.dataset in ("file-v0", "user-v0") and u.path:
        return u.path
    return ""


def _use_custom_bc() -> bool:
    return bool(FLAGS.bc_dir) or bool([x for x in FLAGS.bc if str(x).strip()])


def _build_benchmarks_list(
    env: LlvmEnv,
) -> Tuple[List[Union[str, Benchmark]], str]:
    """返回 (每条评测项：URI 字符串或 Benchmark 对象, 描述标签)。"""
    if _use_custom_bc():
        try:
            paths = _collect_bc_paths(
                [str(x) for x in FLAGS.bc if x and str(x).strip()],
                FLAGS.bc_dir,
                bool(FLAGS.bc_recursive),
            )
        except app.UsageError:
            raise
        except (OSError, ValueError) as e:
            raise app.UsageError(str(e)) from e
        if not paths:
            raise app.UsageError(
                "已启用 --bc / --bc_dir 但未找到任何 .bc 文件。"
            )
        return [_bc_path_to_benchmark(p) for p in paths], "custom-bc"

    benchmarks = env.datasets[FLAGS.test_dataset].benchmark_uris()
    if FLAGS.max_benchmarks:
        benchmarks = islice(benchmarks, FLAGS.max_benchmarks)
    uris = list(benchmarks)
    return uris, FLAGS.test_dataset


def _llvm_extra_observations(env: LlvmEnv) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    if not FLAGS.eval_record_ir_ic:
        return out
    for key in (
        "IrInstructionCount",
        "IrInstructionCountO0",
        "IrInstructionCountO3",
        "IrInstructionCountOz",
    ):
        try:
            v = env.observation[key]
            if hasattr(v, "item"):
                v = v.item()
            out[key] = float(v) if v is not None else None
        except Exception:
            out[key] = None
    return out


def _llvm_ic_counts_for_csv(env: LlvmEnv) -> Dict[str, Optional[float]]:
    """为 CSV 记录：当前 IR、O0、-O3 / -Oz 基线（不随 eval_record_ir_ic 关闭）。"""
    out: Dict[str, Optional[float]] = {}
    for key in (
        "IrInstructionCount",
        "IrInstructionCountO0",
        "IrInstructionCountO3",
        "IrInstructionCountOz",
    ):
        try:
            v = env.observation[key]
            if hasattr(v, "item"):
                v = v.item()
            out[key] = float(v) if v is not None else None
        except Exception:
            out[key] = None
    return out


def _float_cell(x: Optional[float]) -> str:
    return "" if x is None else f"{x:g}"


def _ratio_num(a: Optional[float], b: Optional[float]) -> str:
    """(a - b) / a；需 a>0 且 b 有效。表示相对 a 的「缩减」比例。"""
    if a is not None and a > 0 and b is not None:
        return f"{(a - b) / a:.6f}"
    return ""


def _ic_reduction_tuple(
    original: Optional[float], optimized: Optional[float]
) -> Tuple[str, str, str]:
    """返回 (ic_reduction, original_str, optimized_str)；ic_reduction = (O0-当前)/O0。"""
    o_str = _float_cell(original)
    z_str = _float_cell(optimized)
    r = _ratio_num(original, optimized)
    return r, o_str, z_str


def _read_states_for_resume(path: str) -> List[CompilerEnvState]:
    """从 leaderboard CSV 恢复已完成评测（只读前四列，兼容扩展列与旧四列表）。"""
    p = Path(path)
    if not p.is_file():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    start = 0
    if rows[0] and str(rows[0][0]).strip().lower() == "benchmark":
        start = 1
    out: List[CompilerEnvState] = []
    for row in rows[start:]:
        if len(row) < 4:
            continue
        bench, rw, wt, cmd = row[0], row[1], row[2], row[3]
        out.append(
            CompilerEnvState(
                benchmark=bench,
                reward=None if rw == "" else float(rw),
                walltime=0 if wt == "" else float(wt),
                commandline=cmd,
            )
        )
    return out


def _write_temp_leaderboard_for_validate(path: str) -> str:
    """生成仅含 benchmark,reward,walltime,commandline 的临时 CSV 供 validate 使用。"""
    p = Path(path)
    if not p.is_file():
        return str(p)
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return str(p)
    start = 0
    if rows[0] and str(rows[0][0]).strip().lower() == "benchmark":
        start = 1
    out_rows = [("benchmark", "reward", "walltime", "commandline")]
    for row in rows[start:]:
        if len(row) < 4:
            continue
        out_rows.append(row[:4])
    fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="validate_", text=True)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerows(out_rows)
    return tmp


def _write_run_meta(
    out_dir: str,
    t_session0: float,
    n_benchmarks: int,
    dataset_label: str,
    final_geomean_reward: Optional[float] = None,
    final_mean_reward: Optional[float] = None,
    completed_count: Optional[int] = None,
) -> None:
    root = _project_root()
    meta: Dict = {
        "finished_at": datetime.now().isoformat(),
        "argv": list(sys.argv),
        "output_dir": os.path.abspath(out_dir),
        "project_root": str(root),
        "git_commit": _git_commit(root),
        "flags": {
            "leaderboard_results": FLAGS.leaderboard_results,
            "leaderboard_logfile": FLAGS.leaderboard_logfile,
            "max_benchmarks": FLAGS.max_benchmarks,
            "n": FLAGS.n,
            "test_dataset": FLAGS.test_dataset,
            "bc": [str(x) for x in FLAGS.bc],
            "bc_dir": FLAGS.bc_dir,
            "bc_recursive": FLAGS.bc_recursive,
            "validate": FLAGS.validate,
            "resume": FLAGS.resume,
            "eval_output_dir": FLAGS.eval_output_dir,
            "eval_record_ir_ic": FLAGS.eval_record_ir_ic,
        },
        "session_wall_s": time.perf_counter() - t_session0,
        "n_eval_runs": n_benchmarks,
        "dataset_label": dataset_label,
        "completed_count": completed_count,
        "final_geomean_reward": final_geomean_reward,
        "final_mean_reward": final_mean_reward,
    }
    path = os.path.join(out_dir, "run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Wrote {path}")


def _write_eval_detail(out_dir: str, rows: List[dict]) -> None:
    path = os.path.join(out_dir, "eval_detail.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote {path}")


class _EvalPolicyWorker(Thread):
    """Worker thread to evaluate a policy, with per-episode extra metrics."""

    def __init__(
        self,
        env: LlvmEnv,
        benchmarks: List[Union[str, Benchmark]],
        policy: Policy,
        init_states: List[CompilerEnvState],
    ):
        super().__init__()
        self.env = env
        self.benchmarks = benchmarks
        self.policy = policy
        self.states: List[CompilerEnvState] = init_states
        self.detail_rows: List[dict] = []
        self.alive = True

    def run(self):
        res_path = FLAGS.leaderboard_results
        is_new = (not Path(res_path).is_file()) or os.stat(res_path).st_size == 0
        with open(res_path, "a", newline="", encoding="utf-8") as out_f:
            w = csv.writer(out_f, lineterminator="\n")
            if is_new:
                w.writerow(
                    (
                        "benchmark",
                        "reward",
                        "walltime",
                        "commandline",
                        "ic_reduction",
                        "original_ir_count",
                        "optimized_ir_count",
                        "o3_ir_count",
                        "oz_ir_count",
                        "ic_reduction_o3_vs_o0",
                        "ic_reduction_oz_vs_o0",
                        "ic_reduction_policy_vs_o3",
                        "ic_reduction_policy_vs_oz",
                    )
                )
            for benchmark in self.benchmarks:
                self.env.reset(benchmark=benchmark)
                with Timer() as timer:
                    self.policy(self.env)

                assert self.env.in_episode, "Environment is no longer in an episode"
                assert self.env.benchmark and (
                    _benchmark_uri_str(self.env.benchmark)
                    == _benchmark_uri_str(benchmark)
                ), "Policy changed environment benchmark"
                assert self.env.reward_space, "Policy unset environment reward space"
                assert self.env.reward_space.name == "IrInstructionCountOz", (
                    "Policy changed environment reward space"
                )

                state = self.env.state.copy()
                state.walltime = timer.time
                num_actions = len(self.env.actions)
                ic = _llvm_ic_counts_for_csv(self.env)
                extra_obs = _llvm_extra_observations(self.env)
                uri_s = _benchmark_uri_str(benchmark)
                o0 = ic.get("IrInstructionCountO0")
                ir = ic.get("IrInstructionCount")
                o3 = ic.get("IrInstructionCountO3")
                oz = ic.get("IrInstructionCountOz")
                ic_r, o0s, z_s = _ic_reduction_tuple(o0, ir)
                o3_vs_o0 = _ratio_num(o0, o3)
                oz_vs_o0 = _ratio_num(o0, oz)
                pol_vs_o3 = _ratio_num(o3, ir)
                pol_vs_oz = _ratio_num(oz, ir)
                row = {
                    "benchmark": uri_s,
                    "bc_path": _bc_path_from_uri_str(uri_s) or None,
                    "reward": state.reward,
                    "walltime": state.walltime,
                    "num_actions": num_actions,
                    "commandline": state.commandline,
                }
                row.update({f"obs_{k}": v for k, v in extra_obs.items()})
                self.detail_rows.append(row)

                w.writerow(
                    (
                        state.benchmark,
                        state.reward,
                        state.walltime,
                        state.commandline,
                        ic_r,
                        o0s,
                        z_s,
                        _float_cell(o3),
                        _float_cell(oz),
                        o3_vs_o0,
                        oz_vs_o0,
                        pol_vs_o3,
                        pol_vs_oz,
                    )
                )
                out_f.flush()
                self.states.append(state)

                if not self.alive:
                    return


def eval_llvm_instcount_policy(policy: Policy) -> None:
    def main(argv):
        assert len(argv) == 1, f"Unknown args: {argv[1:]}"
        assert FLAGS.n > 0, "n must be > 0"

        t_session0 = time.perf_counter()
        out_dir: Optional[str] = FLAGS.eval_output_dir
        if not out_dir:
            out_dir = os.path.dirname(os.path.abspath(FLAGS.leaderboard_results)) or "."
        os.makedirs(out_dir, exist_ok=True)

        worker = None
        total_count = 0
        dataset_label = ""
        with gym.make("llvm-ic-v0") as env:
            logger = logging.getLogger("compiler_gym")
            logger.setLevel(logging.DEBUG)
            log_handler = logging.FileHandler(FLAGS.leaderboard_logfile)
            logger.addHandler(log_handler)
            logger.propagate = False

            print(f"Writing results to {FLAGS.leaderboard_results}")
            print(f"Writing logs to {FLAGS.leaderboard_logfile}")

            base_benchmarks, dataset_label = _build_benchmarks_list(env)
            n_unique = len(base_benchmarks)
            benchmarks = base_benchmarks * FLAGS.n
            total_count = len(benchmarks)

            init_states = []
            if FLAGS.resume and Path(FLAGS.leaderboard_results).is_file():
                for state in _read_states_for_resume(FLAGS.leaderboard_results):
                    init_states.append(state)
                    for i, b in enumerate(benchmarks):
                        if _benchmark_uri_str(b) == state.benchmark:
                            del benchmarks[i]
                            break

            worker = _EvalPolicyWorker(env, benchmarks, policy, init_states)
            worker.start()
            timer = Timer().reset()
            if _use_custom_bc():
                print(
                    f"=== 自定义 .bc 评测集 ({dataset_label}): "
                    f"{n_unique} 个唯一样本 × {FLAGS.n} 轮 ==="
                )
            else:
                print(
                    f"=== Evaluating policy on {humanize.intcomma(total_count)} "
                    f"{FLAGS.test_dataset} benchmarks ===\n\n"
                )
            try:
                while worker.is_alive():
                    done_count = len(worker.states)
                    remaining_count = total_count - done_count
                    elapsed = timer.time
                    gmean_reward = geometric_mean([s.reward for s in worker.states])
                    mean_walltime = (
                        arithmetic_mean([s.walltime for s in worker.states]) or elapsed
                    )
                    print(
                        "\r\033[2A"
                        "\033[K"
                        f"Runtime: {humanize_duration_hms(elapsed)}. "
                        f"Estimated completion: {humanize_duration_hms(mean_walltime * remaining_count)}. "
                        f"Completed: {humanize.intcomma(done_count)} / {humanize.intcomma(total_count)} "
                        f"({done_count / max(total_count,1):.1%})."
                        "\n\033[K"
                        f"Current mean walltime: {mean_walltime:.3f}s / benchmark."
                        "\n\033[K"
                        f"Current geomean reward: {gmean_reward:.4f}.",
                        flush=True,
                        end="",
                    )
                    sleep(1)
            except KeyboardInterrupt:
                print("\nkeyboard interrupt", flush=True)
                worker.alive = False
                FLAGS.validate = False
            finally:
                if worker is not None:
                    worker.join(timeout=3600)

        if worker is not None:
            detail: List[dict] = list(worker.detail_rows)
            if out_dir and detail:
                _write_eval_detail(out_dir, detail)
            final_rewards = [s.reward for s in worker.states]
            final_geomean_reward = (
                geometric_mean(final_rewards) if final_rewards else None
            )
            final_mean_reward = (
                arithmetic_mean(final_rewards) if final_rewards else None
            )
            if out_dir:
                _write_run_meta(
                    out_dir,
                    t_session0,
                    total_count,
                    dataset_label,
                    final_geomean_reward=final_geomean_reward,
                    final_mean_reward=final_mean_reward,
                    completed_count=len(worker.states),
                )

        if FLAGS.validate and _use_custom_bc():
            print(
                "Skip validate for custom --bc/--bc_dir benchmarks "
                "(benchmark://user-v0 is not registered in env.datasets)."
            )
        elif FLAGS.validate:
            FLAGS.env = "llvm-ic-v0"
            tmp_v = _write_temp_leaderboard_for_validate(FLAGS.leaderboard_results)
            try:
                validate(["argv0", tmp_v])
            finally:
                if tmp_v != FLAGS.leaderboard_results:
                    try:
                        os.remove(tmp_v)
                    except OSError:
                        pass

    app.run(main)
