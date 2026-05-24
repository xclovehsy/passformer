# Passformer 解码方式对编译优化效果的实验设计

本文档说明如何系统研究**解码策略（及超参数）**在固定模型下对 **LLVM 指令数 / CompilerGym 环境回报** 等指标的影响。实现上可对齐仓库中的 `PassformerModel.generate`（HuggingFace `GenerationMixin`）、`src/reinforce/test.py` 中的多温度 rollout + fork 选优，以及 `src/inference/passformer_inference.py` 的封装。

---

## 1. 研究问题（建议先写清 1～2 条）

在**固定预训练/微调权重与相同 checkpoint** 的前提下，可检验例如：

- **Q1**：贪婪解码、束搜索、随机采样（及温度 / top-p 等）对 **官方 Leaderboard 指标（如 `eval_llvm_instcount`）** 的相对差异有多大？
- **Q2**：性能差异主要来自**「尝试的候选序列数量/算力」**，还是来自**「解码算法本身」**？（需用预算对齐或成本归一化。）

问题写清楚后，再选实验因子与表头，避免实验变成无结构的调参表。

---

## 2. 变量与指标

| 类型 | 内容 |
|------|------|
| **自变量** | 解码模式：`greedy`；`beam`（`num_beams`）；`sample`（`do_sample=True`，`temperature` / `top_p` / `top_k`）；以及 **多温度 + 多 rollout + 在 fork 环境中选累计 reward 最大** 的策略（与 `test.py` 一致时）。 |
| **必须控制的变量** | 同一 `model_path`、同一评测 **benchmark 集合**、同一 **encoder** 截断（如 `max_input_length`）、同一 **生成长度上界**（`max_length` 或 `max_new_tokens`，各组必须一致且明确是否含 BOS 等）。融合若使用 autophase，各组**一致提供**；随机解码需可复现时**固定随机种子**。 |
| **因变量（主结果）** | 与 `eval_llvm_instcount_policy` 等官方流程一致的主指标；若另记 fork 内累计 reward，需说明与最终榜单指标的关系。 |
| **辅助与成本** | 每 benchmark **生成+评测总耗时**、**候选条数**、**GPU 显存**峰值；用于「效果—成本」讨论。 |
| **稳健性** | 对**随机采样**类设置，建议**多随机种子**（如 3～5）报 **均值 ± 标准差** 或区间。 |

---

## 3. 实验组（由简到繁）

### 3.1 阶段 A：主对比（先跑通、少而精）

1. **Greedy**：`do_sample=False`，`num_beams=1` — 基线，计算最省。
2. **Beam search**：在固定 `max_gen_length` 下扫描 `num_beams ∈ {1, 4, 8, 16}`（或硬件允许的上限）。
3. **单温度随机采样**：`do_sample=True`，扫描 `temperature`（如 0.2, 0.4, 0.6, 0.8, 1.0 或对数间隔的若干点）；可选加 `top_p < 1.0` 的 1～2 档。
4. **多温度 + 多 rollout + fork 选优**（与 `test.py` 策略对齐）：固定 `num_rollouts`、多 `temperatures`，在 fork 环境中评估每条候选的累计 reward，将最优序列写回**原始**环境 — 作为「强基线 / 近部署」形态。

### 3.2 阶段 B：公平性（成本与预算）

- 若某方法会生成/评估**多条**候选（多 rollout、多 beam），需二选一或并列说明：
  - **总评估预算相同**：例如均「最多评估 16 条序列再取最优」，比较 **i.i.d. 采样**、**多温度切分**、**beam-16** 在**相同预算**下的表现；或
  - **总计算量/总生成 token 可比的归一化**：报告每条 benchmark 的 **前向次数** 或 **生成 token 总量**，避免用「多采样堆出来的提升」假装成「解码器结构优势」。

### 3.3 阶段 C：消融（可选，有篇幅再做）

- 固定解码策略，只改变 **`max_gen_length`（或 `max_new_tokens`）**，区分是**序列长度**还是**解码类型**在主导。
- 固定从 **k 个候选**中选 1 个，改变**选择规则**（如 env 累计 reward 最大 vs. 仅模型似然）— 若关注「与 CompilerGym 目标是否一致」。

---

## 4. 实现与记录方式（与仓库对齐）

- **统一入口**：尽量通过同一套「对 `LlvmEnv` 的 policy」调用 `model.generate`，只切换解码参数，避免**代码路径**不一致（例如 `test.py` 与 `test_cbench.py` 行为漂移）。
- **落盘内容**：为每种（解码配置 × 随机种子）输出独立结果表（如 CSV/JSON），字段至少含：`benchmark`、**主指标**、**是否早停/结束**、**时间戳**；采样实验保存 **seed**。
- **单因子原则**：改解码时尽量**不要同时**改 `max_gen_length` 或换 checkpoint，避免混淆因素。

参考文件：

- `src/reinforce/test.py`：多温度采样、fork 内评估、`eval_llvm_instcount_policy`。
- `src/inference/passformer_inference.py`：`num_beams`、`do_sample`、`temperature`、`top_p` 等向 `model.generate` 的传递方式。

---

## 5. 结果呈现建议

- **总表**：各方法在验证集/榜单子集上的 **平均主指标**、相对 greedy 的 **相对提升 (%)**。
- **方差**：采样类给出跨 seed 的 **均值 ± 标准差** 或分位数；beam/greedy 可单次或有限次重复以估计波动。
- **分桶**（若数据支持）：按 IR 规模、Autophase 范数、或难例/易例分桶，观察「哪类程序更受益于探索（采样/beam）」。

---

## 6. 常见风险与规避

- **训评不一致**：若训练时默认解码与实验解码差异大，结论需圈定为「**仅推理阶段**的解码选择」，并说明与训练目标的关系。
- **长度与特殊 token 计数**：`max_length` 是否包含 decoder 起始部分，**各组必须一致**；`pad_token_id` / `eos_token_id` 与 tokenizer 对齐。
- **非确定性**：GPU 上即使用同一 seed 也可能有微小差异；关键对比在采样设置上**多次重复**更稳妥。

---

## 7. 最小可行方案（MVP）清单

1. 固定 1 个 checkpoint、1 个 benchmark 子集（可先用小规模再扩全量）。
2. 跑通：**Greedy**、**Beam(4, 8)**、**Sample(2～3 个温度)**、**多 rollout + 多温度 + fork-best**（与 `test.py` 对齐）四块。
3. 对采样与多随机候选，使用 **3 个 seed** 报均值，附录可附标准差。
4. 在附录或同表记清：**每条 benchmark 的候选数/束宽** 与 **Wall-clock**，便于复现与对比。

（文档版本：与 2026-04-26 讨论稿一致，可按实际代码与 `transformers` 版本增补具体 `generate` 参数名。）
