# PPO 强化学习训练工作汇报

## 工作概述

这边在 `src/reinforce/train_sequential.py` 里实现了基于 PPO 的强化学习训练流程，主要在 cbench-v1 上做 Passformer 的训练和验证。整体思路是：用监督学习预训练好的模型，通过 CompilerGym 环境里的真实优化反馈来做策略微调。

## 实现情况

**训练方式**：按 benchmark 顺序逐个训练，每个 benchmark 训到收敛或提前停止（reward 达标、patience 用尽）再切到下一个。样本通过 `build_single_sample()` 从 bc 文件加载 IR 和 Autophase 特征。

**PPO 训练器**：参考 CompilerDream 等工作的思路，实现了带 per-step reward 的 PPO。相比之前试的 GRPO，主要改进有：

1. **Per-step Reward**：每个 pass 执行后通过 `env.step()` 拿到即时 reward，能把奖励更细地归因到每个动作上，模型能学到每一步选哪个 pass 更合适。
2. **Clipped Surrogate**：限制策略更新幅度，避免大更新导致发散。
3. **Multi-epoch PPO**：每轮 rollout 后多轮 PPO 更新，提高样本利用率。
4. **Discount + Mean-rollout Baseline**：用 γ=0.99 的折扣回报，同一输入的 K 条 rollout 取均值做 baseline，降低方差。
5. **Multi-temperature Sampling**：温度 [0.3, 0.5, 0.7, 1.0] 混合采样，平衡探索和利用。

环境用的是 CompilerGym 的 `llvm-autophase-ic-v0`，reward 基于 IR 指令数（code size）的变化。

## 训练结果

在 cbench-v1 上跑了几轮，表现最好的一次（20260304）geomean reward 到 1.032，23 个 benchmark 都有有效结果。像 bzip2、qsort、sha、rijndael 等都能到 1.1 左右，tiff 系列也在 1.03 上下。结果会写到 `work_dirs/reinforce_sequential/<timestamp>/benchmark_results.csv`。

也有几次跑得不太好，比如 20260307 那轮有多个 benchmark reward 直接为 0，geomean 掉到 8e-5 量级，可能是超时或 early stop 导致。训练稳定性这块还在调。

## 和现有工作的对比

进展文档里对比过 CompilerDream、PPO+Guided Search、Greedy Search、GATv2 等，从 code size reduction 来看我们监督学习阶段的 Passformer 和这些 RL/搜索方法指标定义可能不太一样，需要统一评估协议才能公平对比。

**当前策略的一个明显局限**：我们只在第一次 `env.reset()` 时提取一次 IR 和 Autophase 特征，然后模型基于这个初始特征一次性生成完整优化序列，中间不再根据执行后的 IR 状态重新观测。而 CompilerDream、GATv2 这类工作是边执行边观察、逐步决策的，能利用每个 pass 执行后的新 IR/Autophase 信息。所以按同样评估方式，我们这种 one-shot 策略预期会差一些，这也是后续可以改进的方向。

## 后续计划

目前主要在调超参数（学习率、temperature、ppo_clip_eps、num_rollouts 等），同时准备在大规模数据集（如 anghabench、github 等）上做训练。配置在 `configs/reinforce_sequential.yaml`，跑的话用：

```bash
python -m src.reinforce.train_sequential --config configs/reinforce_sequential.yaml
```




# PPO 强化学习训练工作汇报

## 工作概述
老师，我来汇报下最近的工作进展

本周主要精力集中在基于PPO的强化学习训练流程，并在cbench-v1上做了训练和验证。对监督学习预训练好的模型，通过CompilerGym环境里的真实优化反馈来做策略微调。
目前小规模验证训练代码是按benchmark顺序逐个训练，训到收敛或提前停止再切到下一个。从bc文件加载IR和Autophase特征。

参考CompilerDream等工作的思路，实现了PPO算法。相比之前实现的GRPO，主要改进有：
1. 每个pass执行后通过编译环境拿到即时reward，能把奖励更细地归因到每个动作上，模型能学到每一步选哪个pass更合适。2. 限制策略更新幅度，避免大更新导致发散。3. 用γ=0.99的折扣回报，同一输入的K条rollout取均值做baseline，降低方差。4.k条rollout采用不同温度混合采样生成，可以提升探索能力。
训练结果是在cbench-v1上跑了几轮，表现最好的一次geomean reward到1.044。和现有工作的对比（CompilerDream、PPO+Guided Search、Greedy Search、GATv2）等还有微小差距。也有几次跑得不太好，训练稳定性这块还在调。

当前策略的一个弱势是只提取一次IR和Autophase特征，然后模型基于这个初始特征一次性生成完整优化序列，中间不再根据执行后的IR状态重新观测。而 CompilerDream、GATv2 这类工作是边执行边观察、逐步决策的，能利用每个pass执行后的新特征信息。所以我们这种one-shot策略预期会差一些。但优势是不用多次和编译环境交互，相比传统方法在推理速度上有优势。

后续计划准备调下超参数（学习率、temperature、ppo_clip_eps、num_rollouts 等），同时准备在大规模数据集上做训练。

