# Passformer 强化学习训练技术方案

## 一、方案概述

将 Passformer 模型从**监督学习（Supervised Learning）**转换为**强化学习（Reinforcement Learning）**训练，参考 CompilerDream 的 RL 训练方式，在 CompilerGym 环境中通过奖励信号学习最优的编译器优化 Pass 序列。

### 核心思路

1. **环境交互**：在 CompilerGym 中执行生成的 Pass 序列，获得真实优化效果
2. **奖励信号**：基于 IR 指令数减少比例（reduction）设计奖励函数
3. **策略梯度**：使用 PPO（Proximal Policy Optimization）或类似算法更新模型
4. **经验回放**：存储交互经验，用于稳定训练

---

## 二、架构设计

### 2.1 整体架构

```
┌─────────────────┐
│  CompilerGym    │ ← 环境：执行 Pass 序列，返回奖励
│    Environment  │
└────────┬────────┘
         │
         │ (state, reward, done)
         ▼
┌─────────────────┐
│  Passformer     │ ← 策略网络：生成 Pass 序列
│   Policy Net    │
└────────┬────────┘
         │
         │ (action probabilities)
         ▼
┌─────────────────┐
│  Value Net      │ ← 价值网络：估计状态价值（可选）
│   (Critic)      │
└────────┬────────┘
         │
         │ (value estimate)
         ▼
┌─────────────────┐
│  RL Trainer     │ ← 训练器：PPO/REINFORCE
│   (PPO)         │
└─────────────────┘
```

### 2.2 关键组件

#### 1. **环境包装器（RL Environment Wrapper）**
- 封装 CompilerGym 环境
- 提供统一的 RL 接口（reset, step, observation_space, action_space）
- 处理状态表示（LLVM IR + Autophase）
- 计算奖励函数

#### 2. **策略网络（Policy Network）**
- **复用现有 Passformer 模型**作为策略网络
- 输入：LLVM IR（tokenized）+ Autophase
- 输出：Pass 序列的概率分布（通过 decoder 生成）

#### 3. **价值网络（Value Network，可选）**
- 估计状态价值 V(s)
- 可以复用 encoder，添加一个价值头（value head）
- 用于 PPO 等 actor-critic 算法

#### 4. **经验回放缓冲区（Replay Buffer）**
- 存储 (state, action, reward, next_state, done)
- 支持采样批次用于训练

#### 5. **RL 训练器**
- 实现 PPO 或 REINFORCE 算法
- 计算策略梯度
- 更新模型参数

---

## 三、详细设计

### 3.1 环境包装器

**文件：`src/rl/env_wrapper.py`**

```python
class PassformerRLEnv:
    """
    RL 环境包装器，封装 CompilerGym 环境
    """
    def __init__(
        self,
        env,  # CompilerGym environment
        encoder_tokenizer,
        decoder_tokenizer,
        reward_config={
            'baseline_thre': 1.0,  # 奖励阈值
            'leakiness': 1.0,      # 奖励泄漏系数
            'step_penalty': 0.0,   # 步数惩罚
        }
    ):
        self.env = env
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.reward_config = reward_config
        
    def reset(self, benchmark=None):
        """重置环境，返回初始状态"""
        obs = self.env.reset(benchmark=benchmark)
        state = self._extract_state(obs)
        return state
    
    def step(self, action_sequence):
        """
        执行动作序列（Pass 序列）
        
        Args:
            action_sequence: Pass 序列字符串，如 "-mem2reg -instcombine"
        
        Returns:
            next_state, reward, done, info
        """
        # 解析 Pass 序列
        passes = action_sequence.split()
        
        total_reward = 0
        initial_ir_count = self.env.observation['IrInstructionCount']
        
        for pass_name in passes:
            if pass_name.startswith('-'):
                pass_name = pass_name[1:]
            
            # 查找动作索引
            try:
                action_idx = self.env.action_space.flags.index(pass_name)
            except ValueError:
                # Pass 不存在，跳过
                continue
            
            # 执行动作
            obs, reward, done, info = self.env.step(action_idx)
            total_reward += reward
            
            if done:
                break
        
        # 计算最终奖励
        final_ir_count = self.env.observation['IrInstructionCount']
        reduction = (initial_ir_count - final_ir_count) / initial_ir_count
        
        # 应用奖励函数
        reward = self._compute_reward(
            initial_ir_count, 
            final_ir_count, 
            reduction,
            len(passes)
        )
        
        next_state = self._extract_state(obs)
        info['reduction'] = reduction
        info['initial_ir_count'] = initial_ir_count
        info['final_ir_count'] = final_ir_count
        
        return next_state, reward, done, info
    
    def _extract_state(self, obs):
        """提取状态（LLVM IR + Autophase）"""
        return {
            'llvm_ir': obs['Ir'],
            'autophase': obs['Autophase'],
            'ir_count': obs['IrInstructionCount'],
        }
    
    def _compute_reward(self, initial_count, final_count, reduction, num_steps):
        """计算奖励"""
        # 基础奖励：reduction
        reward = reduction
        
        # 应用阈值和泄漏
        if reward > self.reward_config['baseline_thre']:
            reward = self.reward_config['baseline_thre'] + \
                     (reward - self.reward_config['baseline_thre']) * \
                     self.reward_config['leakiness']
        
        # 步数惩罚
        reward -= self.reward_config['step_penalty'] * num_steps
        
        return reward
```

### 3.2 策略网络适配

**文件：`src/rl/policy_network.py`**

```python
class PassformerPolicy:
    """
    基于 Passformer 的策略网络
    """
    def __init__(self, model, encoder_tokenizer, decoder_tokenizer, device='cuda'):
        self.model = model.to(device)
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.device = device
    
    def get_action_log_probs(self, states, action_sequences):
        """
        计算动作序列的对数概率
        
        Args:
            states: List of state dicts {'llvm_ir': str, 'autophase': np.array}
            action_sequences: List of action sequences (strings)
        
        Returns:
            log_probs: Tensor of log probabilities
        """
        # Tokenize states
        llvm_irs = [s['llvm_ir'] for s in states]
        autophases = torch.tensor([s['autophase'] for s in states]).to(self.device)
        
        encoder_inputs = self.encoder_tokenizer(
            llvm_irs, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)
        
        # Tokenize actions
        decoder_inputs = self.decoder_tokenizer(
            action_sequences,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=encoder_inputs['input_ids'],
            attention_mask=encoder_inputs['attention_mask'],
            autophase=autophases,
            labels=decoder_inputs['input_ids']
        )
        
        # Compute log probabilities
        logits = outputs.logits
        labels = decoder_inputs['input_ids']
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        log_probs_selected = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding tokens
        mask = (shift_labels != self.decoder_tokenizer.pad_token_id).float()
        log_probs_selected = log_probs_selected * mask
        
        # Sum over sequence length
        total_log_probs = log_probs_selected.sum(dim=-1)
        
        return total_log_probs
    
    def sample_action(self, state, max_length=256, temperature=1.0):
        """
        采样动作序列
        
        Args:
            state: State dict {'llvm_ir': str, 'autophase': np.array}
            max_length: Maximum sequence length
            temperature: Sampling temperature
        
        Returns:
            action_sequence: String of Pass sequence
            log_prob: Log probability of the sequence
        """
        self.model.eval()
        
        llvm_ir = state['llvm_ir']
        autophase = torch.tensor(state['autophase']).unsqueeze(0).to(self.device)
        
        encoder_inputs = self.encoder_tokenizer(
            llvm_ir,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=encoder_inputs['input_ids'],
                attention_mask=encoder_inputs['attention_mask'],
                autophase=autophase,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.decoder_tokenizer.pad_token_id,
                eos_token_id=self.decoder_tokenizer.eos_token_id,
            )
        
        action_sequence = self.decoder_tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        # Compute log prob
        log_prob = self.get_action_log_probs([state], [action_sequence])[0]
        
        return action_sequence, log_prob
```

### 3.3 价值网络（可选）

**文件：`src/rl/value_network.py`**

```python
class PassformerValueHead(nn.Module):
    """
    价值网络头，添加到 Passformer encoder
    """
    def __init__(self, encoder_hidden_size, hidden_dim=256):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(encoder_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, encoder_hidden_states):
        """
        Args:
            encoder_hidden_states: [batch_size, seq_len, hidden_size]
        
        Returns:
            values: [batch_size, 1]
        """
        # Use [CLS] token or mean pooling
        pooled = encoder_hidden_states[:, 0, :]  # [batch_size, hidden_size]
        values = self.value_head(pooled)
        return values


class PassformerValueNetwork:
    """
    完整的价值网络（encoder + value head）
    """
    def __init__(self, model, encoder_tokenizer, value_head, device='cuda'):
        self.model = model.to(device)
        self.encoder_tokenizer = encoder_tokenizer
        self.value_head = value_head.to(device)
        self.device = device
    
    def get_value(self, states):
        """估计状态价值"""
        llvm_irs = [s['llvm_ir'] for s in states]
        autophases = torch.tensor([s['autophase'] for s in states]).to(self.device)
        
        encoder_inputs = self.encoder_tokenizer(
            llvm_irs,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                input_ids=encoder_inputs['input_ids'],
                attention_mask=encoder_inputs['attention_mask']
            )
        
        # Fuse autophase if needed
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Project autophase and fuse
        if hasattr(self.model, '_fuse_autophase'):
            encoder_hidden_states = self.model._fuse_autophase(
                encoder_hidden_states,
                autophases
            )
        
        values = self.value_head(encoder_hidden_states)
        return values
```

### 3.4 PPO 训练器

**文件：`src/rl/ppo_trainer.py`**

```python
class PPOTrainer:
    """
    PPO 训练器
    """
    def __init__(
        self,
        policy_network,
        value_network=None,  # 可选
        lr=3e-5,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cuda'
    ):
        self.policy = policy_network
        self.value_network = value_network
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.model.parameters(),
            lr=lr
        )
        
        if self.value_network:
            self.value_optimizer = torch.optim.Adam(
                self.value_network.parameters(),
                lr=lr
            )
        
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def update(self, batch):
        """
        更新策略网络
        
        Args:
            batch: {
                'states': List of states,
                'actions': List of action sequences,
                'rewards': List of rewards,
                'old_log_probs': Tensor of old log probs,
                'values': Tensor of value estimates (if using value network),
                'advantages': Tensor of advantages
            }
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        old_log_probs = batch['old_log_probs'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        
        # Get new log probs
        new_log_probs = self.policy.get_action_log_probs(states, actions)
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (if using value network)
        value_loss = 0
        if self.value_network:
            values = self.value_network.get_value(states)
            value_targets = torch.tensor(rewards).to(self.device).unsqueeze(-1)
            value_loss = F.mse_loss(values, value_targets)
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss
        
        # Backward
        self.optimizer.zero_grad()
        if self.value_network:
            self.value_optimizer.zero_grad()
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.model.parameters(),
            self.max_grad_norm
        )
        
        self.optimizer.step()
        if self.value_network:
            self.value_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item() if self.value_network else 0,
            'total_loss': total_loss.item()
        }
```

### 3.5 经验回放缓冲区

**文件：`src/rl/replay_buffer.py`**

```python
class ReplayBuffer:
    """
    经验回放缓冲区
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, info=None):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info or {}
        }
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """采样批次"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch
    
    def __len__(self):
        return len(self.buffer)
```

### 3.6 主训练脚本

**文件：`src/training/passformer_rl_train.py`**

```python
def main():
    # 1. 加载配置
    config = load_config(args.config)
    
    # 2. 创建环境
    env = compiler_gym.make("llvm-ic-v0")
    rl_env = PassformerRLEnv(
        env,
        encoder_tokenizer,
        decoder_tokenizer,
        reward_config=config['reward']
    )
    
    # 3. 加载模型
    model = PassformerModel.from_pretrained(config['model']['pretrained_path'])
    policy = PassformerPolicy(model, encoder_tokenizer, decoder_tokenizer)
    
    # 4. 创建训练器
    trainer = PPOTrainer(policy, lr=config['training']['lr'])
    
    # 5. 创建回放缓冲区
    replay_buffer = ReplayBuffer(capacity=config['training']['replay_capacity'])
    
    # 6. 训练循环
    for episode in range(config['training']['num_episodes']):
        # 收集经验
        state = rl_env.reset(benchmark=...)
        episode_rewards = []
        episode_states = []
        episode_actions = []
        
        for step in range(config['training']['max_steps']):
            # 采样动作
            action, log_prob = policy.sample_action(state)
            
            # 执行动作
            next_state, reward, done, info = rl_env.step(action)
            
            # 存储经验
            replay_buffer.push(state, action, reward, next_state, done, info)
            episode_rewards.append(reward)
            episode_states.append(state)
            episode_actions.append(action)
            
            state = next_state
            if done:
                break
        
        # 计算优势（使用 GAE）
        advantages = compute_gae(episode_rewards, ...)
        
        # 更新策略
        if len(replay_buffer) >= config['training']['min_buffer_size']:
            batch = replay_buffer.sample(config['training']['batch_size'])
            metrics = trainer.update(batch)
            logger.info(f"Episode {episode}, Metrics: {metrics}")
        
        # 保存检查点
        if episode % config['training']['save_interval'] == 0:
            model.save_pretrained(f"{work_dir}/checkpoint-{episode}")
```

---

## 四、实施步骤

### Phase 1: 基础环境搭建（1-2 天）

1. ✅ 创建 `src/rl/` 目录结构
2. ✅ 实现环境包装器 `env_wrapper.py`
3. ✅ 测试 CompilerGym 环境集成

### Phase 2: 策略网络适配（2-3 天）

1. ✅ 实现 `policy_network.py`
2. ✅ 适配 Passformer 模型为策略网络
3. ✅ 实现动作采样和概率计算

### Phase 3: RL 训练器实现（3-4 天）

1. ✅ 实现 PPO 训练器
2. ✅ 实现经验回放缓冲区
3. ✅ 实现优势估计（GAE）

### Phase 4: 训练脚本和配置（2-3 天）

1. ✅ 实现主训练脚本
2. ✅ 创建配置文件
3. ✅ 添加日志和监控

### Phase 5: 测试和优化（3-5 天）

1. ✅ 小规模测试
2. ✅ 超参数调优
3. ✅ 性能优化

---

## 五、配置文件示例

**文件：`configs/passformer_rl_train.yaml`**

```yaml
model:
  pretrained_path: /path/to/pretrained/passformer
  encoder_tokenizer_path: /path/to/encoder_tokenizer
  decoder_tokenizer_path: /path/to/decoder_tokenizer

env:
  env_name: llvm-ic-v0
  benchmark_dataset: cbench-v1  # 或自定义数据集
  max_steps: 45

reward:
  baseline_thre: 1.0
  leakiness: 1.0
  step_penalty: 0.0

training:
  num_episodes: 10000
  max_steps: 45
  batch_size: 32
  lr: 3e-5
  clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  gamma: 0.99  # 折扣因子
  gae_lambda: 0.95  # GAE lambda
  replay_capacity: 10000
  min_buffer_size: 1000
  update_frequency: 10  # 每 N 个 episode 更新一次
  save_interval: 100

output:
  work_dir: /path/to/output
  log_dir: /path/to/logs
```

---

## 六、关键技术点

### 6.1 奖励函数设计

```python
def compute_reward(initial_ir_count, final_ir_count, num_steps):
    """
    奖励函数设计
    
    选项1: 简单 reduction
    reward = (initial - final) / initial
    
    选项2: 对数缩放（避免奖励过大）
    reward = log(initial / final)
    
    选项3: 归一化到 [-1, 1]
    reward = 2 * (reduction) - 1
    
    选项4: 带步数惩罚
    reward = reduction - alpha * num_steps
    """
    reduction = (initial_ir_count - final_ir_count) / initial_ir_count
    reward = reduction - 0.01 * num_steps  # 步数惩罚
    return reward
```

### 6.2 优势估计（GAE）

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lambda_=0.95):
    """
    计算广义优势估计（Generalized Advantage Estimation）
    """
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages)
```

### 6.3 动作空间处理

- **离散动作空间**：每个 Pass 是一个动作，序列是动作序列
- **序列生成**：使用 decoder 自回归生成 Pass 序列
- **EOS 处理**：遇到 EOS token 时停止生成

---

## 七、与 CompilerDream 的对比

| 特性 | CompilerDream | Passformer RL |
|------|---------------|---------------|
| **模型架构** | DreamerV2 (World Model) | Passformer (Seq2Seq) |
| **训练方式** | World Model + Actor-Critic | 直接策略梯度（PPO） |
| **状态表示** | Autophase + InstCount | LLVM IR + Autophase |
| **动作空间** | Autophase (56维) | Pass 序列（序列生成） |
| **优势** | 可以想象未来状态 | 更直接，训练简单 |
| **劣势** | 需要训练世界模型 | 需要更多环境交互 |

---

## 八、预期效果

1. **性能提升**：通过 RL 训练，模型可以学习到更好的优化策略
2. **泛化能力**：在不同 benchmark 上表现更好
3. **探索能力**：可以发现监督学习学不到的优化序列

---

## 九、风险和挑战

1. **训练不稳定**：RL 训练可能不稳定，需要仔细调参
2. **环境交互成本**：每次训练都需要实际编译，成本较高
3. **奖励稀疏**：优化效果可能不明显，奖励信号稀疏
4. **序列长度**：Pass 序列长度不固定，需要处理变长序列

---

## 十、后续优化方向

1. **混合训练**：结合监督学习和强化学习
2. **课程学习**：从简单到复杂的 benchmark
3. **多任务学习**：同时优化多个目标（性能、代码大小等）
4. **分布式训练**：并行多个环境，加速训练

