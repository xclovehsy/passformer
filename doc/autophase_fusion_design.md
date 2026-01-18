# Autophase 静态特征融合技术方案

## 1. 背景

### 1.1 当前架构

当前模型采用 Encoder-Decoder 架构：

- **Encoder**: ModernBERT (InstBERT)
  - hidden_size: 768
  - num_hidden_layers: 22
  - 输入: LLVM IR 文本序列
  
- **Decoder**: GPT2
  - n_embd: 768
  - n_layer: 6
  - 输出: 优化 pass 序列

### 1.2 Autophase 特征

Autophase 是一个 56 维的静态特征向量，包含以下类型的信息：

| 类别 | 特征示例 |
|------|---------|
| 基本块信息 | TotalBlocks, BlockLow, BlockMid, onePred, twoSuccessor 等 |
| 指令计数 | TotalInsts, NumAddInst, NumLoadInst, NumCallInst 等 |
| PHI 节点 | NumPHIInst, BeginPhi, ArgsPhi, BBNoPhi, BB03Phi 等 |
| 控制流 | BranchCount, UncondBranches, CriticalCount, NumEdges |
| 常量信息 | const32Bit, const64Bit, numConstZeroes, numConstOnes |
| 函数/内存 | TotalFuncs, TotalMemInst |

### 1.3 问题

当前代码虽然在数据预处理时加载了 Autophase 特征，但**模型并未实际使用该特征**。我们希望让模型能够感知 IR 的静态特征，以提升优化序列生成的质量。

---

## 2. 技术方案

### 方案1: 特征拼接到 Encoder 输出（最简单）

#### 原理

将 Autophase 通过线性层映射到 encoder 的隐藏维度（768），然后作为额外的 token 拼接到 encoder 输出序列中。

#### 架构图

```
LLVM IR → [InstBERT Encoder] → [seq_len, 768]
                                      ↓
Autophase → [Linear: 56→768] → [1, 768] → concat → [seq_len+1, 768]
                                                          ↓
                                                   [GPT2 Decoder]
                                                          ↓
                                                   优化 Pass 序列
```

#### 代码实现

```python
import torch
import torch.nn as nn
from transformers import EncoderDecoderModel

class AutophaseProjection(nn.Module):
    """将 Autophase 特征投影到 encoder 隐藏维度"""
    
    def __init__(self, autophase_dim=56, hidden_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(autophase_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, autophase):
        # autophase: [batch, 56] -> [batch, 1, 768]
        return self.proj(autophase).unsqueeze(1)


class AutophaseFusedEncoderDecoder(nn.Module):
    """在 encoder 输出后拼接 Autophase 特征"""
    
    def __init__(self, encoder_decoder_model, autophase_dim=56):
        super().__init__()
        self.model = encoder_decoder_model
        hidden_dim = encoder_decoder_model.config.encoder.hidden_size
        self.autophase_proj = AutophaseProjection(autophase_dim, hidden_dim)
    
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        autophase,
        labels=None,
        **kwargs
    ):
        # 获取 encoder 输出
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        encoder_hidden = encoder_outputs.last_hidden_state  # [batch, seq_len, 768]
        
        # 投影 autophase 并拼接
        autophase_emb = self.autophase_proj(autophase)  # [batch, 1, 768]
        fused_hidden = torch.cat([autophase_emb, encoder_hidden], dim=1)
        
        # 更新 attention_mask
        batch_size = attention_mask.shape[0]
        autophase_mask = torch.ones(batch_size, 1, device=attention_mask.device)
        fused_attention_mask = torch.cat([autophase_mask, attention_mask], dim=1)
        
        # 通过 decoder
        outputs = self.model.decoder(
            encoder_hidden_states=fused_hidden,
            encoder_attention_mask=fused_attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 实现简单，改动最小 | 融合方式简单，特征交互有限 |
| 可快速验证 autophase 是否有用 | autophase 信息可能被 IR 特征淹没 |
| 不需要修改 encoder/decoder 内部结构 | 只在 encoder 输出处融合一次 |

---

### 方案2: 作为 Decoder 前缀（Prefix Tuning 风格）

#### 原理

将 Autophase 转换为若干个"虚拟 token"作为 decoder 输入的前缀，让 decoder 通过 attention 机制主动查询静态特征。

#### 架构图

```
                                    Autophase → [MLP] → [num_prefix, 768]
                                                              ↓
LLVM IR → [Encoder] → encoder_hidden → [Decoder] ← prefix_tokens
                                              ↓
                                       优化 Pass 序列
```

#### 代码实现

```python
class AutophasePrefix(nn.Module):
    """将 Autophase 转换为前缀 token 序列"""
    
    def __init__(self, autophase_dim=56, hidden_dim=768, num_prefix_tokens=4):
        super().__init__()
        self.num_prefix = num_prefix_tokens
        self.hidden_dim = hidden_dim
        
        # 两层 MLP 投影
        self.proj = nn.Sequential(
            nn.Linear(autophase_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * num_prefix_tokens),
            nn.LayerNorm(hidden_dim * num_prefix_tokens)
        )
    
    def forward(self, autophase):
        # autophase: [batch, 56]
        # output: [batch, num_prefix, hidden_dim]
        prefix = self.proj(autophase)
        return prefix.view(-1, self.num_prefix, self.hidden_dim)


class PrefixConditionedDecoder(nn.Module):
    """带有 Autophase 前缀的 Decoder"""
    
    def __init__(self, decoder, autophase_dim=56, num_prefix_tokens=4):
        super().__init__()
        self.decoder = decoder
        hidden_dim = decoder.config.n_embd
        self.prefix_generator = AutophasePrefix(
            autophase_dim, hidden_dim, num_prefix_tokens
        )
        self.num_prefix = num_prefix_tokens
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        autophase=None,
        labels=None,
        **kwargs
    ):
        batch_size = input_ids.shape[0]
        
        # 生成前缀 embeddings
        if autophase is not None:
            prefix_embeds = self.prefix_generator(autophase)  # [batch, num_prefix, hidden]
            
            # 获取 decoder 的 input embeddings
            inputs_embeds = self.decoder.transformer.wte(input_ids)
            
            # 拼接前缀
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
            
            # 更新 attention mask
            prefix_mask = torch.ones(batch_size, self.num_prefix, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
            # 调整 labels
            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, self.num_prefix), 
                    -100, 
                    device=labels.device
                )
                labels = torch.cat([prefix_labels, labels], dim=1)
            
            return self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels,
                **kwargs
            )
        else:
            return self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels,
                **kwargs
            )
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| Decoder 可主动查询静态特征 | 增加序列长度 |
| 符合 Prefix Tuning 的成功范式 | 需要调整 position embedding |
| 前缀 token 数量可调 | 实现复杂度中等 |

---

### 方案3: Cross-Attention 融合（推荐）

#### 原理

在 decoder 的每一层（或特定层）添加一个专门的 cross-attention 模块，用于关注 autophase 特征。这样 decoder 可以在每一层都利用静态特征信息。

#### 架构图

```
                              Autophase → [Projection] → autophase_emb
                                                              ↓
LLVM IR → [Encoder] → encoder_hidden                          ↓
                          ↓                                   ↓
              [Decoder Layer 1] ← Cross-Attn ← autophase_emb
                          ↓                                   ↓
              [Decoder Layer 2] ← Cross-Attn ← autophase_emb
                          ↓                                   ↓
                         ...                                 ...
                          ↓
                   优化 Pass 序列
```

#### 代码实现

```python
class AutophaseCrossAttention(nn.Module):
    """Autophase 特征的 Cross-Attention 模块"""
    
    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, autophase_emb):
        # hidden_states: [batch, seq_len, hidden]
        # autophase_emb: [batch, 1, hidden] 或 [batch, num_features, hidden]
        attn_output, _ = self.cross_attn(
            query=hidden_states,
            key=autophase_emb,
            value=autophase_emb
        )
        hidden_states = self.norm(hidden_states + self.dropout(attn_output))
        return hidden_states


class AutophaseConditionedGPT2(nn.Module):
    """带有 Autophase Cross-Attention 的 GPT2 Decoder"""
    
    def __init__(self, gpt2_model, autophase_dim=56, inject_layers=None):
        super().__init__()
        self.gpt2 = gpt2_model
        self.config = gpt2_model.config
        hidden_dim = self.config.n_embd
        num_layers = self.config.n_layer
        
        # Autophase 投影层
        self.autophase_proj = nn.Sequential(
            nn.Linear(autophase_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 决定在哪些层注入 autophase 信息
        # 默认在所有层注入
        if inject_layers is None:
            inject_layers = list(range(num_layers))
        self.inject_layers = set(inject_layers)
        
        # 为每个注入层创建 cross-attention
        self.autophase_cross_attns = nn.ModuleDict({
            str(i): AutophaseCrossAttention(hidden_dim, self.config.n_head)
            for i in inject_layers
        })
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        autophase=None,
        labels=None,
        **kwargs
    ):
        # 获取 input embeddings
        inputs_embeds = self.gpt2.transformer.wte(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        position_embeds = self.gpt2.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.gpt2.transformer.drop(hidden_states)
        
        # 投影 autophase
        autophase_emb = None
        if autophase is not None:
            autophase_emb = self.autophase_proj(autophase).unsqueeze(1)  # [batch, 1, hidden]
        
        # 逐层前向传播
        for i, block in enumerate(self.gpt2.transformer.h):
            # 原始 transformer block
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )[0]
            
            # 注入 autophase 信息
            if autophase_emb is not None and i in self.inject_layers:
                hidden_states = self.autophase_cross_attns[str(i)](
                    hidden_states, autophase_emb
                )
        
        hidden_states = self.gpt2.transformer.ln_f(hidden_states)
        lm_logits = self.gpt2.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {"loss": loss, "logits": lm_logits}
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 每一层都可利用静态特征 | 需要修改 decoder 结构 |
| 特征交互充分 | 增加计算量和参数 |
| 可灵活选择注入层 | 实现复杂度较高 |

---

### 方案4: FiLM 条件调制（Feature-wise Linear Modulation）

#### 原理

使用 Autophase 特征来动态调制 encoder 或 decoder 中各层的隐藏表示，通过 scale (γ) 和 shift (β) 操作实现条件控制。

#### 架构图

```
Autophase → [γ_net] → γ (scale)
         → [β_net] → β (shift)
                          ↓
hidden_states → LayerNorm → h' = γ * h + β → 下一层
```

#### 代码实现

```python
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation 层"""
    
    def __init__(self, autophase_dim=56, hidden_dim=768):
        super().__init__()
        # 生成 scale 参数
        self.gamma_net = nn.Sequential(
            nn.Linear(autophase_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        # 生成 shift 参数
        self.beta_net = nn.Sequential(
            nn.Linear(autophase_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 初始化为恒等变换
        nn.init.ones_(self.gamma_net[-1].weight.data.mean(dim=1))
        nn.init.zeros_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)
    
    def forward(self, hidden_states, autophase):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            autophase: [batch, autophase_dim]
        Returns:
            modulated hidden_states: [batch, seq_len, hidden_dim]
        """
        gamma = self.gamma_net(autophase).unsqueeze(1)  # [batch, 1, hidden]
        beta = self.beta_net(autophase).unsqueeze(1)    # [batch, 1, hidden]
        
        return gamma * hidden_states + beta


class FiLMConditionedEncoder(nn.Module):
    """使用 FiLM 调制的 Encoder"""
    
    def __init__(self, encoder, autophase_dim=56, film_layers=None):
        super().__init__()
        self.encoder = encoder
        hidden_dim = encoder.config.hidden_size
        num_layers = encoder.config.num_hidden_layers
        
        # 决定在哪些层应用 FiLM
        if film_layers is None:
            # 默认在每隔一层应用
            film_layers = list(range(0, num_layers, 2))
        self.film_layers = set(film_layers)
        
        # 为每个 FiLM 层创建调制器
        self.film_modulators = nn.ModuleDict({
            str(i): FiLMLayer(autophase_dim, hidden_dim)
            for i in film_layers
        })
    
    def forward(self, input_ids, attention_mask, autophase=None, **kwargs):
        # 这里需要 hook 或修改 encoder 内部
        # 简化示例：在 encoder 输出后应用 FiLM
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs.last_hidden_state
        
        if autophase is not None:
            # 对最终输出应用 FiLM 调制
            # 完整实现需要修改 encoder 内部，在每层后应用
            for layer_idx in sorted(self.film_layers):
                if str(layer_idx) in self.film_modulators:
                    hidden_states = self.film_modulators[str(layer_idx)](
                        hidden_states, autophase
                    )
        
        outputs.last_hidden_state = hidden_states
        return outputs
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 参数高效 | 需要深入修改 transformer 内部 |
| 可在任意层注入条件信息 | 调制方式可能不如 attention 灵活 |
| 计算开销小 | 实现需要 hook 或重写 forward |

---

### 方案5: 双编码器架构

#### 原理

为 Autophase 设计一个专门的小型编码器，将 IR 语义特征和静态特征分别编码后，通过 attention 机制进行融合。

#### 架构图

```
LLVM IR → [InstBERT Encoder] → ir_features [seq_len, 768]
                                      ↓
                              [Fusion Attention]  ← autophase_features [1, 768]
                                      ↓                      ↑
                              fused_features         [Autophase Encoder]
                                      ↓                      ↑
                              [GPT2 Decoder]            Autophase [56]
                                      ↓
                              优化 Pass 序列
```

#### 代码实现

```python
class AutophaseEncoder(nn.Module):
    """Autophase 特征专用编码器"""
    
    def __init__(self, autophase_dim=56, hidden_dim=768, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(autophase_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for i in range(num_layers)
        ])
    
    def forward(self, autophase):
        # autophase: [batch, 56]
        hidden = autophase
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden.unsqueeze(1)  # [batch, 1, 768]


class FeatureFusion(nn.Module):
    """IR 特征与 Autophase 特征融合模块"""
    
    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        # IR 作为 query，Autophase 作为 key/value
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # 自注意力融合
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(self, ir_features, autophase_features, attention_mask=None):
        # Cross attention: IR 关注 Autophase
        attn_out, _ = self.cross_attn(
            query=ir_features,
            key=autophase_features,
            value=autophase_features
        )
        ir_features = self.norm1(ir_features + attn_out)
        
        # Self attention
        if attention_mask is not None:
            # 转换为 attention mask 格式
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
            
        attn_out, _ = self.self_attn(
            ir_features, ir_features, ir_features,
            key_padding_mask=key_padding_mask
        )
        ir_features = self.norm2(ir_features + attn_out)
        
        # FFN
        ir_features = self.norm3(ir_features + self.ffn(ir_features))
        
        return ir_features


class DualEncoderModel(nn.Module):
    """双编码器模型"""
    
    def __init__(
        self, 
        ir_encoder,  # InstBERT
        decoder,     # GPT2
        autophase_dim=56,
        hidden_dim=768
    ):
        super().__init__()
        self.ir_encoder = ir_encoder
        self.decoder = decoder
        self.autophase_encoder = AutophaseEncoder(autophase_dim, hidden_dim)
        self.fusion = FeatureFusion(hidden_dim)
    
    def forward(
        self,
        input_ids,
        attention_mask,
        autophase,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        **kwargs
    ):
        # 编码 IR
        ir_outputs = self.ir_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        ir_features = ir_outputs.last_hidden_state  # [batch, seq_len, 768]
        
        # 编码 Autophase
        autophase_features = self.autophase_encoder(autophase)  # [batch, 1, 768]
        
        # 融合
        fused_features = self.fusion(ir_features, autophase_features, attention_mask)
        
        # Decoder
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=fused_features,
            encoder_attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 模块化设计，职责清晰 | 参数量增加较多 |
| 两种特征有独立处理流程 | 训练成本更高 |
| 融合方式灵活 | 架构复杂度高 |
| 可扩展支持更多特征类型 | |

---

## 3. 方案对比与建议

### 3.1 方案对比表

| 方案 | 实现复杂度 | 参数增量 | 效果潜力 | 推荐场景 |
|------|-----------|---------|---------|---------|
| **方案1: 特征拼接** | ⭐ | ~0.6M | ⭐⭐ | 快速验证 autophase 有效性 |
| **方案2: Decoder 前缀** | ⭐⭐ | ~2.4M | ⭐⭐⭐ | 希望 decoder 主动利用特征 |
| **方案3: Cross-Attention** | ⭐⭐⭐ | ~7M | ⭐⭐⭐⭐ | 追求更好效果 |
| **方案4: FiLM 调制** | ⭐⭐⭐ | ~1.5M | ⭐⭐⭐ | 参数效率优先 |
| **方案5: 双编码器** | ⭐⭐⭐⭐ | ~10M | ⭐⭐⭐⭐ | 完整架构设计 |

### 3.2 推荐实施路径

1. **第一阶段：验证可行性**
   - 实现方案1（特征拼接）
   - 对比有无 autophase 的模型性能
   - 确认静态特征对任务是否有帮助

2. **第二阶段：优化融合方式**
   - 如果方案1有效，尝试方案2或方案3
   - 比较不同融合方式的效果

3. **第三阶段：完整架构**
   - 根据前两阶段的结论
   - 如需要更强的特征交互，实现方案5

### 3.3 训练注意事项

1. **数据准备**：确保 dataloader 正确传递 autophase 特征
2. **特征归一化**：建议对 autophase 特征进行标准化或归一化
3. **学习率**：新增模块可使用更高的学习率
4. **渐进训练**：可先冻结原模型，只训练新增模块

---

## 4. 附录

### 4.1 Autophase 特征完整列表

```python
AUTOPHASE_FEATURE_NAMES = [
    "BBNumArgsHi", "BBNumArgsLo", "onePred", "onePredOneSuc", "onePredTwoSuc",
    "oneSuccessor", "twoPred", "twoPredOneSuc", "twoEach", "twoSuccessor",
    "morePreds", "BB03Phi", "BBHiPhi", "BBNoPhi", "BeginPhi",
    "BranchCount", "returnInt", "CriticalCount", "NumEdges", "const32Bit",
    "const64Bit", "numConstZeroes", "numConstOnes", "UncondBranches", "binaryConstArg",
    "NumAShrInst", "NumAddInst", "NumAllocaInst", "NumAndInst", "BlockMid",
    "BlockLow", "NumBitCastInst", "NumBrInst", "NumCallInst", "NumGetElementPtrInst",
    "NumICmpInst", "NumLShrInst", "NumLoadInst", "NumMulInst", "NumOrInst",
    "NumPHIInst", "NumRetInst", "NumSExtInst", "NumSelectInst", "NumShlInst",
    "NumStoreInst", "NumSubInst", "NumTruncInst", "NumXorInst", "NumZExtInst",
    "TotalBlocks", "TotalInsts", "TotalMemInst", "TotalFuncs", "ArgsPhi", "testUnary"
]

AUTOPHASE_FEATURE_DIM = 56
```

### 4.2 HuggingFace Trainer 兼容性

自定义模型可以无缝使用 HuggingFace Trainer，只需满足以下条件：

#### 模型要求

1. 继承 `nn.Module` 或 `PreTrainedModel`
2. `forward()` 返回包含 `loss` 的对象（dict、tuple 或 ModelOutput）

```python
# 方式1：简单继承 nn.Module
class MyModel(nn.Module):
    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.compute_logits(input_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

# 方式2：继承 PreTrainedModel（推荐，支持 save/load）
class MyModel(PreTrainedModel):
    config_class = MyConfig
    
    def forward(self, input_ids, labels=None, **kwargs):
        # 同上
        return MyModelOutput(loss=loss, logits=logits)
```

#### 自定义输入字段（如 autophase）

对于自定义输入字段，需要：

1. **设置 `remove_unused_columns=False`**：防止 Trainer 移除非标准列

```python
training_args = TrainingArguments(
    output_dir=work_dir,
    remove_unused_columns=False,  # 重要！
    ...
)
```

2. **使用自定义 DataCollator**：正确批处理自定义字段

```python
class AutophaseDataCollator:
    def __call__(self, features):
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
            "autophase": torch.stack([f["autophase"] for f in features]),  # 自定义字段
        }
        return batch

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=AutophaseDataCollator(),
)
```

#### 已实现的代码

本项目已实现完整的 HuggingFace 兼容模型：

- **模型**: `src/model/autophase_encoder_decoder.py`
  - `AutophaseEncoderDecoderModel`: 支持多种融合方法的 Encoder-Decoder
  - `AutophaseEncoderDecoderConfig`: 配置类
  - `AutophaseDataCollator`: 数据批处理器

- **训练脚本**: `src/training/passformer_autophase_train.py`
  - 完整的训练流程，使用 HuggingFace Trainer

#### 使用示例

```bash
python -m src.training.passformer_autophase_train \
    --config configs/passformer_autophase.yaml
```

配置文件示例：

```yaml
model:
  instbert_id: "path/to/instbert"
  inst2vec_tokenizer_id: "path/to/tokenizer"
  opti_seq_tokenizer_id: "path/to/opti_seq_tokenizer"

autophase:
  fusion_method: "concat"  # 可选: concat, add, cross_attention, prefix
  dim: 56
  num_prefix_tokens: 4  # 仅 prefix 方法使用

gpt2_config:
  n_embd: 768
  n_head: 12
  n_layer: 6
  vocab_size: 128

data:
  data_dir: "path/to/dataset"
  encoder_maxlen: 1024
  decoder_maxlen: 256

training_args:
  per_device_train_batch_size: 8
  learning_rate: 5e-5
  num_train_epochs: 10
```

### 4.3 参考文献

1. Huang, Q., et al. (2019). *Autophase: Compiler phase-ordering for HLS with deep reinforcement learning.* FCCM.
2. Perez, E., et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI.
3. Li, X. L., & Liang, P. (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* ACL.
4. HuggingFace Transformers Documentation. *Custom Models.* https://huggingface.co/docs/transformers/custom_models

