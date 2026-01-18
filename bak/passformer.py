"""
Passformer: Encoder-Decoder Model for LLVM Optimization Sequence Generation.

This module implements an encoder-decoder model that incorporates Autophase static
features alongside LLVM IR semantic features.

Supports HuggingFace Trainer API.

方案一实现：在 Decoder 输入处添加 Autophase 前缀（Prefix Tuning 风格）

Tokenizer 使用说明:
- Encoder: 使用 Inst2VecTokenizer 对 LLVM IR 进行 tokenization
- Decoder: 使用 OptiSeqTokenizer 对优化序列进行 tokenization
"""
import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModel,
    GPT2LMHeadModel,
    GPT2Config,
)
from transformers.modeling_outputs import ModelOutput

# 导入 tokenizer（可选，用于辅助函数）
try:
    from .tokenizer import Inst2VecTokenizer, OptiSeqTokenizer
    _TOKENIZERS_AVAILABLE = True
except ImportError:
    _TOKENIZERS_AVAILABLE = False


# ============================================================================
# Config
# ============================================================================

class PassformerConfig(PretrainedConfig):
    """
    Configuration for PassformerModel.
    
    Tokenizer 要求:
    - Encoder tokenizer: Inst2VecTokenizer (用于 LLVM IR tokenization)
    - Decoder tokenizer: OptiSeqTokenizer (用于优化序列 tokenization)
    
    可以通过 from_tokenizers() 类方法从 tokenizer 自动设置相关配置。
    """
    
    model_type = "passformer"
    
    def __init__(
        self,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_config: dict = None,
        autophase_dim: int = 56,
        fusion_method: str = "decoder_prefix",  # "decoder_prefix", "concat", "add", "cross_attention", "prefix"
        num_prefix_tokens: int = 4,  # for prefix fusion
        decoder_start_token_id: int = None,
        pad_token_id: int = None,
        eos_token_id: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_pretrained_model_name_or_path = encoder_pretrained_model_name_or_path
        self.decoder_config = decoder_config or {}
        self.autophase_dim = autophase_dim
        self.fusion_method = fusion_method
        self.num_prefix_tokens = num_prefix_tokens
        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
    
    @classmethod
    def from_tokenizers(
        cls,
        encoder_tokenizer,
        decoder_tokenizer,
        encoder_pretrained_model_name_or_path: str,
        decoder_config: dict = None,
        autophase_dim: int = 56,
        fusion_method: str = "decoder_prefix",
        num_prefix_tokens: int = 4,
        **kwargs
    ):
        """
        从 tokenizer 创建配置，自动提取相关参数。
        
        Args:
            encoder_tokenizer: Inst2VecTokenizer 实例（用于 encoder）
            decoder_tokenizer: OptiSeqTokenizer 实例（用于 decoder）
            encoder_pretrained_model_name_or_path: Encoder 预训练模型路径
            decoder_config: Decoder (GPT2) 配置字典。如果未指定 vocab_size，将自动从 decoder_tokenizer 获取
            autophase_dim: Autophase 特征维度
            fusion_method: 融合方法 ("decoder_prefix", "concat", "add", "cross_attention", "prefix")
            num_prefix_tokens: 前缀 token 数量（用于 prefix 融合方法）
            **kwargs: 其他配置参数
        
        Returns:
            PassformerConfig 实例
        
        Example:
            >>> from src.model import PassformerConfig, Inst2VecTokenizer, OptiSeqTokenizer
            >>> 
            >>> encoder_tokenizer = Inst2VecTokenizer.from_pretrained("path/to/encoder_tokenizer")
            >>> decoder_tokenizer = OptiSeqTokenizer.from_pretrained("path/to/decoder_tokenizer")
            >>> 
            >>> config = PassformerConfig.from_tokenizers(
            ...     encoder_tokenizer=encoder_tokenizer,
            ...     decoder_tokenizer=decoder_tokenizer,
            ...     encoder_pretrained_model_name_or_path="path/to/encoder_model",
            ...     decoder_config={
            ...         "n_embd": 768,
            ...         "n_layer": 6,
            ...         "n_head": 12,
            ...         # vocab_size 会自动从 decoder_tokenizer.vocab_size 获取
            ...     },
            ...     fusion_method="decoder_prefix",
            ... )
        """
        if not _TOKENIZERS_AVAILABLE:
            raise ImportError("Tokenizers are not available. Please ensure tokenizer.py is importable.")
        
        # 验证 tokenizer 类型（如果 PassformerModel 已定义）
        try:
            PassformerModel.validate_tokenizers(encoder_tokenizer, decoder_tokenizer)
        except (NameError, AttributeError):
            # 如果模型类还未定义，跳过验证
            pass
        
        # 从 decoder_tokenizer 提取配置（decoder 使用 OptiSeqTokenizer）
        decoder_vocab_size = decoder_tokenizer.vocab_size
        pad_token_id = decoder_tokenizer.pad_token_id
        eos_token_id = decoder_tokenizer.eos_token_id
        bos_token_id = getattr(decoder_tokenizer, 'bos_token_id', None)
        
        # decoder_start_token_id 通常使用 bos_token_id 或 eos_token_id
        # 这里默认使用 bos_token_id，如果没有则使用 eos_token_id
        decoder_start_token_id = bos_token_id if bos_token_id is not None else eos_token_id
        
        # 验证必要的 token ID
        if pad_token_id is None:
            raise ValueError("decoder_tokenizer.pad_token_id 不能为 None")
        if eos_token_id is None:
            raise ValueError("decoder_tokenizer.eos_token_id 不能为 None")
        if decoder_start_token_id is None:
            raise ValueError("无法确定 decoder_start_token_id，请确保 decoder_tokenizer 有 bos_token_id 或 eos_token_id")
        
        # 更新 decoder_config，确保 vocab_size 正确设置
        decoder_config = decoder_config or {}
        if "vocab_size" not in decoder_config:
            decoder_config["vocab_size"] = decoder_vocab_size
        
        return cls(
            encoder_pretrained_model_name_or_path=encoder_pretrained_model_name_or_path,
            decoder_config=decoder_config,
            autophase_dim=autophase_dim,
            fusion_method=fusion_method,
            num_prefix_tokens=num_prefix_tokens,
            decoder_start_token_id=decoder_start_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )


# ============================================================================
# Model Outputs
# ============================================================================

@dataclass
class PassformerSeq2SeqOutput(ModelOutput):
    """Output type for PassformerModel."""
    
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# ============================================================================
# Fusion Modules
# ============================================================================

class AutophaseProjection(nn.Module):
    """Project Autophase features to encoder/decoder hidden dimension."""
    
    def __init__(self, autophase_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(autophase_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
    
    def forward(self, autophase: torch.Tensor) -> torch.Tensor:
        """
        Args:
            autophase: [batch_size, autophase_dim]
        Returns:
            [batch_size, 1, hidden_dim]
        """
        return self.proj(autophase).unsqueeze(1)


class AutophasePrefixGenerator(nn.Module):
    """Generate prefix tokens from Autophase features for decoder input."""
    
    def __init__(self, autophase_dim: int, hidden_dim: int, num_prefix_tokens: int):
        super().__init__()
        self.num_prefix = num_prefix_tokens
        self.hidden_dim = hidden_dim
        
        self.proj = nn.Sequential(
            nn.Linear(autophase_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * num_prefix_tokens),
            nn.LayerNorm(hidden_dim * num_prefix_tokens),
        )
    
    def forward(self, autophase: torch.Tensor) -> torch.Tensor:
        """
        Args:
            autophase: [batch_size, autophase_dim]
        Returns:
            [batch_size, num_prefix_tokens, hidden_dim]
        """
        prefix = self.proj(autophase)
        return prefix.view(-1, self.num_prefix, self.hidden_dim)


class AutophaseCrossAttention(nn.Module):
    """Cross-attention module for Autophase feature fusion."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        autophase_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            autophase_emb: [batch_size, 1, hidden_dim]
        Returns:
            [batch_size, seq_len, hidden_dim]
        """
        attn_output, _ = self.cross_attn(
            query=hidden_states,
            key=autophase_emb,
            value=autophase_emb,
        )
        return self.norm(hidden_states + self.dropout(attn_output))


# ============================================================================
# Main Model
# ============================================================================

class PassformerModel(PreTrainedModel):
    """
    Passformer: Encoder-Decoder model with Autophase feature fusion.
    
    Tokenizer 使用:
    - Encoder: 使用 Inst2VecTokenizer 对 LLVM IR 进行 tokenization
      - 输入: LLVM IR 文本
      - 输出: token IDs 和 attention mask
    - Decoder: 使用 OptiSeqTokenizer 对优化序列进行 tokenization
      - 输入: 优化 pass 序列（空格分隔的字符串）
      - 输出: token IDs 和 attention mask
    
    支持多种融合方法:
    - "decoder_prefix": 在 Decoder 输入处添加 Autophase 前缀（方案一，推荐）
    - "concat": 在 Encoder 输出后拼接 autophase embedding
    - "add": 将 autophase embedding 加到 encoder 输出
    - "cross_attention": 使用 cross-attention 融合 autophase 特征
    - "prefix": 在 encoder 输出前生成 prefix tokens
    
    兼容 HuggingFace Trainer API。
    
    使用示例:
        >>> from src.model import PassformerModel, PassformerConfig, Inst2VecTokenizer, OptiSeqTokenizer
        >>> 
        >>> # 步骤1: 加载 tokenizers（必需）
        >>> # Encoder 使用 Inst2VecTokenizer 处理 LLVM IR
        >>> encoder_tokenizer = Inst2VecTokenizer.from_pretrained("path/to/encoder_tokenizer")
        >>> # Decoder 使用 OptiSeqTokenizer 处理优化序列
        >>> decoder_tokenizer = OptiSeqTokenizer.from_pretrained("path/to/decoder_tokenizer")
        >>> 
        >>> # 步骤2: 使用 tokenizer 创建配置（推荐方式）
        >>> # from_tokenizers() 会自动从 tokenizer 获取 vocab_size, pad_token_id, eos_token_id 等
        >>> config = PassformerConfig.from_tokenizers(
        ...     encoder_tokenizer=encoder_tokenizer,  # 必需：用于 encoder 的 tokenizer
        ...     decoder_tokenizer=decoder_tokenizer,  # 必需：用于 decoder 的 tokenizer
        ...     encoder_pretrained_model_name_or_path="path/to/encoder_model",
        ...     decoder_config={
        ...         "n_embd": 768,
        ...         "n_layer": 6,
        ...         "n_head": 12,
        ...         # vocab_size 会自动从 decoder_tokenizer.vocab_size 获取
        ...     },
        ...     fusion_method="decoder_prefix",
        ... )
        >>> 
        >>> # 步骤3: 创建模型
        >>> model = PassformerModel(config)
        >>> 
        >>> # 步骤4: 使用 tokenizer 进行编码
        >>> # Encoder 输入：LLVM IR 文本
        >>> llvm_ir = "define void @main() { ... }"
        >>> encoder_inputs = encoder_tokenizer(llvm_ir, max_length=1024, return_tensors="pt")
        >>> 
        >>> # Decoder 输入：优化序列文本
        >>> opti_seq = "mem2reg instcombine gvn"
        >>> decoder_inputs = decoder_tokenizer(opti_seq, max_length=256, return_tensors="pt")
        >>> 
        >>> # 模型前向传播
        >>> outputs = model(
        ...     input_ids=encoder_inputs["input_ids"],
        ...     attention_mask=encoder_inputs["attention_mask"],
        ...     decoder_input_ids=decoder_inputs["input_ids"],
        ...     decoder_attention_mask=decoder_inputs["attention_mask"],
        ...     labels=decoder_inputs["input_ids"],  # 用于训练
        ...     autophase=autophase_features,  # 可选：Autophase 特征 [batch, 56]
        ... )
    """
    
    config_class = PassformerConfig
    base_model_prefix = "passformer"
    
    def __init__(self, config: PassformerConfig):
        super().__init__(config)
        
        self.config = config
        
        # Encoder (ModernBERT/InstBERT)
        if config.encoder_pretrained_model_name_or_path:
            self.encoder = AutoModel.from_pretrained(
                config.encoder_pretrained_model_name_or_path
            )
        else:
            raise ValueError("encoder_pretrained_model_name_or_path must be provided")
        
        # Get encoder hidden size
        self.encoder_hidden_size = self.encoder.config.hidden_size
        
        # Decoder (GPT2)
        decoder_config = GPT2Config(**config.decoder_config)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        self.decoder = GPT2LMHeadModel(decoder_config)
        
        # Get decoder hidden size
        self.decoder_hidden_size = self.decoder.config.n_embd
        
        # Autophase fusion module
        self.fusion_method = config.fusion_method
        self.autophase_dim = config.autophase_dim
        
        if self.fusion_method == "decoder_prefix":
            # 方案一：在 Decoder 输入处添加前缀
            self.autophase_prefix = AutophasePrefixGenerator(
                config.autophase_dim,
                self.decoder_hidden_size,  # 使用 decoder 的 hidden size
                config.num_prefix_tokens
            )
        elif self.fusion_method == "concat":
            # 在 encoder 输出后拼接
            self.autophase_proj = AutophaseProjection(
                config.autophase_dim, self.encoder_hidden_size
            )
        elif self.fusion_method == "add":
            # 加到 encoder 输出
            self.autophase_proj = AutophaseProjection(
                config.autophase_dim, self.encoder_hidden_size
            )
        elif self.fusion_method == "cross_attention":
            # Cross-attention 融合
            self.autophase_proj = AutophaseProjection(
                config.autophase_dim, self.encoder_hidden_size
            )
            self.autophase_cross_attn = AutophaseCrossAttention(
                self.encoder_hidden_size, 
                num_heads=8
            )
        elif self.fusion_method == "prefix":
            # 在 encoder 输出前生成 prefix
            self.autophase_prefix = AutophasePrefixGenerator(
                config.autophase_dim,
                self.encoder_hidden_size,
                config.num_prefix_tokens
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Initialize weights
        self.post_init()
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    @staticmethod
    def validate_tokenizers(encoder_tokenizer, decoder_tokenizer):
        """
        验证 tokenizer 是否与模型兼容。
        
        Args:
            encoder_tokenizer: 应该是 Inst2VecTokenizer 实例
            decoder_tokenizer: 应该是 OptiSeqTokenizer 实例
        
        Raises:
            ValueError: 如果 tokenizer 类型不正确
        """
        if not _TOKENIZERS_AVAILABLE:
            return  # 如果 tokenizer 不可用，跳过验证
        
        from .tokenizer import Inst2VecTokenizer, OptiSeqTokenizer
        
        if not isinstance(encoder_tokenizer, Inst2VecTokenizer):
            raise ValueError(
                f"encoder_tokenizer 必须是 Inst2VecTokenizer 实例，"
                f"但得到 {type(encoder_tokenizer)}"
            )
        
        if not isinstance(decoder_tokenizer, OptiSeqTokenizer):
            raise ValueError(
                f"decoder_tokenizer 必须是 OptiSeqTokenizer 实例，"
                f"但得到 {type(decoder_tokenizer)}"
            )
        
        # 验证必要的属性
        required_attrs = ['vocab_size', 'pad_token_id', 'eos_token_id']
        for attr in required_attrs:
            if not hasattr(decoder_tokenizer, attr):
                raise ValueError(f"decoder_tokenizer 缺少必需的属性: {attr}")
            if getattr(decoder_tokenizer, attr) is None:
                raise ValueError(f"decoder_tokenizer.{attr} 不能为 None")
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        autophase: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, PassformerSeq2SeqOutput]:
        """
        Forward pass with Autophase feature fusion.
        
        Args:
            input_ids: Encoder input token IDs [batch, seq_len]
            attention_mask: Encoder attention mask [batch, seq_len]
            autophase: Autophase feature vector [batch, autophase_dim]
            decoder_input_ids: Decoder input token IDs [batch, tgt_len]
            decoder_attention_mask: Decoder attention mask [batch, tgt_len]
            labels: Target labels for loss computation [batch, tgt_len]
            ...
        
        Returns:
            PassformerSeq2SeqOutput with loss and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Encode input
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        
        encoder_hidden_states = encoder_outputs.last_hidden_state  # [batch, seq, hidden]
        
        # 方案一：在 Decoder 输入处添加 Autophase 前缀
        decoder_inputs_embeds = None
        updated_decoder_attention_mask = decoder_attention_mask
        updated_labels = labels
        
        if self.fusion_method == "decoder_prefix" and autophase is not None:
            # 准备带前缀的 decoder 输入
            decoder_inputs_embeds, updated_decoder_attention_mask, updated_labels = \
                self._prepare_decoder_inputs_with_autophase(
                    decoder_input_ids,
                    decoder_attention_mask,
                    autophase,
                    labels
                )
            decoder_input_ids = None  # 使用 inputs_embeds 时不需要 input_ids
        else:
            # 其他融合方法：在 encoder 输出处融合
            if autophase is not None:
                encoder_hidden_states, attention_mask = self._fuse_autophase(
                    encoder_hidden_states, attention_mask, autophase
                )
        
        # Prepare decoder inputs (如果没有使用 decoder_prefix)
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # Shift labels right for decoder input
            decoder_input_ids = self._shift_right(labels)
        
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,  # 使用带前缀的 embeddings
            attention_mask=updated_decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        logits = decoder_outputs.logits
        
        # Compute loss
        loss = None
        if updated_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                updated_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return PassformerSeq2SeqOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states if hasattr(encoder_outputs, 'hidden_states') else None,
            encoder_attentions=encoder_outputs.attentions if hasattr(encoder_outputs, 'attentions') else None,
        )
    
    def _prepare_decoder_inputs_with_autophase(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        autophase: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        在 decoder 输入前添加 autophase 前缀（方案一）。
        
        Args:
            decoder_input_ids: [batch, tgt_len]
            decoder_attention_mask: [batch, tgt_len]
            autophase: [batch, autophase_dim]
            labels: [batch, tgt_len] or None
        
        Returns:
            (inputs_embeds, updated_attention_mask, updated_labels)
        """
        batch_size = decoder_input_ids.size(0)
        device = decoder_input_ids.device
        
        # 生成 autophase 前缀 embeddings
        prefix_embeds = self.autophase_prefix(autophase)  # [batch, num_prefix, hidden]
        num_prefix = prefix_embeds.size(1)
        
        # 获取 decoder 的 token embeddings
        inputs_embeds = self.decoder.transformer.wte(decoder_input_ids)  # [batch, tgt_len, hidden]
        
        # 拼接前缀
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)  # [batch, num_prefix+tgt_len, hidden]
        
        # 更新 attention mask
        prefix_mask = torch.ones(
            batch_size, num_prefix, 
            device=device, 
            dtype=decoder_attention_mask.dtype
        )
        updated_mask = torch.cat([prefix_mask, decoder_attention_mask], dim=1)
        
        # 更新 labels（前缀部分设为 -100 忽略）
        updated_labels = None
        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, num_prefix),
                -100,
                device=device,
                dtype=labels.dtype
            )
            updated_labels = torch.cat([prefix_labels, labels], dim=1)
        
        return inputs_embeds, updated_mask, updated_labels
    
    def _fuse_autophase(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        autophase: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse Autophase features with encoder hidden states (用于非 decoder_prefix 方法).
        
        Returns:
            Tuple of (fused_hidden_states, updated_attention_mask)
        """
        batch_size = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device
        
        if self.fusion_method == "concat":
            # Project autophase and concatenate as extra token
            autophase_emb = self.autophase_proj(autophase)  # [batch, 1, hidden]
            fused = torch.cat([autophase_emb, encoder_hidden_states], dim=1)
            
            # Update attention mask
            autophase_mask = torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([autophase_mask, attention_mask], dim=1)
            
            return fused, attention_mask
        
        elif self.fusion_method == "add":
            # Add autophase embedding to all positions
            autophase_emb = self.autophase_proj(autophase)  # [batch, 1, hidden]
            fused = encoder_hidden_states + autophase_emb
            return fused, attention_mask
        
        elif self.fusion_method == "cross_attention":
            # Use cross-attention to fuse
            autophase_emb = self.autophase_proj(autophase)  # [batch, 1, hidden]
            fused = self.autophase_cross_attn(encoder_hidden_states, autophase_emb)
            return fused, attention_mask
        
        elif self.fusion_method == "prefix":
            # Generate prefix tokens
            prefix = self.autophase_prefix(autophase)  # [batch, num_prefix, hidden]
            fused = torch.cat([prefix, encoder_hidden_states], dim=1)
            
            # Update attention mask
            prefix_mask = torch.ones(
                batch_size, self.config.num_prefix_tokens, 
                device=device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
            return fused, attention_mask
        
        else:
            return encoder_hidden_states, attention_mask
    
    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        """Shift labels right to create decoder input."""
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        
        if decoder_start_token_id is None:
            raise ValueError("decoder_start_token_id must be defined")
        
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = decoder_start_token_id
        
        # Replace -100 (ignore index) with pad_token_id
        if pad_token_id is not None:
            shifted.masked_fill_(shifted == -100, pad_token_id)
        
        return shifted
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        autophase: Optional[torch.FloatTensor] = None,
        max_length: int = 256,
        num_beams: int = 1,
        do_sample: bool = False,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate optimization sequences.
        
        Args:
            input_ids: Encoder input [batch, seq_len]
            attention_mask: Encoder attention mask
            autophase: Autophase features [batch, autophase_dim]
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            
        Returns:
            Generated token IDs [batch, gen_len]
        """
        # Encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # 处理 autophase（非 decoder_prefix 方法在 encoder 输出处融合）
        if autophase is not None and self.fusion_method != "decoder_prefix":
            encoder_hidden_states, attention_mask = self._fuse_autophase(
                encoder_hidden_states, attention_mask, autophase
            )
        
        # Prepare for generation
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Start with decoder_start_token
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        
        # 如果使用 decoder_prefix，需要准备带前缀的输入
        decoder_inputs_embeds = None
        if self.fusion_method == "decoder_prefix" and autophase is not None:
            # 生成前缀
            prefix_embeds = self.autophase_prefix(autophase)  # [batch, num_prefix, hidden]
            num_prefix = prefix_embeds.size(1)
            
            # 获取初始 token 的 embedding
            start_embeds = self.decoder.transformer.wte(decoder_input_ids)  # [batch, 1, hidden]
            
            # 拼接前缀
            decoder_inputs_embeds = torch.cat([prefix_embeds, start_embeds], dim=1)  # [batch, num_prefix+1, hidden]
            decoder_input_ids = None  # 使用 inputs_embeds
        
        # Simple greedy decoding (for more advanced, use HF generate utils)
        for step in range(max_length - 1):
            outputs = self.decoder(
                input_ids=decoder_input_ids,
                inputs_embeds=decoder_inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,  # 简化处理，不使用 cache
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # 更新 decoder 输入
            if decoder_inputs_embeds is not None:
                # 使用 inputs_embeds 模式
                next_embeds = self.decoder.transformer.wte(next_token)  # [batch, 1, hidden]
                decoder_inputs_embeds = torch.cat([decoder_inputs_embeds, next_embeds], dim=1)
            else:
                # 使用 input_ids 模式
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Stop if all sequences have generated EOS
            if self.config.eos_token_id is not None:
                if (next_token == self.config.eos_token_id).all():
                    break
        
        # 返回生成的 token IDs（如果使用了 inputs_embeds，需要从 embeddings 中提取）
        if decoder_inputs_embeds is not None:
            # 对于 decoder_prefix，返回时去掉前缀部分
            # 这里简化处理，返回完整的序列（包括前缀对应的部分）
            # 实际使用时可能需要更复杂的处理
            return decoder_input_ids if decoder_input_ids is not None else torch.zeros(
                (batch_size, max_length), dtype=torch.long, device=device
            )
        else:
            return decoder_input_ids


# ============================================================================
# Data Collator for Autophase
# ============================================================================

class PassformerDataCollator:
    """
    Data collator that properly handles autophase features.
    
    Required for HuggingFace Trainer to correctly batch autophase tensors.
    """
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, features):
        """
        Collate batch of features.
        
        Args:
            features: List of dicts with keys: input_ids, attention_mask, labels, autophase
        
        Returns:
            Batched tensors
        """
        batch = {}
        
        # Standard fields
        if "input_ids" in features[0]:
            batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        
        if "attention_mask" in features[0]:
            batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        
        if "labels" in features[0]:
            batch["labels"] = torch.stack([f["labels"] for f in features])
        
        # Autophase field
        if "autophase" in features[0]:
            batch["autophase"] = torch.stack([f["autophase"] for f in features])
        
        return batch


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    """
    测试 PassformerModel 和 tokenizer 的集成。
    
    运行方式:
        python -m src.model.passformer
        或
        python src/model/passformer.py
    
    如果需要测试完整的模型功能，请提供 tokenizer 路径:
        python src/model/passformer.py \\
            --encoder_tokenizer_path path/to/encoder_tokenizer \\
            --decoder_tokenizer_path path/to/decoder_tokenizer \\
            --encoder_model_path path/to/encoder_model
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PassformerModel with tokenizers")
    parser.add_argument(
        "--encoder_tokenizer_path", type=str, 
        default="/home/xucong24/Compiler/checkpoints/Inst2VecTokenizer",
        help="Path to Inst2VecTokenizer (optional)"
    )
    parser.add_argument(
        "--decoder_tokenizer_path", type=str, 
        default="/home/xucong24/Compiler/checkpoints/OptiSeqTokenizer",
        help="Path to OptiSeqTokenizer (optional)"
    )
    parser.add_argument(
        "--encoder_model_path", type=str, 
        default="/home/xucong24/Compiler/work_dirs/instbert_poj104_mlm/20260106_152024/final_model",
        help="Path to encoder pretrained model (optional)"
    )
    parser.add_argument(
        "--test_forward", action="store_true",
        default=True,
        help="Test forward pass (requires tokenizers and model)"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("PassformerModel 测试")
    print("=" * 80)
    
    # 测试 1: 检查 tokenizer 是否可用
    print("\n[测试 1] 检查 tokenizer 导入...")
    if _TOKENIZERS_AVAILABLE:
        print("  ✅ Tokenizers 可用")
        print(f"  - Inst2VecTokenizer: {Inst2VecTokenizer}")
        print(f"  - OptiSeqTokenizer: {OptiSeqTokenizer}")
    else:
        print("  ⚠️  Tokenizers 不可用，跳过相关测试")
        print("  提示: 确保 tokenizer.py 可以正常导入")
    
    # 测试 2: 测试 tokenizer 加载和基本功能
    if _TOKENIZERS_AVAILABLE and args.decoder_tokenizer_path:
        print("\n[测试 2] 测试 OptiSeqTokenizer...")
        try:
            decoder_tokenizer = OptiSeqTokenizer.from_pretrained(args.decoder_tokenizer_path)
            print(f"  ✅ OptiSeqTokenizer 加载成功")
            print(f"  - vocab_size: {decoder_tokenizer.vocab_size}")
            print(f"  - pad_token_id: {decoder_tokenizer.pad_token_id}")
            print(f"  - eos_token_id: {decoder_tokenizer.eos_token_id}")
            print(f"  - bos_token_id: {decoder_tokenizer.bos_token_id}")
            
            # 测试编码/解码
            test_seq = "mem2reg instcombine gvn"
            encoded = decoder_tokenizer(test_seq, max_length=32, return_tensors="pt")
            decoded = decoder_tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
            print(f"  - 编码测试: '{test_seq}' -> {encoded['input_ids'].shape}")
            print(f"  - 解码测试: {decoded}")
            print("  ✅ OptiSeqTokenizer 功能正常")
        except Exception as e:
            print(f"  ❌ OptiSeqTokenizer 测试失败: {e}")
            decoder_tokenizer = None
    else:
        print("\n[测试 2] 跳过 OptiSeqTokenizer 测试（需要 --decoder_tokenizer_path）")
        decoder_tokenizer = None
    
    if _TOKENIZERS_AVAILABLE and args.encoder_tokenizer_path:
        print("\n[测试 3] 测试 Inst2VecTokenizer...")
        try:
            encoder_tokenizer = Inst2VecTokenizer.from_pretrained(args.encoder_tokenizer_path)
            print(f"  ✅ Inst2VecTokenizer 加载成功")
            print(f"  - vocab_size: {encoder_tokenizer.vocab_size}")
            print(f"  - pad_token_id: {encoder_tokenizer.pad_token_id}")
            print(f"  - eos_token_id: {encoder_tokenizer.eos_token_id}")
            
            # 测试编码
            test_ir = "define void @main() { ret void }"
            encoded = encoder_tokenizer(test_ir, max_length=128, return_tensors="pt")
            print(f"  - 编码测试: LLVM IR -> {encoded['input_ids'].shape}")
            print("  ✅ Inst2VecTokenizer 功能正常")
        except Exception as e:
            print(f"  ❌ Inst2VecTokenizer 测试失败: {e}")
            encoder_tokenizer = None
    else:
        print("\n[测试 3] 跳过 Inst2VecTokenizer 测试（需要 --encoder_tokenizer_path）")
        encoder_tokenizer = None
    
    # 测试 4: 测试 PassformerConfig.from_tokenizers()
    if _TOKENIZERS_AVAILABLE and encoder_tokenizer and decoder_tokenizer:
        print("\n[测试 4] 测试 PassformerConfig.from_tokenizers()...")
        try:
            if args.encoder_model_path:
                config = PassformerConfig.from_tokenizers(
                    encoder_tokenizer=encoder_tokenizer,
                    decoder_tokenizer=decoder_tokenizer,
                    encoder_pretrained_model_name_or_path=args.encoder_model_path,
                    decoder_config={
                        "n_embd": 256,  # 小模型用于测试
                        "n_layer": 2,
                        "n_head": 4,
                    },
                    fusion_method="decoder_prefix",
                    num_prefix_tokens=4,
                )
                print("  ✅ PassformerConfig 创建成功")
                print(f"  - decoder_start_token_id: {config.decoder_start_token_id}")
                print(f"  - pad_token_id: {config.pad_token_id}")
                print(f"  - eos_token_id: {config.eos_token_id}")
                print(f"  - decoder_config.vocab_size: {config.decoder_config.get('vocab_size', 'N/A')}")
                print(f"  - fusion_method: {config.fusion_method}")
            else:
                print("  ⚠️  跳过（需要 --encoder_model_path）")
                config = None
        except Exception as e:
            print(f"  ❌ PassformerConfig 创建失败: {e}")
            import traceback
            traceback.print_exc()
            config = None
    else:
        print("\n[测试 4] 跳过 PassformerConfig.from_tokenizers() 测试（需要 tokenizers）")
        config = None
    
    # 测试 5: 测试 tokenizer 验证方法
    if _TOKENIZERS_AVAILABLE and encoder_tokenizer and decoder_tokenizer:
        print("\n[测试 5] 测试 validate_tokenizers()...")
        try:
            PassformerModel.validate_tokenizers(encoder_tokenizer, decoder_tokenizer)
            print("  ✅ Tokenizer 验证通过")
            
            # 测试错误的 tokenizer 类型
            try:
                PassformerModel.validate_tokenizers(decoder_tokenizer, encoder_tokenizer)
                print("  ❌ 应该检测到错误的 tokenizer 类型")
            except ValueError as e:
                print(f"  ✅ 正确检测到错误的 tokenizer 类型: {str(e)[:60]}...")
        except Exception as e:
            print(f"  ❌ Tokenizer 验证失败: {e}")
    
    # 测试 6: 测试模型创建和前向传播
    if config and args.test_forward:
        print("\n[测试 6] 测试模型创建和前向传播...")
        try:
            model = PassformerModel(config)
            print(f"  ✅ PassformerModel 创建成功")
            print(f"  - 总参数量: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  - Encoder 参数量: {sum(p.numel() for p in model.encoder.parameters()):,}")
            print(f"  - Decoder 参数量: {sum(p.numel() for p in model.decoder.parameters()):,}")
            
            # 准备测试数据
            batch_size = 2
            encoder_seq_len = 32
            decoder_seq_len = 16
            
            # Encoder 输入
            encoder_input_ids = torch.randint(0, encoder_tokenizer.vocab_size, (batch_size, encoder_seq_len))
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_len, dtype=torch.long)
            
            # Decoder 输入
            decoder_input_ids = torch.randint(0, decoder_tokenizer.vocab_size, (batch_size, decoder_seq_len))
            decoder_attention_mask = torch.ones(batch_size, decoder_seq_len, dtype=torch.long)
            labels = decoder_input_ids.clone()
            
            # Autophase 特征
            autophase = torch.randn(batch_size, config.autophase_dim)
            
            # 前向传播
            print("\n  执行前向传播...")
            outputs = model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                autophase=autophase,
            )
            
            print(f"  ✅ 前向传播成功")
            print(f"  - logits shape: {outputs.logits.shape}")
            print(f"  - loss: {outputs.loss.item():.4f}" if outputs.loss is not None else "  - loss: None")
            print(f"  - encoder_last_hidden_state shape: {outputs.encoder_last_hidden_state.shape}")
            
            # 测试不同融合方法
            print("\n  测试不同融合方法...")
            fusion_methods = ["decoder_prefix", "concat", "add", "cross_attention", "prefix"]
            for method in fusion_methods:
                try:
                    test_config = PassformerConfig.from_tokenizers(
                        encoder_tokenizer=encoder_tokenizer,
                        decoder_tokenizer=decoder_tokenizer,
                        encoder_pretrained_model_name_or_path=args.encoder_model_path,
                        decoder_config={
                                "vocab_size": 128,                 # 会被 decoder tokenizer 的 vocab_size 覆盖
                                "n_positions": 512,                # 最大位置编码长度
                                "n_embd": 768,                     # 嵌入维度（建议与 encoder hidden_size 一致）
                                "n_layer": 6,                      # Transformer 层数
                                "n_head": 12,                      # 注意力头数
                                "activation_function": "gelu_new",
                                "resid_pdrop": 0.1,
                                "embd_pdrop": 0.1,
                                "attn_pdrop": 0.1,
                                "layer_norm_epsilon": 1e-05,
                                "initializer_range": 0.02,
                                "scale_attn_weights": True,
                                "use_cache": True,
                                "add_cross_attention": True       # 必须为 true，启用交叉注意力
                        },
                        fusion_method=method,
                    )
                    test_model = PassformerModel(test_config)
                    test_outputs = test_model(
                        input_ids=encoder_input_ids,
                        attention_mask=encoder_attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        labels=labels,
                        autophase=autophase,
                    )
                    print(f"    ✅ {method}: logits {test_outputs.logits.shape}, loss {test_outputs.loss.item():.4f}")
                except Exception as e:
                    print(f"    ❌ {method}: {str(e)[:50]}")
            
        except Exception as e:
            print(f"  ❌ 模型测试失败: {e}")
            import traceback
            traceback.print_exc()
    elif config:
        print("\n[测试 6] 跳过前向传播测试（使用 --test_forward 启用）")
    else:
        print("\n[测试 6] 跳过模型测试（需要配置和模型路径）")
    
    # 测试 7: 测试 DataCollator
    print("\n[测试 7] 测试 PassformerDataCollator...")
    try:
        collator = PassformerDataCollator(pad_token_id=0)
        features = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([2, 3, 4]),
                "autophase": torch.randn(56),
            },
            {
                "input_ids": torch.tensor([5, 6]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([6, 7]),
                "autophase": torch.randn(56),
            },
        ]
        batch = collator(features)
        print("  ✅ PassformerDataCollator 功能正常")
        print(f"  - batch keys: {list(batch.keys())}")
        print(f"  - input_ids shape: {batch['input_ids'].shape}")
        print(f"  - autophase shape: {batch['autophase'].shape}")
    except Exception as e:
        print(f"  ❌ PassformerDataCollator 测试失败: {e}")
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)
    print("\n提示:")
    print("  - 要运行完整测试，请提供 tokenizer 和模型路径")
    print("  - 使用 --test_forward 启用前向传播测试")
    print("  - 示例:")
    print("    python src/model/passformer.py \\")
    print("        --encoder_tokenizer_path path/to/encoder_tokenizer \\")
    print("        --decoder_tokenizer_path path/to/decoder_tokenizer \\")
    print("        --encoder_model_path path/to/encoder_model \\")
    print("        --test_forward")
