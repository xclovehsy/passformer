from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional


class AutophaseProjection(nn.Module):

    def __init__(
        self, 
        autophase_dim: int, 
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        if autophase_dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"autophase_dim and hidden_dim must be positive, got {autophase_dim} and {hidden_dim}")
        
        immediate_dim = hidden_dim // 2
        if immediate_dim == 0:
            raise ValueError(f"hidden_dim must be at least 2, got {hidden_dim}")

        self.proj = nn.Sequential(
            nn.Linear(autophase_dim, immediate_dim),
            nn.LayerNorm(immediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(immediate_dim, immediate_dim),
            nn.LayerNorm(immediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(immediate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward(self, autophase: torch.Tensor) -> torch.Tensor:
        if autophase.dim() != 2:
            raise ValueError(f"Expected 2D input [batch_size, autophase_dim], got shape {autophase.shape}")
        return self.proj(autophase).unsqueeze(1)


class AutophaseCrossAttention(nn.Module):
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        
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
        if hidden_states.dim() != 3 or autophase_emb.dim() != 3:
            raise ValueError(
                f"Expected 3D tensors, got hidden_states shape {hidden_states.shape}, "
                f"autophase_emb shape {autophase_emb.shape}"
            )
        
        attn_output, _ = self.cross_attn(
            query=hidden_states,
            key=autophase_emb,
            value=autophase_emb,
        )
        # Residual connection with layer norm (Post-LN architecture)
        return self.norm(hidden_states + self.dropout(attn_output))
