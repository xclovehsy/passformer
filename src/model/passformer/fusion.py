from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional


class AutophaseProjection(nn.Module):

    def __init__(
        self, 
        autophase_dim: int, 
        hidden_dim: int,
        dropout: float = 0.1,
        intermediate_dim: Optional[int] = None
    ):
        """
        Args:
            autophase_dim: Input autophase feature dimension
            hidden_dim: Output hidden dimension
            dropout: Dropout rate
            intermediate_dim: Intermediate dimension for MLP (default: hidden_dim // 2)
        """
        super().__init__()
        if autophase_dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"autophase_dim and hidden_dim must be positive, got {autophase_dim} and {hidden_dim}")
        
        # Default intermediate_dim: hidden_dim // 2
        if intermediate_dim is None:
            intermediate_dim = hidden_dim // 2
            if intermediate_dim == 0:
                raise ValueError(f"hidden_dim must be at least 2, got {hidden_dim}")
        elif intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim must be positive, got {intermediate_dim}")

        # 2-layer MLP: input -> intermediate -> output
        self.proj = nn.Sequential(
            nn.Linear(autophase_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
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


class AutophaseConcatFusion(nn.Module):
    """
    Concat fusion method:
    1. Project autophase features to embedding dimension via MLP
    2. Concat with encoder hidden states
    3. Project concat features back to decoder hidden size via MLP
    """
    
    def __init__(
        self,
        autophase_dim: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        autophase_emb_dim: Optional[int] = None,
        dropout: float = 0.1,
        autophase_intermediate_dim: Optional[int] = None,
        concat_intermediate_dim: Optional[int] = None
    ):
        """
        Args:
            autophase_dim: Input autophase feature dimension
            encoder_hidden_size: Encoder hidden dimension
            decoder_hidden_size: Decoder hidden dimension
            autophase_emb_dim: Target dimension for autophase embedding (default: encoder_hidden_size)
            dropout: Dropout rate
            autophase_intermediate_dim: Intermediate dim for autophase projection MLP (default: autophase_emb_dim // 2)
            concat_intermediate_dim: Intermediate dim for concat projection MLP (default: max(decoder_hidden_size, concat_dim // 2))
        """
        super().__init__()
        if autophase_dim <= 0:
            raise ValueError(f"autophase_dim must be positive, got {autophase_dim}")
        if encoder_hidden_size <= 0:
            raise ValueError(f"encoder_hidden_size must be positive, got {encoder_hidden_size}")
        if decoder_hidden_size <= 0:
            raise ValueError(f"decoder_hidden_size must be positive, got {decoder_hidden_size}")
        
        # Default: autophase_emb_dim = encoder_hidden_size
        if autophase_emb_dim is None:
            autophase_emb_dim = encoder_hidden_size
        elif autophase_emb_dim <= 0:
            raise ValueError(f"autophase_emb_dim must be positive, got {autophase_emb_dim}")
        
        # MLP to project autophase to embedding dimension
        # 2-layer MLP: input -> intermediate -> output
        if autophase_intermediate_dim is None:
            autophase_intermediate_dim = autophase_emb_dim // 2
            if autophase_intermediate_dim == 0:
                autophase_intermediate_dim = 1
        elif autophase_intermediate_dim <= 0:
            raise ValueError(f"autophase_intermediate_dim must be positive, got {autophase_intermediate_dim}")
        
        self.autophase_proj = nn.Sequential(
            nn.Linear(autophase_dim, autophase_intermediate_dim),
            nn.LayerNorm(autophase_intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(autophase_intermediate_dim, autophase_emb_dim),
            nn.LayerNorm(autophase_emb_dim),
        )
        
        # MLP to project concat features to decoder hidden size
        # 2-layer MLP: input -> intermediate -> output
        concat_dim = encoder_hidden_size + autophase_emb_dim
        if concat_intermediate_dim is None:
            concat_intermediate_dim = max(decoder_hidden_size, concat_dim // 2)
        elif concat_intermediate_dim <= 0:
            raise ValueError(f"concat_intermediate_dim must be positive, got {concat_intermediate_dim}")
        
        self.concat_proj = nn.Sequential(
            nn.Linear(concat_dim, concat_intermediate_dim),
            nn.LayerNorm(concat_intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(concat_intermediate_dim, decoder_hidden_size),
            nn.LayerNorm(decoder_hidden_size),
        )
        
        self.autophase_emb_dim = autophase_emb_dim
    
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        autophase: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: [batch_size, seq_len, encoder_hidden_size]
            autophase: [batch_size, autophase_dim]
        
        Returns:
            fused_states: [batch_size, seq_len, decoder_hidden_size]
        """
        if encoder_hidden_states.dim() != 3:
            raise ValueError(
                f"Expected 3D encoder_hidden_states [batch_size, seq_len, hidden_size], "
                f"got shape {encoder_hidden_states.shape}"
            )
        if autophase.dim() != 2:
            raise ValueError(
                f"Expected 2D autophase [batch_size, autophase_dim], got shape {autophase.shape}"
            )
        
        batch_size, seq_len, encoder_hidden_size = encoder_hidden_states.shape
        
        # Project autophase to embedding dimension: [batch_size, autophase_emb_dim]
        autophase_emb = self.autophase_proj(autophase)  # [batch_size, autophase_emb_dim]
        
        # Broadcast autophase_emb to match sequence length: [batch_size, seq_len, autophase_emb_dim]
        autophase_emb = autophase_emb.unsqueeze(1).expand(batch_size, seq_len, self.autophase_emb_dim)
        
        # Concat encoder hidden states and autophase embedding
        # [batch_size, seq_len, encoder_hidden_size + autophase_emb_dim]
        concat_states = torch.cat([encoder_hidden_states, autophase_emb], dim=-1)
        
        # Project concat features to decoder hidden size
        # [batch_size, seq_len, decoder_hidden_size]
        fused_states = self.concat_proj(concat_states)
        
        return fused_states