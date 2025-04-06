import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head self-attention with residual connection and layer norm
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))  # residual + norm

        # Feed-forward network with residual connection and layer norm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))  # residual + norm

        return x
