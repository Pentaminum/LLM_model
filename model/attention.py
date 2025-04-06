import torch
import torch.nn.functional as F
import math
import torch.nn as nn

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: shape [batch_size, num_heads, seq_len, head_dim]
    mask: optional mask (e.g., for causal or padding)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B, H, L, L]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)  # [B, H, L, L]
    output = torch.matmul(attention_weights, V)    # [B, H, L, D]

    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projection layers
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, L, D = x.shape  # batch, seq_len, embed_dim
        H = self.num_heads
        D_h = self.head_dim

        # Project inputs to Q, K, V
        Q = self.W_q(x)  # [B, L, D]
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head: [B, L, D] -> [B, H, L, D_h]
        Q = Q.view(B, L, H, D_h).transpose(1, 2)  # [B, H, L, D_h]
        K = K.view(B, L, H, D_h).transpose(1, 2)
        V = V.view(B, L, H, D_h).transpose(1, 2)

        # Apply scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads: [B, H, L, D_h] -> [B, L, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)

        # Final linear projection
        output = self.W_o(attn_output)

        return output  # optionally, return attn_weights too