import torch
import torch.nn as nn

from model.embedding import TokenEmbedding
from model.positional_encoding import get_positional_encoding
from model.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, ff_hidden_dim: int,
                 num_layers: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)

        # Register positional encoding as buffer (non-learnable parameter)
        pos_encoding = get_positional_encoding(max_seq_len, embed_dim)  # [max_seq_len, embed_dim]
        self.register_buffer("pos_encoding", pos_encoding)  # won't be updated during training

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """
        B, L = x.shape
        assert L <= self.pos_encoding.size(0), "Input sequence length exceeds max_seq_len"

        # Embedding + Positional Encoding
        tok_emb = self.token_embedding(x)                      # [B, L, D]
        pos_emb = self.pos_encoding[:L, :].unsqueeze(0)        # [1, L, D]
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.output_layer(x)                          # [B, L, vocab_size]
        return logits
