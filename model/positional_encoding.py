import torch
import math

def get_positional_encoding(seq_len: int, dim: int) -> torch.Tensor:
    """
    Returns a [seq_len x dim] tensor of sinusoidal positional encodings.
    """
    pe = torch.zeros(seq_len, dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

    return pe  # shape: [seq_len, dim]

