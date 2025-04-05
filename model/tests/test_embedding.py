import torch
import json

from model.embedding import TokenEmbedding
from model.positional_encoding import get_positional_encoding
from tokenizer.tokenizer import CharTokenizer

# hyperparameter
seq_len = 32
embed_dim = 64

# 1. load tokens from tokens.json
with open("data/tokens.json", "r") as f:
    tokens = json.load(f)

# 2. input sequence for testing (first 32)
input_ids = tokens[:seq_len]
input_tensor = torch.tensor(input_ids).unsqueeze(0)  # shape: [1, seq_len]

# 3. load tokenizer for vocab_size용
tokenizer = CharTokenizer.load("data/tokenizer.json")
vocab_size = tokenizer.vocab_size

# 4. embedding layer
embedding_layer = TokenEmbedding(vocab_size, embed_dim)
token_embeddings = embedding_layer(input_tensor)  # shape: [1, seq_len, dim]

# 5. positional_encoding creation
positional_encoding = get_positional_encoding(seq_len, embed_dim)  # shape: [seq_len, dim]
positional_encoding = positional_encoding.unsqueeze(0)  # shape: [1, seq_len, dim]

# 6. add them
final_input = token_embeddings + positional_encoding  # shape: [1, seq_len, dim]

# 7. check
print(f"token_embeddings.shape: {token_embeddings.shape}")
print(f"positional_encoding.shape: {positional_encoding.shape}")
print(f"final_input.shape: {final_input.shape}")
