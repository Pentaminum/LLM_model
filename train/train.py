import torch
import torch.nn as nn
import torch.optim as optim
import json
from model.transformer import GPTModel
from tokenizer.BPE_tokenizer import BPETokenizer
import os

# ----- hyperparameter -----
BATCH_SIZE = 16
SEQ_LEN = 128
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 512
NUM_LAYERS = 8
EPOCHS = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- data load -----
with open("data/tokens.json", "r") as f:
    tokens = json.load(f)

tokenizer = BPETokenizer.load("data/tokenizer.json")
vocab_size = tokenizer.vocab_size

# tokens to tensor
data = torch.tensor(tokens, dtype=torch.long)

# ----- batch creation function -----
def get_batch(data, batch_size, seq_len):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ----- model -----
model = GPTModel(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_hidden_dim=FF_DIM,
    num_layers=NUM_LAYERS,
    max_seq_len=SEQ_LEN
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ----- train loop -----
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for step in range(50): 
        x, y = get_batch(data, BATCH_SIZE, SEQ_LEN)  # x: [B, L], y: [B, L]
        logits = model(x)  # [B, L, vocab_size]

        # CrossEntropy는 [B * L, V] vs [B * L]
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / 100
    print(f"Epoch {epoch + 1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/model.pth")
print("Model saved: checkpoints/model.pth")