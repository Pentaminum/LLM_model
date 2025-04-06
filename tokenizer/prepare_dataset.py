from datasets import load_dataset
from tokenizer.BPE_tokenizer import BPETokenizer
from tqdm import tqdm
import json
import os

# 1. Load daily_dialog dataset
print("📦 Loading daily_dialog dataset...")
dataset = load_dataset("daily_dialog")

# 2. Flatten dialog lines
print("📄 Extracting dialog lines...")
train_lines = []
for dialog in dataset["train"]:
    train_lines.extend([line.strip() for line in dialog["dialog"] if len(line.strip()) > 0])

print(f"✔️ Total dialog lines extracted: {len(train_lines):,}")

# 3. Write to temp text file (for BPE training)
os.makedirs("data", exist_ok=True)
with open("data/train_text.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_lines))

# 4. Train BPE Tokenizer
print("🔧 Training BPE Tokenizer...")
tokenizer = BPETokenizer.train("data/train_text.txt", vocab_size=30000)

# 5. Encode dataset with progress bar
print("🔠 Encoding text into token IDs...")
tokens = []
with open("data/train_text.txt", "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Encoding"):
        tokens.extend(tokenizer.encode(line.strip()))

# 6. Save tokens + tokenizer
with open("data/tokens.json", "w") as f:
    json.dump(tokens, f)

tokenizer.save("data/tokenizer.json")

print(f"\n✅ Tokenization complete!")
print(f"   Total tokens: {len(tokens):,}")
print(f"   Vocab size  : {tokenizer.vocab_size:,}")
