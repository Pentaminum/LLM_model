from datasets import load_dataset
from tokenizer import CharTokenizer
import json
import os

# dataset load
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
train_lines = [line for line in dataset["train"]["text"] if len(line.strip()) > 0]

# partial use (5000 lines for speed)
text = "\n".join(train_lines[:5000])

# Tokenizer creation
tokenizer = CharTokenizer(text=text)
tokens = tokenizer.encode(text)

# save
os.makedirs("data", exist_ok=True)

with open("data/tokens.json", "w") as f:
    json.dump(tokens, f)

tokenizer.save("data/tokenizer.json")

print(f"Tokenization complete. Total tokens: {len(tokens)}")
