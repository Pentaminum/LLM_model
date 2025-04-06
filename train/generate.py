import torch
from model.transformer import GPTModel
from tokenizer.BPE_tokenizer import BPETokenizer

# settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 100
TEMPERATURE = 1.0

# Tokenizer & Model
tokenizer = BPETokenizer.load("data/tokenizer.json")

model = GPTModel(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_heads=4,
    ff_hidden_dim=512,
    num_layers=8,
    max_seq_len=128
).to(DEVICE)

model.load_state_dict(torch.load("checkpoints/model.pth", map_location=DEVICE))
model.eval()

# === chat roop ===
print("GPT mini chatbot start!")
while True:
    user_input = input("\n💬 You: ")

    input_ids = tokenizer.encode(user_input)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    generated = input_tensor
    start_len = generated.size(1)  # save user input size

    for _ in range(MAX_NEW_TOKENS):
        if generated.size(1) > model.pos_encoding.size(0):
            break

        with torch.no_grad():
            logits = model(generated)[:, -1, :]
            logits = logits / TEMPERATURE
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

    # decode only the generated part
    new_tokens = generated[0][start_len:].tolist()
    output_text = tokenizer.decode(new_tokens)

    print("🤖 GPT:", output_text)

