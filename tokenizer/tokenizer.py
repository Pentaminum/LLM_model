import json
from typing import List, Dict

class CharTokenizer:
    def __init__(self, text: str = None, vocab: Dict[str, int] = None):
        if vocab:
            self.token2id = vocab
        elif text:
            unique_chars = sorted(set(text))
            self.token2id = {ch: idx for idx, ch in enumerate(unique_chars)}
        else:
            raise ValueError("Must provide either `text` or `vocab`.")

        self.id2token = {idx: ch for ch, idx in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    def encode(self, text: str) -> List[int]:
        return [self.token2id[ch] for ch in text if ch in self.token2id]

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.id2token[token] for token in tokens])

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.token2id, f)

    @staticmethod
    def load(path: str):
        with open(path, 'r') as f:
            vocab = json.load(f)
        return CharTokenizer(vocab=vocab)
