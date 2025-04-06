from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import json

class BPETokenizer:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    @classmethod
    def train(cls, text_file: str, vocab_size: int = 30000):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]"])
        tokenizer.train([text_file], trainer)
        return cls(tokenizer)

    def encode(self, text: str):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def save(self, path: str):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str):
        tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
