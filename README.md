# LLM_model

STEP 1: Tokenization
Write a very simple tokenizer (start with character-level or word-level, skip BPE if needed).

Save token-to-id and id-to-token mappings.

STEP 2: Dataset
Use a tiny subset of a public dataset like WikiText or even just a few paragraphs of classic text 

Preprocess the text into sequences of token IDs.

STEP 3: Embeddings + Positional Encoding
Create token embeddings (embedding matrix: vocab_size x embed_dim).

