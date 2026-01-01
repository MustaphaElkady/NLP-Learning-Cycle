import torch
from collections import Counter

def load_wikitext(path, max_tokens=None):

    with open(path,"r", encoding = "utf-8" ) as f :
        text = f.read()

    # Simple tokenization by whitespace
    tokens = text.split()
    
    if max_tokens:
        tokens = tokens[:max_tokens]
    return tokens 






def build_vocab(tokens):
    counter = Counter()
    vocab = {word: 1+i for i, word in enumerate(counter.keys())}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = len(vocab)
    return vocab



def encode(tokens, vocab):
    encoded =[
        vocab.get(tok,  vocab["<UNK>"] )
            for tok in tokens
    ]
        # for tok in tokens:
        #     if tok in vocab:
        #         encoded.append(vocab[tok])
        #     else:
        #         encoded.append(vocab["<UNK>"])

    return torch.tensor(encoded, dtype=torch.long)

def create_sequences(encoded, seq_len):

    X, y = [], []

    # Sliding window over the encoded tokens
    for i in range(len(encoded) - seq_len):
        X.append(encoded[i : i + seq_len])   # Input sequence
        y.append(encoded[i + seq_len])       # Next token (target)

    # Stack inputs into a single tensor
    return torch.stack(X), torch.tensor(y)
