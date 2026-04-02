import torch
from collections import Counter

def load_wikitext(path, max_tokens=None):

    with open(path,"r", encoding = "utf-8" ) as f :
        text = f.read()

    # Simple tokenization by whitespace
    tokens = preprocess(text) #0010
    
    if max_tokens:
        tokens = tokens[:max_tokens]
    return tokens 



def build_vocab(tokens):
    counter = Counter()
    counter.update(tokens)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for i, word in enumerate(counter.keys()):
        vocab[word] = i + 2

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
    for i in range(len(encoded) - seq_len):
        X_seq = encoded[i : i + seq_len]
        y_seq = encoded[i + seq_len]
        yield X_seq, y_seq

def preprocess(text: str) -> list[str]: #0010
    text = remove_wikitext
