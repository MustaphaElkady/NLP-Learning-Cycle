import pandas as pd
import torch
from collections import Counter

def load_imdb_csv(csv_path, max_samples = 2000, max_length = 50):
    df = pd.read_csv(csv_path)

    texts = []
    labels = []

    for i, row in df.iterrows():
        if i >= max_samples:
            break

        tokens = row['review'].lower().split()[:max_length]
        texts.append(tokens)

        labels.append(1 if row['sentiment'] == "pos" else 0)

    return texts, torch.tensor(labels, dtype=torch.float)

def build_vocab(token_lists):
    counter = Counter()

    for tokens in token_lists:
        counter.update(tokens)

        vocab = {word: 1+i for i, word in enumerate(counter.keys())}
        vocab["<PAD>"] = 0
    return vocab

def encode_and_pad(texts, vocab, max_length):
    encoded =[]
    for tokens in texts:
        seq = [vocab.get(tok,0) for tok in tokens ]
        seq += [0] * (max_length - len(seq))
        encoded.append(seq)


    return torch.tensor(encoded, dtype=torch.long)

