from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from collections import Counter
import torch



def load_imdb_data(max_length=50):
    tokenizer = get_tokenizer("basic_english")

    train_iter, test_iter = IMDB()

    texts = []
    labels = []

    for label, text in train_iter:
        tokens = tokenizer(text)[:max_length]
        texts.append(tokens)
        labels.append(1 if label == 'pos' else 0)

    return texts, labels



def build_vocab(token_lists):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {word: i + 1 for i, word in enumerate(counter.keys())}
    vocab["<PAD>"] = 0
    return vocab



def build_vocab(token_lists):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {word: i + 1 for i, word in enumerate(counter.keys())}
    vocab["<PAD>"] = 0
    return vocab

