import torch
import torch.nn as nn
import torch.optim as optim

from data.wikitext.load_wikitext import *
from models.rnn import RNNLanguageModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def run_experiment():
    SEQ_LEN = 1000
    MAX_TOKENS = 20000

    tokens = load_wikitext("data/wikitext/wikitext-2/train.txt", MAX_TOKENS)
    vocab = build_vocab(tokens)
    encoded = encode(tokens, vocab)
    X, y = create_sequences(encoded, SEQ_LEN)

    model = RNNLanguageModel(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(5):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} | Loss {loss.item():.4f}")
