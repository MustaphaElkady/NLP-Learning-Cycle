import torch
import torch.nn as nn
import torch.optim as optim
from data.wikitext.load_wikitext import *
from models.rnn import RNNLanguageModel
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def run_experiment(dataset_path, dataset_selected,seq_len, max_tokens, epochs, lr):
    tokens = load_wikitext(dataset_path, max_tokens)
    vocab = build_vocab(tokens)
    encoded = encode(tokens, vocab)
    X, y = create_sequences(encoded, seq_len)
    model = RNNLanguageModel(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} | Loss {loss.item():.4f}")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/rnn_{dataset_selected}.pt")
    torch.save(vocab, f"checkpoints/vocab_{dataset_selected}.pt")
    return model, vocab