import torch
import torch.nn as nn
import torch.optim as optim

from data.wikitext.load_wikitext import *
from models.lstm import LSTMLanguageModel
import os

def run_experiment(dataset_path, dataset_selected, seq_len, max_tokens, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokens = load_wikitext(dataset_path, max_tokens)
    vocab = build_vocab(tokens)
    encoded = encode(tokens, vocab)
    # X, y = create_sequences(encoded, SEQ_LEN)

    model = LSTMLanguageModel(len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        steps = 0

        for X_seq, y_seq in create_sequences(encoded, seq_len):
            X_seq = X_seq.unsqueeze(0).to(device)   # batch = 1
            y_seq = torch.tensor([y_seq]).to(device)

            optimizer.zero_grad()
            out = model(X_seq)
            loss = criterion(out, y_seq)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / steps:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/lstm_{dataset_selected}.pt")
    torch.save(vocab, f"checkpoints/vocab_lstm_{dataset_selected}.pt")
    return model, vocab