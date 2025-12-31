import torch
import torch.nn as nn
import torch.optim as optim

from data.smalldataset.load_imdb import (
    load_imdb_csv,
    build_vocab,
    encode_and_pad
)

from models.lstm import LSTMClassifier


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)

    # ---------- Configuration ----------
    CSV_PATH = "data/imdb/IMDB Dataset.csv"
    MAX_SAMPLES = 2000
    MAX_LENGTH = 1000   # نفس الضغط اللي طبقناه على RNN
    EPOCHS = 10
    LR = 0.001

    # ---------- Load Data ----------
    texts, labels = load_imdb_csv(
        csv_path=CSV_PATH,
        max_samples=MAX_SAMPLES,
        max_length=MAX_LENGTH
    )

    vocab = build_vocab(texts)
    X = encode_and_pad(texts, vocab, MAX_LENGTH)

    X = X.to(device)
    labels = labels.to(device)

    # ---------- Model ----------
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=64,
        hidden_dim=32   # نفس hidden_dim
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---------- Training ----------
    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        outputs = model(X).squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == labels).float().mean()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {acc.item():.4f}"
        )


if __name__ == "__main__":
    run_experiment()