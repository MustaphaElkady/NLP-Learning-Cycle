import torch
import torch.nn as nn
import torch.optim as optim

from data.smalldataset.load_imdb import (
    load_imdb_csv,
    build_vocab,
    encode_and_pad
)
from models.rnn import RNNClassifier


def run_experiment():
    # 1.Experiment Configuration
    CSV_PATH = "data\data\imdb\IMDB Dataset.csv"
    MAX_SAMPLES = 2000
    MAX_LENGTH = 50

    EMBED_DIM = 64
    HIDDEN_DIM = 128
    EPOCHS = 5
    LR = 0.001

    # 2.Load & Prepare Data
    texts, labels = load_imdb_csv(
        csv_path=CSV_PATH,
        max_samples=MAX_SAMPLES,
        max_length=MAX_LENGTH
    )
    vocab = build_vocab(texts)
    X = encode_and_pad(texts, vocab, MAX_LENGTH)

    # 3.Model, Loss, Optimizer

    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

     # 4.Training Loop
    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        outputs = model(X).squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Accuracy
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == labels).float().mean()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {acc.item():.4f}"
        )


if __name__ == "__main__":
    run_experiment()