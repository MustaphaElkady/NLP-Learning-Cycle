
from data.smalldataset.load_imdb import (
    load_imdb_small,
    build_vocab,
    encode_and_pad
)

from models.rnn import RNNClassifier


def run_experiment():
    MAX_LEN = 50
    EPOCHS = 5
    LR = 0.001

    texts, labels = load_imdb_small(max_samples=2000, max_length=MAX_LEN)
    vocab = build_vocab(texts)
    X = encode_and_pad(texts, vocab, MAX_LEN)

    model = RNNClassifier(vocab_size=len(vocab))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        outputs = model(X).squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == labels).mean()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")


if __name__ == "__main__":
    run_experiment()
