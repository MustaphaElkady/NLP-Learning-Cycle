import torch.nn as nn


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim = 64, hidden_dim = 128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True) # batch_first(true) => (batch, seq_len, features)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x =  self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)