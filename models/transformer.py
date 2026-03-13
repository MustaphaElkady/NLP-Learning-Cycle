import torch.nn as nn

class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=64,
        n_heads=4,
        n_layers=2
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out = self.encoder(x)
        out = out[:, -1, :]     # last token
        return self.fc(out)
