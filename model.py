import torch
from torch import nn

class DAN(nn.Module):
    def __init__(self, vocab_size, vector_size,
                hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vector_size)

        self.enc = nn.Sequential(nn.LayerNorm(vector_size),
                                 nn.Linear(vector_size, hidden_size),
                                 nn.LayerNorm(hidden_size),
                                 nn.Dropout(p=dropout),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, output_size),
                                 nn.LayerNorm(output_size))

    def forward(self, X):
        out = self.emb(X)
        out = out.mean(dim=1)
        out = self.enc(out)

        return out
