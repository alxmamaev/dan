import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli

class DAN(nn.Module):
    def __init__(self, vocab_size, vector_size,
                hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vector_size)

        self.enc = nn.Sequential(nn.BatchNorm1d(vector_size),
                                 nn.Linear(vector_size, hidden_size),
                                 nn.BatchNorm1d(hidden_size),
                                 nn.Dropout(p=dropout),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, output_size),
                                 nn.BatchNorm1d(output_size))

    def forward(self, X):
        out = self.emb(X)
        out = out.mean(dim=1)
        out = self.enc(out)

        return out
