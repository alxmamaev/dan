import torch
from torch import nn
from torch import nn
from torch.distributions.bernoulli import Bernoulli

class DAN(nn.Module):
    def __init__(self, vocab_size, vector_size, output_size, dropout=0.3):
        super().__init__()
        self.bernoulli = Bernoulli(dropout)
        self.emb = nn.Embedding(vocab_size, vector_size)
        self.fc = nn.Linear(vector_size, output_size)

    def forward(self, X):
        mask = self.bernoulli.sample(X.shape).float().unsqueeze(2)

        out = self.emb(X)
        out = out * mask
        out = out.sum(dim=1)
        out = out/(mask.sum(dim=1))

        out = self.fc(out)
        return out
