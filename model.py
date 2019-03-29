import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli

class DAN(nn.Module):
    def __init__(self, vocab_size, vector_size,
                hidden_size, output_size,
                token_drop_rate=0.3, dropout=0.1):
        super().__init__()
        self.bernoulli = Bernoulli(dropout)
        self.emb = nn.Embedding(vocab_size, vector_size)

        self.enc = nn.Sequential(nn.BatchNorm1d(vector_size),
                                 nn.Linear(vector_size, hidden_size),
                                 nn.BatchNorm1d(hidden_size),
                                 nn.Dropout(p=dropout),
                                 nn.Sigmoid(),
                                 nn.Linear(hidden_size, output_size),
                                 nn.BatchNorm1d(output_size))

    def forward(self, X):
        # Masking
        with torch.no_grad():
            if self.training:
                mask = self.bernoulli.sample(X.shape).byte()
                mask = mask.to(X.device)

                X[mask] = 0
                emb_mask = mask.float().unsqueeze(2).to(X.device)
            else:
                emb_mask = torch.ones(X.shape).unsqueeze(2).to(X.device)
            n_tokens = (X != 0).sum(dim=1)

        out = self.emb(X) * emb_mask
        out = out.sum(dim=1) / n_tokens
        out = self.enc(out)

        return out
