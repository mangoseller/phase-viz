import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_size=48):
        super().__init__()
        self.embedding = nn.Embedding(53, 12)
        self.linear1r = nn.Linear(12, hidden_size, bias=True)
        self.linear1l = nn.Linear(12, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, 53, bias=False)
        self.act = nn.GELU()
        self.vocab_size = 53

    def forward(self, x):
        x1 = self.embedding(x[..., 0])
        x2 = self.embedding(x[..., 1])
        x1 = self.linear1l(x1)
        x2 = self.linear1r(x2)
        x = x1 + x2
        x = self.act(x)
        x = self.linear2(x)
        return x