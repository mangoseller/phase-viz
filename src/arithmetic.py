# DEMO file
import torch.nn as nn # type: ignore
import torch # type: ignore
import torch.nn.functional as F # type: ignore

class ArithmeticNet(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()

        # Input processing
        self.num_embedder = nn.Linear(2, hidden_size)
        self.op_embedding = nn.Embedding(3, hidden_size)

        # Core processing - adding one more layer but keeping it simple
        self.layer1 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size//2)

        # Single output - no separate abstention head
        self.output = nn.Linear(hidden_size//2, 1)
