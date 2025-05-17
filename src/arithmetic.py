import torch.nn as nn
import torch
import torch.nn.functional as F

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

    def forward(self, numbers, operator):
        # Embed inputs
        num_features = self.num_embedder(numbers)
        op_features = self.op_embedding(operator)

        # Combine features
        x = torch.cat([num_features, op_features], dim=1)

        # Process with residual connections for better gradient flow
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1)) + x1
        x3 = self.layer3(x2)

        return self.output(x3)