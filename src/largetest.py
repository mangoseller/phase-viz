import torch
import torch.nn as nn
import torch.nn.functional as F

class LargeNet(nn.Module):
    """
    A significantly larger neural network for testing progress bars.
    This model has many more parameters than ArithmeticNet.
    """
    def __init__(self, hidden_size=512, num_blocks=4, input_dim=10):
        super().__init__()
        
        # Store configuration parameters
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.input_dim = input_dim
        
        # Input processing
        self.input_proj = nn.Linear(input_dim, hidden_size)
        
        # Create a deep stack of residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            self.blocks.append(block)
        
        # Some convolutional layers to add even more parameters
        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
        )
        
        # Parallel attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=8, 
            batch_first=True
        )
        
        # Final output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.output = nn.Linear(hidden_size // 4, 1)
        
    def forward(self, x):
        # Assume x is of shape [batch_size, input_dim]
        x = self.input_proj(x)  # [batch_size, hidden_size]
        
        # Apply residual blocks
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
        
        # Apply convolutional layers (need to reshape for 1D convolution)
        conv_input = x.unsqueeze(-1)  # [batch_size, hidden_size, 1]
        conv_input = conv_input.transpose(1, 2)  # [batch_size, 1, hidden_size]
        conv_output = self.conv_layers(conv_input)
        conv_output = conv_output.transpose(1, 2).squeeze(-1)  # [batch_size, hidden_size]
        
        # Mix with original representation
        x = x + conv_output
        
        # Apply self-attention
        attn_output, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = x + attn_output.squeeze(1)  # Add attention output as residual
        
        # Final processing
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)
