"""Test model architectures for phase-viz testing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleNet(nn.Module):
    """Minimal model for basic testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ConfigurableNet(nn.Module):
    """Model with configurable architecture."""
    def __init__(self, input_dim=10, hidden_size=64, num_layers=3, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TransformerModel(nn.Module):
    """Transformer-based model for testing."""
    def __init__(self, vocab_size=1000, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, max_seq_length=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        output = self.transformer(src, src_mask)
        output = self.fc_out(output)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ConvNet(nn.Module):
    """Convolutional neural network for testing."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Assuming input size of 32x32
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class RNNModel(nn.Module):
    """Recurrent neural network for testing."""
    def __init__(self, input_size=100, hidden_size=256, num_layers=2, 
                 output_size=10, rnn_type='LSTM'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                             batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                             batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        # Take the last output
        out = out[:, -1, :]
        out = self.fc(out)
        return out
class CustomConfigModel(nn.Module):
    """Model that requires a config dict for initialization."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.fc1 = nn.Linear(config['input_dim'], config['hidden_dim'])
        self.fc2 = nn.Linear(config['hidden_dim'], config['output_dim'])
        
        if config.get('use_dropout', False):
            self.dropout = nn.Dropout(config.get('dropout_rate', 0.5))
        else:
            self.dropout = None

class NoParamsModel(nn.Module):
    """Model with no parameters for edge case testing."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class DynamicModel(nn.Module):
    """Model with dynamic architecture based on input."""
    def __init__(self, base_dim=64):
        super().__init__()
        self.base_dim = base_dim
        self.input_proj = nn.Linear(10, base_dim)
        
        # Create multiple paths
        self.path1 = nn.Sequential(
            nn.Linear(base_dim, base_dim * 2),
            nn.ReLU(),
            nn.Linear(base_dim * 2, base_dim)
        )
        
        self.path2 = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.ReLU(),
            nn.Linear(base_dim // 2, base_dim)
        )
        
        self.output = nn.Linear(base_dim, 1)
    
    def forward(self, x, use_path1=True):
        x = self.input_proj(x)
        
        if use_path1:
            x = self.path1(x)
        else:
            x = self.path2(x)
        
        return self.output(x)