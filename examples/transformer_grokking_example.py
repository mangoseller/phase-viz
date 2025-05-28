import torch
import torch.nn as nn
import numpy as np

class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))
    
    def forward(self, x):
        return torch.einsum('dbp -> bpd', self.W_E[:, x])

class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_vocab))
    
    def forward(self, x):
        return x @ self.W_U

class PosEmbed(nn.Module):
    def __init__(self, n_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d_model) / np.sqrt(d_model))
    
    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads) / np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type='ReLU'):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type='ReLU'):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, n_ctx)
        self.mlp = MLP(d_model, d_mlp, act_type)
    
    def forward(self, x):
        return x  

class Transformer(nn.Module):
    def __init__(self, d_vocab=114, d_model=128, n_ctx=3, d_mlp=512, 
                 num_heads=4, num_layers=1, act_type='ReLU'):
        """
        Initialize the Transformer model.
        
        Args:
            d_vocab: Vocabulary size (default: 114)
            d_model: Model dimension (default: 128) 
            n_ctx: Context length (default: 3)
            d_mlp: MLP hidden dimension (default: 512)
            num_heads: Number of attention heads (default: 4)
            num_layers: Number of transformer blocks (default: 1)
            act_type: Activation type, 'ReLU' or 'GeLU' (default: 'ReLU')
        """
        super().__init__()
        

        d_head = d_model // num_heads
        

        self.embed = Embed(d_vocab=d_vocab, d_model=d_model)
        self.pos_embed = PosEmbed(n_ctx=n_ctx, d_model=d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                d_mlp=d_mlp,
                d_head=d_head,
                num_heads=num_heads,
                n_ctx=n_ctx,
                act_type=act_type
            ) for _ in range(num_layers)
        ])
        self.unembed = Unembed(d_vocab=d_vocab, d_model=d_model)
        
    def forward(self, x):

        return x
# alias for CLI calling
GrokkingMLP = Transformer