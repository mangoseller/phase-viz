
import torch
import numpy as np

def participation_ratio_of_model(model: torch.nn.Module) -> float:
    """
    Compute the participation ratio of all trainable parameters.
    
    The participation ratio measures how evenly distributed the values are.
    A higher value means more even distribution of weights.
    """
    # flatten parameters into one long vector
    flat = torch.cat(
        [p.data.reshape(-1).float() for p in model.parameters() if p.requires_grad]
    )
    s2 = torch.sum(flat.pow(2))
    s4 = torch.sum(flat.pow(4))
    return (s2.pow(2) / s4).item()

def sparsity_of_model(model: torch.nn.Module) -> float:
    """
    Compute the sparsity of trainable parameters.
    
    Returns the fraction of parameters that are close to zero.
    """
    # Threshold for considering a parameter as "zero"
    threshold = 1e-3
    
    # Get all parameters
    flat = torch.cat(
        [p.data.reshape(-1).float() for p in model.parameters() if p.requires_grad]
    )
    
    # Compute sparsity as fraction of parameters below threshold
    near_zero = torch.sum((torch.abs(flat) < threshold).float())
    return (near_zero / flat.numel()).item()

def spectral_norm_of_model(model: torch.nn.Module) -> float:
    """
    Approximate the spectral norm of the model's linear layers.
    
    Returns the maximum singular value across all weight matrices.
    """
    max_norm = 0.0
    
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.weight is not None:
            # Use torch.linalg.svdvals to get singular values
            try:
                s = torch.linalg.svdvals(module.weight)
                if s.numel() > 0:
                    max_norm = max(max_norm, s[0].item())
            except:
                # Fallback in case of numerical issues
                u, s, v = torch.svd(module.weight, compute_uv=False)
                if s.numel() > 0:
                    max_norm = max(max_norm, s[0].item())
    
    return max_norm

def weight_norm_of_model(model: torch.nn.Module) -> float:
    """
    Compute the Frobenius norm of all trainable parameters.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def max_activation_of_model(model: torch.nn.Module) -> float:
    """
    Estimate the maximum potential activation in the model.
    
    This is a proxy that looks at weight magnitudes.
    """
    max_activation = 0.0
    
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            if hasattr(module, 'weight'):
                max_val = torch.max(torch.abs(module.weight)).item()
                max_activation = max(max_activation, max_val)
    
    return max_activation