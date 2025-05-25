"""Built-in metrics for phase-viz."""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import skew, kurtosis


def l2_norm_of_model(model: torch.nn.Module) -> float:
    """Compute the L2 norm of all trainable parameters in a model."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if not trainable_params:
        return 0.0
    
    device = trainable_params[0].device
    with torch.no_grad():
        squared_sum = torch.tensor(0.0, device=device)
        for p in trainable_params:
            squared_sum += p.norm(2).pow(2)
        result = torch.sqrt(squared_sum).item()
    
    return result


def weight_entropy_of_model(model: torch.nn.Module) -> float:
    """
    Compute the Shannon entropy of weight distribution.
    
    Higher entropy indicates more uniform weight distribution,
    lower entropy indicates weights concentrated around specific values.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if not trainable_params:
        return 0.0
    
    device = trainable_params[0].device
    with torch.no_grad():
        all_weights = torch.cat([p.data.flatten() for p in trainable_params])
        
        # Create histogram of weight values
        num_bins = 100
        hist, bin_edges = torch.histogram(all_weights, bins=num_bins)
        
        # Convert to probabilities
        hist = hist.float()
        hist = hist / hist.sum()
        
        # Calculate entropy (avoid log(0))
        epsilon = 1e-10
        entropy = -(hist * torch.log(hist + epsilon)).sum().item()
        
        # Normalize by log(num_bins) to get value between 0 and 1
        normalized_entropy = entropy / np.log(num_bins)
    
    return normalized_entropy


def layer_connectivity_of_model(model: torch.nn.Module) -> float:
    """
    Compute average absolute weight per layer.
    
    This metric indicates the average strength of connections in the network.
    Higher values suggest stronger connections between neurons.
    """
    layer_weights = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            if hasattr(module, 'weight') and module.weight is not None:
                with torch.no_grad():
                    avg_weight = module.weight.data.abs().mean().item()
                    layer_weights.append(avg_weight)
    
    if not layer_weights:
        return 0.0
    
    return float(np.mean(layer_weights))


def parameter_variance_of_model(model: torch.nn.Module) -> float:
    """
    Compute the variance of all trainable parameters.
    
    This metric helps track how spread out the parameter values are,
    which can indicate model complexity or training dynamics.
    """
    all_params = []
    for p in model.parameters():
        if p.requires_grad:
            all_params.extend(p.cpu().data.numpy().flatten())
    
    if len(all_params) == 0:
        return 0.0
    
    return float(np.var(all_params))


def layer_wise_norm_ratio_of_model(model: torch.nn.Module) -> float:
    """
    Compute the ratio of norms between first and last layers.
    
    This can help identify gradient vanishing/exploding issues.
    """
    layers_with_params = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            if hasattr(module, 'weight') and module.weight is not None:
                norm = module.weight.data.norm(2).item()
                layers_with_params.append((name, norm))
    
    if len(layers_with_params) < 2:
        return 1.0
    
    first_norm = layers_with_params[0][1]
    last_norm = layers_with_params[-1][1]
    
    # Avoid division by zero
    if first_norm == 0:
        return float('inf') if last_norm > 0 else 1.0
    
    return last_norm / first_norm


def activation_capacity_of_model(model: torch.nn.Module) -> float:
    """
    Estimate the model's activation capacity based on layer dimensions.
    
    This is a proxy for the model's representational capacity.
    """
    total_capacity = 0.0
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # For linear layers, capacity is related to output dimension
            total_capacity += np.log(module.out_features + 1)
        elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
            # For conv layers, consider both channels and kernel size
            kernel_size = (module.kernel_size[0] if isinstance(module.kernel_size, tuple) 
                          else module.kernel_size)
            total_capacity += np.log(module.out_channels * kernel_size + 1)
        elif isinstance(module, nn.MultiheadAttention):
            # For attention layers, consider embedding dimension and heads
            total_capacity += np.log(module.embed_dim * module.num_heads + 1)
    
    return float(total_capacity)


def dead_neuron_percentage_of_model(model: torch.nn.Module) -> float:
    """
    Estimate percentage of "dead" neurons (near-zero weights).
    
    This can indicate over-regularization or training issues.
    """
    threshold = 1e-6
    total_params = 0
    near_zero_params = 0
    
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            if hasattr(module, 'weight') and module.weight is not None:
                weight_data = module.weight.data.abs()
                total_params += weight_data.numel()
                near_zero_params += (weight_data < threshold).sum().item()
    
    if total_params == 0:
        return 0.0
    
    return 100.0 * near_zero_params / total_params


def weight_rank_of_model(model: torch.nn.Module) -> float:
    """
    Compute average effective rank of weight matrices.
    
    Lower rank might indicate redundancy in the model.
    """
    ranks = []
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            if weight.numel() > 0:
                # Compute singular values
                try:
                    singular_values = torch.linalg.svdvals(weight)
                    # Normalize singular values
                    normalized_sv = singular_values / singular_values.sum()
                    # Compute Shannon entropy as proxy for effective rank
                    entropy = -(normalized_sv * torch.log(normalized_sv + 1e-10)).sum()
                    effective_rank = torch.exp(entropy).item()
                    ranks.append(effective_rank)
                except:
                    # Fallback for numerical issues
                    continue
    
    if len(ranks) == 0:
        return 0.0
    
    return float(np.mean(ranks))


def gradient_flow_score_of_model(model: torch.nn.Module) -> float:
    """
    Compute a score representing potential gradient flow quality.
    
    Based on the initialization scale of parameters across layers.
    """
    scores = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) >= 2:
            # Compute the standard deviation
            std = param.data.std().item()
            # Expected std for good gradient flow (Xavier/He initialization)
            fan_in = param.shape[1] if len(param.shape) >= 2 else 1
            expected_std = np.sqrt(2.0 / fan_in)
            
            # Score based on how close we are to expected
            score = 1.0 / (1.0 + abs(std - expected_std))
            scores.append(score)
    
    if len(scores) == 0:
        return 0.0
    
    return float(np.mean(scores))


def effective_rank_of_model(model: torch.nn.Module) -> float:
    """
    Compute the average effective rank of all weight matrices using entropy of singular values.
    """
    ranks = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            if w.numel() == 0:
                continue
            try:
                s = torch.linalg.svdvals(w)
            except:
                s = torch.svd(w, compute_uv=False).S
            s = s[s > 1e-8]  # Avoid log(0)
            if s.numel() > 0:
                p = s / s.sum()
                entropy = -(p * torch.log(p)).sum()
                ranks.append(torch.exp(entropy).item())  # Effective rank
    return float(np.mean(ranks)) if ranks else 0.0


def avg_condition_number_of_model(model: torch.nn.Module) -> float:
    """
    Compute the average condition number of weight matrices.
    
    High condition numbers indicate potential numerical instability.
    """
    condition_numbers = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            try:
                s = torch.linalg.svdvals(W)
            except:
                s = torch.svd(W, compute_uv=False).S
            if s.numel() >= 2 and s[-1] > 1e-8:
                condition_numbers.append((s[0] / s[-1]).item())
    return float(np.mean(condition_numbers)) if condition_numbers else 0.0


def flatness_proxy_of_model(model: torch.nn.Module) -> float:
    """
    Compute a proxy for loss landscape flatness.
    
    Product of Frobenius and spectral norms.
    """
    frob_norm = weight_norm_of_model(model)
    spec_norm = spectral_norm_of_model(model)
    return frob_norm * spec_norm


def mean_weight_of_model(model: torch.nn.Module) -> float:
    """
    Compute the mean of all trainable weights.
    
    Useful for detecting weight drift or bias.
    """
    weights = [p.data.flatten() for p in model.parameters() if p.requires_grad]
    if not weights:
        return 0.0
    flat = torch.cat(weights)
    return flat.mean().item()


def weight_skew_of_model(model: torch.nn.Module) -> float:
    """
    Compute the skewness of weight distribution.
    
    Measures asymmetry of weight distribution.
    """
    flat = torch.cat([p.data.flatten().cpu().float() for p in model.parameters() if p.requires_grad])
    if flat.numel() == 0:
        return 0.0
    return float(skew(flat.numpy()))


def weight_kurtosis_of_model(model: torch.nn.Module) -> float:
    """
    Compute the kurtosis of weight distribution.
    
    Measures the "tailedness" of weight distribution.
    """
    flat = torch.cat([p.data.flatten().cpu().float() for p in model.parameters() if p.requires_grad])
    if flat.numel() == 0:
        return 0.0
    return float(kurtosis(flat.numpy()))


def isotropy_of_model(model: torch.nn.Module) -> float:
    """
    Compute the isotropy of weight matrices.
    
    Values closer to 1 indicate more isotropic (uniform) distributions.
    """
    isotropies = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            if W.numel() > 0:
                cov = W @ W.T
                trace = torch.trace(cov)
                frob_sq = torch.norm(cov, 'fro')**2
                if frob_sq > 0:
                    isotropies.append((trace ** 2 / frob_sq).item())  # closer to 1 → isotropic
    return float(np.mean(isotropies)) if isotropies else 0.0


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


def spectral_norm_of_model(model: torch.nn.Module) -> float:
    """
    Approximate the spectral norm of the model's linear layers.
    
    Returns the maximum singular value across all weight matrices.
    """
    max_norm = 0.0
    
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
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
    if flat.numel() == 0:
        return 0.0
    s2 = torch.sum(flat.pow(2))
    s4 = torch.sum(flat.pow(4))
    if s4 == 0:
        return 0.0
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
    
    if flat.numel() == 0:
        return 0.0
    
    # Compute sparsity as fraction of parameters below threshold
    near_zero = torch.sum((torch.abs(flat) < threshold).float())
    return (near_zero / flat.numel()).item()


def max_activation_of_model(model: torch.nn.Module) -> float:
    """
    Estimate the maximum potential activation in the model.
    
    This is a proxy that looks at weight magnitudes.
    """
    max_activation = 0.0
    
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if hasattr(module, 'weight'):
                max_val = torch.max(torch.abs(module.weight)).item()
                max_activation = max(max_activation, max_val)
    
    return max_activation