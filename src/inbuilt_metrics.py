import torch
import torch.nn as nn
import numpy as np
import scipy
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
    from compatible_wrapper import get_weight_matrices_for_metrics
    
    matrices = get_weight_matrices_for_metrics(model)
    if not matrices:
        return 0.0
    
    connectivities = []
    for w in matrices:
        connectivities.append(w.abs().mean().item())
    
    return sum(connectivities) / len(connectivities) if connectivities else 0.0


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
    all_norms = []
    
    for name, param in model.named_parameters():
        if param.dim() >= 2 and any(ind in name.upper() for ind in ['W', 'WEIGHT', 'KERNEL', 'EMBED']):
            norm = torch.norm(param, p='fro').item()
            all_norms.append((name, norm))
    
    if len(all_norms) >= 2:
        # Use first and last in the parameter list
        first_norm = all_norms[0][1]
        last_norm = all_norms[-1][1]
        return first_norm / last_norm if last_norm > 1e-10 else 0.0
    
    return 1.0  # Default ratio if not enough layers


def activation_capacity_of_model(model: torch.nn.Module) -> float:
    """
    Estimate the model's activation capacity based on weight properties.
    
    This is a proxy for the model's representational capacity.
    """
    from compatible_wrapper import get_weight_matrices_for_metrics
    
    matrices = get_weight_matrices_for_metrics(model)
    if not matrices:
        return 0.0
    
    # Instead of requiring square matrices and using determinants,
    # use a more general measure of capacity
    capacity = 0.0
    
    for w in matrices:
        if w.shape[0] > 0 and w.shape[1] > 0:
            # Use the log of the product of the shape dimensions
            # as a measure of the "degrees of freedom"
            dim_capacity = torch.log(torch.tensor(w.shape[0] * w.shape[1], dtype=torch.float32))
            
            # Weight by the "spread" of the singular values
            try:
                svd_vals = torch.linalg.svdvals(w)
                if len(svd_vals) > 0 and svd_vals[0] > 0:
                    # Use entropy of normalized singular values
                    svd_normalized = svd_vals / svd_vals.sum()
                    entropy = -(svd_normalized * torch.log(svd_normalized + 1e-10)).sum()
                    
                    # Higher entropy means more uniform singular values = higher capacity
                    spread_factor = torch.exp(entropy) / len(svd_vals)
                else:
                    spread_factor = 0.1
            except:
                spread_factor = 0.1
            
            capacity += (dim_capacity * spread_factor).item()
    
    # Normalize by number of matrices to get average capacity
    return capacity / len(matrices) if matrices else 0.0


def dead_neuron_percentage_of_model(model: torch.nn.Module) -> float:
    """
    Estimate percentage of "dead" neurons (near-zero weights).
    
    This can indicate over-regularization or training issues.
    """
    from compatible_wrapper import get_weight_matrices_for_metrics
    
    # Use a more reasonable threshold for "dead" neurons
    threshold = 1e-3  # Increased from 1e-5
    matrices = get_weight_matrices_for_metrics(model)
    if not matrices:
        return 0.0
    
    total_neurons = 0
    dead_neurons = 0
    
    for w in matrices:
        # Only consider matrices that look like they have "neurons" (rows)
        # Skip embedding-like matrices
        if w.shape[0] > 0:
            # Check each row (neuron) - use L2 norm instead of L1
            row_norms = w.norm(p=2, dim=1)
            total_neurons += row_norms.shape[0]
            dead_neurons += (row_norms < threshold).sum().item()
    
    return (dead_neurons / total_neurons * 100) if total_neurons > 0 else 0.0



def weight_rank_of_model(model: torch.nn.Module) -> float:
    """
    Compute average effective rank of weight matrices.
    
    Lower rank might indicate redundancy in the model.
    """
    from compatible_wrapper import get_weight_matrices_for_metrics
    
    matrices = get_weight_matrices_for_metrics(model)
    if not matrices:
        return 0.0
    
    ranks = []
    for w in matrices:
        if w.shape[0] > 0 and w.shape[1] > 0:
            try:
                # Use a more robust rank calculation with tolerance
                # Default tolerance in matrix_rank might be too strict
                rank = torch.linalg.matrix_rank(w, tol=1e-4).item()
                ranks.append(rank)
            except Exception as e:
                # If rank calculation fails, estimate it
                try:
                    # Alternative: count singular values above threshold
                    svd_vals = torch.linalg.svdvals(w)
                    rank = (svd_vals > 1e-4).sum().item()
                    ranks.append(rank)
                except:
                    # Last resort: use minimum dimension
                    ranks.append(min(w.shape))
    
    return sum(ranks) / len(ranks) if ranks else 0.0


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
    from compatible_wrapper import get_weight_matrices_for_metrics
    
    matrices = get_weight_matrices_for_metrics(model)
    if not matrices:
        return 0.0
    
    ranks = []
    for w in matrices:
        if w.shape[0] > 0 and w.shape[1] > 0:
            try:
                sv = torch.linalg.svdvals(w)
                # Normalize singular values
                sv_normalized = sv / sv.sum()
                # Calculate entropy
                entropy = -(sv_normalized * torch.log(sv_normalized + 1e-10)).sum()
                # Convert to effective rank
                eff_rank = torch.exp(entropy).item()
                ranks.append(eff_rank)
            except:
                continue
    
    return sum(ranks) / len(ranks) if ranks else 0.0


def avg_condition_number_of_model(model: torch.nn.Module) -> float:
    """
    Compute the average condition number of weight matrices.
    
    High condition numbers indicate potential numerical instability.
    """
    from compatible_wrapper import get_weight_matrices_for_metrics
    
    matrices = get_weight_matrices_for_metrics(model)
    if not matrices:
        return 0.0
    
    condition_numbers = []
    for w in matrices:
        if w.shape[0] > 0 and w.shape[1] > 0:
            try:
                cond = torch.linalg.cond(w).item()
                if not torch.isnan(torch.tensor(cond)) and not torch.isinf(torch.tensor(cond)):
                    condition_numbers.append(cond)
            except:
                continue
    
    return sum(condition_numbers) / len(condition_numbers) if condition_numbers else 0.0


def flatness_proxy_of_model(model: torch.nn.Module) -> float:
    """
    Compute a proxy for loss landscape flatness.
    
    Product of Frobenius and spectral norms.
    """
    from compatible_wrapper import get_weight_matrices_for_metrics
    
    matrices = get_weight_matrices_for_metrics(model)
    if not matrices:
        return 0.0
    
    # Use ratio of Frobenius norm to spectral norm as proxy
    total_frob = 0.0
    total_spec = 0.0
    
    for w in matrices:
        if w.shape[0] > 0 and w.shape[1] > 0:
            total_frob += torch.norm(w, p='fro').item() ** 2
            try:
                sv = torch.linalg.svdvals(w)
                if len(sv) > 0:
                    total_spec += sv[0].item() ** 2
            except:
                continue
    
    return (total_frob / total_spec) if total_spec > 0 else 0.0


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
    from compatible_wrapper import get_weight_matrices_for_metrics
    
    matrices = get_weight_matrices_for_metrics(model)
    if not matrices:
        return 0.0
    
    isotropies = []
    for w in matrices:
        if w.shape[0] > 0 and w.shape[1] > 0:
            try:
                # Calculate covariance matrix
                w_centered = w - w.mean(dim=0, keepdim=True)
                cov = torch.mm(w_centered.T, w_centered) / (w.shape[0] - 1)
                
                # Calculate eigenvalues
                eigenvalues = torch.linalg.eigvalsh(cov)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter small values
                
                if len(eigenvalues) > 1:
                    # Isotropy as ratio of smallest to largest eigenvalue
                    isotropy = (eigenvalues.min() / eigenvalues.max()).item()
                    isotropies.append(isotropy)
            except:
                continue
    
    return sum(isotropies) / len(isotropies) if isotropies else 0.0


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
    
    """Calculate maximum singular value across all weight matrices."""
    from compatible_wrapper import get_weight_matrices_for_metrics
    
    max_sv = 0.0
    matrices = get_weight_matrices_for_metrics(model)
    
    for w in matrices:
        if w.shape[0] > 0 and w.shape[1] > 0:
            try:
                sv = torch.linalg.svdvals(w)
                if len(sv) > 0:
                    max_sv = max(max_sv, sv[0].item())
            except:
                continue
    
    return max_sv


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
    from compatible_wrapper import get_weight_matrices_for_metrics
    
    matrices = get_weight_matrices_for_metrics(model)
    if not matrices:
        return 0.0
    
    max_acts = []
    for w in matrices:
        # Maximum activation is the maximum absolute row sum
        max_act = w.abs().sum(dim=1).max().item()
        max_acts.append(max_act)
    
    return max(max_acts) if max_acts else 0.0