"""Example custom metrics for phase-viz testing and demonstration."""

import torch
import torch.nn as nn
import numpy as np


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


# Test the metrics if running directly
if __name__ == "__main__":
    from test_models import SimpleNet, TransformerModel
    
    print("Testing custom metrics...")
    
    # Test on SimpleNet
    model = SimpleNet()
    print("\nSimpleNet metrics:")
    print(f"  Parameter variance: {parameter_variance_of_model(model):.6f}")
    print(f"  Layer norm ratio: {layer_wise_norm_ratio_of_model(model):.6f}")
    print(f"  Activation capacity: {activation_capacity_of_model(model):.6f}")
    print(f"  Dead neuron %: {dead_neuron_percentage_of_model(model):.2f}%")
    print(f"  Weight rank: {weight_rank_of_model(model):.6f}")
    print(f"  Gradient flow score: {gradient_flow_score_of_model(model):.6f}")
    
    # Test on TransformerModel
    transformer = TransformerModel(vocab_size=100, d_model=64, nhead=4, num_layers=2)
    print("\nTransformerModel metrics:")
    print(f"  Parameter variance: {parameter_variance_of_model(transformer):.6f}")
    print(f"  Activation capacity: {activation_capacity_of_model(transformer):.6f}")
    print(f"  Dead neuron %: {dead_neuron_percentage_of_model(transformer):.2f}%")