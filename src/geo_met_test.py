import torch
from scipy.stats import skew, kurtosis
import numpy as np
def effective_rank_of_model(model: torch.nn.Module) -> float:
    """
    Compute the average effective rank of all weight matrices using entropy of singular values.
    """
    ranks = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data
            if w.numel() == 0: continue
            try:
                s = torch.linalg.svdvals(w)
            except:
                s = torch.svd(w, compute_uv=False).S
            s = s[s > 1e-8]  # Avoid log(0)
            p = s / s.sum()
            entropy = -(p * torch.log(p)).sum()
            ranks.append(torch.exp(entropy).item())  # Effective rank
    return np.mean(ranks) if ranks else 0.0

def avg_condition_number_of_model(model: torch.nn.Module) -> float:
    condition_numbers = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            W = module.weight.data
            try:
                s = torch.linalg.svdvals(W)
            except:
                s = torch.svd(W, compute_uv=False).S
            if s.numel() >= 2 and s[-1] > 1e-8:
                condition_numbers.append((s[0] / s[-1]).item())
    return float(np.mean(condition_numbers)) if condition_numbers else 0.0
def flatness_proxy(model: torch.nn.Module) -> float:
    frob = weight_norm_of_model(model)
    spec = spectral_norm_of_model(model)
    return frob * spec
def mean_weight_of_model(model: torch.nn.Module) -> float:
    weights = [p.data.flatten() for p in model.parameters() if p.requires_grad]
    if not weights:
        return 0.0
    flat = torch.cat(weights)
    return flat.mean().item()
from scipy.stats import skew, kurtosis

def weight_skew_kurtosis(model: torch.nn.Module) -> tuple:
    flat = torch.cat([p.data.flatten().cpu().float() for p in model.parameters() if p.requires_grad])
    return float(skew(flat.numpy())), float(kurtosis(flat.numpy()))
def isotropy_of_model(model: torch.nn.Module) -> float:
    isotropies = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            W = module.weight.data
            cov = W @ W.T
            trace = torch.trace(cov)
            frob_sq = torch.norm(cov, 'fro')**2
            isotropies.append((trace ** 2 / frob_sq).item())  # closer to 1 → isotropic
    return float(np.mean(isotropies)) if isotropies else 0.0
