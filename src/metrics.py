import os
from typing import Callable, Sequence, List
import torch
import matplotlib.pyplot as plt
from loader import load_model_from_checkpoint

def l2_norm_of_model(model: torch.nn.Module) -> float:
    """Compute the L2 norm of all *trainable* parameters in a model."""
    return torch.sqrt(
        sum(p.data.norm(2).pow(2) for p in model.parameters() if p.requires_grad)
    ).item()


def compute_metric_over_checkpoints( # TODO: import C++ code
    metric_fn: Callable[[torch.nn.Module], float],
    checkpoints: Sequence[str],
    device: str = "cpu",
) -> List[float]:
    """Load each checkpoint, apply *metric_fn(model)*, return the list."""
    values: List[float] = []
    for path in checkpoints:
        model = load_model_from_checkpoint(path, device)
        values.append(metric_fn(model))
    return values


def plot_metric_over_checkpoints( # TODO: make this better
    checkpoint_names: Sequence[str], values: Sequence[float], metric_name: str
) -> None:
    """Simple line-plot helper (unchanged except arg order)."""
    plt.figure(figsize=(10, 5))
    plt.plot(checkpoint_names, values, marker="o", linestyle="-")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Checkpoint")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Checkpoints")
    plt.tight_layout()
    plt.grid(True)
    plt.show()
