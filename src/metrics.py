import torch
from loader import load_model_from_checkpoint
import matplotlib.pyplot as plt

def l2_norm_of_model(model: torch.nn.Module) -> float:
    """Compute the L2 norm of all parameters in the model."""
    return torch.sqrt(sum(p.data.norm(2).pow(2) for p in model.parameters() if p.requires_grad)).item()


def compute_l2_from_checkpoint(path: str, device="cpu") -> float:
    """Compute L2 norm of parameters from checkpoint."""
    model = load_model_from_checkpoint(path, device)
    return l2_norm_of_model(model)

def plot_metric_over_checkpoints(checkpoint_names: list[str], values: list[float], metric_name="L2 Norm"):
    """
    Plots a line chart of a metric (e.g. L2 norm) over checkpoint steps.

    Args:
        checkpoint_names: List of checkpoint filenames (used as x-axis labels)
        values: List of metric values (same length as checkpoint_names)
        metric_name: Label for the Y-axis and plot title
    """
    plt.figure(figsize=(10, 5))
    plt.plot(checkpoint_names, values, marker='o', linestyle='-')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Checkpoint")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Over Checkpoints")
    plt.tight_layout()
    plt.grid(True)
    plt.show()
