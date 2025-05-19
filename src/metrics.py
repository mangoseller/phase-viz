import os
from typing import Callable, Sequence, List, Dict
import torch
import matplotlib.pyplot as plt
import importlib.util
import inspect
import sys
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


def import_metric_functions(file_path: str) -> Dict[str, Callable[[torch.nn.Module], float]]:
    """Import all metric functions from a Python file.
    
    A valid metric function must:
    1. End with '_of_model'
    2. Take exactly one argument (the model)
    3. Return a float
    
    Returns:
        A dictionary mapping metric names to metric functions
    """
    spec = importlib.util.spec_from_file_location("custom_metric", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {file_path}")
        
    mod = importlib.util.module_from_spec(spec)
    sys.modules["custom_metric"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise ImportError(f"Failed importing {file_path}: {e}")
    
    # Find all functions that end with _of_model
    metric_fns = {}
    for name, fn in mod.__dict__.items():
        if (
            inspect.isfunction(fn) 
            and fn.__module__ == mod.__name__
            and name.endswith("_of_model")
        ):
            # Check that the function takes exactly one argument
            sig = inspect.signature(fn)
            if len(sig.parameters) != 1:
                print(f"Warning: Function {name} does not take exactly one argument. Skipping.")
                continue
                
            # Format the metric name for display
            metric_name = name.replace("_of_model", "").replace("_", " ").title()
            metric_fns[metric_name] = fn
            
    return metric_fns


# def plot_metric_over_checkpoints( # TODO: make this better
#     checkpoint_names: Sequence[str], values: Sequence[float], metric_name: str
# ) -> None:
#     """Simple line-plot helper (unchanged except arg order)."""
#     plt.figure(figsize=(10, 5))
#     plt.plot(checkpoint_names, values, marker="o", linestyle="-")
#     plt.xticks(rotation=45, ha="right")
#     plt.xlabel("Checkpoint")
#     plt.ylabel(metric_name)
#     plt.title(f"{metric_name} over Checkpoints")
#     plt.tight_layout()
#     plt.grid(True)
#     plt.show()

__all__ = ["plot_metric_interactive"]






