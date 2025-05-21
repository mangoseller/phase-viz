import os
import time
import logging
from typing import Callable, Sequence, List, Dict, Optional, Any, Union
import torch
import importlib.util
import inspect
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from loader import load_model_from_checkpoint, clear_model_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase-viz")

# Global metric cache for memoization
_metric_cache = {}

def with_memory_optimization(func):
    """Decorator to optimize memory usage for metric functions."""
    def wrapper(model, *args, **kwargs):
        try:
            # Run metric function
            result = func(model, *args, **kwargs)
            
            # Force CUDA synchronization if model is on GPU
            if hasattr(model, 'parameters'):
                first_param = next(model.parameters(), None)
                if first_param is not None and first_param.is_cuda:
                    torch.cuda.synchronize()
            
            return result
        except Exception as e:
            logger.exception(f"Error calculating metric: {e}")
            raise
        finally:
            # Clean up cache explicitly
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return wrapper

@with_memory_optimization
def l2_norm_of_model(model: torch.nn.Module) -> float:
    """Compute the L2 norm of all *trainable* parameters in a model."""
    # Cache key based on the model's state_dict hash
    cache_key = f"l2_norm_{hash(tuple(p.sum().item() for p in model.parameters() if p.requires_grad))}"
    if cache_key in _metric_cache:
        return _metric_cache[cache_key]
    
    # Optimize computation on GPU if available
    device = next(model.parameters()).device
    with torch.no_grad():  # Prevent tracking history
        squared_sum = torch.tensor(0.0, device=device)
        for p in model.parameters():
            if p.requires_grad:
                squared_sum += p.norm(2).pow(2)
        result = torch.sqrt(squared_sum).item()
    
    # Cache the result
    _metric_cache[cache_key] = result
    return result


def compute_metric_batch(
    metric_functions: Dict[str, Callable[[torch.nn.Module], float]],
    checkpoint_path: str,
    device: str = "auto",
) -> Dict[str, float]:
    """Compute multiple metrics for a single model checkpoint.
    
    This is more efficient than loading the model multiple times.
    
    Args:
        metric_functions: Dict mapping metric names to metric functions
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Dict mapping metric names to metric values
    """
    # Load the model only once
    model = load_model_from_checkpoint(checkpoint_path, device)
    
    # Calculate all metrics
    results = {}
    for name, func in metric_functions.items():
        try:
            results[name] = func(model)
        except Exception as e:
            logger.exception(f"Error calculating {name}: {e}")
            results[name] = float('nan')
    
    # No need to delete model explicitly - Python's garbage collection will handle it
    return results


def compute_metrics_over_checkpoints(
    metric_functions: Dict[str, Callable[[torch.nn.Module], float]],
    checkpoints: Sequence[str],
    device: str = "auto",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    parallel: bool = False
) -> Dict[str, List[float]]:
    """Compute multiple metrics over multiple checkpoints efficiently.
    
    Args:
        metric_functions: Dict mapping metric names to metric functions
        checkpoints: List of checkpoint paths
        device: Device to load models on ('auto', 'cpu', 'cuda')
        progress_callback: Callback for progress updates
        parallel: If True, uses parallel processing for checkpoints
        
    Returns:
        Dict mapping metric names to lists of values for each checkpoint
    """
    # Auto-detect device if set to auto
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize result structure
    results = {name: [None] * len(checkpoints) for name in metric_functions}
    
    # Track metrics progress
    metrics_progress = {
        name: {"completed": 0, "total": len(checkpoints)}
        for name in metric_functions
    }
    
    # Define the processing function
    def process_checkpoint(idx, path):
        try:
            # Get the checkpoint's base name for logging
            checkpoint_name = os.path.basename(path)
            
            # Compute all metrics for this checkpoint
            checkpoint_results = compute_metric_batch(metric_functions, path, device)
            
            # Store results and update progress info
            for name, value in checkpoint_results.items():
                results[name][idx] = value
                metrics_progress[name]["completed"] += 1
            
            # Report progress
            if progress_callback:
                progress_info = {
                    "current": idx + 1,
                    "total": len(checkpoints),
                    "checkpoint": checkpoint_name,
                    "metrics": checkpoint_results,
                    "metrics_progress": metrics_progress
                }
                progress_callback(progress_info)
            
            return idx, checkpoint_results
        except Exception as e:
            logger.exception(f"Error processing checkpoint {path}: {e}")
            raise
    
    # Process checkpoints either in parallel or sequentially
    if parallel and len(checkpoints) > 1:
        # Determine appropriate number of workers
        max_workers = min(len(checkpoints), os.cpu_count() or 4)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_checkpoint, i, path) 
                for i, path in enumerate(checkpoints)
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Error in parallel processing: {e}")
    else:
        # Sequential processing - better for displaying progress
        for i, path in enumerate(checkpoints):
            try:
                process_checkpoint(i, path)
            except Exception as e:
                logger.exception(f"Error processing checkpoint {path}: {e}")
                # Continue with other checkpoints even if one fails
    
    # Clean up the model cache to free memory
    clear_model_cache()
    
    return results


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
                logger.warning(f"Function {name} does not take exactly one argument. Skipping.")
                continue
                
            # Format the metric name for display
            metric_name = name.replace("_of_model", "").replace("_", " ").title()
            
            # Apply memory optimization decorator
            metric_fns[metric_name] = with_memory_optimization(fn)
            
    return metric_fns


def clear_metric_cache():
    """Clear the metric cache to free memory."""
    global _metric_cache
    _metric_cache = {}


__all__ = ["compute_metrics_over_checkpoints", "l2_norm_of_model", 
           "import_metric_functions", "clear_metric_cache"]