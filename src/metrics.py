import os
import time
import tempfile
from typing import Callable, Sequence, List, Dict, Optional, Any, Union
import torch
import importlib.util
import inspect
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from load_models import load_model_from_checkpoint, clear_model_cache
from utils import logger
import typer as t
import functools
import numpy as np

# Import all built-in metrics
from inbuilt_metrics import (
    l2_norm_of_model as _l2_norm,
    weight_entropy_of_model as _weight_entropy,
    layer_connectivity_of_model as _layer_connectivity,
    parameter_variance_of_model as _parameter_variance,
    layer_wise_norm_ratio_of_model as _layer_wise_norm_ratio,
    activation_capacity_of_model as _activation_capacity,
    dead_neuron_percentage_of_model as _dead_neuron_percentage,
    weight_rank_of_model as _weight_rank,
    gradient_flow_score_of_model as _gradient_flow_score,
    effective_rank_of_model as _effective_rank,
    avg_condition_number_of_model as _avg_condition_number,
    flatness_proxy_of_model as _flatness_proxy,
    mean_weight_of_model as _mean_weight,
    weight_skew_of_model as _weight_skew,
    weight_kurtosis_of_model as _weight_kurtosis,
    isotropy_of_model as _isotropy,
    weight_norm_of_model as _weight_norm,
    spectral_norm_of_model as _spectral_norm,
    participation_ratio_of_model as _participation_ratio,
    sparsity_of_model as _sparsity,
    max_activation_of_model as _max_activation
)

# Global metric cache for memoization
_metric_cache = {}

def with_memory_optimization(func):
    """
    Decorator that adds GPU-sync / cache-clear without
    breaking picklability.
    """
    @functools.wraps(func)          # <--- gives the wrapper the *same*
    def wrapper(model, *args, **kw):#      __name__/__qualname__/__module__
        try:
            # Create cache key for memoization
            if hasattr(func, '_cache_key_func'):
                cache_key = func._cache_key_func(model)
                if cache_key in _metric_cache:
                    return _metric_cache[cache_key]
            
            result = func(model, *args, **kw)

            # Sync if first param is on CUDA
            first = next(model.parameters(), None)
            if first is not None and first.is_cuda:
                torch.cuda.synchronize()

            # Cache result if cache key exists
            if hasattr(func, '_cache_key_func'):
                _metric_cache[cache_key] = result
                
            return result
        except Exception as e:
            logger.exception("Error calculating metric: %s", e)
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ensure the wrapper is reachable as module-level attr for pickle
    mod = sys.modules[func.__module__]
    setattr(mod, func.__name__, wrapper)

    return wrapper

# Apply decorator to all imported metrics
l2_norm_of_model = with_memory_optimization(_l2_norm)
weight_entropy_of_model = with_memory_optimization(_weight_entropy)
layer_connectivity_of_model = with_memory_optimization(_layer_connectivity)
parameter_variance_of_model = with_memory_optimization(_parameter_variance)
layer_wise_norm_ratio_of_model = with_memory_optimization(_layer_wise_norm_ratio)
activation_capacity_of_model = with_memory_optimization(_activation_capacity)
dead_neuron_percentage_of_model = with_memory_optimization(_dead_neuron_percentage)
weight_rank_of_model = with_memory_optimization(_weight_rank)
gradient_flow_score_of_model = with_memory_optimization(_gradient_flow_score)
effective_rank_of_model = with_memory_optimization(_effective_rank)
avg_condition_number_of_model = with_memory_optimization(_avg_condition_number)
flatness_proxy_of_model = with_memory_optimization(_flatness_proxy)
mean_weight_of_model = with_memory_optimization(_mean_weight)
weight_skew_of_model = with_memory_optimization(_weight_skew)
weight_kurtosis_of_model = with_memory_optimization(_weight_kurtosis)
isotropy_of_model = with_memory_optimization(_isotropy)
weight_norm_of_model = with_memory_optimization(_weight_norm)
spectral_norm_of_model = with_memory_optimization(_spectral_norm)
participation_ratio_of_model = with_memory_optimization(_participation_ratio)
sparsity_of_model = with_memory_optimization(_sparsity)
max_activation_of_model = with_memory_optimization(_max_activation)

# Add cache key functions for metrics that use caching
def _l2_cache_key(model):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return "l2_norm_empty"
    return "l2_norm_" + str(hash(tuple(p.sum().item() for p in trainable_params)))

def _entropy_cache_key(model):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return "weight_entropy_empty"
    return "weight_entropy_" + str(hash(tuple(p.sum().item() for p in trainable_params)))

l2_norm_of_model._cache_key_func = _l2_cache_key
weight_entropy_of_model._cache_key_func = _entropy_cache_key

# Import built-in metrics at module level for ProcessPoolExecutor
def _get_builtin_metrics():
    """Get built-in metrics that are always available."""
    return {
        "L2 Norm": l2_norm_of_model,
        "Weight Entropy": weight_entropy_of_model,
        "Layer Connectivity": layer_connectivity_of_model,
        "Parameter Variance": parameter_variance_of_model,
        "Layer Wise Norm Ratio": layer_wise_norm_ratio_of_model,
        "Activation Capacity": activation_capacity_of_model,
        "Dead Neuron Percentage": dead_neuron_percentage_of_model,
        "Weight Rank": weight_rank_of_model,
        "Gradient Flow Score": gradient_flow_score_of_model,
        "Effective Rank": effective_rank_of_model,
        "Avg Condition Number": avg_condition_number_of_model,
        "Flatness Proxy": flatness_proxy_of_model,
        "Mean Weight": mean_weight_of_model,
        "Weight Skew": weight_skew_of_model,
        "Weight Kurtosis": weight_kurtosis_of_model,
        "Isotropy": isotropy_of_model,
        "Weight Norm": weight_norm_of_model,
        "Spectral Norm": spectral_norm_of_model,
        "Participation Ratio": participation_ratio_of_model,
        "Sparsity": sparsity_of_model,
        "Max Activation": max_activation_of_model
    }


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
            value = func(model)
            # Ensure the result is a float
            if value is None:
                logger.warning(f"Metric {name} returned None, converting to NaN")
                results[name] = float('nan')
            else:
                results[name] = float(value)
        except Exception as e:
            logger.exception(f"Error calculating {name}: {e}")
            results[name] = float('nan')
    
    # No need to delete model explicitly - Python's garbage collection will handle it
    return results

import load_models
from state import load_state
def _process_checkpoint_cpu(args):  
    """
    args: (idx, path, metric_names, device, metrics_file)
    """
    idx, path, metric_names, device, metrics_file = args
    if load_models._model_class is None:      # ❷ check the live value
        state = load_state()
        load_models.load_model_class(         # ❸ sets load_models._model_class
            state["model_path"],
            state["class_name"],
        )

    try:
        # Resolve metric functions inside the worker
        builtin = _get_builtin_metrics()          # {'L2 Norm': l2_norm_of_model, ...}
        metric_funcs = {n: builtin[n] for n in metric_names if n in builtin}

        if metrics_file:
            custom = import_metric_functions(metrics_file)
            metric_funcs.update({n: custom[n] for n in metric_names if n in custom})
        if not metric_funcs:
            raise RuntimeError("No metric functions resolved inside worker.")

        checkpoint_name    = os.path.basename(path)
        checkpoint_results = compute_metric_batch(metric_funcs, path, device)

        return {
            "idx": idx,
            "checkpoint_name": checkpoint_name,
            "results": checkpoint_results,
            "error": None,
        }

    except Exception as e:
        logger.exception("Worker error on %s: %s", path, e)
        return {
            "idx": idx,
            "checkpoint_name": os.path.basename(path),
            "results": {n: float("nan") for n in metric_names},
            "error": str(e),
        }



def compute_metrics_over_checkpoints(
    metric_functions: Dict[str, Callable[[torch.nn.Module], float]],
    checkpoints: Sequence[str],
    device: str = "auto",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    parallel: bool = True,
    metrics_file: str = None
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
    
    # Determine if we're using GPU
    using_gpu = device.startswith("cuda")
    
    results = {name: [None] * len(checkpoints) for name in metric_functions}
    
    # Track progress
    metrics_progress = {
        name: {"completed": 0, "total": len(checkpoints)}
        for name in metric_functions
        
    }
    
    def process_checkpoint(idx, path):
        """Process checkpoint for ThreadPoolExecutor (GPU)."""
        try:
            checkpoint_name = os.path.basename(path)
            checkpoint_results = compute_metric_batch(metric_functions, path, device)
            
            # Store results and update progress info
            for name, value in checkpoint_results.items():
                results[name][idx] = value
                metrics_progress[name]["completed"] += 1
            
                
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
    
    if parallel and len(checkpoints) > 1:
        if using_gpu:
            # Use ThreadPoolExecutor for GPU
            max_workers = min(len(checkpoints), 4)  # Limit threads on GPU
            executor_type = "threads"
            
            # Print info about execution
            t.secho(f"Using {max_workers} threads on GPU", fg=t.colors.BLUE)
            logger.info(f"Using ThreadPoolExecutor with {max_workers} threads on GPU")
            
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
            # Use ProcessPoolExecutor for CPU
            max_workers = min(len(checkpoints), mp.cpu_count() or 4)
            
            # Print info about execution
            t.secho(f"\nCreated {max_workers} processes on CPU", fg=t.colors.BLUE)
            logger.info(f"Using ProcessPoolExecutor with {max_workers} processes on CPU")
            
            # Prepare arguments for multiprocessing
            args_list = [
                (i, p, list(metric_functions.keys()), device, metrics_file)  
                for i, p in enumerate(checkpoints)
            ]
            
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = [exe.submit(_process_checkpoint_cpu, a) for a in args_list]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        idx = result['idx']
                        
                        # Store results
                        for name, value in result['results'].items():
                            results[name][idx] = value
                            metrics_progress[name]["completed"] += 1
                        
                        # Report progress
                        if progress_callback:
                            progress_info = {
                                "current": idx + 1,
                                "total": len(checkpoints),
                                "checkpoint": result['checkpoint_name'],
                                "metrics": result['results'],
                                "metrics_progress": metrics_progress
                            }
                            progress_callback(progress_info)
                    except Exception as e:
                        logger.exception(f"Error in parallel processing: {e}")
    else:
        # Sequential processing
        t.secho("Processing checkpoints sequentially", fg=t.colors.BLUE)
        logger.info("Processing checkpoints sequentially")
        
        for i, path in enumerate(checkpoints):
            try:
                if using_gpu:
                    process_checkpoint(i, path)
                else:
                    # For CPU, we can still use the same approach
                    checkpoint_name = os.path.basename(path)
                    checkpoint_results = compute_metric_batch(metric_functions, path, device)
                    
                    # Store results and update progress info
                    for name, value in checkpoint_results.items():
                        results[name][i] = value
                        metrics_progress[name]["completed"] += 1
                    
                    # Report progress
                    if progress_callback:
                        progress_info = {
                            "current": i + 1,
                            "total": len(checkpoints),
                            "checkpoint": checkpoint_name,
                            "metrics": checkpoint_results,
                            "metrics_progress": metrics_progress
                        }
                        progress_callback(progress_info)
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