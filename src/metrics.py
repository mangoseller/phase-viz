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

# Global metric cache for memoization
_metric_cache = {}

# Import built-in metrics at module level for ProcessPoolExecutor
def _get_builtin_metrics():
    """Get built-in metrics that are always available."""
    return {
        "L2 Norm": l2_norm_of_model
    }

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
    with torch.no_grad():  
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


# Helper function for ProcessPoolExecutor (needs to be at module level)
def _process_checkpoint_cpu(args):
    """Process a checkpoint on CPU (for ProcessPoolExecutor)."""
    idx, path, metric_names, metric_module_path, device = args
    try:
        # Get built-in metrics
        builtin_metrics = _get_builtin_metrics()
        metric_functions = {}
        
        # Add built-in metrics
        for name in metric_names:
            if name in builtin_metrics:
                metric_functions[name] = builtin_metrics[name]
        
        # Import custom metrics if provided
        if metric_module_path and os.path.exists(metric_module_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("_metrics", metric_module_path)
            metrics_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metrics_module)
            
            # Add custom metrics
            for name in metric_names:
                if name not in metric_functions:
                    # Find the function in the module
                    func_name = name.lower().replace(" ", "_") + "_of_model"
                    if hasattr(metrics_module, func_name):
                        metric_functions[name] = getattr(metrics_module, func_name)
        
        checkpoint_name = os.path.basename(path)
        checkpoint_results = compute_metric_batch(metric_functions, path, device)
        
        return {
            'idx': idx,
            'path': path,
            'checkpoint_name': checkpoint_name,
            'results': checkpoint_results,
            'error': None
        }
    except Exception as e:
        logger.exception(f"Error processing checkpoint {path}: {e}")
        return {
            'idx': idx,
            'path': path,
            'checkpoint_name': os.path.basename(path),
            'results': {name: float('nan') for name in metric_names},
            'error': str(e)
        }


def compute_metrics_over_checkpoints(
    metric_functions: Dict[str, Callable[[torch.nn.Module], float]],
    checkpoints: Sequence[str],
    device: str = "auto",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    parallel: bool = True
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
            t.secho(f"Created {max_workers} processes on CPU", fg=t.colors.BLUE)
            logger.info(f"Using ProcessPoolExecutor with {max_workers} processes on CPU")
            
            # Prepare arguments for multiprocessing
            args_list = [
                (i, path, metric_functions, device)
                for i, path in enumerate(checkpoints)
            ]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_checkpoint_cpu, args) for args in args_list]
                
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