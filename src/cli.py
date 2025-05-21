# Clean up, might be nice to print out using x threads, logging , delete html on kill sig, test with diff archs, metrics inc physics, run inference to get loss
# and what not, ProcessPoolExecutor if CPU, else normal threading (what I have now), ofc fixing up graph and implementing nice visual features there. 
import os
# Suppress all GTK-related warnings
os.environ['GTK_DEBUG'] = '0'
os.environ['NO_AT_BRIDGE'] = '1'
os.environ['ACCESSIBILITY_ENABLED'] = '0'

import typer as t # type: ignore
import typing
import sys
import time
import importlib.util
import inspect
import logging
import threading
from pathlib import Path
from contextlib import contextmanager
import torch
sys.path.append(os.path.dirname(__file__))
from loader import load_model_class, contains_checkpoints, clear_model_cache
from state import load_state, save_state
from metrics import (
    l2_norm_of_model,
    compute_metrics_over_checkpoints,
    import_metric_functions,
    clear_metric_cache
)
from generate_plot import plot_metric_interactive
from loading import SimpleLoadingAnimation

# Configure logging - only show critical errors
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger("phase-viz")
app = t.Typer(no_args_is_help=False)

@contextmanager
def suppress_stdout_stderr():
    """
    Context manager to suppress stdout and stderr.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    class NullIO:
        def write(self, *args, **kwargs):
            pass
        def flush(self, *args, **kwargs):
            pass
    
    try:
        sys.stdout = NullIO()
        sys.stderr = NullIO()
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def welcome():
    title = t.style("phase-viz", fg=t.colors.CYAN, bold=True)
    subtitle = t.style("Visualize the developmental trajectory of a neural network", fg=t.colors.MAGENTA)
    instruction = t.style("Provide a directory path with -dir {PATH} to begin analysis.", fg=t.colors.GREEN)
    help_hint = t.style("Use --help for more options.", fg=t.colors.YELLOW, dim=True)

    t.echo()
    t.secho("=" * 60, fg="white")
    t.echo(f"{title}")
    t.echo(f"{subtitle}\n")
    t.echo(f"{instruction}")
    t.echo(f"{help_hint}")
    t.secho("=" * 60, fg="white")
    t.echo()

@app.command()
def load_dir(
    dir: str = t.Option(..., help="Directory containing model checkpoints"),
    model: str = t.Option(..., help="Path to Python file defining the model class"),
    class_name: str = t.Option(..., help="Name of the model class inside the file")
):
    with suppress_stdout_stderr():
        model_class = load_model_class(model, class_name)

    if not os.path.isdir(dir):
        t.secho(f"The path '{dir}' is not a directory.", fg=t.colors.RED, bold=True)
        raise t.Exit(code=1)

    try:
        checkpoints = contains_checkpoints(dir)
    except Exception as e:
        t.secho(f"{str(e)}", fg=t.colors.RED, bold=True)
        raise t.Exit(code=1)

    state = {
        "dir": dir,
        "model_path": model,
        "class_name": class_name,
        "checkpoints": checkpoints,
    }
    save_state(state)

    t.secho(f"Loaded {len(checkpoints)} checkpoint(s) successfully.", fg=t.colors.GREEN, bold=True)


@app.command()
def plot_metric(
    device: str = t.Option("cuda", help="Device to use for calculations ('cuda', 'cpu', specific 'cuda:n')"),
    parallel: bool = t.Option(True, help="Use parallel processing for calculations (default: True)")
):
    """Compute and plot metrics over model checkpoints.
    
    This command allows you to select metrics to calculate over checkpoints,
    and then visualizes them in an interactive plot.
    """
    try:
        state = load_state() # load json state file if it exists
    except RuntimeError as e:
        t.secho(str(e), fg=t.colors.RED, bold=True)
        raise t.Exit(code=1)
    
    # Check if CUDA is available when device is set to "cuda"
    using_cpu = False
    if device == "cuda" and not torch.cuda.is_available():
        t.secho("WARNING: CUDA requested but not available. Falling back to CPU.", 
               fg=t.colors.YELLOW, bold=True)
        device = "cpu"
        using_cpu = True
    elif device == "cpu":
        using_cpu = True
    
    # Display warning if using CPU
    if using_cpu:
        t.secho("WARNING: Running on CPU. Consider using GPU for large models or many checkpoints.", 
               fg=t.colors.YELLOW, bold=True)
    
    with suppress_stdout_stderr():
        load_model_class(state["model_path"], state["class_name"])
    
    checkpoints = state["checkpoints"]
    
    # Dict to store metrics to be calculated: {metric_name: metric_function}
    metrics_to_calculate = {}
    
    # Loop to collect multiple metrics (but without calculating them yet)
    while True:
        raw = t.prompt(
            "Enter metric name (e.g. 'l2'), path to a custom .py file, or 'done' to finish selecting metrics"
        ).strip()
        
        if raw.lower() == 'done':
            if not metrics_to_calculate:
                t.secho("No metrics selected. Please select at least one metric.", fg=t.colors.RED)
                continue
            break
            
        if os.path.isfile(raw):
            path = Path(raw).resolve()
            if path.suffix != ".py":
                _err(f"Custom metric must be a .py file (got {path})")
                
            # Import all metric functions from the file
            try:
                with suppress_stdout_stderr():
                    metric_functions = import_metric_functions(str(path))
                
                if not metric_functions:
                    _err(f"No valid metric functions found in {path}")
                
                # Add each metric to the to-calculate list
                for metric_name, metric_fn in metric_functions.items():
                    if metric_name in metrics_to_calculate:
                        t.secho(f"Metric '{metric_name}' already added. Skipping.", fg=t.colors.YELLOW)
                        continue
                    
                    t.secho(f"Added metric: {metric_name} (will be calculated when you type 'done')", fg=t.colors.GREEN)
                    metrics_to_calculate[metric_name] = metric_fn
            except Exception as e:
                _err(f"Failed importing metric functions from {path}: {e}")
        else:
            # Look for built-in metrics
            match raw.lower():
                case "l2":
                    if "L2 Norm" in metrics_to_calculate:
                        t.secho("Metric 'L2 Norm' already added. Skipping.", fg=t.colors.YELLOW)
                        continue
                    
                    t.secho("Added metric: {0} (will be calculated when you type 'done')".format(
                        "L2 Norm"), fg=t.colors.GREEN)
                    metrics_to_calculate["L2 Norm"] = l2_norm_of_model
                case _:
                    t.secho("Could not locate metric/file.", fg=t.colors.RED)
                    continue
    
    # Create a single loading animation for all metrics
    t.secho(f"\nCalculating {len(metrics_to_calculate)} metrics across {len(checkpoints)} checkpoints...", 
           fg=t.colors.BLUE, bold=True)
    
    # Create the loading animation
    loading_animation = SimpleLoadingAnimation(
        base_text="Calculating metrics",
        color="blue"
    )
    loading_animation.start(total_metrics=len(metrics_to_calculate))
    
    # Thread-safe progress tracking
    progress_lock = threading.Lock()
    completed_metrics = {name: 0 for name in metrics_to_calculate}
    
    # Define a thread-safe progress callback for the animation
    def progress_callback(progress_info):
        with progress_lock:
            # Update completed count for each metric
            for name, progress in progress_info["metrics_progress"].items():
                completed = progress["completed"]
                if completed > completed_metrics.get(name, 0):
                    completed_metrics[name] = completed
            
            # Count metrics that are 100% complete
            fully_completed = sum(1 for name, count in completed_metrics.items()
                               if count >= len(checkpoints))
            
            # Update the loading animation
            loading_animation.update(current=fully_completed, total=len(metrics_to_calculate))
    
    # Calculate all metrics
    try:
        metrics_data = compute_metrics_over_checkpoints(
            metrics_to_calculate,
            checkpoints,
            device=device,
            progress_callback=progress_callback,
            parallel=parallel
        )
    except Exception as e:
        loading_animation.stop(f"Error calculating metrics: {str(e)}")
        _err(f"Error calculating metrics: {str(e)}")
    
    # Stop the loading animation
    loading_animation.stop(f"Metrics calculation complete")
    
    # Clear caches to free memory
    with suppress_stdout_stderr():
        clear_model_cache()
        clear_metric_cache()
    
    # Save all metrics data for later use
    state["metrics_data"] = metrics_data
    save_state(state)
    
    # Get checkpoint names for display
    checkpoint_names = [os.path.basename(p) for p in checkpoints]
    
    # Show usage instructions BEFORE opening the browser
    t.secho("\nOpening interactive visualization. In the plot, you can:", fg=t.colors.CYAN)
    t.secho("- Use the dropdown menu to toggle between metrics or show all at once", fg=t.colors.CYAN)
    t.secho("- Interact with the data points for detailed information", fg=t.colors.CYAN)
    t.secho("- Download the plot as PNG using the button in the top-right", fg=t.colors.CYAN)
    t.secho("- Export the data as CSV for further analysis", fg=t.colors.CYAN)
    
    # Show plotting animation (much simpler now)
    plotting_animation = SimpleLoadingAnimation("Opening visualization", t.colors.BLUE)
    plotting_animation.start(total_metrics=1)
    
    try:
        # Generate the interactive plot
        with suppress_stdout_stderr():
            plot_metric_interactive(
                checkpoint_names=checkpoint_names,
                metrics_data=metrics_data
            )
    except Exception as e:
        plotting_animation.stop(f"Error generating plot: {str(e)}")
        _err(f"Error generating plot: {str(e)}")
    finally:
        # Ensure animation stops even if plotting fails
        plotting_animation.stop("Interactive plot opened in browser")


def _err(msg: str) -> None:  
    t.secho(msg, fg=t.colors.RED, bold=True)
    raise t.Exit(code=1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        welcome()
    else:
        app()

# Add the required import at the top level
import torch