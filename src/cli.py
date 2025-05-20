# Set these at the very beginning of the file, before any imports
import os
# Suppress all GTK-related warnings (these must be set before any imports)
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
from loading import ProgressBar, AnimatedText

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
    device: str = t.Option("auto", help="Device to use for calculations ('auto', 'cpu', 'cuda')"),
    parallel: bool = t.Option(True, help="Use parallel processing for calculations"),
    simulate_slow: bool = t.Option(True, help="Simulate slow calculations for testing (default: True)")
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
    
    # Force a small delay to ensure everything is properly initialized
    time.sleep(0.1)
    
    # Now calculate all metrics with progress bars
    t.secho(f"\nCalculating {len(metrics_to_calculate)} metrics...", fg=t.colors.BLUE, bold=True)
    
    # Set up progress bars for each metric
    progress_bars = {}
    for metric_name in metrics_to_calculate.keys():
        progress_bars[metric_name] = ProgressBar(
            total=len(checkpoints),
            desc=f"Computing {metric_name}",
            color="blue",
            unit="checkpoint"
        )
    
    # Calculate all metrics in batch with progress reporting
    def batch_progress_callback(progress_info):
        # Update progress for each metric
        current = progress_info["current"]
        total = progress_info["total"]
        
        # Update each metric's progress bar
        for metric_name, _ in progress_info["metrics"].items():
            if metric_name in progress_bars:
                progress_bars[metric_name].update(current)
    
    # Start calculation with optimized batch processing
    with suppress_stdout_stderr():
        metrics_data = compute_metrics_over_checkpoints(
            metrics_to_calculate,
            checkpoints,
            device=device,
            progress_callback=batch_progress_callback,
            simulate_slow=simulate_slow,  # Make this True by default for testing
            parallel=parallel
        )
    
    # Complete all progress bars
    for metric_name, progress_bar in progress_bars.items():
        progress_bar.finish(f"Completed {metric_name} calculation")
    
    # Small delay to ensure animations are complete
    time.sleep(0.2)
    
    # Clear caches to free memory
    with suppress_stdout_stderr():
        clear_model_cache()
        clear_metric_cache()
    
    # Save all metrics data for later use
    state["metrics_data"] = metrics_data
    save_state(state)
    
    # Plot all metrics using the enhanced plot function
    checkpoint_names = [os.path.basename(p) for p in checkpoints]
    
    # Show plotting animation
    plotting_animation = AnimatedText("Plotting metrics", t.colors.BLUE)
    plotting_animation.start()
    
    try:
        # Capture and suppress all output from plot generation
        with suppress_stdout_stderr():
            plot_metric_interactive(
                checkpoint_names=checkpoint_names,
                metrics_data=metrics_data
            )
    finally:
        # Ensure animation is stopped even if plotting fails
        plotting_animation.stop(f"Plot generated with {len(metrics_data)} metrics.")
    
    t.secho("Use the dropdown menu to toggle between metrics or show all at once.", fg=t.colors.CYAN)


def _err(msg: str) -> None:  
    t.secho(msg, fg=t.colors.RED, bold=True)
    raise t.Exit(code=1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        welcome()
    else:
        app()
