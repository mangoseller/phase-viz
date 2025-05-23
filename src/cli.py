# Fix overlay title - should not just say All metrics when two metrics are overlayed - fix x-axis cut off and legend
# cut off, fix metrics and test with multiple metrics properly to make sure overlay and phase-transitions buttons
# do in fact work.

#TODO: pass final tests, ensure differing architectures can be loaded, test cuda on vast or something
# trim slop, fix metrics - all custom metrics arent working right now, get good models to test with (checks)
# bugtest, find good metrics to compute LLC and stuff, --help arg for metrics list metrics

# Stretch Goals - Metrics that require inference, multiple model direct comparison


import os
import typer as t 
import typing
import sys
import time
import importlib.util
import inspect
import threading
from pathlib import Path
import torch

sys.path.append(os.path.dirname(__file__))
from load_models import load_model_class, contains_checkpoints, clear_model_cache
from state import load_state, save_state
from metrics import (
    l2_norm_of_model,
    compute_metrics_over_checkpoints,
    import_metric_functions,
    clear_metric_cache
)
from generate_plot import plot_metric_interactive
from loading import SimpleLoadingAnimation
from utils import suppress_stdout_stderr, logger

# Initialize typer app
app = t.Typer(no_args_is_help=False)


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
    logger.info(f"Loading directory: {dir}")
    logger.info(f"Model file: {model}")
    logger.info(f"Class name: {class_name}")
    
    with suppress_stdout_stderr():
        model_class = load_model_class(model, class_name)

    if not os.path.isdir(dir):
        error_msg = f"The path '{dir}' is not a directory."
        logger.error(error_msg)
        t.secho(error_msg, fg=t.colors.RED, bold=True)
        raise t.Exit(code=1)

    try:
        checkpoints = contains_checkpoints(dir)
        logger.info(f"Found {len(checkpoints)} checkpoints")
    except Exception as e:
        logger.error(f"Error loading checkpoints: {str(e)}")
        t.secho(f"{str(e)}", fg=t.colors.RED, bold=True)
        raise t.Exit(code=1)

    state = {
        "dir": dir,
        "model_path": model,
        "class_name": class_name,
        "checkpoints": checkpoints,
    }
    save_state(state)
    logger.info("State saved successfully")

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
    logger.info(f"Starting plot-metric command with device={device}, parallel={parallel}")
    
    try:
        state = load_state() # Try to load state file
        logger.info("State loaded successfully")
    except RuntimeError as e:
        logger.error(f"Failed to load state: {str(e)}")
        t.secho(str(e), fg=t.colors.RED, bold=True)
        raise t.Exit(code=1)
    

    using_cpu = False
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        t.secho("WARNING: CUDA requested but not available. Falling back to CPU.", 
               fg=t.colors.YELLOW, bold=True)
        device = "cpu"
        using_cpu = True

    elif device == "cpu":
        using_cpu = True
    
    # Display warning if using CPU
    if using_cpu:
        logger.info("Running on CPU")
        t.secho("WARNING: Running on CPU. Consider using GPU for large models or many checkpoints.", 
               fg=t.colors.YELLOW, bold=True)
    else:
        logger.info(f"Running on {device}")
    
    with suppress_stdout_stderr():
        load_model_class(state["model_path"], state["class_name"])
    
    checkpoints = state["checkpoints"]
    
    # Dict to store metrics to be calculated: {metric_name: metric_function}
    metrics_to_calculate = {}
    
    # Loop to collect multiple metrics (but without calculating them yet)
    while True:
        raw = t.prompt(
            "Enter a metric name, a path to a custom .py file, 'metrics' or 'done' to finish selecting metrics"
        ).strip()
        
        logger.info(f"User input: {raw}")
        
        if raw.lower() == 'done':
            if not metrics_to_calculate:
                logger.warning("User tried to continue without selecting metrics")
                t.secho("No metrics selected. Please select at least one metric.", fg=t.colors.RED)
                t.secho("To exit, type 'exit'", fg=t.colors.YELLOW)
                continue
            break
        if raw.lower() == 'exit':
            logger.info("User exited")
            t.secho("Exiting, goodbye.", fg=t.colors.GREEN)
            raise t.Exit(code=0)
            
        if not raw.endswith(".py"):
            candidate = raw + ".py"
            if os.path.isfile(candidate):
                raw = candidate

        if os.path.isfile(raw):
            path = Path(raw).resolve()
            if path.suffix != ".py":
                _err(f"Custom metrics must come from a .py file (got {path})")
            # Import all metric functions from the file
            try:
                with suppress_stdout_stderr():
                    metric_functions = import_metric_functions(str(path))
                
                if not metric_functions:
                    _err(f"No valid metric functions found in {path}")
                
                # Add each metric to the to-calculate list
                for metric_name, metric_fn in metric_functions.items():
                    if metric_name in metrics_to_calculate:
                        logger.info(f"Metric '{metric_name}' already added, skipping")
                        t.secho(f"Metric '{metric_name}' already added. Skipping.", fg=t.colors.YELLOW)
                        continue
                    
                    logger.info(f"Added metric: {metric_name}")
                    t.secho(f"Added metric: {metric_name}", fg=t.colors.GREEN)
                    metrics_to_calculate[metric_name] = metric_fn
            except Exception as e: # Check this, we should import valid functions and not break
                _err(f"Failed importing metric functions from {path}: {e}")
        else:
            # Look for built-in metrics
            match raw.lower():
                case "l2":
                    if "L2 Norm" in metrics_to_calculate:
                        logger.info("L2 Norm already added, skipping")
                        t.secho("Metric 'L2 Norm' already added. Skipping.", fg=t.colors.YELLOW)
                        continue
                    
                    logger.info("Added metric: L2 Norm")
                    t.secho("Added metric: {0} (will be calculated when you type 'done')".format(
                        "L2 Norm"), fg=t.colors.GREEN)
                    metrics_to_calculate["L2 Norm"] = l2_norm_of_model
                case _:
                    logger.warning(f"Unknown metric: {raw}")
                    t.secho(f"{raw} is not an inbuilt metric. For a list of available metrics, enter 'metrics'.", fg=t.colors.RED)
                    continue
    
    
    logger.info(f"Starting calculation of {len(metrics_to_calculate)} metrics across {len(checkpoints)} checkpoints")
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
    
    
    def progress_callback(progress_info):
        with progress_lock:
            
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
        logger.info("Successfully computed all metrics")
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        loading_animation.stop(f"Error calculating metrics: {str(e)}")
        _err(f"Error calculating metrics: {str(e)}")
    
    # Stop the loading animation
    loading_animation.stop(f"Successfully computed {len(completed_metrics.keys())} metrics.")
    
    # Clear caches to free memory
    with suppress_stdout_stderr():
        clear_model_cache()
        clear_metric_cache()
    logger.info("Cleared model and metric caches")
    
    # Save all metrics data for later use
    state["metrics_data"] = metrics_data
    save_state(state)
    logger.info("Saved metrics data to state")
    
    # Get checkpoint names for display
    checkpoint_names = [os.path.basename(p) for p in checkpoints]
    
    # Show usage instructions 
    t.secho("\nOpening interactive visualization. In the plot, you can:", fg=t.colors.CYAN)
    t.secho("- Use the dropdown menu to toggle between metrics or show all at once", fg=t.colors.CYAN)
    t.secho("- Interact with the data points for detailed information", fg=t.colors.CYAN)
    t.secho("- Download the plot as PNG using the button in the top-right", fg=t.colors.CYAN)
    t.secho("- Export the data as a CSV for further analysis", fg=t.colors.CYAN)
    
    
    
    try:
        # Generate the plot
        logger.info("Generating interactive plot")
        with suppress_stdout_stderr():
            plot_metric_interactive(
                checkpoint_names=checkpoint_names,
                metrics_data=metrics_data
            )
        logger.info("Plot generation completed")
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}")
        _err(f"Error generating plot: {str(e)}")



def _err(msg: str) -> None:  
    logger.error(f"Fatal error: {msg}")
    t.secho(msg, fg=t.colors.RED, bold=True)
    raise t.Exit(code=1)


if __name__ == "__main__":
    logger.info("Starting phase-viz CLI")
    if len(sys.argv) == 1:
        welcome()
    else:
        app()
    logger.info("phase-viz CLI exiting")