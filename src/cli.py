# TODO:
# get good models to test with (+ test with actual models)
# clean up in general
# readme - still to be improved but draft is ok, screenshots, fix math rendering
# update requirements.txt
# package it together nicely
# test end to end and do slides

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
from metrics import (import_metric_functions, compute_metrics_over_checkpoints, clear_metric_cache)
sys.path.append(os.path.dirname(__file__))
from load_models import load_model_class, contains_checkpoints, clear_model_cache
from state import load_state, save_state
from generate_plot import plot_metric_interactive
from loading import SimpleLoadingAnimation
from utils import suppress_stdout_stderr, logger, BUILTIN_METRICS

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
    parallel: bool = t.Option(True, help="Use parallel processing for calculations (default: True)") # This needs to go back to True
):
    """Compute and plot metrics over model checkpoints."""
    metrics_file = ""
    logger.info(f"Starting plot-metric command with device={device}, parallel={parallel}")   
    try:
        state = load_state() 
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
    
    if using_cpu:
        logger.info("Running on CPU")
        t.secho("WARNING: Running on CPU. Consider using GPU for large models or many checkpoints.", 
               fg=t.colors.YELLOW, bold=True)
    else:
        logger.info(f"Running on {device}")
    
    with suppress_stdout_stderr():
        load_model_class(state["model_path"], state["class_name"])

    checkpoints = state["checkpoints"]
    metrics_to_calculate = {}
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
            
        if not raw.endswith(".py") and raw != "metrics":
            candidate = raw + ".py"
            if os.path.isfile(candidate):
                raw = candidate

        if os.path.isfile(raw):
            metrics_file = raw
            path = Path(raw).resolve()
            if path.suffix != ".py":
                _err(f"Custom metrics must come from a .py file (got {path})")
            try:
                with suppress_stdout_stderr():
                    metric_functions = import_metric_functions(str(path))           
                if not metric_functions:
                    _err(f"No valid metric functions found in {path}")
                for metric_name, metric_fn in metric_functions.items():
                    if metric_name in metrics_to_calculate:
                        logger.info(f"Metric '{metric_name}' already added, skipping")
                        t.secho(f"Metric '{metric_name}' already added. Skipping.", fg=t.colors.YELLOW)
                        continue
                    logger.info(f"Added metric: {metric_name}")
                    t.secho(f"Added metric: {metric_name}", fg=t.colors.GREEN)
                    metrics_to_calculate[metric_name] = metric_fn
            except Exception as e: 
                _err(f"Failed importing metric functions from {path}: {e}")
        else:
            match raw.lower(): 
                case "metrics":
                    t.secho("\nAvailable built-in metrics:", fg=t.colors.CYAN, bold=True)
                    t.secho("• L2 Norm (l2) - L2 norm of all trainable parameters", fg=t.colors.CYAN)
                    t.secho("• Weight Entropy (entropy) - Shannon entropy of weight distribution", fg=t.colors.CYAN)
                    t.secho("• Layer Connectivity (connectivity) - Average absolute weight per layer", fg=t.colors.CYAN)
                    t.secho("• Parameter Variance (variance) - Variance of all trainable parameters", fg=t.colors.CYAN)
                    t.secho("• Layer Wise Norm Ratio (norm_ratio) - Ratio of norms between first and last layers", fg=t.colors.CYAN)
                    t.secho("• Activation Capacity (capacity) - Model's representational capacity", fg=t.colors.CYAN)
                    t.secho("• Dead Neuron Percentage (dead_neurons) - Percentage of near-zero weights", fg=t.colors.CYAN)
                    t.secho("• Weight Rank (rank) - Average effective rank of weight matrices", fg=t.colors.CYAN)
                    t.secho("• Gradient Flow Score (gradient_flow) - Gradient flow quality score", fg=t.colors.CYAN)
                    t.secho("• Effective Rank (effective_rank) - Effective rank using entropy of singular values", fg=t.colors.CYAN)
                    t.secho("• Avg Condition Number (condition) - Average condition number of weight matrices", fg=t.colors.CYAN)
                    t.secho("• Flatness Proxy (flatness) - Proxy for loss landscape flatness", fg=t.colors.CYAN)
                    t.secho("• Mean Weight (mean) - Mean of all trainable weights", fg=t.colors.CYAN)
                    t.secho("• Weight Skew (skew) - Skewness of weight distribution", fg=t.colors.CYAN)
                    t.secho("• Weight Kurtosis (kurtosis) - Kurtosis of weight distribution", fg=t.colors.CYAN)
                    t.secho("• Isotropy (isotropy) - Isotropy of weight matrices", fg=t.colors.CYAN)
                    t.secho("• Weight Norm (weight_norm) - Frobenius norm of all trainable parameters", fg=t.colors.CYAN)
                    t.secho("• Spectral Norm (spectral) - Maximum singular value across weight matrices", fg=t.colors.CYAN)
                    t.secho("• Participation Ratio (participation) - How evenly distributed weight values are", fg=t.colors.CYAN)
                    t.secho("• Sparsity (sparsity) - Fraction of near-zero parameters", fg=t.colors.CYAN)
                    t.secho("• Max Activation (max_activation) - Maximum potential activation", fg=t.colors.CYAN)
                    t.secho("\nYou can also provide a .py file with custom metrics ending in '_of_model'", fg=t.colors.YELLOW)

                case "all":
                    all_metrics = True
                    for pretty_name, fn in BUILTIN_METRICS.values():
                        if pretty_name in metrics_to_calculate:
                            logger.info(f"Metric '{pretty_name}' already added, skipping")
                            continue
                        logger.info(f"Added metric: {pretty_name}")
                        t.secho(f"Added metric: {pretty_name}", fg=t.colors.GREEN)
                        metrics_to_calculate[pretty_name] = fn
                    break

                case key if key in BUILTIN_METRICS:
                    pretty_name, fn = BUILTIN_METRICS[key]
                    if pretty_name in metrics_to_calculate:
                        logger.info(f"Metric '{pretty_name}' already added, skipping")
                        t.secho(f"Metric '{pretty_name}' already added. Skipping.",
                                fg=t.colors.YELLOW)
                        continue
                    logger.info(f"Added metric: {pretty_name}")
                    t.secho(f"Added metric: {pretty_name}", fg=t.colors.GREEN)
                    metrics_to_calculate[pretty_name] = fn
            
                case _:
                    logger.warning(f"Unknown metric: {raw}")
                    t.secho(f"{raw} is not an in-built metric. For a list of available "
                            "metrics, enter 'metrics'.", fg=t.colors.RED)
                  
    logger.info(f"Starting calculation of {len(metrics_to_calculate)} metrics across {len(checkpoints)} checkpoints")
    t.secho(f"\nCalculating {len(metrics_to_calculate)} metrics across {len(checkpoints)} checkpoints...", 
           fg=t.colors.BLUE, bold=True)
    
    loading_animation = SimpleLoadingAnimation(
        base_text="Calculating metrics",
        color="blue"
    )
    loading_animation.start(total_metrics=len(metrics_to_calculate))
    progress_lock = threading.Lock()
    completed_metrics = {name: 0 for name in metrics_to_calculate}
    
    def progress_callback(progress_info):
        with progress_lock: 
            for name, progress in progress_info["metrics_progress"].items():
                completed = progress["completed"]
                if completed > completed_metrics.get(name, 0):
                    completed_metrics[name] = completed
            fully_completed = sum(1 for name, count in completed_metrics.items()
                               if count >= len(checkpoints))
        
            loading_animation.update(current=fully_completed, total=len(metrics_to_calculate))
    
    try:
        metrics_data = compute_metrics_over_checkpoints(
            metrics_to_calculate,
            checkpoints,
            device=device,
            progress_callback=progress_callback,
            parallel=parallel,
            metrics_file=metrics_file,
        )
        logger.info("Successfully computed all metrics")
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        loading_animation.stop(f"Error calculating metrics: {str(e)}")
        _err(f"Error calculating metrics: {str(e)}")
    loading_animation.stop(f"Successfully computed {len(completed_metrics.keys())} metrics.")
    
    with suppress_stdout_stderr():
        clear_model_cache()
        clear_metric_cache()
    logger.info("Cleared model and metric caches")
    state["metrics_data"] = metrics_data
    save_state(state)
    logger.info("Saved metrics data to state")

    checkpoint_names = [os.path.basename(p) for p in checkpoints]
    t.secho("\nOpening interactive visualization. In the plot, you can:", fg=t.colors.CYAN)
    t.secho("- Use the dropdown menu to toggle between metrics or show all at once", fg=t.colors.CYAN)
    t.secho("- Interact with the data points for detailed information", fg=t.colors.CYAN)
    t.secho("- Download the plot as PNG using the button in the top-right", fg=t.colors.CYAN)
    t.secho("- Export the data as a CSV for further analysis", fg=t.colors.CYAN)
    
    try:
        logger.info("Generating interactive plot")
        with suppress_stdout_stderr():
            plot_metric_interactive(
                checkpoint_names=checkpoint_names,
                metrics_data=metrics_data,
                many_metrics=len(metrics_to_calculate) > 7
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