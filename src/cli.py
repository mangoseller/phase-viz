#TODO: support mac and windows w parallel processing

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
# Module packages
from load_models import load_model_class, contains_checkpoints, clear_model_cache
from state import load_state, save_state
from generate_plot import plot_metric_interactive
from loading import SimpleLoadingAnimation
from utils import (
    suppress_stdout_stderr, logger, 
    BUILTIN_METRICS, 
    get_model_classes,
    ClassNameLoadingError
)
from metrics import (
    import_metric_functions, 
    compute_metrics_over_checkpoints, 
    clear_metric_cache
    )

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


def _select_class_name(model_path: str, suggested_name: str = None) -> str:
    """Helper to select or auto-detect class name."""
    if suggested_name:
        return suggested_name
        
    try:
        available_classes = get_model_classes(model_path)
        if len(available_classes) == 0:
            _err(f"No nn.Module subclasses found in {model_path}")
        elif len(available_classes) == 1:
            class_name = available_classes[0]
            logger.info(f"Auto-detected class: {class_name}")
            t.secho(f"Auto-detected class: {class_name}", fg=t.colors.BLUE, bold=True)
            return class_name
        else:
            # Multiple classes found - let user choose
            t.secho(f"Multiple model classes found: {', '.join(available_classes)}", fg=t.colors.YELLOW, bold=True)
            t.secho("Please specify which class to use with --class-name", fg=t.colors.YELLOW, bold=True)
            raise t.Exit(code=1)
    except Exception as e:
        logger.error(f"{str(e)}")
        if isinstance(e, ClassNameLoadingError):
            t.secho(f"Error auto-detecting class: {str(e)}", fg=t.colors.RED, bold=True)
            t.secho("Please specify the class name manually with --class-name", fg=t.colors.YELLOW)
            raise t.Exit(code=1)
        else:
            t.secho(f"Something went wrong - please view logs for more details.", fg=t.colors.RED, bold=True)
            raise t.Exit(code=1)


def _select_metrics(max_metrics: int = None) -> tuple:
    """Interactive metric selection. Returns (metrics_dict, metrics_file)."""
    metrics_to_calculate = {}
    metrics_file = ""
    
    while True:
        if max_metrics and len(metrics_to_calculate) >= max_metrics:
            t.secho(f"Reached {max_metrics} metric limit.", fg=t.colors.YELLOW)
            break
            
        remaining_text = f" ({max_metrics - len(metrics_to_calculate)} slots remaining)" if max_metrics else ""
        prompt_text = f"Enter a metric name, a path to a custom .py file, 'metrics' or 'done' to finish selecting metrics{remaining_text}"
        
        raw = t.prompt(prompt_text).strip()
        logger.info(f"User input: {raw}")
        
        if raw.lower() == 'done':
            if not metrics_to_calculate:
                t.secho("No metrics selected. Please select at least one metric.", fg=t.colors.RED)
                continue
            break
            
        if raw.lower() == 'exit':
            t.secho("Exiting, goodbye.", fg=t.colors.GREEN)
            raise t.Exit(code=0)
            
        # Check if the user input a file name with .py truncated
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
                    if max_metrics and len(metrics_to_calculate) >= max_metrics:
                        t.secho(f"Reached {max_metrics} metric limit. Skipping remaining metrics.", fg=t.colors.YELLOW)
                        break
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
            if raw.lower() == "metrics":
                _show_available_metrics()
                if max_metrics:
                    t.secho(f"\nRemember: You can select up to {max_metrics - len(metrics_to_calculate)} more metric(s)", 
                           fg=t.colors.YELLOW)
            elif raw.lower() == "all":
                if max_metrics:
                    t.secho("Cannot select 'all' when there's a metric limit.", fg=t.colors.RED)
                    continue
                for pretty_name, fn in BUILTIN_METRICS.values():
                    if pretty_name not in metrics_to_calculate:
                        logger.info(f"Added metric: {pretty_name}")
                        t.secho(f"Added metric: {pretty_name}", fg=t.colors.GREEN)
                        metrics_to_calculate[pretty_name] = fn
                break
            elif raw.lower() in BUILTIN_METRICS:
                if max_metrics and len(metrics_to_calculate) >= max_metrics:
                    t.secho(f"Already selected {max_metrics} metrics. Type 'done' to continue.", fg=t.colors.YELLOW)
                    continue
                    
                pretty_name, fn = BUILTIN_METRICS[raw.lower()]
                if pretty_name in metrics_to_calculate:
                    logger.info(f"Metric '{pretty_name}' already added, skipping")
                    t.secho(f"Metric '{pretty_name}' already added. Skipping.", fg=t.colors.YELLOW)
                    continue
                logger.info(f"Added metric: {pretty_name}")
                t.secho(f"Added metric: {pretty_name}", fg=t.colors.GREEN)
                metrics_to_calculate[pretty_name] = fn
            else:
                logger.warning(f"Unknown metric: {raw}")
                t.secho(f"{raw} is not an in-built metric. For a list of available metrics, enter 'metrics'.", 
                       fg=t.colors.RED)
    
    return metrics_to_calculate, metrics_file


def _show_available_metrics():
    """Display available built-in metrics."""
    t.secho("\nAvailable built-in metrics:", fg=t.colors.CYAN, bold=True)
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


def _compute_metrics_with_animation(
    metrics_to_calculate: dict,
    checkpoints: list,
    device: str,
    parallel: bool,
    metrics_file: str,
    model_path: str,
    class_name: str,
    model_label: str = ""
) -> dict:
    """Compute metrics with loading animation."""
    label = f" for {model_label}" if model_label else ""
    logger.info(f"Starting calculation of {len(metrics_to_calculate)} metrics across {len(checkpoints)} checkpoints{label}")
    t.secho(f"\nCalculating {len(metrics_to_calculate)} metrics across {len(checkpoints)} checkpoints{label}...", 
           fg=t.colors.BLUE, bold=True)
    
    loading_animation = SimpleLoadingAnimation(
        base_text=f"Calculating metrics{label}",
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
            model_path=model_path,
            class_name=class_name
        )
        loading_animation.stop(f"Successfully computed {len(completed_metrics.keys())} metrics{label}.")
        return metrics_data
    except Exception as e:
        loading_animation.stop(f"Error calculating metrics: {str(e)}")
        raise


def _validate_device(device: str) -> str:
    """Validate and return the device to use."""
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        t.secho("WARNING: CUDA requested but not available. Falling back to CPU.", 
               fg=t.colors.YELLOW, bold=True)
        return "cpu"
    return device


def _load_model_interactively(model_label: str) -> dict:
    """Interactive model loading with error handling and retry."""
    while True:
        # Get directory path
        while True:
            dir_path = t.prompt(f"Enter directory path for {model_label} model").strip()
            if os.path.isdir(dir_path):
                break
            t.secho(f"'{dir_path}' is not a valid directory. Please try again.", fg=t.colors.RED)
        
        # Get model file path
        while True:
            model_path = t.prompt(f"Enter model file path for {model_label} model").strip()
            if os.path.isfile(model_path) and model_path.endswith('.py'):
                break
            if not os.path.isfile(model_path):
                t.secho(f"'{model_path}' is not a valid file. Please try again.", fg=t.colors.RED)
            elif not model_path.endswith('.py'):
                t.secho(f"'{model_path}' is not a Python file. Please provide a .py file.", fg=t.colors.RED)
        
        # Auto-detect or ask for class name
        try:
            available_classes = get_model_classes(model_path)
            if len(available_classes) == 0:
                t.secho(f"No nn.Module subclasses found in {model_path}. Please check the file and try again.", 
                       fg=t.colors.RED)
                continue
            elif len(available_classes) == 1:
                class_name = available_classes[0]
                logger.info(f"Auto-detected class: {class_name}")
                t.secho(f"Auto-detected class: {class_name}", fg=t.colors.BLUE, bold=True)
            else:
                t.secho(f"Multiple model classes found: {', '.join(available_classes)}", fg=t.colors.YELLOW)
                while True:
                    class_name = t.prompt(f"Enter class name for {model_label} model").strip()
                    if class_name in available_classes:
                        break
                    t.secho(f"'{class_name}' not found in available classes. Please choose from: {', '.join(available_classes)}", 
                           fg=t.colors.RED)
        except Exception as e:
            logger.warning(f"Could not auto-detect classes: {e}")
            class_name = t.prompt(f"Enter class name for {model_label} model").strip()
        
        # Try to load model class
        try:
            with suppress_stdout_stderr():
                model_class = load_model_class(model_path, class_name)
        except Exception as e:
            t.secho(f"Failed to load model class '{class_name}' from '{model_path}': {str(e)}", fg=t.colors.RED)
            t.secho("Please check your inputs and try again.", fg=t.colors.YELLOW)
            continue
        
        # Find checkpoints
        try:
            checkpoints = contains_checkpoints(dir_path)
            logger.info(f"Found {len(checkpoints)} checkpoints for {model_label} model")
            t.secho(f"Found {len(checkpoints)} checkpoint(s) for {class_name}.", fg=t.colors.GREEN)
            
            return {
                "dir": dir_path,
                "model_path": model_path,
                "class_name": class_name,
                "checkpoints": checkpoints
            }
        except Exception as e:
            t.secho(f"Error loading checkpoints from '{dir_path}': {str(e)}", fg=t.colors.RED)
            t.secho("Please check the directory and try again.", fg=t.colors.YELLOW)
            continue


@app.command()
def load_dir(
    dir: str = t.Option(..., help="Directory containing model checkpoints"),
    model: str = t.Option(..., help="Path to Python file defining the model class"),
    class_name: str = t.Option(None, help="Name of the model class (auto-detected if not provided)")
):
    logger.info(f"Loading directory: {dir}")
    logger.info(f"Model file: {model}")

    class_name = _select_class_name(model, class_name)
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
def plot_metrics(
    device: str = t.Option("cuda", help="Device to use for calculations ('cuda', 'cpu', specific 'cuda:n')"),
    parallel: bool = t.Option(True, help="Use parallel processing for calculations (by default: True)") 
):   
    logger.info(f"Starting plot-metric command with device={device}, parallel={parallel}")   
    try:
        state = load_state() 
        logger.info("State loaded successfully")
    except RuntimeError as e:
        logger.error(f"Failed to load state: {str(e)}")
        t.secho(str(e), fg=t.colors.RED, bold=True)
        raise t.Exit(code=1)
    
    device = _validate_device(device)
    if device == "cpu":
        logger.info("Running on CPU")
        t.secho("Running on CPU. Consider using CUDA for large models or many checkpoints.", 
               fg=t.colors.YELLOW, bold=True)
    else:
        t.secho("Running on CUDA", fg=t.colors.GREEN, bold=True)
        logger.info(f"Running on {device}")
    
    with suppress_stdout_stderr():
        load_model_class(state["model_path"], state["class_name"])

    checkpoints = state["checkpoints"]
    metrics_to_calculate, metrics_file = _select_metrics()
    
    try:
        metrics_data = _compute_metrics_with_animation(
            metrics_to_calculate,
            checkpoints,
            device,
            parallel,
            metrics_file,
            state["model_path"],
            state["class_name"]
        )
    except Exception as e:
        _err(f"Error calculating metrics: {str(e)}")
    
    with suppress_stdout_stderr():
        clear_model_cache()
        clear_metric_cache()
    state["metrics_data"] = metrics_data
    save_state(state)
    logger.info("Saved metrics data to state file.")

    checkpoint_names = [os.path.basename(p) for p in checkpoints]
    t.secho("\nOpening visualization. In the plot, you can:", fg=t.colors.CYAN)
    t.secho("- Use the dropdown menu to toggle between metrics or show all at once", fg=t.colors.CYAN)
    t.secho("- Interact with the data points for detailed information", fg=t.colors.CYAN)
    t.secho("- Download the plot as PNG using the button in the top-right", fg=t.colors.CYAN)
    t.secho("- Export the data as a CSV for further analysis", fg=t.colors.CYAN)
    
    try:
        logger.info("Trying to create plot...")
        with suppress_stdout_stderr():
            plot_metric_interactive(
                checkpoint_names=checkpoint_names,
                metrics_data=metrics_data,
                many_metrics=len(metrics_to_calculate) > 7
            )
        logger.info("Successfully created plot")
    except Exception as e:
        _err(f"Error generating plot: {str(e)}")


@app.command()
def compare_models(
    device: str = t.Option("cuda", help="Device to use for calculations ('cuda', 'cpu', specific 'cuda:n')"),
    parallel: bool = t.Option(True, help="Use parallel processing for calculations (by default: True)")
):
    """Compare metrics between multiple models (up to 3)."""
    logger.info(f"Starting compare-models command with device={device}, parallel={parallel}")
    
    device = _validate_device(device)
    if device == "cpu":
        logger.info("Running on CPU")
        t.secho("Running on CPU. Consider using CUDA for large models or many checkpoints.", 
               fg=t.colors.YELLOW, bold=True)
    else:
        t.secho("Running on CUDA", fg=t.colors.GREEN, bold=True)
        logger.info(f"Running on {device}")
    
    t.secho("\n=== Model Comparison Mode ===", fg=t.colors.CYAN, bold=True)
    
    # Ask how many models to compare
    while True:
        num_models_str = t.prompt("How many models would you like to compare? (2-3)")
        try:
            num_models = int(num_models_str)
            if 2 <= num_models <= 3:
                break
            t.secho("Please enter a number between 2 and 3.", fg=t.colors.RED)
        except ValueError:
            t.secho("Please enter a valid number.", fg=t.colors.RED)
    
    t.secho(f"You will load {num_models} models to compare their metrics.", fg=t.colors.BLUE)
    
    # Load all models
    models_data = []
    model_labels = ["first", "second", "third"]
    
    for i in range(num_models):
        t.secho(f"\n--- Loading {model_labels[i].capitalize()} Model ---", fg=t.colors.MAGENTA, bold=True)
        model_data = _load_model_interactively(model_labels[i])
        models_data.append(model_data)
    
    # Select metrics (limit to 5)
    t.secho("\n--- Select Metrics to Compare ---", fg=t.colors.MAGENTA, bold=True)
    t.secho("You can select up to 5 metrics for comparison.", fg=t.colors.YELLOW)
    metrics_to_calculate, metrics_file = _select_metrics(max_metrics=5)
    
    # Calculate metrics for all models
    t.secho(f"\n--- Calculating Metrics for All {num_models} Models ---", fg=t.colors.MAGENTA, bold=True)
    
    all_metrics_data = []
    try:
        for i, model_data in enumerate(models_data):
            metrics_data = _compute_metrics_with_animation(
                metrics_to_calculate,
                model_data['checkpoints'],
                device,
                parallel,
                metrics_file,
                model_data['model_path'],
                model_data['class_name'],
                model_data['class_name']
            )
            all_metrics_data.append(metrics_data)
    except Exception as e:
        _err(f"Error calculating metrics: {str(e)}")
    
    # Clear caches
    with suppress_stdout_stderr():
        clear_model_cache()
        clear_metric_cache()
    
    # Prepare comparison data
    comparison_data = {}
    model_keys = ["model1", "model2", "model3"]
    
    for i, (model_data, metrics_data) in enumerate(zip(models_data, all_metrics_data)):
        checkpoint_names = [os.path.basename(p) for p in model_data['checkpoints']]
        comparison_data[model_keys[i]] = {
            "name": model_data['class_name'],
            "checkpoints": checkpoint_names,
            "metrics": metrics_data
        }
    
    # Use first model's data for the base metrics structure
    combined_metrics = all_metrics_data[0]
    checkpoint_names = [os.path.basename(p) for p in models_data[0]['checkpoints']]
    
    t.secho("\nOpening comparison visualization. In the plot, you can:", fg=t.colors.CYAN)
    t.secho(f"- See metrics for all {num_models} models with different line styles", fg=t.colors.CYAN)
    t.secho("- Use all the same features as single model visualization", fg=t.colors.CYAN)
    t.secho("- Toggle between metrics, overlay mode, and phase transitions", fg=t.colors.CYAN)
    t.secho("- Export comparison data as CSV", fg=t.colors.CYAN)
    
    try:
        logger.info(f"Creating comparison plot for {num_models} models...")
        with suppress_stdout_stderr():
            # Use the same plot_metric_interactive but with comparison data
            plot_metric_interactive(
                checkpoint_names=checkpoint_names,  # Primary model checkpoints
                metrics_data=combined_metrics,
                many_metrics=len(metrics_to_calculate) > 3,
                comparison_mode=True,
                comparison_data=comparison_data,
                num_models=num_models
            )
        logger.info("Successfully created comparison plot")
    except Exception as e:
        _err(f"Error generating comparison plot: {str(e)}")


def _err(msg: str) -> None:
    """Log and raise errors."""
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