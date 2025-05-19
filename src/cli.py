import typer as t # type: ignore
import typing
import os
import sys
import importlib.util
import inspect
from pathlib import Path
sys.path.append(os.path.dirname(__file__))
from loader import load_model_class, contains_checkpoints
from state import load_state, save_state
from metrics import (
    l2_norm_of_model,
    compute_metric_over_checkpoints,
    import_metric_functions
)
from generate_plot import plot_metric_interactive

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
def load_dir( #TODO: test with differing architecture
    dir: str = t.Option(..., help="Directory containing model checkpoints"),
    model: str = t.Option(..., help="Path to Python file defining the model class"),
    class_name: str = t.Option(..., help="Name of the model class inside the file")
):
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
def plot_metric():
    try:
        state = load_state() # load json state file if it exists
    except RuntimeError as e:
        t.secho(str(e), fg=t.colors.RED, bold=True)
        raise t.Exit(code=1)
    
    load_model_class(state["model_path"], state["class_name"])
    checkpoints = state["checkpoints"]
    
    # Dict to store metrics: {metric_name: [values]}
    metrics_data = {}
    
    # Loop to collect multiple metrics
    while True:
        raw = t.prompt(
            "Enter metric name (e.g. 'l2'), path to a custom .py file, or 'done' to finish selecting metrics"
        ).strip()
        
        if raw.lower() == 'done':
            if not metrics_data:
                t.secho("No metrics selected. Please select at least one metric.", fg=t.colors.RED)
                continue
            break
            
        if os.path.isfile(raw):
            path = Path(raw).resolve()
            if path.suffix != ".py":
                _err(f"Custom metric must be a .py file (got {path})")
                
            # Import all metric functions from the file
            try:
                metric_functions = import_metric_functions(str(path))
                if not metric_functions:
                    _err(f"No valid metric functions found in {path}")
                
                # Compute each metric from the file
                for metric_name, metric_fn in metric_functions.items():
                    if metric_name in metrics_data:
                        t.secho(f"Metric '{metric_name}' already added. Skipping.", fg=t.colors.YELLOW)
                        continue
                    
                    t.secho(f"Computing {metric_name}...", fg=t.colors.BLUE)
                    values = compute_metric_over_checkpoints(metric_fn, checkpoints)
                    metrics_data[metric_name] = values
                    t.secho(f"Added metric: {metric_name}", fg=t.colors.GREEN)
            except Exception as e:
                _err(f"Failed importing metric functions from {path}: {e}")
        else:
            # Look for built-in metrics
            match raw.lower():
                case "l2":
                    if "L2 Norm" in metrics_data:
                        t.secho("Metric 'L2 Norm' already added. Skipping.", fg=t.colors.YELLOW)
                        continue
                    
                    t.secho("Computing L2 Norm...", fg=t.colors.BLUE)
                    values = compute_metric_over_checkpoints(l2_norm_of_model, checkpoints)
                    metrics_data["L2 Norm"] = values
                    t.secho("Added metric: L2 Norm", fg=t.colors.GREEN)
                case _:
                    t.secho("Could not locate metric/file.", fg=t.colors.RED)
                    continue
    
    # Save all metrics data for later use
    state["metrics_data"] = metrics_data
    save_state(state)
    
    # Plot all metrics using the enhanced plot function
    checkpoint_names = [os.path.basename(p) for p in checkpoints]
    
    t.secho(f"Plotting {len(metrics_data)} metrics...", fg=t.colors.BLUE)
    plot_metric_interactive(
        checkpoint_names=checkpoint_names,
        metrics_data=metrics_data
    )
    
    t.secho(f"Plot generated with {len(metrics_data)} metrics.", fg=t.colors.GREEN)
    t.secho("Use the dropdown menu to toggle between metrics or show all at once.", fg=t.colors.CYAN)


def _err(msg: str) -> None:  
    t.secho(msg, fg=t.colors.RED, bold=True)
    raise t.Exit(code=1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        welcome()
    else:
        app()

