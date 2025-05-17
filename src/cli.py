import typer as t
import typing
import os
import sys
sys.path.append(os.path.dirname(__file__))
import sys
from loader import load_model_class, contains_checkpoints
from state import load_state, save_state
from metrics import compute_l2_from_checkpoint, plot_metric_over_checkpoints
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
def plot_norm():
    try:
        state = load_state()
    except RuntimeError as e:
        t.secho(f"{str(e)}", fg=t.colors.RED, bold=True)
        raise t.Exit(code=1)

    model_class = load_model_class(state["model_path"], state["class_name"])
    checkpoints = state["checkpoints"]

    norms = []
    for path in checkpoints:
        norm = compute_l2_from_checkpoint(path)
        norms.append(norm)

    plot_metric_over_checkpoints(
        checkpoint_names=[os.path.basename(p) for p in checkpoints],
        values=norms,
        metric_name="L2 Norm"
    )



if __name__ == "__main__":
    if len(sys.argv) == 1:
        welcome()
    else:
        app()
