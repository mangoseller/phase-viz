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
    plot_metric_over_checkpoints,
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

    raw = t.prompt(
        "Enter metric name (e.g. 'l2') or the path to a custom .py file"
    ).strip()

    metric_fn = None
    metric_name = None

    if os.path.isfile(raw):
        path = Path(raw).resolve()
        if path.suffix != ".py":
            _err(f"Custom metric must be a .py file (got {path})")

        spec = importlib.util.spec_from_file_location("custom_metric", str(path)) # get module
        if spec is None or spec.loader is None:
            _err(f"Could not import module from {path}")

        mod = importlib.util.module_from_spec(spec) # type: ignore
        sys.modules["custom_metric"] = mod
        try:
            spec.loader.exec_module(mod) # import module
        except Exception as e:
            _err(f"Failed importing {path}: {e}")

        # get functions that end with _of_model from module, e.g. compute_LLC_of_model()
        cands = [
            fn
            for fn in mod.__dict__.values()
            if inspect.isfunction(fn)
            and fn.__module__ == mod.__name__
            and fn.__name__.endswith("_of_model")
        ]
        if len(cands) != 1: # only one function per file - TODO: compute them in sequence, allow for big files of funcs
            _err(
                "File must define exactly one top-level function whose name "
                "ends with '_of_model'."
            )

        metric_fn = cands[0]
        # metric func must have 1 parameter
        if len(inspect.signature(metric_fn).parameters) != 1:
            _err(
                f"{metric_fn.__qualname__} must take exactly one argument "
                "(the model)."
            )

        metric_name = metric_fn.__name__.replace("_of_model", "").replace("_", " ").title() # get a title

    else:
        # compute inbuilt-funcs
        match raw.lower():
            case "l2":
                metric_fn = l2_norm_of_model
                metric_name = "L2 Norm"
            case _:
                _err(f"'{raw}' is not a built-in metric and is not a file.")

    values = compute_metric_over_checkpoints(metric_fn, checkpoints)
    plot_metric_over_checkpoints(
        checkpoint_names=[os.path.basename(p) for p in checkpoints],
        values=values,
        metric_name=metric_name,
    )


def _err(msg: str) -> None:  
    t.secho(msg, fg=t.colors.RED, bold=True)
    raise t.Exit(code=1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        welcome()
    else:
        app()
