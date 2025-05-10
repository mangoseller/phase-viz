import typer

app = typer.Typer()
t = typer  

@app.command()
def main():
    title = t.style("phase-viz", fg=t.colors.CYAN, bold=True)
    subtitle = t.style("Visualize the developmental trajectory of a neural network/", fg=t.colors.MAGENTA)
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
def load_dir():
    pass

if __name__ == "__main__":
    app()
