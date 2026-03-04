"""MIND CLI — Unified command-line interface for the MIND pipeline."""
from typing import Optional

import typer

from mind.cli.commands import data, detect, tm
from mind.cli._console import set_verbosity

app = typer.Typer(
    name="mind",
    help="MIND — Multilingual Information Discrepancy Detection",
    invoke_without_command=True,
    rich_markup_mode="rich",
)

# Register subcommand groups
app.add_typer(data.app, name="data", help="Data preprocessing (segment, translate, prepare)")
app.add_typer(detect.app, name="detect", help="Discrepancy detection pipeline")
app.add_typer(tm.app, name="tm", help="Topic modeling")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Show version and exit."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress all output except errors."),
):
    """MIND — Multilingual Information Discrepancy Detection CLI."""
    if verbose and quiet:
        typer.echo("Error: --verbose and --quiet are mutually exclusive.", err=True)
        raise typer.Exit(code=1)
    if quiet:
        set_verbosity(0)
    elif verbose:
        set_verbosity(2)
    else:
        set_verbosity(1)

    if version:
        from importlib.metadata import version as get_version
        try:
            typer.echo(f"mind {get_version('mind')}")
        except Exception:
            typer.echo("mind (version unknown — not installed via pip)")
        raise typer.Exit()
    # If no subcommand was given, show help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def entrypoint():
    """Console script entry point."""
    app()


if __name__ == "__main__":
    entrypoint()
