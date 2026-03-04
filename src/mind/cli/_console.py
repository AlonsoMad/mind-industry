"""Shared Rich console and output helpers."""
import json as _json
from contextlib import contextmanager

from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Verbosity control
# ---------------------------------------------------------------------------
# 0 = quiet (errors only), 1 = normal, 2 = verbose
_verbosity: int = 1


def set_verbosity(level: int) -> None:
    """Set global output verbosity (0=quiet, 1=normal, 2=verbose)."""
    global _verbosity
    _verbosity = level


def is_quiet() -> bool:
    return _verbosity == 0


def is_verbose() -> bool:
    return _verbosity >= 2


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    """Print a styled section header (suppressed in quiet mode)."""
    if not is_quiet():
        console.rule(f"[bold cyan]{title}[/bold cyan]")


def print_success(message: str) -> None:
    """Print a success message with a checkmark."""
    if not is_quiet():
        console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Print an error message with an X mark (always shown)."""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message (suppressed in quiet mode)."""
    if not is_quiet():
        console.print(f"[bold yellow]⚠[/bold yellow]  {message}")


def print_config_table(config: dict, title: str = "Configuration") -> None:
    """Render a key-value config dict as a Rich table (suppressed in quiet mode)."""
    if is_quiet():
        return
    table = Table(title=title, show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")
    for key, value in config.items():
        table.add_row(str(key), str(value))
    console.print(table)


def print_json_result(data: dict) -> None:
    """Emit a JSON result to stdout (bypasses verbosity — always shown)."""
    console.print_json(_json.dumps(data, default=str))


# ---------------------------------------------------------------------------
# Progress spinner
# ---------------------------------------------------------------------------

@contextmanager
def spinner(description: str):
    """Context manager that shows a Rich spinner, or is a no-op in quiet mode."""
    if is_quiet():
        yield
    else:
        with console.status(f"[bold cyan]{description}[/bold cyan]", spinner="dots"):
            yield
