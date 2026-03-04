"""Topic modeling commands — `mind tm {train, init-config}`."""
import pathlib
import time
from typing import Optional

import typer

from mind.cli._config_loader import load_config, validate_config_keys, DEFAULT_SYSTEM_CONFIG
from mind.cli._console import (
    console, print_header, print_success, print_error, print_warning,
    print_config_table, spinner,
)

app = typer.Typer(no_args_is_help=True)

# --------------------------------------------------------------------------- #
#  YAML template — tm section only
# --------------------------------------------------------------------------- #
_TM_CONFIG_TEMPLATE = """\
# ---------------------------------------------------------------
# MIND Topic Modeling Configuration Template
# ---------------------------------------------------------------
# Use with:  mind tm train --config this_file.yaml
# ---------------------------------------------------------------

tm:
  train:
    input: data/processed/prepared.parquet
    lang1: EN
    lang2: DE
    model_folder: data/models/tm_ende
    num_topics: 30
    alpha: 1.0
"""


# --------------------------------------------------------------------------- #
#  Commands
# --------------------------------------------------------------------------- #

@app.command("init-config")
def init_config():
    """Print a commented YAML tm-section config template to stdout.

    Usage:  mind tm init-config > my_run.yaml
    """
    typer.echo(_TM_CONFIG_TEMPLATE)


@app.command("train")
def train(
    config: Optional[pathlib.Path] = typer.Option(None, "--config", "-c", help="Path to run config YAML."),
    system_config: pathlib.Path = typer.Option(DEFAULT_SYSTEM_CONFIG, "--system-config", help="Path to system config."),
    input: Optional[str] = typer.Option(None, "--input", "-i", help="Path to prepared parquet file."),
    lang1: Optional[str] = typer.Option(None, "--lang1", help="First language (e.g. EN)."),
    lang2: Optional[str] = typer.Option(None, "--lang2", help="Second language (e.g. DE)."),
    model_folder: Optional[str] = typer.Option(None, "--model-folder", help="Directory to save trained model."),
    num_topics: Optional[int] = typer.Option(None, "--num-topics", help="Number of topics."),
    alpha: Optional[float] = typer.Option(None, "--alpha", help="Dirichlet alpha parameter."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip overwrite confirmation."),
):
    """Train a Polylingual Topic Model."""
    try:
        cli_overrides: dict = {}
        tm: dict = {}
        if input is not None:
            tm["input"] = input
        if lang1 is not None:
            tm["lang1"] = lang1
        if lang2 is not None:
            tm["lang2"] = lang2
        if model_folder is not None:
            tm["model_folder"] = model_folder
        if num_topics is not None:
            tm["num_topics"] = num_topics
        if alpha is not None:
            tm["alpha"] = alpha
        if tm:
            cli_overrides["tm"] = {"train": tm}

        cfg = load_config(
            run_config_path=config,
            system_config_path=system_config,
            cli_overrides=cli_overrides if cli_overrides else None,
        )

        for w in validate_config_keys(cfg):
            print_warning(w)

        tm_cfg = cfg.get("tm", {}).get("train", {})
        resolved_input = tm_cfg.get("input")
        resolved_lang1 = tm_cfg.get("lang1")
        resolved_lang2 = tm_cfg.get("lang2")
        resolved_model_folder = tm_cfg.get("model_folder")
        resolved_num_topics = tm_cfg.get("num_topics")
        resolved_alpha = tm_cfg.get("alpha", 1.0)

        if not all([resolved_input, resolved_lang1, resolved_lang2, resolved_model_folder, resolved_num_topics]):
            print_error("'tm.train.{input, lang1, lang2, model_folder, num_topics}' are all required.")
            raise typer.Exit(code=1)

        # Early path validation
        if not pathlib.Path(resolved_input).exists():
            print_error(f"Input file not found: {resolved_input}")
            raise typer.Exit(code=1)

        # Overwrite confirmation for model folder
        mf = pathlib.Path(resolved_model_folder)
        if mf.exists() and any(mf.iterdir()):
            if not yes and not typer.confirm(
                f"Model folder '{resolved_model_folder}' already contains files. Overwrite?",
                default=False,
            ):
                raise typer.Exit()

        print_header("MIND Topic Modeling — Train")
        print_config_table({
            "Input": resolved_input,
            "Language 1": resolved_lang1,
            "Language 2": resolved_lang2,
            "Model folder": resolved_model_folder,
            "Num topics": resolved_num_topics,
            "Alpha": resolved_alpha,
        })

        from mind.topic_modeling.polylingual_tm import PolylingualTM

        start = time.time()
        with spinner("Training topic model…"):
            ptm = PolylingualTM(
                lang1=resolved_lang1,
                lang2=resolved_lang2,
                model_folder=pathlib.Path(resolved_model_folder),
                num_topics=resolved_num_topics,
                alpha=resolved_alpha,
            )
            ptm.train(df_path=pathlib.Path(resolved_input))
        elapsed = time.time() - start

        print_success(f"Topic model trained in {elapsed:.1f}s — model saved to {resolved_model_folder}")

    except (KeyboardInterrupt, SystemExit):
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)
    except typer.Exit:
        raise
    except Exception as exc:
        print_error(f"Topic model training failed: {exc}")
        console.print_exception()
        raise typer.Exit(code=1)
