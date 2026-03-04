"""Data preprocessing commands — `mind data {segment, translate, prepare, init-config}`."""
import json
import time
from pathlib import Path
from typing import Optional

import typer

from mind.cli._config_loader import load_config, validate_config_keys, DEFAULT_SYSTEM_CONFIG
from mind.cli._console import (
    console, print_header, print_success, print_error, print_warning,
    print_config_table, spinner,
)

app = typer.Typer(no_args_is_help=True)

# --------------------------------------------------------------------------- #
#  YAML template — data section only
# --------------------------------------------------------------------------- #
_DATA_CONFIG_TEMPLATE = """\
# ---------------------------------------------------------------
# MIND Data Preprocessing Configuration Template
# ---------------------------------------------------------------
# Use with:  mind data segment --config this_file.yaml
#            mind data translate --config this_file.yaml
#            mind data prepare --config this_file.yaml
# ---------------------------------------------------------------

data:
  segment:
    input: data/raw/documents.parquet
    output: data/processed/segmented.parquet
    text_col: text
    min_length: 100
    separator: "\\n"
  translate:
    input: data/processed/segmented.parquet
    output: data/processed/translated.parquet
    src_lang: en
    tgt_lang: de
    text_col: text
    lang_col: lang
  prepare:
    anchor: data/processed/segmented.parquet
    comparison: data/processed/translated.parquet
    output: data/processed/prepared.parquet
    schema:
      chunk_id: id_preproc
      text: text
      lang: lang
      full_doc: full_doc
      doc_id: doc_id
"""

_REQUIRED_SCHEMA_KEYS = {"chunk_id", "text", "lang", "full_doc", "doc_id"}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _prompt_overwrite(path: str, yes: bool) -> None:
    """Prompt before overwriting an existing file."""
    p = Path(path)
    if p.exists():
        if yes:
            return
        if not typer.confirm(f"Output file '{path}' already exists. Overwrite?", default=False):
            raise typer.Exit()


def _validate_input_path(path: str, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _parse_schema(schema_str: str) -> dict:
    """Parse a schema from a JSON string, YAML string, or file path."""
    # Try as file path first
    if Path(schema_str).exists():
        p = Path(schema_str)
        if p.suffix in (".json",):
            with open(p) as f:
                return json.load(f)
        else:
            import yaml
            with open(p) as f:
                return yaml.safe_load(f)
    # Try JSON parse
    try:
        result = json.loads(schema_str)
    except json.JSONDecodeError:
        import yaml
        try:
            result = yaml.safe_load(schema_str)
        except Exception:
            raise ValueError(f"Could not parse --schema as JSON or YAML: {schema_str!r}")
    if not isinstance(result, dict):
        raise ValueError("--schema must be a key-value mapping.")
    missing = _REQUIRED_SCHEMA_KEYS - result.keys()
    if missing:
        raise ValueError(f"--schema is missing required keys: {sorted(missing)}")
    return result


# --------------------------------------------------------------------------- #
#  Commands
# --------------------------------------------------------------------------- #

@app.command("init-config")
def init_config():
    """Print a commented YAML data-section config template to stdout.

    Usage:  mind data init-config > my_run.yaml
    """
    typer.echo(_DATA_CONFIG_TEMPLATE)


@app.command("segment")
def segment(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to run config YAML."),
    system_config: Path = typer.Option(DEFAULT_SYSTEM_CONFIG, "--system-config", help="Path to system config."),
    input: Optional[str] = typer.Option(None, "--input", "-i", help="Path to input parquet file."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Path to save segmented output."),
    text_col: Optional[str] = typer.Option(None, "--text-col", help="Name of text column."),
    min_length: Optional[int] = typer.Option(None, "--min-length", help="Minimum paragraph length (chars)."),
    separator: Optional[str] = typer.Option(None, "--separator", help="Paragraph separator string."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip overwrite confirmation."),
):
    """Segment documents into passages/paragraphs."""
    try:
        cli_overrides: dict = {}
        seg: dict = {}
        if input is not None:
            seg["input"] = input
        if output is not None:
            seg["output"] = output
        if text_col is not None:
            seg["text_col"] = text_col
        if min_length is not None:
            seg["min_length"] = min_length
        if separator is not None:
            seg["separator"] = separator
        if seg:
            cli_overrides["data"] = {"segment": seg}

        cfg = load_config(
            run_config_path=config,
            system_config_path=system_config,
            cli_overrides=cli_overrides if cli_overrides else None,
        )

        for w in validate_config_keys(cfg):
            print_warning(w)

        seg_cfg = cfg.get("data", {}).get("segment", {})
        resolved_input = seg_cfg.get("input")
        resolved_output = seg_cfg.get("output")
        if not resolved_input or not resolved_output:
            print_error("'data.segment.input' and 'data.segment.output' are required.")
            raise typer.Exit(code=1)

        # Early path validation
        _validate_input_path(resolved_input, "data.segment.input")

        resolved_text_col = seg_cfg.get("text_col", "text")
        resolved_min_length = seg_cfg.get("min_length", 100)
        resolved_separator = seg_cfg.get("separator", "\n")

        _prompt_overwrite(resolved_output, yes)

        print_header("MIND Data — Segment")
        print_config_table({
            "Input": resolved_input,
            "Output": resolved_output,
            "Text column": resolved_text_col,
            "Min length": resolved_min_length,
            "Separator": repr(resolved_separator),
        })

        from mind.corpus_building.segmenter import Segmenter

        start = time.time()
        with spinner("Segmenting documents…"):
            segmenter = Segmenter(config_path=system_config)
            result_path = segmenter.segment(
                path_df=Path(resolved_input),
                path_save=Path(resolved_output),
                text_col=resolved_text_col,
                min_length=resolved_min_length,
                sep=resolved_separator,
            )
        elapsed = time.time() - start

        import pandas as pd
        row_count = len(pd.read_parquet(result_path))
        print_success(f"Segmentation complete in {elapsed:.1f}s — {row_count} rows → {resolved_output}")

    except (KeyboardInterrupt, SystemExit):
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)
    except typer.Exit:
        raise
    except Exception as exc:
        print_error(f"Segmentation failed: {exc}")
        console.print_exception()
        raise typer.Exit(code=1)


@app.command("translate")
def translate(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to run config YAML."),
    system_config: Path = typer.Option(DEFAULT_SYSTEM_CONFIG, "--system-config", help="Path to system config."),
    input: Optional[str] = typer.Option(None, "--input", "-i", help="Path to input parquet file."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Path to save translated output."),
    src_lang: Optional[str] = typer.Option(None, "--src-lang", help="Source language code (e.g. en)."),
    tgt_lang: Optional[str] = typer.Option(None, "--tgt-lang", help="Target language code (e.g. de)."),
    text_col: Optional[str] = typer.Option(None, "--text-col", help="Name of text column."),
    lang_col: Optional[str] = typer.Option(None, "--lang-col", help="Name of language column."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip overwrite confirmation."),
):
    """Translate passages between languages."""
    try:
        cli_overrides: dict = {}
        tr: dict = {}
        if input is not None:
            tr["input"] = input
        if output is not None:
            tr["output"] = output
        if src_lang is not None:
            tr["src_lang"] = src_lang
        if tgt_lang is not None:
            tr["tgt_lang"] = tgt_lang
        if text_col is not None:
            tr["text_col"] = text_col
        if lang_col is not None:
            tr["lang_col"] = lang_col
        if tr:
            cli_overrides["data"] = {"translate": tr}

        cfg = load_config(
            run_config_path=config,
            system_config_path=system_config,
            cli_overrides=cli_overrides if cli_overrides else None,
        )

        for w in validate_config_keys(cfg):
            print_warning(w)

        tr_cfg = cfg.get("data", {}).get("translate", {})
        resolved_input = tr_cfg.get("input")
        resolved_output = tr_cfg.get("output")
        resolved_src = tr_cfg.get("src_lang")
        resolved_tgt = tr_cfg.get("tgt_lang")
        if not all([resolved_input, resolved_output, resolved_src, resolved_tgt]):
            print_error("'data.translate.{input, output, src_lang, tgt_lang}' are all required.")
            raise typer.Exit(code=1)

        _validate_input_path(resolved_input, "data.translate.input")

        resolved_text_col = tr_cfg.get("text_col", "text")
        resolved_lang_col = tr_cfg.get("lang_col", "lang")

        _prompt_overwrite(resolved_output, yes)

        print_header("MIND Data — Translate")
        print_config_table({
            "Input": resolved_input,
            "Output": resolved_output,
            "Source lang": resolved_src,
            "Target lang": resolved_tgt,
            "Text column": resolved_text_col,
            "Lang column": resolved_lang_col,
        })

        from mind.corpus_building.translator import Translator

        start = time.time()
        with spinner("Translating passages…"):
            translator = Translator(config_path=system_config)
            translated_df = translator.translate(
                path_df=Path(resolved_input),
                src_lang=resolved_src,
                tgt_lang=resolved_tgt,
                text_col=resolved_text_col,
                lang_col=resolved_lang_col,
                save_path=resolved_output,
            )
        elapsed = time.time() - start

        print_success(f"Translation complete in {elapsed:.1f}s — {len(translated_df)} rows → {resolved_output}")

    except (KeyboardInterrupt, SystemExit):
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)
    except typer.Exit:
        raise
    except Exception as exc:
        print_error(f"Translation failed: {exc}")
        console.print_exception()
        raise typer.Exit(code=1)


@app.command("prepare")
def prepare(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to run config YAML."),
    system_config: Path = typer.Option(DEFAULT_SYSTEM_CONFIG, "--system-config", help="Path to system config."),
    anchor: Optional[str] = typer.Option(None, "--anchor", help="Path to anchor language parquet."),
    comparison: Optional[str] = typer.Option(None, "--comparison", help="Path to comparison language parquet."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Path to save prepared output."),
    schema: Optional[str] = typer.Option(None, "--schema", help="JSON/YAML schema mapping, or path to schema file."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip overwrite confirmation."),
):
    """Prepare and merge datasets for the MIND pipeline."""
    try:
        cli_overrides: dict = {}
        pr: dict = {}
        if anchor is not None:
            pr["anchor"] = anchor
        if comparison is not None:
            pr["comparison"] = comparison
        if output is not None:
            pr["output"] = output
        if schema is not None:
            try:
                pr["schema"] = _parse_schema(schema)
            except Exception as e:
                print_error(f"Failed to parse --schema: {e}")
                raise typer.Exit(code=1)
        if pr:
            cli_overrides["data"] = {"prepare": pr}

        cfg = load_config(
            run_config_path=config,
            system_config_path=system_config,
            cli_overrides=cli_overrides if cli_overrides else None,
        )

        for w in validate_config_keys(cfg):
            print_warning(w)

        pr_cfg = cfg.get("data", {}).get("prepare", {})
        resolved_anchor = pr_cfg.get("anchor")
        resolved_comparison = pr_cfg.get("comparison")
        resolved_output = pr_cfg.get("output")
        resolved_schema = pr_cfg.get("schema")
        if not all([resolved_anchor, resolved_comparison, resolved_output, resolved_schema]):
            print_error("'data.prepare.{anchor, comparison, output, schema}' are all required.")
            raise typer.Exit(code=1)

        # Validate schema keys when coming from YAML config
        if isinstance(resolved_schema, dict):
            missing = _REQUIRED_SCHEMA_KEYS - resolved_schema.keys()
            if missing:
                print_error(f"data.prepare.schema is missing required keys: {sorted(missing)}")
                raise typer.Exit(code=1)

        _validate_input_path(resolved_anchor, "data.prepare.anchor")
        _validate_input_path(resolved_comparison, "data.prepare.comparison")
        _prompt_overwrite(resolved_output, yes)

        print_header("MIND Data — Prepare")
        print_config_table({
            "Anchor": resolved_anchor,
            "Comparison": resolved_comparison,
            "Output": resolved_output,
            "Schema": resolved_schema,
        })

        from mind.corpus_building.data_preparer import DataPreparer

        start = time.time()
        with spinner("Merging and preparing datasets…"):
            preparer = DataPreparer(schema=resolved_schema)
            final_df = preparer.format_dataframes(
                anchor_path=Path(resolved_anchor),
                comparison_path=Path(resolved_comparison),
                path_save=Path(resolved_output),
            )
        elapsed = time.time() - start

        print_success(f"Preparation complete in {elapsed:.1f}s — {len(final_df)} rows → {resolved_output}")

    except (KeyboardInterrupt, SystemExit):
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)
    except typer.Exit:
        raise
    except Exception as exc:
        print_error(f"Data preparation failed: {exc}")
        console.print_exception()
        raise typer.Exit(code=1)
