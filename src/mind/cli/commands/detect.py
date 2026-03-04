"""Detection commands — `mind detect {run, init-config, validate-config}`."""
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import typer

from mind.cli._config_loader import load_config, validate_config_keys, DEFAULT_SYSTEM_CONFIG
from mind.cli._console import (
    console, print_header, print_success, print_error, print_warning,
    print_config_table, print_json_result, spinner, is_quiet,
)

app = typer.Typer(no_args_is_help=True)

# --------------------------------------------------------------------------- #
#  YAML templates
# --------------------------------------------------------------------------- #
_DETECT_CONFIG_TEMPLATE = """\
# ---------------------------------------------------------------
# MIND Detection Run Configuration Template
# ---------------------------------------------------------------
# Use with:  mind detect run --config this_file.yaml
# Values here override config/config.yaml (system defaults).
# CLI flags and MIND_* env vars override both.
# ---------------------------------------------------------------

# Override system LLM if needed (optional)
# llm:
#   default:
#     backend: ollama
#     model: llama3.3:70b

detect:
  topics: [7, 15]
  sample_size: 200
  path_save: data/mind_runs/ende/results
  dry_run: false
  no_entailment: false
  source:
    corpus_path: data/corpora/polylingual_df.parquet
    thetas_path: data/corpora/thetas_EN.npz
    id_col: doc_id          # default
    passage_col: text       # default
    full_doc_col: full_doc  # default
    lang_filter: EN         # required — e.g. EN, ES, DE
    filter_ids_path: null
    previous_check: null
  target:
    corpus_path: data/corpora/polylingual_df.parquet
    thetas_path: data/corpora/thetas_DE.npz
    id_col: doc_id
    passage_col: text
    full_doc_col: full_doc
    lang_filter: DE         # required — e.g. DE, ES, FR
    index_path: data/mind_runs/ende/indexes
    filter_ids_path: null
  load_thetas: true
"""

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _comma_separated_ints(value: str) -> List[int]:
    """Parse a comma-separated string of ints, e.g. '7,15'."""
    try:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError:
        raise typer.BadParameter("Topics must be comma-separated integers.")


def _read_filter_ids(path: Optional[str]) -> Optional[List[str]]:
    """Read filter IDs from a file (one per line), or return None."""
    if not path:
        return None
    try:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Filter IDs file not found: {path}")


def _env_override(key: str, cast=str, default=None):
    """Read a MIND_* environment variable with optional type cast."""
    val = os.environ.get(f"MIND_{key}")
    if val is None:
        return default
    try:
        return cast(val)
    except (ValueError, TypeError):
        print_warning(f"Ignoring invalid env var MIND_{key}={val!r}")
        return default


def _prompt_overwrite(path: str, yes: bool) -> None:
    """Prompt the user to confirm overwriting an existing non-empty directory."""
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        if yes:
            return
        if not typer.confirm(f"Output directory '{path}' already contains files. Overwrite?", default=False):
            raise typer.Exit()


def _validate_paths(paths: list[tuple[str, str]]) -> None:
    """Validate that required file paths exist. paths = [(value, label), ...]"""
    for value, label in paths:
        if value and not Path(value).exists():
            raise FileNotFoundError(f"{label} not found: {value}")


# --------------------------------------------------------------------------- #
#  Commands
# --------------------------------------------------------------------------- #

@app.command("init-config")
def init_config():
    """Print a commented YAML run-config template to stdout.

    Usage:  mind detect init-config > my_run.yaml
    """
    typer.echo(_DETECT_CONFIG_TEMPLATE)


@app.command("validate-config")
def validate_config(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to run config YAML."),
    system_config: Path = typer.Option(DEFAULT_SYSTEM_CONFIG, "--system-config", help="Path to system config."),
):
    """Validate a detection config file without running the pipeline.

    Checks that all required fields are present and referenced files exist.
    """
    try:
        cfg = load_config(run_config_path=config, system_config_path=system_config)

        # Key typo warnings
        for warning in validate_config_keys(cfg):
            print_warning(warning)

        detect_cfg = cfg.get("detect")
        errors = []

        if not detect_cfg:
            errors.append("Missing 'detect' section.")
        else:
            if not detect_cfg.get("topics"):
                errors.append("Missing 'detect.topics'.")
            if not detect_cfg.get("path_save"):
                errors.append("Missing 'detect.path_save'.")

            src = detect_cfg.get("source", {})
            tgt = detect_cfg.get("target", {})

            for key, label in [
                ("corpus_path", "detect.source.corpus_path"),
                ("thetas_path", "detect.source.thetas_path"),
            ]:
                val = src.get(key)
                if not val:
                    errors.append(f"Missing '{label}'.")
                elif not Path(val).exists():
                    errors.append(f"File not found: {label} = {val}")

            for key, label in [
                ("corpus_path", "detect.target.corpus_path"),
                ("thetas_path", "detect.target.thetas_path"),
            ]:
                val = tgt.get(key)
                if not val:
                    errors.append(f"Missing '{label}'.")
                elif not Path(val).exists():
                    errors.append(f"File not found: {label} = {val}")

            if not src.get("lang_filter"):
                errors.append("Missing 'detect.source.lang_filter' (e.g. EN).")
            if not tgt.get("lang_filter"):
                errors.append("Missing 'detect.target.lang_filter' (e.g. DE).")

        if errors:
            print_header("Config Validation — FAILED")
            for e in errors:
                print_error(e)
            raise typer.Exit(code=1)
        else:
            print_header("Config Validation — OK")
            print_success(f"Config is valid. Topics: {detect_cfg.get('topics')}")

    except (KeyboardInterrupt, SystemExit):
        raise
    except typer.Exit:
        raise
    except Exception as exc:
        print_error(f"Validation error: {exc}")
        raise typer.Exit(code=1)


@app.command("run")
def run(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to run config YAML."),
    system_config: Path = typer.Option(DEFAULT_SYSTEM_CONFIG, "--system-config", help="Path to system config."),
    topics: Optional[str] = typer.Option(None, "--topics", help="Comma-separated topic IDs, e.g. '7,15'. Env: MIND_TOPICS."),
    sample_size: Optional[int] = typer.Option(None, "--sample-size", help="Sample size for passage subsampling. Env: MIND_SAMPLE_SIZE."),
    path_save: Optional[str] = typer.Option(None, "--path-save", help="Directory to save results. Env: MIND_PATH_SAVE."),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="LLM model override. Env: MIND_LLM_MODEL."),
    llm_server: Optional[str] = typer.Option(None, "--llm-server", help="LLM server URL override. Env: MIND_LLM_SERVER."),
    dry_run: Optional[bool] = typer.Option(None, "--dry-run", help="Run without writing outputs. Env: MIND_DRY_RUN."),
    no_entailment: Optional[bool] = typer.Option(None, "--no-entailment", help="Disable entailment checking. Env: MIND_NO_ENTAILMENT."),
    print_config_flag: bool = typer.Option(False, "--print-config", help="Print resolved config and exit."),
    output_format: str = typer.Option("text", "--output-format", help="Output format: text or json."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts."),
):
    """Run the MIND discrepancy detection pipeline."""
    json_mode = output_format.lower() == "json"

    try:
        # --- Read env vars (lower priority than CLI flags) ---
        env_topics = _env_override("TOPICS")
        env_sample = _env_override("SAMPLE_SIZE", cast=int)
        env_path_save = _env_override("PATH_SAVE")
        env_llm_model = _env_override("LLM_MODEL")
        env_llm_server = _env_override("LLM_SERVER")
        env_dry_run = _env_override("DRY_RUN", cast=lambda v: v.lower() in ("1", "true", "yes"))
        env_no_entailment = _env_override("NO_ENTAILMENT", cast=lambda v: v.lower() in ("1", "true", "yes"))

        # --- Build CLI overrides dict (CLI flags beat env vars) ---
        cli_overrides: dict = {}

        resolved_topics_str = topics or env_topics
        resolved_sample_size = sample_size or env_sample
        resolved_path_save = path_save or env_path_save
        resolved_llm_model = llm_model or env_llm_model
        resolved_llm_server = llm_server or env_llm_server
        resolved_dry_run = dry_run if dry_run is not None else env_dry_run
        resolved_no_entailment = no_entailment if no_entailment is not None else env_no_entailment

        if resolved_topics_str is not None:
            cli_overrides.setdefault("detect", {})["topics"] = _comma_separated_ints(resolved_topics_str)
        if resolved_sample_size is not None:
            cli_overrides.setdefault("detect", {})["sample_size"] = resolved_sample_size
        if resolved_path_save is not None:
            cli_overrides.setdefault("detect", {})["path_save"] = resolved_path_save
        if resolved_dry_run is not None:
            cli_overrides.setdefault("detect", {})["dry_run"] = resolved_dry_run
        if resolved_no_entailment is not None:
            cli_overrides.setdefault("detect", {})["no_entailment"] = resolved_no_entailment
        if resolved_llm_model is not None:
            cli_overrides.setdefault("llm", {}).setdefault("default", {})["model"] = resolved_llm_model
        if resolved_llm_server is not None:
            cli_overrides["llm_server"] = resolved_llm_server

        # --- Load & merge config ---
        cfg = load_config(
            run_config_path=config,
            system_config_path=system_config,
            cli_overrides=cli_overrides if cli_overrides else None,
        )

        # --- Typo warnings ---
        for w in validate_config_keys(cfg):
            print_warning(w)

        # --- Validate required fields ---
        detect_cfg = cfg.get("detect")
        if not detect_cfg:
            print_error("No 'detect' section found in configuration. Use --config or provide required flags.")
            raise typer.Exit(code=1)

        final_topics = detect_cfg.get("topics")
        final_path_save = detect_cfg.get("path_save")
        if not final_topics or not final_path_save:
            print_error("'detect.topics' and 'detect.path_save' are required (via config, env vars, or CLI flags).")
            raise typer.Exit(code=1)

        src_cfg = detect_cfg.get("source", {})
        tgt_cfg = detect_cfg.get("target", {})

        if not src_cfg.get("lang_filter"):
            print_error("'detect.source.lang_filter' is required (e.g. EN).")
            raise typer.Exit(code=1)
        if not tgt_cfg.get("lang_filter"):
            print_error("'detect.target.lang_filter' is required (e.g. DE).")
            raise typer.Exit(code=1)

        # --- Validate file paths early ---
        _validate_paths([
            (src_cfg.get("corpus_path"), "detect.source.corpus_path"),
            (src_cfg.get("thetas_path"), "detect.source.thetas_path"),
            (tgt_cfg.get("corpus_path"), "detect.target.corpus_path"),
            (tgt_cfg.get("thetas_path"), "detect.target.thetas_path"),
        ])
        if src_cfg.get("filter_ids_path"):
            _validate_paths([(src_cfg["filter_ids_path"], "detect.source.filter_ids_path")])
        if tgt_cfg.get("filter_ids_path"):
            _validate_paths([(tgt_cfg["filter_ids_path"], "detect.target.filter_ids_path")])

        # --- Print config if requested ---
        if print_config_flag:
            if json_mode:
                print_json_result(detect_cfg)
            else:
                print_header("Resolved Detection Configuration")
                console.print_json(json.dumps(detect_cfg, indent=2, default=str))
            raise typer.Exit()

        # --- Overwrite confirmation ---
        _prompt_overwrite(final_path_save, yes)

        # --- Show summary table ---
        if not json_mode:
            print_header("MIND Detection Pipeline")
            print_config_table({
                "Topics": final_topics,
                "Sample size": detect_cfg.get("sample_size", "all"),
                "Output": final_path_save,
                "Dry run": detect_cfg.get("dry_run", False),
                "Entailment": not detect_cfg.get("no_entailment", False),
                "LLM model": cfg.get("llm", {}).get("default", {}).get("model", "from config"),
            }, title="Run Configuration")

        # --- Build corpus dicts ---
        src_filter_ids = _read_filter_ids(src_cfg.get("filter_ids_path"))
        tgt_filter_ids = _read_filter_ids(tgt_cfg.get("filter_ids_path"))

        source_corpus = {
            "corpus_path": src_cfg["corpus_path"],
            "thetas_path": src_cfg["thetas_path"],
            "id_col": src_cfg.get("id_col", "doc_id"),
            "passage_col": src_cfg.get("passage_col", "text"),
            "full_doc_col": src_cfg.get("full_doc_col", "full_doc"),
            "language_filter": src_cfg["lang_filter"],
            "filter_ids": src_filter_ids,
            "load_thetas": detect_cfg.get("load_thetas", False),
        }

        target_corpus = {
            "corpus_path": tgt_cfg["corpus_path"],
            "thetas_path": tgt_cfg["thetas_path"],
            "id_col": tgt_cfg.get("id_col", "doc_id"),
            "passage_col": tgt_cfg.get("passage_col", "text"),
            "full_doc_col": tgt_cfg.get("full_doc_col", "full_doc"),
            "language_filter": tgt_cfg["lang_filter"],
            "filter_ids": tgt_filter_ids,
            "load_thetas": detect_cfg.get("load_thetas", False),
        }
        if tgt_cfg.get("index_path"):
            target_corpus["index_path"] = tgt_cfg["index_path"]

        mind_cfg = {
            "llm_model": cfg.get("llm", {}).get("default", {}).get("model"),
            "llm_server": cfg.get("llm_server"),
            "source_corpus": source_corpus,
            "target_corpus": target_corpus,
            "dry_run": detect_cfg.get("dry_run", False),
            "do_check_entailement": not detect_cfg.get("no_entailment", False),
        }

        # --- Instantiate & run ---
        from mind.pipeline.pipeline import MIND

        start = time.time()
        with spinner("Initialising pipeline…"):
            mind = MIND(**mind_cfg)

        run_kwargs = {
            "topics": final_topics,
            "path_save": final_path_save,
            "previous_check": src_cfg.get("previous_check"),
        }
        if detect_cfg.get("sample_size") is not None:
            run_kwargs["sample_size"] = detect_cfg["sample_size"]

        with spinner("Running detection pipeline…"):
            mind.run_pipeline(**run_kwargs)

        elapsed = time.time() - start

        if json_mode:
            print_json_result({
                "status": "ok",
                "elapsed_seconds": round(elapsed, 1),
                "topics": final_topics,
                "path_save": final_path_save,
            })
        else:
            print_success(f"Detection complete in {elapsed:.1f}s — results saved to {final_path_save}")

    except (KeyboardInterrupt, SystemExit):
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)
    except typer.Exit:
        raise
    except Exception as exc:
        print_error(f"Detection failed: {exc}")
        console.print_exception()
        raise typer.Exit(code=1)
