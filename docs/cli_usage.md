# MIND CLI Usage Guide

The MIND CLI consolidates all pipeline scripts into a single `mind` command with semantic subcommands, YAML-based configuration, and Rich terminal output.

## Installation

```bash
# From the project root (requires Python 3.12)
pip install -e .

# Verify
mind --help
mind -V          # version
```

## Global Flags

These flags are available on every `mind` invocation (before any subcommand):

| Flag | Description |
|---|---|
| `-V, --version` | Show version and exit |
| `--verbose` | Enable verbose output (more diagnostic info) |
| `-q, --quiet` | Suppress all output except errors |

```bash
mind --verbose detect run --config my_run.yaml
mind -q detect run --config my_run.yaml
```

## Quick Start

```bash
# 1. Generate a config template for detection
mind detect init-config > my_run.yaml

# 2. Validate config without running
mind detect validate-config --config my_run.yaml

# 3. Run the full pipeline
mind detect run --config my_run.yaml
```

## Commands

### `mind detect` — Discrepancy Detection

```bash
# Generate a config template (detect section only)
mind detect init-config > my_run.yaml

# Validate a config file — checks required fields and file paths
mind detect validate-config --config my_run.yaml
mind detect validate-config --config my_run.yaml --system-config config/config.yaml

# Run the detection pipeline
mind detect run --config my_run.yaml

# Override specific settings inline
mind detect run --config my_run.yaml --topics 7,15 --sample-size 100 --llm-model gemini-2.5-flash

# Skip confirmation prompts (e.g. in CI)
mind detect run --config my_run.yaml --yes

# Print resolved config without running
mind detect run --config my_run.yaml --print-config

# Machine-readable JSON output (suppresses Rich output)
mind detect run --config my_run.yaml --output-format json
```

**Environment variable overrides** (lower priority than CLI flags):

| Variable | Equivalent flag |
|---|---|
| `MIND_TOPICS` | `--topics` |
| `MIND_PATH_SAVE` | `--path-save` |
| `MIND_SAMPLE_SIZE` | `--sample-size` |
| `MIND_LLM_MODEL` | `--llm-model` |
| `MIND_LLM_SERVER` | `--llm-server` |
| `MIND_DRY_RUN` | `--dry-run` |
| `MIND_NO_ENTAILMENT` | `--no-entailment` |

```bash
# Useful in CI / Docker
MIND_TOPICS=7,15 MIND_PATH_SAVE=/results mind detect run --config my_run.yaml --yes
```

---

### `mind data` — Data Preprocessing

```bash
# Generate a config template (data section only)
mind data init-config > my_data.yaml

# Segment documents into passages
mind data segment --config run.yaml
mind data segment --input data/raw/docs.parquet --output data/processed/segmented.parquet

# Translate passages between languages
mind data translate --config run.yaml
mind data translate --input data/processed/segmented.parquet \
  --output data/processed/translated.parquet \
  --src-lang en --tgt-lang de

# Prepare and merge datasets
mind data prepare --config run.yaml
mind data prepare --anchor data/processed/segmented.parquet \
  --comparison data/processed/translated.parquet \
  --output data/processed/prepared.parquet \
  --schema '{"chunk_id":"id_preproc","text":"text","lang":"lang","full_doc":"full_doc","doc_id":"doc_id"}'
```

The `--schema` flag accepts:
- A JSON string inline (as above)
- A path to a `.json` or `.yaml` file

All data commands accept `--yes` / `-y` to skip the overwrite confirmation prompt.

---

### `mind tm` — Topic Modeling

```bash
# Generate a config template (tm section only)
mind tm init-config > my_tm.yaml

# Train a polylingual topic model
mind tm train --config run.yaml
mind tm train --input data/processed/prepared.parquet \
  --lang1 EN --lang2 DE \
  --model-folder data/models/tm_ende \
  --num-topics 30

# Skip overwrite confirmation
mind tm train --config run.yaml --yes
```

---

## Configuration

All commands accept `--config path.yaml` for YAML-based configuration and `--system-config path.yaml` to override the system defaults.

### Config Priority (highest → lowest)

1. **CLI flags** (e.g. `--topics 7,15`)
2. **Environment variables** (`MIND_TOPICS=7,15` — detection only)
3. **Run config file** (`--config run.yaml`)
4. **System config** (`config/config.yaml` — resolved relative to the installed package)
5. **Hardcoded defaults** in code

### Generate Templates

Each command group has its own `init-config` that emits only its relevant section:

```bash
mind detect init-config > my_detect.yaml   # detect: section only
mind data init-config   > my_data.yaml     # data: section only
mind tm init-config     > my_tm.yaml       # tm: section only
```

---

## Shell Completion

Typer supports shell completion for Bash, Zsh, and Fish:

```bash
# Bash
mind --install-completion bash

# Zsh
mind --install-completion zsh

# Fish
mind --install-completion fish

# Or show the completion script without installing
mind --show-completion bash
```

---

## Legacy Compatibility

The old `mind-legacy` entry point preserves the original `cli.py` interface for backward compatibility. It shows a **deprecation warning** and will be removed in a future release:

```bash
mind-legacy --topics 7 --path_save results/ \
  --src_corpus_path data/src.parquet \
  --src_thetas_path data/thetas.npz \
  --tgt_corpus_path data/tgt.parquet \
  --tgt_thetas_path data/thetas_tgt.npz \
  --tgt_lang_filter DE
```

Migrate to the new commands at your convenience:

```bash
# Old               →  New
mind-legacy ...     →  mind detect run --config my_run.yaml
```
