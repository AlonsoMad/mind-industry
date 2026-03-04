"""YAML configuration loader with deep-merge support."""
from pathlib import Path
from typing import Any, Dict, Optional
import copy
import yaml

# Resolve relative to the package root (3 levels up from src/mind/cli/)
# This works regardless of the user's working directory.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_SYSTEM_CONFIG_CANDIDATE = _PACKAGE_ROOT / "config" / "config.yaml"
DEFAULT_SYSTEM_CONFIG: Path = (
    _SYSTEM_CONFIG_CANDIDATE
    if _SYSTEM_CONFIG_CANDIDATE.exists()
    else Path("config/config.yaml")  # last-resort CWD fallback
)

# Known top-level sections for typo detection
_KNOWN_TOP_LEVEL_KEYS = {
    "logger", "optimization", "mind", "llm", "detect", "data", "tm",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    run_config_path: Optional[Path] = None,
    system_config_path: Path = DEFAULT_SYSTEM_CONFIG,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    """Load and merge system config, run config, and CLI overrides.

    Priority (highest to lowest):
        1. Explicit CLI flags (cli_overrides)
        2. Run config file (run_config_path)
        3. System config file (system_config_path) — always loaded as base
        4. Hardcoded defaults in code

    Raises:
        FileNotFoundError: if a config file is missing.
        ValueError: if a config file contains invalid YAML.
    """
    # 1. System config (always loaded as base)
    try:
        with open(system_config_path) as f:
            system = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(
            f"System config not found: {system_config_path}\n"
            f"Tip: pass --system-config <path> to specify a custom path."
        )
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in system config {system_config_path}: {exc}")

    # 2. Run config (user-supplied, optional)
    merged = system
    if run_config_path:
        try:
            with open(run_config_path) as f:
                run = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Run config not found: {run_config_path}")
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in run config {run_config_path}: {exc}")
        merged = _deep_merge(system, run)

    # 3. CLI overrides (highest priority)
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    return merged


def validate_config_keys(cfg: dict) -> list[str]:
    """Check for unknown top-level keys that may indicate typos.

    Returns a list of warning strings (empty if no issues found).
    """
    warnings = []
    for key in cfg:
        if key not in _KNOWN_TOP_LEVEL_KEYS:
            import difflib
            close = difflib.get_close_matches(key, _KNOWN_TOP_LEVEL_KEYS, n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            warnings.append(f"Unknown config key '{key}'{hint}")
    return warnings
