"""Unit tests for mind.cli._config_loader."""
import tempfile
from pathlib import Path

import pytest
import yaml

from mind.cli._config_loader import _deep_merge, load_config


class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_nested_override(self):
        base = {"x": {"a": 1, "b": 2}, "y": 10}
        override = {"x": {"b": 99, "c": 3}}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 99, "c": 3}, "y": 10}

    def test_deeply_nested(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": {"c": 99, "d": 2}}}

    def test_does_not_mutate_base(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"b": 1}}

    def test_new_nested_key(self):
        base = {"a": 1}
        override = {"b": {"c": 2}}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": {"c": 2}}

    def test_override_dict_with_scalar(self):
        base = {"a": {"b": 1}}
        override = {"a": "replaced"}
        result = _deep_merge(base, override)
        assert result == {"a": "replaced"}

    def test_empty_dicts(self):
        assert _deep_merge({}, {}) == {}
        assert _deep_merge({"a": 1}, {}) == {"a": 1}
        assert _deep_merge({}, {"a": 1}) == {"a": 1}


class TestLoadConfig:
    def _write_yaml(self, dir_path: Path, filename: str, data: dict) -> Path:
        p = dir_path / filename
        p.write_text(yaml.dump(data))
        return p

    def test_system_only(self, tmp_path):
        sys_cfg = self._write_yaml(tmp_path, "sys.yaml", {"llm": {"model": "test"}, "logger": {"level": "INFO"}})
        result = load_config(system_config_path=sys_cfg)
        assert result["llm"]["model"] == "test"
        assert result["logger"]["level"] == "INFO"

    def test_system_plus_run(self, tmp_path):
        sys_cfg = self._write_yaml(tmp_path, "sys.yaml", {"llm": {"model": "old", "backend": "ollama"}, "logger": {"level": "INFO"}})
        run_cfg = self._write_yaml(tmp_path, "run.yaml", {"llm": {"model": "new"}, "detect": {"topics": [7]}})
        result = load_config(run_config_path=run_cfg, system_config_path=sys_cfg)
        assert result["llm"]["model"] == "new"
        assert result["llm"]["backend"] == "ollama"  # preserved from system
        assert result["detect"]["topics"] == [7]

    def test_system_plus_run_plus_overrides(self, tmp_path):
        sys_cfg = self._write_yaml(tmp_path, "sys.yaml", {"llm": {"model": "old"}})
        run_cfg = self._write_yaml(tmp_path, "run.yaml", {"llm": {"model": "mid"}, "detect": {"topics": [7]}})
        result = load_config(
            run_config_path=run_cfg,
            system_config_path=sys_cfg,
            cli_overrides={"llm": {"model": "cli-override"}, "detect": {"topics": [1, 2, 3]}},
        )
        assert result["llm"]["model"] == "cli-override"
        assert result["detect"]["topics"] == [1, 2, 3]

    def test_no_run_config(self, tmp_path):
        sys_cfg = self._write_yaml(tmp_path, "sys.yaml", {"a": 1})
        result = load_config(system_config_path=sys_cfg)
        assert result == {"a": 1}

    def test_empty_files(self, tmp_path):
        sys_cfg = tmp_path / "sys.yaml"
        sys_cfg.write_text("")
        result = load_config(system_config_path=sys_cfg)
        assert result == {}
