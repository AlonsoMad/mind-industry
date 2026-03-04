"""Unit tests for CLI command structure, help output, init-config, and error scenarios."""
import tempfile
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from mind.cli.main import app

runner = CliRunner()


class TestRootApp:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "data" in result.output
        assert "detect" in result.output
        assert "tm" in result.output

    def test_version_V(self):
        """--version short flag is now -V (was -v)."""
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert "mind" in result.output

    def test_version_long(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "mind" in result.output

    def test_quiet_and_verbose_mutually_exclusive(self):
        result = runner.invoke(app, ["--quiet", "--verbose", "detect", "--help"])
        assert result.exit_code != 0

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "data" in result.output


class TestDetectCommands:
    def test_run_help(self):
        result = runner.invoke(app, ["detect", "run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--topics" in result.output
        assert "--path-save" in result.output
        assert "--dry-run" in result.output
        assert "--yes" in result.output
        assert "--output-format" in result.output

    def test_init_config_is_valid_yaml(self):
        result = runner.invoke(app, ["detect", "init-config"])
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert "detect" in parsed
        # init-config for detect should NOT contain data or tm sections
        assert "data" not in parsed
        assert "tm" not in parsed

    def test_detect_help(self):
        result = runner.invoke(app, ["detect", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "init-config" in result.output
        assert "validate-config" in result.output

    def test_validate_config_help(self):
        result = runner.invoke(app, ["detect", "validate-config", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_validate_config_missing_config_section(self, tmp_path):
        """Running validate-config with a YAML that has no 'detect' key should fail."""
        cfg = tmp_path / "sys.yaml"
        cfg.write_text(yaml.dump({"llm": {"default": {"model": "test"}}}))
        result = runner.invoke(app, [
            "detect", "validate-config",
            "--system-config", str(cfg),
        ])
        assert result.exit_code == 1

    def test_validate_config_good_config(self, tmp_path):
        """validate-config with valid fields and existing files should succeed."""
        # Create dummy data files
        for name in ("corpus.parquet", "thetas.npz"):
            (tmp_path / name).touch()

        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(yaml.dump({
            "detect": {
                "topics": [1],
                "path_save": str(tmp_path / "results"),
                "source": {
                    "corpus_path": str(tmp_path / "corpus.parquet"),
                    "thetas_path": str(tmp_path / "thetas.npz"),
                    "lang_filter": "EN",
                },
                "target": {
                    "corpus_path": str(tmp_path / "corpus.parquet"),
                    "thetas_path": str(tmp_path / "thetas.npz"),
                    "lang_filter": "DE",
                },
                "load_thetas": True,
            },
        }))
        sys_cfg = tmp_path / "sys.yaml"
        sys_cfg.write_text(yaml.dump({"llm": {"default": {"model": "test"}}}))

        result = runner.invoke(app, [
            "detect", "validate-config",
            "--config", str(run_cfg),
            "--system-config", str(sys_cfg),
        ])
        assert result.exit_code == 0

    def test_run_errors_missing_required_fields(self, tmp_path):
        """detect run with a config missing required fields should exit with code 1."""
        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(yaml.dump({"detect": {"sample_size": 10}}))  # no topics, no path_save
        sys_cfg = tmp_path / "sys.yaml"
        sys_cfg.write_text(yaml.dump({"llm": {"default": {"model": "test"}}}))

        result = runner.invoke(app, [
            "detect", "run",
            "--config", str(run_cfg),
            "--system-config", str(sys_cfg),
        ])
        assert result.exit_code == 1

    def test_run_errors_missing_lang_filter(self, tmp_path):
        """detect run with missing lang_filter in target should exit with code 1."""
        for name in ("corpus.parquet", "thetas.npz"):
            (tmp_path / name).touch()
        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(yaml.dump({
            "detect": {
                "topics": [1],
                "path_save": str(tmp_path / "results"),
                "source": {
                    "corpus_path": str(tmp_path / "corpus.parquet"),
                    "thetas_path": str(tmp_path / "thetas.npz"),
                    "lang_filter": "EN",
                },
                "target": {
                    "corpus_path": str(tmp_path / "corpus.parquet"),
                    "thetas_path": str(tmp_path / "thetas.npz"),
                    # lang_filter intentionally omitted
                },
                "load_thetas": True,
            },
        }))
        sys_cfg = tmp_path / "sys.yaml"
        sys_cfg.write_text(yaml.dump({"llm": {"default": {"model": "test"}}}))
        result = runner.invoke(app, [
            "detect", "run",
            "--config", str(run_cfg),
            "--system-config", str(sys_cfg),
        ])
        assert result.exit_code == 1

    def test_run_errors_nonexistent_input_path(self, tmp_path):
        """detect run with a corpus_path that doesn't exist should exit with code 1."""
        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(yaml.dump({
            "detect": {
                "topics": [1],
                "path_save": str(tmp_path / "results"),
                "source": {
                    "corpus_path": "/totally/does/not/exist.parquet",
                    "thetas_path": "/totally/does/not/exist.npz",
                    "lang_filter": "EN",
                },
                "target": {
                    "corpus_path": "/totally/does/not/exist.parquet",
                    "thetas_path": "/totally/does/not/exist.npz",
                    "lang_filter": "DE",
                },
                "load_thetas": True,
            },
        }))
        sys_cfg = tmp_path / "sys.yaml"
        sys_cfg.write_text(yaml.dump({"llm": {"default": {"model": "test"}}}))
        result = runner.invoke(app, [
            "detect", "run",
            "--config", str(run_cfg),
            "--system-config", str(sys_cfg),
        ])
        assert result.exit_code == 1

    def test_run_missing_config_file(self, tmp_path):
        """detect run pointing to a nonexistent config file should exit with code 1."""
        sys_cfg = tmp_path / "sys.yaml"
        sys_cfg.write_text(yaml.dump({"llm": {"default": {"model": "test"}}}))
        result = runner.invoke(app, [
            "detect", "run",
            "--config", "/nonexistent/path.yaml",
            "--system-config", str(sys_cfg),
        ])
        assert result.exit_code == 1

    def test_run_invalid_yaml_config(self, tmp_path):
        """detect run with a malformed YAML config should exit with code 1."""
        broken = tmp_path / "broken.yaml"
        broken.write_text("detect: {topics: [1\n  invalid yaml !!")
        sys_cfg = tmp_path / "sys.yaml"
        sys_cfg.write_text(yaml.dump({"llm": {"default": {"model": "test"}}}))
        result = runner.invoke(app, [
            "detect", "run",
            "--config", str(broken),
            "--system-config", str(sys_cfg),
        ])
        assert result.exit_code == 1


class TestDataCommands:
    def test_segment_help(self):
        result = runner.invoke(app, ["data", "segment", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--output" in result.output
        assert "--text-col" in result.output
        assert "--yes" in result.output

    def test_translate_help(self):
        result = runner.invoke(app, ["data", "translate", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--src-lang" in result.output
        assert "--tgt-lang" in result.output
        assert "--yes" in result.output

    def test_prepare_help(self):
        result = runner.invoke(app, ["data", "prepare", "--help"])
        assert result.exit_code == 0
        assert "--anchor" in result.output
        assert "--comparison" in result.output
        assert "--schema" in result.output
        assert "--yes" in result.output

    def test_data_help(self):
        result = runner.invoke(app, ["data", "--help"])
        assert result.exit_code == 0
        assert "segment" in result.output
        assert "translate" in result.output
        assert "prepare" in result.output
        assert "init-config" in result.output

    def test_data_init_config_is_valid_yaml(self):
        result = runner.invoke(app, ["data", "init-config"])
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert "data" in parsed
        assert "detect" not in parsed
        assert "tm" not in parsed

    def test_segment_errors_on_nonexistent_input(self, tmp_path):
        sys_cfg = tmp_path / "sys.yaml"
        sys_cfg.write_text(yaml.dump({}))
        result = runner.invoke(app, [
            "data", "segment",
            "--input", "/nonexistent.parquet",
            "--output", str(tmp_path / "out.parquet"),
            "--system-config", str(sys_cfg),
        ])
        assert result.exit_code == 1

    def test_prepare_errors_on_missing_schema_keys(self, tmp_path):
        """A --schema dict missing required keys should fail fast."""
        for name in ("anchor.parquet", "comparison.parquet"):
            (tmp_path / name).touch()
        sys_cfg = tmp_path / "sys.yaml"
        sys_cfg.write_text(yaml.dump({}))
        import json
        bad_schema = json.dumps({"chunk_id": "id"})  # missing text, lang, full_doc, doc_id
        result = runner.invoke(app, [
            "data", "prepare",
            "--anchor", str(tmp_path / "anchor.parquet"),
            "--comparison", str(tmp_path / "comparison.parquet"),
            "--output", str(tmp_path / "out.parquet"),
            "--schema", bad_schema,
            "--system-config", str(sys_cfg),
        ])
        assert result.exit_code == 1


class TestTmCommands:
    def test_train_help(self):
        result = runner.invoke(app, ["tm", "train", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--lang1" in result.output
        assert "--num-topics" in result.output
        assert "--yes" in result.output

    def test_tm_help(self):
        result = runner.invoke(app, ["tm", "--help"])
        assert result.exit_code == 0
        assert "train" in result.output
        assert "init-config" in result.output

    def test_tm_init_config_is_valid_yaml(self):
        result = runner.invoke(app, ["tm", "init-config"])
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert "tm" in parsed
        assert "data" not in parsed
        assert "detect" not in parsed

    def test_train_errors_on_nonexistent_input(self, tmp_path):
        sys_cfg = tmp_path / "sys.yaml"
        sys_cfg.write_text(yaml.dump({}))
        result = runner.invoke(app, [
            "tm", "train",
            "--input", "/nonexistent.parquet",
            "--lang1", "EN",
            "--lang2", "DE",
            "--model-folder", str(tmp_path / "model"),
            "--num-topics", "10",
            "--system-config", str(sys_cfg),
        ])
        assert result.exit_code == 1


class TestLegacyCLI:
    def test_build_parser_required_args(self):
        """Legacy parser should accept required arguments without raising."""
        from mind.cli._legacy import build_parser
        args = build_parser().parse_args([
            "--topics", "7,15",
            "--path_save", "/tmp/results",
            "--src_corpus_path", "/tmp/src.parquet",
            "--src_thetas_path", "/tmp/thetas.npz",
            "--tgt_corpus_path", "/tmp/tgt.parquet",
            "--tgt_thetas_path", "/tmp/thetas_tgt.npz",
            "--tgt_lang_filter", "DE",
        ])
        assert args.topics == [7, 15]
        assert args.path_save == "/tmp/results"

    def test_build_parser_comma_topics(self):
        from mind.cli._legacy import build_parser, comma_separated_ints
        result = comma_separated_ints("7,15,23")
        assert result == [7, 15, 23]
