"""Tests for the pipeline self-test and the configuration."""
from trainer2.cli.pipeline import self_test
from trainer2.config import ROOT, Config, build_config


def test_pipeline_self_test_runs_both_architectures_and_beam_search():
    """The built-in self-test passes (metrics, GRU, Transformer, beam search)."""
    # Runs the metric checks, one tiny model of each type and beam search
    self_test()


def test_default_data_directory_is_the_package_data_folder():
    """The data folder defaults to trainer2/data."""
    # No flags given: the default data folder
    assert Config().data_dir == ROOT / "data"


def test_cli_flags_override_defaults():
    """Command-line flags override the default settings."""
    # Parse two flags
    config = build_config(["--limit-train", "123", "--beam", "3"])
    # Both values come from the flags
    assert config.n_train == 123 and config.beam == 3

