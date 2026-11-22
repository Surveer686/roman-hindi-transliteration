"""Tests for the metric functions (evaluation/metrics.py)."""
from trainer2.evaluation.metrics import (
    cluster_bootstrap,
    edit_distance,
    evaluate,
    paired_diff_ci,
)


def test_edit_distance():
    """Edit distance on known examples."""
    # Classic example: 3 edits
    assert edit_distance("kitten", "sitting") == 3
    # Empty string: every character counts
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3
    # Identical strings: 0 edits
    assert edit_distance("same", "same") == 0


def test_top_k_and_cer():
    """Top-1 / top-2 accuracy and CER on a tiny example."""
    # Ranked candidates for each word
    predictions = [["ब", "क"], ["अ", "क"]]
    # Accepted spellings for each word
    references = [{"ब"}, {"ख", "क"}]
    # Compute the metrics
    metrics = evaluate(predictions, references, ks=(1, 2))
    # Word 1 is right at rank 1, word 2 only at rank 2
    assert metrics["top1"] == 0.5
    assert metrics["top2"] == 1.0
    # Word 2 is one edit off a one-character answer: 1 / 2
    assert metrics["cer_micro"] == 0.5


def test_intervals_bracket_point_estimates():
    """Every 95% interval contains its own estimate."""
    predictions = [["ब", "क"], ["अ", "क"], ["ब", "ख"], ["क", "ब"]]
    references = [{"ब"}, {"क"}, {"ब"}, {"ब"}]
    metrics = evaluate(predictions, references, ks=(1, 2))
    # Each interval must contain its estimate
    for key in ("top1", "top2", "cer_micro", "cer_macro"):
        assert metrics[f"{key}_lo"] <= metrics[key] <= metrics[f"{key}_hi"]


def test_cluster_bootstrap_ratio():
    """The group bootstrap gives the right ratio and a sensible interval."""
    # 3 of 4 groups are right
    point, low, high = cluster_bootstrap([1, 0, 1, 1], [1, 1, 1, 1])
    assert point == 0.75
    assert 0.0 <= low <= point <= high <= 1.0


def test_paired_difference_detects_a_clear_gap():
    """A system that is always right beats one that is always wrong."""
    # System A is always right, B never
    diff, low, high = paired_diff_ci([1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1])
    assert diff == low == high == 1.0


def test_paired_difference_of_identical_systems_is_zero():
    """A system compared with itself differs by 0."""
    # The same results for both systems
    values = [1, 0, 1, 0, 1, 1]
    # Compare the system with itself
    diff, low, high = paired_diff_ci(values, values, [1] * 6)
    # The difference is 0 and the interval includes 0
    assert diff == 0.0 and low <= 0.0 <= high
