"""Top-k accuracy, character error rate and bootstrap intervals."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def edit_distance(reference: str, hypothesis: str) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for row_index, reference_char in enumerate(reference, 1):
        current = [row_index]
        for column_index, hypothesis_char in enumerate(hypothesis, 1):
            cost = 0 if reference_char == hypothesis_char else 1
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def evaluate(
    predictions: Sequence[Sequence[str]],
    references: Sequence[Iterable[str]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    hits: dict[int, list[float]] = {k: [] for k in ks}
    edits = 0
    reference_chars = 0
    character_errors: list[float] = []
    word_edits: list[int] = []
    word_chars: list[int] = []
    for hypotheses, reference_values in zip(predictions, references):
        refs = list(reference_values)
        for k in ks:
            hits[k].append(float(any(hypothesis in refs for hypothesis in hypotheses[:k])))
        top1 = hypotheses[0] if hypotheses else ""
        best = min(refs, key=lambda item: (edit_distance(top1, item), -len(item))) if refs else ""
        distance = edit_distance(top1, best)
        character_errors.append(distance / max(1, len(best)))
        edits += distance
        reference_chars += len(best)
        word_edits.append(distance)
        word_chars.append(len(best))
    output: dict[str, float] = {
        f"top{k}": float(np.mean(hits[k])) for k in ks
    }
    output["cer_micro"] = edits / max(reference_chars, 1)
    output["cer_macro"] = float(np.mean(character_errors)) if character_errors else 0.0

    # 95% bootstrap intervals over test words; one set of resamples is shared by
    # every metric so the intervals are consistent with each other.
    n_words = len(character_errors)
    if n_words:
        samples = np.random.default_rng(0).integers(0, n_words, (BOOTSTRAP_SAMPLES, n_words))
        for k in ks:
            means = np.array(hits[k], dtype=float)[samples].mean(axis=1)
            output[f"top{k}_lo"], output[f"top{k}_hi"] = _interval(means)
        means = np.array(character_errors, dtype=float)[samples].mean(axis=1)
        output["cer_macro_lo"], output["cer_macro_hi"] = _interval(means)
        micro = np.array(word_edits, dtype=float)[samples].sum(axis=1) / np.maximum(
            np.array(word_chars, dtype=float)[samples].sum(axis=1), 1.0
        )
        output["cer_micro_lo"], output["cer_micro_hi"] = _interval(micro)
    else:
        for k in ks:
            output[f"top{k}_lo"] = output[f"top{k}_hi"] = 0.0
        for name in ("cer_macro", "cer_micro"):
            output[f"{name}_lo"] = output[f"{name}_hi"] = 0.0
    return output


BOOTSTRAP_SAMPLES = 1000


def _interval(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def cluster_bootstrap(
    numerators: Sequence[float],
    denominators: Sequence[float],
    n: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Ratio sum(numerators)/sum(denominators) with a 95% interval, resampling
    whole groups (e.g. tweets) so tokens inside one group are not treated as
    independent."""
    num = np.asarray(numerators, dtype=float)
    den = np.asarray(denominators, dtype=float)
    if len(num) == 0:
        return 0.0, 0.0, 0.0
    point = float(num.sum() / max(den.sum(), 1.0))
    samples = np.random.default_rng(seed).integers(0, len(num), (n, len(num)))
    ratios = num[samples].sum(axis=1) / np.maximum(den[samples].sum(axis=1), 1.0)
    low, high = _interval(ratios)
    return point, low, high


def paired_diff_ci(
    numerators_a: Sequence[float],
    numerators_b: Sequence[float],
    denominators: Sequence[float],
    n: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Difference (A - B) of two ratios scored on the same groups, with a 95%
    interval from resampling those groups jointly. An interval that excludes 0
    means the gap is unlikely to be sampling noise."""
    a = np.asarray(numerators_a, dtype=float)
    b = np.asarray(numerators_b, dtype=float)
    den = np.asarray(denominators, dtype=float)
    if len(a) == 0:
        return 0.0, 0.0, 0.0
    point = float((a.sum() - b.sum()) / max(den.sum(), 1.0))
    samples = np.random.default_rng(seed).integers(0, len(a), (n, len(a)))
    diffs = (a[samples].sum(axis=1) - b[samples].sum(axis=1)) / np.maximum(
        den[samples].sum(axis=1), 1.0
    )
    low, high = _interval(diffs)
    return point, low, high


def self_test() -> None:
    assert edit_distance("kitten", "sitting") == 3
    metrics = evaluate([["ब", "क"], ["अ", "क"]], [{"ब"}, {"ख", "क"}], ks=(1, 2))
    assert metrics["top1"] == 0.5 and metrics["top2"] == 1.0, metrics
    for key in ("top1", "top2", "cer_micro", "cer_macro"):
        assert metrics[f"{key}_lo"] <= metrics[key] <= metrics[f"{key}_hi"], (key, metrics)
    point, low, high = cluster_bootstrap([1, 0, 1, 1], [1, 1, 1, 1])
    assert point == 0.75 and 0.0 <= low <= point <= high <= 1.0, (point, low, high)
    diff, low, high = paired_diff_ci([1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1])
    assert diff == 1.0 and low == 1.0 and high == 1.0, (diff, low, high)
