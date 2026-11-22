"""Error categories, error-breakdown tables and qualitative examples.

Two kinds of analysis on the word-level predictions:

* ``error_rows`` / ``save_error_breakdown``: every top-1 error is put into ONE
  linguistic category (nukta, vowel length, ...) by ``tag_error``, and the share of each
  category is reported per system and noise level (``error_breakdown.csv``, Table 5).
* ``qualitative_lines`` / ``save_qualitative_examples``: a reproducible random sample of
  words with each system's top-3 output (``qualitative_examples.txt``).

The categories come from simple string rules, so they are only a rough guide: a word that
differs from the gold in more than one way falls into "other" (65-70% of errors).

Called from ``evaluation/word_eval.py`` (``save_evaluation``).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..config import Config
from .metrics import edit_distance

# Long vowel -> its short counterpart (ii -> i, uu -> u, for both the vowel signs and the
# independent vowels). If two words become equal after this mapping, they differ only in
# vowel length. Example: "पीला" vs "पिला".
LONG2SHORT = {"ी": "ि", "ू": "ु", "ई": "इ", "ऊ": "उ"}
# Aspirated consonant -> its unaspirated partner (kh -> k, gh -> g, chh -> ch, ...).
# If two words become equal after this mapping, they differ only in aspiration.
ASPIRATE = {
    "ख": "क",
    "घ": "ग",
    "छ": "च",
    "झ": "ज",
    "ठ": "ट",
    "ढ": "ड",
    "थ": "त",
    "ध": "द",
    "फ": "प",
    "भ": "ब",
}
# The categories, in the order they appear as columns of error_breakdown.csv.
CATEGORIES = (
    "anusvara/chandrabindu",
    "vowel length",
    "nukta",
    "aspiration",
    "conjunct/halant",
    "schwa/ā",
    "other",
)
# Shape of the predictions: (system name, noise level) -> for each test word, a ranked
# list of (devanagari_text, log_probability) candidates, best first.
Predictions = dict[tuple[str, float], list[list[tuple[str, float]]]]


def _map(text: str, table: dict[str, str]) -> str:
    """Replace each character by ``table[char]`` if it has an entry, else keep it."""
    return "".join(table.get(char, char) for char in text)


def tag_error(prediction: str, gold: str) -> str:
    """Name the single kind of difference between a wrong prediction and the gold word.

    The tests below are tried IN ORDER and the first one that makes the two strings equal
    decides the category. Each test "forgives" one kind of difference and asks whether
    the words are then identical:

    1. anusvara/chandrabindu: treat U+0901 (chandrabindu) and U+0902 (anusvara) as equal.
    2. vowel length: map long vowels to short ones (LONG2SHORT).
    3. nukta: delete the nukta dot U+093C (e.g. "ज़" vs "ज").
    4. aspiration: map aspirated consonants to unaspirated ones (ASPIRATE).
    5. conjunct/halant: delete the halant U+094D (a joined consonant cluster vs separate).
    6. schwa/ā: delete the vowel sign "ा" U+093E (an extra or missing "aa", which is how
       schwa deletion shows up, e.g. "खरब" vs "खराब").
    7. other: none of the above, or two or more kinds of difference at once.

    Only top-1 errors are tagged, and always against the closest accepted spelling.
    """
    if prediction.replace("ँ", "ं") == gold.replace("ँ", "ं"):
        return "anusvara/chandrabindu"
    if _map(prediction, LONG2SHORT) == _map(gold, LONG2SHORT):
        return "vowel length"
    if prediction.replace("़", "") == gold.replace("़", ""):
        return "nukta"
    if _map(prediction, ASPIRATE) == _map(gold, ASPIRATE):
        return "aspiration"
    if prediction.replace("्", "") == gold.replace("्", ""):
        return "conjunct/halant"
    if prediction.replace("ा", "") == gold.replace("ा", ""):
        return "schwa/ā"
    return "other"


def error_rows(
    predictions: Predictions,
    test_refs_list: Sequence[Sequence[str]],
    model_names: Sequence[str],
    levels: Sequence[float],
) -> list[dict[str, Any]]:
    """Count top-1 errors per category for every (noise level, system).

    Returns one row per (level, system) with ``#errors`` (how many words were wrong) and,
    for each category, the SHARE of those errors that fall in it (shares sum to 1).
    """
    rows: list[dict[str, Any]] = []
    for level in levels:
        for name in model_names:
            counts: dict[str, int] = {}
            errors = 0
            for hypotheses, references in zip(predictions[(name, level)], test_refs_list):
                # The system's best guess (rank 1); empty string if it produced nothing.
                top1 = hypotheses[0][0] if hypotheses else ""
                refs = list(references)
                if top1 in refs:
                    continue  # correct, so not an error
                errors += 1
                # Compare against the closest accepted spelling, so a word with several
                # valid forms is judged by the form the system came nearest to.
                best = min(refs, key=lambda ref: edit_distance(top1, ref)) if refs else ""
                category = tag_error(top1, best)
                counts[category] = counts.get(category, 0) + 1
            row: dict[str, Any] = {"model": name, "noise": level, "#errors": errors}
            for category in CATEGORIES:
                # max(errors, 1) avoids dividing by zero when a system makes no errors.
                row[category] = counts.get(category, 0) / max(errors, 1)
            rows.append(row)
    return rows


def save_error_breakdown(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the category shares to ``error_breakdown.csv`` (rounded to 3 decimals)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for category in CATEGORIES:
            frame[category] = frame[category].round(3)
    frame.to_csv(path, index=False)


def qualitative_lines(
    predictions: Predictions,
    test_words: Sequence[str],
    noisy_inputs: dict[float, list[str]],
    test_refs_list: Sequence[Sequence[str]],
    model_names: Sequence[str],
    level: float = 0.3,
    count: int = 12,
    seed: int = 7,
) -> list[str]:
    """Text lines showing ``count`` random test words with each system's top-3 outputs.

    For every sampled word it prints the noisy input, the clean word, the accepted gold
    spellings, then one line per system with its three best candidates and their
    log-probabilities. A fixed ``seed`` makes the same words appear on every run.
    """
    if not test_words:
        return []
    indices = random.Random(seed).sample(
        range(len(test_words)), min(count, len(test_words))
    )
    lines: list[str] = []
    for index in indices:
        lines.append(
            f"\ninput (noise {level}): {noisy_inputs[level][index]:14s} "
            f"clean: {test_words[index]:12s} gold: {sorted(test_refs_list[index])}"
        )
        for name in model_names:
            hypotheses = predictions[(name, level)][index][:3]
            lines.append(f"   {name:18s} {hypotheses}")
    return lines


def save_qualitative_examples(
    predictions: Predictions,
    test_words: Sequence[str],
    noisy_inputs: dict[float, list[str]],
    test_refs_list: Sequence[Sequence[str]],
    config: Config,
    model_names: Sequence[str] | None = None,
) -> None:
    """Write ``qualitative_examples.txt`` using noise level 0.3 (or the highest level run)."""
    path = config.out_dir / "qualitative_examples.txt"
    lines = qualitative_lines(
        predictions,
        test_words,
        noisy_inputs,
        test_refs_list,
        model_names if model_names is not None else config.models,
        level=0.3 if 0.3 in config.noise_levels else config.noise_levels[-1],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
