"""Aksharantar word pairs: cleaning, leakage check, reference lists and the cached DataBundle.

What this module does, in order (see ``prepare_data``):

1. Read the three Aksharantar Hindi files (train / valid / test). Every line is a JSON
   object such as ``{"english word": "kharab", "native word": "खराब", ...}``.
2. Clean each pair: normalise Unicode, lowercase the Roman side, keep only pairs whose
   Roman side is ``a-z`` letters and whose Devanagari side is Devanagari letters, drop
   overlong words and exact duplicates.
3. Leakage check: remove any training pair that also occurs in validation or test, so the
   models are never scored on something they were trained on.
4. Sample the training subset (100,000 pairs in our runs) and build the *reference lists*:
   for each test word, every Devanagari spelling accepted as correct.
5. Cache everything in ``prepared.json`` so later runs skip the 195 MB training file.

Used by: ``training/trainer.py`` (train pairs, validation), ``evaluation/word_eval.py``
(test words and references) and ``cli/pipeline.py`` (``load_bundle``).
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..config import Config

# Roman side must be plain lowercase letters only (no digits, spaces, hyphens, accents).
ROMAN_RE = re.compile(r"^[a-z]+$")
# Devanagari side must consist only of characters in the Devanagari Unicode block
# U+0900-U+097F (letters, vowel signs, virama, anusvara, nukta, ...).
DEVANAGARI_RE = re.compile(r"^[\u0900-\u097F]+$")


@dataclass
class DataBundle:
    """Everything the rest of the pipeline needs from the word data, in one object."""

    # Roman/Devanagari pairs used to train the models: [("kharab", "खराब"), ...]
    train_pairs: list[tuple[str, str]]
    # Validation pairs used only to compute validation *loss* during training.
    val_pairs_loss: list[tuple[str, str]]
    # Roman word -> all accepted Devanagari spellings, for validation / test scoring.
    # Several spellings per word are possible, e.g. "pearson" -> ["पियरसन", "पियर्सन"].
    val_refs: dict[str, list[str]]
    test_refs: dict[str, list[str]]
    # The (possibly capped) list of unique Roman words actually scored.
    val_words: list[str]
    test_words: list[str]
    # Counts and checks (cleaned sizes, leaked pairs removed, ...) saved to data_stats.json.
    stats: dict[str, Any]


def iter_jsonl(path: Path, limit: int | None = None) -> Iterator[tuple[str, str, Any]]:
    """Stream ``(roman, native, source)`` triples from a JSON-lines file.

    Reads lazily (one line at a time) because the training file is ~195 MB. Rows missing
    either word, or with non-string values, are skipped. ``limit`` stops early (handy for
    quick experiments).
    """
    seen = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue  # ignore blank lines
            row = json.loads(line)
            # Aksharantar's column names contain a space: "english word" / "native word".
            roman = row.get("english word")
            native = row.get("native word")
            if not isinstance(roman, str) or not isinstance(native, str):
                continue
            yield roman, native, row.get("source")
            seen += 1
            if limit is not None and seen >= limit:
                return


def clean_pairs(rows: Iterator[tuple[str, str, Any]], max_len: int) -> list[tuple[str, str]]:
    """Normalise, filter and de-duplicate raw rows into clean ``(roman, native)`` pairs."""
    seen: set[tuple[str, str]] = set()
    output: list[tuple[str, str]] = []
    for roman, native, _ in rows:
        # NFC makes visually identical Devanagari strings byte-identical (e.g. a letter
        # plus nukta written as one code point or two). Roman text is also lowercased.
        source = unicodedata.normalize("NFC", roman).strip().lower()
        target = unicodedata.normalize("NFC", native).strip()
        if not ROMAN_RE.match(source) or not DEVANAGARI_RE.match(target):
            continue  # not a plain Roman word / not a pure Devanagari word
        if len(source) > max_len or len(target) > max_len:
            continue  # very long entries are rare and slow to train on
        pair = (source, target)
        if pair in seen:
            continue  # exact duplicate pair
        seen.add(pair)
        output.append(pair)
    return output


def build_refs(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Group pairs by Roman word: ``{"pearson": ["पियरसन", "पियर्सन"], ...}``.

    A Roman spelling can map to several correct Devanagari spellings. During scoring a
    prediction counts as correct if it equals *any* of them.
    """
    refs: dict[str, set[str]] = defaultdict(set)
    for roman, native in pairs:
        refs[roman].add(native)
    # Sorted lists give a stable, reproducible order in saved files.
    return {roman: sorted(targets) for roman, targets in refs.items()}


def cap_words(words: Iterable[str], count: int, seed: int) -> list[str]:
    """Return at most ``count`` words, chosen reproducibly.

    Sorting first removes any dependence on set/dict ordering, so the same ``seed`` always
    selects the same subset. (Our test set has 9,991 words, just under the 10,000 cap.)
    """
    ordered = sorted(words)
    if len(ordered) > count:
        return sorted(random.Random(seed).sample(ordered, count))
    return ordered


def signature(config: Config) -> str:
    """A string describing the settings that change *which* data gets selected.

    It is stored inside ``prepared.json``. If any of these settings change, the cached
    file no longer matches and is rebuilt (see ``load_bundle``).
    """
    return json.dumps(
        {
            "n_train": config.n_train,
            "n_val_loss": config.n_val_loss,
            "n_val_acc": config.n_val_acc,
            "n_test_max": config.n_test_max,
            "max_len": config.max_len,
            "seed": config.seed,
        },
        sort_keys=True,
    )


def save_bundle(bundle: DataBundle, path: Path, config: Config) -> None:
    """Write the cache (``prepared.json``) and a small human-readable ``data_stats.json``."""
    payload = {
        "signature": signature(config),
        "train_pairs": bundle.train_pairs,
        "val_pairs_loss": bundle.val_pairs_loss,
        "val_refs": bundle.val_refs,
        "test_refs": bundle.test_refs,
        "val_words": bundle.val_words,
        "test_words": bundle.test_words,
        "stats": bundle.stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config.stats_path.parent.mkdir(parents=True, exist_ok=True)
    config.stats_path.write_text(
        json.dumps(bundle.stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def prepare_data(config: Config) -> DataBundle:
    """Build a ``DataBundle`` from the raw Aksharantar files (slow; result is cached)."""
    # Step 1-2: read and clean all three splits.
    train_all = clean_pairs(iter_jsonl(config.train_path), config.max_len)
    val_all = clean_pairs(iter_jsonl(config.valid_path), config.max_len)
    test_all = clean_pairs(iter_jsonl(config.test_path), config.max_len)

    # Step 3: leakage check. A training pair identical to a validation or test pair is
    # dropped. (``removed`` is 0 for Aksharantar Hindi: its splits are already disjoint.)
    held_out = set(val_all) | set(test_all)
    before = len(train_all)
    train_all = [pair for pair in train_all if pair not in held_out]
    removed = before - len(train_all)

    # Step 4a: pick the training subset (100,000 pairs by default) with a fixed seed.
    rng = random.Random(config.seed)
    if len(train_all) > config.n_train:
        train_pairs = rng.sample(train_all, config.n_train)
    else:
        train_pairs = list(train_all)

    # Step 4b: reference lists and the words that will be scored. Validation accuracy is
    # computed on val_words; the final results use test_words.
    val_refs = build_refs(val_all)
    test_refs = build_refs(test_all)
    val_words = cap_words(val_refs.keys(), config.n_val_acc, config.seed)
    test_words = cap_words(test_refs.keys(), config.n_test_max, config.seed)
    # A separate sample of validation *pairs* for the validation loss.
    val_pairs_loss = random.Random(config.seed).sample(
        val_all, min(config.n_val_loss, len(val_all))
    )

    # Extra sanity statistic: what share of test Roman words also appear in the training
    # subset? It was 0.0 in our run, i.e. test words are genuinely unseen.
    train_roman = {roman for roman, _ in train_pairs}
    overlap = sum(word in train_roman for word in test_words) / max(1, len(test_words))
    stats = {
        "cleaned_train": before,
        "cleaned_valid": len(val_all),
        "cleaned_test": len(test_all),
        "removed_leaked_train_pairs": removed,
        "train_pairs_used": len(train_pairs),
        "val_loss_pairs": len(val_pairs_loss),
        "val_words": len(val_words),
        "test_words": len(test_words),
        "test_roman_overlap_with_train": overlap,
        "max_len": config.max_len,
        "seed": config.seed,
    }
    bundle = DataBundle(
        train_pairs=train_pairs,
        val_pairs_loss=val_pairs_loss,
        val_refs=val_refs,
        test_refs=test_refs,
        val_words=val_words,
        test_words=test_words,
        stats=stats,
    )
    # Step 5: cache for next time.
    save_bundle(bundle, config.prepared_path, config)
    return bundle


def load_bundle(config: Config) -> DataBundle:
    """Return the cached bundle if it matches the current settings, else rebuild it."""
    path = config.prepared_path
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Reuse the cache only if the data-selection settings are unchanged.
        if payload.get("signature") == signature(config):
            return DataBundle(
                # JSON turns tuples into lists, so convert the pairs back to tuples.
                train_pairs=[tuple(pair) for pair in payload["train_pairs"]],
                val_pairs_loss=[tuple(pair) for pair in payload["val_pairs_loss"]],
                val_refs=payload["val_refs"],
                test_refs=payload["test_refs"],
                val_words=payload["val_words"],
                test_words=payload["test_words"],
                stats=payload["stats"],
            )
    return prepare_data(config)
