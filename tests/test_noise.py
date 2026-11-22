"""Tests for the synthetic spelling-noise generator (datasets/noise.py)."""
import random

import pytest

from trainer2.datasets.noise import noisify

WORDS = ["kharab", "achhi", "nahi", "mein", "kaafi", "bahut", "sunset", "padhai"]


def test_level_zero_is_identity():
    """Noise level 0 is the 'clean' test: every word must come back unchanged."""
    assert all(noisify(word, 0.0, random.Random(0)) == word for word in WORDS)


def test_single_character_words_are_untouched():
    """One-letter words are never edited (noisify returns them as they are)."""
    assert noisify("a", 0.9, random.Random(0)) == "a"


@pytest.mark.parametrize("level", [0.1, 0.3, 0.5])
def test_same_seed_gives_same_noise(level):
    """Same seed -> same noise, so all systems get identical noisy test words."""
    first = [noisify(word, level, random.Random(7)) for word in WORDS]
    second = [noisify(word, level, random.Random(7)) for word in WORDS]
    assert first == second


@pytest.mark.parametrize("level", [0.1, 0.5, 1.0])
def test_output_stays_lowercase_alphanumeric_and_non_empty(level):
    """No rule may produce an empty string or characters outside a-z and 0-9."""
    rng = random.Random(3)
    for _ in range(200):
        for word in WORDS:
            noisy = noisify(word, level, rng)
            assert noisy and noisy.isalnum() and noisy == noisy.lower()


def test_high_noise_changes_most_words():
    """Sanity check: at level 1.0 most words are actually changed."""
    rng = random.Random(5)
    changed = sum(noisify(word, 1.0, rng) != word for word in WORDS * 20)
    assert changed > len(WORDS) * 20 * 0.5
