"""Rule-based synthetic spelling noise and the paired training dataset.

Why this exists: dictionary data has careful spellings (``kharab``), but people on social
media write ``kharaab``, ``kharb`` or ``kharabbb``. To study that gap we *simulate* messy
spellings by applying small random edits to the Roman side of clean pairs.

How it works:
* There are six noise **rules** (phonetic swap, ending swap, dropped vowels, elongation,
  single-character edit, digit substitution), each a small function ``word -> word``.
* ``noisify(word, level)`` runs every rule in turn; each rule fires with probability
  ``min(1, level * weight)``. A higher ``level`` therefore means more edits.
* ``PairDataset`` uses ``noisify`` while training the *noisy* models (half of the
  examples get noised input; the Devanagari target is never changed).
* ``evaluation/word_eval.py`` uses ``noisify`` at levels 0.0 ... 0.5 to build the noisy
  test inputs.

Important limitation (also stated in the report): the same rules are used to train the
noisy models and to create the noisy test words, so gains on synthetic noise do not by
themselves prove gains on real spellings.
"""
from __future__ import annotations

import math
import random
import re

from torch.utils.data import Dataset

from ..evaluation.metrics import edit_distance

VOWELS = "aeiou"

# Phonetic swaps: (text to find, replacement). One matching swap is chosen at random and
# applied at one random position. They mimic common alternative Roman spellings of the
# same Hindi sound, e.g. "kam" -> "kaam", "khush" -> "kush", "film" -> "philm".
PHON_SWAPS = [
    ("ee", "i"), ("i", "ee"), ("ii", "i"), ("aa", "a"), ("a", "aa"),
    ("oo", "u"), ("u", "oo"), ("v", "w"), ("w", "v"), ("ph", "f"),
    ("f", "ph"), ("sh", "s"), ("s", "sh"), ("z", "j"), ("j", "z"),
    ("kh", "k"), ("ck", "k"), ("c", "k"), ("ai", "ae"), ("ai", "e"),
    ("au", "ow"),
]
# Ending swaps: only applied when the word *ends* with the pattern, e.g. "hai" -> "h",
# "kya" -> "kyaa", "hoon" -> "hu", "mein" -> "men" (texting style).
FINAL_SWAPS = [
    ("ai", "e"), ("ai", "ay"), ("ain", "an"), ("ein", "en"),
    ("ya", "yaa"), ("hai", "h"), ("oon", "un"), ("hoon", "hu"),
]
# SMS-style digits: "to" -> "2", "for" -> "4" (rare, hence its low weight below).
DIGIT_SWAPS = [("to", "2"), ("for", "4")]

# (rule name, weight). The weight scales how often the rule fires: probability =
# min(1, level * weight). Example at level 0.3: phonetic 0.36, final 0.30, drop_vowels
# 0.30, elongate 0.24, char_edit 0.15, digit 0.03. Order matters: rules are applied in
# this order and each works on the output of the previous one.
NOISE_RULES = [
    ("phonetic", 1.2),
    ("final", 1.0),
    ("drop_vowels", 1.0),
    ("elongate", 0.8),
    ("char_edit", 0.5),
    ("digit", 0.1),
]


def _replace_one(word: str, old: str, new: str, rng: random.Random) -> str:
    """Replace ONE randomly chosen occurrence of ``old`` by ``new`` (if there is one)."""
    # re.escape: treat ``old`` as plain text, not as a regular expression.
    positions = [match.start() for match in re.finditer(re.escape(old), word)]
    if not positions:
        return word
    position = rng.choice(positions)
    return word[:position] + new + word[position + len(old):]


def _phonetic(word: str, rng: random.Random) -> str:
    """Apply one random phonetic swap that is applicable to this word."""
    options = [(old, new) for old, new in PHON_SWAPS if old in word]
    if not options:
        return word  # nothing in the word can be swapped
    return _replace_one(word, *rng.choice(options), rng)


def _final(word: str, rng: random.Random) -> str:
    """Rewrite the word's ending (``hai`` -> ``h``, ``hoon`` -> ``hu``)."""
    options = [(old, new) for old, new in FINAL_SWAPS if word.endswith(old)]
    if not options:
        return word
    old, new = rng.choice(options)
    return word[:len(word) - len(old)] + new


def _drop_vowels(word: str, rng: random.Random) -> str:
    """Delete some vowels (``bahut`` -> ``bhut``), a very common texting habit."""
    # Index > 0: never drop the first letter, so the word stays recognisable.
    indices = [i for i, char in enumerate(word) if char in VOWELS and i > 0]
    if not indices:
        return word
    # Drop between 1 and half of the (rounded-up) vowels.
    count = rng.randint(1, max(1, (len(indices) + 1) // 2))
    drop = set(rng.sample(indices, min(count, len(indices))))
    result = "".join(char for i, char in enumerate(word) if i not in drop)
    # Guard: do not shrink a word to a single letter.
    return result if len(result) >= 2 else word


def _elongate(word: str, rng: random.Random) -> str:
    """Repeat a vowel or the last letter 1-3 extra times (``kaam`` -> ``kaaaam``)."""
    candidates = [i for i, char in enumerate(word) if char in VOWELS] + [len(word) - 1]
    index = rng.choice(candidates)
    return word[:index + 1] + word[index] * rng.randint(1, 3) + word[index + 1:]


def _char_edit(word: str, rng: random.Random) -> str:
    """One typo: swap two neighbouring letters, duplicate a letter, or delete a letter."""
    operation = rng.choice(["swap", "dup", "del"])
    if operation == "swap" and len(word) >= 3:
        position = rng.randrange(0, len(word) - 1)
        # Exchange the letters at ``position`` and ``position + 1``.
        return word[:position] + word[position + 1] + word[position] + word[position + 2:]
    if operation == "dup":
        position = rng.randrange(0, len(word))
        return word[:position] + word[position] + word[position:]
    if operation == "del" and len(word) >= 4:
        # Start at 1 so the first letter is never deleted.
        position = rng.randrange(1, len(word))
        return word[:position] + word[position + 1:]
    return word  # the chosen operation was not allowed for such a short word


def _digit(word: str, rng: random.Random) -> str:
    """Replace ``to`` / ``for`` inside the word by a digit (``toh`` -> ``2h``)."""
    options = [(old, new) for old, new in DIGIT_SWAPS if old in word]
    if not options:
        return word
    return _replace_one(word, *rng.choice(options), rng)


# Name -> function, so ``noisify`` can look a rule up by the names in NOISE_RULES.
RULES = {
    "phonetic": _phonetic,
    "final": _final,
    "drop_vowels": _drop_vowels,
    "elongate": _elongate,
    "char_edit": _char_edit,
    "digit": _digit,
}


def default_weights() -> dict[str, float]:
    """The built-in rule weights as a ``{name: weight}`` dict."""
    return {name: weight for name, weight in NOISE_RULES}


def resolve_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Merge user-supplied weights over the defaults (unknown rule names are ignored)."""
    resolved = default_weights()
    if weights:
        for name, weight in weights.items():
            if name in resolved:
                resolved[name] = float(weight)
    return resolved


def noisify(
    word: str,
    level: float,
    rng: random.Random,
    weights: dict[str, float] | None = None,
) -> str:
    """Return a noisy spelling of ``word``.

    Args:
        word: clean lowercase Roman word, e.g. ``"kharab"``.
        level: noise strength. ``0`` returns the word unchanged; ``0.5`` applies many
            edits; at ``1.0`` every rule with weight >= 1 always fires.
        rng: random generator. Passing a seeded ``random.Random`` makes the result
            reproducible, which is how all systems get exactly the same noisy test words.
        weights: optional per-rule weights replacing the defaults.
    """
    if level <= 0 or len(word) < 2:
        return word  # level 0 = clean; one-letter words are left alone
    result = word
    for name, default in NOISE_RULES:
        weight = weights.get(name, default) if weights else default
        # Each rule independently fires with probability min(1, level * weight).
        if rng.random() < min(1.0, level * weight):
            result = RULES[name](result, rng)
    # ``or word`` is a safety net: never return an empty string.
    return result or word


def tune_noise_weights(
    pairs: list[tuple[str, str]],
    level: float = 0.3,
    trials: int = 60,
    samples: int = 3,
    seed: int = 0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit the rule weights so synthetic noise looks like REAL observed spellings.

    ``pairs`` are ``(clean_spelling, observed_noisy_spelling)`` taken from annotated
    tweets. This is a simple random search:

    1. Score the default weights: for each pair, generate ``samples`` noisy versions of
       the clean word and keep the one closest (edit distance) to the observed spelling;
       average that distance over all pairs. Lower is better.
    2. Propose ``trials`` candidates, each multiplying every default weight by a random
       factor between 0.25x and 4x (log-uniform), and keep the best-scoring one.

    Returns ``(best_weights, report)``. It runs only with ``--tune-noise`` and annotated
    dev tweets; in our reported experiments no such pairs existed, so the default weights
    were used.
    """
    base = default_weights()
    report: dict[str, float] = {
        "pairs": float(len(pairs)),
        "trials": 0.0,
        "level": level,
        "base_score": 0.0,
        "best_score": 0.0,
    }
    if not pairs:
        return base, report  # nothing to fit against

    def score(weights: dict[str, float], rng_seed: int) -> float:
        """Average best-of-``samples`` edit distance to the observed spelling."""
        local_rng = random.Random(rng_seed)
        total = 0.0
        for clean, observed in pairs:
            best = min(
                edit_distance(noisify(clean, level, local_rng, weights), observed)
                for _ in range(samples)
            )
            total += best
        return total / len(pairs)

    base_score = score(base, seed)
    best_weights = dict(base)
    best_score = base_score
    proposal_rng = random.Random(seed + 1)
    # Multiplicative range for a candidate weight: exp(uniform(log 0.25, log 4)).
    low, high = math.log(0.25), math.log(4.0)
    for trial in range(max(0, trials)):
        candidate = {
            name: value * math.exp(proposal_rng.uniform(low, high))
            for name, value in base.items()
        }
        # A different seed per trial gives each candidate its own noise draws.
        candidate_score = score(candidate, seed + 1_000_003 * (trial + 1))
        if candidate_score < best_score:
            best_score = candidate_score
            best_weights = candidate

    report.update(
        {
            "trials": float(max(0, trials)),
            "base_score": round(base_score, 6),
            "best_score": round(best_score, 6),
        }
    )
    return best_weights, report


class PairDataset(Dataset):
    """Training dataset of ``(roman, native)`` pairs with optional on-the-fly noise.

    * ``noisy_prob = 0`` -> the *clean* models: the Roman input is returned unchanged.
    * ``noisy_prob = 0.5`` -> the *noisy* models: half of the examples get a noised Roman
      input, with a noise level drawn uniformly from ``level_range`` (0.1-0.6). The
      Devanagari target is always unchanged, so the model learns that noisy spellings map
      to the same word.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        noisy_prob: float = 0.0,
        level_range: tuple[float, float] = (0.1, 0.6),
        seed: int = 0,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.pairs = list(pairs)
        self.noisy_prob = noisy_prob
        self.level_range = level_range
        self.seed = seed
        self.weights = weights
        self.epoch = 0
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the noise generator: new noise every epoch, yet reproducible."""
        self.epoch = epoch
        self.rng = random.Random(self.seed + epoch)

    def __getitem__(self, index: int) -> tuple[str, str]:
        roman, native = self.pairs[index]
        # With probability ``noisy_prob`` replace the Roman input by a noised version.
        if self.noisy_prob > 0 and self.rng.random() < self.noisy_prob:
            roman = noisify(
                roman,
                self.rng.uniform(*self.level_range),
                self.rng,
                self.weights,
            )
        return roman, native
