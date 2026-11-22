"""Character vocabularies and batch collation.

The models work on *characters*: the Roman input ``"kharab"`` and the Devanagari output
``"खराब"`` are each turned into a list of integer ids. This module defines

* the special ids (padding, start, end, unknown),
* ``Vocab``: a two-way map between characters and ids,
* ``build_vocabs`` / ``save_vocabs`` / ``load_vocabs``: create and persist the source
  (Roman) and target (Devanagari) vocabularies,
* ``pad_batch`` / ``collate``: turn a list of words into one padded tensor for training.

Used by the models (padding id), the decoder, the trainer and every evaluation.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from string import ascii_lowercase, digits
from typing import Iterable

import torch

# Ids 0-3 are reserved and must stay in this order: saved checkpoints and vocab.json
# files depend on it.
PAD, SOS, EOS, UNK = 0, 1, 2, 3  # padding, start-of-sequence, end-of-sequence, unknown
SPECIAL = ["<pad>", "<sos>", "<eos>", "<unk>"]


class Vocab:
    """Maps characters to integer ids and back (ids 0-3 are the special tokens)."""

    def __init__(self, chars: Iterable[str]) -> None:
        # Special tokens first, then the characters in sorted order -> deterministic ids.
        self.itos = list(SPECIAL) + sorted(set(chars))  # id -> character
        self.stoi = {char: index for index, char in enumerate(self.itos)}  # character -> id

    def encode(self, text: str) -> list[int]:
        """Text -> ids. A character not in the vocabulary becomes ``UNK``."""
        return [self.stoi.get(char, UNK) for char in text]

    def decode(self, ids: Iterable[int]) -> str:
        """Ids -> text. Stops at ``EOS`` and skips padding and start tokens."""
        pieces: list[str] = []
        for index in ids:
            if index == EOS:
                break  # everything after the end marker is ignored
            if index in (PAD, SOS):
                continue
            if 0 <= index < len(self.itos):
                pieces.append(self.itos[index])
        return "".join(pieces)

    def __len__(self) -> int:
        return len(self.itos)


def build_vocabs(train_pairs: list[tuple[str, str]]) -> tuple[Vocab, Vocab]:
    """Create the (source, target) vocabularies from the training pairs.

    * Source (Roman): always all of a-z and 0-9, plus any other character seen in training.
      Including every letter means test words never contain an unseen Roman letter.
    * Target (Devanagari): only characters that occur at least twice in the training data.
      A character seen once would be learned from a single example, so it is dropped and
      mapped to ``UNK`` if it ever appears.
    """
    source_chars = set(ascii_lowercase + digits)
    source_chars.update(char for roman, _ in train_pairs for char in roman)
    counts = Counter(char for _, native in train_pairs for char in native)
    target_chars = {char for char, count in counts.items() if count >= 2}
    return Vocab(source_chars), Vocab(target_chars)


def save_vocabs(path: Path, source_vocab: Vocab, target_vocab: Vocab) -> None:
    """Write both vocabularies to ``vocab.json`` (as the full id -> character lists)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"src": source_vocab.itos, "tgt": target_vocab.itos}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_vocabs(path: Path) -> tuple[Vocab, Vocab]:
    """Read ``vocab.json`` back. Re-creating from the saved lists gives identical ids."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    # Skip the 4 special tokens: ``Vocab`` adds them itself.
    return Vocab(payload["src"][len(SPECIAL):]), Vocab(payload["tgt"][len(SPECIAL):])


def pad_batch(sequences: list[list[int]]) -> torch.Tensor:
    """Stack variable-length id lists into one ``(batch, longest)`` tensor, padding with PAD."""
    length = max(len(sequence) for sequence in sequences)
    output = torch.full((len(sequences), length), PAD, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        output[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return output


def collate(
    batch: list[tuple[str, str]], source_vocab: Vocab, target_vocab: Vocab
) -> tuple[torch.Tensor, torch.Tensor]:
    """DataLoader collate function: a list of (roman, native) pairs -> (source, target) tensors.

    The source is the encoded Roman word. The target is ``<sos> chars... <eos>`` so the
    decoder learns where a word starts and ends.
    """
    source = [source_vocab.encode(roman) for roman, _ in batch]
    target = [[SOS, *target_vocab.encode(native), EOS] for _, native in batch]
    return pad_batch(source), pad_batch(target)
