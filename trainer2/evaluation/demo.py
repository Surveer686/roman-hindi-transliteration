"""Demo: transliterate whole sentences (not used by the pipeline)."""
from __future__ import annotations

import re
from typing import Sequence

import torch

from ..config import Config
from ..datasets.vocab import Vocab
from ..models.decode import predict

# Matches links
URL_RE = re.compile(r"^(https?://|www\.)", re.I)
# Example sentences for the demo
DEMO_SENTENCES = (
    "kya haal hai",
    "mujhe tumse milna hai yaar",
    "aaj ka din bahut accha tha #blessed",
    "bhai kya scene hai @rahul check http://t.co/x 2 baje",
)


def transliterate_sentence(
    sentence: str,
    model: torch.nn.Module,
    config: Config,
    source_vocab: Vocab,
    target_vocab: Vocab,
    device: torch.device,
    beam: int = 5,
) -> str:
    """Transliterate the Roman words of a sentence; keep links, mentions and hashtags."""
    # Pieces of the result
    output: list[str] = []
    # Split into words and spaces
    for part in re.split(r"(\s+)", sentence):
        # Keep spaces as they are
        if not part or part.isspace():
            output.append(part)
            continue
        # Split off punctuation around the word
        match = re.match(r"^(\W*)(.*?)(\W*)$", part, re.S)
        # Punctuation before, the word, punctuation after
        before, core, after = match.groups() if match else ("", part, "")
        # Leave links, mentions, hashtags and non-letters unchanged
        if (
            URL_RE.match(part)
            or "#" in before
            or "@" in before
            or not re.fullmatch(r"[A-Za-z]+", core or "")
        ):
            output.append(part)
            continue
        # Transliterate the word
        predictions = predict(
            model,
            [core.lower()],
            source_vocab,
            target_vocab,
            device,
            beam=beam,
            batch_size=1,
            max_len=config.max_decode_len,
        )
        # Take the best candidate
        best = predictions[0][0][0] if predictions and predictions[0] else core
        # Put the punctuation back
        output.append(before + best + after)
    # Join the pieces
    return "".join(output)


def save_sentence_demo(
    models: dict[str, torch.nn.Module],
    best_name: str,
    config: Config,
    source_vocab: Vocab,
    target_vocab: Vocab,
    device: torch.device,
    sentences: Sequence[str] = DEMO_SENTENCES,
) -> None:
    """Write sentence_demo.txt with the demo sentences and their transliteration."""
    # The first line names the model
    lines = [f"using: {best_name}"]
    # Transliterate each sentence
    for sentence in sentences:
        # Get the transliteration
        translated = transliterate_sentence(
            sentence, models[best_name], config, source_vocab, target_vocab, device
        )
        # Add input and output
        lines.append(f"{sentence}\n  -> {translated}")
    # Write the demo file
    path = config.out_dir / "sentence_demo.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
