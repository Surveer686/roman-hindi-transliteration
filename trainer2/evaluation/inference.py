"""Predict candidates for single tokens with any system (our models or IndicXlit)."""
from __future__ import annotations

import torch

from ..config import Config
from ..datasets.vocab import Vocab
from ..models.decode import hypothesis_texts, predict
from ..models.indicxlit import IndicXlitBaseline

# A system is one of our models or the IndicXlit baseline
System = torch.nn.Module | IndicXlitBaseline


def predict_tokens_topk(
    tokens: list[str],
    system: System,
    config: Config,
    source_vocab: Vocab,
    target_vocab: Vocab,
    device: torch.device,
    beam: int | None = None,
    batch_size: int | None = None,
) -> list[list[str]]:
    """Ranked candidates (best first) for every token."""
    # Our own model: use beam search
    if isinstance(system, torch.nn.Module):
        # Ranked (text, score) pairs for each token
        hypotheses = predict(
            system,
            tokens,
            source_vocab,
            target_vocab,
            device,
            beam=beam if beam is not None else config.beam,
            batch_size=batch_size if batch_size is not None else config.eval_batch_size,
            max_len=config.max_decode_len,
        )
    # IndicXlit has its own predict method
    else:
        hypotheses = system.predict(tokens)
    # Drop the scores
    return hypothesis_texts(hypotheses)


def predict_tokens(
    tokens: list[str],
    system: System,
    config: Config,
    source_vocab: Vocab,
    target_vocab: Vocab,
    device: torch.device,
    beam: int | None = None,
    batch_size: int | None = None,
) -> list[str]:
    """Best candidate only, for every token."""
    # Get all candidates
    candidates = predict_tokens_topk(
        tokens, system, config, source_vocab, target_vocab, device, beam, batch_size
    )
    # Keep the first candidate of each token
    return [options[0] if options else "" for options in candidates]
