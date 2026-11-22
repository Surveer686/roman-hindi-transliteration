"""Beam-search decoding shared by the GRU and the Transformer."""
from __future__ import annotations

import torch

from ..datasets.vocab import EOS, PAD, SOS, UNK, Vocab, pad_batch

NEG = -1e9


@torch.no_grad()
def beam_search(
    model: torch.nn.Module,
    src: torch.Tensor,
    target_vocab: Vocab,
    beam: int = 10,
    max_len: int = 40,
    alpha: float = 0.0,
) -> list[list[tuple[str, float]]]:
    model.eval()
    batch_size, device = src.size(0), src.device
    memory, state = model.encode(src)
    expansion = torch.arange(batch_size, device=device).repeat_interleave(beam)
    memory = tuple(item.index_select(0, expansion) for item in memory)
    state = model.reorder_state(state, expansion)
    tokens = torch.full((batch_size * beam, 1), SOS, dtype=torch.long, device=device)
    scores = torch.zeros(batch_size, beam, device=device)
    scores[:, 1:] = NEG
    finished = torch.zeros(batch_size * beam, dtype=torch.bool, device=device)
    base = (torch.arange(batch_size, device=device) * beam).unsqueeze(1)
    for _ in range(max_len):
        log_probabilities, state = model.step(tokens, state, memory)
        vocab_size = log_probabilities.size(-1)
        log_probabilities = log_probabilities.clone()
        log_probabilities[:, PAD] = NEG
        log_probabilities[:, SOS] = NEG
        log_probabilities[:, UNK] = NEG
        log_probabilities[finished] = NEG
        log_probabilities[finished, PAD] = 0.0
        candidates = (scores.view(-1, 1) + log_probabilities).view(
            batch_size, beam * vocab_size
        )
        scores, top = candidates.topk(beam, dim=1)
        beam_indices, token_indices = top // vocab_size, top % vocab_size
        selected = (base + beam_indices).view(-1)
        tokens = torch.cat([tokens[selected], token_indices.view(-1, 1)], dim=1)
        state = model.reorder_state(state, selected)
        finished = (
            finished[selected]
            | (token_indices.view(-1) == EOS)
            | (token_indices.view(-1) == PAD)
        )
        if bool(finished.all()):
            break
    sequences = tokens[:, 1:].view(batch_size, beam, -1).tolist()
    sequence_scores = scores.tolist()
    results: list[list[tuple[str, float]]] = []
    for batch_index in range(batch_size):
        hypotheses = []
        for beam_index in range(beam):
            ids = sequences[batch_index][beam_index]
            length = ids.index(EOS) + 1 if EOS in ids else len(ids)
            score = (
                sequence_scores[batch_index][beam_index] / (length ** alpha)
                if alpha > 0
                else sequence_scores[batch_index][beam_index]
            )
            hypotheses.append((target_vocab.decode(ids), score))
        hypotheses.sort(key=lambda item: -item[1])
        seen: set[str] = set()
        unique: list[tuple[str, float]] = []
        for text, score in hypotheses:
            if text not in seen:
                seen.add(text)
                unique.append((text, score))
        results.append(unique)
    return results


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    words: list[str],
    source_vocab: Vocab,
    target_vocab: Vocab,
    device: torch.device,
    beam: int = 10,
    batch_size: int = 64,
    max_len: int = 40,
    alpha: float = 0.0,
) -> list[list[tuple[str, float]]]:
    model.eval()
    order = sorted(range(len(words)), key=lambda index: len(words[index]))
    results: list[list[tuple[str, float]] | None] = [None] * len(words)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        batch_words = [words[index] if words[index] else "a" for index in indices]
        source = pad_batch([source_vocab.encode(word) for word in batch_words]).to(device)
        hypotheses = beam_search(
            model, source, target_vocab, beam=beam, max_len=max_len, alpha=alpha
        )
        for index, result in zip(indices, hypotheses):
            results[index] = result
    return [result if result is not None else [] for result in results]


def hypothesis_texts(predictions: list[list[tuple[str, float]]]) -> list[list[str]]:
    return [[text for text, _ in prediction] for prediction in predictions]
