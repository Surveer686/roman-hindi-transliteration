"""IndicXlit baseline (CTranslate2 conversion), used only for prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import ctranslate2
from huggingface_hub import snapshot_download

from ..config import Config


class IndicXlitBaseline:
    """Runs the pretrained IndicXlit model like one of our own models."""
    # Name shown in the results
    name = "IndicXlit-raw"

    def __init__(self, config: Config) -> None:
        # Download the model (cached after the first time)
        root = Path(
            snapshot_download(
                repo_id=config.indicxlit_repo,
                allow_patterns=[f"{config.indicxlit_model_dir}/*"],
            )
        )
        # Folder of the converted model
        model_path = root / config.indicxlit_model_dir
        # Load the translator on the CPU
        self.translator = ctranslate2.Translator(
            str(model_path),
            device="cpu",
            inter_threads=config.indicxlit_inter_threads,
            intra_threads=config.indicxlit_intra_threads,
        )
        # Beam width (at least the number of candidates)
        self.beam_size = max(config.indicxlit_beam, config.indicxlit_topk)
        # How many candidates to return
        self.topk = config.indicxlit_topk
        # Words per batch
        self.batch_size = max(1, config.indicxlit_batch_size)
        # Longest output
        self.max_decoding_length = config.max_decode_len

    def predict(self, words: list[str]) -> list[list[tuple[str, float]]]:
        """Return ranked (text, score) candidates for each Roman word."""
        # Each word becomes its letters, after the Hindi language tag
        sources = [
            ["__hi__", *[char for char in word.strip().lower() if char.isalpha()]]
            for word in words
        ]
        # Raw translator results
        results: list[Any] = []
        # Translate in batches
        for start in range(0, len(sources), self.batch_size):
            results.extend(
                self.translator.translate_batch(
                    sources[start : start + self.batch_size],
                    beam_size=self.beam_size,
                    num_hypotheses=self.topk,
                    max_decoding_length=self.max_decoding_length,
                    return_scores=True,
                )
            )
        # Ranked candidates for every word
        output: list[list[tuple[str, float]]] = []
        # Convert each result
        for result in results:
            # Join the characters into text and keep the score
            hypotheses = [
                ("".join(tokens), float(score))
                for tokens, score in zip(result.hypotheses, result.scores)
            ][:self.topk]
            # An empty candidate if nothing came back
            output.append(hypotheses if hypotheses else [("", 0.0)])
        # One ranked list per word
        return output
