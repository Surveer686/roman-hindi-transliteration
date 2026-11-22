"""Word-level evaluation: every system on clean and noised dictionary words.

What happens here (``evaluate_models``):

1. Take the test words (9,991 unique Roman words) and their accepted Devanagari spellings.
2. Create the noisy versions of those words ONCE, at every noise level, with a fixed seed.
   Every system is then scored on *exactly the same inputs*, which keeps the comparison fair.
3. For each system (4 trained models + IndicXlit) and each noise level, produce ranked
   candidate lists (beam search, up to 10 candidates per word) and score them with
   ``evaluation.metrics.evaluate`` -> top-1 / top-5 / top-10 accuracy and CER.
4. ``save_evaluation`` writes ``results.csv``, the per-word predictions, the
   accuracy-vs-noise plot, the error breakdown and a few qualitative examples.

Called from ``cli/pipeline.py``. Results of our run: ``runs/limit100k_ep15_bs256/``.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # draw plots to files, without needing a display window
import matplotlib.pyplot as plt
import pandas as pd
import torch

from ..config import Config
from ..datasets.noise import noisify
from ..datasets.vocab import Vocab
from ..datasets.word_data import DataBundle
from ..models.decode import hypothesis_texts, predict
from .error_analysis import (
    Predictions,
    error_rows,
    save_error_breakdown,
    save_qualitative_examples,
)
from .metrics import evaluate


@dataclass
class EvaluationResult:
    """Everything produced by one evaluation run, passed on to ``save_evaluation``."""

    # One dict per (system, noise level): top1/top5/top10, CER, intervals.
    rows: list[dict[str, Any]]
    # (system name, noise level) -> for each test word, a ranked list of (text, log-score).
    predictions: Predictions
    # noise level -> the noisy Roman input used for each test word.
    noisy_inputs: dict[float, list[str]]
    test_words: list[str]
    # For each test word, the list of accepted Devanagari spellings.
    test_refs_list: list[list[str]]
    # Every system that was evaluated (name -> model / baseline object).
    models: dict[str, Any]


def make_noisy_inputs(
    words: list[str],
    levels: tuple[float, ...],
    seed: int = 1234,
    weights: dict[str, float] | None = None,
) -> dict[float, list[str]]:
    """Noise every test word once per level, reproducibly.

    The fixed ``seed`` and the single shared generator mean that rerunning gives the same
    noisy words, and that all systems see identical noisy inputs. Level 0.0 returns the
    words unchanged (the "clean" test).
    """
    rng = random.Random(seed)
    return {
        level: [noisify(word, level, rng, weights) for word in words]
        for level in levels
    }


def _result_row(name: str, level: float, metrics: dict[str, float]) -> dict[str, Any]:
    """Flatten one metrics dict into a CSV row for ``results.csv``.

    Column meaning: ``top1/top5/top10`` = accuracy at that rank (with ``_lo``/``_hi`` =
    95% bootstrap interval); ``cer`` = micro CER (total edits / total gold characters);
    ``cer_macro`` = average of the per-word CER values.
    """
    row: dict[str, Any] = {"model": name, "noise": level}
    for key in ("top1", "top5", "top10"):
        row[key] = metrics[key]
        row[f"{key}_lo"] = metrics[f"{key}_lo"]
        row[f"{key}_hi"] = metrics[f"{key}_hi"]
    row["cer"] = metrics["cer_micro"]
    row["cer_lo"] = metrics["cer_micro_lo"]
    row["cer_hi"] = metrics["cer_micro_hi"]
    row["cer_macro"] = metrics["cer_macro"]
    row["cer_macro_lo"] = metrics["cer_macro_lo"]
    row["cer_macro_hi"] = metrics["cer_macro_hi"]
    return row


def evaluate_models(
    config: Config,
    bundle: DataBundle,
    models: dict[str, Any],
    source_vocab: Vocab,
    target_vocab: Vocab,
    device: torch.device,
    baselines: dict[str, Any] | None = None,
    noise_weights: dict[str, float] | None = None,
) -> EvaluationResult:
    """Score all systems on the clean and noised test words.

    Args:
        models: the four trained PyTorch models (name -> model).
        baselines: external systems with a ``predict(words)`` method, i.e. IndicXlit.
        noise_weights: optional tuned noise-rule weights (``None`` = defaults).
    """
    config.make_dirs()
    test_words = list(bundle.test_words)
    test_refs_list = [list(bundle.test_refs[word]) for word in test_words]
    # Seed 1234 is fixed: the noisy test set never changes between runs.
    noisy_inputs = make_noisy_inputs(
        test_words, config.noise_levels, 1234, noise_weights
    )
    predictions: Predictions = {}
    rows: list[dict[str, Any]] = []

    # --- the four models trained from scratch (GRU / Transformer, clean / noisy) ---
    for name, model in models.items():
        model.eval()  # inference mode: no dropout
        for level in config.noise_levels:
            # Beam search returns up to ``beam`` candidates per word, best first, each as
            # (devanagari_text, log_probability).
            hypotheses = predict(
                model,
                noisy_inputs[level],
                source_vocab,
                target_vocab,
                device,
                beam=config.beam,
                batch_size=config.eval_batch_size,
                max_len=config.max_decode_len,
            )
            predictions[(name, level)] = hypotheses
            # Drop the scores, then compute top-1/5/10 and CER against the references.
            metrics = evaluate(
                hypothesis_texts(hypotheses), test_refs_list, ks=(1, 5, 10)
            )
            rows.append(_result_row(name, level, metrics))
        print(f"[eval] evaluated {name}", flush=True)

    # --- external baselines (IndicXlit): same inputs, same scoring ---
    for name, baseline in (baselines or {}).items():
        for level in config.noise_levels:
            hypotheses = baseline.predict(noisy_inputs[level])
            predictions[(name, level)] = hypotheses
            metrics = evaluate(
                hypothesis_texts(hypotheses), test_refs_list, ks=(1, 5, 10)
            )
            rows.append(_result_row(name, level, metrics))
        print(f"[eval] evaluated {name}", flush=True)

    all_systems = dict(models)
    all_systems.update(baselines or {})
    return EvaluationResult(
        rows=rows,
        predictions=predictions,
        noisy_inputs=noisy_inputs,
        test_words=test_words,
        test_refs_list=test_refs_list,
        models=all_systems,
    )


def best_model(rows: list[dict[str, Any]], level: float = 0.3) -> str:
    """Name of the system with the highest top-1 at ``level`` (falls back to the largest
    available noise level). Used to pick which model to show in a demo."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return ""
    available = set(frame["noise"].unique())
    chosen = level if level in available else max(available)
    subset = frame[frame["noise"] == chosen]
    return str(subset.sort_values("top1", ascending=False).iloc[0]["model"])


def plot_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    """Draw top-1, top-5 and CER against noise level, one line per system (Figure 1)."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for name in frame["model"].unique():
        subset = frame[frame["model"] == name]
        axes[0].plot(subset["noise"], subset["top1"], marker="o", label=name)
        axes[1].plot(subset["noise"], subset["top5"], marker="o", label=name)
        axes[2].plot(subset["noise"], subset["cer"], marker="o", label=name)
    for axis, title in zip(
        axes, ("Top-1 accuracy", "Top-5 accuracy", "CER (lower is better)")
    ):
        axis.set_title(title + " vs synthetic noise level")
        axis.set_xlabel("noise level")
        axis.grid(alpha=0.3)
    axes[0].legend()  # one legend (first panel) is enough: all panels share the lines
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)  # free the figure's memory


def save_evaluation(result: EvaluationResult, config: Config) -> None:
    """Write every output file of the word-level evaluation to ``config.out_dir``.

    Files: ``results.csv`` (the main table), ``test_predictions.jsonl`` (every word's top-10
    candidates for every system and noise level; large), ``accuracy_vs_noise.png``,
    ``error_breakdown.csv`` and ``qualitative_examples.txt``.
    """
    config.make_dirs()
    system_names = list(result.models.keys())
    frame = pd.DataFrame(result.rows)
    frame.to_csv(config.out_dir / "results.csv", index=False)

    # One JSON object per line: (system, noise level, word) with its candidates and scores.
    # Kept for follow-up analysis (for example, reading a sample of errors by hand).
    with (config.out_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for name in system_names:
            for level in config.noise_levels:
                key = (name, level)
                if key not in result.predictions:
                    continue
                for word_index, (word, noisy_word, hypotheses) in enumerate(
                    zip(
                        result.test_words,
                        result.noisy_inputs[level],
                        result.predictions[key],
                    )
                ):
                    payload = {
                        "model": name,
                        "noise": level,
                        "word": word,
                        "input": noisy_word,
                        "refs": result.test_refs_list[word_index],
                        "top_k": [[text, score] for text, score in hypotheses],
                    }
                    handle.write(
                        json.dumps(payload, ensure_ascii=False) + "\n"
                    )

    plot_metrics(result.rows, config.out_dir / "accuracy_vs_noise.png")
    # Classify every top-1 error into a linguistic category (see error_analysis.py).
    rows = error_rows(
        result.predictions, result.test_refs_list, system_names, config.noise_levels
    )
    save_error_breakdown(rows, config.out_dir / "error_breakdown.csv")
    save_qualitative_examples(
        result.predictions,
        result.test_words,
        result.noisy_inputs,
        result.test_refs_list,
        config,
        system_names,
    )
    print(f"[eval] saved artifacts to {config.out_dir}", flush=True)
