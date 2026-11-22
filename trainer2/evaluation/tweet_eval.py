"""Tweet-level evaluation: token top-k, CER, sentence accuracy and intervals."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from ..config import Config
from ..datasets.noise import noisify
from ..datasets.tweet_data import TweetBundle
from ..datasets.vocab import Vocab
from .inference import System, predict_tokens_topk
from .metrics import cluster_bootstrap, edit_distance


def noisy_tweet_tokens(
    bundle: TweetBundle,
    level: float,
    rng: random.Random,
    weights: dict[str, float] | None = None,
) -> dict[int, list[str]]:
    """Add noise to the scored tokens of every tweet (level 0 = unchanged)."""
    # Noisy tokens of each tweet
    output: dict[int, list[str]] = {}
    # Go through the tweets
    for index, tweet in enumerate(bundle.tweets):
        # Only the scored tokens get noise
        scored = set(tweet["scored_indices"])
        # Noise the scored tokens, keep the others
        output[index] = [
            noisify(token, level, rng, weights) if position in scored else token
            for position, token in enumerate(tweet["tokens"])
        ]
    return output


def _predict_tweets(
    config: Config,
    bundle: TweetBundle,
    system: System,
    noisy: dict[int, list[str]],
    source_vocab: Vocab,
    target_vocab: Vocab,
    device: torch.device,
) -> list[list[list[str]]]:
    """Ranked candidates for every scored token, grouped by tweet."""
    # All scored tokens of all tweets in one list
    flat_tokens: list[str] = []
    # Where each tweet's tokens sit in that list
    spans: list[tuple[int, int]] = []
    # Collect the scored tokens of each tweet
    for index, tweet in enumerate(bundle.tweets):
        # Where this tweet starts
        start = len(flat_tokens)
        for position in tweet["scored_indices"]:
            flat_tokens.append(noisy[index][position])
        # Where it ends
        spans.append((start, len(flat_tokens)))
    # Nothing to predict
    if not flat_tokens:
        return [[] for _ in bundle.tweets]
    # Predict all tokens in one go
    predicted = predict_tokens_topk(
        flat_tokens, system, config, source_vocab, target_vocab, device
    )
    # Split the candidates back per tweet
    return [predicted[start:end] for start, end in spans]


def _top1(candidates: list[list[str]]) -> list[str]:
    """Keep only the best candidate of each token."""
    # Best candidate, or empty if there is none
    return [options[0] if options else "" for options in candidates]


def _render(tokens: list[str], positions: list[int], values: list[str]) -> str:
    """Put predicted words back into the tweet and join them."""
    # Start from the original tokens
    output = list(tokens)
    # Replace the scored positions
    for position, value in zip(positions, values):
        output[position] = value
    # Join into one sentence
    return " ".join(output)


def evaluate_tweets(
    config: Config,
    bundle: TweetBundle,
    systems: dict[str, System],
    source_vocab: Vocab,
    target_vocab: Vocab,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score every system on the tweets; returns result rows and per-tweet predictions."""
    # Optional tuned noise weights
    weights = config.noise_weights or None
    # Noisy tokens for each noise level (fixed seeds)
    noisy_by_level = {
        level: noisy_tweet_tokens(
            bundle, level, random.Random(config.seed + int(level * 100)), weights
        )
        for level in config.noise_levels
    }
    # One result row per system and noise level
    rows: list[dict[str, Any]] = []
    # One prediction row per tweet
    prediction_rows: list[dict[str, Any]] = []
    # Scored token positions of each tweet
    scored_per_tweet = [tweet["scored_indices"] for tweet in bundle.tweets]

    # Evaluate each system
    for name, system in systems.items():
        # At each noise level
        for level in config.noise_levels:
            # Noisy tokens for this level
            noisy = noisy_by_level[level]
            # Ranked candidates for every scored token
            predicted_topk = _predict_tweets(
                config, bundle, system, noisy, source_vocab, target_vocab, device
            )
            # Best candidate of each token
            predicted_sentences = [_top1(tweet) for tweet in predicted_topk]

            # Counters for this system and level
            correct_tokens = 0
            hits_top5 = 0
            hits_top10 = 0
            total_tokens = 0
            edits = 0
            characters = 0
            sentence_cers: list[float] = []
            sentence_matches: list[int] = []
            # Per-tweet counts (for intervals and paired tests)
            per_tweet: dict[str, list[int]] = {
                key: [] for key in ("hits1", "hits5", "hits10", "n_tokens", "edits", "chars")
            }

            # Go through the tweets
            for tweet_index, (tweet, positions, predictions) in enumerate(
                zip(bundle.tweets, scored_per_tweet, predicted_sentences)
            ):
                # Accepted spellings of each scored token
                references = [tweet["gold"][position] for position in positions]
                # Candidates of this tweet
                tweet_topk = predicted_topk[tweet_index]
                # Tokens found within the top 5 and top 10
                tweet_hits5 = tweet_hits10 = 0
                # Is a correct spelling among the first 5 or 10?
                for options, refs in zip(tweet_topk, references):
                    tweet_hits5 += int(any(option in refs for option in options[:5]))
                    tweet_hits10 += int(any(option in refs for option in options[:10]))
                # Add to the totals
                hits_top5 += tweet_hits5
                hits_top10 += tweet_hits10
                # Top-1 hits, edits and gold characters of this tweet
                tweet_hits1 = tweet_edits = tweet_chars = 0
                # Closest accepted spelling of each token
                best_refs: list[str] = []
                # Score each token
                for prediction, refs in zip(predictions, references):
                    total_tokens += 1
                    # Is the top-1 candidate correct?
                    tweet_hit = int(prediction in refs)
                    correct_tokens += tweet_hit
                    tweet_hits1 += tweet_hit
                    # Closest accepted spelling
                    best = (
                        min(refs, key=lambda ref: edit_distance(prediction, ref))
                        if refs
                        else ""
                    )
                    best_refs.append(best)
                    # Character edits for this token
                    distance_token = edit_distance(prediction, best)
                    edits += distance_token
                    characters += len(best)
                    tweet_edits += distance_token
                    tweet_chars += len(best)
                # Counts of this tweet
                tweet_counts = (
                    tweet_hits1,
                    tweet_hits5,
                    tweet_hits10,
                    len(predictions),
                    tweet_edits,
                    tweet_chars,
                )
                # Store them
                for key, value in zip(per_tweet, tweet_counts):
                    per_tweet[key].append(value)

                # Whole-tweet text of the gold and of the prediction
                gold_text = " ".join(best_refs)
                prediction_text = " ".join(predictions)
                # Edits for the whole tweet
                distance = edit_distance(gold_text, prediction_text)
                # CER of this tweet
                sentence_cers.append(distance / max(1, len(gold_text)))
                # 1 if every token of the tweet is exactly right
                sentence_matches.append(
                    int(
                        all(
                            prediction in refs
                            for prediction, refs in zip(predictions, references)
                        )
                    )
                )
                # Save the details of this tweet
                prediction_rows.append(
                    {
                        "system": name,
                        "noise": level,
                        "id": tweet["id"],
                        "roman": tweet["text"],
                        "input": " ".join(noisy[tweet_index]),
                        "gold": " ".join(best_refs),
                        "gold_refs": references,
                        "top5": [options[:5] for options in tweet_topk],
                        "hits1": per_tweet["hits1"][-1],
                        "hits5": per_tweet["hits5"][-1],
                        "hits10": per_tweet["hits10"][-1],
                        "n_tokens": per_tweet["n_tokens"][-1],
                        "prediction": _render(
                            tweet["tokens"], positions, predictions
                        ),
                    }
                )

            # Result row: accuracy, CER and counts
            row: dict[str, Any] = {
                "system": name,
                "noise": level,
                "token_accuracy": correct_tokens / max(total_tokens, 1),
                "token_top5": hits_top5 / max(total_tokens, 1),
                "token_top10": hits_top10 / max(total_tokens, 1),
                "sentence_accuracy": sum(sentence_matches)
                / max(len(sentence_matches), 1),
                "cer": edits / max(characters, 1),
                "cer_macro": sum(sentence_cers) / max(len(sentence_cers), 1),
                "n_tweets": len(bundle.tweets),
                "n_scored_tokens": total_tokens,
            }
            # 95% intervals, resampling whole tweets
            for column, numerator, denominator in (
                ("token_accuracy", "hits1", "n_tokens"),
                ("token_top5", "hits5", "n_tokens"),
                ("token_top10", "hits10", "n_tokens"),
                ("cer", "edits", "chars"),
            ):
                # Interval from the per-tweet counts
                _, low, high = cluster_bootstrap(per_tweet[numerator], per_tweet[denominator])
                row[f"{column}_lo"] = low
                row[f"{column}_hi"] = high
            # Keep the row
            rows.append(row)
        print(f"[tweet-eval] evaluated {name}", flush=True)
    # Results and predictions
    return rows, prediction_rows


def plot_tweet_results(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot token accuracy, sentence accuracy and CER against noise."""
    # Results as a table
    frame = pd.DataFrame(rows)
    # Nothing to plot
    if frame.empty:
        return
    # Three panels side by side
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    # One line per system
    for name in frame["system"].unique():
        subset = frame[frame["system"] == name]
        axes[0].plot(subset["noise"], subset["token_accuracy"], marker="o", label=name)
        axes[1].plot(
            subset["noise"], subset["sentence_accuracy"], marker="o", label=name
        )
        axes[2].plot(subset["noise"], subset["cer"], marker="o", label=name)
    # Titles and labels
    for axis, title in zip(
        axes,
        (
            "Token accuracy (real tweets)",
            "Sentence accuracy (real tweets)",
            "CER, lower is better (real tweets)",
        ),
    ):
        axis.set_title(title)
        axis.set_xlabel("added synthetic noise (0 = as tweeted)")
        axis.grid(alpha=0.3)
    # One legend on the first panel
    axes[0].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save the figure
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_tweet_results(
    config: Config, rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]
) -> None:
    """Save the results table, per-tweet predictions and the plot."""
    config.make_dirs()
    # Results table
    pd.DataFrame(rows).to_csv(config.out_dir / "tweet_results.csv", index=False)
    # Per-tweet predictions, one JSON object per line
    with (config.out_dir / "tweet_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Draw the plot
    plot_tweet_results(rows, config.out_dir / "tweet_noise_comparison.png")


def save_tweet_demo(
    config: Config,
    bundle: TweetBundle,
    systems: dict[str, System],
    source_vocab: Vocab,
    target_vocab: Vocab,
    device: torch.device,
    count: int = 5,
) -> None:
    """Save a readable demo: input, gold and the output of each system."""
    # Nothing to show
    if not bundle.tweets:
        return
    # Text lines of the demo
    lines: list[str] = []
    # The first few tweets
    sample = bundle.tweets[:count]
    # Show the tweets as written and with noise
    for level in (0.0, 0.3):
        # Fixed seed
        rng = random.Random(config.seed + int(level * 100))
        # Noisy tokens
        noisy = noisy_tweet_tokens(bundle, level, rng, config.noise_weights or None)
        # Best output of each system
        predictions = {
            name: [
                _top1(tweet)
                for tweet in _predict_tweets(
                    config, bundle, system, noisy, source_vocab, target_vocab, device
                )
            ]
            for name, system in systems.items()
        }
        lines.append(f"\n===== added noise level {level} =====")
        # Show input, gold and outputs of each tweet
        for index, tweet in enumerate(sample):
            # Positions of the scored tokens
            positions = tweet["scored_indices"]
            lines.append(f"id: {tweet['id']}")
            lines.append(f"tweet: {tweet['text']}")
            lines.append(f"input: {' '.join(noisy[index])}")
            lines.append(
                "gold: "
                + _render(
                    tweet["tokens"],
                    positions,
                    [tweet["gold"][position][0] for position in positions],
                )
            )
            # One line per system
            for name in systems:
                lines.append(
                    f"  {name:18s} {_render(tweet['tokens'], positions, predictions[name][index])}"
                )
            lines.append("")
    # Write the demo file
    path = config.out_dir / "tweet_demo.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
