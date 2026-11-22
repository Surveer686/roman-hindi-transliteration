"""Evaluate all systems on a tweet-style CSV (id, roman_input, gold_devanagari).
Run: python -m trainer2.cli.evaluate_tweets"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from ..config import ROOT, Config, resolve_device, set_seed
from ..datasets.tweet_data import load_tweet_bundle
from ..datasets.vocab import load_vocabs
from ..evaluation.metrics import paired_diff_ci
from ..evaluation.tweet_eval import evaluate_tweets, save_tweet_demo, save_tweet_results
from ..training.trainer import load_model


def convert(csv_path: Path, jsonl_path: Path) -> tuple[int, list[str]]:
    """Turn the CSV into tweet JSONL; returns the rows kept and the problems."""
    # Rows kept so far
    kept = 0
    # Reasons why rows were skipped
    problems: list[str] = []
    # Create the output folder
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    # Read the CSV and write the JSONL
    with csv_path.open(encoding="utf-8-sig", newline="") as source, jsonl_path.open(
        "w", encoding="utf-8"
    ) as target:
        # One tweet per row
        for row in csv.DictReader(source):
            # Roman words of the tweet
            roman = (row.get("roman_input") or "").split()
            # Devanagari words of the gold
            gold = (row.get("gold_devanagari") or "").split()
            # Roman and gold need the same number of words
            if not roman or len(roman) != len(gold):
                # Skip the row and say why
                problems.append(
                    f"row {row.get('id')}: {len(roman)} roman tokens vs {len(gold)} gold tokens"
                )
                continue
            # The Roman input must be plain ASCII
            if any(not token.isascii() for token in roman):
                problems.append(f"row {row.get('id')}: non-ASCII character in roman_input")
                continue
            # Record in the format the evaluator reads
            record = {
                "id": f"tweet-{row.get('id')}",
                "text": " ".join(roman),
                "tokens": roman,
                "gold": [[word] for word in gold],
                "langs": ["hi"] * len(roman),
            }
            # Write the record
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    # How many were kept, and why others were skipped
    return kept, problems


def main() -> None:
    """Command-line entry: evaluate every system on the CSV and save the results."""
    # Command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ROOT / "tweets" / "roman_hindi_tweet_eval_50.csv"))
    parser.add_argument("--checkpoints", default=str(ROOT / "runs" / "limit100k_ep15_bs256"))
    parser.add_argument("--out", default=str(ROOT / "runs" / "tweet_eval50"))
    parser.add_argument("--no-indicxlit", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Where the trained models are, and where the results go
    checkpoint_dir = Path(args.checkpoints).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert the CSV
    jsonl_path = out_dir / "tweets50.jsonl"
    kept, problems = convert(Path(args.csv), jsonl_path)
    print(f"[tweets50] kept {kept} tweets, skipped {len(problems)}")
    for problem in problems:
        print("  skipped", problem)
    # Stop if no tweet is usable
    if not kept:
        raise SystemExit("no usable tweets")

    # Config used to load the models
    load_config = Config(out_dir=checkpoint_dir, device=args.device)
    # Config for the evaluation: tweets as written, no added noise
    eval_config = replace(
        load_config,
        out_dir=out_dir,
        tweet_file=str(jsonl_path),
        noise_levels=(0.0,),
        tweet_benchmark=True,
    )
    set_seed(eval_config.seed)
    # Pick CPU or GPU
    device = resolve_device(eval_config.device)

    # Vocabularies saved with the models
    source_vocab, target_vocab = load_vocabs(checkpoint_dir / "vocab.json")
    # Load the four trained models
    systems = {}
    for name in load_config.models:
        model, _ = load_model(name, load_config, source_vocab, target_vocab, device)
        systems[name] = model
    # Add IndicXlit unless it is switched off
    if not args.no_indicxlit:
        # Imported here because it needs ctranslate2
        from ..models.indicxlit import IndicXlitBaseline

        baseline = IndicXlitBaseline(eval_config)
        systems[baseline.name] = baseline

    # Load the converted tweets
    bundle = load_tweet_bundle(eval_config)
    if bundle is None:
        raise SystemExit("tweet bundle failed to load")
    # Note where the gold comes from
    bundle.stats["source"] = "human-written gold, roman_hindi_tweet_eval_50.csv"
    print("[tweets50] stats", json.dumps(bundle.stats))

    # Score every system
    rows, predictions = evaluate_tweets(
        eval_config, bundle, systems, source_vocab, target_vocab, device
    )
    # Save the results table and predictions
    save_tweet_results(eval_config, rows, predictions)
    # Save a readable demo of 10 tweets
    save_tweet_demo(eval_config, bundle, systems, source_vocab, target_vocab, device, count=10)
    # Save statistics of the tweet set
    (out_dir / "tweet_stats.json").write_text(
        json.dumps(bundle.stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    def fmt(row: dict, key: str) -> str:
        """Format a value with its 95% interval."""
        return f"{row[key]:.3f} [{row[key + '_lo']:.3f},{row[key + '_hi']:.3f}]"

    # Print the results with their intervals
    for row in rows:
        print(
            f"{row['system']:18s} top1={fmt(row, 'token_accuracy')} "
            f"top5={fmt(row, 'token_top5')} top10={fmt(row, 'token_top10')} "
            f"cer={fmt(row, 'cer')} sent_acc={row['sentence_accuracy']:.3f} "
            f"tokens={row['n_scored_tokens']}"
        )
    # Save the paired comparisons
    write_paired(predictions, out_dir / "tweet_paired.csv")


# Paired comparisons: (system A, system B, label)
COMPARISONS = (
    ("Transformer-clean", "GRU-clean", "Q1 architecture, clean-trained"),
    ("Transformer-noisy", "GRU-noisy", "Q1 architecture, noise-trained"),
    ("GRU-noisy", "GRU-clean", "Q2 noise training, GRU"),
    ("Transformer-noisy", "Transformer-clean", "Q2 noise training, Transformer"),
    ("IndicXlit-raw", "GRU-clean", "Q3 IndicXlit vs GRU-clean"),
    ("IndicXlit-raw", "Transformer-clean", "Q3 IndicXlit vs Transformer-clean"),
)


def write_paired(predictions: list[dict], path: Path) -> None:
    """Paired top-1 differences (A - B) with 95% intervals, resampling tweets."""
    # system -> tweet -> (correct tokens, tokens)
    by_system: dict[str, dict[str, tuple[int, int]]] = {}
    # Collect the per-tweet counts
    for record in predictions:
        # Only the tweets as written (no added noise)
        if record["noise"] != 0.0:
            continue
        by_system.setdefault(record["system"], {})[record["id"]] = (
            record["hits1"],
            record["n_tokens"],
        )
    # CSV header
    lines = ["comparison,system_a,system_b,diff_top1,ci_lo,ci_hi,significant"]
    # Compare each pair of systems
    for a, b, label in COMPARISONS:
        # Skip pairs that were not evaluated
        if a not in by_system or b not in by_system:
            continue
        # Use the same tweets for both systems
        ids = sorted(by_system[a])
        # Difference with its 95% interval
        diff, low, high = paired_diff_ci(
            [by_system[a][i][0] for i in ids],
            [by_system[b][i][0] for i in ids],
            [by_system[a][i][1] for i in ids],
        )
        # Significant if the interval excludes zero
        significant = low > 0 or high < 0
        # Add a CSV row
        lines.append(f"{label},{a},{b},{diff:.4f},{low:.4f},{high:.4f},{significant}")
        # Print it
        print(f"[paired] {label}: {diff:+.3f} [{low:+.3f},{high:+.3f}] significant={significant}")
    # Write the CSV
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
