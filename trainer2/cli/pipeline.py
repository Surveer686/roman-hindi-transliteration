"""Full pipeline: prepare data, train, evaluate on words and tweets (python -m trainer2.run)."""
from __future__ import annotations

import json
from typing import Any, Sequence

import torch

from ..config import Config, build_parser, platform_info, resolve_device, set_seed
from ..datasets.noise import tune_noise_weights
from ..datasets.tweet_data import dev_noise_pairs, import_annotation_csvs, load_tweet_bundle
from ..datasets.vocab import EOS, PAD, SOS, Vocab, build_vocabs, load_vocabs, save_vocabs
from ..datasets.word_data import load_bundle
from ..evaluation.metrics import self_test as metrics_self_test
from ..evaluation.tweet_eval import evaluate_tweets, save_tweet_demo, save_tweet_results
from ..evaluation.word_eval import evaluate_models, save_evaluation
from ..models.decode import predict
from ..models.gru import GRUSeq2Seq
from ..models.indicxlit import IndicXlitBaseline
from ..models.transformer import CharTransformer
from ..training.trainer import load_model, train_all

# Switches that choose what the run does
RUN_OPTIONS = (
    "prepare_only",
    "train_only",
    "eval_only",
    "overwrite",
    "selftest",
    "import_tweets",
)


def build_run_parser() -> Any:
    """Command-line parser: the config flags plus the run-mode switches."""
    # Start from the config flags
    parser = build_parser()
    # Add the run-mode switches
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--import-tweets", dest="import_tweets", action="store_true")
    return parser


def parse_options(argv: Sequence[str] | None = None) -> tuple[Config, dict[str, bool]]:
    """Parse the command line into a Config and the run-mode switches."""
    # Parse the command line
    payload = vars(build_run_parser().parse_args(argv))
    # The run-mode switches
    options = {name: bool(payload[name]) for name in RUN_OPTIONS}
    # Build the config from the other flags
    config = Config.from_dict(
        {key: value for key, value in payload.items() if value is not None}
    )
    return config, options


def self_test() -> None:
    """Fast check that both model types run a forward pass and decode."""
    # Check the metric functions first
    metrics_self_test()
    # Tiny vocabularies for the test
    source_vocab = Vocab("abc")
    target_vocab = Vocab("कख")
    # A tiny batch of two words
    source = torch.tensor([[SOS, 4, 5, PAD], [SOS, 4, PAD, PAD]])
    target = torch.tensor([[SOS, 4, EOS, PAD], [SOS, 5, EOS, PAD]])
    # One small GRU and one small Transformer
    models = (
        GRUSeq2Seq(len(source_vocab), len(target_vocab), 32, 32, 1, 0.0),
        CharTransformer(
            len(source_vocab),
            len(target_vocab),
            dimension=32,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            feedforward=64,
            dropout=0.0,
        ),
    )
    # Check each model
    for model in models:
        model.eval()
        # The forward pass must give the right shape
        logits = model(source, target[:, :-1])
        assert logits.shape[:2] == (2, 3), logits.shape
        # Beam search must return candidates
        predictions = predict(
            model,
            ["abc", "ab"],
            source_vocab,
            target_vocab,
            torch.device("cpu"),
            beam=3,
            max_len=5,
        )
        assert len(predictions) == 2 and all(predictions), predictions
    # Everything passed
    print("self-test ok")


def get_vocabs(config: Config) -> tuple[Vocab, Vocab]:
    """Load the vocabularies, or build and save them from the training data."""
    # Reuse the saved vocabularies
    if config.vocab_path.exists():
        return load_vocabs(config.vocab_path)
    # Otherwise build them from the training pairs
    bundle = load_bundle(config)
    source_vocab, target_vocab = build_vocabs(bundle.train_pairs)
    # Save them for next time
    save_vocabs(config.vocab_path, source_vocab, target_vocab)
    return source_vocab, target_vocab


def run(argv: Sequence[str] | None = None) -> None:
    """Run the whole pipeline according to the command-line flags."""
    # Read the command line
    config, options = parse_options(argv)
    # --selftest: quick built-in test
    if options["selftest"]:
        self_test()
        return
    # --import-tweets: build the tweet files, then stop
    if options["import_tweets"]:
        report = import_annotation_csvs(config)
        print("[import-tweets]", json.dumps(report, ensure_ascii=False), flush=True)
        return

    # Prepare the output folder and save the settings
    config.make_dirs()
    config.write_json()
    # Make the run repeatable
    set_seed(config.seed)
    # Pick CPU or GPU
    device = resolve_device(config.device)
    print("[run]", json.dumps(platform_info()), "device", device, flush=True)

    # Load the word data (cached)
    bundle = load_bundle(config)
    # Load or build the vocabularies
    source_vocab, target_vocab = get_vocabs(config)
    print(
        "[run] prepared",
        len(bundle.train_pairs),
        "train pairs,",
        len(bundle.val_words),
        "val words,",
        len(bundle.test_words),
        "test words",
        flush=True,
    )
    # --prepare-only: stop after preparing the data
    if options["prepare_only"]:
        return

    # Load the tweets, if there are any
    tweet_bundle = load_tweet_bundle(config)
    if tweet_bundle is None:
        print("[run] no tweet data found; tweet benchmark and tuning skipped", flush=True)
    # Optionally tune the noise weights on dev tweets
    elif config.tune_noise:
        # (clean, as-written) pairs from the dev tweets
        pairs = dev_noise_pairs(tweet_bundle)
        if not pairs:
            print(
                "[run] no annotated dev tweet pairs; keeping default noise weights",
                flush=True,
            )
        else:
            # Search for better noise weights
            weights, report = tune_noise_weights(
                pairs,
                level=config.noise_tuning_level,
                trials=config.noise_tuning_trials,
                seed=config.seed,
            )
            # Use them from now on
            config.noise_weights = weights
            # Save the weights and the search report
            config.noise_weights_path.write_text(
                json.dumps(
                    {
                        "weights": weights,
                        "report": report,
                        "stats": tweet_bundle.stats,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            config.write_json()
            print(f"[run] tuned noise weights {json.dumps(report)}", flush=True)

    # Train the four models (skipped with --eval-only)
    if not options["eval_only"]:
        train_all(
            config, bundle, source_vocab, target_vocab, device, options["overwrite"]
        )
    # --train-only: stop after training
    if options["train_only"]:
        return

    # Load the trained models
    systems: dict[str, Any] = {}
    for name in config.models:
        model, _ = load_model(name, config, source_vocab, target_vocab, device)
        systems[name] = model

    # Optional IndicXlit baseline
    baselines: dict[str, Any] = {}
    if config.compare_indicxlit:
        print("[run] loading IndicXlit baseline", flush=True)
        baseline = IndicXlitBaseline(config)
        baselines[baseline.name] = baseline

    # Evaluate all systems on the word test set
    result = evaluate_models(
        config,
        bundle,
        systems,
        source_vocab,
        target_vocab,
        device,
        baselines=baselines,
        noise_weights=config.noise_weights or None,
    )
    # Save the tables, plots and predictions
    save_evaluation(result, config)

    # Models plus baseline, for the tweet evaluation
    all_systems = dict(systems)
    all_systems.update(baselines)
    # Evaluate on the tweets if requested
    if config.tweet_benchmark and tweet_bundle is not None:
        # Score every system on the tweets
        tweet_rows, tweet_predictions = evaluate_tweets(
            config,
            tweet_bundle,
            all_systems,
            source_vocab,
            target_vocab,
            device,
        )
        # Save the tweet results
        save_tweet_results(config, tweet_rows, tweet_predictions)
        # Save the readable tweet demo
        save_tweet_demo(
            config, tweet_bundle, all_systems, source_vocab, target_vocab, device
        )
        # Save statistics about the tweet set
        (config.out_dir / "tweet_stats.json").write_text(
            json.dumps(tweet_bundle.stats, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    # Requested, but there is no tweet data
    elif config.tweet_benchmark:
        print("[run] tweet benchmark skipped: no tweet data found", flush=True)


def main() -> None:
    """Command-line entry point."""
    run()


if __name__ == "__main__":
    main()
