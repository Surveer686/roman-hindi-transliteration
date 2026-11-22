"""Run configuration: the Config dataclass, CLI flags, and device and seed helpers."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PATH_FIELDS = {"data_dir", "out_dir"}
TUPLE_FIELDS = {"models", "noise_levels"}


@dataclass
class Config:
    seed: int = 42
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    out_dir: Path = field(default_factory=lambda: ROOT / "runs" / "four_models")
    train_file: str = "hin_train.json"
    valid_file: str = "hin_valid.json"
    test_file: str = "hin_test.json"
    n_train: int = 100_000
    n_val_loss: int = 5_000
    n_val_acc: int = 2_000
    n_test_max: int = 10_000
    max_len: int = 30
    batch_size: int = 64
    eval_batch_size: int = 64
    epochs: int = 30
    patience: int = 5
    noisy_prob: float = 0.5
    noise_low: float = 0.1
    noise_high: float = 0.6
    beam: int = 10
    max_decode_len: int = 40
    device: str = "auto"
    amp: bool = True
    num_workers: int = 0
    models: tuple[str, ...] = (
        "GRU-clean",
        "GRU-noisy",
        "Transformer-clean",
        "Transformer-noisy",
    )
    noise_levels: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)
    compare_indicxlit: bool = False
    indicxlit_beam: int = 10
    indicxlit_topk: int = 10
    indicxlit_repo: str = "Singla0009/all-indic-transliteration"
    indicxlit_model_dir: str = "indicxlit_ct2_fp32"
    indicxlit_batch_size: int = 256
    indicxlit_inter_threads: int = 1
    indicxlit_intra_threads: int = 2
    tweet_benchmark: bool = False
    tweet_file: str = "tweets.jsonl"
    tweet_dev_file: str = "tweets_dev.jsonl"
    tune_noise: bool = False
    noise_tuning_trials: int = 60
    noise_tuning_level: float = 0.3
    noise_weights: dict[str, float] = field(default_factory=dict)

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_file

    @property
    def valid_path(self) -> Path:
        return self.data_dir / self.valid_file

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_file

    @property
    def tweet_path(self) -> Path:
        return self.data_dir / self.tweet_file

    @property
    def tweet_dev_path(self) -> Path:
        return self.data_dir / self.tweet_dev_file

    @property
    def noise_weights_path(self) -> Path:
        return self.out_dir / "noise_weights.json"

    @property
    def prepared_path(self) -> Path:
        return self.out_dir / "prepared.json"

    @property
    def vocab_path(self) -> Path:
        return self.out_dir / "vocab.json"

    @property
    def stats_path(self) -> Path:
        return self.out_dir / "data_stats.json"

    @property
    def config_path(self) -> Path:
        return self.out_dir / "config.json"

    def make_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, tuple):
                result[key] = list(value)
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Config:
        converted: dict[str, Any] = {}
        for config_field in fields(cls):
            if config_field.name not in payload:
                continue
            value = payload[config_field.name]
            if config_field.name in PATH_FIELDS and value is not None:
                value = Path(value).expanduser()
            if config_field.name in TUPLE_FIELDS and value is not None:
                value = tuple(value)
            converted[config_field.name] = value
        return cls(**converted)

    def write_json(self) -> None:
        self.make_dirs()
        self.config_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local four-model transliteration pipeline")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-dir", dest="data_dir", type=str)
    parser.add_argument("--out", dest="out_dir", type=str)
    parser.add_argument("--train-file", type=str)
    parser.add_argument("--valid-file", type=str)
    parser.add_argument("--test-file", type=str)
    parser.add_argument("--limit-train", dest="n_train", type=int)
    parser.add_argument("--n-val-loss", type=int)
    parser.add_argument("--n-val-acc", type=int)
    parser.add_argument("--n-test-max", type=int)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--noisy-prob", type=float)
    parser.add_argument("--noise-low", type=float)
    parser.add_argument("--noise-high", type=float)
    parser.add_argument("--beam", type=int)
    parser.add_argument("--max-decode-len", type=int)
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "cuda", "gpu", "mps"))
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=None)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--models", nargs="+", type=str)
    parser.add_argument("--noise-levels", nargs="+", type=float)
    parser.add_argument(
        "--compare-indicxlit", dest="compare_indicxlit", action="store_true", default=None
    )
    parser.add_argument("--indicxlit-beam", type=int)
    parser.add_argument("--indicxlit-topk", type=int)
    parser.add_argument("--indicxlit-repo", type=str)
    parser.add_argument("--indicxlit-model-dir", type=str)
    parser.add_argument("--indicxlit-batch-size", type=int)
    parser.add_argument("--indicxlit-inter-threads", type=int)
    parser.add_argument("--indicxlit-intra-threads", type=int)
    parser.add_argument(
        "--tweet-benchmark",
        "--sentence-benchmark",
        dest="tweet_benchmark",
        action="store_true",
        default=None,
    )
    parser.add_argument("--tweet-file", type=str)
    parser.add_argument("--tweet-dev-file", type=str)
    parser.add_argument("--tune-noise", dest="tune_noise", action="store_true", default=None)
    parser.add_argument("--noise-tuning-trials", type=int)
    parser.add_argument("--noise-tuning-level", type=float)
    return parser


def build_config(argv: Sequence[str] | None = None) -> Config:
    args = vars(build_parser().parse_args(argv))
    config = Config()
    for key, value in args.items():
        if value is None:
            continue
        if key in PATH_FIELDS:
            value = Path(value).expanduser().resolve()
        setattr(config, key, value)
    return config


def resolve_device(name: str) -> torch.device:
    requested = (name or "auto").lower()
    if requested in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but it is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def module_main() -> bool:
    return __name__ == "__main__"


def platform_info() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": str(torch.cuda.is_available()),
    }
