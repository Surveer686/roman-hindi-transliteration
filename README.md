# Roman → Devanagari (Hindi) Transliteration

Transliterating Romanized Hindi (`kharab`, `yaar aaj ka din...`) into Devanagari (`खराब`, `यार आज का दिन...`).
The project compares three systems on clean dictionary words, on words with synthetic spelling noise, and on
tweet-style sentences:

| System | Description |
|---|---|
| **GRU** | Character-level seq2seq GRU with beam search |
| **Transformer** | Character-level encoder–decoder Transformer with beam search |
| **IndicXlit** | Pretrained IndicXlit model (CTranslate2 export), used as an off-the-shelf baseline |

The GRU and Transformer are each trained twice: **clean** (original spellings only) and **noisy** (half of the
training inputs get rule-based spelling noise; the Devanagari target is unchanged).

## Project layout

```
trainer2/
├── config.py            # Config dataclass + CLI flags
├── run.py               # python -m trainer2.run
├── cli/                 # pipeline.py (train/eval), evaluate_tweets.py
├── datasets/            # vocab, word/tweet loading, synthetic noise (noise.py)
├── models/              # gru.py, transformer.py, indicxlit.py, decode.py (beam search)
├── training/            # trainer.py
├── evaluation/          # metrics, word_eval, tweet_eval, error_analysis, demo
├── tweets/              # 50-tweet evaluation set (CSV)
└── runs/                # saved results, plots and configs of the reported runs
tests/                   # pytest suite
```

## Setup

Requires Python ≥ 3.10.

```bash
pip install -e .          # torch, numpy, pandas, matplotlib, ctranslate2, huggingface_hub
pip install -e ".[dev]"   # adds pytest and ruff
```

## Data

Training data is **not** in the repo. Place the word-pair files in a data directory (default `trainer2/data/`):
`hin_train.json`, `hin_valid.json`, `hin_test.json`. Point to another location with `--data-dir`.
The IndicXlit model is downloaded from Hugging Face (`Singla0009/all-indic-transliteration`) on first use.

## Usage

```bash
# Quick sanity check
python -m trainer2.run --selftest

# Full pipeline: prepare data -> train -> evaluate (reported run configuration)
python -m trainer2.run --data-dir path/to/data --limit-train 100000 --epochs 15 \
    --batch-size 256 --out trainer2/runs/limit100k_ep15_bs256

# Individual stages
python -m trainer2.run --prepare-only
python -m trainer2.run --train-only
python -m trainer2.run --eval-only

# Evaluate trained models on the tweet set
python -m trainer2.cli.evaluate_tweets --checkpoints trainer2/runs/limit100k_ep15_bs256
```

After `pip install -e .` the console scripts `translit-run` and `translit-eval-tweets` are also available.
See `python -m trainer2.run --help` for all options (noise levels, beam size, device, AMP, models to train, ...).
Model checkpoints (`*.pt`) are not committed.

## Results

Reported run: 100k training pairs, 15 epochs, batch size 256, seed 42. Word test set: ~10k held-out words
(no Roman-side overlap with training). Numbers are top-1 accuracy / character error rate (CER).

### Dictionary words, by synthetic noise level (top-1)

| Model | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| GRU-clean | 46.7% | 36.6% | 28.5% | 22.4% | 12.4% |
| GRU-noisy | 45.1% | 38.9% | 33.3% | 28.4% | 20.4% |
| Transformer-clean | 43.0% | 34.2% | 26.4% | 21.1% | 12.1% |
| Transformer-noisy | 41.6% | 35.7% | 30.0% | 25.3% | 18.5% |
| IndicXlit (raw) | **50.3%** | 39.3% | 29.7% | 22.4% | 12.5% |

Noise-aware training costs a little on clean input but helps substantially as noise increases.

### Tweet-style sentences (50 tweets, 493 scored tokens)

| System | Token top-1 | Token top-5 | CER |
|---|---|---|---|
| GRU-clean | 61.9% | 87.4% | 0.199 |
| GRU-noisy | 55.4% | 84.0% | 0.211 |
| Transformer-clean | **62.3%** | 88.6% | 0.194 |
| Transformer-noisy | 57.2% | 84.2% | 0.229 |
| IndicXlit (raw) | 58.6% | 85.0% | 0.227 |

Full tables (top-5/top-10, CER, confidence intervals), training curves, error breakdowns and qualitative
examples are in `trainer2/runs/`.

### Caveats

- The same noise rules generate the noisy training data and the noisy test words, so gains on synthetic noise
  do not by themselves prove gains on real-world spellings.
- The tweet set is small (50 tweets); differences between systems there are within the confidence intervals.

## Tests

```bash
pytest
```

## License

See [LICENSE](LICENSE).
