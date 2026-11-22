"""Load tweet evaluation data, and build it from annotation CSVs if needed."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config

# Language tag of Hindi tokens
HINDI = "hi"
# The trainer2 package folder
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
# Columns that mark a tweet table
TWEET_HEADERS = {"tweet_id", "split", "text", "tokens_with_lang_tag"}
# Columns that mark a word-annotation table
WORD_HEADERS = {"word", "gold_devanagari"}


@dataclass
class TweetBundle:
    """Tweets for evaluation, plus the dev tweets and statistics."""
    # Tweets used for evaluation
    tweets: list[dict[str, Any]]
    # File they were read from
    path: Path
    # Dev tweets (only used for tuning)
    dev_tweets: list[dict[str, Any]]
    # File of the dev tweets
    dev_path: Path
    # Counts about the set
    stats: dict[str, Any]


def _as_list(value: Any) -> list[Any]:
    """Wrap a value in a list (None becomes an empty list)."""
    # No value: empty list
    if value is None:
        return []
    # Already a list
    if isinstance(value, (list, tuple)):
        return list(value)
    # A single value becomes a one-item list
    return [value]


def _as_gold(value: Any) -> list[list[str]]:
    """Turn a gold value into one list of accepted spellings per token."""
    # No gold at all
    if value is None:
        return []
    # A string: one spelling per word
    if isinstance(value, str):
        return [[item] for item in value.split() if item]
    # A list: each item holds the spellings of one token
    if isinstance(value, (list, tuple)):
        # Spellings per token
        gold: list[list[str]] = []
        for item in value:
            if isinstance(item, str):
                if item:
                    gold.append([item])
            elif isinstance(item, (list, tuple)):
                # Several accepted spellings for one token
                refs = [str(reference) for reference in item if str(reference)]
                gold.append(refs)
        return gold
    return []


def _infer_lang(token: str) -> str:
    """Guess if a token is Hindi: plain letters yes; mentions, links and numbers no."""
    # Mentions, hashtags and links are not Hindi
    if token.startswith(("@", "#", "http://", "https://", "www.")):
        return "other"
    # Plain letters count as Hindi
    return HINDI if token.isalpha() else "other"


def _normalise(row: dict[str, Any], line_no: int) -> dict[str, Any] | None:
    """Convert one raw tweet row into a clean record (None if unusable)."""
    # The tweet text, under any of several column names
    text = row.get("text", row.get("roman", row.get("tweet")))
    # Tokens, if given directly
    raw_tokens = row.get("tokens")
    if isinstance(raw_tokens, str):
        tokens = raw_tokens.split()
    elif isinstance(raw_tokens, (list, tuple)):
        tokens = [str(token) for token in raw_tokens]
    elif isinstance(text, str):
        tokens = text.split()
    else:
        print(f"[tweet-data] skipped line {line_no}: no text or tokens")
        return None

    # Reject an empty token list
    if not tokens:
        print(f"[tweet-data] skipped line {line_no}: empty token list")
        return None
    # Rebuild the text from the tokens
    if not isinstance(text, str) or not text:
        text = " ".join(tokens)

    # Accepted spellings of every token
    gold = _as_gold(row.get("gold", row.get("gold_devanagari", row.get("native"))))
    # Gold must match the tokens one to one
    if len(gold) != len(tokens):
        print(
            f"[tweet-data] skipped line {line_no}: {len(tokens)} tokens, "
            f"{len(gold)} gold entries"
        )
        return None

    # Language tag of every token
    raw_langs = row.get("langs", row.get("lang"))
    if isinstance(raw_langs, str):
        langs = [raw_langs.lower()] * len(tokens)
    elif isinstance(raw_langs, (list, tuple)) and len(raw_langs) == len(tokens):
        langs = [str(lang).lower() for lang in raw_langs]
    else:
        langs = [_infer_lang(token) for token in tokens]

    # Clean spellings (only used for noise tuning)
    raw_clean = row.get("clean_tokens", row.get("clean", row.get("canonical")))
    if isinstance(raw_clean, str):
        clean_candidates = raw_clean.split()
    else:
        clean_candidates = _as_list(raw_clean)
    # One clean spelling per token, or None
    clean_tokens = [
        str(clean_candidates[index]).lower()
        if index < len(clean_candidates) and clean_candidates[index] is not None
        else None
        for index in range(len(tokens))
    ]

    # Scored tokens: Hindi ones that have a gold spelling
    scored_indices = [
        index
        for index, lang in enumerate(langs)
        if lang == HINDI and gold[index]
    ]
    # The finished record
    return {
        "id": str(row.get("id", row.get("tweet_id", line_no))),
        "text": text,
        "tokens": tokens,
        "gold": gold,
        "langs": langs,
        "clean_tokens": clean_tokens,
        "scored_indices": scored_indices,
    }


def _load_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read a tweet JSONL file; returns the records and the number skipped."""
    # Usable tweets
    records: list[dict[str, Any]] = []
    # Number of unusable lines
    skipped = 0
    # Read the file line by line
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            # Skip lines that are not valid JSON
            except json.JSONDecodeError as error:
                print(f"[tweet-data] skipped line {line_no}: {error}")
                skipped += 1
                continue
            # Skip lines that are not objects
            if not isinstance(row, dict):
                skipped += 1
                continue
            # Clean the row
            record = _normalise(row, line_no)
            # Skip unusable rows
            if record is None:
                skipped += 1
                continue
            records.append(record)
    return records, skipped


def _candidate_dirs(config: Config) -> list[Path]:
    """Folders that are searched for tweet files."""
    # Where to look: data, tweets, package and output folders
    directories = (
        config.data_dir,
        PACKAGE_ROOT / "tweets",
        PACKAGE_ROOT,
        config.out_dir,
    )
    output: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        # Keep each existing folder once
        if directory.exists() and directory not in seen:
            seen.add(directory)
            output.append(directory)
    return output


def find_tweet_file(config: Config, filename: str) -> Path | None:
    """Find a tweet file by name in the search folders."""
    # An absolute path is used as it is
    candidate_path = Path(filename)
    if candidate_path.is_absolute():
        return candidate_path if candidate_path.exists() else None
    # Otherwise search the folders in order
    for directory in _candidate_dirs(config):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def load_tweet_bundle(config: Config) -> TweetBundle | None:
    """Load the tweet set (built from the CSVs first if needed)."""
    # Look for the tweet file
    tweet_path = find_tweet_file(config, config.tweet_file)
    if tweet_path is None:
        try:
            # Not found: build it from the CSVs
            import_annotation_csvs(config)
        except (FileNotFoundError, ValueError, OSError) as error:
            print(f"[tweet-data] tweet import skipped: {error}", flush=True)
            return None
        tweet_path = find_tweet_file(config, config.tweet_file)
        if tweet_path is None:
            return None
    # Read the tweets
    tweets, skipped = _load_records(tweet_path)
    if not tweets:
        print(f"[tweet-data] no usable tweets in {tweet_path}")
        return None

    # Optional dev tweets
    dev_path = find_tweet_file(config, config.tweet_dev_file)
    dev_tweets: list[dict[str, Any]] = []
    dev_skipped = 0
    if dev_path is not None:
        dev_tweets, dev_skipped = _load_records(dev_path)

    # Counts for the report
    stats = {
        "tweet_path": str(tweet_path),
        "dev_path": str(dev_path) if dev_tweets else None,
        "n_tweets": len(tweets),
        "n_scored_tokens": sum(len(tweet["scored_indices"]) for tweet in tweets),
        "n_skipped_tweets": skipped,
        "n_dev_tweets": len(dev_tweets),
        "n_dev_pairs": len(dev_noise_pairs_from(dev_tweets)),
        "n_skipped_dev_tweets": dev_skipped,
    }
    # Everything together
    return TweetBundle(
        tweets=tweets,
        path=tweet_path,
        dev_tweets=dev_tweets,
        dev_path=dev_path if dev_tweets else config.tweet_dev_path,
        stats=stats,
    )


def dev_noise_pairs_from(tweets: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(clean, as-written) spelling pairs from tweets, for noise tuning."""
    # (clean, as-written) pairs
    pairs: list[tuple[str, str]] = []
    for tweet in tweets:
        for index in tweet["scored_indices"]:
            # The clean spelling and the one actually written
            clean = tweet["clean_tokens"][index]
            observed = tweet["tokens"][index]
            # Skip tokens without a clean spelling
            if not clean or not clean.isalpha():
                continue
            # Skip tokens written the clean way
            if clean == observed.lower():
                continue
            # Keep the pair
            pairs.append((clean, observed.lower()))
    return pairs


def dev_noise_pairs(bundle: TweetBundle) -> list[tuple[str, str]]:
    """Noise-tuning pairs from the dev tweets of a bundle."""
    return dev_noise_pairs_from(bundle.dev_tweets)


def _read_table(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read a CSV file into (column names, rows)."""
    # utf-8-sig drops the Excel byte-order mark
    with path.open(encoding="utf-8-sig", newline="") as handle:
        # Read the rows as dicts
        reader = csv.DictReader(handle)
        # Column names and rows
        return list(reader.fieldnames or []), list(reader)


def _candidate_tables(config: Config) -> list[Path]:
    """Table-like files found in the search folders."""
    # Files to inspect
    tables: list[Path] = []
    seen: set[Path] = set()
    # Check every search folder
    for directory in _candidate_dirs(config):
        # Each file in the folder
        for path in sorted(directory.iterdir()):
            if (
                # Only table-like files, each one once
                path.is_file()
                and path.suffix.lower() in (".csv", ".md", ".tsv", ".txt")
                and path not in seen
            ):
                seen.add(path)
                tables.append(path)
    return tables


def _detect_tables(
    config: Config,
) -> tuple[tuple[Path, list[str], list[dict[str, Any]]] | None, ...]:
    """Find the tweet table and the word table among those files."""
    # Best tables found so far
    tweet_source = None
    word_source = None
    for path in _candidate_tables(config):
        try:
            # Read the file (unreadable files are skipped)
            fields, rows = _read_table(path)
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        # Column names in lower case
        header = {field.strip().lower() for field in fields}
        # First table with tweet columns
        if tweet_source is None and TWEET_HEADERS <= header:
            tweet_source = (path, fields, rows)
        # First table with word columns
        if word_source is None and WORD_HEADERS <= header:
            word_source = (path, fields, rows)
        # Stop once both are found
        if tweet_source is not None and word_source is not None:
            break
    return tweet_source, word_source


def _find_tweet_csv(
    config: Config,
) -> tuple[Path, list[str], list[dict[str, Any]]] | None:
    """Find the CSV that holds the tweets."""
    # All tables that look like tweet tables
    candidates: list[tuple[Path, list[str], list[dict[str, Any]]]] = []
    for path in _candidate_tables(config):
        try:
            # Read the file (unreadable files are skipped)
            fields, rows = _read_table(path)
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        # Column names in lower case
        header = {field.strip().lower() for field in fields}
        # Has the tweet columns
        if TWEET_HEADERS <= header:
            candidates.append((path, fields, rows))
    # Prefer a file with 'tweet' in its name
    for candidate in candidates:
        if "tweet" in candidate[0].name.lower():
            return candidate
    # Otherwise take the first one
    return candidates[0] if candidates else None


def _find_word_table(
    config: Config,
) -> tuple[Path, list[str], list[dict[str, Any]]] | None:
    """Find the CSV with word annotations (word + gold)."""
    for path in _candidate_tables(config):
        # Only CSV or TSV files
        if path.suffix.lower() not in (".csv", ".tsv"):
            continue
        try:
            # Read the file (unreadable files are skipped)
            fields, rows = _read_table(path)
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        # Column names in lower case
        header = {field.strip().lower() for field in fields}
        # Has the word and gold columns
        if WORD_HEADERS <= header:
            return path, fields, rows
    return None


def _local_word_lookup(config: Config) -> dict[str, tuple[list[str], bool]]:
    """Automatic gold: spellings taken from the Aksharantar word pairs."""
    # Imported here: only needed for this fallback
    from .word_data import build_refs, clean_pairs, iter_jsonl

    # word -> (spellings, is Hindi)
    lookup: dict[str, tuple[list[str], bool]] = {}
    # Use all three Aksharantar files
    for path in (config.train_path, config.valid_path, config.test_path):
        # Skip missing files
        if not path.exists():
            continue
        # Spellings of every Roman word
        for roman, references in build_refs(
            clean_pairs(iter_jsonl(path), config.max_len)
        ).items():
            # Spellings already known
            existing = list(lookup.get(roman, ([], False))[0])
            # Add only new spellings
            for reference in references:
                if reference not in existing:
                    existing.append(reference)
            # Dictionary words count as Hindi
            lookup[roman] = (existing, True)
    return lookup


def _field(fields: list[str], predicate: Any) -> str | None:
    """Find a column name with a test on its lower-case name."""
    # First column whose name passes the test
    for field in fields:
        if predicate(field.strip().lower()):
            return field
    return None


def _word_lookup(
    fields: list[str], rows: list[dict[str, Any]]
) -> dict[str, tuple[list[str], bool]]:
    """Word -> (accepted spellings, is Hindi) from an annotation sheet."""
    # Find the needed columns
    word_key = _field(fields, lambda name: name == "word")
    gold_key = _field(fields, lambda name: name == "gold_devanagari")
    alternate_key = _field(fields, lambda name: "alternate" in name)
    hindi_key = _field(fields, lambda name: name.startswith("is_hindi"))
    # The word and gold columns are required
    if word_key is None or gold_key is None:
        raise ValueError("word annotation sheet is missing word or gold_devanagari")

    # word -> (spellings, is Hindi)
    lookup: dict[str, tuple[list[str], bool]] = {}
    # One row per word
    for row in rows:
        # The word, in lower case
        word = (row.get(word_key) or "").strip().lower()
        # Skip empty rows
        if not word:
            continue
        # The main Devanagari spelling
        gold = (row.get(gold_key) or "").strip()
        # Accepted spellings start with it
        references = [gold] if gold else []
        # Add alternative spellings (separated by |)
        if alternate_key:
            for reference in re.split(r"[|]", row.get(alternate_key) or ""):
                reference = reference.strip()
                if reference and reference not in references:
                    references.append(reference)
        # Marked Y when the word is Hindi
        is_hindi = False
        if hindi_key:
            is_hindi = (row.get(hindi_key) or "").strip().upper() == "Y"
        # Store the word
        lookup[word] = (references, is_hindi)
    return lookup


def _tagged_tokens(value: str) -> tuple[list[str], list[str]]:
    """Split 'word/TAG word/TAG' text into tokens and tags."""
    # Words and their tags
    tokens: list[str] = []
    tags: list[str] = []
    # Each item looks like word/TAG
    for item in value.split():
        # Split at the last slash
        if "/" in item:
            token, tag = item.rsplit("/", 1)
        else:
            token, tag = item, ""
        # Store the word and its lower-case tag
        tokens.append(token.strip())
        tags.append(tag.strip().lower())
    return tokens, tags


def _token_language(token: str, tag: str, word_info: tuple[list[str], bool]) -> str:
    """Decide whether a token counts as Hindi."""
    references, is_hindi = word_info
    # Numbers and symbols are not Hindi
    if not token.isalpha():
        return "other"
    # Hindi if annotated, known or tagged HI
    if references or is_hindi or tag == HINDI:
        return HINDI
    return "other"


def import_annotation_csvs(config: Config) -> dict[str, Any]:
    """Build the tweet JSONL files from the annotation CSVs."""
    # Find the tweet table
    tweet_source = _find_tweet_csv(config)
    # Nothing to import
    if tweet_source is None:
        raise FileNotFoundError("no tweet CSV with tweet_id/split/text/tokens_with_lang_tag")

    # Unpack the table
    tweet_path, tweet_fields, tweet_rows = tweet_source
    # Find the word-annotation sheet, if any
    word_source = _find_word_table(config)
    # No sheet: use the dictionary pairs as gold
    if word_source is None:
        lookup = _local_word_lookup(config)
        gold_source = "local_word_pairs"
    # A sheet exists: read its annotations
    else:
        word_path, word_fields, word_rows = word_source
        lookup = _word_lookup(word_fields, word_rows)
        # Sheet has no gold yet: fall back to the dictionary
        if any(references for references, _ in lookup.values()):
            gold_source = str(word_path)
        else:
            lookup = _local_word_lookup(config)
            gold_source = "local_word_pairs"

    # Find the tweet columns
    id_key = _field(tweet_fields, lambda name: name == "tweet_id")
    split_key = _field(tweet_fields, lambda name: name == "split")
    text_key = _field(tweet_fields, lambda name: name == "text")
    tag_key = _field(tweet_fields, lambda name: name == "tokens_with_lang_tag")
    # All four columns are required
    if not all((id_key, split_key, text_key, tag_key)):
        raise ValueError("tweet CSV is missing required columns")

    # Tweets grouped by split (test or dev)
    grouped: dict[str, list[dict[str, Any]]] = {"test": [], "dev": []}
    # Counters for the report
    total_tokens = 0
    hi_tokens = 0
    gold_tokens = 0
    # Convert each tweet
    for row in tweet_rows:
        # Tokens and their language tags
        tokens, tags = _tagged_tokens(row.get(tag_key) or "")
        # No tagged tokens: split the plain text
        if not tokens:
            tokens = (row.get(text_key) or "").split()
            tags = [""] * len(tokens)
        # Pad missing tags
        if len(tags) < len(tokens):
            tags.extend([""] * (len(tokens) - len(tags)))

        # Gold, language and clean spelling per token
        gold: list[list[str]] = []
        langs: list[str] = []
        clean: list[str | None] = []
        # Look up each token
        for token, tag in zip(tokens, tags):
            # Spellings known for this token
            word_info = lookup.get(token.lower(), ([], False))
            gold.append(word_info[0])
            langs.append(_token_language(token, tag, word_info))
            clean.append(token.lower() if token.isalpha() else None)
        # Update the counters
        total_tokens += len(tokens)
        hi_tokens += sum(lang == HINDI for lang in langs)
        gold_tokens += sum(
            1 for refs, lang in zip(gold, langs) if refs and lang == HINDI
        )
        # Store the record under its split
        grouped.setdefault((row.get(split_key) or "test").strip().lower(), []).append(
            {
                "id": (row.get(id_key) or "").strip(),
                "text": (row.get(text_key) or " ").strip(),
                "tokens": tokens,
                "gold": gold,
                "langs": langs,
                "clean_tokens": clean,
            }
        )

    # Write one JSONL file per split
    config.make_dirs()
    outputs: dict[str, str] = {}
    for split, filename in (("test", config.tweet_file), ("dev", config.tweet_dev_file)):
        path = config.out_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for record in grouped.get(split, []):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        outputs[split] = str(path)

    # Summary of the import
    report = {
        "tweet_source": str(tweet_path),
        "test_path": outputs["test"],
        "dev_path": outputs["dev"],
        "n_test": len(grouped.get("test", [])),
        "n_dev": len(grouped.get("dev", [])),
        "n_tokens": total_tokens,
        "n_hindi_tokens": hi_tokens,
        "n_gold_tokens": gold_tokens,
        "gold_source": gold_source,
        "n_lookup_words": len(lookup),
        "gold_coverage": gold_tokens / max(hi_tokens, 1),
    }
    # Print the summary
    print("[tweet-data] imported " + json.dumps(report, ensure_ascii=False), flush=True)
    # Hand back the summary
    return report
