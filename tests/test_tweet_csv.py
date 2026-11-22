"""Tests for converting the tweet CSV into evaluation records."""
import json

from trainer2.cli.evaluate_tweets import convert


def write_csv(path, rows):
    """Write a small test CSV."""
    lines = ["id,roman_input,gold_devanagari"] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path):
    """Read a JSONL file into a list of dicts."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_aligned_rows_are_kept(tmp_path):
    """A well-formed row becomes a record with tokens, gold and language tags."""
    # Temporary input and output files
    csv_path, out = tmp_path / "in.csv", tmp_path / "out.jsonl"
    write_csv(csv_path, [("1", "kya haal hai", "क्या हाल है")])
    # Run the converter
    kept, problems = convert(csv_path, out)
    # One row kept, no problems
    assert (kept, problems) == (1, [])
    # Read the converted row
    record = read_jsonl(out)[0]
    # Three Roman tokens
    assert record["tokens"] == ["kya", "haal", "hai"]
    # Three gold words
    assert record["gold"] == [["क्या"], ["हाल"], ["है"]]
    # Every token is tagged Hindi
    assert record["langs"] == ["hi", "hi", "hi"]


def test_misaligned_and_non_ascii_rows_are_skipped_with_a_reason(tmp_path):
    """Bad rows are skipped and the reason is reported."""
    csv_path, out = tmp_path / "in.csv", tmp_path / "out.jsonl"
    # Three rows: good, wrong word count, Devanagari in the Roman column
    write_csv(
        csv_path,
        [
            ("1", "kya haal hai", "क्या हाल है"),
            ("2", "kya haal", "क्या हाल है"),
            ("3", "आज ka din", "आज का दिन"),
        ],
    )
    kept, problems = convert(csv_path, out)
    # Only the good row is kept
    assert kept == 1
    # The two bad rows are reported
    assert len(problems) == 2
    # Row 2: the word counts differ
    assert any("row 2" in problem for problem in problems)
    # Row 3: non-ASCII Roman text
    assert any("row 3" in problem for problem in problems)
    # Only one line was written
    assert len(read_jsonl(out)) == 1
