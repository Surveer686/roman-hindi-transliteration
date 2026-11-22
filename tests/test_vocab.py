"""Tests for the character vocabularies and batch collation (datasets/vocab.py)."""
from trainer2.datasets.vocab import (
    EOS,
    PAD,
    SOS,
    UNK,
    Vocab,
    build_vocabs,
    collate,
    load_vocabs,
    save_vocabs,
)


def test_encode_decode_round_trip():
    """Encoding a word to ids and decoding it again returns the same word."""
    vocab = Vocab("abc")
    assert vocab.decode(vocab.encode("cab")) == "cab"


def test_unknown_characters_map_to_unk():
    """A character outside the vocabulary becomes the <unk> id instead of failing."""
    vocab = Vocab("abc")
    assert vocab.encode("az") == [vocab.stoi["a"], UNK]


def test_decode_stops_at_eos_and_skips_specials():
    """Decoding ignores <sos> and <pad> and stops at <eos>, as a decoder's output needs."""
    vocab = Vocab("abc")
    ids = [SOS, vocab.stoi["a"], PAD, vocab.stoi["b"], EOS, vocab.stoi["c"]]
    assert vocab.decode(ids) == "ab"


def test_build_vocabs_keeps_only_repeated_target_characters():
    """Target characters seen fewer than twice are dropped; all Roman letters are kept."""
    pairs = [("ka", "का"), ("kaa", "का"), ("ki", "की")]
    source, target = build_vocabs(pairs)
    assert "का"[0] in target.stoi  # appears in 3 words
    assert "ई" not in target.stoi
    assert all(char in source.stoi for char in "kai")


def test_save_and_load(tmp_path):
    """vocab.json round-trips: reloaded vocabularies have identical ids."""
    source, target = Vocab("ab"), Vocab("कख")
    path = tmp_path / "vocab.json"
    save_vocabs(path, source, target)
    loaded_source, loaded_target = load_vocabs(path)
    assert loaded_source.itos == source.itos and loaded_target.itos == target.itos


def test_collate_pads_to_longest():
    """A batch is padded to its longest word, and targets get <sos> ... <eos>."""
    source, target = Vocab("ab"), Vocab("क")
    src, tgt = collate([("ab", "क"), ("a", "कक")], source, target)
    assert src.shape == (2, 2)
    assert tgt.shape == (2, 4)  # <sos> + 2 chars + <eos>
    assert tgt[0, 0].item() == SOS
