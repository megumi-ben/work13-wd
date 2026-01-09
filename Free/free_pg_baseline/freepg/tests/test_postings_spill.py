import os

from freepg.postings_build import (
    _line_iter,
    _python_sort,
    _read_bin_records,
    _spill_bin_to_text,
    _write_record,
)
from freepg.exec_postings import ByteLRU


def test_spill_roundtrip_special_chars(tmp_path):
    bin_path = tmp_path / "shard.bin"
    # write grams containing tab and newline
    with open(bin_path, "ab") as f:
        _write_record(f, "a\tb", 1)
        _write_record(f, "a\tb", 2)
        _write_record(f, "line\nbreak", 3)
        _write_record(f, "punct!?", 3)
        _write_record(f, "punct!?", 3)  # duplicate doc id to be deduped
    unsorted = bin_path.with_suffix(".unsorted")
    sorted_path = bin_path.with_suffix(".sorted")
    _spill_bin_to_text(str(bin_path), str(unsorted))
    _python_sort(str(unsorted), str(sorted_path))
    agg = {}
    for gram_hex, doc in _line_iter(str(sorted_path)):
        gram = bytes.fromhex(gram_hex).decode("utf-8")
        agg.setdefault(gram, set()).add(doc)
    assert agg["a\tb"] == {1, 2}
    assert agg["line\nbreak"] == {3}
    assert agg["punct!?"] == {3}


def test_byte_lru_skips_large_entries():
    cache = ByteLRU(max_bytes=16, max_entry_bytes=8)
    cache.put("big", [1] * 5, size_bytes=16)  # exceeds entry limit
    assert cache.get("big") is None
    cache.put("small", [1], size_bytes=4)
    assert cache.get("small") == [1]


def test_marker_atomic(tmp_path):
    from freepg.postings_build import _write_marker_atomic

    marker = tmp_path / "mark.db_loaded"
    _write_marker_atomic(str(marker))
    assert marker.exists()
    assert marker.read_text() == "done"
