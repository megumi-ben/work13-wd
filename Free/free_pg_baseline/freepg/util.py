import contextlib
import datetime
import os
import tempfile
import time
import zlib
from array import array
from typing import Iterable, Iterator, List, Optional, Sequence

from . import config


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{config.LOG_PREFIX} {ts} {msg}")


def chunked(iterable: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def chunked_iter(iterable: Iterable, size: int) -> Iterator[list]:
    bucket = []
    for item in iterable:
        bucket.append(item)
        if len(bucket) >= size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


@contextlib.contextmanager
def timer(label: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    log(f"{label} took {(end - start):0.3f}s")


def try_import_pyroaring():
    try:
        import pyroaring

        return pyroaring
    except Exception:
        return None


def try_import_ahocorasick():
    try:
        import ahocorasick

        return ahocorasick
    except Exception:
        return None


def delta_encode(sorted_ids: List[int]) -> bytes:
    if not sorted_ids:
        return b""
    deltas = array("Q")
    prev = 0
    for doc_id in sorted_ids:
        delta = doc_id - prev
        if delta < 0:
            raise ValueError("IDs must be non-decreasing for delta encoding")
        deltas.append(delta)
        prev = doc_id
    raw = deltas.tobytes()
    return zlib.compress(raw, level=9)


def delta_decode(blob: bytes) -> List[int]:
    if not blob:
        return []
    raw = zlib.decompress(blob)
    deltas = array("Q")
    deltas.frombytes(raw)
    ids: List[int] = []
    prev = 0
    for delta in deltas:
        prev += delta
        ids.append(prev)
    return ids


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_temp_dir(prefix: str = "freepg") -> str:
    return tempfile.mkdtemp(prefix=prefix)
