import json
import os
import tempfile
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from . import config
from .db import (
    count_docs,
    get_conn,
    insert_free_keys,
    insert_index_meta,
    load_keys,
    run_schema,
    stream_docs,
)
from .grams import generate_grams_for_doc
from .presuf_shell import presuf_shell
from .util import log


class ShardedSpillCounter:
    def __init__(self, spill_threshold: int, num_shards: int):
        self.spill_threshold = spill_threshold
        self.num_shards = num_shards
        self.counts: Counter = Counter()
        self.tempdir = tempfile.mkdtemp(prefix="freepg_counts_")
        self.files: Dict[int, str] = {}

    def _file_for(self, shard: int) -> str:
        if shard not in self.files:
            path = os.path.join(self.tempdir, f"shard_{shard}.txt")
            self.files[shard] = path
        return self.files[shard]

    def add(self, gram: str) -> None:
        self.counts[gram] += 1
        if len(self.counts) >= self.spill_threshold:
            self.spill()

    def spill(self) -> None:
        if not self.counts:
            return
        shard_buffers: Dict[int, List[str]] = {}
        for gram, cnt in self.counts.items():
            shard = hash(gram) % self.num_shards
            shard_buffers.setdefault(shard, []).append(json.dumps([gram, cnt]))
        for shard, lines in shard_buffers.items():
            path = self._file_for(shard)
            with open(path, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
        self.counts.clear()

    def finalize(self) -> Counter:
        self.spill()
        final_counts: Counter = Counter()
        final_counts.update(self.counts)
        for shard, path in self.files.items():
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    gram, cnt = json.loads(line.rstrip("\n"))
                    final_counts[gram] += int(cnt)
        return final_counts


def discover_keys(
    dsn: str,
    c: float = config.DEFAULT_C,
    lmax: int = config.DEFAULT_LMAX,
    sample_rows: Optional[int] = config.DEFAULT_SAMPLE_ROWS,
    use_shell: bool = config.DEFAULT_USE_SHELL,
    spill_threshold: int = config.DEFAULT_SPILL_THRESHOLD,
    num_shards: int = config.DEFAULT_SHARDS,
    docs_table: str = "docs",
) -> Tuple[str, Dict[str, Tuple[int, float]], Dict[str, str]]:
    """
    Returns (index_id, mapping gram -> (df, sel), meta dict).
    """
    conn = get_conn(dsn)
    schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "sql", "schema.sql"))
    run_schema(conn, schema_path)
    total_docs = count_docs(conn, docs_table=docs_table)
    if total_docs == 0:
        raise RuntimeError("No docs found. Ingest data first.")
    if sample_rows is None or sample_rows <= 0:
        sample_rows = total_docs
    N = min(sample_rows, total_docs)
    log(f"Key discovery scanning {N} docs (of {total_docs}) with lmax={lmax}, c={c}")
    spiller = ShardedSpillCounter(spill_threshold=spill_threshold, num_shards=num_shards)
    scanned = 0
    for doc_id, text in stream_docs(conn, limit=N, docs_table=docs_table):
        seen: Set[str] = set()
        for gram in generate_grams_for_doc(text, lmax):
            seen.add(gram)
        for gram in seen:
            spiller.add(gram)
        scanned += 1
        if scanned % 100000 == 0:
            log(f"Scanned {scanned} docs...")
    counts = spiller.finalize()
    log(f"Counted {len(counts)} unique grams")
    keys = compute_minimal_useful(counts, N, c, lmax)
    log(f"Found {len(keys)} minimal useful grams")
    if use_shell:
        shell_keys = presuf_shell(list(keys.keys()))
        shell = {g: keys[g] for g in shell_keys if g in keys}
        log(f"Shell reduced to {len(shell)} grams")
    else:
        shell = keys
    index_id = config.make_index_id(c, lmax, use_shell, N, docs_table=docs_table)
    meta = {
        "index_id": index_id,
        "discovery_N": str(N),
        "discovery_c": str(c),
        "discovery_lmax": str(lmax),
        "discovery_sample_rows": str(sample_rows),
        "use_shell": str(use_shell),
        "key_count": str(len(shell)),
        "docs_table": docs_table,
    }
    insert_index_meta(
        conn,
        index_id,
        c=c,
        lmax=lmax,
        use_shell=use_shell,
        discovery_rows=N,
        notes=config.DEFAULT_INDEX_NOTES,
        postings_codec="zlib_delta",
        postings_params={"lmax": lmax},
        docs_table=docs_table,
    )
    insert_free_keys(conn, index_id, shell)
    conn.close()
    return index_id, shell, meta


def compute_minimal_useful(
    counts: Counter, N: int, c: float, lmax: int
) -> Dict[str, Tuple[int, float]]:
    sel_map: Dict[str, float] = {}
    for gram, df in counts.items():
        sel_map[gram] = df / N
    useful: Set[str] = {g for g, sel in sel_map.items() if sel <= c}
    minimal: Dict[str, Tuple[int, float]] = {}
    for gram in useful:
        if len(gram) > lmax:
            continue
        prefix = gram[:-1]
        if len(prefix) == 0:
            prefix_useful = False
        else:
            prefix_useful = prefix in useful
        if not prefix_useful:
            minimal[gram] = (counts[gram], sel_map[gram])
    return minimal


def load_keys_from_meta(conn) -> List[str]:
    latest = latest_index(conn=conn)
    if not latest:
        raise RuntimeError("No key index found. Run build-keys first.")
    return list(load_keys(conn, latest).keys())


def latest_index(conn):
    from .db import latest_index_id

    return latest_index_id(conn)
