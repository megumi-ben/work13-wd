import heapq
import os
import shutil
import struct
import subprocess
import tempfile
from math import ceil
from typing import Dict, Iterable, List, Set, Tuple

from . import config
from .db import (
    clear_free_index,
    count_docs,
    get_conn,
    get_index_meta,
    insert_free_index,
    insert_index_meta,
    load_keys,
    run_schema,
    set_meta,
    stream_docs,
)
from .grams import grams_by_length
from .util import delta_encode, ensure_dir, log, make_temp_dir, try_import_ahocorasick, try_import_pyroaring


def build_postings(
    dsn: str,
    index_id: str,
    backend: str = config.DEFAULT_BACKEND,
    shards: int = config.DEFAULT_POSTINGS_SHARDS,
    tmpdir: str = config.DEFAULT_TMPDIR,
    resume: bool = False,
    logged: bool = config.DEFAULT_LOGGED,
    sort_mem: str = config.DEFAULT_SORT_MEM,
    sort_cmd: str = config.DEFAULT_SORT_CMD,
) -> None:
    conn = get_conn(dsn)
    schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "sql", "schema.sql"))
    run_schema(conn, schema_path)
    meta = get_index_meta(conn, index_id)
    if not meta:
        raise RuntimeError(f"No meta found for index_id={index_id}. Run build-keys first.")
    lmax = meta["lmax"]
    key_map = load_keys(conn, index_id)
    if not key_map:
        raise RuntimeError(f"No keys found for index_id={index_id}. Run build-keys first.")
    keys = [k for k in key_map.keys() if 2 <= len(k) <= lmax]
    if not keys:
        raise RuntimeError("No keys of length >=2 found.")
    docs_table = meta.get("docs_table", "docs")
    total_docs = count_docs(conn, docs_table=docs_table)
    if total_docs == 0:
        raise RuntimeError("No docs to build postings on.")
    # decide backend
    if backend == "auto":
        passes = ceil(len(keys) / config.AHO_CHUNK_SIZE)
        backend = "enum_ngrams" if passes > 2 else "ac_chunk"
    log(f"Using backend={backend} for index_id={index_id} (keys={len(keys)}, lmax={lmax})")
    if not resume:
        clear_free_index(conn, index_id)
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE free_index SET {}LOGGED".format("" if logged else "UN"))
    pyroaring_mod = try_import_pyroaring()
    use_roaring = pyroaring_mod is not None
    postings_codec = "roaring" if use_roaring else "zlib_delta"
    work_tmp = tmpdir or make_temp_dir(prefix="freepg_postings_")
    ensure_dir(work_tmp)
    if backend == "enum_ngrams":
        _build_enum_ngrams(
            conn,
            index_id,
            keys,
            lmax,
            total_docs,
            shards=shards,
            tmpdir=work_tmp,
            resume=resume,
            use_roaring=use_roaring,
            pyroaring_mod=pyroaring_mod,
            sort_mem=sort_mem,
            sort_cmd=sort_cmd,
            docs_table=docs_table,
        )
    elif backend == "ac_chunk" and try_import_ahocorasick():
        _build_with_aho(
            conn,
            index_id,
            keys,
            total_docs,
            use_roaring=use_roaring,
            pyroaring_mod=pyroaring_mod,
            lmax=lmax,
            docs_table=docs_table,
        )
    else:
        log("Using length-bucket scan backend")
        _build_with_length_buckets(conn, index_id, keys, total_docs, use_roaring, pyroaring_mod, docs_table)
    insert_index_meta(
        conn,
        index_id,
        c=meta["c"],
        lmax=lmax,
        use_shell=meta["use_shell"],
        discovery_rows=meta["discovery_rows"],
        notes=meta.get("notes") or "",
        postings_codec=postings_codec,
        postings_params={"backend": backend, "shards": shards},
        docs_table=docs_table,
    )
    set_meta(conn, f"postings_total_docs:{index_id}", str(total_docs))
    log(f"Postings build complete for index_id={index_id} with encoding={postings_codec}")
    conn.close()


def _build_with_length_buckets(conn, index_id: str, keys: List[str], total_docs: int, use_roaring: bool, pyroaring_mod, docs_table: str) -> None:
    postings: Dict[str, object] = {}
    length_sets: Dict[int, Set[str]] = grams_by_length(keys)
    for doc_id, text in stream_docs(conn, docs_table=docs_table):
        found: Set[str] = set()
        n = len(text)
        for length, grams_set in length_sets.items():
            if length == 0 or length > n:
                continue
            max_start = n - length + 1
            for i in range(max_start):
                gram = text[i : i + length]
                if gram in grams_set:
                    found.add(gram)
        for gram in found:
            postings.setdefault(gram, pyroaring_mod.BitMap() if use_roaring else set()).add(doc_id)
    _flush_postings(conn, index_id, postings, total_docs, use_roaring)


def _build_with_aho(conn, index_id: str, keys: List[str], total_docs: int, use_roaring: bool, pyroaring_mod, lmax: int, docs_table: str):
    postings: Dict[str, object] = {}
    chunk_size = config.AHO_CHUNK_SIZE
    for chunk_start in range(0, len(keys), chunk_size):
        chunk = keys[chunk_start : chunk_start + chunk_size]
        automaton = _build_aho(chunk)
        log(f"Scanning docs for chunk {chunk_start}..{chunk_start + len(chunk)}")
        for doc_id, text in stream_docs(conn, docs_table=docs_table):
            found: Set[str] = set()
            for _, gram in automaton.iter(text):
                found.add(gram)
            for gram in found:
                postings.setdefault(gram, pyroaring_mod.BitMap() if use_roaring else set()).add(doc_id)
    _flush_postings(conn, index_id, postings, total_docs, use_roaring)


def _build_aho(keys: Iterable[str]):
    import ahocorasick

    automaton = ahocorasick.Automaton()
    for gram in keys:
        automaton.add_word(gram, gram)
    automaton.make_automaton()
    return automaton


def _flush_postings(conn, index_id: str, postings: Dict[str, object], total_docs: int, use_roaring: bool):
    rows: List[tuple] = []
    for gram, ids in postings.items():
        if use_roaring:
            df = len(ids)
            blob = ids.serialize()
        else:
            df = len(ids)
            blob = delta_encode(sorted(ids))
        sel = df / total_docs
        rows.append((index_id, gram, blob, df, sel))
        if len(rows) >= config.INSERT_BATCH:
            insert_free_index(conn, rows)
            rows = []
    if rows:
        insert_free_index(conn, rows)


def _build_enum_ngrams(
    conn,
    index_id: str,
    keys: List[str],
    lmax: int,
    total_docs: int,
    shards: int,
    tmpdir: str,
    resume: bool,
    use_roaring: bool,
    pyroaring_mod,
    sort_mem: str,
    sort_cmd: str,
    docs_table: str,
):
    log(f"enum_ngrams backend: single pass over docs, shards={shards}, tmpdir={tmpdir}")
    keys_by_len: Dict[int, Set[str]] = grams_by_length(keys)
    shard_paths = [os.path.join(tmpdir, f"shard_{i}.bin") for i in range(shards)]
    shard_spill_done = [os.path.join(tmpdir, f"shard_{i}.spill_done") for i in range(shards)]
    shard_sorted = [os.path.join(tmpdir, f"shard_{i}.sorted") for i in range(shards)]
    shard_db = [os.path.join(tmpdir, f"shard_{i}.db_loaded") for i in range(shards)]
    handles = {}
    for i in range(shards):
        if resume and os.path.exists(shard_db[i]):
            continue
        if resume and os.path.exists(shard_spill_done[i]):
            continue
        if not resume and os.path.exists(shard_paths[i]):
            os.remove(shard_paths[i])
        handles[i] = open(shard_paths[i], "ab")
    scanned = 0
    for doc_id, text in stream_docs(conn, docs_table=docs_table):
        seen: Set[Tuple[int, str]] = set()
        n = len(text)
        for length, keyset in keys_by_len.items():
            if length < 2 or length > lmax or length > n:
                continue
            max_start = n - length + 1
            for i in range(max_start):
                gram = text[i : i + length]
                if gram in keyset:
                    seen.add((length, gram))
        for _, gram in seen:
            shard = hash(gram) % shards
            if shard in handles:
                _write_record(handles[shard], gram, doc_id)
        scanned += 1
        if scanned % 100000 == 0:
            log(f"Scanned {scanned} docs...")
    for h in handles.values():
        h.close()
    for i, h in list(handles.items()):
        if not h.closed:
            h.flush()
            h.close()
        _write_marker_atomic(shard_spill_done[i])
    log("Shard writes complete. Aggregating...")
    for i in range(shards):
        if os.path.exists(shard_db[i]):
            log(f"Shard {i} already db_loaded; skipping.")
            continue
        if not os.path.exists(shard_paths[i]) and not os.path.exists(shard_sorted[i]):
            continue
        _aggregate_shard(
            conn,
            index_id,
            shard_paths[i],
            shard_sorted[i],
            total_docs,
            use_roaring,
            pyroaring_mod,
            sort_mem,
            sort_cmd,
            shard_db[i],
        )


def _aggregate_shard(
    conn,
    index_id: str,
    shard_bin: str,
    shard_sorted: str,
    total_docs: int,
    use_roaring: bool,
    pyroaring_mod,
    sort_mem: str,
    sort_cmd: str,
    marker_db: str,
):
    if os.path.exists(marker_db):
        return
    if shard_bin and os.path.exists(shard_bin) and os.path.getsize(shard_bin) == 0:
        _write_marker_atomic(marker_db)
        return
    if shard_bin and os.path.exists(shard_bin) and (not os.path.exists(shard_sorted)):
        unsorted_txt = shard_bin + ".unsorted"
        _spill_bin_to_text(shard_bin, unsorted_txt)
        sorted_ok = _external_sort(unsorted_txt, shard_sorted, sort_mem=sort_mem, sort_cmd=sort_cmd, tmpdir=os.path.dirname(shard_sorted))
        if not sorted_ok:
            _python_sort(unsorted_txt, shard_sorted)
        os.remove(unsorted_txt)
        _write_marker_atomic(shard_sorted + ".sorted_done")
    elif not os.path.exists(shard_sorted):
        return
    rows: List[tuple] = []
    iterator = _line_iter(shard_sorted)
    current_gram = None
    current_ids = pyroaring_mod.BitMap() if use_roaring else []
    last_doc = None
    for gram_hex, doc_id in iterator:
        gram_bytes = bytes.fromhex(gram_hex)
        gram = gram_bytes.decode("utf-8")
        if current_gram is None:
            current_gram = gram
        if gram != current_gram:
            rows.extend(_finalize_row(index_id, current_gram, current_ids, total_docs, use_roaring))
            current_gram = gram
            current_ids = pyroaring_mod.BitMap() if use_roaring else []
            last_doc = None
        if last_doc == doc_id:
            continue
        if use_roaring:
            current_ids.add(doc_id)
        else:
            current_ids.append(doc_id)
        last_doc = doc_id
        if len(rows) >= config.INSERT_BATCH:
            insert_free_index(conn, rows)
            rows = []
    if current_gram is not None:
        rows.extend(_finalize_row(index_id, current_gram, current_ids, total_docs, use_roaring))
    if rows:
        insert_free_index(conn, rows)
    _write_marker_atomic(marker_db)


def _line_iter(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue
            gram, doc = line.strip("\n").split("\t")
            yield gram, int(doc)


def _merge_chunks(chunks: List[str]):
    files = [open(p, "r", encoding="utf-8") for p in chunks]
    heap = []
    for idx, fh in enumerate(files):
        line = fh.readline()
        if line:
            gram, doc = line.strip("\n").split("\t")
            heapq.heappush(heap, (gram, int(doc), idx))
    while heap:
        gram, doc, idx = heapq.heappop(heap)
        yield gram, doc
        nxt = files[idx].readline()
        if nxt:
            g2, d2 = nxt.strip("\n").split("\t")
            heapq.heappush(heap, (g2, int(d2), idx))
    for fh in files:
        fh.close()


def _finalize_row(index_id: str, gram: str, ids_obj, total_docs: int, use_roaring: bool):
    rows = []
    if use_roaring:
        df = len(ids_obj)
        blob = ids_obj.serialize()
    else:
        ids_sorted = sorted(ids_obj)
        df = len(ids_sorted)
        blob = delta_encode(ids_sorted)
    sel = df / total_docs
    rows.append((index_id, gram, blob, df, sel))
    return rows


def _write_record(handle, gram: str, doc_id: int):
    g_bytes = gram.encode("utf-8")
    handle.write(struct.pack("<I", len(g_bytes)))
    handle.write(g_bytes)
    handle.write(struct.pack("<Q", doc_id))


def _read_bin_records(bin_path: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    with open(bin_path, "rb") as src:
        while True:
            len_bytes = src.read(4)
            if not len_bytes:
                break
            if len(len_bytes) < 4:
                break
            (g_len,) = struct.unpack("<I", len_bytes)
            gram = src.read(g_len)
            doc_bytes = src.read(8)
            if len(doc_bytes) < 8:
                break
            (doc_id,) = struct.unpack("<Q", doc_bytes)
            out.append((gram.decode("utf-8"), doc_id))
    return out


def _spill_bin_to_text(bin_path: str, out_path: str):
    with open(bin_path, "rb") as src, open(out_path, "w", encoding="utf-8") as dst:
        while True:
            len_bytes = src.read(4)
            if not len_bytes:
                break
            if len(len_bytes) < 4:
                break
            (g_len,) = struct.unpack("<I", len_bytes)
            gram = src.read(g_len)
            doc_bytes = src.read(8)
            if len(doc_bytes) < 8:
                break
            (doc_id,) = struct.unpack("<Q", doc_bytes)
            dst.write(f"{gram.hex()}\t{doc_id}\n")


def _external_sort(unsorted: str, sorted_path: str, sort_mem: str, sort_cmd: str, tmpdir: str) -> bool:
    cmd = [sort_cmd or "sort", "-t", "\t", "-k1,1", "-k2,2n", unsorted, "-o", sorted_path]
    if sort_mem:
        cmd.insert(1, f"-S{sort_mem}")
    if tmpdir:
        cmd.insert(1, f"-T{tmpdir}")
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    try:
        res = subprocess.run(cmd, check=False, env=env, capture_output=True)
        if res.returncode != 0:
            log(f"External sort failed: {res.stderr.decode('utf-8', 'ignore')}")
            return False
        return True
    except Exception as e:
        log(f"External sort exception: {e}")
        return False


def _python_sort(unsorted: str, sorted_path: str):
    chunk_dir = tempfile.mkdtemp(prefix="freepg_sort_")
    chunks = []
    chunk_size = 500000
    with open(unsorted, "r", encoding="utf-8") as f:
        buf = []
        for line in f:
            buf.append(line.strip("\n"))
            if len(buf) >= chunk_size:
                buf.sort()
                cpath = os.path.join(chunk_dir, f"chunk_{len(chunks)}.txt")
                with open(cpath, "w", encoding="utf-8") as cf:
                    cf.write("\n".join(buf) + "\n")
                chunks.append(cpath)
                buf = []
        if buf:
            buf.sort()
            cpath = os.path.join(chunk_dir, f"chunk_{len(chunks)}.txt")
            with open(cpath, "w", encoding="utf-8") as cf:
                cf.write("\n".join(buf) + "\n")
            chunks.append(cpath)
    if not chunks:
        open(sorted_path, "w", encoding="utf-8").close()
        return
    if len(chunks) == 1:
        shutil.move(chunks[0], sorted_path)
        shutil.rmtree(chunk_dir, ignore_errors=True)
        return
    with open(sorted_path, "w", encoding="utf-8") as out:
        for line in _merge_sorted_files(chunks):
            out.write(line + "\n")
    shutil.rmtree(chunk_dir, ignore_errors=True)


def _merge_sorted_files(paths: List[str]):
    files = [open(p, "r", encoding="utf-8") for p in paths]
    heap = []
    for idx, fh in enumerate(files):
        line = fh.readline().strip("\n")
        if line:
            gram, doc = line.split("\t")
            heapq.heappush(heap, (gram, int(doc), idx))
    while heap:
        gram, doc, idx = heapq.heappop(heap)
        yield f"{gram}\t{doc}"
        nxt = files[idx].readline().strip("\n")
        if nxt:
            g2, d2 = nxt.split("\t")
            heapq.heappush(heap, (g2, int(d2), idx))
    for fh in files:
        fh.close()


def _write_marker_atomic(path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("done")
    os.replace(tmp, path)
