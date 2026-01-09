import io
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

from psycopg2.extras import execute_values

from . import config
from .db import count_docs, fetch_postings, get_conn
from .plan import PlanNode
from .util import delta_decode, log, try_import_pyroaring


class ByteLRU:
    def __init__(self, max_bytes: int, max_entry_bytes: int):
        self.max_bytes = max_bytes
        self.max_entry_bytes = max_entry_bytes
        self.data: OrderedDict = OrderedDict()
        self.current_bytes = 0

    def get(self, key):
        if key not in self.data:
            return None
        val, sz = self.data.pop(key)
        self.data[key] = (val, sz)
        return val

    def put(self, key, value, size_bytes: int):
        if size_bytes > self.max_entry_bytes or size_bytes > self.max_bytes:
            return
        if key in self.data:
            _, prev = self.data.pop(key)
            self.current_bytes -= prev
        while self.current_bytes + size_bytes > self.max_bytes and self.data:
            _, (_, ev_sz) = self.data.popitem(last=False)
            self.current_bytes -= ev_sz
        if self.current_bytes + size_bytes > self.max_bytes:
            return
        self.data[key] = (value, size_bytes)
        self.current_bytes += size_bytes


_POSTINGS_CACHE = ByteLRU(config.POSTINGS_CACHE_MAX_BYTES, config.POSTINGS_CACHE_MAX_ENTRY_BYTES)


def execute_query(
    dsn: str,
    index_id: str,
    plan_node: PlanNode,
    regex: str,
    docs_table: str = "docs",
    verify_limit: int = 10000,
) -> Dict[str, object]:
    conn = get_conn(dsn)
    timings: Dict[str, float] = {}
    total_docs = count_docs(conn, docs_table=docs_table)
    if plan_node.node_type == "TRUE":
        log("Plan is TRUE; skipping postings filter.")
        t0 = time.perf_counter()
        final_ids = verify_regex(conn, regex, None, docs_table=docs_table)[0]
        timings["verify_ms"] = (time.perf_counter() - t0) * 1000
        conn.close()
        return {
            "candidates": None,
            "final_ids": final_ids,
            "timings": timings,
        }
    leaves = sorted(set(collect_leaves(plan_node)))
    t0 = time.perf_counter()
    encoding = get_encoding(conn, index_id)
    postings_sets, fetch_ms = fetch_and_decode(conn, index_id, leaves, encoding)
    timings["postings_fetch_ms"] = fetch_ms
    t1 = time.perf_counter()
    candidate_ids = eval_plan(plan_node, postings_sets)
    timings["set_ops_ms"] = (time.perf_counter() - t1) * 1000
    if candidate_ids is None:
        candidate_ids = None
    ratio = None
    if candidate_ids is not None:
        ratio = len(candidate_ids) / total_docs if total_docs else 1.0
        if ratio > config.CAND_RATIO_GATE:
            log(f"Candidate ratio {ratio:0.3f} exceeds gate {config.CAND_RATIO_GATE}; running full scan.")
            candidate_ids = None
    t2 = time.perf_counter()
    final_ids, materialize_ms, verify_ms = verify_regex(conn, regex, candidate_ids, docs_table=docs_table)
    timings["materialize_ms"] = materialize_ms
    timings["verify_ms"] = verify_ms
    conn.close()
    return {
        "candidates": None if candidate_ids is None else sorted(candidate_ids),
        "final_ids": final_ids,
        "timings": timings,
        "candidate_count": None if candidate_ids is None else len(candidate_ids),
        "final_count": len(final_ids),
        "candidate_ratio": ratio,
    }


def collect_leaves(node: PlanNode) -> List[str]:
    if node.node_type == "LEAF":
        return [node.value or ""]
    leaves: List[str] = []
    for c in node.children:
        leaves.extend(collect_leaves(c))
    return leaves


def get_encoding(conn, index_id: str) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT postings_codec FROM free_index_meta WHERE index_id=%s", (index_id,))
        res = cur.fetchone()
        return res[0] if res else "zlib_delta"


def fetch_and_decode(conn, index_id: str, grams: List[str], encoding: str):
    t0 = time.perf_counter()
    decoded: Dict[str, object] = {}
    missing: List[str] = []
    for g in grams:
        cached = _POSTINGS_CACHE.get((index_id, g))
        if cached is not None:
            decoded[g] = copy_postings(cached)
        else:
            missing.append(g)
    if missing:
        raw = _fetch_postings_bulk(conn, index_id, missing)
        for gram, (blob, df, sel) in raw.items():
            obj = decode_posting(blob, encoding)
            size_est = _estimate_size(obj, blob, encoding)
            _POSTINGS_CACHE.put((index_id, gram), obj, size_est)
            decoded[gram] = copy_postings(obj)
    return decoded, (time.perf_counter() - t0) * 1000


def decode_posting(blob: bytes, encoding: str):
    if encoding == "roaring":
        pyroaring = try_import_pyroaring()
        if not pyroaring:
            raise RuntimeError("Postings encoded as roaring but pyroaring is not installed.")
        return pyroaring.BitMap.deserialize(blob)
    return set(delta_decode(blob))


def _estimate_size(obj, blob: bytes, encoding: str) -> int:
    if encoding == "roaring":
        try:
            return obj.getSizeInBytes()
        except Exception:
            return len(blob) if blob is not None else 0
    try:
        return len(obj) * 16  # conservative for Python int
    except Exception:
        return len(blob) if blob is not None else 0


def copy_postings(obj):
    return obj.copy()


def eval_plan(node: PlanNode, postings_sets: Dict[str, object]):
    if node.node_type == "LEAF":
        return postings_sets.get(node.value or "", None)
    if node.node_type == "TRUE":
        return None
    child_sets = [eval_plan(c, postings_sets) for c in node.children]
    if node.node_type == "AND":
        filtered = [s for s in child_sets if s is not None]
        if not filtered:
            return None
        filtered.sort(key=lambda s: len(s))
        base = copy_postings(filtered[0])
        for s in filtered[1:]:
            base &= s
        return base
    elif node.node_type == "OR":
        any_none = False
        result = None
        for s in child_sets:
            if s is None:
                any_none = True
            else:
                if result is None:
                    result = copy_postings(s)
                else:
                    result |= s
        if any_none:
            return None
        return result if result is not None else set()
    return None


def _fetch_postings_bulk(conn, index_id: str, grams: List[str]) -> Dict[str, Tuple[bytes, int, float]]:
    if not grams:
        return {}
    if len(grams) <= config.LEAF_TEMP_THRESHOLD:
        return fetch_postings(conn, index_id, grams)
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE leaf_grams(gram TEXT) ON COMMIT DROP")
        buf = io.StringIO("\n".join(grams) + "\n")
        cur.copy_expert("COPY leaf_grams(gram) FROM STDIN WITH (FORMAT text)", buf)
        cur.execute(
            """
            SELECT f.gram, f.postings, f.df_full, f.sel_full
            FROM free_index f
            JOIN leaf_grams g ON f.gram = g.gram
            WHERE f.index_id=%s
            """,
            (index_id,),
        )
        rows = cur.fetchall()
    return {g: (p, d, s) for g, p, d, s in rows}


def verify_regex(conn, regex: str, candidate_ids: Optional[Set[int]], docs_table: str = "docs") -> Tuple[List[int], float, float]:
    materialize_ms = 0.0
    verify_ms = 0.0
    res: List[int] = []
    t_start = time.perf_counter()
    with conn.cursor() as cur:
        if candidate_ids is None:
            cur.execute(
                f"SELECT id FROM {docs_table} WHERE text_content ~ %s",
                (regex,),
            )
            res = [row[0] for row in cur.fetchall()]
            verify_ms = (time.perf_counter() - t_start) * 1000
        else:
            ids_list = list(candidate_ids)
            if not ids_list:
                return [], materialize_ms, verify_ms
            if len(ids_list) <= config.ARRAY_ANY_MAX:
                cur.execute(
                    f"SELECT id FROM {docs_table} WHERE id = ANY(%s) AND text_content ~ %s",
                    (ids_list, regex),
                )
                res = [row[0] for row in cur.fetchall()]
                verify_ms = (time.perf_counter() - t_start) * 1000
            else:
                restore_autocommit = conn.autocommit
                if restore_autocommit:
                    conn.autocommit = False
                try:
                    cur.execute("CREATE TEMP TABLE IF NOT EXISTS cand(id BIGINT) ON COMMIT DROP")
                    copy_buf = io.StringIO("\n".join(str(i) for i in ids_list) + "\n")
                    cur.copy_expert("COPY cand(id) FROM STDIN WITH (FORMAT text)", copy_buf)
                    cur.execute("ANALYZE cand")
                    materialize_ms = (time.perf_counter() - t_start) * 1000
                    t_verify = time.perf_counter()
                    cur.execute(
                        f"SELECT d.id FROM {docs_table} d JOIN cand c ON d.id = c.id WHERE d.text_content ~ %s",
                        (regex,),
                    )
                    res = [row[0] for row in cur.fetchall()]
                    verify_ms = (time.perf_counter() - t_verify) * 1000
                    conn.commit()
                finally:
                    if restore_autocommit:
                        conn.autocommit = True
    return res, materialize_ms, verify_ms
