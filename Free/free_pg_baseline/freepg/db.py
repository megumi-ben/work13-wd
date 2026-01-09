import json
import os
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

from . import config
from .util import log


def get_conn(dsn: str):
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    return conn


def run_schema(conn, schema_path: str) -> None:
    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    migrate_legacy(conn)
    log("Schema ensured and migration checked.")


def stream_docs(
    conn,
    limit: Optional[int] = None,
    offset: int = 0,
    docs_table: str = "docs",
) -> Generator[Tuple[int, str], None, None]:
    name = f"freepg_docs_{os.getpid()}"
    restore_autocommit = conn.autocommit
    if restore_autocommit:
        conn.autocommit = False
    try:
        with conn.cursor(name=name) as cur:
            cur.itersize = config.FETCH_BATCH
            base_query = sql.SQL("SELECT id, text_content FROM {tbl} ORDER BY id OFFSET %s").format(
                tbl=sql.Identifier(docs_table)
            )
            if limit is None:
                cur.execute(base_query, (offset,))
            else:
                query = base_query + sql.SQL(" LIMIT %s")
                cur.execute(query, (offset, limit))
            while True:
                rows = cur.fetchmany(config.FETCH_BATCH)
                if not rows:
                    break
                for row in rows:
                    yield row[0], row[1]
        conn.commit()
    finally:
        if restore_autocommit:
            conn.autocommit = True


def count_docs(conn, docs_table: str = "docs") -> int:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {tbl}").format(tbl=sql.Identifier(docs_table)))
        return int(cur.fetchone()[0])


def set_meta(conn, k: str, v: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO free_meta (k, v)
            VALUES (%s, %s)
            ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v
            """,
            (k, v),
        )


def get_meta(conn, k: str) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT v FROM free_meta WHERE k=%s", (k,))
        res = cur.fetchone()
        return res[0] if res else None


def get_all_meta(conn) -> Dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT k, v FROM free_meta")
        return {k: v for k, v in cur.fetchall()}


def clear_free_index(conn, index_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM free_index WHERE index_id=%s", (index_id,))


def insert_free_index(conn, rows: List[Tuple[str, str, bytes, int, float]]) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO free_index (index_id, gram, postings, df_full, sel_full)
            VALUES %s
            ON CONFLICT (index_id, gram) DO UPDATE
              SET postings = EXCLUDED.postings,
                  df_full = EXCLUDED.df_full,
                  sel_full = EXCLUDED.sel_full
            """,
            rows,
        )


def fetch_postings(conn, index_id: str, grams: List[str]) -> Dict[str, Tuple[bytes, int, float]]:
    if not grams:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT gram, postings, df_full, sel_full FROM free_index WHERE index_id=%s AND gram = ANY(%s)",
            (index_id, grams),
        )
        rows = cur.fetchall()
        return {g: (p, d, s) for g, p, d, s in rows}


def insert_index_meta(
    conn,
    index_id: str,
    c: float,
    lmax: int,
    use_shell: bool,
    discovery_rows: int,
    notes: str = "",
    postings_codec: str = "zlib_delta",
    postings_params: Optional[dict] = None,
    docs_table: str = "docs",
):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO free_index_meta (index_id, c, lmax, use_shell, discovery_rows, notes, postings_codec, postings_params, docs_table)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (index_id) DO UPDATE
              SET c = EXCLUDED.c,
                  lmax = EXCLUDED.lmax,
                  use_shell = EXCLUDED.use_shell,
                  discovery_rows = EXCLUDED.discovery_rows,
                  notes = EXCLUDED.notes,
                  postings_codec = EXCLUDED.postings_codec,
                  postings_params = EXCLUDED.postings_params,
                  docs_table = EXCLUDED.docs_table
            """,
            (
                index_id,
                c,
                lmax,
                use_shell,
                discovery_rows,
                notes,
                postings_codec,
                json.dumps(postings_params) if postings_params else None,
                docs_table,
            ),
        )


def insert_free_keys(conn, index_id: str, key_map: Dict[str, Tuple[int, float]]) -> None:
    rows = [(index_id, gram, df, sel) for gram, (df, sel) in key_map.items()]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO free_keys (index_id, gram, df_discovery, sel_discovery)
            VALUES %s
            ON CONFLICT (index_id, gram) DO UPDATE
              SET df_discovery = EXCLUDED.df_discovery,
                  sel_discovery = EXCLUDED.sel_discovery
            """,
            rows,
        )


def load_keys(conn, index_id: str) -> Dict[str, Tuple[int, float]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT gram, df_discovery, sel_discovery FROM free_keys WHERE index_id=%s",
            (index_id,),
        )
        return {g: (d, s) for g, d, s in cur.fetchall()}


def latest_index_id(conn) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT index_id FROM free_index_meta ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
        return row[0] if row else None


def get_index_meta(conn, index_id: str) -> Optional[Dict[str, object]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT index_id, c, lmax, use_shell, discovery_rows, created_at, notes, postings_codec, postings_params, docs_table FROM free_index_meta WHERE index_id=%s",
            (index_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "index_id": row[0],
            "c": row[1],
            "lmax": row[2],
            "use_shell": row[3],
            "discovery_rows": row[4],
            "created_at": row[5],
            "notes": row[6],
            "postings_codec": row[7],
            "postings_params": row[8],
            "docs_table": row[9],
        }


def migrate_legacy(conn) -> None:
    # Detect legacy free_index without index_id
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name='free_index'
            """
        )
        cols = {r[0] for r in cur.fetchall()}
    if cols and "index_id" not in cols and {"gram", "postings", "df", "sel"}.issubset(cols):
        log("Legacy free_index detected; migrating to new schema with index_id='default'.")
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE free_index ADD COLUMN IF NOT EXISTS index_id TEXT")
            cur.execute("ALTER TABLE free_index RENAME COLUMN df TO df_full")
            cur.execute("ALTER TABLE free_index RENAME COLUMN sel TO sel_full")
            # Drop old primary key if present
            cur.execute(
                """
                SELECT constraint_name FROM information_schema.table_constraints
                WHERE table_name='free_index' AND constraint_type='PRIMARY KEY'
                """
            )
            pk = cur.fetchone()
            if pk:
                cur.execute(f'ALTER TABLE free_index DROP CONSTRAINT IF EXISTS "{pk[0]}"')
            cur.execute("UPDATE free_index SET index_id='default' WHERE index_id IS NULL")
            cur.execute("ALTER TABLE free_index ADD PRIMARY KEY (index_id, gram)")
        log("free_index migration complete.")
    # Ensure new columns exist in free_index_meta
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='free_index_meta'"
        )
        meta_cols = {r[0] for r in cur.fetchall()}
        if meta_cols:
            if "postings_codec" not in meta_cols:
                cur.execute("ALTER TABLE free_index_meta ADD COLUMN IF NOT EXISTS postings_codec TEXT NOT NULL DEFAULT 'zlib_delta'")
            if "postings_params" not in meta_cols:
                cur.execute("ALTER TABLE free_index_meta ADD COLUMN IF NOT EXISTS postings_params JSONB")
            if "docs_table" not in meta_cols:
                cur.execute("ALTER TABLE free_index_meta ADD COLUMN IF NOT EXISTS docs_table TEXT NOT NULL DEFAULT 'docs'")
    # Ensure new tables exist
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS free_index_meta (
                index_id TEXT PRIMARY KEY,
                c DOUBLE PRECISION NOT NULL,
                lmax INTEGER NOT NULL,
                use_shell BOOLEAN NOT NULL,
                discovery_rows BIGINT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                postings_codec TEXT NOT NULL DEFAULT 'zlib_delta',
                postings_params JSONB,
                docs_table TEXT NOT NULL DEFAULT 'docs',
                notes TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS free_keys (
                index_id TEXT NOT NULL,
                gram TEXT NOT NULL,
                df_discovery INTEGER NOT NULL,
                sel_discovery DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (index_id, gram)
            )
            """
        )
    # If legacy keys exist in free_index but not in free_keys, attempt to seed free_keys/index_meta
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM free_keys")
        has_keys = cur.fetchone()[0] > 0
    if not has_keys and cols and "index_id" in cols:
        with conn.cursor() as cur:
            cur.execute("SELECT gram, df_full, sel_full FROM free_index")
            rows = cur.fetchall()
        if rows:
            log("Seeding free_keys from existing free_index into index_id='default'.")
            insert_free_keys(conn, "default", {g: (int(d), float(s)) for g, d, s in rows})
            insert_index_meta(conn, "default", c=0.0, lmax=0, use_shell=True, discovery_rows=0, notes="migrated", docs_table="docs")


def ensure_docs_table(conn, table_name: str):
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {tbl} (
                    id BIGSERIAL PRIMARY KEY,
                    text_content TEXT NOT NULL
                )
                """
            ).format(tbl=sql.Identifier(table_name))
        )
