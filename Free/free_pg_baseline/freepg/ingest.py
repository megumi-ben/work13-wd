import csv
import os
from typing import Optional

from psycopg2.extras import execute_values

from . import config
from .db import ensure_docs_table, get_conn, run_schema
from .util import log


def ingest_csv(dsn: str, csv_path: str, text_col: str, limit: Optional[int] = None, docs_table: str = "docs") -> None:
    conn = get_conn(dsn)
    schema_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "sql", "schema.sql")
    )
    run_schema(conn, schema_path)
    ensure_docs_table(conn, docs_table)
    total = 0
    batch = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if text_col not in reader.fieldnames:
            raise ValueError(f"text column '{text_col}' not found in CSV headers {reader.fieldnames}")
        for row in reader:
            val = row[text_col]
            batch.append((val,))
            total += 1
            if limit and total >= limit:
                break
            if len(batch) >= config.INSERT_BATCH:
                _insert_batch(conn, batch, docs_table)
                batch = []
    if batch:
        _insert_batch(conn, batch, docs_table)
    log(f"Ingest complete. Inserted {total} rows.")
    conn.close()


def _insert_batch(conn, batch, docs_table: str):
    with conn.cursor() as cur:
        execute_values(cur, f"INSERT INTO {docs_table} (text_content) VALUES %s", batch)
