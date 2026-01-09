# FREE PostgreSQL Baseline

Runnable baseline reproduction of the ICDE 2002 paper “A Fast Regular Expression Indexing Engine (FREE)” on top of PostgreSQL. The goal is to keep the code simple and working end-to-end: build keys, build postings, run regex queries with zero false negatives.

## Layout
- `freepg/` Python package with CLI and core modules.
- `sql/schema.sql` schema and pg_trgm extension.
- `scripts/demo_small.sh` tiny end-to-end demo.

## Quick start
```bash
pip install -r requirements.txt
# Set DSN (e.g., postgresql://user:password@localhost/dbname)
export DSN=postgresql://localhost/postgres

# Small demo
bash scripts/demo_small.sh
```

## CLI
All commands run via `python -m freepg.cli ...`.

- Ingest CSV: `freepg ingest --dsn "$DSN" --csv data.csv --text-col text [--limit N]`
- Discover keys: `freepg build-keys --dsn "$DSN" [--c 0.1] [--lmax 10] [--sample 3000000] [--use-shell/--no-shell]` (prints deterministic `index_id`)
- Build postings: `freepg build-postings --dsn "$DSN" [--index-id <id>] [--backend auto|enum_ngrams|ac_chunk|bucket] [--shards N] [--tmpdir DIR] [--resume] [--sort-mem 2G] [--sort-cmd sort]` (defaults to latest index, backend auto)
- Query: `freepg query --dsn "$DSN" --regex "<pattern>" [--index-id <id>] [--limit N] [--show-plan]`

## What it does
1. **Key discovery** (`build-keys`): streams docs (optionally sampled), counts grams up to `lmax`, keeps minimal useful grams where selectivity `<= c` and prefix is useless, optionally applies the FREE presuf shell, and writes rows into `free_index_meta` and `free_keys` with a deterministic `index_id`.
2. **Postings build** (`build-postings`): reads keys from `free_keys`, builds postings with selectable backends:
   - `enum_ngrams` (default for large keysets): single-pass substring enumeration with sharded binary spill (length-prefixed records safe for tabs/newlines) + deterministic external sort + aggregation (resume-friendly, atomic markers).
   - `ac_chunk`: chunked Aho–Corasick if available.
   - `bucket`: length-bucket sliding windows fallback.
   Postings encoded as Roaring (if `pyroaring`) or zlib-delta and stored in `free_index` keyed by `index_id`.
3. **Query** (`query`): safe regex literal extractor (generator template fast-path; falls back to full scan if uncertain), builds logical plan (AND of literals, simple OR groups), substitutes literals with available keys or selective/pruned substrings, executes postings set ops with batch fetch + LRU cache, applies candidate-ratio gate to avoid useless materialization, and verifies candidates with PostgreSQL regex (`~`) to return exact matches.

## Notes
- PostgreSQL `pg_trgm` extension is enabled in `schema.sql` (not required for the baseline).
- `pyroaring` and `pyahocorasick` are optional; the code falls back to pure Python structures if missing. If postings were built with Roaring, queries require `pyroaring` installed to decode.
- Index variants live in tables (`free_index_meta`, `free_keys`, `free_index`); `free_meta` holds only small scalars (e.g., postings encoding per index). A simple migration path in code upgrades legacy schemas automatically (legacy data migrates to `index_id='default'` when possible).
- Resume markers: per shard `.spill_done`, `.sorted_done`, `.db_loaded` written atomically; `--resume` skips shards already loaded. Old text spill files can be discarded; rebuild is acceptable.
