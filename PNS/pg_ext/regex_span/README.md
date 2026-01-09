# regex_span PostgreSQL extension

Helpers for running PostgreSQL's native regex engine on character ranges without substring slicing. All positions are 1-based character offsets (not bytes) and boundary-sensitive constructs (e.g., `\y`) are preserved because the match is evaluated against the original string.

## Functions

- `regex_match_span(txt text, pattern text, start_pos int, end_pos int) -> boolean`  
  Returns true when a match exists whose start/end (inclusive) equal the supplied offsets. Out-of-range or inverted offsets return false without error.
- `regex_match_from_upto(txt text, pattern text, start_pos int, end_pos int) -> boolean`  
  Requires the match to start at `start_pos` and end no later than `end_pos`.
- `regex_find_from(txt text, pattern text, start_pos int) -> int[]`  
  First match at or after `start_pos`; returns `{mstart, mend}` (`mend` inclusive) or NULL if none. `allow_empty` (default false) skips zero-length matches.
- `regex_find_at_upto(txt text, pattern text, start_pos int, end_pos int) -> int[]`  
  First match that starts at `start_pos` and ends no later than `end_pos`; NULL if none. `allow_empty` controls zero-length inclusion (default false).
- `regex_find_all(txt text, pattern text, start_pos int default 1, overlap boolean default true, allow_empty boolean default false) -> setof int[]`  
  All matches from `start_pos` forward, ordered by start. `overlap=true` advances by `mstart+1` to keep overlaps; `overlap=false` advances by `mend+1`. Zero-length matches are filtered unless `allow_empty=true`; cursor always advances at least one character.
- `regex_verify_windows(txt text, pattern text, windows int4[], allow_empty boolean default false) -> int4`  
  Batch-verify windows `[s1,e1,s2,e2,...]`; returns the first hit window_id (0-based) where the match starts at `si` and ends no later than `ei`, or -1 if none.

## Build & install (PGXS)

```sh
cd pg_ext/regex_span
make USE_PGXS=1
make USE_PGXS=1 install
```

Then in psql:

```sql
CREATE EXTENSION IF NOT EXISTS regex_span;
```

## Quick test

```
psql -f sql/test_regex_span.sql
```

## Notes / assumptions

- Targeted at PostgreSQL 13.8; uses `RE_compile_and_cache` and `pg_regexec` with `REG_ADVANCED` flags and the current collation.
- Character offsets are computed via wide-character conversion (`pg_mb2wchar_with_len`) to support multibyte text.
- Overlap handling is explicit; zero-length matches are filtered by default and the search cursor always advances by at least one character to avoid infinite loops.
- Regex compilation errors raise the standard PostgreSQL error; out-of-range positions return false/NULL instead of errors.
