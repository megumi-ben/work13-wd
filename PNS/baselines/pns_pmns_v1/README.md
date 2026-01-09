# PNS/PMNS v1 baseline

Baseline verifier that routes all regex checks to the `regex_span` PostgreSQL extension. It keeps the PNS/PMNS control-flow hooks (min-heap merge, FINDMAX-style suffix selection) but defers performance work to later versions.

## Python API

- `match_span(conn, txt, pattern, start_pos, end_pos) -> bool`
- `match_from_upto(conn, txt, pattern, start_pos, end_pos) -> bool`
- `find_from(conn, txt, pattern, start_pos) -> (mstart, mend) | None`
- `find_at_upto(conn, txt, pattern, start_pos, end_pos) -> (mstart, mend) | None`
- `find_all(conn, txt, pattern, start_pos=1, overlap=True, allow_empty=False) -> list[(mstart, mend)]`
- `verify_windows(conn, txt, pattern, windows_flat) -> int` (batch verify, returns first hit window_id or -1)
- `PNSPMNSVerifier.matches(txt, pattern, lmin=1, factors=None) -> bool`

All offsets are 1-based character positions. Pass a live `psycopg2` connection that has `CREATE EXTENSION regex_span` executed.

## PositionProvider v1 (anchor-derived)

`AnchorPositionProvider` supplies candidate positions with anchors when possible:

- LP: uses first island anchor hits, expanded backward by `leading_gap_max` when regex starts with a `(?.|\n){0,N}?` gap; falls back to full-range enumeration if parsing fails/anchors为空（日志标记）。
- LS: uses last island anchor hits; falls back to full-range enumeration if parsing失败/空。
- N/M: runs `regex_find_all` (overlap enabled) for each factor and returns the hit starts.

## Verification flow

1. Build candidate windows using LP/LS plus merged N/M hits (heap merge + binary search for suffix).  
2. Batch-verify all windows via `regex_verify_windows` (single regex compile, first hit wins).  
3. Safety: only when anchors parsing fell back, call `regex_find_from(..., allow_empty=True)` as a last resort.

## Notes

- Bit-parallel/bitset acceleration is intentionally absent in v1; the structure is ready for drop-in replacements by swapping the `PositionProvider`.
- Stats (`pns_pmns_stats=...`) are logged: |LP|, |LS|, windows, pruned counts, verify_calls, avg_verify_len, batch_ms, first_hit_window_id, leading_gap_max, fallback flags.
- Zero-length matches在 find_all/find_from/find_at_upto 默认被过滤；需要时传 allow_empty=True。
- Tests (`tests/test_correctness.py`) compare baseline decisions with `txt ~ pattern`, cover leading_gap、overlap 与批量验证路径。
