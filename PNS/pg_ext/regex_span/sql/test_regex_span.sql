-- Minimal manual test script for regex_span
\echo 'Loading extension...'
CREATE EXTENSION IF NOT EXISTS regex_span;

\echo 'Word boundary span check'
SELECT regex_match_span('abc def', E'\\ydef\\y', 5, 7) AS boundary_ok;

\echo 'match_from_upto and find_at_upto upper-bound semantics'
SELECT regex_match_from_upto('abcde', 'bc', 2, 4) AS match_from_upto_ok;
SELECT regex_match_from_upto('abcde', 'bcde', 2, 4) AS match_from_upto_fail; -- ends after bound
SELECT regex_find_at_upto('abcde', 'bcd', 2, 4) AS find_at_upto_ok;
SELECT regex_find_at_upto('abcde', 'bcd', 3, 4) AS find_at_upto_null;

\echo 'Non-greedy gap with newline'
SELECT regex_find_from('a' || E'\n' || E'\n' || 'b', E'a(?:.|\\n){0,3}?b', 1) AS gap_match;

\echo 'OR grouping'
SELECT * FROM regex_find_all('foo and bar', '(?:foo|bar)', 1, true);

\echo 'Character class with count'
SELECT * FROM regex_find_all('lego mego 99ego _xego', '[A-Za-z0-9_]{2}ego', 1, true);

\echo 'Multibyte safety'
SELECT regex_match_span('汉字abc', '字a', 2, 3) AS multibyte_ok;

\echo 'Overlap enabled'
SELECT * FROM regex_find_all('ababa', 'aba', 1, true);
\echo 'Overlap disabled'
SELECT * FROM regex_find_all('ababa', 'aba', 1, false);

\echo 'Zero-length handling (should not loop, filtered by default)'
SELECT * FROM regex_find_all('aaaa', 'a*?', 1, true, false) LIMIT 5;
\echo 'Zero-length allowed explicitly'
SELECT * FROM regex_find_all('aaaa', 'a*?', 1, true, true) LIMIT 5;

\echo 'Bounds handling'
SELECT regex_match_span('abc', 'a', 0, 1) AS start_zero;
SELECT regex_find_from('abc', 'a', 0) AS find_start_zero;
SELECT regex_match_span('abc', 'abc', 1, 5) AS end_out_of_range;

\echo 'Batch verify windows (first hit expected at index 1)'
SELECT regex_verify_windows('foo bar baz', '\ybar\y', ARRAY[1,3,5,7,5,8]) AS first_hit_window;
