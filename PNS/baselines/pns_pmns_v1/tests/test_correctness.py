import os
import random
import string
import logging

import psycopg2
import pytest

from baselines.pns_pmns_v1.pns_pmns import AnchorPositionProvider, PNSPMNSVerifier
from baselines.pns_pmns_v1.verify_pg import find_all, verify_windows
from baselines.pns_pmns_v1.factor_extractor import extract_factors


@pytest.fixture(scope="session")
def conn():
    dsn = os.getenv("PG_DSN", "dbname=postgres")
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS regex_span;")
    yield conn
    conn.close()


def pg_regex_matches(conn, txt: str, pattern: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT %s ~ %s", (txt, pattern))
        return cur.fetchone()[0]


def random_word(rng: random.Random, min_len: int = 3, max_len: int = 6) -> str:
    letters = string.ascii_lowercase + string.digits + "_"
    length = rng.randint(min_len, max_len)
    return "".join(rng.choice(letters) for _ in range(length))


def random_text(rng: random.Random) -> str:
    tokens = ["foo", "bar", "ego", "abc", "def", "a", "b", "汉字"]
    pieces = []
    for _ in range(rng.randint(4, 9)):
        choice = rng.choice(["token", "newline", "random"])
        if choice == "newline":
            pieces.append("\n")
        elif choice == "random":
            pieces.append(random_word(rng))
        else:
            pieces.append(rng.choice(tokens))
    return " ".join(pieces)


def random_pattern(rng: random.Random) -> str:
    kind = rng.choice(["boundary", "gap", "or", "class"])
    if kind == "boundary":
        word = rng.choice(["foo", "bar", "ego", random_word(rng, 2, 4)])
        return rf"\y{word}\y"
    if kind == "gap":
        k = rng.randint(0, 3)
        return rf"a(?:.|\n){{0,{k}}}?b"
    if kind == "or":
        opt1 = rng.choice(["foo", "bar", random_word(rng, 2, 3)])
        opt2 = rng.choice(["baz", "ego", random_word(rng, 2, 3)])
        return rf"(?:{opt1}|{opt2})"
    k = rng.randint(2, 3)
    return rf"[A-Za-z0-9_]{{{k}}}ego"


def test_multibyte_span(conn):
    verifier = PNSPMNSVerifier(conn)
    assert verifier.matches("汉字abc", "字a", lmin=2)


def test_random_agreement(conn):
    verifier = PNSPMNSVerifier(conn)
    rng = random.Random(42)
    for _ in range(30):
        txt = random_text(rng)
        pattern = random_pattern(rng)
        expected = pg_regex_matches(conn, txt, pattern)
        got = verifier.matches(txt, pattern, lmin=1)
        assert got == expected, f"text={txt!r}, pattern={pattern!r}"


def test_overlap_handling(conn):
    spans = find_all(conn, "ababa", "aba", start_pos=1, overlap=True)
    assert spans == [(1, 3), (3, 5)]
    spans_non = find_all(conn, "ababa", "aba", start_pos=1, overlap=False)
    assert spans_non == [(1, 3)]


def test_anchor_prunes(conn):
    provider = AnchorPositionProvider(conn)
    txt = "foo gap foo bar baz foo"
    pattern = r"\yfoo\y(?:.|\n){0,2}?\ybar\y"
    lmin = 1
    full_range = max(len(txt) - lmin + 1, 0)
    lp_hits, _ = provider.get_LPs(txt, pattern, lmin, factors=None)
    ls_hits, _ = provider.get_LSs(txt, pattern, lmin, factors=None)
    assert len(lp_hits) < full_range
    assert len(ls_hits) < full_range


def test_factor_extractor_no_or_necessary():
    pat = r"\yLeft\y(?:.|\n){0,5}?\yaccident\y"
    f = extract_factors(pat)
    assert f is not None
    assert "Left" in f.islands[0].necessary_literals
    assert "accident" in f.islands[1].necessary_literals
    assert all(len(lit) >= 2 for lit in f.islands[0].necessary_literals + f.islands[1].necessary_literals)


def test_factor_extractor_or_common_substring():
    pat = r"\y(?:fooBAR|zzBAR)\y"
    f = extract_factors(pat)
    assert f is not None
    all_nec = [lit for isl in f.islands for lit in isl.necessary_literals]
    assert any("BAR" in lit for lit in all_nec)


def test_factor_extractor_or_non_literal_disable():
    pat = r"\y(?:ab[A-Z]{2}|abCD)\y"
    f = extract_factors(pat)
    assert f is not None
    all_nec = [lit for isl in f.islands for lit in isl.necessary_literals]
    assert all_nec == []


def test_pmns_lite_prunes_windows_but_keeps_correctness(conn, caplog):
    txt = "Left xxx closed --- Left yyy closed --- Left accident closed"
    pattern = r"\yLeft\y(?:.|\n){0,30}?\yaccident\y(?:.|\n){0,30}?\yclosed\y"
    verifier_pns = PNSPMNSVerifier(conn, mode="pns")
    verifier_pmns = PNSPMNSVerifier(conn, mode="pmns-lite")
    expected = pg_regex_matches(conn, txt, pattern)
    ok1 = verifier_pns.matches(txt, pattern, lmin=1)
    caplog.set_level(logging.INFO)
    ok2 = verifier_pmns.matches(txt, pattern, lmin=1)
    assert ok1 == expected
    assert ok2 == expected
    # inspect last log line
    record = None
    for r in caplog.records:
        if "pns_pmns_stats=" in r.message:
            record = r
    assert record is not None
    payload_str = record.message.split("pns_pmns_stats=")[-1]
    import ast
    payload = ast.literal_eval(payload_str)
    assert payload.get("M_enabled") is True or payload.get("M_enabled") == 1
    assert payload.get("windows_after_M") <= payload.get("windows_before_M")


def test_lp_ls_factor_literal_path_used(conn):
    txt = "xxx Left ... accident ... closed ..."
    pattern = r"\yLeft\y(?:.|\n){0,10}?\yaccident\y"
    verifier = PNSPMNSVerifier(conn, mode="pns")
    ok = verifier.matches(txt, pattern, lmin=1)
    expected = pg_regex_matches(conn, txt, pattern)
    assert ok == expected
    stats = verifier.last_stats
    assert stats is not None
    assert stats.get("LP_source") == "factor_lit"
    assert stats.get("LS_source") == "factor_lit"


def test_lp_leading_gap_expansion_avoids_fn(conn):
    txt = "ALeft accident"
    pattern = r"(?:.|\n){0,1}?\yLeft\y(?:.|\n){0,5}?\yaccident\y"
    verifier = PNSPMNSVerifier(conn, mode="pns")
    ok = verifier.matches(txt, pattern, lmin=1)
    expected = pg_regex_matches(conn, txt, pattern)
    assert ok == expected
    stats = verifier.last_stats
    assert stats is not None
    assert stats.get("LP_source") == "factor_lit"
    assert stats.get("LP_expansion_ratio", 1) >= 1


def test_anchor_missing_fallback(conn):
    txt = "abcde"
    pattern = r"\y[A-Z]{2}\y"
    verifier = PNSPMNSVerifier(conn, mode="pns")
    ok = verifier.matches(txt, pattern, lmin=1)
    expected = pg_regex_matches(conn, txt, pattern)
    assert ok == expected
    stats = verifier.last_stats
    assert stats is not None
    assert stats.get("LP_source") != "factor_lit" or stats.get("LS_source") != "factor_lit"


def test_factor_extractor_max_len_basic():
    pat = r"\yAB\y(?:.|\n){0,3}?\yCD\y"
    f = extract_factors(pat)
    assert f is not None
    assert f.min_len == 4
    assert f.max_len == 7


def test_factor_extractor_max_len_with_class_quant():
    pat = r"\y[A-Za-z0-9_]{1,3}AB\y(?:.|\n){0,2}?\yCD[0-9]{2,4}\y"
    f = extract_factors(pat)
    assert f is not None
    assert f.max_len == 13
    assert f.min_len == 7


def test_max_len_bound_prunes_but_correct(conn):
    txt = "ABxxxCD ABxCD ABxxxxxCD"
    pattern = r"\yAB\y(?:.|\n){0,1}?\yCD\y"
    verifier = PNSPMNSVerifier(conn, mode="pns")
    ok = verifier.matches(txt, pattern, lmin=1)
    expected = pg_regex_matches(conn, txt, pattern)
    assert ok == expected
    stats = verifier.last_stats
    assert stats is not None
    assert stats.get("bound_enabled") in (True, 1)
    assert stats.get("windows_after_bound") <= stats.get("windows_before_bound")


def test_leading_gap_no_miss(conn):
    verifier = PNSPMNSVerifier(conn)
    txt = "xyz ABC tail"
    pattern = r"(?:.|\n){0,3}?\yABC\y"
    expected = pg_regex_matches(conn, txt, pattern)
    assert verifier.matches(txt, pattern, lmin=1) == expected


def test_verify_windows_path(conn):
    txt = "foo bar baz"
    pattern = r"\ybar\y"
    windows = [1, 3, 5, 7, 5, 8]  # (1,3),(5,7),(5,8) => first hit at index 1
    idx = verify_windows(conn, txt, pattern, windows)
    assert idx == 1


def test_nfactor_lite_correctness(conn):
    txt = "foo bar baz"
    pattern = r"\ybar\y"
    verifier_pns = PNSPMNSVerifier(conn, mode="pns")
    verifier_n = PNSPMNSVerifier(conn, mode="pns-n")
    expected = pg_regex_matches(conn, txt, pattern)
    assert verifier_pns.matches(txt, pattern, lmin=1) == expected
    assert verifier_n.matches(txt, pattern, lmin=1) == expected


def test_nfactor_prunes_when_available(conn):
    txt = "foo xxx yyy foo zzz foo"
    pattern = r"\yfoo\y"
    verifier = PNSPMNSVerifier(conn, mode="pns-n")
    expected = pg_regex_matches(conn, txt, pattern)
    assert verifier.matches(txt, pattern, lmin=1) == expected
    stats = verifier.last_stats or {}
    if stats.get("N_enabled"):
        assert stats.get("windows_after_N", stats.get("windows_before_N")) <= stats.get("windows_before_N")


def test_nfactor_prefix_partial_overlap_must_not_prune(conn):
    txt = "ABCDX"
    pattern = r"ABCD"
    verifier = PNSPMNSVerifier(conn, mode="pns-n")
    expected = pg_regex_matches(conn, txt, pattern)
    ok = verifier.matches(txt, pattern, lmin=1)
    assert ok == expected
    stats = verifier.last_stats or {}
    if stats.get("N_enabled"):
        # Because CDX crosses the must-cover prefix, pruning should not drop the only window.
        assert stats.get("windows_after_N", stats.get("windows_before_N")) == stats.get("windows_before_N")
