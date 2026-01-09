from freepg.regex_literals import extract_literals


def test_simple_boundary_literal():
    res = extract_literals(r"\yCA\-274\y")
    assert res.safe
    assert res.literals == ["CA-274"]
    assert res.used_template


def test_simple_alternation():
    res = extract_literals(r"(?:foo|bar)")
    assert res.safe
    assert res.literals == [["foo", "bar"]]


def test_gap_then_literal():
    res = extract_literals(r"(?:.|\n){0,3}?abc")
    assert res.safe
    assert res.literals == ["abc"]
    assert res.used_template


def test_backreference_is_unsafe():
    res = extract_literals(r"(foo)\1")
    assert not res.safe
    assert res.literals == []


def test_lookahead_is_unsafe():
    res = extract_literals(r"(?=foo)bar")
    assert not res.safe
    assert res.literals == []


def test_charclass_literal_tail():
    res = extract_literals(r"[A-Za-z0-9_]{2,2}ego[A-Za-z0-9_]{0,0}")
    assert res.safe
    assert "ego" in res.literals


def test_boundary_token():
    res = extract_literals(r"\yI\-5\y")
    assert res.safe
    assert res.literals == ["I-5"]
