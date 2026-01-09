import pytest

from baselines.pns_pmns_v1.prefilter_sql import (
    build_where_for_necessary_literals,
    escape_like_literal,
    merge_with_user_where,
)


def test_escape_like_literal():
    assert escape_like_literal("a%b_c\\d") == "a\\%b\\_c\\\\d"
    assert escape_like_literal("") == ""


def test_build_where_for_necessary_literals_basic():
    frag, params = build_where_for_necessary_literals("col", ["ab", "c%_\\d"], use_ilike=False, k=2)
    assert "LIKE %s ESCAPE E'\\\\'" in frag
    assert set(params) == {"%ab%", "%c\\%\\_\\\\d%"}


def test_merge_with_user_where():
    user_where = "x > 1"
    extra = "(col LIKE %s)"
    merged = merge_with_user_where(user_where, extra)
    assert merged == f"({user_where}) AND ({extra})"
    merged_only = merge_with_user_where(None, extra)
    assert merged_only == extra
    merged_none = merge_with_user_where(user_where, "")
    assert merged_none == user_where
