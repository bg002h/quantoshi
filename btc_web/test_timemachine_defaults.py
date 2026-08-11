"""Task 8: asof_date must be a first-class key in bubble_defaults() so the
as-of chart is cache-keyed correctly (Time Machine feature)."""
from tab_defaults import bubble_defaults


def test_asof_default_none():
    assert bubble_defaults().get("asof_date", "MISSING") is None
