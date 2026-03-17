"""Type coercion helpers and lot formatting."""

from btc_core import fmt_price


def _format_lots_for_table(lots: list[dict]) -> list[dict]:
    """Format lot dicts for the Stack Tracker DataTable display."""
    return [
        {**l,
         "total_paid": fmt_price(l["btc"] * l["price"]),
         "pct_q":      f"Q{l['pct_q']*100:.2f}%"}
        for l in lots
    ]


# ── Type coercion helpers ───────────────────────────────────────────────────
# Use `is not None` instead of `or` to handle 0 as a valid value.

def _ci(val: object, default: int, lo: int | None = None, hi: int | None = None) -> int:
    """Coerce to int. Returns default when val is None."""
    v = int(float(val)) if val is not None else default
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return v


def _cf(val: object, default: float, lo: float | None = None, hi: float | None = None) -> float:
    """Coerce to float. Returns default when val is None."""
    v = float(val) if val is not None else default
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return v
