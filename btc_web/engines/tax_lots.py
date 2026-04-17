"""Lot-level BTC tracking for capital gains tax computation.

Pure Python — depends only on dataclasses and datetime.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class TaxLot:
    date: str          # ISO format YYYY-MM-DD
    btc: float         # BTC amount in this lot
    cost_basis: float  # USD per BTC at acquisition
    source: str        # "initial", "rebal_buy", "scf", "low_q"


@dataclass
class LotGain:
    btc: float
    cost_basis: float    # per-BTC basis of the lot
    sale_price: float    # per-BTC sale price
    proceeds: float      # btc * sale_price
    cost: float          # btc * cost_basis
    gain: float          # proceeds - cost
    is_long_term: bool   # held >= 365 days
    holding_days: int


@dataclass
class SaleResult:
    btc_sold: float
    gains: list[LotGain]
    remaining_lots: list[TaxLot]


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _make_lot_gain(btc_used: float, lot: TaxLot, sale_price: float,
                   holding_days: int, is_long_term: bool) -> LotGain:
    """Construct a LotGain from a lot sale (whole or partial)."""
    proceeds = btc_used * sale_price
    cost = btc_used * lot.cost_basis
    return LotGain(
        btc=btc_used,
        cost_basis=lot.cost_basis,
        sale_price=sale_price,
        proceeds=proceeds,
        cost=cost,
        gain=proceeds - cost,
        is_long_term=is_long_term,
        holding_days=holding_days,
    )


def sell_lots(lots: list[TaxLot], btc_to_sell: float, sale_price: float,
              sale_date: str, method: str = "fifo") -> SaleResult:
    """Sell BTC from lots using FIFO or LIFO cost basis method.

    Args:
        lots: Current lot inventory.
        btc_to_sell: Amount of BTC to sell.
        sale_price: USD per BTC at time of sale.
        sale_date: ISO date string of the sale.
        method: "fifo" (oldest first) or "lifo" (newest first).

    Returns:
        SaleResult with actual btc_sold, list of LotGain records, and
        remaining lots sorted by date ascending.
    """
    sale_dt = _parse_date(sale_date)

    if method == "lifo":
        ordered = sorted(lots, key=lambda x: x.date, reverse=True)
    else:  # fifo (default)
        ordered = sorted(lots, key=lambda x: x.date)

    gains: list[LotGain] = []
    remaining_lots: list[TaxLot] = []
    btc_remaining = btc_to_sell
    btc_sold = 0.0

    for lot in ordered:
        if btc_remaining <= 0:
            remaining_lots.append(lot)
            continue

        lot_dt = _parse_date(lot.date)
        holding_days = (sale_dt - lot_dt).days
        # IRS Pub 544: LT requires holding period to EXCEED one year. The
        # holding period begins the day AFTER acquisition, so a sale on
        # the anniversary date is exactly 365 days and remains short-term.
        is_long_term = holding_days > 365

        if lot.btc <= btc_remaining:
            # Consume entire lot
            btc_used = lot.btc
            btc_remaining -= btc_used
            btc_sold += btc_used
            gains.append(_make_lot_gain(btc_used, lot, sale_price,
                                        holding_days, is_long_term))
        else:
            # Partial lot consumption
            btc_used = btc_remaining
            btc_sold += btc_used
            btc_remaining = 0.0
            gains.append(_make_lot_gain(btc_used, lot, sale_price,
                                        holding_days, is_long_term))
            # Remainder stays in inventory
            remaining_lots.append(TaxLot(
                date=lot.date,
                btc=lot.btc - btc_used,
                cost_basis=lot.cost_basis,
                source=lot.source,
            ))

    # Canonical order: date ascending
    remaining_lots.sort(key=lambda x: x.date)

    return SaleResult(btc_sold=btc_sold, gains=gains, remaining_lots=remaining_lots)


def seed_lots(stack_tracker_lots: list[dict], *, start_stack: float = 0.0,
              start_price: float = 0.0, start_date: str = "") -> list[TaxLot]:
    """Build an initial list of TaxLots for the simulation.

    Priority:
    1. If stack_tracker_lots is non-empty, convert each dict to TaxLot.
    2. Else if start_stack > 0 and start_date provided, create a single lot.
    3. Else return an empty list.

    Args:
        stack_tracker_lots: List of dicts with keys "date", "btc", "price".
        start_stack: BTC amount for the fallback single-lot path.
        start_price: USD/BTC cost basis for the fallback path.
        start_date: ISO date string for the fallback path.

    Returns:
        List of TaxLot, sorted by date ascending.
    """
    if stack_tracker_lots:
        lots = [
            TaxLot(
                date=lot["date"],
                btc=float(lot["btc"]),
                cost_basis=float(lot["price"]),
                source="initial",
            )
            for lot in stack_tracker_lots
        ]
        lots.sort(key=lambda x: x.date)
        return lots

    if start_stack > 0 and start_date:
        return [TaxLot(
            date=start_date,
            btc=float(start_stack),
            cost_basis=float(start_price),
            source="initial",
        )]

    return []
