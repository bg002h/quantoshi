"""Citadel Planner — BTC and investment transaction helpers with cost basis tracking."""
from __future__ import annotations

from .citadel_types import CitadelState, SimConfig, _SATOSHI

__all__ = ["_sell_btc_tracked", "_buy_btc_tracked", "_sell_investments_tracked"]


def _sell_btc_tracked(state: CitadelState, config: SimConfig,
                      btc_to_sell: float) -> "SaleResult":
    """Sell BTC with lot tracking + accumulator update. Returns SaleResult.

    If lots exist, uses sell_lots() for proper cost basis tracking.
    If lots are empty (defensive), does raw stack decrement.
    Accumulator gains are only recorded when state.tax_year_accum is not None.
    """
    from .tax_lots import sell_lots, SaleResult

    if btc_to_sell <= 0:
        return SaleResult(btc_sold=0.0, gains=[], remaining_lots=list(state.tax_lots))

    if state.tax_lots:
        result = sell_lots(
            state.tax_lots, btc_to_sell, state.btc_price,
            state.sim_date, method=config.cost_basis_method,
        )
        state.btc_stack -= result.btc_sold
        state.tax_lots = result.remaining_lots
        if state.tax_year_accum is not None:
            for g in result.gains:
                if g.is_long_term:
                    if g.gain >= 0:
                        state.tax_year_accum.lt_capital_gains += g.gain
                    else:
                        state.tax_year_accum.lt_capital_losses += abs(g.gain)
                else:
                    if g.gain >= 0:
                        state.tax_year_accum.st_capital_gains += g.gain
                    else:
                        state.tax_year_accum.st_capital_losses += abs(g.gain)
        return result
    else:
        btc_sold = min(btc_to_sell, state.btc_stack)
        state.btc_stack -= btc_sold
        return SaleResult(btc_sold=btc_sold, gains=[], remaining_lots=[])


def _buy_btc_tracked(state: CitadelState, config: SimConfig,
                     btc_bought: float, source: str = "rebal_buy") -> None:
    """Buy BTC and create a tax lot for cost basis tracking."""
    if btc_bought <= 0:
        return
    from .tax_lots import TaxLot
    state.btc_stack += btc_bought
    state.tax_lots.append(TaxLot(
        date=state.sim_date, btc=btc_bought,
        cost_basis=state.btc_price, source=source,
    ))


def _sell_investments_tracked(state: CitadelState, config: SimConfig,
                              bin_index: int, amount: float) -> tuple:
    """Sell from investment bin with cost basis tracking + accumulator update.

    Returns (amount_drawn, gain). Gain is positive for profit, negative for loss.
    Accumulator update only when state.tax_year_accum is not None.
    """
    current = state.investments[bin_index]
    if current <= 0 or amount <= 0:
        return (0.0, 0.0)
    draw = min(current, amount)
    fraction = draw / current
    basis_sold = state.invest_cost_basis[bin_index] * fraction
    state.invest_cost_basis[bin_index] -= basis_sold
    state.investments[bin_index] -= draw
    gain = draw - basis_sold
    if state.tax_year_accum is not None:
        if gain >= 0:
            state.tax_year_accum.lt_capital_gains += gain
        else:
            state.tax_year_accum.lt_capital_losses += abs(gain)
    return (draw, gain)
