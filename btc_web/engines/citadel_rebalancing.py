"""Citadel Planner — threshold-based BTC rebalancing triggers."""
from __future__ import annotations

from .citadel_types import CitadelState, SimConfig
from .citadel_transactions import _sell_btc_tracked, _buy_btc_tracked
from .citadel_floors import _distribute_to_accounts, _source_from_accounts

__all__ = ["_evaluate_rebalancing", "_execute_sell_btc", "_execute_buy_btc"]


def _execute_sell_btc(state: CitadelState, config: SimConfig,
                      rate_pct: float, split: dict) -> dict:
    """Sell rate_pct% of BTC stack, distribute proceeds via split."""
    btc_to_sell = state.btc_stack * (rate_pct / 100.0)
    if btc_to_sell <= 0 or state.btc_price <= 0:
        return {}
    result = _sell_btc_tracked(state, config, btc_to_sell)
    if result.btc_sold <= 0:
        return {}
    proceeds = result.btc_sold * state.btc_price
    _distribute_to_accounts(state, proceeds, split)
    return {"action": "sell_btc", "btc_sold": result.btc_sold, "proceeds": proceeds}

def _execute_buy_btc(state: CitadelState, config: SimConfig,
                     rate_pct: float, split: dict) -> dict:
    """Source funds from accounts via split, buy BTC.
    Respects floor rules — won't draw accounts below floors."""
    total_dollar = state.cash + sum(state.reserves) + sum(state.investments)
    target = total_dollar * (rate_pct / 100.0)
    if target <= 0 or state.btc_price <= 0:
        return {}
    sourced = _source_from_accounts(state, target, split, config=config)
    if sourced <= 0:
        return {}
    btc_bought = sourced / state.btc_price
    _buy_btc_tracked(state, config, btc_bought, source="rebal_buy")
    return {"action": "buy_btc", "btc_bought": btc_bought, "cost": sourced}

def _evaluate_rebalancing(state: CitadelState, config: SimConfig,
                          btc_quantile: float) -> None:
    """Evaluate and execute rebalancing triggers. Mutates state."""
    state.rebal_event = None
    if state.rebal_cooldown > 0:
        state.rebal_cooldown -= 1
    # If gradual is active, continue it (ignoring new triggers)
    if state.grad_active:
        if state.grad_remaining > 0:
            if state.grad_direction == "sell_btc":
                evt = _execute_sell_btc(state, config, state.grad_rate, state.grad_split)
            else:
                evt = _execute_buy_btc(state, config, state.grad_rate, state.grad_split)
            state.grad_remaining -= 1
            if evt:
                evt["type"] = "gradual_continue"
                state.rebal_event = evt
        if state.grad_remaining <= 0:
            state.grad_active = False
        return
    # Check high-Q trigger
    if btc_quantile >= config.high_q_trigger:
        action = config.high_q_action
        split = action.get("split", {})
        if action["mode"] == "lump" and state.rebal_cooldown <= 0:
            evt = _execute_sell_btc(state, config, action["rate"], split)
            if evt:
                evt["type"] = "lump_sell"
                state.rebal_event = evt
                state.rebal_cooldown = config.lump_cooldown
        elif action["mode"] == "gradual":
            state.grad_active = True
            state.grad_remaining = action.get("duration", 6)
            state.grad_rate = action["rate"]
            state.grad_direction = "sell_btc"
            state.grad_split = split
            evt = _execute_sell_btc(state, config, state.grad_rate, split)
            state.grad_remaining -= 1
            if evt:
                evt["type"] = "gradual_start"
                state.rebal_event = evt
        return
    # Check low-Q trigger
    if btc_quantile <= config.low_q_trigger:
        action = config.low_q_action
        split = action.get("split", {})
        if action["mode"] == "lump" and state.rebal_cooldown <= 0:
            evt = _execute_buy_btc(state, config, action["rate"], split)
            if evt:
                evt["type"] = "lump_buy"
                state.rebal_event = evt
                state.rebal_cooldown = config.lump_cooldown
        elif action["mode"] == "gradual":
            state.grad_active = True
            state.grad_remaining = action.get("duration", 6)
            state.grad_rate = action["rate"]
            state.grad_direction = "buy_btc"
            state.grad_split = split
            evt = _execute_buy_btc(state, config, state.grad_rate, split)
            state.grad_remaining -= 1
            if evt:
                evt["type"] = "gradual_start"
                state.rebal_event = evt
