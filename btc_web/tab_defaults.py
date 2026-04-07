"""
Single source of truth for all tab default values in Quantoshi.

Constraints:
- Never pass a MappingProxyType directly to json.dumps() — raises TypeError.
  Use dict(DEFAULTS) or a _defaults() function to get a plain dict first.
- Never pass a MappingProxyType directly to a figure builder — use _defaults()
  or dict(DEFAULTS, **overrides) to produce a mutable copy.
- All inner collection values must be tuples/frozensets, not lists/sets.
- Dynamic values (yr_now, _ALL_QS) must NOT be resolved at import time —
  only inside _defaults() functions.
"""

from types import MappingProxyType

BUBBLE = MappingProxyType({
    "selected_qs": (0.5,),
    "xscale": "log", "yscale": "log",
    "auto_y": ("yes",),
    "ymin": 0.03, "ymax": 10**6.05,
    "shade": True, "show_data": True, "show_today": True,
    "show_legend": False, "minor_grid": False,
    "show_ols": False, "show_ucl": False,
    "show_comp": True, "show_sup": True,
    "n_future": 3,
    "pt_size": 8, "pt_alpha": 0.3,
    "stack": 0, "show_stack": False, "use_lots": False,
    "legend_pos": "top-left",
    "comp_color": "#FFD700", "comp_lw": 2.0,
    "sup_color": "#888888", "sup_lw": 1.5,
    "active_models": ("bub",),
    "palette": "default",
    "scanner_lines": (),
    "qs_mode": (),
})

HEATMAP = MappingProxyType({
    "exit_qs": (),
    "color_mode": 0,
    "b1": 0, "b2": 5,
    "hm_palette": "mono",
    "c_lo": "#1a1a1a", "c_mid1": "#555555",
    "c_mid2": "#999999", "c_hi": "#e0e0e0",
    "n_disc": 32,
    "vfmt": "cagr_mult",
    "cell_font_size": 9,
    "show_colorbar": True,
    "stack": 0, "use_lots": False,
    "hm_model": "bub",
    "active_models": (),
    "palette": "default",
})

DCA = MappingProxyType({
    "start_stack": 0, "use_lots": False,
    "amount": 100, "freq": "Monthly", "inflation": 0.0,
    "selected_qs": (0.5,),
    "disp_mode": "btc",
    "annotate": True, "show_today": False,
    "show_legend": False, "minor_grid": False,
    "log_y": False,
    "legend_pos": "bottom-right",
    "active_models": (),
    "palette": "default",
    "sc_enabled": False, "sc_loan_amount": 1200,
    "sc_rate": 13.0, "sc_loan_type": "interest_only",
    "sc_term_months": 12, "sc_repeats": 0, "sc_rollover": False,
    "sc_entry_mode": "live", "sc_custom_price": 80000.0,
    "sc_tax_rate": 0.33,
    "show_qr": True, "show_mc": False,
    "qs_mode": (),
})

RETIRE = MappingProxyType({
    "start_stack": 1.0, "use_lots": False,
    "wd_amount": 3000, "freq": "Monthly",
    "inflation": 4.0,
    "selected_qs": (0.15, 0.85),
    "start_yr": 2031, "end_yr": 2075,
    "disp_mode": "btc",
    "annotate": True, "log_y": True,
    "shade": True,
    "show_legend": False, "minor_grid": True,
    "legend_pos": "bottom-right",
    "active_models": (),
    "palette": "default",
    "show_qr": True, "show_mc": False,
    "qs_mode": (),
})

SUPERCHARGE = MappingProxyType({
    "mode": "a", "start_stack": 1.0, "use_lots": False,
    "start_yr": 2033,
    "delays": (0.0, 0.0, 0.0, 0.0, 2.0),
    "freq": "Monthly", "inflation": 4.0,
    "selected_qs": (0.15, 0.85),
    "chart_layout": 2,
    "display_q": 0.05,
    "wd_amount": 5000, "end_yr": 2075,
    "disp_mode": "usd",
    "annotate": True, "log_y": True,
    "shade": True,
    "show_legend": False, "minor_grid": True,
    "legend_pos": "top-left",
    "target_yr": 2060,
    "active_models": (),
    "palette": "default",
    "show_qr": True, "show_mc": False,
    "qs_mode": (),
})

STACK = MappingProxyType({
    "lot_btc": 0.01,
    "lot_price": 69420,
    "lot_notes": "",
})

CITADEL = MappingProxyType({
    "start_stack": 1.0, "use_lots": False,
    "cash_initial": 20000, "cash_rate": 4.0,
    "res_short_init": 10000, "res_short_rate": 5.0, "res_short_vol": 2.0,
    "res_med_init": 10000, "res_med_rate": 4.5, "res_med_vol": 8.0,
    "res_long_init": 10000, "res_long_rate": 4.0, "res_long_vol": 15.0,
    "inv_eq_init": 100000, "inv_eq_basis": 100000, "inv_eq_rate": 10.0, "inv_eq_vol": 16.0,
    "inv_bd_init": 50000, "inv_bd_basis": 50000, "inv_bd_rate": 5.0, "inv_bd_vol": 7.0,
    "monthly_spend": 5000, "inflation": 4.0, "spend_growth": 0.0,
    "high_q_trigger": 95, "high_q_mode": "gradual", "high_q_rate": 2.0, "high_q_dur": 6,
    "high_q_split_cash": 20, "high_q_split_rs": 20, "high_q_split_rm": 20,
    "high_q_split_rl": 10, "high_q_split_eq": 20, "high_q_split_bd": 10,
    "low_q_trigger": 5, "low_q_mode": "lump", "low_q_rate": 10.0, "low_q_dur": 1,
    "low_q_split_cash": 20, "low_q_split_rs": 20, "low_q_split_rm": 20,
    "low_q_split_rl": 10, "low_q_split_eq": 20, "low_q_split_bd": 10,
    "lump_cooldown": 12,
    "cash_floor": 10000, "res_short_floor": 0, "res_med_floor": 0, "res_long_floor": 0,
    "cash_floor_growth": 0, "reserve_floor_growth": 0,
    "scf_enabled": False, "scf_amount": 100000, "scf_type": "term",
    "scf_rate": 8.0, "scf_term": 60, "scf_repay_trigger": 1.0,
    "start_yr": 2031, "end_yr": 2075, "freq": "Monthly",
    "price_model": "bub", "asset_return_model": "lognormal",
    "selected_qs": (0.25,),
    "disp_mode": "usd_per_asset",
    "annotate": True, "log_y": True, "show_legend": True, "minor_grid": True,
    "legend_pos": "bottom-right",
    "palette": "default",
    # Tax system (off by default)
    "tax_enabled": False,
    "filing_status": "single",
    "state_code": "TX",
    "state_rate_override": None,
    "tcja_sunset": False,
    "birth_year": None,
    "cost_basis_method": "fifo",
    "other_income": 0,
    "other_income_growth": 0,
    "td_btc": 0.5, "td_cash": 20000, "td_res_short": 30000, "td_res_med": 50000, "td_res_long": 0,
    "td_inv_eq": 200000, "td_inv_bd": 100000,
    "tf_btc": 0.5, "tf_cash": 20000, "tf_res_short": 30000, "tf_res_med": 50000, "tf_res_long": 0,
    "tf_inv_eq": 200000, "tf_inv_bd": 100000,
})


def bubble_defaults() -> dict:
    import pandas as pd
    yr_now = pd.Timestamp.today().year
    d = dict(BUBBLE)
    d["xmin"] = 2010
    d["xmax"] = 2033
    d["selected_qs"] = list(BUBBLE["selected_qs"])
    d["active_models"] = list(BUBBLE["active_models"])
    d["scanner_lines"] = list(BUBBLE["scanner_lines"])
    d["auto_y"] = list(BUBBLE["auto_y"])
    d["lots"] = []
    d["user_model"] = None
    return d


def heatmap_defaults() -> dict:
    import pandas as pd
    yr_now = pd.Timestamp.today().year
    d = dict(HEATMAP)
    d["entry_yr"] = yr_now
    d["entry_q"] = 50.0
    d["exit_yr_lo"] = yr_now
    d["exit_yr_hi"] = yr_now + 15
    d["exit_qs"] = list(HEATMAP["exit_qs"])
    d["active_models"] = list(HEATMAP["active_models"])
    d["lots"] = []
    return d


def dca_defaults() -> dict:
    import pandas as pd
    yr_now = pd.Timestamp.today().year
    d = dict(DCA)
    d["start_yr"] = yr_now
    d["end_yr"] = yr_now + 10
    d["selected_qs"] = list(DCA["selected_qs"])
    d["active_models"] = list(DCA["active_models"])
    d["lots"] = []
    d["user_model"] = None
    d["sc_live_price"] = None
    return d


def retire_defaults() -> dict:
    d = dict(RETIRE)
    d["selected_qs"] = list(RETIRE["selected_qs"])
    d["active_models"] = list(RETIRE["active_models"])
    d["lots"] = []
    d["user_model"] = None
    return d


def supercharge_defaults() -> dict:
    d = dict(SUPERCHARGE)
    d["delays"] = list(SUPERCHARGE["delays"])
    d["selected_qs"] = list(SUPERCHARGE["selected_qs"])
    d["active_models"] = list(SUPERCHARGE["active_models"])
    d["lots"] = []
    d["user_model"] = None
    return d


def citadel_defaults() -> dict:
    d = dict(CITADEL)
    d["selected_qs"] = list(CITADEL["selected_qs"])
    d["lots"] = []
    d["user_model"] = None
    return d


# ── Defaults fingerprint (for L0 cache invalidation) ────────────────────────
import hashlib as _hashlib


def _compute_defaults_hash() -> str:
    """Hash all frozen dicts. Changes when any default value changes.

    Uses repr(sorted(items)) — deterministic for primitives and tuples.
    If a set or dict value is ever added, this may become non-deterministic.
    The test_inner_collections_are_tuples test guards against this.
    """
    h = _hashlib.md5()
    for d in (BUBBLE, HEATMAP, DCA, RETIRE, SUPERCHARGE, STACK, CITADEL):
        h.update(repr(sorted(d.items())).encode())
    return h.hexdigest()[:12]


_DEFAULTS_HASH = _compute_defaults_hash()
