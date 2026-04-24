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
from colors import (LOT_MARKER_COLOR, FALLBACK_MODEL_GRAY, NEAR_BLACK,
                    DIM_TEXT, SPINE_COLOR_FALLBACK, PROGRESS_TRACK,
                    PT_SIZE_DEFAULT, PT_ALPHA_DEFAULT,
                    TRACE_WIDTH_COMPOSITE, TRACE_WIDTH_SUPPORT,
                    CHART_FONT_WATERMARK,
                    HM_DEFAULT_RED, HM_DEFAULT_GREEN, WHITE)

def _build_bubble_dict():
    """Derive BUBBLE figure-params dict from SNAPSHOT_DEFAULTS widget values.

    bub-qs widget stores BAND LABELS (e.g. ['median']); _bands_to_qs
    translates to quantile floats. bub-qs-adv stores quantile floats
    directly. Default qs_mode is [] (default mode) so we use bub-qs.
    """
    from snapshot_defaults import SNAPSHOT_DEFAULTS as _S

    # Inlined to avoid circular import via layout.common -> layout.bubble ->
    # tab_defaults. Mirrors layout.common._DEFAULT_BANDS / _bands_to_qs.
    _BANDS = {"inner": [0.15, 0.85], "outer": [0.01, 0.99], "median": [0.5]}
    def _bands_to_qs(band_values):
        qs = []
        for b in (band_values or []):
            qs.extend(_BANDS.get(b, []))
        return sorted(set(qs))

    def sd(k, default=None):
        return _S.get(k, default)

    yr = sd("bub-yrange:value", [-1.5, 6.05])
    xr = sd("bub-xrange:value", [2010, 2033])
    toggles = set(sd("bub-toggles:value", []) or [])
    btoggles = set(sd("bub-bubble-toggles:value", []) or [])
    qs_mode_val = tuple(sd("bub-qs-mode:value", []) or [])
    if "advanced" in qs_mode_val:
        selected_qs = tuple(sd("bub-qs-adv:value", [0.5]) or [0.5])
    else:
        selected_qs = tuple(_bands_to_qs(sd("bub-qs:value", ["median"])))
        if not selected_qs:
            selected_qs = (0.5,)
    return {
        "selected_qs": selected_qs,
        "xscale":      sd("bub-xscale:value", "log"),
        "yscale":      sd("bub-yscale:value", "log"),
        # ymin/ymax must match bub-yrange so prewarm key aligns with first-
        # render callback key. xrange feeds xmin/xmax in bubble_defaults().
        "ymin":        10 ** float(yr[0]),
        "ymax":        10 ** float(yr[1]),
        "shade":       "shade"       in toggles,
        "show_data":   "show_data"   in toggles,
        "show_today":  "show_today"  in toggles,
        "show_legend": "show_legend" in toggles,
        "minor_grid":  "minor_grid"  in toggles,
        "show_ols":    "show_ols"    in toggles,
        "show_ucl":    "show_ucl"    in toggles,
        "show_comp":   "show_comp"   in btoggles,
        "show_sup":    "show_sup"    in btoggles,
        "n_future":    sd("bub-n-future:value", 3),
        "pt_size":     sd("bub-ptsize:value",  PT_SIZE_DEFAULT),
        "pt_alpha":    sd("bub-ptalpha:value", PT_ALPHA_DEFAULT),
        "stack":       sd("bub-stack:value", 0),
        "show_stack":  False,
        "use_lots":    False,
        "legend_pos":  sd("bub-legend-pos:value", "top-left"),
        # Composite/support style constants — not exposed as widgets.
        "comp_color":  LOT_MARKER_COLOR,
        "comp_lw":     TRACE_WIDTH_COMPOSITE,
        "sup_color":   FALLBACK_MODEL_GRAY,
        "sup_lw":      TRACE_WIDTH_SUPPORT,
        "active_models": tuple(sd("bub-model-show:value", ["bub"]) or ["bub"]),
        "palette":     sd("palette-store:data", "default"),
        "scanner_lines": (),
        "qs_mode":     qs_mode_val,
        "sigma_mode":  sd("bub-sigma-mode:value", "resqr"),
        # Derived-from-State keys callbacks always inject; must appear in
        # defaults so the prewarm cache key aligns.
        "decomp_model": "",
        "decomp_mode":  "individual",
        "decomp_components": (),
        "lppl_n_freqs": (),
        "lppl_weighted": (),
        "lppl_no_13":  (),
        "config_b_keys": (),
        "_xrange": (int(xr[0]), int(xr[1])),  # internal: feeds xmin/xmax
    }


_BUBBLE_RAW = _build_bubble_dict()
_BUBBLE_XRANGE = _BUBBLE_RAW.pop("_xrange")
BUBBLE = MappingProxyType(_BUBBLE_RAW)

def _build_heatmap_dict():
    """Derive HEATMAP figure-params from SNAPSHOT_DEFAULTS widget values.

    hm_palette is a UI-only control — seeds the dropdown but the heatmap
    callback never puts it in its runtime params dict. heatmap_defaults()
    pops it before the prewarm cache key is built. live_price is stripped
    from the cache key in utils._get_heatmap_fig.
    """
    from snapshot_defaults import SNAPSHOT_DEFAULTS as _S
    sd = lambda k, default=None: _S.get(k, default)
    toggles = set(sd("hm-toggles:value", []) or [])
    return {
        "exit_qs":       (),  # heatmap_defaults() overrides with _DEF_QS
        "color_mode":    sd("hm-mode:value", 0),
        "b1":            sd("hm-b1:value", 0),
        "b2":            sd("hm-b2:value", 5),
        "hm_palette":    sd("hm-palette:value", "rwg"),
        "c_lo":          sd("hm-c-lo:value",   HM_DEFAULT_RED),
        "c_mid1":        sd("hm-c-mid1:value", WHITE),
        "c_mid2":        sd("hm-c-mid2:value", WHITE),
        "c_hi":          sd("hm-c-hi:value",   HM_DEFAULT_GREEN),
        "n_disc":        sd("hm-grad:value", 32),
        "vfmt":          sd("hm-vfmt:value", "cagr_mult"),
        "cell_font_size": sd("hm-cell-fs:value", CHART_FONT_WATERMARK),
        "show_colorbar": "colorbar" in toggles,
        "stack":         sd("hm-stack:value", 0),
        "use_lots":      bool(sd("hm-use-lots:value", []) or []),
        "hm_model":      sd("hm-active-model:data", "bub"),
        "active_models": (),
        "palette":       sd("palette-store:data", "default"),
    }


HEATMAP = MappingProxyType(_build_heatmap_dict())

DCA = MappingProxyType({
    "start_stack": 0, "use_lots": False,
    "amount": 100, "freq": "Monthly", "inflation": 0.0,
    "selected_qs": (0.5,),
    "disp_mode": "btc",
    "annotate": True, "show_today": False,
    "show_legend": False, "minor_grid": False,
    "log_y": False,
    "discrete": False, "shade": False,
    "legend_pos": "bottom-right",
    "active_models": (),
    "palette": "default",
    "sc_enabled": False, "sc_loan_amount": 1200,
    "sc_rate": 13.0, "sc_loan_type": "interest_only",
    "sc_term_months": 12, "sc_repeats": 0, "sc_rollover": False,
    "sc_entry_mode": "live", "sc_custom_price": 80000.0,
    "sc_tax_rate": 0.33,
    "show_qr": True, "show_mc": False,
    # qs_mode is a UI control (sel-qs vs adv-qs selector) consumed by
    # routing logic in the callback — never placed into the figure builder's
    # params dict. dca_defaults() pops it before the prewarm cache key is
    # built so the key matches the runtime key.
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
    "shade": True, "discrete": False,
    "show_legend": False, "minor_grid": False,
    "legend_pos": "bottom-right",
    "active_models": (),
    "palette": "default",
    "show_qr": True, "show_mc": False,
    # qs_mode — see DCA note. retire_defaults() pops it.
    "qs_mode": (),
})

SUPERCHARGE = MappingProxyType({
    "mode": "a", "start_stack": 1.0, "use_lots": False,
    "start_yr": 2033,
    "delays": (0.0, 0.0, 0.0, 0.0, 2.0),
    "freq": "Monthly", "inflation": 4.0,
    "selected_qs": (0.15, 0.85),
    # chart_layout=2 encodes "shade bands on" (the shade checkbox is the
    # UI-level toggle; chart_layout is what the figure builder consumes).
    "chart_layout": 2,
    "display_q": 0.05,
    "wd_amount": 5000, "end_yr": 2075,
    "disp_mode": "usd",
    "annotate": True, "log_y": True,
    "discrete": False,
    "show_legend": False, "minor_grid": False,
    "legend_pos": "top-left",
    "target_yr": 2060,
    "active_models": (),
    "palette": "default",
    "show_qr": True, "show_mc": False,
    # qs_mode — see DCA note. supercharge_defaults() pops it.
    "qs_mode": (),
    # is_mobile is a per-client viewport-derived flag in the runtime params
    # dict; kept here so defaults baseline matches a desktop viewer.
    "is_mobile": False,
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
    "annotate": True, "log_y": True, "show_legend": True, "minor_grid": False,
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
    d = dict(BUBBLE)
    d["xmin"] = _BUBBLE_XRANGE[0]
    d["xmax"] = _BUBBLE_XRANGE[1]
    d["selected_qs"] = list(BUBBLE["selected_qs"])
    d["active_models"] = list(BUBBLE["active_models"])
    d["scanner_lines"] = list(BUBBLE["scanner_lines"])
    d["decomp_components"] = list(BUBBLE["decomp_components"])
    d["lppl_n_freqs"] = list(BUBBLE["lppl_n_freqs"])
    d["lppl_weighted"] = list(BUBBLE["lppl_weighted"])
    d["lppl_no_13"] = list(BUBBLE["lppl_no_13"])
    d["config_b_keys"] = list(BUBBLE["config_b_keys"])
    d["qs_mode"] = list(BUBBLE["qs_mode"])
    d["lots"] = []
    d["user_model"] = None
    return d


def heatmap_defaults() -> dict:
    import pandas as pd
    import _app_ctx
    yr_now = pd.Timestamp.today().year
    d = dict(HEATMAP)
    d["entry_yr"] = yr_now
    d["entry_q"] = 50.0
    d["exit_yr_lo"] = yr_now
    d["exit_yr_hi"] = yr_now + 10
    # Use _app_ctx._DEF_QS so the prewarm cache matches the live layout
    # default in heatmap.py's hm-exit-qs checklist. HEATMAP["exit_qs"] is
    # legacy empty () and would bake a "No data" error into the L1 cache.
    d["exit_qs"] = list(_app_ctx._DEF_QS)
    d["active_models"] = list(HEATMAP["active_models"])
    d["lots"] = []
    # hm_palette is a layout-only key that the heatmap callback never puts
    # in its runtime params dict; strip it so the prewarm key aligns.
    d.pop("hm_palette", None)
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
    # qs_mode is a UI-only selector — the DCA callback routes it to choose
    # between sel_qs and adv_qs but never passes it to the figure builder.
    d.pop("qs_mode", None)
    return d


def retire_defaults() -> dict:
    d = dict(RETIRE)
    d["selected_qs"] = list(RETIRE["selected_qs"])
    d["active_models"] = list(RETIRE["active_models"])
    d["lots"] = []
    d["user_model"] = None
    d.pop("qs_mode", None)
    return d


def supercharge_defaults() -> dict:
    d = dict(SUPERCHARGE)
    d["delays"] = list(SUPERCHARGE["delays"])
    d["selected_qs"] = list(SUPERCHARGE["selected_qs"])
    d["active_models"] = list(SUPERCHARGE["active_models"])
    d["lots"] = []
    d["user_model"] = None
    d.pop("qs_mode", None)
    return d


def citadel_defaults() -> dict:
    d = dict(CITADEL)
    d["selected_qs"] = list(CITADEL["selected_qs"])
    d["lots"] = []
    d["user_model"] = None
    return d


# Leverage Calculator tab — see docs/superpowers/specs/2026-04-18-leverage-calculator-design.md
LEVERAGE_DEFAULTS = MappingProxyType({
    "lev_date":     None,   # resolved to date.today() in leverage_defaults()
    "lev_price":    None,   # resolved to most recent model_data close in leverage_defaults()
    "lev_model":    "bub",
    "lev_floor_q":  0.01,
    "lev_rb":       13.0,
    "lev_rl":       4.5,
    "lev_horizon":  4.0,
    "lev_cagr":     20.0,
    "lev_toggles":  ("legend",),
})


def leverage_defaults():
    """Return a plain dict with dynamic fields resolved.

    Current price is the live ticker value (Binance/CoinGecko via
    utils._fetch_btc_price, 20-min cache), falling back to the model's
    latest daily close if the fetch fails.
    """
    import datetime as _dt
    import _app_ctx
    md = _app_ctx.M  # ModelData instance — set at app.py
    model_close = round(float(md.price_prices[-1]), 2) if md is not None else 65000.0
    try:
        from utils import _fetch_btc_price
        live = _fetch_btc_price()
        current_price = round(float(live), 2) if live is not None else model_close
    except Exception:
        current_price = model_close
    d = dict(LEVERAGE_DEFAULTS)
    d["lev_date"] = _dt.date.today().isoformat()
    d["lev_price"] = current_price
    return d


# ── Defaults fingerprint (for L0 cache invalidation) ────────────────────────
import hashlib as _hashlib


def _compute_defaults_hash() -> str:
    """Cache-invalidation fingerprint. Now sourced from
    snapshot_defaults._compute_snapshot_defaults_fingerprint() — broader
    scope (covers all 310 _SNAPSHOT_CONTROLS entries, not just the 7
    frozen tab dicts), more accurate L0 invalidation."""
    from snapshot_defaults import _compute_snapshot_defaults_fingerprint
    return _compute_snapshot_defaults_fingerprint()


_DEFAULTS_HASH = _compute_defaults_hash()
