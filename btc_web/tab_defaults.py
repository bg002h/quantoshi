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
        # MC controls — prewarm-cache parity with update_bubble runtime.
        # Inner collections stored as tuples to satisfy
        # test_defaults.py::test_inner_collections_are_tuples; bubble_defaults()
        # promotes mc_regime/mc_window to lists for runtime.
        "mc_enabled":     bool(sd("bub-mc-enable:value", []) or []),
        "mc_amount":      sd("bub-mc-amount:value", 100),
        "mc_bins":        sd("bub-mc-bins:value", 5),
        "mc_entry_q":     sd("bub-mc-entry-q:value", 10),
        "mc_freq":        sd("bub-mc-freq:value", "Monthly"),
        "mc_infl":        sd("bub-mc-infl:value", 4),
        "mc_model_src":   sd("bub-mc-model-src:value", "bub"),
        "mc_regime":      tuple(sd("bub-mc-regime:value", [0, 1, 2, 3, 4]) or []),
        "mc_sims":        sd("bub-mc-sims:value", 200),
        "mc_start_yr":    sd("bub-mc-start-yr:value", 2031),
        "mc_window":      tuple(sd("bub-mc-window:value", [2010, 2026]) or []),
        "mc_years":       sd("bub-mc-years:value", 40),
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
        "line_width":    sd("hm-line-width:value", 1.0),
        "line_color":    sd("hm-line-color:value", "#000000"),
        "show_colorbar": "colorbar" in toggles,
        "stack":         sd("hm-stack:value", 0),
        "use_lots":      bool(sd("hm-use-lots:value", []) or []),
        "hm_model":      sd("hm-active-model:data", "bub"),
        "active_models": (),
        "palette":       sd("palette-store:data", "default"),
    }


HEATMAP = MappingProxyType(_build_heatmap_dict())

def _build_dca_dict():
    """Derive DCA figure-params from SNAPSHOT_DEFAULTS widget values.

    Translation notes:
    - dca-sc-tax:value is PERCENT (33); sc_tax_rate is FRACTION (0.33).
    - active_models stays empty tuple (widget has ['bub'] but DCA fig
      callback sources active_models separately).
    - qs_mode is UI-only; popped in dca_defaults() before prewarm key.
    """
    from snapshot_defaults import SNAPSHOT_DEFAULTS as _S
    _BANDS = {"inner": [0.15, 0.85], "outer": [0.01, 0.99], "median": [0.5]}

    def sd(k, default=None):
        return _S.get(k, default)

    def _bands(vals):
        qs = []
        for b in (vals or []):
            qs.extend(_BANDS.get(b, []))
        return tuple(sorted(set(qs)))

    toggles = set(sd("dca-toggles:value", []) or [])
    qs_mode_val = tuple(sd("dca-qs-mode:value", []) or [])
    if "advanced" in qs_mode_val:
        sel_qs = tuple(sd("dca-qs-adv:value", [0.5]) or [0.5])
    else:
        sel_qs = _bands(sd("dca-qs:value", ["median"])) or (0.5,)
    return {
        "start_stack":  sd("dca-stack:value", 0),
        "use_lots":     bool(sd("dca-use-lots:value", []) or []),
        "amount":       sd("dca-amount:value", 100),
        "freq":         sd("dca-freq:value", "Monthly"),
        "inflation":    sd("dca-infl:value", 0.0),
        "selected_qs":  sel_qs,
        "disp_mode":    sd("dca-disp:value", "btc"),
        "annotate":     "annotate"    in toggles,
        "show_today":   "show_today"  in toggles,
        "show_legend":  "show_legend" in toggles,
        "minor_grid":   "minor_grid"  in toggles,
        "log_y":        "log_y"       in toggles,
        "discrete":     "discrete"    in toggles,
        "shade":        "shade"       in toggles,
        "legend_pos":   sd("dca-legend-pos:value", "bottom-right"),
        "active_models": (),
        "palette":      sd("palette-store:data", "default"),
        "sc_enabled":   bool(sd("dca-sc-enable:value", []) or []),
        "sc_loan_amount": sd("dca-sc-loan:value", 1200),
        "sc_rate":      sd("dca-sc-rate:value", 13.0),
        "sc_loan_type": sd("dca-sc-type:value", "interest_only"),
        "sc_term_months": sd("dca-sc-term:value", 12),
        "sc_repeats":   sd("dca-sc-repeats:value", 0),
        "sc_rollover":  bool(sd("dca-sc-rollover:value", []) or []),
        "sc_entry_mode": sd("dca-sc-entry-mode:value", "live"),
        "sc_custom_price": sd("dca-sc-custom-price:value", 80000.0),
        "sc_tax_rate":  float(sd("dca-sc-tax:value", 33)) / 100.0,
        "show_qr":      True,
        "show_mc":      False,
        "qs_mode":      qs_mode_val,
    }


DCA = MappingProxyType(_build_dca_dict())

def _build_retire_dict():
    """Derive RETIRE figure-params from SNAPSHOT_DEFAULTS widget values.

    show_mc=False is the prewarm baseline; the runtime callback computes
    the effective value from ret-mc-enable. Keeping the static False
    preserves cache-key alignment.
    """
    from snapshot_defaults import SNAPSHOT_DEFAULTS as _S
    _BANDS = {"inner": [0.15, 0.85], "outer": [0.01, 0.99], "median": [0.5]}

    def sd(k, default=None):
        return _S.get(k, default)

    def _bands(vals):
        qs = []
        for b in (vals or []):
            qs.extend(_BANDS.get(b, []))
        return tuple(sorted(set(qs)))

    toggles = set(sd("ret-toggles:value", []) or [])
    yr = sd("ret-yr-range:value", [2031, 2075])
    qs_mode_val = tuple(sd("ret-qs-mode:value", []) or [])
    if "advanced" in qs_mode_val:
        sel_qs = tuple(sd("ret-qs-adv:value", [0.15, 0.85]) or [0.15, 0.85])
    else:
        sel_qs = _bands(sd("ret-qs:value", ["inner"])) or (0.15, 0.85)
    return {
        "start_stack": sd("ret-stack:value", 1.0),
        "use_lots":    bool(sd("ret-use-lots:value", []) or []),
        "wd_amount":   sd("ret-wd:value", 3000),
        "freq":        sd("ret-freq:value", "Monthly"),
        "inflation":   sd("ret-infl:value", 4.0),
        "selected_qs": sel_qs,
        "start_yr":    int(yr[0]),
        "end_yr":      int(yr[1]),
        "disp_mode":   sd("ret-disp:value", "btc"),
        "annotate":    "annotate"    in toggles,
        "log_y":       "log_y"       in toggles,
        "shade":       "shade"       in toggles,
        "discrete":    "discrete"    in toggles,
        "show_legend": "show_legend" in toggles,
        "minor_grid":  "minor_grid"  in toggles,
        "legend_pos":  sd("ret-legend-pos:value", "bottom-right"),
        "active_models": (),
        "palette":     sd("palette-store:data", "default"),
        "show_qr":     True,
        "show_mc":     False,
        "qs_mode":     qs_mode_val,
    }


RETIRE = MappingProxyType(_build_retire_dict())

def _build_supercharge_dict():
    """Derive SUPERCHARGE figure-params from SNAPSHOT_DEFAULTS.

    Translation: sc-chart-layout:value is a checklist (['shade'] = bands
    on = layout 2; [] = bands off = layout 1). The 5 sc-d0..sc-d4 scalars
    pack into delays tuple.
    """
    from snapshot_defaults import SNAPSHOT_DEFAULTS as _S
    _BANDS = {"inner": [0.15, 0.85], "outer": [0.01, 0.99], "median": [0.5]}

    def sd(k, default=None):
        return _S.get(k, default)

    def _bands(vals):
        qs = []
        for b in (vals or []):
            qs.extend(_BANDS.get(b, []))
        return tuple(sorted(set(qs)))

    toggles = set(sd("sc-toggles:value", []) or [])
    layout_val = sd("sc-chart-layout:value", []) or []
    qs_mode_val = tuple(sd("sc-qs-mode:value", []) or [])
    if "advanced" in qs_mode_val:
        sel_qs = tuple(sd("sc-qs-adv:value", [0.15, 0.85]) or [0.15, 0.85])
    else:
        sel_qs = _bands(sd("sc-qs:value", ["inner"])) or (0.15, 0.85)
    delays = tuple(float(sd(f"sc-d{i}:value", 0.0) or 0.0) for i in range(5))
    return {
        "mode":        sd("sc-mode:value", "a"),
        "start_stack": sd("sc-stack:value", 1.0),
        "use_lots":    bool(sd("sc-use-lots:value", []) or []),
        "start_yr":    sd("sc-start-yr:value", 2033),
        "delays":      delays,
        "freq":        sd("sc-freq:value", "Monthly"),
        "inflation":   sd("sc-infl:value", 4.0),
        "selected_qs": sel_qs,
        "chart_layout": 2 if "shade" in layout_val else 1,
        "display_q":   sd("sc-display-q:value", 0.05),
        "wd_amount":   sd("sc-wd:value", 5000),
        "end_yr":      sd("sc-end-yr:value", 2075),
        "disp_mode":   sd("sc-disp:value", "usd"),
        "annotate":    "annotate"    in toggles,
        "log_y":       "log_y"       in toggles,
        "discrete":    "discrete"    in toggles,
        "show_legend": "show_legend" in toggles,
        "minor_grid":  "minor_grid"  in toggles,
        "legend_pos":  sd("sc-legend-pos:value", "top-left"),
        "target_yr":   sd("sc-target-yr:value", 2060),
        "active_models": (),
        "palette":     sd("palette-store:data", "default"),
        "show_qr":     True,
        "show_mc":     False,
        "qs_mode":     qs_mode_val,
        "is_mobile":   False,
    }


SUPERCHARGE = MappingProxyType(_build_supercharge_dict())

STACK = MappingProxyType({
    "lot_btc": 0.01,
    "lot_price": 69420,
    "lot_notes": "",
})

def _build_citadel_dict():
    """Derive CITADEL figure-params from SNAPSHOT_DEFAULTS.

    cp-qs:value stores a single FLOAT (0.25), not a list — unique to CITADEL.
    Tax fields kept as static constants here; cp-tax-config:data widget
    stores them as a single nested dict (default {}); the figure builder
    applies its own per-field defaults when the dict is empty.
    """
    from snapshot_defaults import SNAPSHOT_DEFAULTS as _S

    def sd(k, default=None):
        return _S.get(k, default)

    toggles = set(sd("cp-toggles:value", []) or [])
    yr = sd("cp-yr-range:value", [2031, 2075])
    cp_qs_val = sd("cp-qs:value", 0.25)
    sel_qs = (float(cp_qs_val),) if cp_qs_val is not None else (0.25,)
    return {
        "start_stack":  sd("cp-stack:value", 1.0),
        "use_lots":     bool(sd("cp-use-lots:value", []) or []),
        "cash_initial": sd("cp-cash-init:value", 20000),
        "cash_rate":    sd("cp-cash-rate:value", 4.0),
        "res_short_init": sd("cp-res-short-init:value", 10000),
        "res_short_rate": sd("cp-res-short-rate:value", 5.0),
        "res_short_vol":  sd("cp-res-short-vol:value", 2.0),
        "res_med_init":  sd("cp-res-med-init:value", 10000),
        "res_med_rate":  sd("cp-res-med-rate:value", 4.5),
        "res_med_vol":   sd("cp-res-med-vol:value", 8.0),
        "res_long_init": sd("cp-res-long-init:value", 10000),
        "res_long_rate": sd("cp-res-long-rate:value", 4.0),
        "res_long_vol":  sd("cp-res-long-vol:value", 15.0),
        "inv_eq_init":  sd("cp-inv-eq-init:value", 100000),
        "inv_eq_basis": sd("cp-inv-eq-basis:value", 100000),
        "inv_eq_rate":  sd("cp-inv-eq-rate:value", 10.0),
        "inv_eq_vol":   sd("cp-inv-eq-vol:value", 16.0),
        "inv_bd_init":  sd("cp-inv-bd-init:value", 50000),
        "inv_bd_basis": sd("cp-inv-bd-basis:value", 50000),
        "inv_bd_rate":  sd("cp-inv-bd-rate:value", 5.0),
        "inv_bd_vol":   sd("cp-inv-bd-vol:value", 7.0),
        "monthly_spend": sd("cp-spend:value", 5000),
        "inflation":    sd("cp-infl:value", 4.0),
        "spend_growth": sd("cp-spend-growth:value", 0.0),
        "high_q_trigger": sd("cp-high-q-thresh:value", 95),
        "high_q_mode":  sd("cp-high-q-mode:value", "gradual"),
        "high_q_rate":  sd("cp-high-q-rate:value", 2.0),
        "high_q_dur":   sd("cp-high-q-dur:value", 6),
        "high_q_split_cash": sd("cp-high-q-split-cash:value", 20),
        "high_q_split_rs":   sd("cp-high-q-split-rs:value", 20),
        "high_q_split_rm":   sd("cp-high-q-split-rm:value", 20),
        "high_q_split_rl":   sd("cp-high-q-split-rl:value", 10),
        "high_q_split_eq":   sd("cp-high-q-split-eq:value", 20),
        "high_q_split_bd":   sd("cp-high-q-split-bd:value", 10),
        "low_q_trigger": sd("cp-low-q-thresh:value", 5),
        "low_q_mode":   sd("cp-low-q-mode:value", "lump"),
        "low_q_rate":   sd("cp-low-q-rate:value", 10.0),
        "low_q_dur":    sd("cp-low-q-dur:value", 1),
        "low_q_split_cash": sd("cp-low-q-split-cash:value", 20),
        "low_q_split_rs":   sd("cp-low-q-split-rs:value", 20),
        "low_q_split_rm":   sd("cp-low-q-split-rm:value", 20),
        "low_q_split_rl":   sd("cp-low-q-split-rl:value", 10),
        "low_q_split_eq":   sd("cp-low-q-split-eq:value", 20),
        "low_q_split_bd":   sd("cp-low-q-split-bd:value", 10),
        "lump_cooldown": sd("cp-lump-cooldown:value", 12),
        "cash_floor":    sd("cp-cash-floor:value", 10000),
        "res_short_floor": sd("cp-res-short-floor:value", 0),
        "res_med_floor":  sd("cp-res-med-floor:value", 0),
        "res_long_floor": sd("cp-res-long-floor:value", 0),
        "cash_floor_growth": sd("cp-cash-floor-growth:value", 0),
        "reserve_floor_growth": sd("cp-res-floor-growth:value", 0),
        "scf_enabled":  bool(sd("cp-scf-enable:value", []) or []),
        "scf_amount":   sd("cp-scf-amount:value", 100000),
        "scf_type":     sd("cp-scf-type:value", "term"),
        "scf_rate":     sd("cp-scf-rate:value", 8.0),
        "scf_term":     sd("cp-scf-term:value", 60),
        "scf_repay_trigger": sd("cp-scf-trigger:value", 1.0),
        "start_yr":     int(yr[0]),
        "end_yr":       int(yr[1]),
        "freq":         sd("cp-freq:value", "Monthly"),
        "price_model":  sd("cp-model-src:value", "bub"),
        "asset_return_model": sd("cp-asset-model:value", "lognormal"),
        "selected_qs":  sel_qs,
        "disp_mode":    sd("cp-disp:value", "usd_per_asset"),
        "annotate":     "annotate"    in toggles,
        "log_y":        "log_y"       in toggles,
        "show_legend":  "show_legend" in toggles,
        "minor_grid":   "minor_grid"  in toggles,
        "legend_pos":   sd("cp-legend-pos:value", "bottom-right"),
        "palette":      sd("palette-store:data", "default"),
        # Tax system (off by default; widget stores nested in cp-tax-config)
        "tax_enabled":  bool(sd("cp-tax-toggle:value", False)),
        "filing_status": "single",
        "state_code":   "TX",
        "state_rate_override": None,
        "tcja_sunset":  False,
        "birth_year":   None,
        "cost_basis_method": "fifo",
        "other_income": 0,
        "other_income_growth": 0,
        "td_btc": sd("cp-td-btc:value", 0.5),
        "td_cash": sd("cp-td-cash:value", 20000),
        "td_res_short": sd("cp-td-res-short:value", 30000),
        "td_res_med": sd("cp-td-res-med:value", 50000),
        "td_res_long": sd("cp-td-res-long:value", 0),
        "td_inv_eq": sd("cp-td-inv-eq:value", 200000),
        "td_inv_bd": sd("cp-td-inv-bd:value", 100000),
        "tf_btc": sd("cp-tf-btc:value", 0.5),
        "tf_cash": sd("cp-tf-cash:value", 20000),
        "tf_res_short": sd("cp-tf-res-short:value", 30000),
        "tf_res_med": sd("cp-tf-res-med:value", 50000),
        "tf_res_long": sd("cp-tf-res-long:value", 0),
        "tf_inv_eq": sd("cp-tf-inv-eq:value", 200000),
        "tf_inv_bd": sd("cp-tf-inv-bd:value", 100000),
    }


CITADEL = MappingProxyType(_build_citadel_dict())


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
    # MC inner collections — stored as tuples in BUBBLE; promote to lists
    # for runtime parity with the update_bubble callback's params dict.
    d["mc_regime"] = list(BUBBLE["mc_regime"])
    d["mc_window"] = list(BUBBLE["mc_window"])
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
