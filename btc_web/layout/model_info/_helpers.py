"""Helper functions for Model Info tab — table/row builders + clickable image.

Split from layout/model_info.py; all helpers share the _app_ctx-backed
live model lookups and the shared coeff_table/qr_table/comparison_table
scaffolding.
"""

from dash import html
import dash_bootstrap_components as dbc  # noqa: F401  (some helpers need it indirectly via imports)

import _app_ctx
from colors import (
    FALLBACK_MODEL_GRAY,
    TABLE_HEADER_BG,
    TABLE_BORDER_LIGHT, TABLE_BORDER_MID, TABLE_BORDER_DARK,
    USER_MODEL_TRACE, _hex_alpha,
    UI_FONT_SM, UI_FONT_MD, UI_FONT_BASE, UI_FONT_LG,
)


def _clickable_img(src, max_width="700px"):
    """Image that opens in a lightbox modal on click."""
    return html.Img(
        src=src,
        className="model-info-img",
        n_clicks=0,
        id={"type": "mi-img", "src": src},
        style={"width": "100%", "maxWidth": max_width,
               "borderRadius": "8px", "marginBottom": "16px",
               "cursor": "zoom-in"},
    )


def _hybppl_dd_rows():
    """Live coefficient table for HybPPL (DD) — pulls from model class."""
    mdl = _app_ctx.PRICE_MODELS.get("hybppl_dd")
    if mdl is None:
        return [("(model not loaded)", "\u2014")]
    T_cal = 2 * 3.14159265358979 / mdl._W_cal
    return [
        ("A (intercept, log\u2081\u2080 USD)",          f"{mdl._A:.4f}"),
        ("B (slope)",                                   f"{mdl._B:.4f}"),
        ("C\u2081 (damped amplitude, log\u2081\u2080)", f"{mdl._C1:.4f}"),
        ("\u03c9_log (log-time freq, rad)",             f"{mdl._W_log:.4f}"),
        ("\u03c6\u2081 (phase, rad)",                    f"{mdl._PHI1:.4f}"),
        ("D\u2081 (log damping exponent)",               f"{mdl._D1:.4f}"),
        ("C\u2082 (cal amplitude, log\u2081\u2080)",    f"{mdl._C2:.4f}"),
        ("\u03c9_cal (calendar freq, rad/yr)",           f"{mdl._W_cal:.4f}"),
        ("T_cal (calendar period)",                     f"{T_cal:.2f} yr"),
        ("\u03c6\u2082 (phase, rad)",                    f"{mdl._PHI2:.4f}"),
        ("D\u2082 (cal damping exponent)",               f"{mdl._D2:.6f}"),
    ]


def _bm_rows():
    """Live coefficient table for Bubble Model — support + band parameters."""
    m = _app_ctx.M
    if m is None:
        return [("(model not loaded)", "\u2014")]
    return [
        ("A_sup (support intercept, log\u2081\u2080 USD)", f"{m.support_intercept:.4f}"),
        ("B_sup (support slope)",                         f"{m.support_slope:.4f}"),
        ("\u03c3\u2080 up (base vol, upper bands)",
            f"{getattr(m, 'bm_sigma0_up', 0.085):.4f}"),
        ("\u03b1 up (vol shrinkage, upper)",
            f"{getattr(m, 'bm_alpha_up', 0.132):.4f}"),
        ("\u03c3\u2080 down (base vol, lower bands)",
            f"{getattr(m, 'bm_sigma0_down', 0.075):.4f}"),
        ("\u03b1 down (vol shrinkage, lower)",
            f"{getattr(m, 'bm_alpha_down', 0.218):.4f}"),
        ("R\u00b2 (composite on full history)",          f"{float(m.bm_r2):.4f}"),
        ("N future bubbles (max)",                       f"{int(m.n_future_max)}"),
    ]


def _ef_rows():
    """Live coefficient table for Empirical Floor Model."""
    ef = _app_ctx.PRICE_MODELS.get("ef")
    if ef is None:
        return [("(EF model not loaded \u2014 model_data_ef.pkl missing)", "\u2014")]
    return [
        ("Support intercept (log\u2081\u2080 USD)",  f"{ef._intercept:.4f}"),
        ("Support slope",                             f"{ef._slope:.4f}"),
        ("\u03c3\u2080 up (base vol, upper)",        f"{ef._sigma0_up:.4f}"),
        ("\u03b1 up (vol shrinkage, upper)",         f"{ef._alpha_up:.4f}"),
        ("\u03c3\u2080 down (base vol, lower)",      f"{ef._sigma0_down:.4f}"),
        ("\u03b1 down (vol shrinkage, lower)",       f"{ef._alpha_down:.4f}"),
        ("R\u00b2 (composite)",                       f"{float(ef._bm_r2):.4f}"),
        ("N future bubbles (max)",                   f"{int(ef._n_future_max)}"),
    ]


def _hyb2l_coeff_table():
    """Live coefficient table for HybPPL +2nd Log."""
    m = _app_ctx.PRICE_MODELS.get("hyb2l")
    if m is None:
        return _coeff_table([("(Hyb2L model not loaded)", "\u2014")])
    return _coeff_table([
        ("A (intercept)", f"{m._A:.6f}"),
        ("B (slope)", f"{m._B:.6f}"),
        ("C\u2081 (log osc 1 amp)", f"{m._C1:.6f}"),
        ("\u03c9\u2081 (log freq 1)", f"{m._W1:.4f}"),
        ("D\u2081 (damping 1)", f"{m._D1:.6f}"),
        ("C\u2082 (cal osc amp)", f"{m._C2:.6f}"),
        ("\u03c9_c (cal freq)", f"{m._Wc:.4f}  (T={2*3.14159/m._Wc:.2f}yr)"),
        ("C\u2083 (log osc 2 amp)", f"{m._C3:.6f}"),
        ("\u03c9\u2082 (log freq 2)", f"{m._W2:.4f}"),
        ("D\u2082 (damping 2)", f"{m._D2:.6f}"),
        ("\u03c3 (residual std)", f"{m._sigma:.6f}"),
    ])


def _hyb2c_coeff_table():
    """Live coefficient table for HybPPL +2nd Cal."""
    m = _app_ctx.PRICE_MODELS.get("hyb2c")
    if m is None:
        return _coeff_table([("(Hyb2C model not loaded)", "\u2014")])
    return _coeff_table([
        ("A (intercept)", f"{m._A:.6f}"),
        ("B (slope)", f"{m._B:.6f}"),
        ("C\u2081 (log osc amp)", f"{m._C1:.6f}"),
        ("\u03c9\u2081 (log freq)", f"{m._W1:.4f}"),
        ("D (damping)", f"{m._D:.6f}"),
        ("C\u2082 (cal osc 1 amp)", f"{m._C2:.6f}"),
        ("\u03c9_c\u2081 (cal freq 1)", f"{m._Wc1:.4f}  (T={2*3.14159/m._Wc1:.2f}yr)"),
        ("C\u2083 (cal osc 2 amp)", f"{m._C3:.6f}"),
        ("\u03c9_c\u2082 (cal freq 2)", f"{m._Wc2:.4f}  (T={2*3.14159/m._Wc2:.2f}yr)"),
        ("\u03c3 (residual std)", f"{m._sigma:.6f}"),
    ])


def _hyb2b_coeff_table():
    """Live coefficient table for HybPPL +Both."""
    m = _app_ctx.PRICE_MODELS.get("hyb2b")
    if m is None:
        return _coeff_table([("(Hyb2B model not loaded)", "\u2014")])
    return _coeff_table([
        ("A (intercept)", f"{m._A:.6f}"),
        ("B (slope)", f"{m._B:.6f}"),
        ("C\u2081 (log osc 1 amp)", f"{m._C1:.6f}"),
        ("\u03c9_l\u2081 (log freq 1)", f"{m._W1:.4f}"),
        ("D\u2081 (damping 1)", f"{m._D1:.6f}"),
        ("C\u2082 (cal osc 1 amp)", f"{m._C2:.6f}"),
        ("\u03c9_c\u2081 (cal freq 1)", f"{m._Wc1:.4f}  (T={2*3.14159/m._Wc1:.2f}yr)"),
        ("C\u2083 (log osc 2 amp)", f"{m._C3:.6f}"),
        ("\u03c9_l\u2082 (log freq 2)", f"{m._W2:.4f}"),
        ("D\u2082 (damping 2)", f"{m._D2:.6f}"),
        ("C\u2084 (cal osc 2 amp)", f"{m._C4:.6f}"),
        ("\u03c9_c\u2082 (cal freq 2)", f"{m._Wc2:.4f}  (T={2*3.14159/m._Wc2:.2f}yr)"),
        ("\u03c3 (residual std)", f"{m._sigma:.6f}"),
    ])


def _pca_coeff_table():
    """Live coefficient table for PCA Model."""
    m = _app_ctx.PRICE_MODELS.get("pca")
    if m is None:
        return _coeff_table([("(PCA model not loaded)", "\u2014")])
    rows = [
        ("k (PCs used)", str(m._N_PCS)),
        ("Basis components", str(len(m._basis_info))),
        ("Source models", ", ".join(sorted(m._source_models.keys()))),
        ("\u03c3 (residual std)", f"{m._sigma:.6f}"),
    ]
    if hasattr(m, '_explained') and len(m._explained) > 0:
        for i, ev in enumerate(m._explained):
            rows.append((f"PC{i+1} explained var", f"{ev:.4%}"))
    return _coeff_table(rows)


def _pca_formula_table():
    """Live formula with numerical coefficients for registered PCA model."""
    m = _app_ctx.PRICE_MODELS.get("pca")
    if m is None:
        return _coeff_table([("(PCA model not loaded)", "\u2014")])
    rows = [
        ("\u03b2\u2080 (intercept)", f"{m._intercept:.6f}"),
    ]
    labels = ["power law trend", "halving cycle", "log-periodic",
              "2nd log harmonic", "residual osc", "fine structure"]
    for i in range(min(m._N_PCS, len(m._beta) - 1)):
        lbl = labels[i] if i < len(labels) else f"PC{i+1}"
        rows.append((
            f"\u03b2{i+1} \u00d7 {lbl}",
            f"{m._beta[i+1]:+.6f}  (var explained: {m._explained[i]:.4%})",
        ))
    rows.append(("\u03c3 (residual std)", f"{m._sigma:.6f}"))
    return _coeff_table(rows)


def _pca_variance_table():
    """Explained variance per PC with cumulative."""
    m = _app_ctx.PRICE_MODELS.get("pca")
    if m is None:
        return _coeff_table([("(PCA model not loaded)", "\u2014")])
    labels = ["power law trend", "halving cycle", "log-periodic",
              "2nd log harmonic", "residual osc", "fine structure"]
    cumvar = 0.0
    rows = []
    for i, ev in enumerate(m._explained):
        cumvar += ev
        lbl = labels[i] if i < len(labels) else f"PC{i+1}"
        rows.append((
            f"PC{i+1}: {lbl}",
            f"{ev:.4%}  (cumulative: {cumvar:.4%})",
        ))
    return _coeff_table(rows)


def _pca_expanded_formula():
    """Full expanded formula with all numerical coefficients for replication."""
    import numpy as _np
    m = _app_ctx.PRICE_MODELS.get("pca")
    if m is None or not m._basis_info:
        return html.P("(PCA model not loaded)", className="text-muted")

    # Compute effective constant and slope
    const_total = m._intercept
    slope_total = 0.0
    osc_rows = []

    for i, ((key, cname), w) in enumerate(zip(m._basis_info, m._weights)):
        mdl = m._source_models.get(key)
        if mdl is None:
            continue
        cl = cname.lower()

        if "constant" in cl or cname.startswith("A "):
            const_total += w * getattr(mdl, "_A", 0)
        elif "log" in cl and "t" in cl and "osc" not in cl and "cos" not in cl:
            slope_total += w * getattr(mdl, "_B", 0)
        else:
            # Oscillatory term — extract params
            if "log osc 2" in cl or "\u03c9\u2082" in cname or "\u03c9_l\u2082" in cname:
                C = getattr(mdl, "_C3", getattr(mdl, "_C", 0))
                D = getattr(mdl, "_D2", getattr(mdl, "_D", 0))
                W = getattr(mdl, "_W2", getattr(mdl, "_W", 0))
                PHI = getattr(mdl, "_PHI3", getattr(mdl, "_PHI", 0))
                kind = "log"
            elif "log osc" in cl or "\u03c9_log" in cname or "\u03c9\u2081" in cname:
                C = getattr(mdl, "_C1", getattr(mdl, "_C", 0))
                D = getattr(mdl, "_D1", getattr(mdl, "_D", 0))
                W = getattr(mdl, "_W1", getattr(mdl, "_W", 0))
                PHI = getattr(mdl, "_PHI1", getattr(mdl, "_PHI", 0))
                kind = "log"
            elif "cal osc 2" in cl or "\u03c9_c\u2082" in cname:
                C = getattr(mdl, "_C4", getattr(mdl, "_C3", 0))
                W = getattr(mdl, "_Wc2", 0)
                PHI = getattr(mdl, "_PHI4", getattr(mdl, "_PHI3", 0))
                Dc = getattr(mdl, "_Dc2", None)
                D = Dc if Dc is not None and Dc > 0.01 else 0
                kind = "cal"
            elif "cal osc" in cl or "\u03c9_cal" in cname or "\u03c9_c\u2081" in cname:
                C = getattr(mdl, "_C2", 0)
                W = getattr(mdl, "_Wc1", getattr(mdl, "_Wc", getattr(mdl, "_W2", 0)))
                PHI = getattr(mdl, "_PHI2", 0)
                Dc = getattr(mdl, "_Dc1", None)
                D = Dc if Dc is not None and Dc > 0.01 else 0
                kind = "cal"
            else:
                continue

            eff_amp = abs(w * C)
            if eff_amp < 0.001:
                continue

            if kind == "log":
                formula = f"{C:.4f}\u00b7t^(\u2212{D:.4f})\u00b7cos({W:.4f}\u00b7ln(t){PHI:+.4f})"
            elif D > 0:
                T = 2 * _np.pi / W if W > 0 else 0
                formula = f"{C:.4f}\u00b7t^(\u2212{D:.4f})\u00b7cos({W:.4f}\u00b7t{PHI:+.4f})  [T={T:.1f}yr]"
            else:
                T = 2 * _np.pi / W if W > 0 else 0
                formula = f"{C:.4f}\u00b7cos({W:.4f}\u00b7t{PHI:+.4f})  [T={T:.1f}yr]"

            osc_rows.append((f"w={w:+.6f}", formula, f"{eff_amp:.4f}"))

    rows = [
        ("\u03b1 (constant)", f"{const_total:.6f}", ""),
        ("\u03b2\u00b7log\u2081\u2080(t)", f"\u03b2 = {slope_total:.6f}", ""),
    ]
    header = html.Thead(html.Tr([
        html.Th("Term", style={"paddingRight": "12px"}),
        html.Th("Formula / Value", style={"paddingRight": "12px"}),
        html.Th("Eff. amp"),
    ]))
    body_rows = []
    for label, val, amp in rows:
        body_rows.append(html.Tr([
            html.Td(html.Strong(label)),
            html.Td(html.Code(val)),
            html.Td(amp),
        ]))
    body_rows.append(html.Tr([
        html.Td(html.Strong(f"Oscillatory terms ({len(osc_rows)})"),
                colSpan=3,
                style={"paddingTop": "8px", "borderTop": f"1px solid {TABLE_BORDER_LIGHT}"}),
    ]))
    for w_str, formula, amp in osc_rows:
        body_rows.append(html.Tr([
            html.Td(html.Code(w_str), style={"fontSize": UI_FONT_MD}),
            html.Td(html.Code(formula), style={"fontSize": UI_FONT_MD}),
            html.Td(html.Code(amp), style={"fontSize": UI_FONT_MD}),
        ]))

    return html.Table(
        [header, html.Tbody(body_rows)],
        style={"fontSize": UI_FONT_BASE, "marginBottom": "12px"},
    )


def _pca_basis_listing():
    """Live listing of all basis components used by the PCA model."""
    m = _app_ctx.PRICE_MODELS.get("pca")
    if m is None or not m._basis_info:
        return html.P("(PCA model not loaded)", className="text-muted")
    # Group by source model
    by_model = {}
    for key, cname in m._basis_info:
        by_model.setdefault(key, []).append(cname)
    items = []
    for key in m._SOURCE_KEYS:
        if key not in by_model:
            continue
        mdl = m._source_models.get(key)
        label = mdl.name if mdl else key
        comps = by_model[key]
        items.append(html.Li([
            html.Strong(f"{label}"), f" ({len(comps)} components): ",
            ", ".join(comps),
        ]))
    return html.Ul(items, style={"fontSize": UI_FONT_BASE})


def _hyb4d_coeff_table():
    """Live coefficient table for HybPPL 4D (all damped)."""
    m = _app_ctx.PRICE_MODELS.get("hyb4d")
    if m is None:
        return _coeff_table([("(Hyb4D model not loaded)", "\u2014")])
    return _coeff_table([
        ("A (intercept)", f"{m._A:.6f}"),
        ("B (slope)", f"{m._B:.6f}"),
        ("C\u2081 (log osc 1 amp)", f"{m._C1:.6f}"),
        ("\u03c9_l\u2081 (log freq 1)", f"{m._W1:.4f}"),
        ("D\u2081 (log damping 1)", f"{m._D1:.6f}"),
        ("C\u2082 (cal osc 1 amp)", f"{m._C2:.6f}"),
        ("\u03c9_c\u2081 (cal freq 1)", f"{m._Wc1:.4f}  (T={2*3.14159/m._Wc1:.2f}yr)"),
        ("D_c\u2081 (cal damping 1)", f"{m._Dc1:.6f}"),
        ("C\u2083 (log osc 2 amp)", f"{m._C3:.6f}"),
        ("\u03c9_l\u2082 (log freq 2)", f"{m._W2:.4f}"),
        ("D\u2082 (log damping 2)", f"{m._D2:.6f}"),
        ("C\u2084 (cal osc 2 amp)", f"{m._C4:.6f}"),
        ("\u03c9_c\u2082 (cal freq 2)", f"{m._Wc2:.4f}  (T={2*3.14159/m._Wc2:.2f}yr)"),
        ("D_c\u2082 (cal damping 2)", f"{m._Dc2:.6f}"),
        ("\u03c3 (residual std)", f"{m._sigma:.6f}"),
    ])


def _eppl_coeff_table():
    """Live coefficient table for Entropy PPL model."""
    m = _app_ctx.PRICE_MODELS.get("eppl")
    if m is None:
        return _coeff_table([("(Entropy PPL model not loaded)", "\u2014")])
    return _coeff_table([
        ("A (intercept)", f"{m._A:.6f}"),
        ("B (slope)", f"{m._B:.6f}"),
        ("C\u2081 (log osc 1 amplitude)", f"{m._C1:.6f}"),
        ("\u03c9\u2081 (log osc 1 freq)", f"{m._W1:.6f}"),
        ("\u03c6\u2081 (log osc 1 phase)", f"{m._P1:.6f}"),
        ("w\u2081 (log osc 1 entropy rate)", f"{m._w1:.6f}"),
        ("C\u2083 (log osc 2 amplitude)", f"{m._C3:.6f}"),
        ("\u03c9\u2082 (log osc 2 freq)", f"{m._W2:.6f}"),
        ("\u03c6\u2083 (log osc 2 phase)", f"{m._P3:.6f}"),
        ("w\u2082 (log osc 2 entropy rate)", f"{m._w2:.6f}"),
        ("C\u2082 (cal osc 1 amplitude)", f"{m._C2:.6f}"),
        ("\u03c9_c\u2081 (cal osc 1 freq)", f"{m._Wc1:.6f}"),
        ("\u03c6\u2082 (cal osc 1 phase)", f"{m._P2:.6f}"),
        ("C\u2084 (cal osc 2 amplitude)", f"{m._C4:.6f}"),
        ("\u03c9_c\u2082 (cal osc 2 freq)", f"{m._Wc2:.6f}"),
        ("\u03c6\u2084 (cal osc 2 phase)", f"{m._P4:.6f}"),
        ("\u03c3 (residual std)", f"{m._sigma:.6f}"),
        ("R\u00b2", "0.993320"),
    ])


def _gompertz_coeff_table():
    """Live coefficient table for Gompertz Model."""
    m = _app_ctx.PRICE_MODELS.get("gomp")
    if m is None:
        return _coeff_table([("(Gompertz model not loaded)", "\u2014")])
    max_price = 10.0 ** m._K
    return _coeff_table([
        ("K (carrying capacity, log\u2081\u2080 USD)", f"{m._K:.4f}  (${max_price:,.0f})"),
        ("r (growth rate)", f"{m._r:.6f}"),
        ("t\u2080 (inflection, years since genesis)", f"{m._t0:.4f}"),
        ("\u03c3 (residual std)", f"{m._sigma:.4f}"),
    ])


def _plo_coeff_table():
    """Live coefficient table for Offset Power Law."""
    m = _app_ctx.PRICE_MODELS.get("plo")
    if m is None:
        return _coeff_table([("(Offset Power Law not loaded)", "\u2014")])
    return _coeff_table([
        ("A (log\u2081\u2080 intercept)", f"{m._A:.6f}"),
        ("m (slope)", f"{m._m:.6f}"),
        ("c (time-origin offset, yr)", f"{m._c:.6f}"),
        ("\u03c3 (residual std)", f"{m._sigma:.4f}"),
    ])


def _sexp_coeff_table():
    """Live coefficient table for Stretched Exponential."""
    m = _app_ctx.PRICE_MODELS.get("sexp")
    if m is None:
        return _coeff_table([("(Stretched Exponential not loaded)", "\u2014")])
    return _coeff_table([
        ("A (log\u2081\u2080 intercept at t=0)", f"{m._A:.6f}"),
        ("B (scale)", f"{m._B:.6f}"),
        ("\u03b2 (stretching exponent)", f"{m._beta:.6f}"),
        ("\u03c3 (residual std)", f"{m._sigma:.4f}"),
    ])


def _grdy_coeff_table():
    """Live coefficient table for Greedy Select — reads GreedyModel._BASIS
    so the table tracks any refit without further edits."""
    m = _app_ctx.PRICE_MODELS.get("grdy")
    if m is None:
        return _coeff_table([("(Greedy model not loaded)", "\u2014")])
    rows = [
        ("\u03b1 (intercept)", f"{m._alpha:.6f}"),
        ("\u03b2 (slope)", f"{m._beta:.6f}"),
        ("\u03c3 (residual std)", f"{m._sigma:.4f}"),
    ]
    return _coeff_table(rows)


def _grdy_basis_table():
    """Live table listing the 5 selected basis terms with their full params.

    Columns: index, space, damping, freq ω, phase, weight w, damping param.
    """
    m = _app_ctx.PRICE_MODELS.get("grdy")
    if m is None or not hasattr(m, "_BASIS"):
        return html.Div("Greedy model not loaded.", style={"color": FALLBACK_MODEL_GRAY})
    header_cells = [
        html.Th("", style={"paddingRight": "10px"}),
        html.Th("Space", style={"paddingRight": "10px"}),
        html.Th("Damping", style={"paddingRight": "10px"}),
        html.Th("\u03c9 (freq)", style={"paddingRight": "10px"}),
        html.Th("Phase", style={"paddingRight": "10px"}),
        html.Th("w (weight)", style={"paddingRight": "10px"}),
        html.Th("Damping param"),
    ]
    body_rows = []
    for i, term in enumerate(m._BASIS, 1):
        space, damping, freq, phase, weight, d_param = term
        period_note = ""
        if space == "cal":
            # Cal osc period in years
            import math
            period_note = f"  (T\u2248{2*math.pi/freq:.2f}yr)"
        dp_str = "\u2014" if d_param is None else f"{d_param:.4f}"
        dp_label = ""
        if damping == "hybrid":
            dp_label = f"D = {dp_str}"
        elif damping == "entropy":
            dp_label = f"w_e = {dp_str}"
        body_rows.append(html.Tr([
            html.Td(f"f{i}"),
            html.Td(space),
            html.Td(damping),
            html.Td(f"{freq:.4f}{period_note}"),
            html.Td(phase),
            html.Td(html.Code(f"{weight:+.4f}")),
            html.Td(dp_label),
        ]))
    return html.Table(
        [html.Thead(html.Tr(header_cells)), html.Tbody(body_rows)],
        style={"marginBottom": "12px", "fontSize": UI_FONT_BASE,
               "fontFamily": "ui-monospace, SFMono-Regular, monospace"},
    )


def _logi_coeff_table():
    """Live coefficient table for Logistic (true S-curve)."""
    m = _app_ctx.PRICE_MODELS.get("logi")
    if m is None:
        return _coeff_table([("(Logistic model not loaded)", "\u2014")])
    max_price = 10.0 ** m._K
    return _coeff_table([
        ("K (saturation, log\u2081\u2080 USD)", f"{m._K:.4f}  (${max_price:,.0f})"),
        ("r (growth rate)", f"{m._r:.6f}"),
        ("t\u2080 (inflection, years since genesis)", f"{m._t0:.4f}"),
        ("\u03c3 (residual std)", f"{m._sigma:.4f}"),
    ])


def _spl_table(headers, rows, first_col_width=None, caption=None):
    """Small bordered table for the Saturating Power Law card.

    Static numbers only — every one of them is a property of a *past* data
    window, so they cannot be read off the live model. Regenerate with
        btc_venv/bin/python3 tools/analyze_spl.py
    and update the constants below. Kept as Dash components rather than
    Markdown because dollar amounts inside a mathjax=True dcc.Markdown are
    parsed as inline-math delimiters.
    """
    hdr = {"paddingRight": "12px", "paddingBottom": "6px", "textAlign": "left",
           "borderBottom": f"1px solid {TABLE_BORDER_DARK}", "fontSize": UI_FONT_BASE}
    cell = {"paddingRight": "12px", "paddingBottom": "4px", "paddingTop": "4px",
            "fontSize": UI_FONT_BASE,
            "borderBottom": f"1px solid {TABLE_BORDER_MID}"}
    body = []
    for row in rows:
        cells = []
        for j, v in enumerate(row):
            strong = isinstance(v, str) and v.startswith("**") and v.endswith("**")
            txt = v[2:-2] if strong else v
            style = dict(cell)
            if j == 0 and first_col_width:
                style["width"] = first_col_width
            cells.append(html.Td(html.Strong(txt) if strong or j == 0 else txt,
                                 style=style))
        body.append(html.Tr(cells))
    table = html.Table(
        [html.Thead(html.Tr([html.Th(h, style=hdr) for h in headers])),
         html.Tbody(body)],
        style={"marginBottom": "4px" if caption else "16px",
               "width": "100%", "borderCollapse": "collapse"},
    )
    if caption is None:
        return table
    return html.Div([
        table,
        html.Div(caption, style={"fontSize": UI_FONT_SM, "color": FALLBACK_MODEL_GRAY,
                                 "marginBottom": "16px"}),
    ])


# ── Saturating Power Law: the pinned analysis figures ──────────────────────
# EVERY number on this card is pinned, including _spl_coeff_table() below.
# The card compares two named snapshots — 2026-06-03 and 2026-08-06 — and a
# figure that quietly recomputed would turn "the later window" into "whenever
# you loaded the page", dissolving the comparison the whole card rests on.
# Refreshed by hand, never automatically.
#
# TO REFRESH: run `btc_venv/bin/python3 tools/analyze_spl.py`, take sections
# [0], [0b] and [4], and update _SPL_WINDOW_LABEL, the date labels and the
# prose dates in _items.py IN THE SAME EDIT — a refreshed number under a stale
# date is worse than no refresh. RMSE stays at 4 decimals: six implies a
# discrimination between these fits that is not there.
_SPL_WINDOW_LABEL = "6 Aug 2026"   # the later of the two pinned windows


def _spl_two_window_table():
    """Section [0] — the same fit, nine weeks apart."""
    return _spl_table(
        ["", "data through 3 Jun 2026", f"data through {_SPL_WINDOW_LABEL}"],
        [
            ["n", "5,792", "5,856"],
            ["last price", "$64,813", "$64,294"],
            ["ceiling L (market cap)", "**$34.3T**", "**$14.8T**"],
            ["t₀ (yr)", "28.31", "23.87"],
            ["β", "5.0910", "5.1040"],
            ["RMSE", "0.2945", "0.2939"],
            ["variance removed vs PL", "0.0394%", "0.2329%"],
            ["likelihood-ratio statistic", "2.2802", "13.6536"],
            ["vs boundary 5% critical value 2.7055",
             "does not reject", "**REJECTS**"],
            ["p", "0.066", "0.0001"],
        ],
        first_col_width="38%",
        caption=f"Both columns are complete refits, pinned. Later window: "
                f"data through {_SPL_WINDOW_LABEL}.",
    )


def _spl_by_cutoff_table():
    """Section [0b] — the instability is systematic, not a one-off."""
    return _spl_table(
        ["data through", "last price", "ceiling L (market cap)",
         "t₀ (yr)", "LRT", "verdict at 5%"],
        [
            ["2024-08-06", "$56,308", "$10.7T", "22.4", "6.01", "REJECTS"],
            ["2025-02-06", "$97,846", "$44.1T", "29.8", "0.51", "does not reject"],
            ["2025-08-06", "$114,663", "$1,000T — pinned at the fitting cap",
             "55.1", "−0.11", "does not reject"],
            ["2026-02-06", "$67,196", "$1,000T — pinned at the fitting cap",
             "55.0", "−0.14", "does not reject"],
            ["2026-06-03", "$64,813", "$34.3T", "28.3", "2.28", "does not reject"],
            ["2026-08-06", "$64,294", "$14.8T", "23.9", "13.65", "REJECTS"],
        ],
        caption=f"Each row is a complete refit on data through that date. "
                f"Pinned; last window {_SPL_WINDOW_LABEL}.",
    )


def _spl_cycle_phase_table():
    """Where each window ended relative to the power-law trend, against the
    ceiling that window's refit returned.

    The residual column is the mean log10 residual about a power law fitted
    to that same window, over the window's final year (regenerate alongside
    section [0b]).
    """
    return _spl_table(
        ["data through", "window ended (mean log₁₀ residual, final year)",
         "ceiling that window returned"],
        [
            ["2024-08-06", "−0.055 — below trend", "$10.7T"],
            ["2025-02-06", "+0.044 — above trend", "$44.1T"],
            ["2025-08-06", "+0.060 — above trend", "no ceiling (pinned at the cap)"],
            ["2026-02-06", "+0.046 — above trend", "no ceiling (pinned at the cap)"],
            ["2026-06-03", "−0.022 — below trend", "$34.3T"],
            ["2026-08-06", "−0.082 — below trend", "$14.8T"],
        ],
        first_col_width="22%",
        caption=f"Residual is the mean log₁₀ residual about a power law "
                f"fitted to that same window, over the window’s final year. "
                f"Pinned; last window {_SPL_WINDOW_LABEL}.",
    )


def _spl_profile_table():
    """Section [4] — SSE profile with t0 held fixed. Steep below, flat above."""
    return _spl_table(
        ["t₀ held fixed (yr)", "RMSE", "implied ceiling (market cap)",
         "ΔSSE vs best"],
        [
            ["20", "0.2944", "$7T", "+1.74"],
            ["25 (best on this grid)", "0.2939", "$18T", "0.00"],
            ["28.4", "0.2940", "$34T", "+0.33"],
            ["35", "0.2941", "$96T", "+0.80"],
            ["50", "0.2942", "$578T", "+1.07"],
            ["100", "0.2942", "$19,244T", "+1.13"],
            ["1000 (effectively no ceiling)", "0.2942", "—", "+1.13"],
        ],
        first_col_width="26%",
        caption=f"Profile computed on data through {_SPL_WINDOW_LABEL}, "
                "pinned. The shape — steep below, flat above — is what the "
                "argument rests on; the row values move with the window.",
    )


# Range of the fitted ceiling across the six data windows in
# tools/analyze_spl.py section [0b] (2024-08-06 .. current). Static: these are
# historical refits, not a live quantity. Regenerate with
#     btc_venv/bin/python3 tools/analyze_spl.py
# and update here if the span moves.
_SPL_L_RANGE_LO_T = 10.7     # $T market cap, window ending 2024-08-06
_SPL_L_RANGE_HI_T = 1000.0   # $T, two windows pinned at the fitting bound
_SPL_L_RANGE_FACTOR = 93     # hi / lo


def _spl_coeff_table():
    """Coefficients of the fit on the pinned 2026-08-06 window.

    DELIBERATELY NOT LIVE — unlike every other _*_coeff_table() here, which
    reads _app_ctx.PRICE_MODELS so a refit shows up on the next page load.
    This card's argument is a comparison of two named snapshots; if the
    "2026-08-06" column silently became "whenever you loaded the page", the
    two dates would stop being two dates and the comparison would dissolve.
    A card arguing that these numbers move has to hold still itself. The
    plotted SatPL curve does follow the live fit, which the card says out
    loud near the top. Do not "fix" this back to a live lookup.

    The ceiling is never shown alone: the row for this window is followed
    immediately by the range the same fit returns on other windows.
    """
    return _coeff_table([
        ("data window (fixed)", _SPL_WINDOW_LABEL),
        ("β (early-time power-law exponent)", "5.1040"),
        ("t₀ (roll-over, yr since 2009-07-25)", "23.8691"),
        ("L (ceiling, log₁₀ USD/BTC)", "5.8468"),
        (f"L on the {_SPL_WINDOW_LABEL} window",
         "$702,828 / BTC  ·  $14.8T market cap"),
        ("L on other windows (same fit, same code)",
         f"${_SPL_L_RANGE_LO_T:.1f}T … ${_SPL_L_RANGE_HI_T:,.0f}T "
         f"(fitting cap) — a {_SPL_L_RANGE_FACTOR}× range"),
        ("σ (residual std = fit RMSE)", "0.2939"),
    ])


def _pl_coeff_table():
    """Live OLS coefficients for the Power Law model. PowerLawModel stores
    bands as {q: intercept_shifted, slope}; sigma is backed out from any
    non-median quantile via intercept[q] = intercept[0.5] + z(q)·σ."""
    m = _app_ctx.PRICE_MODELS.get("pl")
    if m is None:
        return _coeff_table([("(PL not loaded)", "\u2014")])
    fit = m.fits.get(0.5) or next(iter(m.fits.values()))
    intercept = fit.get("intercept")
    slope = fit.get("slope")
    sigma = getattr(m, "_sigma", None)
    if sigma is None and intercept is not None:
        from scipy.stats import norm
        for q_test in (0.9, 0.95, 0.99, 0.1, 0.05):
            if q_test in m.fits:
                z = norm.ppf(q_test)
                if z != 0:
                    sigma = abs(
                        (m.fits[q_test]["intercept"] - intercept) / z)
                    break
    r2 = getattr(m, "r2_per_quantile", {}).get(0.5)
    rows = [
        ("\u03b1 (intercept)", f"{intercept:.6f}" if intercept is not None else "\u2014"),
        ("\u03b2 (slope)", f"{slope:.6f}" if slope is not None else "\u2014"),
    ]
    if sigma is not None:
        rows.append(("\u03c3 (residual std)", f"{sigma:.4f}"))
    if r2 is not None:
        rows.append(("R\u00b2 (log-space, median)", f"{r2:.4f}"))
    return _coeff_table(rows)


def _lppl_coeff_table():
    """Live fitted coefficients for the single-frequency LPPL model."""
    m = _app_ctx.PRICE_MODELS.get("lppl")
    if m is None:
        return _coeff_table([("(LPPL not loaded)", "\u2014")])
    r2 = getattr(m, "r2_per_quantile", {}).get(0.5)
    rows = [
        ("A (intercept, log\u2081\u2080 USD)", f"{m._A:.6f}"),
        ("B (slope)", f"{m._B:.6f}"),
        ("C (osc. amplitude, log\u2081\u2080)", f"{m._C:.6f}"),
        ("\u03c9 (log-time freq, rad)", f"{m._W:.6f}"),
        ("\u03c6 (phase, rad)", f"{m._PHI:.6f}"),
        ("D (damping exponent)", f"{m._D:.6f}"),
        ("\u03c3 (residual, log\u2081\u2080)", f"{m._sigma:.4f}"),
    ]
    if r2 is not None:
        rows.append(("R\u00b2 (median)", f"{r2:.4f}"))
    return _coeff_table(rows)


def _lp2_coeff_table():
    """Live fitted coefficients for the two-frequency LPPL₂ model."""
    m = _app_ctx.PRICE_MODELS.get("lp2")
    if m is None:
        return _coeff_table([("(LPPL\u2082 not loaded)", "\u2014")])
    r2 = getattr(m, "r2_per_quantile", {}).get(0.5)
    rows = [
        ("A (intercept, log\u2081\u2080 USD)", f"{m._A:.6f}"),
        ("B (slope)", f"{m._B:.6f}"),
        ("C\u2081 (amp, primary)", f"{m._C:.6f}"),
        ("\u03c9\u2081 (freq, primary)", f"{m._W:.6f}"),
        ("\u03c6\u2081 (phase, primary)", f"{m._PHI:.6f}"),
        ("D (damping, primary only)", f"{m._D:.6f}"),
        ("C\u2082 (amp, secondary)", f"{m._C2:.6f}"),
        ("\u03c9\u2082 (freq, secondary)", f"{m._W2:.6f}"),
        ("\u03c6\u2082 (phase, secondary)", f"{m._PHI2:.6f}"),
        ("\u03c3 (residual, log\u2081\u2080)", f"{m._sigma:.4f}"),
    ]
    if r2 is not None:
        rows.append(("R\u00b2 (median)", f"{r2:.4f}"))
    return _coeff_table(rows)


def _linppl_coeff_table():
    """Live fitted coefficients for LinPPL (calendar-periodic LPPL)."""
    m = _app_ctx.PRICE_MODELS.get("linppl")
    if m is None:
        return _coeff_table([("(LinPPL not loaded)", "\u2014")])
    import math
    period_yr = 2 * math.pi / m._W if m._W else None
    r2 = getattr(m, "r2_per_quantile", {}).get(0.5)
    rows = [
        ("A (intercept, log\u2081\u2080 USD)", f"{m._A:.6f}"),
        ("B (slope)", f"{m._B:.6f}"),
        ("C (osc. amplitude)", f"{m._C:.6f}"),
        ("\u03c9_cal (freq, rad/yr)",
         f"{m._W:.6f}" + (f"  (T\u2248{period_yr:.2f}yr)" if period_yr else "")),
        ("\u03c6 (phase, rad)", f"{m._PHI:.6f}"),
        ("D (damping exponent)", f"{m._D:.6f}"),
        ("\u03c3 (residual, log\u2081\u2080)", f"{m._sigma:.4f}"),
    ]
    if r2 is not None:
        rows.append(("R\u00b2 (median)", f"{r2:.4f}"))
    return _coeff_table(rows)


def _hybppl_coeff_table():
    """Live fitted coefficients for HybPPL (log + calendar frequencies)."""
    m = _app_ctx.PRICE_MODELS.get("hybppl")
    if m is None:
        return _coeff_table([("(HybPPL not loaded)", "\u2014")])
    import math
    period_yr = 2 * math.pi / m._W2 if m._W2 else None
    r2 = getattr(m, "r2_per_quantile", {}).get(0.5)
    rows = [
        ("A (intercept, log\u2081\u2080 USD)", f"{m._A:.6f}"),
        ("B (slope)", f"{m._B:.6f}"),
        ("C\u2081 (log-osc amp)", f"{m._C:.6f}"),
        ("\u03c9\u2081 (log-osc freq)", f"{m._W:.6f}"),
        ("\u03c6\u2081 (log-osc phase)", f"{m._PHI:.6f}"),
        ("D (log-osc damping)", f"{m._D:.6f}"),
        ("C\u2082 (cal-osc amp)", f"{m._C2:.6f}"),
        ("\u03c9\u2082 (cal-osc freq, rad/yr)",
         f"{m._W2:.6f}" + (f"  (T\u2248{period_yr:.2f}yr)" if period_yr else "")),
        ("\u03c6\u2082 (cal-osc phase)", f"{m._PHI2:.6f}"),
        ("\u03c3 (residual, log\u2081\u2080)", f"{m._sigma:.4f}"),
    ]
    if r2 is not None:
        rows.append(("R\u00b2 (median)", f"{r2:.4f}"))
    return _coeff_table(rows)


def _exp_coeff_table():
    """Live fitted coefficients for the Exponential model."""
    m = _app_ctx.PRICE_MODELS.get("exp")
    if m is None:
        return _coeff_table([("(Exp not loaded)", "\u2014")])
    r2 = getattr(m, "r2_per_quantile", {}).get(0.5)
    rows = [
        ("a (intercept, log\u2081\u2080 USD)", f"{m._intercept:.6f}"),
        ("b (slope, per yr)", f"{m._slope:.6f}"),
        ("\u03c3 (residual std)", f"{m._sigma:.4f}"),
    ]
    if r2 is not None:
        rows.append(("R\u00b2 (median)", f"{r2:.4f}"))
    return _coeff_table(rows)


def _bpl_coeff_table():
    """Live coefficient table for Broken Power Law Model."""
    import pandas as pd
    m = _app_ctx.PRICE_MODELS.get("bpl")
    if m is None:
        return _coeff_table([("(BPL model not loaded)", "\u2014")])
    genesis = pd.Timestamp("2009-07-25")
    break_date = genesis + pd.Timedelta(days=m._t_break * 365.25)
    return _coeff_table([
        ("a\u2081 (early intercept)", f"{m._a1:.6f}"),
        ("b\u2081 (early slope)", f"{m._b1:.6f}"),
        ("t_break (breakpoint)", f"{m._t_break:.4f}  ({break_date.strftime('%Y-%m')})"),
        ("b\u2082 (late slope)", f"{m._b2:.6f}"),
        ("a\u2082 (late intercept, derived)", f"{m._a2:.6f}"),
        ("\u03c3 (residual std)", f"{m._sigma:.4f}"),
    ])


def _coeff_table(rows):
    """Small two-column coefficient table."""
    return html.Table([
        html.Tbody([
            html.Tr([
                html.Td(html.Strong(label), style={"paddingRight": "20px",
                         "paddingBottom": "4px", "whiteSpace": "nowrap"}),
                html.Td(html.Code(value) if not isinstance(value, str) or
                         any(c in value for c in "0123456789.\u2212") else value,
                         style={"paddingBottom": "4px"}),
            ]) for label, value in rows
        ])
    ], style={"marginBottom": "12px", "fontSize": UI_FONT_LG})


def _qr_table():
    """Quantile regression coefficient table from live model data."""
    m = _app_ctx.M
    if m is None:
        return html.P("Model data not loaded.", className="text-muted")
    # Show a representative subset
    show_qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    rows = []
    for q in show_qs:
        if q in m.qr_fits:
            f = m.qr_fits[q]
            rows.append(html.Tr([
                html.Td(f"Q{q*100:.0f}%", style={"paddingRight": "12px"}),
                html.Td(html.Code(f"{f['intercept']:.6f}"),
                         style={"paddingRight": "12px"}),
                html.Td(html.Code(f"{f['slope']:.6f}")),
            ]))
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Quantile", style={"paddingRight": "12px"}),
            html.Th("\u03b1 (intercept)", style={"paddingRight": "12px"}),
            html.Th("\u03b2 (slope)"),
        ])),
        html.Tbody(rows),
    ], style={"marginBottom": "12px", "fontSize": UI_FONT_LG})


def _comparison_table():
    """Model comparison summary table."""
    hdr_style = {"paddingRight": "12px", "paddingBottom": "6px",
                 "borderBottom": f"1px solid {TABLE_BORDER_DARK}", "fontSize": UI_FONT_BASE}
    cell_style = {"paddingRight": "12px", "paddingBottom": "4px",
                  "paddingTop": "4px", "fontSize": UI_FONT_BASE,
                  "borderBottom": f"1px solid {TABLE_BORDER_MID}"}
    return html.Table([
        html.Thead(html.Tr([
            html.Th("", style=hdr_style),
            html.Th("QR (Bubble)", style=hdr_style),
            html.Th("Power Law", style=hdr_style),
            html.Th("LPPL", style=hdr_style),
            html.Th("Exponential", style=hdr_style),
            html.Th("S2F", style=hdr_style),
            html.Th("Monte Carlo", style=hdr_style),
        ])),
        html.Tbody([
            html.Tr([
                html.Td(html.Strong("Type"), style=cell_style),
                html.Td("Quantile regression", style=cell_style),
                html.Td("OLS + Gaussian shift", style=cell_style),
                html.Td("Damped log-periodic + Gaussian", style=cell_style),
                html.Td("OLS (linear time) + Gaussian", style=cell_style),
                html.Td("Supply-driven regression", style=cell_style),
                html.Td("Stochastic simulation", style=cell_style),
            ]),
            html.Tr([
                html.Td(html.Strong("Bands"), style=cell_style),
                html.Td("Independent slopes", style=cell_style),
                html.Td("Parallel (same slope)", style=cell_style),
                html.Td("Parallel + oscillating", style=cell_style),
                html.Td("Parallel (very wide)", style=cell_style),
                html.Td("None (single line)", style=cell_style),
                html.Td("Fan (P1\u2013P95)", style=cell_style),
            ]),
            html.Tr([
                html.Td(html.Strong("Captures cycles"), style=cell_style),
                html.Td("No", style=cell_style),
                html.Td("No", style=cell_style),
                html.Td("Yes (damped)", style=cell_style),
                html.Td("No", style=cell_style),
                html.Td("Halvings only", style=cell_style),
                html.Td("Empirically", style=cell_style),
            ]),
            html.Tr([
                html.Td(html.Strong("Parameters"), style=cell_style),
                html.Td("2 per quantile", style=cell_style),
                html.Td("3 (\u03b1, \u03b2, \u03c3)", style=cell_style),
                html.Td("7 (A,B,C,\u03c9,\u03c6,D,\u03c3)", style=cell_style),
                html.Td("3 (\u03b1, \u03b2, \u03c3)", style=cell_style),
                html.Td("2 (\u03b1, \u03b2)", style=cell_style),
                html.Td("5\u00d75 matrix", style=cell_style),
            ]),
            html.Tr([
                html.Td(html.Strong("Dash style"), style=cell_style),
                html.Td("Solid", style=cell_style),
                html.Td("Dotted", style=cell_style),
                html.Td("Dash-dot", style=cell_style),
                html.Td("Long dash-dot", style=cell_style),
                html.Td("Long dash", style=cell_style),
                html.Td("Fan shading", style=cell_style),
            ]),
        ]),
    ], style={"marginBottom": "16px", "width": "100%", "borderCollapse": "collapse"})


def _regime_data_tables():
    """Build summary tables + transition matrices for all asset classes."""
    try:
        from data.asset_matrices import load_asset_matrices
        matrices = load_asset_matrices()
    except Exception as e:
        return html.P(f"Data not available: {e}", className="text-muted")

    sections = []
    _cell = {"fontSize": UI_FONT_MD, "padding": "2px 6px",
             "border": f"1px solid {TABLE_BORDER_LIGHT}", "textAlign": "right"}
    _hdr = {**_cell, "fontWeight": "bold", "backgroundColor": TABLE_HEADER_BG, "textAlign": "center"}

    for key in ("equity", "bond", "tres_short", "tres_med", "tres_long"):
        m = matrices.get(key)
        if not m:
            continue

        label = m.get("label", key)
        n_bins = len(m["bin_means"])

        # Summary stats
        sections.append(html.H6(f"{label}", style={"marginTop": "12px"}))
        sections.append(html.Table([
            html.Tbody([
                html.Tr([
                    html.Td("Observations", style={**_cell, "fontWeight": "bold"}),
                    html.Td(f"{m['n_obs']} months", style=_cell),
                    html.Td("Ann. Return", style={**_cell, "fontWeight": "bold"}),
                    html.Td(f"{m['ann_return']*100:.1f}%", style=_cell),
                    html.Td("Ann. Vol", style={**_cell, "fontWeight": "bold"}),
                    html.Td(f"{m['ann_vol']*100:.1f}%", style=_cell),
                ]),
            ]),
        ], style={"marginBottom": "4px", "borderCollapse": "collapse"}))

        # Regime bins: mean return + volatility per bin
        bin_header = [html.Th("Regime", style=_hdr)] + [
            html.Th(f"Bin {i+1}", style=_hdr) for i in range(n_bins)
        ]
        bin_means_row = [html.Td("Mean mo. return", style={**_cell, "fontWeight": "bold"})] + [
            html.Td(f"{m['bin_means'][i]*100:+.2f}%", style=_cell) for i in range(n_bins)
        ]
        bin_vols_row = [html.Td("Mo. volatility", style={**_cell, "fontWeight": "bold"})] + [
            html.Td(f"{m['bin_vols'][i]*100:.2f}%", style=_cell) for i in range(n_bins)
        ]
        bin_edges_row = [html.Td("Return range", style={**_cell, "fontWeight": "bold"})] + [
            html.Td(f"{m['bin_edges'][i]*100:+.1f} to {m['bin_edges'][i+1]*100:+.1f}%",
                     style={**_cell, "fontSize": UI_FONT_SM})
            for i in range(n_bins)
        ]

        sections.append(html.Table([
            html.Thead(html.Tr(bin_header)),
            html.Tbody([
                html.Tr(bin_means_row),
                html.Tr(bin_vols_row),
                html.Tr(bin_edges_row),
            ]),
        ], style={"marginBottom": "4px", "borderCollapse": "collapse", "width": "100%"}))

        # Transition matrix
        trans = m["trans"]
        t_header = [html.Th("From \u2193 To \u2192", style=_hdr)] + [
            html.Th(f"Bin {j+1}", style=_hdr) for j in range(n_bins)
        ]
        t_rows = []
        for i in range(n_bins):
            cells = [html.Td(f"Bin {i+1}", style={**_cell, "fontWeight": "bold"})]
            for j in range(n_bins):
                p = trans[i, j]
                # Color-code: darker for higher probability
                bg = _hex_alpha(USER_MODEL_TRACE, round(min(p * 1.5, 0.4), 2)) if p > 0.1 else "transparent"
                cells.append(html.Td(f"{p:.0%}", style={**_cell, "backgroundColor": bg}))
            t_rows.append(html.Tr(cells))

        sections.append(html.Details([
            html.Summary("Transition matrix", style={"fontSize": UI_FONT_BASE, "cursor": "pointer",
                                                      "color": FALLBACK_MODEL_GRAY, "marginBottom": "4px"}),
            html.Table([
                html.Thead(html.Tr(t_header)),
                html.Tbody(t_rows),
            ], style={"borderCollapse": "collapse", "width": "100%"}),
        ], style={"marginBottom": "12px"}))

    return html.Div(sections)
