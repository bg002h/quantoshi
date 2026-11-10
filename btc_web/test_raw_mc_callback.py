"""Raw MC, Task 5: Tab-1 Raw controls, their visibility, and the Raw branch of
``update_bubble``.

Layout and wiring are pinned structurally (every piece fails silently -- the
chart draws either way); the Raw branch is driven through ``update_bubble``
on the app's registered prices.
"""
import ast
import json
import pathlib
import shutil
import subprocess

import dash
import numpy as np
import pandas as pd
import pytest

from conftest import _app_ctx, _patch_ctx, update_bubble  # noqa: F401
import raw_mc
from raw_mc import (RAW_LINES_DEFAULT, RAW_SIMS_MAX, RAW_SRCS)
from raw_mc_testing import raw_prices  # noqa: F401

_APP = _app_ctx.app
_HERE = pathlib.Path(__file__).parent
_NODE = shutil.which("node")
needs_node = pytest.mark.skipif(_NODE is None, reason="node not installed")
YR = pd.Timestamp.today().year

RAW_INPUT_IDS = ["bub-mc-raw-sims", "bub-mc-raw-years", "bub-mc-raw-window",
                 "bub-mc-raw-trend", "bub-mc-raw-bins", "bub-mc-raw-regime",
                 "bub-mc-lines", "bub-mc-extremes"]
RAW_PARAMS = ["mc_raw_sims", "mc_raw_years", "mc_raw_window", "mc_raw_trend",
              "mc_raw_bins", "mc_raw_regime", "mc_lines", "mc_extremes"]
NEW_IDS = RAW_INPUT_IDS + ["bub-mc-raw-status", "bub-mc-panel", "bub-mc-lines-wrap"]


# ── layout helpers ──────────────────────────────────────────────────────────

def _walk(node, anc=()):
    """Yield (component, ancestors) depth-first, in document order."""
    if isinstance(node, (list, tuple)):
        for c in node:
            yield from _walk(c, anc)
        return
    if node is None or isinstance(node, (str, int, float)):
        return
    yield node, anc
    ch = getattr(node, "children", None)
    if ch is not None:
        yield from _walk(ch, anc + (node,))


def _by_id(root, cid):
    hits = [(n, a) for n, a in _walk(root) if getattr(n, "id", None) == cid]
    assert len(hits) == 1, (cid, len(hits))
    return hits[0]


def _classes(n):
    return (getattr(n, "className", None) or "").split()


def _has_cls_ancestor(anc, node, cls):
    return any(cls in _classes(a) for a in (*anc, node))


def _texts(root):
    out = []

    def rec(n):
        if isinstance(n, str):
            out.append(n)
        elif isinstance(n, (list, tuple)):
            for c in n:
                rec(c)
        elif n is not None and not isinstance(n, (int, float)):
            rec(getattr(n, "children", None))
            for o in (getattr(n, "options", None) or []):
                if isinstance(o, dict):
                    rec(o.get("label"))
    rec(root)
    return " ".join(out)


def _bub():
    from layout.bubble import _bubble_controls
    return _bubble_controls()


def _ids_in_order(root):
    return [n.id for n, _ in _walk(root)
            if isinstance(getattr(n, "id", None), str)]


# ── dropdown, resolver ──────────────────────────────────────────────────────

def test_dropdown_has_raw_entries_on_tab1_only():
    from layout.mc_controls import _mc_controls
    dd, _ = _by_id(_bub(), "bub-mc-model-src")
    by_val = {o["value"]: o["label"] for o in dd.options}
    assert "Raw · independent" in str(by_val["raw_iid"])
    assert "Raw · halving cycle" in str(by_val["raw_cycle"])
    for p in ("dca", "ret", "hm", "sc", "cp"):
        root = _mc_controls(p)
        vals = [o["value"] for o in _by_id(root, f"{p}-mc-model-src")[0].options]
        assert not set(vals) & set(RAW_SRCS), p


def test_resolver_passes_raw_through():
    from callbacks.charts._resolvers import _resolve_mc_model_src
    for src in RAW_SRCS:
        assert _resolve_mc_model_src(
            src, [3], [], [], 1, 1, "d", "d", "u", "u",
            1, 1, "d", "d", "u", "u") == src


# ── visibility chrome ───────────────────────────────────────────────────────

def test_paid_badge_hidden_in_raw_mode():
    root = _bub()
    panel, _ = _by_id(root, "bub-mc-panel")
    badges = [(n, a) for n, a in _walk(panel)
              if "model-panel-paid-badge" in _classes(n)]
    assert len(badges) == 1
    assert "qs-mc-model-only" in _classes(badges[0][0])
    # bub-mc-body is inside the same panel (I-1: one ancestor for both)
    assert len([n for n, _ in _walk(panel) if getattr(n, "id", None) == "bub-mc-body"]) == 1


def _entries(output):
    from dash import _callback as dc
    return [e for e in list(_APP._callback_list) + list(dc.GLOBAL_CALLBACK_LIST)
            if str(e["output"]) == output]


def test_mode_toggle_targets_panel():
    from layout.mc_controls import _mc_controls
    hits = _entries("bub-mc-panel.className")
    assert len(hits) == 1, hits
    e = hits[0]
    assert [(i["id"], i["property"]) for i in e["inputs"]] == [
        ("bub-mc-model-src", "value")]
    assert e["clientside_function"]
    for p in ("dca", "ret", "hm", "sc", "cp"):
        assert [n for n, _ in _walk(_mc_controls(p))
                if getattr(n, "id", None) == "bub-mc-panel"] == [], p


_MODEL_ONLY = ["bub-mc-start-yr", "bub-mc-entry-q", "bub-mc-sims", "bub-mc-years",
               "bub-mc-window", "bub-mc-bins", "bub-mc-regime", "bub-mc-regime-note",
               "bub-mc-cost", "bub-mc-status", "bub-mc-dl-btn", "bub-mc-upload",
               "bub-mc-restore-btn", "bub-mc-advanced", "bub-mc-adv-body",
               "bub-mc-freq", "bub-mc-ppy"]


def test_model_only_and_raw_only_classes():
    root = _bub()
    for cid in _MODEL_ONLY:
        n, anc = _by_id(root, cid)
        assert _has_cls_ancestor(anc, n, "qs-mc-model-only"), cid
        assert not _has_cls_ancestor(anc, n, "qs-mc-raw-only"), cid
    for key in ("mc-run-btn", "mc-run-status"):
        n, anc = _by_id(root, {"type": key, "tab": "bub"})
        assert _has_cls_ancestor(anc, n, "qs-mc-model-only"), key
    for cid in RAW_INPUT_IDS + ["bub-mc-raw-status"]:
        n, anc = _by_id(root, cid)
        assert _has_cls_ancestor(anc, n, "qs-mc-raw-only"), cid
        assert not _has_cls_ancestor(anc, n, "qs-mc-model-only"), cid
    for cid in ("bub-mc-paths", "bub-mc-model-src", "bub-mc-enable"):   # shared
        n, anc = _by_id(root, cid)
        assert not _has_cls_ancestor(anc, n, "qs-mc-model-only"), cid
        assert not _has_cls_ancestor(anc, n, "qs-mc-raw-only"), cid


def test_mode_classes_not_defeated_by_inline_display():
    """An inline style display on a classed element wins over the class rules,
    so the element would show in both modes."""
    for n, _ in _walk(_bub()):
        if {"qs-mc-model-only", "qs-mc-raw-only"} & set(_classes(n)):
            assert "display" not in (getattr(n, "style", None) or {}), \
                (getattr(n, "id", None), _classes(n))


def test_positions_of_lines_extremes_and_status():
    ids = _ids_in_order(_bub())
    i = ids.index("bub-mc-paths")
    assert ids[i + 1:i + 4] == ["bub-mc-lines-wrap", "bub-mc-lines", "bub-mc-extremes"]
    assert ids[ids.index("bub-mc-model-src") + 1] == "bub-mc-raw-status"


def test_control_values_and_ranges():
    root = _bub()
    g = lambda cid: _by_id(root, cid)[0]  # noqa: E731
    s = g("bub-mc-raw-sims")
    # M-10: no HTML min/max (out-of-range would arrive as null); server clamps
    mm = lambda n: (getattr(n, "min", None), getattr(n, "max", None))  # noqa: E731
    assert (s.type, *mm(s), s.step, s.value, s.debounce) == (
        "number", None, None, 1, 1000, True)
    y = g("bub-mc-raw-years")
    assert [o["value"] for o in y.options] == [10, 20, 30, 40] and y.value == 40
    assert all(isinstance(o["label"], str) for o in y.options)   # no bolding
    w = g("bub-mc-raw-window")
    assert (w.min, w.max, w.value) == (2010, YR, [2010, YR])
    t = g("bub-mc-raw-trend")
    assert [(o["value"], o["label"]) for o in t.options] == [
        ("power", "Power law"), ("exp", "Exponential decay"),
        ("none", "None (honest replay)")]
    assert t.value == "power"
    b = g("bub-mc-raw-bins")
    assert [o["value"] for o in b.options] == list(range(3, 11)) and b.value == 5
    r = g("bub-mc-raw-regime")
    assert r.value == [0, 1, 2, 3, 4]
    assert r.options == raw_mc.regime_options(5, (2010, YR), "power")
    ln = g("bub-mc-lines")
    assert (ln.type, *mm(ln), ln.step, ln.value, ln.debounce) == (
        "number", None, None, 1, 20, True)
    ex = g("bub-mc-extremes")
    assert [o["value"] for o in ex.options] == ["yes"] and ex.value == []
    assert "Show extremes" in str(ex.options[0]["label"])


def test_number_inputs_state_their_range_in_the_label():
    """M-10: with no HTML min/max the label is the only range hint."""
    labels = [_texts(n) for n, _ in _walk(_bub()) if type(n).__name__ == "Label"]
    assert f"Simulations (1–{RAW_SIMS_MAX}, free)" in labels
    assert "Lines shown (5–200)" in labels


def test_source_label_says_source_in_raw_only():
    """M-6: Raw draws no model's bands, so its label is "Source"; the model
    label text is unchanged (and the other tabs keep the plain label)."""
    from layout import mc_controls
    root = _bub()
    ids = _ids_in_order(root)
    lab = next(n for n, _ in _walk(root) if type(n).__name__ == "Label"
               and "Model source (quantile bands)" in _texts(n))
    spans = {(_texts(c), " ".join(_classes(c))) for c in lab.children}
    assert spans == {("Model source (quantile bands)", "qs-mc-model-only"),
                     ("Source", "qs-mc-raw-only")}
    assert "bub-mc-model-src" in ids
    for p in ("dca", "ret", "hm", "sc", "cp"):
        other = mc_controls._mc_controls(p)
        labs = [n for n, _ in _walk(other) if type(n).__name__ == "Label"
                and "Model source" in _texts(n)]
        assert labs and all(n.children == "Model source (quantile bands)"
                            for n in labs), p


def test_lines_block_hidden_until_paths():
    """M-6: Lines shown / Show extremes sit in bub-mc-lines-wrap, hidden by
    default and toggled by ONE clientside Input, bub-mc-paths.value."""
    wrap = _by_id(_bub(), "bub-mc-lines-wrap")[0]
    assert wrap.style == {"display": "none"}
    assert not _classes(wrap)               # the class rules live on the parent
    hits = _entries("bub-mc-lines-wrap.style")
    assert len(hits) == 1
    e = hits[0]
    assert [(i["id"], i["property"]) for i in e["inputs"]] == [("bub-mc-paths", "value")]
    assert e["clientside_function"]


@needs_node
def test_lines_block_visibility_js():
    from layout.mc_controls import RAW_LINES_VIS_JS
    script = ("const fn = (%s); console.log(JSON.stringify("
              "[['yes'], [], null].map(fn)));" % RAW_LINES_VIS_JS)
    res = subprocess.run([_NODE, "-e", script], capture_output=True, text=True,
                         timeout=30)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout) == [{}, {"display": "none"}, {"display": "none"}]


def test_layout_survives_any_regime_options_failure(monkeypatch):
    """I-1: the Tab-1 layout calls into raw_mc; a non-RuntimeError there (e.g.
    pandas < 2.2 rejecting resample("ME") with ValueError) must fall back to
    plain bin labels, not break the page."""
    def boom(*a, **k):
        raise ValueError("Invalid frequency: ME")
    monkeypatch.setattr(raw_mc, "regime_options", boom)
    r = _by_id(_bub(), "bub-mc-raw-regime")[0]
    assert r.options == [{"label": f"Bin {i + 1}", "value": i} for i in range(5)]
    assert r.value == [0, 1, 2, 3, 4]
    from layout import _build_layout
    assert _build_layout("bubble") is not None     # the whole page still builds
    from layout.mc_controls import raw_regime_options
    monkeypatch.setattr(raw_mc, "regime_options",
                        lambda *a, **k: (_ for _ in ()).throw(KeyError("x")))
    assert raw_regime_options(3, (2010, YR), "power") == [
        {"label": f"Bin {i + 1}", "value": i} for i in range(3)]


def test_raw_labels_and_hint():
    root = _bub()
    raw_text = " ".join(_texts(n) for n, _ in _walk(root)
                        if "qs-mc-raw-only" in _classes(n))
    assert "Raw starts today at the current price" in raw_text
    assert "Markov transition matrix" not in raw_text
    assert "≤ 200 is free" not in raw_text and "≤200 is free" not in raw_text


def test_css_rules():
    css = (_HERE / "assets" / "style.css").read_text()
    flat = " ".join(css.split())
    assert ".qs-mc-raw-mode .qs-mc-model-only { display: none" in flat
    assert ".qs-mc-raw-only { display: none" in flat
    assert ".qs-mc-raw-mode .qs-mc-raw-only { display: block" in flat


def test_placeholders_without_markov(monkeypatch):
    from layout import mc_controls
    monkeypatch.setattr(mc_controls._app_ctx, "_HAS_MARKOV", False)
    root = mc_controls._mc_controls("bub", show_paths_toggle=True,
                                    include_raw_src=True)
    for cid in NEW_IDS:
        _by_id(root, cid)
    for p in ("dca", "ret", "hm", "sc", "cp"):
        ids = _ids_in_order(mc_controls._mc_controls(p))
        assert not set(ids) & set(NEW_IDS), p


@needs_node
def test_raw_mode_class_toggle_js():
    from layout.mc_controls import RAW_MODE_CLASS_JS
    script = ("const fn = (%s); console.log(JSON.stringify("
              "['raw_cycle','raw_iid','bub',null,'qr'].map(fn)));" % RAW_MODE_CLASS_JS)
    res = subprocess.run([_NODE, "-e", script], capture_output=True, text=True,
                         timeout=30)
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout)
    assert "qs-mc-raw-mode" in got[0].split() and "qs-mc-raw-mode" in got[1].split()
    assert all("qs-mc-raw-mode" not in g.split() for g in got[2:])


# ── regime options callback ─────────────────────────────────────────────────

def test_raw_regime_options_callback(monkeypatch):
    from callbacks.mc_controls import _update_bub_raw_regime_opts as fn
    hits = _entries("bub-mc-raw-regime.options")
    assert len(hits) == 1
    e = hits[0]
    assert [(i["id"], i["property"]) for i in e["inputs"]] == [
        ("bub-mc-raw-bins", "value"), ("bub-mc-raw-window", "value"),
        ("bub-mc-raw-trend", "value")]
    assert e["prevent_initial_call"] is True
    assert not e["clientside_function"]
    got = fn(4, [2012, YR], "power")                     # list window
    assert got == raw_mc.regime_options(4, (2012, YR), "power")
    assert len(got) == 4 and "σ" in got[0]["label"]
    assert fn(None, None, None) == raw_mc.regime_options(5, (2010, YR), "power")
    # M-9: a failed exp fit gives plain bin labels, never an empty checklist
    monkeypatch.setattr(raw_mc, "fit_trend", lambda *a, **k: None)
    raw_mc.regime_options.cache_clear()
    try:
        assert fn(3, [2010, YR], "exp") == [
            {"label": f"Bin {i + 1}", "value": i} for i in range(3)]
    finally:
        monkeypatch.undo()
        raw_mc.regime_options.cache_clear()


def test_inner_cache_renamed():
    """F-23 N-3: distinct from layout.mc_controls._regime_options."""
    assert hasattr(raw_mc, "_regime_options_cached")
    assert not hasattr(raw_mc, "_regime_options")
    assert raw_mc.regime_options.cache_clear in raw_mc._CACHE_CLEARERS


def test_stale_data_fp_is_a_value_error(raw_prices):  # noqa: F811
    """F-23 N-5: not an assert (stripped under -O)."""
    from raw_mc_testing import _req
    with pytest.raises(ValueError, match="data_fp"):
        raw_mc.simulate(_req(data_fp="stale"))


# ── update_bubble: Raw branch ───────────────────────────────────────────────

def _kw(**over):
    kw = dict(
        _first_render=1,
        sel_qs=[0.5], adv_qs=[], toggles=["shade", "show_data"], ma_sel=[],
        bubble_toggles=["show_comp"], xscale="log", yscale="log",
        xrange=[2010, 2040], yrange=[-1.5, 7.0], n_future=3,
        ptsize=3, ptalpha=0.3, stack=0, show_stack=[], use_lots=[],
        legend_pos="top-left", model_show=["bub"],
        lppl_n_freqs=[3], lppl_weighted=[], lppl_no_13=[],
        hyb_a_nlog=1, hyb_a_ncal=1, hyb_a_log1d="d", hyb_a_log2d="d",
        hyb_a_cal1d="u", hyb_a_cal2d="u",
        hyb_b_enabled=[], hyb_b_nlog=0, hyb_b_ncal=0,
        hyb_b_log1d="d", hyb_b_log2d="d", hyb_b_cal1d="u", hyb_b_cal2d="u",
        ep_a_nlog=1, ep_a_ncal=1, ep_a_log1d="d", ep_a_log2d="d",
        ep_a_cal1d="u", ep_a_cal2d="u",
        ep_b_enabled=[], ep_b_nlog=0, ep_b_ncal=0,
        ep_b_log1d="d", ep_b_log2d="d", ep_b_cal1d="u", ep_b_cal2d="u",
        decomp_model="", decomp_components=[], decomp_mode="individual",
        lots_data=[], palette_key="default", user_model_store=None,
        _redraw_tick=0, _hybppl_commit=0, _eppl_commit=0, _bm_commit=0,
        mc_enable=["yes"], mc_bins=5, mc_regime=[0, 1, 2, 3, 4],
        mc_sims=8, mc_years=40, mc_window=[2010, 2026],
        mc_start_yr=2028, mc_entry_q=10, _mc_loaded=None,
        mc_model_src="raw_cycle",
        pay_trigger=1, price_data=70000,
        mc_cached=None, pay_token=None, mc_unblocked=None, mc_auth=None,
        cta_active=[], qs_mode=[], scan_active=[], scan_q_val=None,
        sigma_mode="resqr", snapshot_pending=False, loaded_hash=None,
        active_chart_committed=None, mc_paths=[],
        mc_raw_sims=200, mc_raw_years=40, mc_raw_window=[2010, YR],
        mc_raw_trend="power", mc_raw_bins=5, mc_raw_regime=[0, 1, 2, 3, 4],
        mc_lines=20, mc_extremes=[],
    )
    kw.update(over)
    return kw


def _run(**over):
    with _patch_ctx("bub-mc-model-src"):
        return update_bubble(**_kw(**over))


def _names(fig):
    return [t.name for t in fig.data if t.name]


def _spaghetti(fig):
    return [t for t in fig.data if getattr(t, "legendgroup", "") == "mc-spaghetti"]


def test_raw_branch_draws_fan_and_status():
    res = _run()
    fig, status = res[0], res[8]
    assert f"MC median · raw cycle, power law, 2010–{YR}" in _names(fig)
    assert status and _texts(status).strip()
    assert "Power law" in _texts(status)
    iid = _run(mc_model_src="raw_iid", mc_raw_trend="exp")[0]
    assert f"MC median · raw independent, exponential decay, 2010–{YR}" in _names(iid)


@pytest.mark.parametrize("exc", [AssertionError("len(k_w) == 47"),
                                 ValueError("bad"), ZeroDivisionError()])
def test_raw_simulate_failure_draws_no_fan_and_never_raises(monkeypatch, caplog, exc):
    """M-7: an exception inside simulate must not 500 update_bubble: the chart
    still draws (no fan), the status says why, and the exception is logged."""
    def boom(req):
        raise exc
    monkeypatch.setattr(raw_mc, "simulate", boom)
    with caplog.at_level("ERROR", logger="callbacks.charts._raw"):
        res = _run(mc_paths=["yes"], mc_extremes=["yes"])
    fig, status = res[0], res[8]
    assert len(res) == 9 and fig is not dash.no_update and len(fig.data) > 0
    assert not any("MC" in n for n in _names(fig))
    assert _spaghetti(fig) == []
    assert _texts(status).strip() == "Raw simulation failed: no fan drawn"
    assert any("Raw simulation failed" in r.getMessage() and r.exc_info
               for r in caplog.records)


def test_raw_refusal_draws_no_fan_but_says_why():
    res = _run(mc_raw_window=[2024, YR])
    assert not any("MC median" in n for n in _names(res[0]))
    assert "Fewer than 36 months" in _texts(res[8])


def test_raw_branch_never_calls_payment(monkeypatch):
    import btcpay
    import mc_cache
    import callbacks.charts as charts

    def boom(*a, **k):
        raise AssertionError("payment / cache path reached in Raw mode")
    monkeypatch.setattr(charts, "_mc_setup", boom)
    monkeypatch.setattr(btcpay, "is_free_tier", boom)
    monkeypatch.setattr(mc_cache, "get_cached_paths", boom)
    if hasattr(btcpay, "create_invoice"):
        monkeypatch.setattr(btcpay, "create_invoice", boom)
    for src in RAW_SRCS:
        fig = _run(mc_model_src=src, mc_paths=["yes"])[0]
        assert any("MC median" in n for n in _names(fig))


def test_raw_outputs_leave_mc_stores_alone():
    res = _run()
    assert len(res) == 9
    for i in (1, 4, 5, 6):          # mc-results, save-modal, save-tab, unblocked
        assert res[i] is dash.no_update, i


def test_lines_and_extremes_only_with_paths():
    assert _spaghetti(_run(mc_paths=[])[0]) == []
    off = _run(mc_paths=[], mc_extremes=["yes"])[0]
    assert "Highest end" not in _names(off)
    on = _run(mc_paths=["yes"])[0]
    assert len(_spaghetti(on)) == RAW_LINES_DEFAULT
    assert "Highest end" not in _names(on)
    ext = _run(mc_paths=["yes"], mc_extremes=["yes"])[0]
    assert len(_spaghetti(ext)) == RAW_LINES_DEFAULT
    assert {"Highest end", "Lowest end"} <= set(_names(ext))
    assert len(ext.data) == len(on.data) + 2
    assert len(_spaghetti(_run(mc_paths=["yes"], mc_lines=7)[0])) == 7


@pytest.mark.parametrize("raw, want", [(None, 20), ("x", 20), (2, 5), (500, 200),
                                       (5, 5), (200, 200), (33, 33)])
def test_lines_coercion(raw, want):
    assert len(_spaghetti(_run(mc_paths=["yes"], mc_lines=raw)[0])) == want


def _spy_simulate(monkeypatch):
    seen = []
    real = raw_mc.simulate

    def spy(req):
        seen.append(req)
        return real(req)
    monkeypatch.setattr(raw_mc, "simulate", spy)
    return seen


def test_raw_start_price_falls_back_to_last_close(monkeypatch):
    seen = _spy_simulate(monkeypatch)
    last = float(raw_mc.get_prices()[0].iloc[-1])
    for pd_ in (None, 0, -5):
        fig = _run(price_data=pd_)[0]
        assert seen[-1].p0 == last
        med = next(t for t in fig.data if t.name and t.name.startswith("MC median"))
        assert med.y[0] == pytest.approx(last)
    _run(price_data=91234.5)
    assert seen[-1].p0 == 91234.5


def test_raw_request_fields(monkeypatch):
    seen = _spy_simulate(monkeypatch)
    _run(mc_model_src="raw_iid", mc_raw_years=20, mc_raw_window=[2013, 2025],
         mc_raw_trend="none", mc_raw_bins=7, mc_raw_regime=[1, 2])
    req = seen[-1]
    assert (req.variant, req.trend, req.window, req.bins, req.checked,
            req.years) == ("raw_iid", "none", (2013, 2025), 7, (1, 2), 20)
    assert req.today == pd.Timestamp.today().normalize()
    assert req.data_fp == raw_mc.get_prices()[1] and req.asof is None


def test_raw_stack_scaling(monkeypatch):
    base = _run()[0]
    scaled = _run(stack=2.0, show_stack=["yes"])[0]
    name = f"MC median · raw cycle, power law, 2010–{YR}"
    a = next(t for t in base.data if t.name == name)
    b = next(t for t in scaled.data if t.name == name)
    assert np.asarray(b.y, float)[0] == pytest.approx(2 * np.asarray(a.y, float)[0])


def test_raw_trimmed_at_visible_right_edge():
    from btc_core import yr_to_t
    fig = _run(xrange=[2010, 2035])[0]
    med = next(t for t in fig.data if t.name and t.name.startswith("MC median"))
    t_hi = yr_to_t(2035, _app_ctx.M.genesis)
    x = np.asarray(med.x, float)
    assert x[-1] >= t_hi and x[-2] < t_hi        # one step past the edge


def test_raw_zoom_lock_applied():
    for zoom in (True, False):
        tg = ["shade", "show_data"] + (["chart_zoom"] if zoom else [])
        raw = _run(toggles=tg)[0]
        model = _run(toggles=tg, mc_enable=[])[0]
        assert raw.layout.dragmode == model.layout.dragmode == ("zoom" if zoom else False)
        assert raw.layout.xaxis.fixedrange is model.layout.xaxis.fixedrange \
            is (not zoom)
        assert raw.layout.yaxis.fixedrange is (not zoom)


def test_app_registers_prices():
    series, fp = raw_mc.get_prices()
    if fp == "csv":                        # a raw_prices test left it: impossible
        pytest.fail("fixture registry leaked")
    assert fp == _app_ctx._MODEL_FP
    assert float(series.iloc[-1]) == float(_app_ctx.M.price_prices[-1])
    assert series.index[-1] == pd.Timestamp(_app_ctx.M.price_dates[-1])


def test_app_registration_in_source():
    src = (_HERE / "app.py").read_text()
    assert src.count("raw_mc.set_prices(") == 1


def test_prices_built_once(monkeypatch, raw_prices):  # noqa: F811
    calls = []
    real = raw_mc.load_months

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(raw_mc, "load_months", counting)
    _run()
    _run(mc_raw_sims=300)
    assert len(calls) == 1


@pytest.mark.parametrize("raw, want", [(None, 1000), (0, 1), (5000, RAW_SIMS_MAX),
                                       ("x", 1000), (250, 250)])
def test_raw_sims_coercion(monkeypatch, raw, want):
    seen = _spy_simulate(monkeypatch)
    _run(mc_raw_sims=raw)
    assert seen[-1].sims == want


def test_raw_other_coercions(monkeypatch):
    seen = _spy_simulate(monkeypatch)
    _run(mc_raw_years=None, mc_raw_window=None, mc_raw_trend="bogus",
         mc_raw_bins=None, mc_raw_regime=None)
    req = seen[-1]
    assert (req.years, req.window, req.trend, req.bins, req.checked) == (
        40, (2010, YR), "power", 5, (0, 1, 2, 3, 4))
    _run(mc_raw_years=15, mc_raw_bins=99, mc_raw_regime="x")
    assert (seen[-1].years, seen[-1].bins, seen[-1].checked) == (40, 5, (0, 1, 2, 3, 4))


def test_model_mode_returns_empty_raw_status():
    assert _run(mc_enable=[])[8] == ""
    assert _run(mc_model_src="bub")[8] == ""


def test_return_tuple_width():
    fn = _update_bubble_ast()
    dec = next(d for d in fn.decorator_list if isinstance(d, ast.Call))
    n_out = sum(1 for a in dec.args if isinstance(a, ast.Call)
                and getattr(a.func, "id", "") == "Output")
    assert n_out == 9
    widths = [n.right.value for n in ast.walk(fn)
              if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult)
              and isinstance(n.left, ast.Tuple)
              and isinstance(n.right, ast.Constant)]
    assert widths and set(widths) == {9}
    rets = [n.value for n in ast.walk(fn) if isinstance(n, ast.Return)
            and isinstance(n.value, ast.Tuple)]
    assert rets and all(len(r.elts) == 9 for r in rets)
    assert len(_run(snapshot_pending=True)) == 9


def _update_bubble_ast():
    path = _HERE / "callbacks" / "charts" / "__init__.py"
    return next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == "update_bubble")


def test_new_inputs_last_and_in_post_restore():
    fn = _update_bubble_ast()
    dec = next(d for d in fn.decorator_list if isinstance(d, ast.Call))
    seq = [(a.func.id, a.args[0].value) for a in dec.args
           if isinstance(a, ast.Call) and isinstance(a.func, ast.Name)
           and a.func.id in ("Input", "State")]
    params = [a.arg for a in fn.args.args]
    assert seq[-8:] == [("Input", i) for i in RAW_INPUT_IDS]
    assert params[-8:] == RAW_PARAMS
    assert seq[-9] == ("Input", "bub-mc-paths")
    found = None
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_POST_RESTORE_TRIGGERS"):
            found = {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    assert set(RAW_INPUT_IDS) <= found


def test_raw_branch_above_mc_setup():
    """F-23 N-6: Raw is decided (mc_visible + RAW_SRCS) before _mc_setup."""
    src = ast.get_source_segment(
        (_HERE / "callbacks" / "charts" / "__init__.py").read_text(),
        _update_bubble_ast())
    assert src.index("mc_visible =") < src.index("_mc_setup(")
    assert src.index("RAW_SRCS") < src.index("_mc_setup(")


def test_raw_params_not_in_bubble_cache_key(monkeypatch):
    import callbacks.charts as charts
    import utils
    seen = []
    real = charts._get_bubble_fig

    def spy(p):
        seen.append(dict(p))
        return real(p)
    monkeypatch.setattr(charts, "_get_bubble_fig", spy)
    monkeypatch.setattr(utils, "_get_mc_bubble_fig",
                        lambda p: pytest.fail("MC builder used in Raw mode"))
    _run()
    assert len(seen) == 1
    assert not [k for k in seen[0] if "raw" in k or k.startswith("mc_")]
    off = []
    monkeypatch.setattr(charts, "_get_bubble_fig",
                        lambda p: (off.append(dict(p)), real(p))[1])
    _run(mc_enable=[])
    assert off[0] == seen[0]        # same key as MC off: an L1 hit
