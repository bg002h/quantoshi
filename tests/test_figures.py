"""Comprehensive test suite for Quantoshi web app."""
import sys, os, json, math, traceback
sys.path.insert(0, "/scratch/code/bitcoinprojections/btc_web")
sys.path.insert(0, "/scratch/code/bitcoinprojections/btc_app")
os.chdir("/scratch/code/bitcoinprojections/btc_web")

import pandas as pd
import numpy as np
from figures import (build_bubble_figure, build_heatmap_figure,
                     build_dca_figure, build_retire_figure,
                     build_supercharge_figure)
from btc_core import (load_model_data, _find_lot_percentile, fmt_price,
                      yr_to_t, today_t, leo_weighted_entry, qr_price)

M = load_model_data()
errors = []
passed = 0

def ok(label):
    global passed
    passed += 1
    print(f"  ✓ {label}")

def fail(label, e):
    errors.append(f"{label}: {e}")
    print(f"  ✗ {label}: {e}")
    traceback.print_exc()

# ── helpers ──────────────────────────────────────────────────────────────────
LOTS_25 = []
for date_str, btc, price in [
    ("2017-01-15",0.50,820),("2017-06-10",0.30,2700),("2017-12-17",0.10,19700),
    ("2018-02-06",0.25,6900),("2018-06-29",0.40,6200),("2018-12-15",0.50,3200),
    ("2019-04-02",0.30,4900),("2019-06-26",0.20,13000),("2019-12-18",0.35,7100),
    ("2020-03-12",1.00,5000),("2020-05-11",0.50,8700),("2020-10-21",0.25,12900),
    ("2020-12-31",0.15,29000),("2021-01-29",0.10,34300),("2021-04-14",0.05,63500),
    ("2021-07-20",0.20,29800),("2021-11-10",0.05,68700),("2022-01-24",0.15,36200),
    ("2022-06-18",0.50,18900),("2022-11-21",0.30,15800),("2023-01-14",0.20,20900),
    ("2023-06-22",0.15,30400),("2023-10-24",0.10,33900),("2024-03-14",0.05,73000),
    ("2024-11-22",0.10,98800),
]:
    t = (pd.Timestamp(date_str) - M.genesis).days / 365.25
    pct_q = _find_lot_percentile(t, float(price), M.qr_fits)
    LOTS_25.append({"date":date_str,"btc":btc,"price":float(price),"pct_q":round(pct_q,6),"notes":""})

TOTAL_BTC = sum(l["btc"] for l in LOTS_25)

# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL DATA INTEGRITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 1. Model Data Integrity ===")

try:
    assert hasattr(M, 'qr_fits') and len(M.qr_fits) > 0
    ok(f"qr_fits loaded: {len(M.qr_fits)} quantiles")
except Exception as e: fail("qr_fits", e)

try:
    assert hasattr(M, 'genesis') and isinstance(M.genesis, pd.Timestamp)
    ok(f"genesis: {M.genesis.date()}")
except Exception as e: fail("genesis", e)

try:
    assert hasattr(M, 'QR_QUANTILES') and len(M.QR_QUANTILES) > 0
    ok(f"QR_QUANTILES: {len(M.QR_QUANTILES)} entries")
except Exception as e: fail("QR_QUANTILES", e)

try:
    qs = sorted(M.qr_fits.keys())
    assert all(0 < q < 1 for q in qs), "quantiles out of range"
    ok(f"quantile range: Q{qs[0]*100}% – Q{qs[-1]*100}%")
except Exception as e: fail("quantile range", e)

try:
    # Check that qr_price works for a few quantiles
    t = 15.0  # ~2024
    for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
        if q in M.qr_fits:
            p = qr_price(q, t, M.qr_fits)
            assert p > 0, f"qr_price({q}, {t}) = {p}"
    ok("qr_price returns positive values")
except Exception as e: fail("qr_price", e)

# ══════════════════════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. Helper Functions ===")

try:
    assert fmt_price(50) == "$50"
    assert "K" in fmt_price(50000)
    assert "M" in fmt_price(1e6)
    ok(f"fmt_price: {fmt_price(50)}, {fmt_price(50000)}, {fmt_price(1e6)}")
except Exception as e: fail("fmt_price", e)

try:
    t = yr_to_t(2024, M.genesis)
    assert 14 < t < 16, f"yr_to_t(2024) = {t}"
    ok(f"yr_to_t(2024) = {t:.2f}")
except Exception as e: fail("yr_to_t", e)

try:
    t = today_t(M.genesis)
    assert t > 15, f"today_t = {t}"
    ok(f"today_t = {t:.4f}")
except Exception as e: fail("today_t", e)

try:
    pct = _find_lot_percentile(15.0, 50000, M.qr_fits)
    assert pct is not None and 0 < pct < 1
    ok(f"_find_lot_percentile(t=15, $50K) = Q{pct*100:.1f}%")
except Exception as e: fail("_find_lot_percentile", e)

try:
    # Edge: very low price → should be near Q0
    pct_low = _find_lot_percentile(15.0, 1.0, M.qr_fits)
    # Edge: very high price → should be near Q100
    pct_high = _find_lot_percentile(15.0, 1e9, M.qr_fits)
    assert pct_low < 0.01 or pct_low is not None
    assert pct_high > 0.99 or pct_high is not None
    ok(f"_find_lot_percentile edge cases: $1→Q{pct_low*100:.2f}%, $1B→Q{pct_high*100:.2f}%")
except Exception as e: fail("_find_lot_percentile edges", e)

try:
    entry = leo_weighted_entry(LOTS_25)
    assert entry is not None and len(entry) == 4
    ok(f"leo_weighted_entry: price=${entry[0]:.0f}, t={entry[1]:.2f}, q={entry[2]:.4f}, btc={entry[3]:.2f}")
except Exception as e: fail("leo_weighted_entry", e)

# ══════════════════════════════════════════════════════════════════════════════
# 3. BUBBLE CHART (Tab 1)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. Bubble Chart ===")

def bubble_params(**kw):
    base = dict(
        selected_qs=[], shade=True, show_ols=False, show_data=True,
        show_today=True, show_legend=False, show_comp=False, show_sup=False,
        xscale="log", yscale="log", xmin=2012, xmax=2030,
        ymin=0.01, ymax=1e8, n_future=3, pt_size=2, pt_alpha=0.2,
        stack=0, show_stack=False, use_lots=False, lots=[],
        comp_color="#FFD700", comp_lw=2.0, sup_color="#888888", sup_lw=1.5,
    )
    base.update(kw)
    return base

# Default (no quantiles)
try:
    fig = build_bubble_figure(M, bubble_params())
    assert len(fig.data) >= 1
    ok(f"default (no qs): {len(fig.data)} traces")
except Exception as e: fail("bubble default", e)

# Multiple quantiles
for qs in [[0.05], [0.25, 0.50, 0.75], [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]]:
    try:
        fig = build_bubble_figure(M, bubble_params(selected_qs=qs, show_legend=True))
        ok(f"{len(qs)} quantiles: {len(fig.data)} traces")
    except Exception as e: fail(f"bubble {len(qs)} qs", e)

# Scale combos
for xs, ys in [("linear","linear"),("linear","log"),("log","linear"),("log","log")]:
    try:
        fig = build_bubble_figure(M, bubble_params(xscale=xs, yscale=ys, selected_qs=[0.50]))
        ok(f"scales {xs}/{ys}")
    except Exception as e: fail(f"bubble {xs}/{ys}", e)

# X range extremes
for xmin, xmax in [(2009, 2050), (2020, 2025), (2024, 2026)]:
    try:
        fig = build_bubble_figure(M, bubble_params(xmin=xmin, xmax=xmax, selected_qs=[0.50]))
        ok(f"xrange [{xmin}, {xmax}]")
    except Exception as e: fail(f"bubble xrange [{xmin},{xmax}]", e)

# Show stack
try:
    fig = build_bubble_figure(M, bubble_params(selected_qs=[0.05,0.50,0.95], stack=2.0, show_stack=True, show_legend=True))
    ok(f"show_stack=2.0: {len(fig.data)} traces")
except Exception as e: fail("bubble show_stack", e)

# Composite + superimposed
try:
    fig = build_bubble_figure(M, bubble_params(show_comp=True, show_sup=True))
    ok(f"composites on: {len(fig.data)} traces")
except Exception as e: fail("bubble composites", e)

# N future bubbles 0 vs 5
for n in [0, 1, 5]:
    try:
        fig = build_bubble_figure(M, bubble_params(n_future=n))
        ok(f"n_future={n}: {len(fig.data)} traces")
    except Exception as e: fail(f"bubble n_future={n}", e)

# Pt size and alpha extremes
for ps, pa in [(1, 0.05), (20, 1.0), (5, 0.5)]:
    try:
        fig = build_bubble_figure(M, bubble_params(pt_size=ps, pt_alpha=pa, show_data=True))
        ok(f"pt_size={ps}, pt_alpha={pa}")
    except Exception as e: fail(f"bubble pt {ps}/{pa}", e)

# With lots
try:
    fig = build_bubble_figure(M, bubble_params(
        selected_qs=[0.05,0.50], use_lots=True, lots=LOTS_25,
        stack=TOTAL_BTC, show_stack=True, show_legend=True))
    ok(f"25 lots + show_stack: {len(fig.data)} traces")
except Exception as e: fail("bubble 25 lots", e)

# All toggles off
try:
    fig = build_bubble_figure(M, bubble_params(
        shade=False, show_data=False, show_today=False, show_legend=False,
        show_comp=False, show_sup=False, selected_qs=[0.50]))
    ok(f"all toggles off: {len(fig.data)} traces")
except Exception as e: fail("bubble all off", e)

# ══════════════════════════════════════════════════════════════════════════════
# 4. HEATMAP (Tab 2)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 4. Heatmap ===")

def heatmap_params(**kw):
    base = dict(
        entry_yr=2025, entry_q=50, live_price=95000,
        exit_yr_lo=2026, exit_yr_hi=2035,
        exit_qs=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
        color_mode=0, b1=0, b2=20,
        c_lo="#d73027", c_mid1="#fee08b", c_mid2="#d9ef8b", c_hi="#1a9850",
        n_disc=32, vfmt="cagr", cell_font_size=10,
        show_colorbar=False, stack=0, use_lots=False, lots=[],
    )
    base.update(kw)
    return base

# All vfmt modes
for vfmt in ["cagr","price","both","stack","port_only","mult_only","cagr_mult","mult_port","none"]:
    try:
        fig = build_heatmap_figure(M, heatmap_params(vfmt=vfmt, stack=1.0))
        ok(f"vfmt={vfmt}")
    except Exception as e: fail(f"heatmap vfmt={vfmt}", e)

# All color modes
for cm in [0, 1, 2]:
    try:
        fig = build_heatmap_figure(M, heatmap_params(color_mode=cm))
        ok(f"color_mode={cm}")
    except Exception as e: fail(f"heatmap color_mode={cm}", e)

# Entry year variations
for yr, lp in [(2015, None), (2020, None), (2025, 95000), (2030, None)]:
    try:
        fig = build_heatmap_figure(M, heatmap_params(entry_yr=yr, live_price=lp,
                                                      exit_yr_lo=yr+1, exit_yr_hi=yr+10))
        ok(f"entry_yr={yr}")
    except Exception as e: fail(f"heatmap entry_yr={yr}", e)

# Extreme entry quantiles
for eq in [0.1, 1.0, 5.0, 50.0, 95.0, 99.0, 99.9]:
    try:
        fig = build_heatmap_figure(M, heatmap_params(entry_q=eq))
        ok(f"entry_q={eq}%")
    except Exception as e: fail(f"heatmap entry_q={eq}", e)

# Few exit quantiles
try:
    fig = build_heatmap_figure(M, heatmap_params(exit_qs=[0.50]))
    ok("single exit quantile")
except Exception as e: fail("heatmap single exit_q", e)

# Wide exit range
try:
    fig = build_heatmap_figure(M, heatmap_params(exit_yr_lo=2026, exit_yr_hi=2060))
    ok("wide exit range 2026-2060")
except Exception as e: fail("heatmap wide exit", e)

# Short exit range (1 year)
try:
    fig = build_heatmap_figure(M, heatmap_params(exit_yr_lo=2026, exit_yr_hi=2027))
    ok("short exit range 2026-2027")
except Exception as e: fail("heatmap short exit", e)

# Custom break points
try:
    fig = build_heatmap_figure(M, heatmap_params(b1=-10, b2=50))
    ok("custom breaks b1=-10 b2=50")
except Exception as e: fail("heatmap custom breaks", e)

# With colorbar
try:
    fig = build_heatmap_figure(M, heatmap_params(show_colorbar=True))
    ok("show_colorbar=True")
except Exception as e: fail("heatmap colorbar", e)

# With lots
try:
    fig = build_heatmap_figure(M, heatmap_params(use_lots=True, lots=LOTS_25, stack=TOTAL_BTC))
    ok(f"25 lots, stack={TOTAL_BTC:.2f}")
except Exception as e: fail("heatmap 25 lots", e)

# Custom colors
try:
    fig = build_heatmap_figure(M, heatmap_params(
        c_lo="#0000ff", c_mid1="#ffffff", c_mid2="#ffffff", c_hi="#ff0000"))
    ok("custom colors")
except Exception as e: fail("heatmap custom colors", e)

# ══════════════════════════════════════════════════════════════════════════════
# 5. DCA (Tab 3)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 5. DCA ===")

def dca_params(**kw):
    base = dict(
        selected_qs=[0.50], amount=100, freq="Monthly",
        yr_start=2025, yr_end=2035, disp="btc",
        dual_y=True, show_legend=True, log_y=False, annotate=False,
        start_stack=0, use_lots=False, lots=[],
        sc_enable=False, sc_loan=1200, sc_rate=13, sc_term=12,
        sc_type="interest_only", sc_repeats=0, sc_entry_mode="live",
        sc_custom_price=None, sc_tax=33, sc_rollover=False,
        live_price=95000,
    )
    base.update(kw)
    return base

# Basic modes
for disp in ["btc", "usd"]:
    try:
        fig = build_dca_figure(M, dca_params(disp=disp))
        ok(f"disp={disp}: {len(fig.data)} traces")
    except Exception as e: fail(f"dca disp={disp}", e)

# All frequencies
for freq in ["Daily", "Weekly", "Monthly", "Quarterly", "Annually"]:
    try:
        fig = build_dca_figure(M, dca_params(freq=freq))
        ok(f"freq={freq}")
    except Exception as e: fail(f"dca freq={freq}", e)

# Multiple quantiles
try:
    fig = build_dca_figure(M, dca_params(selected_qs=[0.01, 0.10, 0.25, 0.50, 0.75, 0.95]))
    ok(f"6 quantiles: {len(fig.data)} traces")
except Exception as e: fail("dca 6 qs", e)

# Amount edge cases
for amt in [1, 10, 1000, 100000]:
    try:
        fig = build_dca_figure(M, dca_params(amount=amt))
        ok(f"amount=${amt}")
    except Exception as e: fail(f"dca amount={amt}", e)

# Year range variations
try:
    fig = build_dca_figure(M, dca_params(yr_start=2020, yr_end=2050))
    ok("yr 2020-2050")
except Exception as e: fail("dca yr 2020-2050", e)

try:
    fig = build_dca_figure(M, dca_params(yr_start=2025, yr_end=2027))
    ok("short range 2025-2027")
except Exception as e: fail("dca short range", e)

# Toggle combos
try:
    fig = build_dca_figure(M, dca_params(log_y=True, annotate=True, dual_y=False, show_legend=False))
    ok("log_y+annotate, no dual_y/legend")
except Exception as e: fail("dca toggles", e)

# With starting stack
try:
    fig = build_dca_figure(M, dca_params(start_stack=1.0))
    ok("start_stack=1.0")
except Exception as e: fail("dca start_stack", e)

# Stack-celerator ON
try:
    fig = build_dca_figure(M, dca_params(sc_enable=True, sc_loan=5000, sc_rate=10, sc_term=24))
    ok(f"SC on: {len(fig.data)} traces")
except Exception as e: fail("dca SC on", e)

# SC with 0% interest
try:
    fig = build_dca_figure(M, dca_params(sc_enable=True, sc_rate=0, sc_term=12))
    ok("SC rate=0%")
except Exception as e: fail("dca SC rate=0", e)

# SC amortizing
try:
    fig = build_dca_figure(M, dca_params(sc_enable=True, sc_type="amortizing", sc_rate=8, sc_term=36))
    ok("SC amortizing")
except Exception as e: fail("dca SC amortizing", e)

# SC with repeats
try:
    fig = build_dca_figure(M, dca_params(sc_enable=True, sc_repeats=3))
    ok("SC repeats=3")
except Exception as e: fail("dca SC repeats", e)

# SC with rollover (interest_only only)
try:
    fig = build_dca_figure(M, dca_params(sc_enable=True, sc_rollover=True))
    ok("SC rollover")
except Exception as e: fail("dca SC rollover", e)

# SC entry modes
for em in ["live", "model", "custom"]:
    try:
        fig = build_dca_figure(M, dca_params(sc_enable=True, sc_entry_mode=em, sc_custom_price=80000))
        ok(f"SC entry_mode={em}")
    except Exception as e: fail(f"dca SC entry={em}", e)

# DCA with lots
try:
    fig = build_dca_figure(M, dca_params(use_lots=True, lots=LOTS_25, start_stack=TOTAL_BTC))
    ok(f"25 lots: {len(fig.data)} traces")
except Exception as e: fail("dca 25 lots", e)

# DCA + SC + lots
try:
    fig = build_dca_figure(M, dca_params(
        use_lots=True, lots=LOTS_25, start_stack=TOTAL_BTC,
        sc_enable=True, sc_loan=10000, sc_rate=8, sc_term=24, sc_repeats=1,
        selected_qs=[0.25, 0.50, 0.75]))
    ok(f"SC + 25 lots: {len(fig.data)} traces")
except Exception as e: fail("dca SC+lots", e)

# ══════════════════════════════════════════════════════════════════════════════
# 6. RETIRE (Tab 4)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 6. Retire ===")

def retire_params(**kw):
    base = dict(
        selected_qs=[0.01, 0.10, 0.25], wd=50000, freq="Annually",
        yr_start=2031, yr_end=2075, infl=4.0,
        disp="usd", log_y=True, dual_y=True, annotate=True, show_legend=True,
        start_stack=1.0, use_lots=False, lots=[],
    )
    base.update(kw)
    return base

# Basic
try:
    fig = build_retire_figure(M, retire_params())
    ok(f"default: {len(fig.data)} traces")
except Exception as e: fail("retire default", e)

# Display modes
for disp in ["usd", "btc"]:
    try:
        fig = build_retire_figure(M, retire_params(disp=disp))
        ok(f"disp={disp}")
    except Exception as e: fail(f"retire disp={disp}", e)

# All frequencies
for freq in ["Daily", "Weekly", "Monthly", "Quarterly", "Annually"]:
    try:
        fig = build_retire_figure(M, retire_params(freq=freq))
        ok(f"freq={freq}")
    except Exception as e: fail(f"retire freq={freq}", e)

# Inflation 0%
try:
    fig = build_retire_figure(M, retire_params(infl=0.0))
    ok("infl=0%")
except Exception as e: fail("retire infl=0", e)

# Inflation high
try:
    fig = build_retire_figure(M, retire_params(infl=20.0))
    ok("infl=20%")
except Exception as e: fail("retire infl=20", e)

# Withdrawal amounts
for wd in [1000, 100000, 1000000]:
    try:
        fig = build_retire_figure(M, retire_params(wd=wd))
        ok(f"wd=${wd:,}")
    except Exception as e: fail(f"retire wd={wd}", e)

# Many quantiles
try:
    fig = build_retire_figure(M, retire_params(
        selected_qs=[0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]))
    ok(f"11 quantiles: {len(fig.data)} traces")
except Exception as e: fail("retire 11 qs", e)

# Year range near-term
try:
    fig = build_retire_figure(M, retire_params(yr_start=2026, yr_end=2035))
    ok("yr 2026-2035")
except Exception as e: fail("retire near yr", e)

# Large stack
try:
    fig = build_retire_figure(M, retire_params(start_stack=100.0))
    ok("stack=100 BTC")
except Exception as e: fail("retire stack=100", e)

# Small stack (likely depletes fast)
try:
    fig = build_retire_figure(M, retire_params(start_stack=0.01, wd=100000))
    ok("stack=0.01 BTC, wd=$100K (fast depletion)")
except Exception as e: fail("retire fast depletion", e)

# With lots
try:
    fig = build_retire_figure(M, retire_params(use_lots=True, lots=LOTS_25, start_stack=TOTAL_BTC))
    ok(f"25 lots: {len(fig.data)} traces")
except Exception as e: fail("retire 25 lots", e)

# Toggle combos
try:
    fig = build_retire_figure(M, retire_params(log_y=False, dual_y=False, annotate=False, show_legend=False))
    ok("all toggles off")
except Exception as e: fail("retire toggles off", e)

# ══════════════════════════════════════════════════════════════════════════════
# 7. SUPERCHARGE (Tab 5)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 7. Supercharge ===")

def sc_params(**kw):
    base = dict(
        selected_qs=[0.001, 0.10], stack=1.0, use_lots=False, lots=[],
        start_yr=2033, delays=[0,0,0,1,2],
        freq="Annually", infl=4.0, wd=100000, end_yr=2075,
        target_yr=2060, disp="usd", mode="A",
        annotate=True, log_y=True, show_legend=True,
        shade=True, display_q=0.05,
    )
    base.update(kw)
    return base

# Mode A default
try:
    fig = build_supercharge_figure(M, sc_params())
    ok(f"Mode A default: {len(fig.data)} traces")
except Exception as e: fail("sc Mode A", e)

# Mode B
try:
    fig = build_supercharge_figure(M, sc_params(mode="B"))
    ok(f"Mode B: {len(fig.data)} traces")
except Exception as e: fail("sc Mode B", e)

# Shade off (single-quantile lines)
try:
    fig = build_supercharge_figure(M, sc_params(shade=False, display_q=0.10))
    ok(f"shade off, display_q=Q10%: {len(fig.data)} traces")
except Exception as e: fail("sc shade off", e)

# All frequencies
for freq in ["Daily", "Weekly", "Monthly", "Quarterly", "Annually"]:
    try:
        fig = build_supercharge_figure(M, sc_params(freq=freq))
        ok(f"freq={freq}")
    except Exception as e: fail(f"sc freq={freq}", e)

# Inflation 0
try:
    fig = build_supercharge_figure(M, sc_params(infl=0.0))
    ok("infl=0%")
except Exception as e: fail("sc infl=0", e)

# Various delay patterns
for delays in [[0,0,0,0,0], [1,2,3,4,5], [0,5,10,15,20]]:
    try:
        fig = build_supercharge_figure(M, sc_params(delays=delays))
        ok(f"delays={delays}")
    except Exception as e: fail(f"sc delays={delays}", e)

# Display modes
for disp in ["usd", "btc"]:
    try:
        fig = build_supercharge_figure(M, sc_params(disp=disp))
        ok(f"disp={disp}")
    except Exception as e: fail(f"sc disp={disp}", e)

# Mode B with different target years
for ty in [2035, 2050, 2075]:
    try:
        fig = build_supercharge_figure(M, sc_params(mode="B", target_yr=ty))
        ok(f"Mode B target_yr={ty}")
    except Exception as e: fail(f"sc Mode B ty={ty}", e)

# More quantiles
try:
    fig = build_supercharge_figure(M, sc_params(
        selected_qs=[0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95]))
    ok(f"8 quantiles: {len(fig.data)} traces")
except Exception as e: fail("sc 8 qs", e)

# With lots
try:
    fig = build_supercharge_figure(M, sc_params(
        use_lots=True, lots=LOTS_25, stack=TOTAL_BTC))
    ok(f"25 lots: {len(fig.data)} traces")
except Exception as e: fail("sc 25 lots", e)

# Mode B + lots
try:
    fig = build_supercharge_figure(M, sc_params(
        mode="B", use_lots=True, lots=LOTS_25, stack=TOTAL_BTC))
    ok(f"Mode B + 25 lots: {len(fig.data)} traces")
except Exception as e: fail("sc Mode B + lots", e)

# Small withdrawal
try:
    fig = build_supercharge_figure(M, sc_params(wd=100))
    ok("wd=$100/yr")
except Exception as e: fail("sc wd=100", e)

# Large withdrawal (fast depletion)
try:
    fig = build_supercharge_figure(M, sc_params(wd=10000000, stack=0.1))
    ok("wd=$10M, stack=0.1 (fast depletion)")
except Exception as e: fail("sc fast depletion", e)

# Toggles off
try:
    fig = build_supercharge_figure(M, sc_params(annotate=False, log_y=False, show_legend=False))
    ok("all toggles off")
except Exception as e: fail("sc toggles off", e)

# ══════════════════════════════════════════════════════════════════════════════
# 8. LRU CACHE QUANTIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 8. LRU Cache Quantization ===")

def _q3(x):
    if x is None or x == 0:
        return x
    exp = math.floor(math.log10(abs(x)))
    factor = 10 ** (exp - 2)
    return round(x / factor) * factor

# Across price magnitudes
for price, expected in [(0.06, 0.06), (1.5, 1.5), (100, 100), (9850, 9850),
                         (43567, 43600), (95437, 95400), (123456, 123000)]:
    try:
        result = _q3(price)
        assert abs(result - expected) < 1e-10, f"_q3({price}) = {result}, expected {expected}"
        ok(f"_q3({price}) = {result}")
    except Exception as e: fail(f"_q3({price})", e)

# Safe quantile range
for q in [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]:
    try:
        result = _q3(q)
        # Verify the quantized value is close to original
        assert abs(result - q) / q < 0.01, f"_q3({q}) = {result}, too far"
        ok(f"_q3(Q{q*100}%) = {result}")
    except Exception as e: fail(f"_q3 quantile {q}", e)

# ══════════════════════════════════════════════════════════════════════════════
# 9. WATERMARK
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 9. Watermark ===")
try:
    fig = build_bubble_figure(M, bubble_params(selected_qs=[0.50]))
    images = fig.layout.images or []
    annotations = [a for a in (fig.layout.annotations or []) if hasattr(a, 'text') and 'quantoshi' in (a.text or '').lower()]
    assert len(images) >= 1, "No watermark image"
    assert len(annotations) >= 1, "No quantoshi.xyz text annotation"
    ok(f"watermark present: {len(images)} image(s), {len(annotations)} text annotation(s)")
except Exception as e: fail("watermark", e)

# Verify watermark on all tabs
for name, builder, params in [
    ("heatmap", build_heatmap_figure, heatmap_params()),
    ("dca", build_dca_figure, dca_params()),
    ("retire", build_retire_figure, retire_params()),
    ("sc", build_supercharge_figure, sc_params()),
]:
    try:
        fig = builder(M, params)
        images = fig.layout.images or []
        assert len(images) >= 1, f"No watermark on {name}"
        ok(f"watermark on {name}")
    except Exception as e: fail(f"watermark {name}", e)

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
total = passed + len(errors)
print(f"RESULTS: {passed}/{total} passed, {len(errors)} failed")
if errors:
    print(f"\nFAILURES:")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED ✓")
