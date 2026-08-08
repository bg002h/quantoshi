#!/usr/bin/env python3
"""Render the two Saturating Power Law (spl) figures for the Model Info card.

    btc_venv/bin/python3 tools/render_spl_figure.py

Writes two PNGs into btc_web/assets/:

  spl_linear_vs_log.png  Log-log price history with the power law, SatPL and
                         the linear-time logistic overlaid. Saturation on a
                         calendar timescale collapses onto the highest price
                         ever observed; saturation in log-time does not.

  spl_profile.png        The best SSE attainable with the ceiling L held
                         fixed, against L. Marks each window's fitted
                         optimum and the $1,000T fitting bound, and shades
                         the ceilings that window's data cannot tell apart
                         from its own best fit at the 5% boundary-corrected
                         likelihood-ratio threshold (2.7055).

DEV-ONLY TOOL. matplotlib is not in btc_web/requirements.txt. Importing it
anywhere under btc_web/ passes every local test and then breaks prod startup.
These PNGs are pre-rendered and committed; nothing generates them at request
time.

PINNED, LIKE THE REST OF THE CARD
---------------------------------
Both figures are fixed to the two data windows the Model Info card argues
about — WINDOW (2026-08-06, primary) and PRIOR_WINDOW (2026-06-03) — not to
"whatever the CSV holds today". The card's thesis is a comparison of two named
snapshots, and a figure that quietly re-rendered on newer prices would turn
"the later window" into "the day someone last ran a tool", dissolving the
comparison (spec section 7.1a).

So re-running this tool is idempotent while WINDOW is unchanged: new price
rows are truncated away. Advancing WINDOW *does* change both figures, and that
is a deliberate card-refresh act, never a side effect of the daily price
update. Refreshing the card means, in one edit: bump WINDOW here, re-run this
tool, re-run tools/analyze_spl.py, and update _SPL_WINDOW_LABEL, the table
values and the prose dates in btc_web/layout/model_info/_helpers.py and
_items.py. That procedure is documented at _SPL_WINDOW_LABEL in
btc_web/layout/model_info/_helpers.py; this tool is one step of it.

The fitting and the profile arithmetic come from tools/analyze_spl.py — the
same optimiser, bounds and mask that produced every number on the card. There
is deliberately no second optimiser here.
"""
from __future__ import annotations

import os
import pathlib
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402
from scipy.optimize import minimize                # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "btc_web"))
os.chdir(str(ROOT))                       # analyze_spl.load() reads a relative CSV

from analyze_spl import (                 # noqa: E402
    BETA_MAX, CAP_FIT_USD, CRIT, GENESIS, SUPPLY, T0_HI, T0_LO,
    fit_lin, fit_spl, lin_log10, load, spl_log10,
)
from colors import (                      # noqa: E402
    CHART_FONT_LEGEND_LG, CHART_FONT_SUBTITLE, CHART_FONT_TICK_LG,
    CHART_FONT_TITLE_LG, DEFAULT, GRID_MAJOR_COLOR, MUTED_TEXT,
    PLOT_BG_COLOR, SCATTER_POINT, SPINE_COLOR, TEXT_COLOR, TITLE_COLOR,
    TRACE_WIDTH_OVERLAY,
)

ASSETS = ROOT / "btc_web" / "assets"

# The two pinned windows. WINDOW is the card's primary one; PRIOR_WINDOW is
# the earlier column of the card's side-by-side table. Changing either is a
# card refresh — see the module docstring.
WINDOW = "2026-08-06"
PRIOR_WINDOW = "2026-06-03"
WINDOW_LABEL = "6 Aug 2026"          # must match _SPL_WINDOW_LABEL in _helpers.py
PRIOR_LABEL = "3 Jun 2026"

MC = DEFAULT["model_colors"]
C_PL, C_SPL = MC["pl"], MC["spl"]
# The linear-time counterpart IS the ordinary logistic in calendar years --
# the existing `logi` model. Borrow its trace colour so the figure agrees
# with the chart.
C_LIN = MC["logi"]

DPI = 130
QUANTIZE_COLORS = 128


# ── fitting ───────────────────────────────────────────────────────────────

def _fit_window(cutoff: str) -> dict:
    """Every fit this file plots, for one data window."""
    t, p = load(cutoff)
    lp = np.log10(p)
    n = len(t)

    # Guard against a silently short window: load() truncates at `cutoff`, so
    # a CSV that has not reached it yet would render a different figure under
    # the pinned label. Fail loudly instead.
    last = GENESIS + pd.Timedelta(days=float(t[-1]) * 365.25)
    gap = (pd.Timestamp(cutoff) - last).days
    if gap > 3:
        raise SystemExit(
            f"BitcoinPricesDaily.csv stops {gap} days before the pinned window "
            f"{cutoff} (last row ~{last.date()}). These figures are pinned; "
            f"rendering a short window under the {cutoff} label would be a lie."
        )

    X = np.vstack([np.ones_like(t), np.log10(t)]).T
    c_pl, *_ = np.linalg.lstsq(X, lp, rcond=None)
    sse_pl = float(np.sum((lp - X @ c_pl) ** 2))

    r = fit_spl(t, lp)
    A, beta, lt0 = r.x

    lo_L, hi_L = np.log10(p.max()), np.log10(CAP_FIT_USD / SUPPLY)
    rl = fit_lin(t, lp)          # analyze_spl section [8]'s own fit, not a copy

    return dict(cutoff=cutoff, t=t, p=p, lp=lp, n=n,
                c_pl=c_pl, sse_pl=sse_pl,
                spl=r.x, sse_spl=float(r.fun),
                l10L=A + beta * lt0, t0=10 ** lt0, beta=beta,
                lin=rl.x, sse_lin=float(rl.fun),
                lo_L=lo_L, hi_L=hi_L)


def _profile(w: dict, grid: np.ndarray) -> np.ndarray:
    """Best SSE with log10(L) held at each grid point, beta and t0 free.

    A and beta are well determined, so fixing L pins t0 through
    A = log10 L - beta*log10 t0 -- this is the same profile the spec's
    section [4] table reports, expressed in the quantity a reader cares
    about (the ceiling) rather than in t0.
    """
    t, lp = w["t"], w["lp"]
    beta0, lt00 = w["spl"][1], w["spl"][2]
    lo_lt0, hi_lt0 = np.log10(T0_LO), np.log10(T0_HI)

    def sse_at(target: float) -> float:
        def f(z):
            b, l = z
            if not (0.01 <= b <= BETA_MAX and lo_lt0 <= l <= hi_lt0):
                return 1e9
            return float(np.sum((lp - spl_log10(t, target - b * l, b, l)) ** 2))

        # Two seeds: the global optimum, and the t0 that reproduces the plain
        # power law at this L. Nelder-Mead from one seed alone drifts on the
        # flat branch.
        seeds = [(beta0, lt00),
                 (beta0, np.clip((target - w["c_pl"][0]) / beta0, lo_lt0, hi_lt0))]
        best = None
        for s in seeds:
            res = minimize(f, s, method="Nelder-Mead",
                           options={"xatol": 1e-11, "fatol": 1e-13,
                                    "maxiter": 40000, "maxfev": 40000})
            if best is None or res.fun < best:
                best = float(res.fun)
        return best

    return np.array([sse_at(g) for g in grid])


def _band(grid: np.ndarray, stat: np.ndarray, i: int) -> tuple[float, float, bool]:
    """Contiguous run around the optimum where the LRT statistic <= CRIT.

    `i` is the index of the optimum the band grows out from, and the caller
    must have found it INSIDE the fitting bound -- see _optimum_index.

    Returns (lo, hi, open_ended) in log10 L, with edges linearly interpolated.
    `open_ended` means the run reaches the right edge of the grid -- the data
    do not bound the ceiling above at all.
    """
    lo_i = i
    while lo_i > 0 and stat[lo_i - 1] <= CRIT:
        lo_i -= 1
    hi_i = i
    while hi_i < len(stat) - 1 and stat[hi_i + 1] <= CRIT:
        hi_i += 1

    def cross(a: int, b: int) -> float:
        """log10 L where stat crosses CRIT between grid points a and b."""
        f = (CRIT - stat[a]) / (stat[b] - stat[a])
        return float(grid[a] + f * (grid[b] - grid[a]))

    lo = cross(lo_i, lo_i - 1) if lo_i > 0 else float(grid[0])
    open_ended = hi_i == len(stat) - 1
    hi = float(grid[-1]) if open_ended else cross(hi_i, hi_i + 1)
    return lo, hi, open_ended


def _optimum_index(prof: np.ndarray, inside: np.ndarray) -> int:
    """Index of the best SSE among ceilings the fit is actually allowed.

    The grid deliberately runs half a decade past the $1,000T bound so the
    figure can show the profile staying flat out there, but that stretch is
    outside the model's parameter space. Taking the minimum over the whole
    grid would let a future window -- one whose profile still fell past the
    bound -- anchor the optimum dot, the threshold and the shaded band to a
    ceiling the fit could never return. It would look right and be wrong.
    """
    return int(np.argmin(np.where(inside, prof, np.inf)))


# ── shared styling ────────────────────────────────────────────────────────

def _new_fig(size):
    fig, ax = plt.subplots(figsize=size, dpi=DPI)
    fig.patch.set_facecolor(PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)
    ax.grid(True, which="major", color=GRID_MAJOR_COLOR, linewidth=0.9, zorder=0)
    ax.tick_params(colors=TEXT_COLOR, labelsize=CHART_FONT_TICK_LG)
    for s in ax.spines.values():
        s.set_color(SPINE_COLOR)
    return fig, ax


def _titles(ax, title, subtitle):
    ax.set_title(title, color=TITLE_COLOR, fontsize=CHART_FONT_TITLE_LG,
                 fontweight="bold", pad=26, loc="left")
    ax.annotate(subtitle, xy=(0, 1.015), xycoords="axes fraction",
                color=MUTED_TEXT, fontsize=CHART_FONT_SUBTITLE, va="bottom")


def _footnote(fig, text, width=137):
    wrapped = "\n".join(textwrap.fill(para, width) for para in text.split("\n"))
    fig.text(0.008, 0.008, wrapped, color=MUTED_TEXT,
             fontsize=CHART_FONT_SUBTITLE - 2, va="bottom", ha="left")


def _save(fig, path):
    fig.savefig(path, facecolor=fig.get_facecolor(), dpi=DPI)
    plt.close(fig)
    try:
        from PIL import Image
        Image.open(path).convert("RGB").quantize(colors=QUANTIZE_COLORS).save(
            path, "PNG", optimize=True)
    except Exception as e:                                # pragma: no cover
        print(f"    (quantize skipped: {type(e).__name__}: {e})")
    print(f"  {path.name}: {path.stat().st_size // 1024} KB")


def _usd(v: float) -> str:
    # matplotlib reads a $...$ pair as mathtext and italicises what is between
    # them, so every dollar sign that reaches a text artist is escaped.
    return rf"\${v:,.0f}" if v >= 1 else rf"\${v:,.2f}"


def _usd_T(v: float) -> str:
    return rf"\${v:,.0f}T"


# ── figure 1: saturation in log-time vs calendar time ─────────────────────

def render_linear_vs_log(w: dict) -> None:
    t, p, n = w["t"], w["p"], w["n"]
    fig, ax = _new_fig((11.0, 6.6))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(t, p, s=2.0, color=SCATTER_POINT, alpha=0.42, linewidths=0,
               zorder=2, label="Bitcoin daily close")

    tg = np.geomspace(t.min(), t.max(), 900)
    a, b = w["c_pl"]
    rmse = lambda sse: np.sqrt(sse / n)                        # noqa: E731

    ax.plot(tg, 10 ** (a + b * np.log10(tg)), color=C_PL,
            linewidth=TRACE_WIDTH_OVERLAY, zorder=4,
            label=f"Power law  —  RMSE {rmse(w['sse_pl']):.4f}")
    ax.plot(tg, 10 ** spl_log10(tg, *w["spl"]), color=C_SPL,
            linewidth=TRACE_WIDTH_OVERLAY + 0.4, linestyle=(0, (7, 2)), zorder=5,
            label=f"SatPL — saturating in log-time  —  RMSE {rmse(w['sse_spl']):.4f}")
    ax.plot(tg, 10 ** lin_log10(tg, *w["lin"]), color=C_LIN,
            linewidth=TRACE_WIDTH_OVERLAY + 0.4, linestyle=(0, (2, 2)), zorder=5,
            label=f"Logistic in calendar time  —  RMSE {rmse(w['sse_lin']):.4f}")

    ceiling = 10 ** w["lin"][0]
    ax.set_ylim(0.035, 8e5)
    ax.axhline(ceiling, color=C_LIN, linewidth=0.9, linestyle=":", alpha=0.6, zorder=3)
    ax.annotate(
        f"the calendar-time fit flattens onto the highest price ever\n"
        f"observed, {_usd(ceiling)} — its own lower bound on L",
        xy=(11.0, ceiling), xytext=(1.28, 2.6e5),
        color=C_LIN, fontsize=CHART_FONT_SUBTITLE, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=C_LIN, linewidth=1.1,
                        connectionstyle="arc3,rad=-0.12"), zorder=6)

    ax.set_xlabel("Years since the 2009-07-25 origin  (log scale)", color=TEXT_COLOR,
                  fontsize=CHART_FONT_TICK_LG)
    ax.set_ylabel("Price, USD  (log scale)", color=TEXT_COLOR,
                  fontsize=CHART_FONT_TICK_LG)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 8, 12, 17]))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_minor_formatter(NullFormatter())

    leg = ax.legend(loc="lower right", fontsize=CHART_FONT_LEGEND_LG,
                    facecolor=PLOT_BG_COLOR, edgecolor=SPINE_COLOR, framealpha=0.94)
    for txt in leg.get_texts():
        txt.set_color(TEXT_COLOR)

    _titles(ax, "Saturation in log-time, not in calendar time",
            f"Same data, same bounds: an S-curve in ln t is indistinguishable "
            f"from the power law; an S-curve in t is not")
    _footnote(fig, f"Fits on data through {WINDOW_LABEL} (n = {n:,}), t ≥ 1 yr from "
                   f"the 2009-07-25 origin. Pinned snapshot — it does not track "
                   f"later prices.")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.875, bottom=0.155)
    _save(fig, ASSETS / "spl_linear_vs_log.png")


# ── figure 2: the profile of the ceiling ──────────────────────────────────

def render_profile(windows: list[dict]) -> None:
    """SSE against the ceiling, for both pinned windows.

    Two curves rather than one because the identification problem and the
    instability are the same fact seen twice: on the 3 Jun window the profile
    is flat from its optimum all the way out to the fitting bound -- the data
    cannot exclude "no ceiling at all" -- and nine weeks later, on 1.1% more
    data, the same computation carves out a narrow band instead. Neither
    picture is the ceiling; the pair is the finding.
    """
    fig, ax = _new_fig((11.0, 6.9))
    ax.set_xscale("log")

    lo_L = min(w["lo_L"] for w in windows)
    hi_L = max(w["hi_L"] for w in windows)
    grid = np.linspace(lo_L, hi_L + 0.5, 96)          # 0.5 dex past the bound
    cap_T = 10 ** grid * SUPPLY / 1e12                # ceiling as market cap, $T
    bound_T = CAP_FIT_USD / 1e12
    floor_T = 10 ** lo_L * SUPPLY / 1e12
    ymax = 2.6
    inside = grid <= np.log10(CAP_FIT_USD / SUPPLY)

    styles = [dict(color=MUTED_TEXT, lw=TRACE_WIDTH_OVERLAY - 0.2, ls=(0, (6, 2))),
              dict(color=C_SPL, lw=TRACE_WIDTH_OVERLAY + 0.6, ls="-")]
    summary = []

    # Everything past the $1000T bound is outside the fit's own parameter
    # space; the profile is continued there dotted only to show it stays flat.
    ax.axvspan(bound_T, cap_T[-1], color=SPINE_COLOR, alpha=0.16, lw=0, zorder=0)

    for w, st in zip(windows, styles):
        prof = _profile(w, grid)
        i_opt = _optimum_index(prof, inside)
        smin = prof[i_opt]
        dsse = prof - smin
        stat = w["n"] * np.log(prof / smin)
        thr = smin * (np.exp(CRIT / w["n"]) - 1)      # ΔSSE at the 5% boundary
        blo, bhi, open_ended = _band(grid, stat, i_opt)
        label = PRIOR_LABEL if w["cutoff"] == PRIOR_WINDOW else WINDOW_LABEL
        blo_T, bhi_T = 10 ** blo * SUPPLY / 1e12, 10 ** bhi * SUPPLY / 1e12
        opt_T = 10 ** w["l10L"] * SUPPLY / 1e12

        # The shaded band: every ceiling whose best fit sits under the 5%
        # boundary — the ceilings this window cannot tell apart.
        ax.fill_between(cap_T, dsse, thr, where=(dsse <= thr),
                        color=st["color"], alpha=0.28, lw=0, zorder=2)
        ax.plot(cap_T[inside], dsse[inside], color=st["color"], linewidth=st["lw"],
                linestyle=st["ls"], zorder=5,
                label=(f"data through {label}   (n = {w['n']:,})\n"
                       f"best fit {_usd_T(opt_T)}   ·   cannot distinguish "
                       f"{_usd_T(blo_T)}"
                       + (" and up" if open_ended else f" – {_usd_T(bhi_T)}")))
        ax.plot(cap_T[~inside], dsse[~inside], color=st["color"],
                linewidth=st["lw"], linestyle=":", alpha=0.55, zorder=4)
        ax.plot([opt_T], [np.interp(w["l10L"], grid, dsse)], "o", color=st["color"],
                markersize=9, markeredgecolor=PLOT_BG_COLOR, markeredgewidth=1.5,
                zorder=7)
        span = (f"{_usd_T(blo_T)} and up (no upper bound)" if open_ended
                else f"{_usd_T(blo_T)} – {_usd_T(bhi_T)}")
        summary.append(dict(label=label, opt_T=opt_T, span=span, n=w["n"],
                            smin=smin, thr=thr, stat_bound=stat[inside][-1],
                            dsse_bound=dsse[inside][-1], open_ended=open_ended))

    # The 5% boundary. The two windows' thresholds differ in the fourth
    # decimal, so one line is drawn, at the primary window's value.
    prim = summary[-1]
    ax.axhline(prim["thr"], color=TEXT_COLOR, linewidth=1.0, linestyle=(0, (4, 3)),
               alpha=0.8, zorder=6)
    ax.annotate(f"5% likelihood-ratio boundary   (2.7055  →  ΔSSE {prim['thr']:.3f})",
                xy=(floor_T * 1.75, prim["thr"]), xytext=(0, 7),
                textcoords="offset points", color=TEXT_COLOR,
                fontsize=CHART_FONT_SUBTITLE, va="bottom")

    for xv, txt, ha, dx in ((floor_T, "highest observed price", "left", 7),
                            (bound_T, r"\$1,000T fitting bound", "right", -7)):
        ax.axvline(xv, color=TEXT_COLOR, linewidth=1.1, alpha=0.5, zorder=6)
        ax.annotate(txt, xy=(xv, 0.035), xytext=(dx, 0), textcoords="offset points",
                    rotation=90, ha=ha, va="bottom", color=TEXT_COLOR,
                    fontsize=CHART_FONT_SUBTITLE - 1, alpha=0.85)

    # The vertical scale, in units a reader can weigh.
    n_p = windows[-1]["n"]
    ax.annotate(f"RMSE {np.sqrt(prim['smin'] / n_p):.4f}",
                xy=(summary[-1]["opt_T"], 0), xytext=(0, -7),
                textcoords="offset points", ha="center", va="top", color=C_SPL,
                fontsize=CHART_FONT_SUBTITLE, fontweight="bold", zorder=8)
    ax.annotate(f"RMSE {np.sqrt((prim['smin'] + prim['dsse_bound']) / n_p):.4f}",
                xy=(bound_T / 2.6, prim["dsse_bound"]), xytext=(0, 9),
                textcoords="offset points", ha="center", va="bottom", color=C_SPL,
                fontsize=CHART_FONT_SUBTITLE, fontweight="bold", zorder=8)

    ax.set_xlim(floor_T / 1.3, cap_T[-1])
    ax.set_ylim(-0.16, ymax)
    ax.set_xlabel("Ceiling L, as total market cap  (log scale)", color=TEXT_COLOR,
                  fontsize=CHART_FONT_TICK_LG)
    ax.set_ylabel("ΔSSE vs that window’s own best fit", color=TEXT_COLOR,
                  fontsize=CHART_FONT_TICK_LG)
    ax.xaxis.set_major_locator(FixedLocator([3, 10, 30, 100, 300, 1000, 3000]))
    ax.xaxis.set_major_formatter(lambda v, _: rf"\${v:,.0f}T")
    ax.xaxis.set_minor_formatter(NullFormatter())

    leg = ax.legend(loc="upper right", fontsize=CHART_FONT_LEGEND_LG,
                    facecolor=PLOT_BG_COLOR, edgecolor=SPINE_COLOR,
                    framealpha=0.94, labelspacing=0.9, borderpad=0.8)
    for txt in leg.get_texts():
        txt.set_color(TEXT_COLOR)

    _titles(ax, "Nothing in the data pins the ceiling down",
            "Best SSE attainable with the ceiling held fixed — steep below the "
            "optimum, all but flat above it")
    _footnote(fig,
              "Shaded: the ceilings each window cannot tell apart from its own best "
              "fit, at the 5% boundary-corrected likelihood-ratio bound. On the June "
              "window it runs from the optimum out past the fitting bound — L → ∞ is "
              "exactly the plain power law, so those data ask for no ceiling.\n"
              "That threshold assumes independent residuals, which these are not "
              "(30-day residual autocorrelation 0.90), so the true indistinguishable "
              f"set is wider still. Top to bottom this panel spans RMSE "
              f"{np.sqrt(prim['smin'] / n_p):.4f} to "
              f"{np.sqrt((prim['smin'] + ymax) / n_p):.4f}. Pinned snapshots.")
    fig.subplots_adjust(left=0.075, right=0.972, top=0.875, bottom=0.225)
    _save(fig, ASSETS / "spl_profile.png")

    print("\n  profile summary (regenerate the card's section [4] table alongside):")
    for s in summary:
        print(f"    {s['label']:>12}  n={s['n']:,}  best {s['opt_T']:,.1f}T  "
              f"band {s['span'].replace(chr(92), '')}   ΔSSE/LRT at the $1,000T bound "
              f"{s['dsse_bound']:+.4f}/{s['stat_bound']:.2f} vs crit {CRIT:.4f}")


def main() -> None:
    print(f"Rendering SatPL figures — pinned to {WINDOW} "
          f"(prior window {PRIOR_WINDOW})...")
    prior = _fit_window(PRIOR_WINDOW)
    cur = _fit_window(WINDOW)
    for w in (prior, cur):
        print(f"  {w['cutoff']}: n={w['n']:,}  beta={w['beta']:.4f}  "
              f"t0={w['t0']:.2f} yr  L=${10 ** w['l10L'] * SUPPLY / 1e12:,.1f}T  "
              f"RMSE spl {np.sqrt(w['sse_spl'] / w['n']):.4f} / "
              f"pl {np.sqrt(w['sse_pl'] / w['n']):.4f} / "
              f"lin {np.sqrt(w['sse_lin'] / w['n']):.4f}")
    render_linear_vs_log(cur)
    render_profile([prior, cur])
    print("Done.")


if __name__ == "__main__":
    main()
