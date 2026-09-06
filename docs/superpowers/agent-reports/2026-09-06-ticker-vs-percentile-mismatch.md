# Navbar ticker vs Tab-1 Percentile view — percentile mismatch

**Date:** 2026-09-06
**Mode:** RECON ONLY — nothing implemented, no application code edited, no commits.
**Base:** `/scratch/code/bitcoinprojections`, `master` (the `r2-no-op-bumps` worktree was not touched).
**Measured with:** `btc_venv/bin/python3`, in-process, real `model_data.pkl`, `_HAS_RESQR=True` (91 resqr bundles bound).

> **Note on this file's location.** The dispatch asked for
> `docs/superpowers/agent-reports/2026-09-06-ticker-vs-percentile-mismatch.md` in the main checkout. This agent was
> hard-isolated to the `r2-no-op-bumps` worktree by the harness (`Write` to the shared checkout is refused;
> `ExitWorktree` is unavailable to a cwd-overridden subagent), and the brief fenced the worktree off. So the report is
> persisted here instead. To land it at the intended path:
> `cp <this file> /scratch/code/bitcoinprojections/docs/superpowers/agent-reports/2026-09-06-ticker-vs-percentile-mismatch.md`

---

## Verdict in one line

**Hypothesis 3 is the defect: the ticker calls `find_percentile` without a `sigma_mode`, so it silently takes the
`"constant"` default, while every Tab-1 view (including `/1.4`) runs at the app default `sigma_mode="resqr"`.**
For the Bubble Model that is worth **19.73 percentile points** today. Hypotheses 1 and 2 (live price, later `t`) are
real but explain only **−0.42 pp** of the 19.31 pp gap — about **2 %**. Hypotheses 4 and 5 are **not** implicated:
the two lookups are byte-identical in behaviour (measured max difference `0.000e+00` over 5,154 dates × 3 models × 2 σ modes).

---

## 1. The measured discrepancy

Environment at measurement time:

| quantity | value |
|---|---|
| today (`today_t`) | `t = 17.117043` (2026-09-06) |
| last CSV close | 2026-09-03, **$81,142.27**, `t = 17.108830` |
| lag | **3.0 days** |
| live price (`_fetch_btc_price`, succeeded) | **$80,260** |
| Tab-1 default `sigma_mode` | `resqr` (`tab_defaults.py:88`, `snapshot_defaults.py:74`) |
| Tab-1 default `active_models` | `["bub"]` — so `/1.4` draws **one** line, BM |
| ticker default model | `qr` (`_MODEL_CYCLE[0]`); **BM is one tap away** |

### Bubble Model — the reported symptom

```
navbar (tap once to BM)   'BM99%'    raw 98.618 %
/1.4 right-hand edge                 raw 79.308 %
                          -------------------------
gap                                       19.31 pp
```

Decomposition, one variable at a time (all via `BubbleModel.find_percentile`):

| step | t | price | sigma_mode | percentile | Δ from previous |
|---|---|---|---|---|---|
| **ticker as shipped** | today | live 80,260 | `constant` | **98.618 %** | — |
| A | today | live 80,260 | `resqr` | 78.891 % | **−19.727 pp ← the defect** |
| B | today | last 81,142 | `resqr` | 79.652 % | +0.761 pp (legitimate: price) |
| **chart as shipped** | t_last | last 81,142 | `resqr` | **79.308 %** | −0.344 pp (legitimate: time) |

So of a 19.31 pp gap, **19.73 pp is the σ-mode defect** and **−0.42 pp net is the legitimate lag**.
Had the ticker used `resqr`, it would print **`BM79%`** against a chart edge of **79.3 %** — a residual of 0.42 pp,
which rounds away entirely.

### All ten models in `_MODEL_CYCLE`

Gap decomposition in percentile points; `sigma` is the defect column, `price`+`time` are legitimate.

```
model         ticker   chart   TOTAL   sigma   price    time   has _resqr?
qr             21.19   23.54   -2.35    0.00   -1.91   -0.44   False
bub            98.62   79.31   19.31   19.73   -0.76   +0.34   True
pl             27.67   22.27    5.40    6.48   -0.88   -0.20   True
spl            32.65   28.85    3.80    4.74   -0.79   -0.15   True
lp3            65.53   66.27   -0.75    0.00   -1.07   +0.33   False
cfg_1d_1u      73.67   76.36   -2.69   -1.55   -1.26   +0.12   True
ecfg_1d_1u     48.87   50.33   -1.46   -0.12   -1.08   -0.26   True
pca            96.21   93.61    2.60    3.04   -0.42   -0.02   True
grdy           99.98  100.00   -0.02   -0.02    0.00    0.00   True
ef             90.80   92.53   -1.73    0.00   -1.41   -0.32   False
```

What the navbar actually prints (integer-rounded) vs the chart's right-hand edge:

```
qr           21%  vs 23.5%      bub          99%  vs 79.3%   <-- reported
pl           28%  vs 22.3%      spl          33%  vs 28.9%
lp3          66%  vs 66.3%      cfg_1d_1u    74%  vs 76.4%
ecfg_1d_1u   49%  vs 50.3%      pca          96%  vs 93.6%
grdy        100%  vs 100.0%     ef           91%  vs 92.5%
```

The legitimate price+time component never exceeds ~2 pp for any model. Everything beyond that is σ-mode.

---

## 2. The mechanism, named to file and line

### The two paths, side by side

**Ticker** — `btc_web/callbacks/ticker.py:64` (and `:101` for U₁):

```python
p = mdl.find_percentile(t, price)          # <-- no sigma_mode argument
```

**Percentile view** — `btc_web/figures/percentile.py:121-122`, fed by
`btc_web/callbacks/charts/__init__.py:828`:

```python
pct = _percentile_series(mdl, t_data, px_data,
                         sigma_mode=p.get("sigma_mode", "constant"))
# ... where p["sigma_mode"] = sigma_mode or "constant", and the
#     bub-sigma-mode RadioItems defaults to "resqr"
```

The signature that closes the loop — `btc_core/_base.py:66`, `:144`, `:240`, `btc_core/_simple.py:704`:

```python
def find_percentile(self, t, price, sigma_mode="constant"):
```

**The default is `"constant"`, but the application's default is `"resqr"`.** Every caller that forgets the keyword
silently gets the legacy band model. `btc_web/callbacks/scanner.py:179` — which sits *on the same tab* — remembers it
(`State("bub-sigma-mode", "value")`, `scanner.py:80`). The ticker does not.

### Why 19.7 pp for BM specifically

`_CompositeModel._sigma_at` (`btc_core/_base.py:181-187`) is a *shrinking* σ, `σ(t) = σ₀·t^(−α)`, with
`σ₀_up=0.08574, α_up=0.13824, σ₀_down=0.07425, α_down=0.22215`. At today's `t=17.12` that band has collapsed:

```
t        sig_up   sig_dn    Q1..Q99 width: constant   resqr
 2.00    0.0779   0.0637                     2.13x   10.27x
 5.00    0.0686   0.0519                     1.91x    2.98x
10.00    0.0624   0.0445                     1.77x    4.68x
17.12    0.0579   0.0395                     1.69x    3.29x
```

At `t=today` the constant-σ BM fan spans Q1 $48,047 → Q99 $80,959 — a **1.69× total width**. The live price of
$80,260 lands between **Q95 ($73,928) and Q99 ($80,959)** → 98.6 %. The resqr fan spans **3.29×**, and the same
$80,260 lands between **Q75 ($75,901) and Q80 ($81,548)** → 78.9 %.

This is not a today-only artefact. Over the full plotted history (n = 5,154 days, `t ≥ 3`), the BM
constant-vs-resqr percentile difference is:

```
mean 11.70 pp   median 11.63 pp   p95 23.23 pp   max 26.82 pp
|delta| >  1 pp : 95.4 % of days
|delta| >  5 pp : 77.8 % of days
|delta| > 10 pp : 56.9 % of days
|delta| > 20 pp : 15.1 % of days
```

Constant-σ also pins the BM percentile to a fan edge on **6.3 %** of days vs **2.1 %** under resqr — i.e. the mode the
ticker is using is the more degenerate one at present-day `t`, which is exactly why `resqr` became the default.

---

## 3. Hypotheses priced out

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | Different price input (live vs last CSV close) | **Legitimate, tiny.** | BM: +0.76 pp. Across all 10 models: −1.91 … +0.00 pp. |
| 2 | Different `t` (today vs last data date) | **Legitimate, tiny.** | 3.0-day lag → BM −0.34 pp. All models: −0.44 … +0.34 pp. |
| 3 | **Different σ / band construction** | **THE DEFECT.** | ticker `constant` (implicit default) vs chart `resqr`. BM 19.73 pp; PL 6.48; SPL 4.74; PCA 3.04; HybPPL −1.55. |
| 4 | Non-monotonic fan lookup (`np.interp` vs bracket scan) | **NOT implicated — both are correct.** | `_bracket_percentile` (`percentile.py:35`) is a faithful mirror of `find_percentile`'s scan. Measured max abs difference **0.000e+00** across 10 models × 2 σ modes, and across the full 5,154-day history for `bub`, `qr`, `pl`. The 2026-08-23 review fix holds. |
| 5 | Composite vs Q50 handling | **NOT implicated.** | Both paths call the *same* `mdl.price_at(q, t, sigma_mode=…)` over the *same* `mdl.quantiles`; neither special-cases the composite. The BM Q50 suppression is a *drawing* decision in `figures/bubble.py`, and does not reach either percentile path. |

Two secondary observations, neither the cause:

- Ticker rounding is `round(p*100)` → integer. Chart hover is `%{y:.1f}%`, and `_round_trace_data` keeps 2 decimals
  (79.30812345 → 79.30). Sub-0.5 pp of apparent disagreement is rounding, no more.
- The ticker's *default* model is `qr`, not `bub` (`_MODEL_CYCLE[0]`). QR carries no resqr bundle, so its 2.35 pp gap
  is 100 % legitimate lag. The operator must have tapped once to reach BM — which is where the defect is worst.

---

## 4. Which models are affected

`sigma_mode` is a no-op for any model with no `_resqr` bundle (`_resqr_price_at` returns `None` → constant fallback,
`btc_core/_helpers.py:99-101`). In `_MODEL_CYCLE`:

- **No bundle → unaffected by the defect:** `qr`, `lp3`, `ef` (LPPL family is deliberately excluded from resqr —
  see the disclaimer in `layout/display_models.py:271-276`). Their gaps are pure lag.
- **Has bundle → affected:** `bub` (19.73 pp), `pl` (6.48), `spl` (4.74), `pca` (3.04), `cfg_1d_1u` (−1.55),
  `ecfg_1d_1u` (−0.12), `grdy` (−0.02).

So **7 of 10 cycle models are wrong to some degree, and BM — the flagship and the one the operator noticed — is wrong
by an order of magnitude more than the rest.**

### Same bug, other call sites (not asked for, but same class)

`grep -rn "find_percentile"` over `btc_web/` shows four more callers that omit `sigma_mode` and therefore get
`constant`:

| file:line | surface | note |
|---|---|---|
| `btc_web/callbacks/lots.py:69` | Stack Tracker — `pct_q` stamped on a newly added lot | persisted into browser `localStorage` |
| `btc_web/callbacks/lots.py:114` | Stack Tracker — `pct_q` recomputed on JSON import | same |
| `btc_web/figures/leverage.py:100` | Tab 7 Max Pay-Price (non-`bub` models) | `bub` takes a hand-rolled `norm.cdf` branch above it, also constant-σ |
| `btc_web/markov.py:68` | `_prices_to_percentiles` — MC transition-matrix binning | **do not change** — see blast radius |

Only `btc_web/callbacks/scanner.py:179` passes it correctly.

Note the σ-radio's own disclaimer (`layout/display_models.py:265-268`): *"Applies to Tab 1 … only — other tabs use the
legacy constant-σ bands regardless of this setting."* That makes `lots.py` and `leverage.py` **documented behaviour,
not defects.** The navbar ticker is the odd one out: it is not a tab, it renders directly above the Tab-1 chart, and
nothing tells the user it is reading a different band model.

---

## 5. Proposed fix

### Recommended — Option A: pin the ticker to the deployment's band model

`btc_web/callbacks/ticker.py`, two call sites (`:64` and `:101`):

```python
_sm = "resqr" if getattr(_app_ctx, "_HAS_RESQR", False) else "constant"
...
p = mdl.find_percentile(t, price, sigma_mode=_sm)
```

Why this shape:

- It matches what `/1` and `/1.4` actually draw under defaults (`tab_defaults.py:88` = `resqr`).
- It **degrades safely**. On a deployment with no bundles `_HAS_RESQR` is False → `constant`, and even with the flag
  True, a model with no bundle (`qr`, `lp3`, `ef`) falls through to constant inside `price_at`. No new failure mode.
- It requires **no new callback Input**, so it cannot regress the lazy-tab surface (below).

**Blast radius: small and bounded.** It changes only what the navbar prints. It does not touch `find_percentile`
itself, so no other caller moves. Measured effect on the ten displayed numbers:

```
qr   21 -> 21    bub  99 -> 79    pl   28 -> 21    spl  33 -> 28    lp3   66 -> 66
cfg  74 -> 75    ecfg 49 -> 49    pca  96 -> 93    grdy 100 -> 100  ef    91 -> 91
```

**It changes what the ticker means, not what the chart means.** The chart is already the app default and stays put.
The honest framing is that the ticker was the stale surface: it is still reading the legacy σ model that the rest of
Tab 1 moved off.

### Rejected — Option B: wire the radio into the ticker

`State("bub-sigma-mode", "value")` on `update_price_ticker` is the obvious idea and it is a trap. Tabs are
lazy-mounted (`layout/__init__.py:252, 830-832`): on `/2`–`/10` the component does not exist, which is exactly the
"nonexistent object" class that memory `feedback_nonexistent_input_perf.md` records as a *real* performance cost, not
a cosmetic warning. Doing it properly means a new always-mounted global `sigma-mode-store` mirrored from the radio
plus a re-trigger of a callback that currently fires once per 20-minute interval. That is a much larger change for a
behaviour that is arguably worse — the navbar number would silently shift while the user is on Tab 6.

### Not recommended — Option C: flip the `find_percentile` default to `"resqr"`

Tempting (it fixes all five forgetful callers at once) and **dangerous**. `btc_web/markov.py:68` uses
`find_percentile` to bin prices for the Markov transition matrix. The ~1.2 GB MC cache and the ~200 MB Citadel band
cache were both built against `constant`-σ bins. Changing that default would silently invalidate both and change paid
MC output. If this route is ever taken, `markov.py:68` must be pinned to `sigma_mode="constant"` explicitly *first*,
in a separate commit.

### Test gap worth closing alongside

`btc_web/test_percentile.py:73-84` (`test_series_matches_find_percentile_qr`) claims in its own docstring to be
*"the consistency guarantee with the navbar ticker"* — but it calls both sides with the **default** `sigma_mode`, so it
verifies algorithm parity while the live UI mismatch sails past. Likewise `figures/percentile.py:42-43` asserts
*"so the oscillator matches the navbar ticker for every model"*, which is currently false for 7 of 10 models. A
regression test should compare **`update_price_ticker`'s actual argument list** against
`tab_defaults.bubble_defaults()["sigma_mode"]`, not two hand-matched calls.

---

## 6. Reproduction

Both numbers, one process, no server needed:

```python
import sys; sys.path[:0] = ["/scratch/code/bitcoinprojections",
                            "/scratch/code/bitcoinprojections/btc_web"]
import numpy as np, _app_ctx, app
from btc_core import today_t
from figures.percentile import _percentile_series

M = _app_ctx.M; bm = _app_ctx.PRICE_MODELS["bub"]
td = today_t(M.genesis)
t_last, px_last = float(M.price_years[-1]), float(M.price_prices[-1])
live = 80260.0

print("ticker:", bm.find_percentile(td, live) * 100)                  # 98.618  (implicit "constant")
print("chart :", _percentile_series(bm, np.array([t_last]),
                                    np.array([px_last]), "resqr")[0]) # 79.308
```

Full measurement scripts used for this report (kept alongside):
`measure.py` (gap table + decomposition), `measure2.py` (lookup parity, fan dump, history stats),
`measure3.py` (σ collapse, edge-pinning, bracket locations) — same scratchpad directory.
