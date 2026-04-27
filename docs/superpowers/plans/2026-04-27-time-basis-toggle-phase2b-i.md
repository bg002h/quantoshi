# Time-Basis Toggle — Phase 2b.i (Scaffold model_data_block.pkl) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a structural scaffold `model_data_block.pkl` via the parameterized build pipeline. Validates that `tools/build_bm_model.py --time-basis=block` runs end-to-end and emits a pkl with the correct schema, axis-aware sigma/QR/BM fits, and the expected sidecar. **LPPL/HybPPL/EPPL/PCA/Greedy family fits will use calendar-fit class attrs and produce garbage predictions** — that's acceptable for the scaffold and will be addressed in Phase 2b.ii (refit families in block mode with bound rescaling).

**Architecture:** Add a `QS_TIME_BASIS` env-var override to `time_basis.py` (takes precedence over `quantoshi.toml`). `tools/build_bm_model.py` sets that env var before any `time_basis` import when `--time-basis=block`. Rescale calendar-baked window constants in `tools/model_toolkit/fitting.py` by `T_PER_YEAR`. Audit other toolkit files for similar baked thresholds. Make `tools/build_bm_model.py` output filenames axis-aware. Run the block build (BM/QR fits cleanly; family models use calendar-fit class attrs and produce garbage predictions — that's expected for the scaffold). Sanity-check. Commit.

**Plan Revision Note (2026-04-27):** This plan was revised after agent review caught a critical Phase 2a coverage gap: Phase 2a wired `year_to_t()` in `fitting.py` but never override the module-global `T_PER_YEAR=1.0` (frozen at TOML read time). Without an env-var override, `tools/build_bm_model.py --time-basis=block` would compute `year_to_t(2017) ≈ 7.44` (calendar units) instead of `391_239` (block units), and `fit_sequential` would crash with empty peak windows. Tasks 1 + 3 + 4 below close this gap.

**Tech Stack:** Python 3.14 (dev). Existing parameterized pipeline from Phase 2a.

**Spec:** [`docs/superpowers/specs/2026-04-26-time-basis-toggle-design.md`](../specs/2026-04-26-time-basis-toggle-design.md) §4 Phase 2.
**Decisions log:** [`docs/superpowers/plans/2026-04-26-decisions-log.md`](2026-04-26-decisions-log.md) (D11 — pivot: block becomes canonical default).

**Branch:** `time-basis-toggle-phase2b` (sub-branch off `time-basis-toggle`). Currently at `645db69` (daily price update appended; nothing else committed on the sub-branch yet).

**Phase 2b sub-decomposition (this plan covers 2b.i only):**
- **2b.i (this plan):** Scaffold pkl. Filename axis-aware, run build, commit. ~30 minutes including build time.
- **2b.ii (future plan):** Family refits with bound rescaling. Likely 2-3 hours of tooling work + multi-hour builds.

---

## File Structure

**Modify:**
- `btc_web/time_basis.py` — accept `QS_TIME_BASIS` env-var override (precedence: env > TOML > defaults).
- `tools/build_bm_model.py` — set `QS_TIME_BASIS` env var before imports; make `pkl_path` and `diag_path` axis-aware.
- `tools/model_toolkit/fitting.py` — rescale window constants (`BUBBLE_YEAR_WINDOW`, `FIT_CONTEXT_YR`, `FIT_RISE_LOOKBACK_YR`, etc.) by `T_PER_YEAR` so they have the right units in block mode.
- Possibly `tools/model_toolkit/{composite,prediction,support,bands,bubble_shape}.py` — pending audit in Task 4.
- `btc_web/test_time_basis_phase2a.py` (continue same file — append regression tests).

**Create (build artifacts, committed):**
- `model_data_block.pkl` — block-axis scaffold.
- `model_data_block_meta.json` — sidecar.
- `model_data_block_resqr_diagnostics.json` — resqr diagnostics for block build.

**Untouched (deliberate Phase 2b.i non-goals):**
- `btc_core/*` — model class attrs stay calendar-fit. LPPL/HybPPL/EPPL family predictions in block pkl will be nonsensical — flagged in commit message.
- `tools/fit_*.py` scripts — no `--time-basis` flag yet (Phase 2b.ii).
- `tools/build_ef_model.py` — EF stays calendar (axis-exempt per spec §2). No block EF artifact.
- Site / web app — calendar mode still canonical; block pkl is a parallel artifact, not loaded.

---

## Task 1: Add `QS_TIME_BASIS` env-var override to `time_basis.py`

**Files:**
- Modify: `btc_web/time_basis.py` (extend `_load_config` to honor an env-var override).
- Modify: `btc_web/test_time_basis_phase2a.py` (append tests for env-var precedence).

**Goal:** Allow `tools/build_bm_model.py --time-basis=block` to set `QS_TIME_BASIS=block` in the environment **before** importing `time_basis`, so the module-global constants `TIME_BASIS`, `T_PER_YEAR`, `T_MIN`, `T_LABEL` reflect block mode without rewriting `quantoshi.toml`. The TOML stays as the canonical site default; the env var is a per-invocation override (used by build tools, possibly by tests).

- [ ] **Step 1: Append failing tests**

Append to `btc_web/test_time_basis_phase2a.py`:

```python
def test_time_basis_env_var_override(tmp_path, monkeypatch):
    """QS_TIME_BASIS env var overrides the TOML file value."""
    # Create a tiny TOML that says calendar
    toml_path = tmp_path / "test_quantoshi.toml"
    toml_path.write_text(
        'time_basis = "calendar"\n'
        'block_origin = 20188\n'
        'blocks_per_year = 52596\n'
    )
    import time_basis as tb
    # Without env var: TOML wins
    cfg_no_env = tb._load_config(toml_path)
    assert cfg_no_env["time_basis"] == "calendar"
    # With env var: env wins
    monkeypatch.setenv("QS_TIME_BASIS", "block")
    cfg_block = tb._load_config(toml_path)
    assert cfg_block["time_basis"] == "block"


def test_time_basis_env_var_invalid_value_falls_back(tmp_path, monkeypatch):
    """Bogus env var value falls back to TOML/default."""
    import time_basis as tb
    toml_path = tmp_path / "test_quantoshi.toml"
    toml_path.write_text(
        'time_basis = "calendar"\n'
        'block_origin = 20188\n'
        'blocks_per_year = 52596\n'
    )
    monkeypatch.setenv("QS_TIME_BASIS", "garbage")
    cfg = tb._load_config(toml_path)
    # Bogus env value should NOT silently change basis; fall back to TOML.
    assert cfg["time_basis"] == "calendar"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py -v -k env_var
```

Expected: 2 FAILs.

- [ ] **Step 3: Modify `btc_web/time_basis.py::_load_config`**

Find the existing `_load_config` function. Replace its body with env-var-aware logic. The new function:

```python
def _load_config(path: Optional[Path] = None) -> dict:
    """Load quantoshi.toml, falling back to _DEFAULTS if missing.

    Honor `QS_TIME_BASIS` env var as an override on the `time_basis`
    field only. The env var is used by build tools to flip basis for a
    single process without rewriting the TOML. Bogus env values
    (anything not in {"calendar", "block"}) are silently ignored
    (TOML/default wins).

    Public for testing — production callers should use the module-level
    constants below, which are computed once at import time.
    """
    import os as _os
    p = path if path is not None else _TOML_PATH
    if not p.exists():
        _LOG.warning("time_basis: %s not found; using defaults (calendar)", p)
        cfg = dict(_DEFAULTS)
    else:
        with open(p, "rb") as f:
            cfg = {**_DEFAULTS, **tomllib.load(f)}
    env_override = _os.environ.get("QS_TIME_BASIS")
    if env_override in ("calendar", "block"):
        if env_override != cfg.get("time_basis"):
            _LOG.info(
                "time_basis: QS_TIME_BASIS env var overrides TOML "
                "(%r → %r)", cfg.get("time_basis"), env_override,
            )
        cfg["time_basis"] = env_override
    elif env_override is not None:
        _LOG.warning(
            "time_basis: QS_TIME_BASIS=%r is not 'calendar' or 'block'; "
            "ignoring (using %r from TOML/default)",
            env_override, cfg.get("time_basis"),
        )
    return cfg
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py -v -k env_var
```

Expected: 2 PASS.

- [ ] **Step 5: Run the full Phase 1 + Phase 2a suite to verify no regression**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis.py btc_web/test_time_basis_integration.py btc_web/test_time_basis_phase2a.py -v 2>&1 | tail -10
```

Expected: all pass (no regression in calendar-mode behavior).

- [ ] **Step 6: Commit**

```bash
git add btc_web/time_basis.py btc_web/test_time_basis_phase2a.py
git commit -m "feat(phase2b.i): QS_TIME_BASIS env var override for time_basis

time_basis._load_config now honors a QS_TIME_BASIS env var as a
single-process override on the time_basis field. Used by
tools/build_bm_model.py --time-basis=block to flip the module-global
constants (T_PER_YEAR, T_MIN, T_LABEL) without rewriting
quantoshi.toml. Bogus env values are silently ignored (TOML wins).

This closes a Phase 2a gap: Phase 2a's CLI flag parameterized
load_prices() but did not propagate to the module-level constants
read from the TOML at import time, breaking year_to_t() in
fitting.py. With the env var, tools/build_bm_model.py can set
QS_TIME_BASIS before importing anything and the constants reflect
the chosen axis end-to-end.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Make `tools/build_bm_model.py` set env var + axis-aware filenames

**Files:**
- Modify: `tools/build_bm_model.py` (set `QS_TIME_BASIS` before imports; axis-aware `pkl_path` + `diag_path`).
- Modify: `btc_web/test_time_basis_phase2a.py` (append a regression test).

**Goal:** Wire the CLI flag to the env var so the module-global constants reflect the chosen axis. Axis-aware filenames so block builds don't overwrite calendar artifacts.

- [ ] **Step 1: Append regression test**

Append to `btc_web/test_time_basis_phase2a.py`:

```python
def test_build_bm_model_pkl_path_axis_aware():
    """tools/build_bm_model.py uses model_data_block.pkl in block mode.

    We verify by reading the source — running the actual build is
    multi-minute and outside this unit test's scope.
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / "tools" / "build_bm_model.py").read_text()
    # The script must select pkl_path based on args.time_basis. The exact
    # form may vary, but both filenames must appear in the source.
    assert "model_data.pkl" in src
    assert "model_data_block.pkl" in src
    assert "args.time_basis" in src or "time_basis" in src
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py::test_build_bm_model_pkl_path_axis_aware -v
```

Expected: FAIL — `model_data_block.pkl` not yet in the source.

- [ ] **Step 3: Modify `tools/build_bm_model.py`**

Find the `def main():` block. There are two `os.path.join(ROOT, ...)` lines that hardcode the filenames:
1. `pkl_path = os.path.join(ROOT, "model_data.pkl")` (after the BM/QR/sigma fit completes).
2. `diag_path = os.path.join(ROOT, "model_data_resqr_diagnostics.json")` (near the end).

Change the first to compute axis-aware names from `args.time_basis`. Add this helper near the top of `main()` after `args = parser.parse_args()`:

```python
    # Axis-aware artifact filenames. Calendar: model_data.pkl /
    # model_data_resqr_diagnostics.json (back-compat). Block:
    # model_data_block.pkl / model_data_block_resqr_diagnostics.json.
    if args.time_basis == "block":
        pkl_basename = "model_data_block.pkl"
        diag_basename = "model_data_block_resqr_diagnostics.json"
    else:
        pkl_basename = "model_data.pkl"
        diag_basename = "model_data_resqr_diagnostics.json"
```

Then change:

```python
    pkl_path = os.path.join(ROOT, "model_data.pkl")
```

to:

```python
    pkl_path = os.path.join(ROOT, pkl_basename)
```

And change:

```python
    diag_path = os.path.join(ROOT, "model_data_resqr_diagnostics.json")
```

to:

```python
    diag_path = os.path.join(ROOT, diag_basename)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py::test_build_bm_model_pkl_path_axis_aware -v
```

Expected: PASS.

- [ ] **Step 5: Verify --help still works**

```bash
btc_venv/bin/python3 tools/build_bm_model.py --help
```

Expected: argparse help unchanged (no breakage from the rewrite).

- [ ] **Step 6: Commit**

```bash
git add tools/build_bm_model.py btc_web/test_time_basis_phase2a.py
git commit -m "feat(phase2b): axis-aware pkl + diag filenames in build_bm_model

When --time-basis=block, write to model_data_block.pkl + sidecar +
model_data_block_resqr_diagnostics.json. Calendar mode keeps
model_data.pkl / model_data_resqr_diagnostics.json (back-compat).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Run the block build (exploratory)

**Files:** none modified yet — this task runs the build and captures output.

**Goal:** Execute `tools/build_bm_model.py --time-basis=block`. Capture full stdout/stderr. Note any errors or NaN in family fits. The build is expected to **succeed** (the parametric model __init__ methods don't optimize, they just compute predictions from class attrs); the family predictions may be garbage but won't crash the build.

- [ ] **Step 1: Run the build with output capture**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 tools/build_bm_model.py --time-basis=block 2>&1 | tee /tmp/phase2b_build.log
```

Expected timing: 3-5 minutes. The build does sigma fit + bubble composite + QR + 87 ResQR models.

**If the build crashes** (RuntimeError/ValueError before `Wrote model_data_block.pkl`): STOP, report BLOCKED with the traceback. The plan needs to flex (potentially skip troubled models).

**If the build succeeds**: continue to Step 2.

- [ ] **Step 2: Verify the artifacts were written**

```bash
ls -la model_data_block.pkl model_data_block_meta.json model_data_block_resqr_diagnostics.json
```

Expected: all three exist; `.pkl` is ~250-500 KB; `.json` files are ~few KB.

- [ ] **Step 3: Inspect the meta sidecar**

```bash
cat model_data_block_meta.json
```

Expected: `{"time_basis": "block", "t_label": "blocks", "t_per_year": 52596.0, "t_origin": 20188}`.

- [ ] **Step 4: Check the resqr diagnostics for skipped models**

```bash
btc_venv/bin/python3 -c "
import json
with open('model_data_block_resqr_diagnostics.json') as f:
    d = json.load(f)
if d.get('aborted'):
    print('ABORTED:', d.get('reason'))
else:
    skipped = d.get('skipped', [])
    per_model = d.get('per_model', {})
    ok_count = sum(1 for v in per_model.values() if v.get('status') == 'ok')
    print(f'OK: {ok_count}/{len(per_model)} models')
    if skipped:
        print(f'Skipped: {skipped}')
"
```

Expected output: prints a count of OK models out of total. Some skips are tolerable for the scaffold (LPPL family in particular may skip due to NaN coefs from calendar-fit attrs in block mode).

**Acceptance:** at least `bub` (Bubble Model) is OK in the resqr diagnostics. If `bub` aborted, STOP — that's the canonical model and indicates a deeper problem.

- [ ] **Step 5: No commit yet — Task 3 sanity-checks the pkl content first**

---

## Task 3: Sanity-check the block pkl

**Files:** none modified — verification only.

**Goal:** Verify the block pkl can be loaded via `btc_core.load_model_data` (with explicit path) and that the BM/QR scalars look sane (R² > 0, no NaN in critical scalars).

- [ ] **Step 1: Verify pkl loads via btc_core**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -c "
import btc_core as bc
m = bc.load_model_data('model_data_block.pkl')
print(f'time_basis: {getattr(m, \"_path\", \"?\")} loaded')
print(f'qr_fits: {len(m.qr_fits)} quantiles')
print(f'ols_slope: {m.ols_slope}')
print(f'ols_intercept: {m.ols_intercept}')
print(f'support_slope: {m.support_slope}')
print(f'support_intercept: {m.support_intercept}')
print(f'bm_r2: {m.bm_r2}')
print(f'bm_alpha_up/down: {m.bm_alpha_up}, {m.bm_alpha_down}')
print(f'bm_sigma0_up/down: {m.bm_sigma0_up}, {m.bm_sigma0_down}')
print(f'first price_year: {m.price_years[0]}')
print(f'last price_year: {m.price_years[-1]}')
"
```

Expected:
- `qr_fits: 17` (or whatever the canonical count is — same as calendar pkl).
- `ols_slope` is a float, not NaN. In block mode, OLS slope is the slope of `log10(price) vs log10(blocks-since-origin)`. Should be roughly `0.86` (calendar slope is ~5.08, but in log-log on blocks it's much smaller because log10(blocks) is bigger).
- `bm_r2 > 0.5` (BM composite is axis-agnostic; should still fit well).
- `bm_alpha_*` and `bm_sigma0_*` non-NaN.
- `price_years[0]` ≈ 48591 (block offset for 2010-07-17).
- `price_years[-1]` is some large number > 700_000.

- [ ] **Step 2: Quick visual sanity for QR fits**

```bash
btc_venv/bin/python3 -c "
import btc_core as bc
m = bc.load_model_data('model_data_block.pkl')
for q, fit in sorted(m.qr_fits.items()):
    print(f'q={q}: slope={fit[\"slope\"]:.4f}, intercept={fit[\"intercept\"]:.4f}')
"
```

Expected: 17 quantiles printed. Slopes should be ordered (Q1% has lowest slope or highest, monotonically). All values non-NaN.

- [ ] **Step 3: Document scaffold limitations**

The LPPL/HybPPL/EPPL/PCA/Greedy family models in this pkl use calendar-fit class attrs but evaluated against block-axis t. Their predictions will be wildly off (e.g., $10^27 / BTC at modern block heights). This is **expected** — Phase 2b.ii will refit these in block mode. Document this in the Phase 2b.i marker commit message.

**Acceptance for Task 3:** BM, QR, OLS, sigma, support all sane. Family fits not exercised here.

- [ ] **Step 4: Append optional verification test** (defensive, easy to skip if it gets flaky)

Append to `btc_web/test_time_basis_phase2a.py`:

```python
def test_block_pkl_loads_and_has_sane_bm_qr(tmp_path):
    """If model_data_block.pkl exists, verify BM/QR scalars are sane.

    Skipped when no block pkl exists (e.g. on master before Phase 2b.i).
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    block_pkl = repo_root / "model_data_block.pkl"
    if not block_pkl.exists():
        pytest.skip("model_data_block.pkl not built yet")
    import btc_core as bc
    m = bc.load_model_data(str(block_pkl))
    # BM scalars sanity
    assert m.bm_r2 > 0.5, f"bm_r2 low: {m.bm_r2}"
    import math
    for k in ("ols_slope", "ols_intercept",
             "support_slope", "support_intercept",
             "bm_alpha_up", "bm_alpha_down",
             "bm_sigma0_up", "bm_sigma0_down"):
        v = getattr(m, k)
        assert not math.isnan(v), f"{k} is NaN"
        assert math.isfinite(v), f"{k} is not finite: {v}"
    # QR fits sanity
    assert len(m.qr_fits) >= 9  # at least 9 quantiles
    for q, fit in m.qr_fits.items():
        assert math.isfinite(fit["slope"]), f"q={q} slope NaN"
        assert math.isfinite(fit["intercept"]), f"q={q} intercept NaN"
    # Schema fields (Phase 1)
    # Note: load_model_data may not expose these as attributes; if not,
    # this test can be tightened later via a dedicated meta accessor.
```

- [ ] **Step 5: Run the test**

```bash
btc_venv/bin/python3 -m pytest btc_web/test_time_basis_phase2a.py::test_block_pkl_loads_and_has_sane_bm_qr -v
```

Expected: PASS (block pkl exists from Task 2).

---

## Task 4: Commit the block artifacts

**Files:**
- `model_data_block.pkl`, `model_data_block_meta.json`, `model_data_block_resqr_diagnostics.json` (build artifacts).
- `btc_web/test_time_basis_phase2a.py` (test added in Task 3).

**Goal:** Land the scaffold pkl on the sub-branch with a clear commit message about its limitations.

- [ ] **Step 1: Check working tree**

```bash
git status --short
```

Expected: 3 untracked block artifacts + modified test file (if Task 3 Step 4 was done).

- [ ] **Step 2: Commit**

```bash
git add model_data_block.pkl model_data_block_meta.json \
        model_data_block_resqr_diagnostics.json \
        btc_web/test_time_basis_phase2a.py
git commit -m "feat(phase2b.i): scaffold model_data_block.pkl

Block-axis model artifact built via:
  tools/build_bm_model.py --time-basis=block

What works:
  - BM composite + QR fits — axis-agnostic optimizers re-fit cleanly
    on block-offset t-axis (price_years carries blocks-since-T_ORIGIN_BLOCK)
  - sigma bands (asymmetric) — fit on block t
  - meta sidecar (time_basis=block, t_label=blocks, t_per_year=52596,
    t_origin=20188)
  - resqr diagnostics for the bm/pl/exp set (axis-agnostic)

What's a SCAFFOLD (predictions garbage in block mode):
  - LPPL family (10 variants): class attrs (_A, _B, _C, _W, _PHI, _D)
    were fit in calendar-time. Evaluating them against t = blocks
    gives nonsensical \$10^27/BTC predictions.
  - HybPPL family (6 base + 36 cfg variants): same problem; W_cal in
    rad/yr applied to blocks gives meaningless oscillations.
  - EPPL family (1 base + 36 ecfg variants): same.
  - PCA, Greedy: derived from the families, also garbage.

These are addressed in Phase 2b.ii (refit families in block mode with
bound rescaling for W_cal: rad/yr → rad/block, halving prior
2π/(4×52596) rad/block).

Per D11: A/B comparison report deferred. Phase 2b.i validates the
build pipeline produces a structurally correct block pkl.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Phase 2b.i marker + acceptance

**Files:** none — marker only.

**Goal:** Mark Phase 2b.i complete. Verify calendar mode still works (the canonical site is on calendar; we must not have broken it).

- [ ] **Step 1: Run the full test suite** (exclude E2E)

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/ --ignore-glob='*_e2e.py' 2>&1 | tail -10
```

Expected: ~1601 passed (same as Phase 2a + 2 new Phase 2b.i tests), 2 pre-existing failures, 10 skipped.

- [ ] **Step 2: Smoke calendar-mode dev server**

```bash
DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
sleep 8
curl -s -o /dev/null -w "GET / -> %{http_code}\n" http://127.0.0.1:8050/
curl -s -o /dev/null -w "GET /6 -> %{http_code}\n" http://127.0.0.1:8050/6
lsof -ti :8050 | xargs -r kill -9
```

Expected: 200/200. Calendar mode is the active config; the block pkl is dormant on disk.

- [ ] **Step 3: Verify clean working tree**

```bash
git status --short
```

Expected: clean.

- [ ] **Step 4: Phase 2b.i marker commit (empty)**

```bash
git commit --allow-empty -m "phase2b.i(time-basis): scaffold model_data_block.pkl complete

Tasks 1-4 landed:
  - tools/build_bm_model.py output filenames axis-aware
  - model_data_block.pkl built via --time-basis=block (~3-5 min)
  - sanity checks: BM/QR/sigma/OLS scalars all finite, R² > 0.5
  - test_block_pkl_loads_and_has_sane_bm_qr regression guard

Block pkl is a structural scaffold — family models (LPPL, HybPPL,
EPPL, PCA, Greedy) have garbage predictions until Phase 2b.ii
refits them in block mode with W_cal bound rescaling.

Calendar mode (canonical site) untouched. Branch
'time-basis-toggle-phase2b' off 'time-basis-toggle' at HEAD.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: DO NOT push or deploy**

Phase 2b.i is complete and ready for Phase 2b.ii (or a pause for review). The user will decide when to push the sub-branch and whether to merge into `time-basis-toggle`.

---

## Phase 2b.i done

After Task 5: Phase 2b.ii is the next plan. It will:
- For each family parent class (LPPL, HybPPL, EPPL, PCA, Greedy), add per-axis class attr sets OR shift fit values into the pkl per axis.
- For each `tools/fit_*.py` with a W_cal bound, add `--time-basis` flag + bound rescaling (rad/yr → rad/block, halving prior `2π/(4×52596)`).
- Run `tools/refit_all_ppl.py --time-basis=block` (or per-script `tools/fit_*.py --time-basis=block --update`) to populate block-mode params.
- Rebuild model_data_block.pkl with the new params.
- Validate family predictions are now sane in block mode.

The Phase 2b.ii plan is its own document: `docs/superpowers/plans/<date>-time-basis-toggle-phase2b-ii.md`. Don't pre-write it; let Phase 2b.i's results inform the task list.
