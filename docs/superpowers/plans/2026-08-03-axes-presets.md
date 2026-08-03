# One-Tap Axes Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a row of one-tap preset buttons to Tab 1's "Axes & Range" panel that set the axis controls in a single click — shipping "Default" and "Current year" on a registry that scales to 4–8 presets.

**Architecture:** A registry tuple in `layout/bubble.py` is the single source of truth. A new `callbacks/axes_presets.py` loops over it and registers **one clientside callback per preset**, each with exactly one Input and five uniform `allow_duplicate` Outputs (the five axis controls). A preset returns `no_update` for any field it does not own, which makes per-preset field ownership structural rather than conventional.

**Tech Stack:** Dash 4.0.0, dash-bootstrap-components 2.0.4, clientside (JS) callbacks, pytest, Playwright (Firefox) for E2E.

**Spec:** `docs/superpowers/specs/2026-08-03-axes-presets-design.md` — read it before starting. Section numbers referenced below (§4, §6.2, …) are that document's.

## Global Constraints

- **One Input per clientside callback — non-negotiable.** `btc_web/callbacks/plot_appearance.py:22-28` documents a Dash 4.0 bug: `allow_duplicate=True` + **multiple** Inputs + `prevent_initial_call` silently fails to fire. Never merge preset callbacks.
- **Register with `_app_ctx.app.clientside_callback`**, never the module-level `dash.clientside_callback` — the latter registers into `dash._callback.GLOBAL_CALLBACK_MAP` instead of `app.callback_map`, and the Task 2 tests would not find it.
- **Never test a JS value with `v || fallback`.** `bub-auto-y`'s legitimate "off" value is `[]`, which is falsy. Use `v !== undefined && v !== null`.
- **Every preset JS body opens with the `if (!n)` guard.** `prevent_initial_call=True` is not sufficient; this repo documents hydration firing callbacks anyway (`btc_web/callbacks/charts/__init__.py:244-245`).
- Python: use `btc_venv/bin/python3` for everything. Dev is 3.14.3, prod 3.12.3.
- Do not modify `snapshot.py`, `snapshot_defaults.py`, `snapshot_defaults_registry.json`, or `tab_defaults.py`. No defaults change means no fingerprint registry update.
- Commit after each task. Branch is `time-basis-toggle-phase2b`.

---

## File Structure

| File | Responsibility |
|---|---|
| `btc_web/layout/bubble.py` | **Modify.** Owns the `AXES_PRESETS` registry, the baked defaults, the JS bodies, and the button row markup. SSOT — callbacks import from here (same direction as `layout/heatmap.py`'s `_HM_PILL_LABELS`). |
| `btc_web/callbacks/axes_presets.py` | **Create.** Nothing but the registration loop. Kept separate from `callbacks/charts/__init__.py` (1800+ lines) so the feature is reviewable in one screen. |
| `btc_web/callbacks/__init__.py` | **Modify.** One import line. Registration is import-driven. |
| `btc_web/test_axes_presets.py` | **Create.** Static tests: registry integrity, markup, baked-default drift, callback registration. |
| `btc_web/test_axes_presets_e2e.py` | **Create.** Playwright behaviour tests. Named `*_e2e.py` so `pytest.ini` excludes it from the default sweep. |

**Task order rationale:** Task 1 ships inert buttons (reviewable as "does the panel look right"). Task 2 adds the wiring and the first working preset — this is where the silent-no-op risk is retired. Task 3 adds "Default", the only preset needing State, view-awareness, and URL precedence.

---

### Task 1: Registry and button row

**Files:**
- Modify: `btc_web/layout/bubble.py`
- Create: `btc_web/test_axes_presets.py`

**Interfaces:**
- Consumes: `SNAPSHOT_DEFAULTS` from `btc_web/snapshot_defaults.py` (a plain dict; imports only stdlib + `time_basis`, so a module-level import is cycle-free).
- Produces, for Tasks 2 and 3:
  - `AXES_CONTROL_IDS: tuple[str, ...]` — the five control ids **in Output order**.
  - `AXES_DEFAULTS: dict[str, Any]` — `{"<id>:value": default}` for those five.
  - `AXES_PRESETS: tuple[dict, ...]` — entries with keys `key: str`, `label: str`, `js: str`, `states: tuple[tuple[str, str], ...]`.
  - Button id convention: `f"bub-axes-preset-{key}"`; row container id `bub-axes-presets`.

- [ ] **Step 1: Write the failing test**

Create `btc_web/test_axes_presets.py`:

```python
"""One-tap axes presets (Tab 1) — registry, markup, callback registration.

Spec: docs/superpowers/specs/2026-08-03-axes-presets-design.md
"""
# Importing conftest imports app, which imports callbacks/__init__.py and so
# registers every callback. Do NOT `import callbacks.axes_presets` directly:
# that would register the callbacks itself, so the registration test in Task 2
# would pass even when the import line in callbacks/__init__.py is missing --
# the exact failure it exists to catch (spec section 9).
from conftest import _app_ctx  # noqa: F401

from layout.bubble import (AXES_CONTROL_IDS, AXES_DEFAULTS, AXES_PRESETS,
                           _bubble_controls)
from snapshot_defaults import SNAPSHOT_DEFAULTS


def _walk(node):
    """Yield every Dash component in a layout tree."""
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _walk(child)


class TestAxesPresetRegistry:
    def test_keys_unique(self):
        keys = [p["key"] for p in AXES_PRESETS]
        assert len(keys) == len(set(keys))

    def test_every_entry_is_complete(self):
        assert AXES_PRESETS, "registry must not be empty"
        for p in AXES_PRESETS:
            assert p["key"].strip(), p
            assert p["label"].strip(), p
            assert p["js"].strip(), p
            assert isinstance(p["states"], tuple), p


def _string_ids(component):
    """Every string id in a layout tree.

    Filters to str deliberately: _mc_controls() embeds Dash pattern-matching
    ids, which are dicts -- {"type": "mc-run-btn", "tab": "bub"} and
    {"type": "mc-run-status", "tab": "bub"} (layout/mc_controls.py:127-128).
    dicts are unhashable, so collecting ids into a set without this filter
    raises TypeError: cannot use 'dict' as a set element.
    """
    return {c.id for c in _walk(component)
            if isinstance(getattr(c, "id", None), str)}


class TestAxesPresetMarkup:
    def test_button_rendered_for_every_preset(self):
        ids = _string_ids(_bubble_controls())
        for p in AXES_PRESETS:
            assert f"bub-axes-preset-{p['key']}" in ids

    def test_preset_row_present(self):
        assert "bub-axes-presets" in _string_ids(_bubble_controls())


class TestAxesBakedDefaults:
    def test_defaults_match_snapshot_defaults_ssot(self):
        assert AXES_DEFAULTS
        for key, value in AXES_DEFAULTS.items():
            assert SNAPSHOT_DEFAULTS[key] == value, key

    def test_every_control_id_has_a_default(self):
        assert len(AXES_CONTROL_IDS) == 5
        for cid in AXES_CONTROL_IDS:
            assert f"{cid}:value" in SNAPSHOT_DEFAULTS
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_axes_presets.py -v
```

Expected: collection error — `ImportError: cannot import name 'AXES_CONTROL_IDS' from 'layout.bubble'`.

- [ ] **Step 3: Add the registry to `btc_web/layout/bubble.py`**

Add `import json` to the existing stdlib imports at the top, and
`from snapshot_defaults import SNAPSHOT_DEFAULTS` to the app-module imports.
(`json` is unused until Task 3; add it now so the import block is touched once.)

Then insert this block immediately **above** `def _bubble_controls():`:

```python
# ── Axes presets (Tab 1) ─────────────────────────────────────────────────
# Registry is the single source of truth for the preset row. Callbacks in
# callbacks/axes_presets.py import it and register one clientside callback per
# entry. Adding a preset = one entry here + its JS body.
# Spec: docs/superpowers/specs/2026-08-03-axes-presets-design.md

# The five Axes & Range controls a preset may write, in Output order.
# Every preset JS body returns a list in exactly this order.
AXES_CONTROL_IDS = ("bub-xscale", "bub-yscale", "bub-xrange",
                    "bub-yrange", "bub-auto-y")

# System defaults, read from the SSOT and baked into the preset JS at import
# time. test_axes_presets.py guards against drift.
AXES_DEFAULTS = {f"{cid}:value": SNAPSHOT_DEFAULTS[f"{cid}:value"]
                 for cid in AXES_CONTROL_IDS}

# Sets the X window to the current calendar year. Y is deliberately left alone:
# when auto-Y is on the existing clientside recompute fits it, and when auto-Y
# is off the user has taken manual control (spec D2).
_JS_CUR_YEAR = """
function(n) {
    var NU = window.dash_clientside.no_update;
    if (!n) { return [NU, NU, NU, NU, NU]; }
    var y = new Date().getFullYear();
    y = Math.max(2010, Math.min(y, 2079));  /* keep y+1 within slider max 2080 */
    return [NU, NU, [y, y + 1], NU, NU];
}
"""

AXES_PRESETS = (
    {"key": "cur_year", "label": "Current year",
     "js": _JS_CUR_YEAR, "states": ()},
)
```

- [ ] **Step 4: Add the button row markup**

In `_bubble_controls()`, inside `_section_card("Axes & Range", …)`, append these
two arguments immediately **after** the `html.Div(id="bub-yrange-wrap", …)`
argument — i.e. they become the card's last children:

```python
            _lbl("Presets"),
            html.Div(
                id="bub-axes-presets",
                className="d-flex flex-wrap gap-1",
                children=[
                    dbc.Button(p["label"], id=f"bub-axes-preset-{p['key']}",
                               size="sm", color="secondary", outline=True,
                               className="flex-fill")
                    for p in AXES_PRESETS
                ],
            ),
```

No conditional logic is needed for the placement requirement: `bub-yrange-wrap`
is shown/hidden by an existing clientside callback
(`btc_web/callbacks/charts/__init__.py:845-853`), so the row sits under the Auto
checkbox when auto-Y is on and under the Y slider when it is off. Verified in
the live DOM: `bub-yrange-wrap` is currently the card body's last child.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_axes_presets.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Verify the app still imports**

```bash
cd /scratch/code/bitcoinprojections/btc_web && PYTHONPATH=".:.." DEV=1 \
  ../btc_venv/bin/python3 -c "import app; print('OK')"
```

Expected: `OK`. This catches a circular-import regression from the new
`snapshot_defaults` import.

Note: `import app` — **not** `import layout` directly. CLAUDE.md's documented
syntax-check command imports `layout` first, which fails with
`AttributeError: 'NoneType' object has no attribute 'clientside_callback'` at
`layout/citadel.py:492`, because `_app_ctx.app` is only populated once `app.py`
has run. That is a pre-existing wart in the documented command, unrelated to
this feature. `app.py` imports `layout`, `callbacks`, and `figures`, so this
still exercises everything.

- [ ] **Step 7: Run the full non-E2E suite for regressions**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/ -q --ignore-glob='*_e2e.py' 2>&1 | tail -5
```

Expected: the new tests pass. **Two failures are pre-existing on this branch and
are NOT yours** — `test_callbacks.py::TestBTCPayPricing::test_free_tier_all_models`
and `test_colors_central.py::test_no_hex_literals_outside_colors_module`. Any
*other* failure is a regression you introduced.

- [ ] **Step 8: Commit**

```bash
cd /scratch/code/bitcoinprojections
git add btc_web/layout/bubble.py btc_web/test_axes_presets.py
git commit -m "feat(tab1): axes-preset registry + button row

Inert for now -- callbacks land in the next commit. Registry is the SSOT for
both the markup and the (upcoming) callback registration.

Spec: docs/superpowers/specs/2026-08-03-axes-presets-design.md"
```

---

### Task 2: Wiring, and the "Current year" preset goes live

**Files:**
- Create: `btc_web/callbacks/axes_presets.py`
- Modify: `btc_web/callbacks/__init__.py`
- Modify: `btc_web/test_axes_presets.py`
- Create: `btc_web/test_axes_presets_e2e.py`

**Interfaces:**
- Consumes: `AXES_CONTROL_IDS`, `AXES_PRESETS` from `layout.bubble` (Task 1).
- Produces: one registered clientside callback per preset, discoverable in
  `_app_ctx.app.callback_map`, keyed by an entry whose `entry["inputs"]` is a
  one-element list `[{"id": "bub-axes-preset-<key>", "property": "n_clicks"}]`.
  States appear separately under `entry["state"]`.

- [ ] **Step 1: Write the failing static tests**

Append to `btc_web/test_axes_presets.py`:

```python
class TestAxesPresetCallbacks:
    """Guards the two ways this feature fails silently.

    app.clientside_callback populates app.callback_map at import time --
    verified empirically against Dash 4.0.0. NOTE: the comment at
    test_callbacks.py:1812-1815 ("callback_map is only populated after
    app.run()") is true only for SERVER @callback registrations, which sit in
    dash._callback.GLOBAL_CALLBACK_MAP until the _setup_server merge.
    App-method clientside registrations never appear there.
    """

    def _preset_entries(self):
        found = {}
        for entry in _app_ctx.app.callback_map.values():
            ids = [i.get("id") for i in entry["inputs"]]
            for p in AXES_PRESETS:
                if f"bub-axes-preset-{p['key']}" in ids:
                    found[p["key"]] = entry
        return found

    def test_callback_registered_for_every_preset(self):
        found = self._preset_entries()
        missing = [p["key"] for p in AXES_PRESETS if p["key"] not in found]
        assert not missing, (
            f"no callback registered for {missing}. Is "
            "`import callbacks.axes_presets` present in callbacks/__init__.py?")

    def test_each_preset_callback_has_exactly_one_input(self):
        # Multiple Inputs + allow_duplicate + prevent_initial_call silently
        # no-ops in Dash 4.0 (plot_appearance.py:22-28). Never merge these.
        for key, entry in self._preset_entries().items():
            assert len(entry["inputs"]) == 1, (
                f"{key} has {len(entry['inputs'])} Inputs; must be exactly 1")

    def test_every_preset_writes_all_five_axis_controls(self):
        # entry["output"] is a list of Output objects (verified against Dash
        # 4.0.0). No isinstance guard -- a guard that skips on an unexpected
        # shape would turn this into a test that passes without asserting.
        for key, entry in self._preset_entries().items():
            out_ids = {o.component_id for o in entry["output"]}
            assert out_ids == set(AXES_CONTROL_IDS), key
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_axes_presets.py -k Callbacks -v
```

Expected: FAIL — `test_callback_registered_for_every_preset` reports
`no callback registered for ['cur_year']`.

- [ ] **Step 3: Create `btc_web/callbacks/axes_presets.py`**

```python
"""One-tap axes presets (Tab 1) — one clientside callback per preset.

MUST be one single-Input clientside callback per preset: Dash 4.0 silently
fails to fire a clientside callback that combines allow_duplicate=True with
MULTIPLE Inputs and prevent_initial_call. See callbacks/plot_appearance.py:22-28.

Registration must go through _app_ctx.app.clientside_callback, not the
module-level dash.clientside_callback, which registers into
dash._callback.GLOBAL_CALLBACK_MAP instead of app.callback_map.

Spec: docs/superpowers/specs/2026-08-03-axes-presets-design.md
"""
from dash import Input, Output, State

import _app_ctx
from layout.bubble import AXES_CONTROL_IDS, AXES_PRESETS

for _preset in AXES_PRESETS:
    _app_ctx.app.clientside_callback(
        _preset["js"],
        # Fresh Output objects per iteration -- do not hoist and reuse.
        *[Output(cid, "value", allow_duplicate=True)
          for cid in AXES_CONTROL_IDS],
        Input(f"bub-axes-preset-{_preset['key']}", "n_clicks"),
        *[State(cid, prop) for cid, prop in _preset["states"]],
        prevent_initial_call=True,
    )
```

- [ ] **Step 4: Register the module in `btc_web/callbacks/__init__.py`**

Add this line to the import block (alongside `import callbacks.plot_appearance`):

```python
import callbacks.axes_presets  # noqa: F401 — callbacks registered at import
```

- [ ] **Step 5: Run the static tests to verify they pass**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_axes_presets.py -v
```

Expected: all pass (9 tests).

- [ ] **Step 6: Write the E2E test**

Create `btc_web/test_axes_presets_e2e.py`:

```python
"""End-to-end Playwright tests for Tab 1's one-tap axes presets.

REQUIRED, not optional: the Dash 4.0 multi-Input bug's failure mode is a
silent no-op, so every preset button must actually be clicked in a browser.

Requires: pip install playwright && python -m playwright install firefox
Run:      cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 \
              -m pytest btc_web/test_axes_presets_e2e.py -v --timeout=90
          (dev server must be running on :8050)
"""
import datetime
import time

import pytest

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

pytestmark = pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")

BASE_URL = "http://localhost:8050"


# Dash 4 renders its own slider (NOT rc-slider). Each thumb is a
# span.dash-slider-thumb carrying aria-valuenow. Verified in the live DOM.
def _slider(page, eid):
    return page.evaluate(
        f'[...document.querySelectorAll("#{eid} [aria-valuenow]")]'
        f'.map(n => Number(n.getAttribute("aria-valuenow")))')


def _radio(page, eid):
    return page.evaluate(
        f'(() => {{ const r = [...document.querySelectorAll('
        f'"#{eid} input[type=radio]")].find(x => x.checked);'
        f' return r ? r.value : null; }})()')


def _n_checked(page, eid):
    return page.evaluate(
        f'[...document.querySelectorAll("#{eid} input[type=checkbox]")]'
        f'.filter(c => c.checked).length')


def _wait_until(predicate, timeout=15.0):
    """Poll until predicate() is truthy. Returns the final value."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.25)
    return last


@pytest.fixture(scope="module")
def page():
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        pg = ctx.new_page()
        pg.goto(f"{BASE_URL}/1", wait_until="networkidle", timeout=30000)
        pg.wait_for_selector("#bub-axes-presets", state="attached", timeout=15000)
        time.sleep(1.5)  # let the restore/first-render cascade settle
        yield pg
        browser.close()


def test_current_year_sets_xrange(page):
    yr = datetime.date.today().year
    page.click("#bub-axes-preset-cur_year")
    got = _wait_until(lambda: _slider(page, "bub-xrange") == [yr, yr + 1])
    assert _slider(page, "bub-xrange") == [yr, yr + 1], (
        f"expected [{yr}, {yr + 1}], got {_slider(page, 'bub-xrange')}. "
        "A silent no-op here means the Dash multi-Input bug, a missing "
        "callbacks/__init__.py import, or a JS error -- check the console.")


def test_current_year_leaves_scales_alone(page):
    """Per-preset field ownership (spec D1): cur_year writes X range only."""
    before = (_radio(page, "bub-xscale"), _radio(page, "bub-yscale"))
    page.click("#bub-axes-preset-cur_year")
    time.sleep(1.0)
    assert (_radio(page, "bub-xscale"), _radio(page, "bub-yscale")) == before
```

- [ ] **Step 7: Run the E2E test**

Start the dev server if it is not already up, then run the file:

```bash
cd /scratch/code/bitcoinprojections
curl -sf -o /dev/null http://127.0.0.1:8050/1 || {
  lsof -ti :8050 | xargs -r kill -9
  DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
  for i in $(seq 1 60); do curl -sf -o /dev/null http://127.0.0.1:8050/ && break; sleep 2; done
}
btc_venv/bin/python3 -m pytest btc_web/test_axes_presets_e2e.py -v --timeout=90
```

Expected: 2 passed. (The `/run-e2e` skill automates the server handling if you
prefer.)

**If `test_current_year_sets_xrange` fails with the range unchanged**, the
single-Input-clientside assumption is broken. Do not work around it by merging
callbacks. Open the browser console for a JS error first; if the callback truly
never fires, fall back to the spec §11 mitigation (a bridge callback baking the
values into a JS global) and update the spec.

- [ ] **Step 8: Commit**

```bash
cd /scratch/code/bitcoinprojections
git add btc_web/callbacks/axes_presets.py btc_web/callbacks/__init__.py \
        btc_web/test_axes_presets.py btc_web/test_axes_presets_e2e.py
git commit -m "feat(tab1): wire axes presets, ship 'Current year'

One clientside callback per preset, single Input each -- merging them would
hit the Dash 4.0 silent-no-op bug documented in plot_appearance.py:22-28.

Static tests assert a callback is registered per button and that each has
exactly one Input; E2E clicks the button, because the failure mode is silent.

Spec: docs/superpowers/specs/2026-08-03-axes-presets-design.md"
```

---

### Task 3: The "Default" preset

**Files:**
- Modify: `btc_web/layout/bubble.py`
- Modify: `btc_web/callbacks/charts/__init__.py` — import the constant, replace 3 literals in `toggle_bub_view`
- Modify: `btc_web/callbacks/routing.py` — import the constant, replace 1 literal at `:428`
- Modify: `btc_web/test_axes_presets.py`
- Modify: `btc_web/test_axes_presets_e2e.py`

**Interfaces:**
- Consumes: `AXES_CONTROL_IDS`, `AXES_DEFAULTS` (Task 1); the registration loop (Task 2), which needs no change — it already reads each entry's `states`.
- Produces: `CAGR_DEFAULT_XRANGE: list[int]`, and a second `AXES_PRESETS` entry with `states = (("snapshot-state-store", "data"), ("bub-view-mode", "data"))`.

**Behaviour being built (spec §6.2).** For each of the five fields: use
`snapshot-state-store["<id>:value"]` when present and non-null, else the baked
system default. View-awareness applies **only to the fallback** — a share link's
X range wins in every view; `[2025, 2050]` is used in CAGR view only when the
link supplied nothing.

- [ ] **Step 1: Write the failing single-source test**

Append to `btc_web/test_axes_presets.py`:

```python
class TestCagrDefaultXrange:
    """[2025, 2050] must have exactly one definition.

    It appears in four places that must agree: the price->CAGR swap and the two
    swap-BACK comparisons in toggle_bub_view, plus the /1.2 deep-link handler in
    routing.py. The swap-back tests exact equality, so a diverged copy silently
    stops CAGR view from restoring [2010, 2033] when you switch back to price.
    """

    def test_no_module_still_hardcodes_the_literal(self):
        import pathlib
        import callbacks.charts as _charts
        import callbacks.routing as _routing

        for mod in (_charts, _routing):
            src = pathlib.Path(mod.__file__).read_text()
            assert "2025, 2050" not in src, (
                f"{mod.__name__} still hardcodes [2025, 2050]. Import "
                "CAGR_DEFAULT_XRANGE from layout.bubble instead -- a second "
                "copy breaks the CAGR<->price swap, which compares for "
                "exact equality.")

    def test_constant_is_a_list_not_a_tuple(self):
        from layout.bubble import CAGR_DEFAULT_XRANGE
        # Compared against JSON-decoded slider values:
        # [2025, 2050] == (2025, 2050) is False, which would break both
        # swap directions silently.
        assert isinstance(CAGR_DEFAULT_XRANGE, list)
        assert CAGR_DEFAULT_XRANGE == [2025, 2050]
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_axes_presets.py -k Cagr -v
```

Expected: FAIL — `ImportError: cannot import name 'CAGR_DEFAULT_XRANGE'`, and
the literal is still present in both modules.

- [ ] **Step 3: Add the constant and the "Default" JS to `btc_web/layout/bubble.py`**

Insert immediately after the `AXES_DEFAULTS` assignment:

```python
# X range that CAGR view uses, where the X slider means exit years rather than
# calendar years. SINGLE SOURCE: callbacks/charts/__init__.py (toggle_bub_view,
# 3 sites) and callbacks/routing.py (the /1.2 deep link) import this. Those
# sites compare it for EXACT EQUALITY to swap back to the price-view range, so
# a second copy silently breaks the CAGR<->price round-trip.
# MUST stay a list -- [2025, 2050] == (2025, 2050) is False in Python.
CAGR_DEFAULT_XRANGE = [2025, 2050]
```

- [ ] **Step 3b: Replace the four hardcoded literals**

In `btc_web/callbacks/charts/__init__.py`, add to the existing top-of-file
import block (next to the `from layout.common import _bands_to_qs` line):

```python
from layout.bubble import CAGR_DEFAULT_XRANGE
```

Then replace all three literals inside `toggle_bub_view` — **all three, not
just the first; replacing one creates intra-function drift, which is worse
than leaving them alone**:

```python
# line ~547, the price -> CAGR swap
-        xr = [2025, 2050] if cur_xrange == [2010, 2033] else dash.no_update
+        xr = CAGR_DEFAULT_XRANGE if cur_xrange == [2010, 2033] else dash.no_update

# line ~553, CAGR -> residuals
-        xr = [2010, 2033] if cur_xrange == [2025, 2050] else dash.no_update
+        xr = [2010, 2033] if cur_xrange == CAGR_DEFAULT_XRANGE else dash.no_update

# line ~558, CAGR -> price
-    xr = [2010, 2033] if cur_xrange == [2025, 2050] else dash.no_update
+    xr = [2010, 2033] if cur_xrange == CAGR_DEFAULT_XRANGE else dash.no_update
```

In `btc_web/callbacks/routing.py` (which already imports from `layout` at
module level), add the same import and replace the literal at ~line 428:

```python
-                [2025, 2050], fwd_yrs, hover_today)
+                CAGR_DEFAULT_XRANGE, fwd_yrs, hover_today)
```

Leave the `[2010, 2033]` operands alone. That value is duplicated far more
widely and is anchored by the snapshot-fingerprint registry workflow;
consolidating it is out of scope.

Then insert this after `_JS_CUR_YEAR`:

```python
# Restores the axes the page loaded with: the share-link URL's values when the
# page came from one, system defaults otherwise. View-awareness applies ONLY to
# the fallback -- a link's X range wins in every view.
#
# The presence test must be `!== undefined && !== null`, never `||`:
# bub-auto-y's legitimate "off" value is [], which is falsy.
_JS_DEFAULT = """
function(n, snap, view_mode) {
    var NU = window.dash_clientside.no_update;
    if (!n) { return [NU, NU, NU, NU, NU]; }
    var IDS = %(ids)s;
    var DEFAULTS = %(defaults)s;
    var out = [];
    for (var i = 0; i < IDS.length; i++) {
        var k = IDS[i] + ":value";
        var fromUrl = (snap && snap[k] !== undefined && snap[k] !== null);
        if (fromUrl) {
            out.push(snap[k]);
        } else if (IDS[i] === "bub-xrange" && view_mode === "cagr") {
            out.push(%(cagr_xrange)s);
        } else {
            out.push(DEFAULTS[k]);
        }
    }
    return out;
}
""" % {
    "ids": json.dumps(list(AXES_CONTROL_IDS)),
    "defaults": json.dumps(AXES_DEFAULTS),
    "cagr_xrange": json.dumps(CAGR_DEFAULT_XRANGE),
}
```

- [ ] **Step 4: Add the registry entry**

Replace the `AXES_PRESETS` tuple with:

```python
AXES_PRESETS = (
    {"key": "default", "label": "Default",
     "js": _JS_DEFAULT,
     "states": (("snapshot-state-store", "data"), ("bub-view-mode", "data"))},
    {"key": "cur_year", "label": "Current year",
     "js": _JS_CUR_YEAR, "states": ()},
)
```

"Default" is listed first so it renders leftmost. The registration loop in
`callbacks/axes_presets.py` needs no change — it already builds States from each
entry's own `states` tuple.

- [ ] **Step 5: Run the static tests**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/test_axes_presets.py -v
```

Expected: all pass, now covering both presets. Confirm
`test_each_preset_callback_has_exactly_one_input` still passes — "Default" has
two States, which must not count as Inputs.

- [ ] **Step 6: Write the E2E tests**

Append to `btc_web/test_axes_presets_e2e.py`:

```python
LINK_XRANGE = [2015, 2040]  # deliberately not the system default


@pytest.fixture(scope="module")
def share_hash():
    """A share link whose bub-xrange differs from the system default.

    Proves it decodes before any browser sees it, so a prefix/encoder mismatch
    fails here with a clear message instead of as a mystery browser failure.
    """
    # btc_web/ is already on sys.path via btc_web/conftest.py, which pytest
    # auto-loads for this directory. No path manipulation needed.
    from snapshot import _encode_snapshot, _decode_snapshot, _SNAP_PREFIX

    blob = f"{_SNAP_PREFIX}{_encode_snapshot({'bub-xrange:value': LINK_XRANGE})}"
    decoded = _decode_snapshot(blob)
    assert decoded is not None, (
        f"{_SNAP_PREFIX!r} does not pair with _encode_snapshot -- use the "
        "encoder matching the current prefix.")
    assert decoded.get("bub-xrange:value") == LINK_XRANGE
    return blob


def test_default_restores_system_defaults(page):
    """No share link: Default returns all five controls to factory values."""
    baseline_y = _slider(page, "bub-yrange")

    # Disturb the axes using the feature itself plus two direct clicks.
    page.click("#bub-axes-preset-cur_year")
    time.sleep(0.8)
    page.click("#bub-xscale input[value='linear']")
    page.click("#bub-auto-y input[type=checkbox]")
    time.sleep(1.0)
    assert _n_checked(page, "bub-auto-y") == 0

    page.click("#bub-axes-preset-default")
    _wait_until(lambda: _slider(page, "bub-xrange") == [2010, 2033])

    assert _slider(page, "bub-xrange") == [2010, 2033]
    assert _radio(page, "bub-xscale") == "log"
    assert _radio(page, "bub-yscale") == "log"
    assert _n_checked(page, "bub-auto-y") == 1
    # Y is asserted against the baseline captured on load, NOT against
    # SNAPSHOT_DEFAULTS: with auto-Y on, auto_bubble_yrange recomputes and
    # rounds, so a freshly loaded page reads [-1.5, 6.0], not [-1.5, 6.05]
    # (spec section 7.2).
    _wait_until(lambda: _slider(page, "bub-yrange") == baseline_y)
    assert _slider(page, "bub-yrange") == baseline_y


def test_default_restores_share_link_xrange(page, share_hash):
    """ACCEPTANCE TEST for the URL requirement.

    On a page loaded from a share link, Default returns to THAT LINK's axes,
    not the system defaults.
    """
    page.goto(f"{BASE_URL}/1{share_hash}", wait_until="networkidle", timeout=30000)
    page.wait_for_selector("#bub-axes-presets", state="attached", timeout=15000)
    _wait_until(lambda: _slider(page, "bub-xrange") == LINK_XRANGE)
    assert _slider(page, "bub-xrange") == LINK_XRANGE, "share link did not restore"

    # Move away from the link's range, then ask for it back.
    page.click("#bub-axes-preset-cur_year")
    _wait_until(lambda: _slider(page, "bub-xrange") != LINK_XRANGE)
    assert _slider(page, "bub-xrange") != LINK_XRANGE

    page.click("#bub-axes-preset-default")
    _wait_until(lambda: _slider(page, "bub-xrange") == LINK_XRANGE)
    assert _slider(page, "bub-xrange") == LINK_XRANGE, (
        "Default fell back to the system default instead of the link's value")

    page.goto(f"{BASE_URL}/1", wait_until="networkidle", timeout=30000)
    page.wait_for_selector("#bub-axes-presets", state="attached", timeout=15000)
    time.sleep(1.0)


def test_default_is_view_aware_in_cagr(page):
    """No share link + CAGR view: the fallback is the CAGR range."""
    page.click("#bub-view-cagr")
    _wait_until(lambda: _slider(page, "bub-xrange") == [2025, 2050])

    page.click("#bub-axes-preset-cur_year")
    _wait_until(lambda: _slider(page, "bub-xrange") != [2025, 2050])

    page.click("#bub-axes-preset-default")
    _wait_until(lambda: _slider(page, "bub-xrange") == [2025, 2050])
    assert _slider(page, "bub-xrange") == [2025, 2050]

    page.click("#bub-view-price")  # restore for any later test
    time.sleep(0.8)
```

- [ ] **Step 7: Run the E2E suite**

```bash
cd /scratch/code/bitcoinprojections
curl -sf -o /dev/null http://127.0.0.1:8050/1 || {
  lsof -ti :8050 | xargs -r kill -9
  DEV=1 nohup bash run_web.sh > /tmp/quantoshi_dev.log 2>&1 &
  for i in $(seq 1 60); do curl -sf -o /dev/null http://127.0.0.1:8050/ && break; sleep 2; done
}
btc_venv/bin/python3 -m pytest btc_web/test_axes_presets_e2e.py -v --timeout=120
```

Expected: 5 passed. The tests share a module-scoped `page`, so run the file
whole rather than cherry-picking with `-k`.

- [ ] **Step 8: Run the full non-E2E suite**

```bash
cd /scratch/code/bitcoinprojections
btc_venv/bin/python3 -m pytest btc_web/ -q --ignore-glob='*_e2e.py' 2>&1 | tail -5
```

Expected: only the two pre-existing failures named in Task 1 Step 7.

- [ ] **Step 9: Visual check**

Open `http://127.0.0.1:8050/1` and confirm:
1. The preset row sits directly under the **Auto** checkbox (auto-Y is on by default).
2. Untick **Auto** — the Y-range slider appears *above* the preset row.
3. Both buttons fit the column and wrap rather than overflow.

- [ ] **Step 10: Commit**

```bash
cd /scratch/code/bitcoinprojections
git add btc_web/layout/bubble.py btc_web/callbacks/charts/__init__.py \
        btc_web/callbacks/routing.py btc_web/test_axes_presets.py \
        btc_web/test_axes_presets_e2e.py
git commit -m "feat(tab1): add 'Default' axes preset

Restores the axes the page loaded with -- the share link's values when the page
came from one, system defaults otherwise. View-awareness applies only to the
fallback, so a link's X range wins in every view while CAGR view (where the
slider means exit years) falls back to [2025, 2050] rather than a range of
exit years in the past.

E2E covers the acceptance case: load a share link, move away, tap Default, and
land back on the link's range rather than the system default.

Spec: docs/superpowers/specs/2026-08-03-axes-presets-design.md"
```

---

## Deferred (documented in the spec, deliberately not built)

Do not "fix" these while implementing — they are accepted behaviour recorded in
spec §7:

- **Custom Time Axis**: with `cta-active` on, `update_bubble` raises
  `PreventUpdate`, so "Default"'s scale/Y writes do not reach the visible chart
  until CTA is switched off. "Current year" works normally.
- **Auto-Y wins**: with auto-Y on, `auto_bubble_yrange` recomputes the Y range
  after "Default" writes it. Expected, and why E2E asserts against a captured
  baseline rather than a literal.
- **Post-restore window**: a preset tapped within the ≤4 s `snapshot-pending`
  window updates the controls without refiring the chart.
- **Slider max drift**: "Default" can restore a Y range above the live slider
  max (which depends on `bub-model-show`), and an X range above the residuals
  view's `current_year + 1` cap. Cosmetic until the slider is dragged.
- **Two renders** per X-range change with auto-Y on — identical to dragging the
  X slider today.
