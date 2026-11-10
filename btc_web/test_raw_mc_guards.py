"""Raw MC, Task 6: Raw never reaches payment or the premium/match chrome, the
real-click MC auto-range starts at the current year in Raw, and a Raw regime
click ticks "Show individual price paths" like the model one does.

Spec: "### Paid and premium chrome in Raw" and "### Auto-range"
(docs/superpowers/specs/2026-11-10-tab1-raw-mc-design.md). The JS runs under
node with the harnesses of test_mc_autowiden.py / test_mc_regime_paths.py.
"""
import json
import pathlib
import shutil
import subprocess

import dash
import pytest

from conftest import _app_ctx, _patch_ctx  # noqa: F401  (registers callbacks)
from raw_mc import RAW_SRCS

_APP = _app_ctx.app      # bound at collection: see test_mc_autowiden.py
_HERE = pathlib.Path(__file__).parent
_NODE = shutil.which("node")
needs_node = pytest.mark.skipif(_NODE is None, reason="node not installed")
NU = "__NU__"


def _node(script):
    res = subprocess.run([_NODE, "-e", script], capture_output=True,
                         text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


# ── payment: a stale-client Run click in Raw never invoices ─────────────────

_TABS = ("dca", "ret", "hm", "sc", "cp", "bub")


def _pay_args(bub_src, bub_price=0):
    """Positional args of _mc_payment_initiate: 1 list Input, (years, start,
    entry_q) x 6, trigger, 6 model sources, 6 prices, 15 config States."""
    srcs = ["bub"] * 6
    prices = [0] * 6
    srcs[_TABS.index("bub")] = bub_src
    prices[_TABS.index("bub")] = bub_price
    return ([[1]] + [40, 2028, 10] * 6 + [0] + srcs + prices + [None] * 15)


@pytest.fixture
def paid_env(monkeypatch):
    """BTCPay 'configured', not DEV, not free tier: the next step after the
    quant check is create_invoice. Records every call instead of raising,
    because the callback swallows exceptions from create_invoice."""
    import btcpay
    calls = []
    monkeypatch.setattr(_app_ctx, "_HAS_BTCPAY", True, raising=False)
    monkeypatch.delenv("DEV", raising=False)
    monkeypatch.setattr(btcpay, "is_free_tier", lambda *a, **k: False)

    def _invoice(*a, **k):
        calls.append(a)
        return {"invoice_id": "x", "checkout_url": "", "amount_sats": 1}
    monkeypatch.setattr(btcpay, "create_invoice", _invoice)
    return calls


def _initiate(args):
    from callbacks.mc_payment import _mc_payment_initiate
    with _patch_ctx({"type": "mc-run-btn", "tab": "bub"}):
        return _mc_payment_initiate(*args)


@pytest.mark.parametrize("src", RAW_SRCS)
def test_run_click_with_raw_source_never_invoices(paid_env, src):
    with pytest.raises(dash.exceptions.PreventUpdate):
        _initiate(_pay_args(src))
    # Above the quant threshold the quant modal would open first: it stays shut.
    with pytest.raises(dash.exceptions.PreventUpdate):
        _initiate(_pay_args(src, bub_price=60_000))
    assert paid_env == []


def test_model_source_still_invoices_and_warns(paid_env):
    """Control: the same click in model mode reaches create_invoice, and above
    50 000 sats opens mc-quant-modal (Output index 7) -- so the Raw test
    above can fail."""
    out = _initiate(_pay_args("bub"))
    assert out[0] is True and len(paid_env) == 1
    assert _initiate(_pay_args("bub", bub_price=60_000))[7] is True


def test_raw_guard_sits_right_after_the_model_source_is_read():
    """M-6: the guard precedes the quant check, the free-tier check and
    create_invoice."""
    import inspect
    from callbacks.mc_payment import _mc_payment_initiate
    src = inspect.getsource(_mc_payment_initiate)
    read = src.index("mc_model_src = model_srcs[tab_idx]")
    guard = src.index("if mc_model_src in RAW_SRCS:")
    assert read < guard < src.index("if tab_price > _MC_QUANT_THRESHOLD")
    assert guard < src.index("btcpay.is_free_tier(") \
        < src.index("btcpay.create_invoice(")


# ── match indicator: neutral beside a Raw fan ───────────────────────────────

_MATCH_HARNESS = r"""
const NU = '__NU__';
globalThis.window = {dash_clientside: {no_update: NU}};
const fn = (%(fn)s);
const out = [];
for (const c of %(cases)s) {
    out.push(fn(c.enable, 40, 2028, 10, c.src, c.key));
}
console.log(JSON.stringify(out));
"""


def _match(cases):
    from callbacks.mc_controls import _MC_MATCH_JS_TPL
    from colors import MC_FREE_GREEN, MC_LIVE_AMBER
    fn = _MC_MATCH_JS_TPL.format(base_cls="", sep="", free_clr=MC_FREE_GREEN,
                                 live_clr=MC_LIVE_AMBER)
    assert fn.strip() in "".join(_APP._inline_scripts)   # what Tab 1 runs
    return _node(_MATCH_HARNESS % {"fn": fn, "cases": json.dumps(cases)})


HIDE = {"display": "none"}
NEUTRAL = ["", HIDE, HIDE, "", HIDE, HIDE]
_KEY = {"years": 40, "start_yr": 2028, "entry_q": 10}


@needs_node
@pytest.mark.parametrize("src", RAW_SRCS)
@pytest.mark.parametrize("key", [_KEY, {"years": 10, "start_yr": 2035,
                                        "entry_q": 90}, None])
def test_match_js_raw_is_neutral(src, key):
    """A rendered key left from an earlier model run (matching or stale), or
    none at all: no text, no premium class, no badge, no restore button, no
    mismatch warning (spec, review M-4)."""
    assert _match([dict(enable=["yes"], src=src, key=key)]) == [NEUTRAL]


@needs_node
def test_match_js_model_unchanged():
    got = _match([dict(enable=["yes"], src="bub", key=_KEY),
                  dict(enable=["yes"], src="bub", key=None),
                  dict(enable=[], src="bub", key=_KEY)])
    assert got[0][0].startswith("✓") and got[0][3] == "mc-premium-chart"
    assert got[1][0].startswith("⚠") and got[1][2] == {}
    assert got[2] == NEUTRAL


# ── auto-range: Raw starts at the current year ──────────────────────────────

def _widen(cases):
    from test_mc_autowiden import _run
    return _run(cases)


@needs_node
@pytest.mark.parametrize("src", RAW_SRCS)
@pytest.mark.parametrize("right", [2033, 2060])
def test_autowiden_raw_uses_current_year(src, right):
    from test_mc_autowiden import _YEAR
    got = _widen([dict(clickAgoMs=5, mc=["yes"], xr=[2010, right],
                       start=2028, src=src)])
    assert got == [[_YEAR, max(right, _YEAR + 10)]]


@needs_node
@pytest.mark.parametrize("start", [None, "", "abc"])
def test_autowiden_raw_ignores_the_model_start_year(start):
    from test_mc_autowiden import _YEAR
    assert _widen([dict(clickAgoMs=5, mc=["yes"], xr=[2010, 2033],
                        start=start, src="raw_cycle")]) \
        == [[_YEAR, _YEAR + 10]]


@needs_node
def test_autowiden_raw_keeps_the_guards():
    """No click, another view, CTA on: no write, exactly as in model mode."""
    base = dict(mc=["yes"], xr=[2010, 2033], start=2028, src="raw_iid")
    assert _widen([dict(base, clickAgoMs=None),
                   dict(base, clickAgoMs=5, mode="resid"),
                   dict(base, clickAgoMs=5, cta=["yes"])]) == [NU, NU, NU]


@needs_node
@pytest.mark.parametrize("src", ["bub", None, "pl"])
def test_autowiden_model_unchanged(src):
    from test_mc_autowiden import _YEAR
    assert _widen([dict(clickAgoMs=5, mc=["yes"], xr=[2010, 2033],
                        start=2028, src=src)]) == [[_YEAR, 2038]]


def test_autowiden_reads_the_model_source_as_state():
    from test_mc_autowiden import _entry
    assert [(s["id"], s["property"]) for s in _entry()["state"]][-1] == \
        ("bub-mc-model-src", "value")


# ── regime auto-tick: the Raw checklist ─────────────────────────────────────

_RAW_ASSET = _HERE / "assets" / "mc_raw_regime_paths.js"


def _regime(cases):
    from test_mc_regime_paths import _run
    return _run(cases)


def _rc(raw, paths=(), raw_bins=5, raw_click=5, model=(0, 1, 2, 3, 4),
        click=None, reset=False):
    d = dict(checked=list(model), bins=5, rawChecked=raw, rawBins=raw_bins,
             paths=list(paths), reset=reset)
    if click is not None:
        d["clickAgoMs"] = click
    if raw_click is not None:
        d["rawClickAgoMs"] = raw_click
    return d


@needs_node
class TestRawRegimeAutotick:
    def test_raw_regime_autotick_decision(self):
        assert _regime([_rc([0, 1], reset=True)]) == [["yes"]]

    def test_no_click_or_stale_click_means_no_change(self):
        assert _regime([_rc([0, 1], raw_click=None, reset=True)]) == [NU]
        assert _regime([_rc([0, 1], raw_click=5000, reset=True)]) == [NU]

    def test_already_on_is_left_alone(self):
        assert _regime([_rc([0, 1], paths=["yes"], reset=True)]) == [NU]

    def test_narrowing_further_does_not_override_a_manual_untick(self):
        assert _regime([_rc([0, 1], reset=True), _rc([0])]) == [["yes"], NU]

    def test_clearing_then_reapplying_ticks_again(self):
        assert _regime([_rc([0, 1], reset=True),
                        _rc([0, 1, 2, 3, 4], paths=["yes"]),
                        _rc([0, 1])]) == [["yes"], NU, ["yes"]]

    def test_default_k_is_five(self):
        """First run starts from the layout default (5 of 5): unticking one
        box is a transition. Raw bins go up to 10."""
        from layout.mc_controls import MC_RAW_REGIME_DEFAULT_K
        from layout.bubble import _bubble_controls
        from test_raw_mc_callback import _by_id
        assert MC_RAW_REGIME_DEFAULT_K == 5 == len(
            _by_id(_bubble_controls(), "bub-mc-raw-regime")[0].value)
        assert _regime([_rc([1, 2, 3, 4], reset=True)]) == [["yes"]]
        assert _regime([_rc(list(range(9)), raw_bins=10, reset=True)]) == [NU]

    def test_out_of_range_indices_are_ignored(self):
        """The Raw value is never reset on a bins change; the server ignores
        indices >= bins (raw_mc.effective_checked). [0..4] at 3 bins is no
        filter; [0, 1, 4] at 3 bins is a filter."""
        assert _regime([_rc([0, 1, 2, 3, 4], raw_bins=3, reset=True)]) == [NU]
        assert _regime([_rc([0, 1, 4], raw_bins=3, reset=True)]) == [["yes"]]

    def test_a_raw_click_is_not_a_model_click(self):
        """Raw stamp fresh, model filter just applied programmatically: only
        the Raw checklist is judged (and its filter is off)."""
        assert _regime([_rc([0, 1, 2, 3, 4], model=[0, 1], reset=True)]) == [NU]

    def test_a_model_click_is_not_a_raw_click(self):
        assert _regime([_rc([0, 1], raw_click=None, model=[0, 1, 2, 3, 4],
                            click=5, reset=True)]) == [NU]

    def test_model_click_still_ticks_with_raw_args_present(self):
        assert _regime([_rc([0, 1, 2, 3, 4], raw_click=None, model=[0, 1],
                            click=5, reset=True)]) == [["yes"]]

    def test_restore_writes_both_then_click_each(self):
        """I-6: a share-link restore writes both checklists in one batch with
        no stamp (filters on, paths off). Both trackers learn 'filtered', so a
        later click on either one (still filtered) is not a transition."""
        r = _regime([_rc([0, 1], model=[0, 1], raw_click=None, reset=True),
                     _rc([0], model=[0, 1], raw_click=5),
                     _rc([0], model=[0], raw_click=None, click=5)])
        assert r == [NU, NU, NU]

    def test_both_trackers_update_on_every_call(self):
        """A model-click call records the Raw count too: a Raw filter written
        while the model checklist fired is remembered."""
        r = _regime([_rc([0, 1], raw_click=None, model=[0, 1, 2, 3, 4],
                         click=5, reset=True),     # model: no filter; raw on
                     _rc([0])])                    # raw click, already on
        assert r == [NU, NU]


def test_raw_asset_and_callback_share_the_click_stamp():
    from layout.mc_controls import MC_REGIME_PATHS_JS
    src = _RAW_ASSET.read_text()
    assert "#bub-mc-raw-regime" in src
    assert "__qsMcRawRegimeClickAt" in src
    assert "__qsMcRawRegimeClickAt" in MC_REGIME_PATHS_JS
    assert "__qsRawRegimeK" in MC_REGIME_PATHS_JS


_LISTENER_HARNESS = r"""
let handler = null, capture = null, evt = null;
globalThis.window = {};
globalThis.document = {addEventListener: (t, h, c) => { evt = t; handler = h; capture = c; }};
Date.now = () => 4242;
%(src)s
const raw = {closest: (sel) => sel === '#bub-mc-raw-regime' ? {} : null};
const model = {closest: (sel) => sel === '#bub-mc-regime' ? {} : null};
const res = {evt, capture};
handler({target: model});
res.afterModel = window.__qsMcRawRegimeClickAt || 0;
handler({target: null});
handler({target: {}});
handler({target: raw});
res.afterRaw = window.__qsMcRawRegimeClickAt || 0;
res.modelStamp = window.__qsMcRegimeClickAt || 0;
console.log(JSON.stringify(res));
"""


@needs_node
def test_raw_asset_stamps_only_raw_regime_clicks():
    got = _node(_LISTENER_HARNESS % {"src": _RAW_ASSET.read_text()})
    assert got == {"evt": "click", "capture": True, "afterModel": 0,
                   "afterRaw": 4242, "modelStamp": 0}


def test_regime_autotick_registration():
    """ONE callback writes bub-mc-paths.value, as its primary (non
    allow_duplicate) Output: an allow_duplicate writer fired update_bubble
    twice per click (auto-tick review I-2)."""
    writers = [e for e in _APP._callback_list
               if "bub-mc-paths.value" in str(e["output"])]
    assert len(writers) == 1
    e = writers[0]
    assert e["output"] == "bub-mc-paths.value"
    assert [(i["id"], i["property"]) for i in e["inputs"]] == [
        ("bub-mc-regime", "value"), ("bub-mc-raw-regime", "value")]
    assert [(s["id"], s["property"]) for s in e["state"]] == [
        ("bub-mc-bins", "value"), ("bub-mc-raw-bins", "value"),
        ("bub-mc-paths", "value")]
    assert e["prevent_initial_call"] is True
