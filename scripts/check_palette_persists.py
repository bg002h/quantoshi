"""A returning colourblind-palette user must NOT get default-palette charts.

STATUS: F-11 is FIXED (2026-09-06) and this script passes. It caught the bug
in the first place, and stays as the regression guard: if a late-mounting
per-tab palette selector ever clobbers `palette-store` again, step 2 fails.
See also `scripts/check_palette_roundtrip.py`, which additionally proves the
opposite direction (switching BACK to Default is not swallowed by the
first-fire guard).

Run it against a dev server (`DEV=1 bash run_web.sh`):

    btc_venv/bin/python3 scripts/check_palette_persists.py

It is committed as the reproduction for F-11 and as the safety net for any
future change to the palette → first-render path: if a change ever suppresses
the palette hydration bump, step 2 below is what catches it.

Original purpose follows.


The no-op-bump guards suppress a palette hydration write that EQUALS the
palette the server rendered with. If the comparison were ever done by fire
count instead of by value, a user whose localStorage holds a colourblind
palette would reload into charts painted in the default palette — silently
wrong, and worst for exactly the user who chose that palette.

1. switch the palette in the UI, confirm the chart repaints
2. RELOAD (palette now hydrates from localStorage to a non-default value)
   and confirm the chart still repaints — i.e. the bump was NOT suppressed
3. confirm the trace colours actually differ from the default palette
"""
import json, sys, time
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8050/1"
posts = []


def on_req(r):
    if "/_dash-update-component" not in r.url:
        return
    b = r.post_data
    d = json.loads(b) if b else {}
    posts.append({"output": str(d.get("output"))[:60],
                  "changed": [str(c) for c in (d.get("changedPropIds") or [])]})


def bubble_colors(pg):
    return pg.evaluate("""() => {
        var gd = document.querySelector('#bubble-graph .js-plotly-plot');
        if (!gd || !gd.data) return [];
        return gd.data.slice(0, 12).map(function(t) {
            return (t.line && t.line.color) || (t.marker && t.marker.color) || '';
        }).filter(function(c) { return typeof c === 'string' && c; });
    }""")


fails = []
with sync_playwright() as p:
    br = p.firefox.launch(headless=True)
    ctx = br.new_context(viewport={"width": 1400, "height": 900})
    pg = ctx.new_page()
    pg.on("request", on_req)

    pg.goto(BASE, wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_selector("#bubble-graph .js-plotly-plot", timeout=40000)
    pg.wait_for_timeout(9000)
    default_colors = bubble_colors(pg)
    print(f"default palette colors: {default_colors[:6]}")

    # ── 1. switch palette in the UI ────────────────────────────────────────
    posts.clear()
    pg.select_option("#palette-select-bub", "cb-brian")
    pg.wait_for_timeout(9000)
    switched_colors = bubble_colors(pg)
    print(f"after switch:           {switched_colors[:6]}")
    if switched_colors == default_colors:
        fails.append("palette switch did not change trace colours")
    else:
        print("  OK  switching the palette repaints")

    # ── 2. reload: palette hydrates from localStorage to cb_brian ──────────
    posts.clear()
    pg.goto(BASE, wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_selector("#bubble-graph .js-plotly-plot", timeout=40000)
    pg.wait_for_timeout(12000)
    reloaded_colors = bubble_colors(pg)
    print(f"after reload:           {reloaded_colors[:6]}")

    bumped = [x for x in posts
              if any("bubble-first-render" in c for c in x["changed"])]
    print(f"  bump-triggered POSTs on the CB reload: {len(bumped)}")
    if not bumped:
        fails.append("no bump on CB-palette hydration — charts would stay default")
    if reloaded_colors == default_colors:
        fails.append("RELOAD PAINTED DEFAULT COLOURS for a cb_brian user")
    elif reloaded_colors == switched_colors:
        print("  OK  reload keeps the colourblind palette")

    # cleanup so the profile does not leak into later runs
    pg.select_option("#palette-select-bub", "default")
    pg.wait_for_timeout(3000)
    ctx.close()
    br.close()

print()
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PASS — the load-bearing palette bump survives the no-op guards")
