"""Full round trip: default -> cb-brian -> reload -> back to default -> reload.

The first-fire guard added for F-11 ignores a selector's opening value when it
equals the server-rendered palette. The risk it introduces is the opposite of
the bug: that switching BACK to Default gets swallowed. Prove it does not.
"""
import sys
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8050/1"
fails = []


def colors(pg):
    return pg.evaluate("""() => {
        var gd = document.querySelector('#bubble-graph .js-plotly-plot');
        return gd && gd.data ? gd.data.map(function(t){
            return (t.line && t.line.color) || ''; }).filter(Boolean).slice(0,3) : []; }""")


def stored(pg):
    return pg.evaluate("() => window.localStorage.getItem('palette-store')")


def load(pg):
    pg.goto(BASE, wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_selector("#bubble-graph .js-plotly-plot", timeout=40000)
    pg.wait_for_timeout(13000)


with sync_playwright() as p:
    br = p.firefox.launch(headless=True)
    ctx = br.new_context(viewport={"width": 1400, "height": 900})
    pg = ctx.new_page()

    load(pg)
    c_default = colors(pg)
    print(f"1. fresh load          colors={c_default}  ls={stored(pg)}")

    pg.select_option("#palette-select-bub", "cb-brian")
    pg.wait_for_timeout(9000)
    c_cb = colors(pg)
    print(f"2. switch to cb-brian  colors={c_cb}  ls={stored(pg)}")
    if c_cb == c_default:
        fails.append("switching to cb-brian did not repaint")

    load(pg)
    c_reload = colors(pg)
    print(f"3. reload              colors={c_reload}  ls={stored(pg)}")
    if c_reload != c_cb:
        fails.append(f"reload lost the palette: {c_reload} != {c_cb}")
    if stored(pg) != '"cb-brian"':
        fails.append(f"localStorage not persisted: {stored(pg)}")

    # the regression risk of the first-fire guard
    pg.select_option("#palette-select-bub", "default")
    pg.wait_for_timeout(9000)
    c_back = colors(pg)
    print(f"4. switch back to def  colors={c_back}  ls={stored(pg)}")
    if c_back != c_default:
        fails.append(f"switching BACK to default was swallowed: {c_back}")

    load(pg)
    c_final = colors(pg)
    print(f"5. reload              colors={c_final}  ls={stored(pg)}")
    if c_final != c_default:
        fails.append(f"default did not persist: {c_final}")

    ctx.close()
    br.close()

print()
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PASS — palette round trip persists in both directions")
