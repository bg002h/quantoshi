# Static Deep-Link Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `/faq`, `/faq.N`, `/mi`, and `/mi.N` as pre-rendered static HTML pages (no React/Dash) to eliminate the 1-2s tab-1 flash on deep links.

**Architecture:** A recursive `_dash_to_html()` serializer converts existing Dash component trees to plain HTML at startup. Flask routes serve the cached HTML with a minimal Bootstrap page template. `dcc.Markdown(mathjax=True)` is rendered via Python `markdown` lib with LaTeX left intact for client-side MathJax.

**Tech Stack:** Python, Flask, dash-bootstrap-components (read-only), markdown lib, MathJax 3 CDN, Bootstrap 5.3 CSS/JS

---

## Context

- Spec: `docs/superpowers/specs/2026-04-09-static-deeplink-pages-design.md`
- FAQ layout: `btc_web/layout/faq.py` — `_FAQ` list of `{"q": str, "a": str|html.Span}`, `_faq_tab()` builds `dbc.Accordion`
- Model Info layout: `btc_web/layout/model_info.py` — `_model_info_tab()` builds `dbc.Accordion` with 26 items, uses `dcc.Markdown(mathjax=True)` (21 times), live model coefficients from `_app_ctx.PRICE_MODELS`
- Flask server: `btc_web/app.py` line 77: `server = app.server`
- Routes registered in: `btc_web/api.py` via `register_routes(server)`
- Flatly Bootstrap CSS: `btc_web/assets/bootstrap_flatly.min.css`
- Lightbox JS: `btc_web/assets/faq_lightbox.js`
- Python venv: `btc_venv/bin/python3`
- Test: `btc_venv/bin/python3 -m pytest btc_web/test_web.py -x -q`

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `btc_web/static_pages.py` | Create | Serializer, renderers, page template, route registration |
| `btc_web/app.py` | Modify (after line ~391) | Call `render_static_pages()` after model load |
| `btc_web/api.py` | Modify (inside `register_routes`) | Import and call `register_static_routes(server)` |
| `btc_web/test_static_pages.py` | Create | Tests for serializer + routes |

---

### Task 1: Create the `_dash_to_html()` serializer

**Files:**
- Create: `btc_web/static_pages.py`
- Create: `btc_web/test_static_pages.py`

- [ ] **Step 1: Write test for basic HTML serialization**

```python
# btc_web/test_static_pages.py
"""Tests for static page HTML serializer."""
import pytest
from dash import html
import dash_bootstrap_components as dbc


def test_plain_string():
    from static_pages import _dash_to_html
    assert _dash_to_html("hello") == "hello"


def test_html_escape():
    from static_pages import _dash_to_html
    assert _dash_to_html("a < b & c") == "a &lt; b &amp; c"


def test_html_div():
    from static_pages import _dash_to_html
    result = _dash_to_html(html.Div("content"))
    assert "<div>" in result
    assert "content" in result
    assert "</div>" in result


def test_html_div_with_class():
    from static_pages import _dash_to_html
    result = _dash_to_html(html.Div("x", className="mb-3 text-muted"))
    assert 'class="mb-3 text-muted"' in result


def test_html_div_with_style():
    from static_pages import _dash_to_html
    result = _dash_to_html(html.Div("x", style={"fontSize": "13px", "marginBottom": "12px"}))
    assert "font-size:13px" in result
    assert "margin-bottom:12px" in result


def test_html_img():
    from static_pages import _dash_to_html
    result = _dash_to_html(html.Img(src="/assets/logo.png", style={"width": "100%"}))
    assert 'src="/assets/logo.png"' in result
    assert "width:100%" in result
    assert "<img" in result


def test_html_br():
    from static_pages import _dash_to_html
    assert _dash_to_html(html.Br()) == "<br>"


def test_html_hr():
    from static_pages import _dash_to_html
    assert _dash_to_html(html.Hr()) == "<hr>"


def test_html_strong():
    from static_pages import _dash_to_html
    result = _dash_to_html(html.Strong("bold"))
    assert "<strong>bold</strong>" == result


def test_html_table():
    from static_pages import _dash_to_html
    tbl = html.Table([
        html.Thead(html.Tr([html.Th("A"), html.Th("B")])),
        html.Tbody([html.Tr([html.Td("1"), html.Td("2")])])
    ])
    result = _dash_to_html(tbl)
    assert "<table>" in result
    assert "<th>" in result
    assert "<td>1</td>" in result


def test_html_list():
    from static_pages import _dash_to_html
    result = _dash_to_html(html.Ul([html.Li("a"), html.Li("b")]))
    assert "<ul>" in result
    assert "<li>a</li>" in result


def test_nested_children():
    from static_pages import _dash_to_html
    comp = html.Span(["text ", html.Strong("bold"), " more"])
    result = _dash_to_html(comp)
    assert "text <strong>bold</strong> more" in result


def test_list_children():
    from static_pages import _dash_to_html
    result = _dash_to_html([html.P("a"), html.P("b")])
    assert "<p>a</p>" in result
    assert "<p>b</p>" in result


def test_string_id():
    from static_pages import _dash_to_html
    result = _dash_to_html(html.Div("x", id="my-div"))
    assert 'id="my-div"' in result


def test_dict_id_ignored():
    from static_pages import _dash_to_html
    result = _dash_to_html(html.Img(src="x.png", id={"type": "mi-img", "src": "x.png"}))
    assert 'id=' not in result


def test_none_returns_empty():
    from static_pages import _dash_to_html
    assert _dash_to_html(None) == ""


def test_col_with_width_offset():
    from static_pages import _dash_to_html
    result = _dash_to_html(dbc.Col("x", width={"size": 8, "offset": 2}))
    assert "col-8" in result
    assert "offset-2" in result


def test_col_with_int_width():
    from static_pages import _dash_to_html
    result = _dash_to_html(dbc.Col("x", width=6))
    assert "col-6" in result


def test_modal_skipped():
    from static_pages import _dash_to_html
    result = _dash_to_html(dbc.Modal("hidden content", is_open=False))
    assert result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd btc_web && PYTHONPATH=".:../:../archive/btc_app" ../btc_venv/bin/python3 -m pytest test_static_pages.py -x -q
```
Expected: ImportError — `static_pages` module doesn't exist yet.

- [ ] **Step 3: Implement `_dash_to_html()` serializer**

```python
# btc_web/static_pages.py
"""Static HTML pages for deep-link routes (/faq, /mi).

Renders Dash component trees to plain HTML at startup, serves via Flask.
No React, no Dash renderer, no Plotly.js — single HTTP round-trip.
"""
import re
from html import escape as _esc


# ── camelCase → kebab-case for CSS properties ──────────────────────────
_CAMEL_RE = re.compile(r"([a-z])([A-Z])")


def _style_to_css(style_dict):
    """Convert a Dash style dict to a CSS string."""
    if not style_dict:
        return ""
    parts = []
    for k, v in style_dict.items():
        css_prop = _CAMEL_RE.sub(r"\1-\2", k).lower()
        parts.append(f"{css_prop}:{v}")
    return ";".join(parts)


# Tags that are self-closing (no children, no closing tag)
_VOID_TAGS = frozenset(["br", "hr", "img", "input", "meta", "link"])

# Dash component type name → HTML tag (lowercase)
# Most html.* components map directly; this handles special cases.
_TAG_MAP = {
    "I": "i",  # html.I → <i>, not <i> (already lowercase, but explicit)
}


def _dash_to_html(component):
    """Recursively convert a Dash component tree to an HTML string."""
    if component is None:
        return ""
    if isinstance(component, str):
        return _esc(component)
    if isinstance(component, (int, float)):
        return _esc(str(component))
    if isinstance(component, list):
        return "".join(_dash_to_html(c) for c in component)

    # ── dcc.Markdown ───────────────────────────────────────────────────
    type_name = type(component).__name__
    if type_name == "Markdown":
        return _render_markdown(component)

    # ── dbc.Accordion ──────────────────────────────────────────────────
    if type_name == "Accordion":
        return _render_accordion(component)
    if type_name == "AccordionItem":
        return _render_accordion_item(component, 0)

    # ── dbc.Row / dbc.Col ──────────────────────────────────────────────
    if type_name == "Row":
        cls = getattr(component, "className", None) or ""
        cls = f"row {cls}".strip()
        inner = _dash_to_html(getattr(component, "children", None))
        return f'<div class="{_esc(cls)}">{inner}</div>'
    if type_name == "Col":
        cls = getattr(component, "className", None) or ""
        width = getattr(component, "width", None)
        if isinstance(width, dict):
            cls += f" col-{width.get('size', '')}"
            if "offset" in width:
                cls += f" offset-{width['offset']}"
        elif isinstance(width, int):
            cls += f" col-{width}"
        else:
            cls = f"col {cls}"
        cls = cls.strip()
        inner = _dash_to_html(getattr(component, "children", None))
        return f'<div class="{_esc(cls)}">{inner}</div>'

    # ── dbc.Modal — skip entirely (lightbox handled by JS) ─────────
    if type_name == "Modal":
        return ""

    # ── Generic html.* components ──────────────────────────────────────
    tag = _TAG_MAP.get(type_name, type_name.lower())

    # Build attributes
    attrs = []
    # id — only string IDs
    comp_id = getattr(component, "id", None)
    if isinstance(comp_id, str):
        attrs.append(f'id="{_esc(comp_id)}"')

    # className → class
    cls = getattr(component, "className", None)
    if cls:
        attrs.append(f'class="{_esc(cls)}"')

    # style
    style = getattr(component, "style", None)
    if style and isinstance(style, dict):
        attrs.append(f'style="{_esc(_style_to_css(style))}"')

    # Common passthrough attributes
    for attr in ("href", "target", "src", "alt", "colSpan", "rowSpan"):
        val = getattr(component, attr, None)
        if val is not None:
            html_attr = attr.lower() if attr in ("colSpan", "rowSpan") else attr
            attrs.append(f'{html_attr}="{_esc(str(val))}"')

    attr_str = (" " + " ".join(attrs)) if attrs else ""

    if tag in _VOID_TAGS:
        return f"<{tag}{attr_str}>"

    children = getattr(component, "children", None)
    inner = _dash_to_html(children)
    return f"<{tag}{attr_str}>{inner}</{tag}>"


def _render_markdown(component):
    """Render dcc.Markdown to HTML. LaTeX $$...$$ left intact for MathJax."""
    import markdown as md
    text = getattr(component, "children", "") or ""
    if isinstance(text, list):
        text = "\n".join(str(t) for t in text)
    cls = getattr(component, "className", "") or ""
    html_content = md.markdown(text, extensions=["tables", "fenced_code"])
    return f'<div class="{_esc(cls)}">{html_content}</div>'


def _render_accordion(component):
    """Render dbc.Accordion to Bootstrap 5 markup."""
    comp_id = getattr(component, "id", None) or "accordion"
    children = getattr(component, "children", []) or []
    if not isinstance(children, list):
        children = [children]
    flush = getattr(component, "flush", False)
    cls = "accordion accordion-flush" if flush else "accordion"
    inner = ""
    for i, child in enumerate(children):
        if type(child).__name__ == "AccordionItem":
            inner += _render_accordion_item(child, i, comp_id)
        else:
            inner += _dash_to_html(child)
    return f'<div class="{cls}" id="{_esc(str(comp_id))}">{inner}</div>'


def _render_accordion_item(component, index, parent_id="accordion"):
    """Render dbc.AccordionItem to Bootstrap 5 collapse markup."""
    item_id = getattr(component, "item_id", None) or f"item-{index}"
    title = getattr(component, "title", "") or ""
    children = getattr(component, "children", None)
    collapse_id = f"collapse-{item_id}"
    heading_id = f"heading-{item_id}"
    body = _dash_to_html(children)
    return (
        f'<div class="accordion-item" id="{_esc(item_id)}">'
        f'<h2 class="accordion-header" id="{_esc(heading_id)}">'
        f'<button class="accordion-button collapsed" type="button" '
        f'data-bs-toggle="collapse" data-bs-target="#{_esc(collapse_id)}" '
        f'aria-expanded="false" aria-controls="{_esc(collapse_id)}">'
        f'{_esc(title)}</button></h2>'
        f'<div id="{_esc(collapse_id)}" class="accordion-collapse collapse" '
        f'aria-labelledby="{_esc(heading_id)}" data-bs-parent="#{_esc(str(parent_id))}">'
        f'<div class="accordion-body">{body}</div></div></div>'
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd btc_web && PYTHONPATH=".:../:../archive/btc_app" ../btc_venv/bin/python3 -m pytest test_static_pages.py -x -q
```
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add btc_web/static_pages.py btc_web/test_static_pages.py
git commit -m "feat: _dash_to_html() serializer for static deep-link pages"
```

---

### Task 2: Add page template and rendering functions

**Files:**
- Modify: `btc_web/static_pages.py`
- Modify: `btc_web/test_static_pages.py`

- [ ] **Step 1: Write tests for page rendering**

Append to `btc_web/test_static_pages.py`:

```python
def test_render_static_faq_returns_html():
    from static_pages import render_static_faq
    html_str = render_static_faq()
    assert "<!DOCTYPE html>" in html_str
    assert "Frequently Asked Questions" in html_str
    assert "quantoshi_logo_nav.png" in html_str
    assert "bootstrap_flatly.min.css" in html_str
    assert "faq_lightbox.js" in html_str
    assert "accordion" in html_str
    # Should NOT contain MathJax (FAQ has no LaTeX)
    assert "mathjax" not in html_str.lower()


def test_render_static_mi_returns_html():
    from static_pages import render_static_model_info
    html_str = render_static_model_info()
    assert "<!DOCTYPE html>" in html_str
    assert "Price Models" in html_str
    assert "mathjax" in html_str.lower()  # Model Info needs MathJax
    assert "accordion" in html_str


def test_faq_has_all_items():
    from static_pages import render_static_faq
    from layout.faq import _FAQ
    html_str = render_static_faq()
    for entry in _FAQ:
        # Each question title should appear in the output
        assert entry["q"][:30] in html_str, f"Missing FAQ: {entry['q'][:40]}"


def test_faq_item_ids():
    from static_pages import render_static_faq
    html_str = render_static_faq()
    assert 'id="faq-0"' in html_str
    assert 'id="faq-1"' in html_str


def test_mi_item_ids():
    from static_pages import render_static_model_info
    html_str = render_static_model_info()
    assert 'id="mi-bub"' in html_str
    assert 'id="mi-eppl"' in html_str
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd btc_web && PYTHONPATH=".:../:../archive/btc_app" ../btc_venv/bin/python3 -m pytest test_static_pages.py::test_render_static_faq_returns_html -x -q
```
Expected: ImportError — `render_static_faq` doesn't exist yet.

- [ ] **Step 3: Implement page template and renderers**

Append to `btc_web/static_pages.py`:

```python
# ── Page template ──────────────────────────────────────────────────────

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Quantoshi — {title}</title>
    <link rel="icon" type="image/png" href="/assets/quantoshi_favicon.png">
    <link rel="stylesheet" href="/assets/bootstrap_flatly.min.css">
    <link rel="stylesheet" href="/assets/style.css">
    {head_extra}
</head>
<body>
    <nav class="navbar navbar-dark" style="background:#2c3e50;padding:6px 16px">
        <div class="container-fluid">
            <a class="navbar-brand d-flex align-items-center" href="/" style="font-family:'Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif">
                <img src="/assets/quantoshi_logo_nav.png" height="40" class="me-2"> Quantoshi
            </a>
            <a href="/{app_tab}" class="btn btn-outline-light btn-sm">Open full app &rarr;</a>
        </div>
    </nav>
    <div class="container mt-3">
        {content}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="/assets/faq_lightbox.js"></script>
    {foot_extra}
</body>
</html>"""

_MATHJAX_HEAD = """<script>MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']]}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>"""

_SCROLL_SCRIPT = """<script>
document.addEventListener('DOMContentLoaded',function(){{
    var el=document.getElementById('{item_id}');
    if(el){{el.scrollIntoView({{behavior:'smooth',block:'start'}});
    var btn=el.querySelector('.accordion-button');
    if(btn&&btn.classList.contains('collapsed'))btn.click();}}
}});
</script>"""

# Cached rendered HTML (populated by render_static_pages())
_STATIC_FAQ_HTML = None
_STATIC_MI_HTML = None


def render_static_faq():
    """Render the FAQ accordion to a complete static HTML page."""
    from layout.faq import _FAQ, _faq_tab
    content_component = _faq_tab()
    content_html = _dash_to_html(content_component)
    return _PAGE_TEMPLATE.format(
        title="FAQ",
        head_extra="",
        app_tab="9",
        content=content_html,
        foot_extra="",
    )


def render_static_model_info():
    """Render the Model Info accordion to a complete static HTML page."""
    from layout.model_info import _model_info_tab
    content_component = _model_info_tab()
    content_html = _dash_to_html(content_component)
    return _PAGE_TEMPLATE.format(
        title="Model Info",
        head_extra=_MATHJAX_HEAD,
        app_tab="8",
        content=content_html,
        foot_extra="",
    )


def render_static_pages():
    """Render and cache both static pages. Call after models are loaded."""
    global _STATIC_FAQ_HTML, _STATIC_MI_HTML
    _STATIC_FAQ_HTML = render_static_faq()
    _STATIC_MI_HTML = render_static_model_info()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd btc_web && PYTHONPATH=".:../:../archive/btc_app" ../btc_venv/bin/python3 -m pytest test_static_pages.py -x -q
```
Expected: All pass. (Some tests may need model data — if so, skip those with `@pytest.mark.skipif` and test them in Task 4.)

- [ ] **Step 5: Commit**

```bash
git add btc_web/static_pages.py btc_web/test_static_pages.py
git commit -m "feat: page template + render functions for static FAQ and Model Info"
```

---

### Task 3: Register Flask routes

**Files:**
- Modify: `btc_web/static_pages.py`
- Modify: `btc_web/api.py`

- [ ] **Step 1: Add route registration to static_pages.py**

Append to `btc_web/static_pages.py`:

```python
def register_static_routes(server):
    """Register /faq, /faq.N, /mi, /mi.N Flask routes."""
    import re

    @server.route("/faq")
    def _static_faq():
        if _STATIC_FAQ_HTML is None:
            return "Static pages not yet rendered", 503
        return _STATIC_FAQ_HTML, 200, {
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "public, max-age=86400",
        }

    @server.route("/faq.<int:item>")
    def _static_faq_item(item):
        if _STATIC_FAQ_HTML is None:
            return "Static pages not yet rendered", 503
        # Inject scroll script for the specific item
        item_id = f"faq-{item - 1}"  # URL is 1-indexed, internal is 0-indexed
        scroll = _SCROLL_SCRIPT.format(item_id=item_id)
        page = _STATIC_FAQ_HTML.replace("</body>", scroll + "</body>")
        return page, 200, {
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "public, max-age=86400",
        }

    @server.route("/mi")
    def _static_mi():
        if _STATIC_MI_HTML is None:
            return "Static pages not yet rendered", 503
        return _STATIC_MI_HTML, 200, {
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "public, max-age=86400",
        }

    @server.route("/mi.<int:item>")
    def _static_mi_item(item):
        if _STATIC_MI_HTML is None:
            return "Static pages not yet rendered", 503
        # Model Info items are identified by item_id, not index
        # Map 1-indexed URL to the Nth accordion item's id
        from layout.model_info import _MODEL_INFO_ITEM_IDS
        if 1 <= item <= len(_MODEL_INFO_ITEM_IDS):
            item_id = _MODEL_INFO_ITEM_IDS[item - 1]
        else:
            item_id = f"item-{item}"
        scroll = _SCROLL_SCRIPT.format(item_id=item_id)
        page = _STATIC_MI_HTML.replace("</body>", scroll + "</body>")
        return page, 200, {
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "public, max-age=86400",
        }
```

- [ ] **Step 2: Export `_MODEL_INFO_ITEM_IDS` from model_info.py**

Add at the end of `btc_web/layout/model_info.py` (after the `_model_info_tab` function):

```python
# Ordered list of accordion item_id values for /mi.N deep linking
_MODEL_INFO_ITEM_IDS = [
    "mi-bub", "mi-qr", "mi-pl", "mi-lppl", "mi-lp2",
    "mi-lppl-weighting", "mi-linppl", "mi-hybppl", "mi-hybppl-dd",
    "mi-hyb2l", "mi-hyb2c", "mi-hyb2b", "mi-hyb4d", "mi-pca",
    "mi-grdy", "mi-eppl", "mi-exp", "mi-gomp", "mi-bpl", "mi-s2f",
    "mi-mc", "mi-ef", "mi-u1", "mi-compare", "mi-regimes", "mi-citadel",
]
```

- [ ] **Step 3: Wire routes into api.py**

Add inside `register_routes(server)` in `btc_web/api.py`, at the top of the function body (before other routes, so Flask matches `/faq` before Dash's catch-all):

```python
    # Static deep-link pages (must be registered before Dash catch-all)
    from static_pages import register_static_routes
    register_static_routes(server)
```

- [ ] **Step 4: Wire rendering into app.py**

Add after `_prewarm_caches()` call (around line 391 in `btc_web/app.py`):

```python
    # Render static deep-link pages (after models are loaded)
    from static_pages import render_static_pages
    render_static_pages()
```

Also add the same call in the DEV path (where prewarm is skipped), so static pages work in dev mode too.

- [ ] **Step 5: Test manually**

```bash
cd /scratch/code/bitcoinprojections
DEV=1 bash run_web.sh &
sleep 4
curl -s http://localhost:8050/faq | head -5
curl -s http://localhost:8050/faq.4 | grep "scroll"
curl -s http://localhost:8050/mi | head -5
curl -s http://localhost:8050/mi.1 | grep "scroll"
kill %1
```

Expected: Each returns a full HTML page. `/faq.4` and `/mi.1` contain scroll scripts.

- [ ] **Step 6: Commit**

```bash
git add btc_web/static_pages.py btc_web/api.py btc_web/app.py btc_web/layout/model_info.py
git commit -m "feat: register /faq, /faq.N, /mi, /mi.N static Flask routes"
```

---

### Task 4: Handle edge cases and route conflicts

**Files:**
- Modify: `btc_web/layout/__init__.py`
- Modify: `btc_web/callbacks/routing.py`

- [ ] **Step 1: Ensure Dash doesn't intercept /faq and /mi**

The Flask routes registered in `api.py` run on the same Flask `server` as Dash. Flask matches routes in registration order — routes registered before Dash's catch-all will be matched first. Verify this works by checking that `register_routes(server)` is called before `app.layout = _serve_layout` in `app.py`.

Read `btc_web/app.py` to confirm the call order. If `register_routes` is called before layout assignment, Flask will match `/faq` and `/mi` before Dash's `/<path:path>` catch-all. No changes needed if the order is correct.

- [ ] **Step 2: Add /mi to _PATH_TO_TAB for SPA fallback**

In `btc_web/layout/__init__.py`, add `/mi` to `_PATH_TO_TAB` so that if static pages fail (503), the SPA can still handle the route:

```python
_PATH_TO_TAB = {
    "/1": "bubble", "/2": "heatmap", "/3": "dca",
    "/4": "retire", "/5": "supercharge", "/6": "citadel",
    "/7": "stack", "/8": "model_info", "/9": "faq",
    "/faq": "faq", "/mi": "model_info",
}
```

Add the same to the clientside callback JS map in `btc_web/callbacks/routing.py` (~line 308):

```javascript
var map = {"/1":"bubble","/2":"heatmap","/3":"dca",
           "/4":"retire","/5":"supercharge","/6":"citadel",
           "/7":"stack","/8":"model_info","/9":"faq",
           "/faq":"faq","/mi":"model_info"};
```

And add `/mi.N` pattern matching (~line 331):

```javascript
if (p && p.indexOf("/mi.") === 0) { return "model_info"; }
```

- [ ] **Step 3: Commit**

```bash
git add btc_web/layout/__init__.py btc_web/callbacks/routing.py
git commit -m "feat: add /mi route to SPA path maps as fallback"
```

---

### Task 5: Run tests and deploy

**Files:**
- No new files

- [ ] **Step 1: Run static page tests**

```bash
cd btc_web && PYTHONPATH=".:../:../archive/btc_app" ../btc_venv/bin/python3 -m pytest test_static_pages.py -x -q
```

- [ ] **Step 2: Run full test suite**

```bash
cd /scratch/code/bitcoinprojections && btc_venv/bin/python3 -m pytest -x -q --tb=short
```

- [ ] **Step 3: Syntax check the web app**

```bash
cd btc_web && PYTHONPATH=".:../:../archive/btc_app" ../btc_venv/bin/python3 -c \
  "import layout, figures, callbacks, cache, engines.adapter; print('OK')"
```

- [ ] **Step 4: Manual smoke test**

```bash
DEV=1 bash run_web.sh &
sleep 5
# Static pages
curl -sI http://localhost:8050/faq | head -5     # should be 200 + text/html
curl -sI http://localhost:8050/mi | head -5      # should be 200 + text/html
curl -sI http://localhost:8050/faq.4 | head -5   # should be 200
curl -sI http://localhost:8050/mi.1 | head -5    # should be 200
# SPA still works
curl -sI http://localhost:8050/9 | head -5       # should be 200 (Dash app)
curl -sI http://localhost:8050/8 | head -5       # should be 200 (Dash app)
kill %1
```

- [ ] **Step 5: Commit all remaining changes**

```bash
git add -u
git commit -m "feat: static deep-link pages for /faq and /mi — instant render"
```

- [ ] **Step 6: Ensure `markdown` is in requirements.txt**

```bash
grep -q "^[Mm]arkdown" btc_web/requirements.txt || echo "Markdown" >> btc_web/requirements.txt
```

Also install on prod if needed:
```bash
ssh root@89.167.70.45 "cd /opt/quantoshi && btc_venv/bin/pip install Markdown"
```

- [ ] **Step 7: Deploy to production**

```bash
git push origin master
ssh root@89.167.70.45 "cd /opt/quantoshi && git pull && redis-cli FLUSHDB && systemctl restart quantoshi"
```

- [ ] **Step 7: Verify in production**

Test in browser:
1. `https://quantoshi.xyz/faq` — FAQ renders instantly, no tab-1 flash
2. `https://quantoshi.xyz/faq.4` — scrolls to item 4
3. `https://quantoshi.xyz/mi` — Model Info renders with LaTeX formulas
4. `https://quantoshi.xyz/mi.1` — scrolls to Bubble Model
5. `https://quantoshi.xyz/9` — still loads full Dash SPA
6. `https://quantoshi.xyz/8` — still loads full Dash SPA
7. Click images in FAQ — lightbox works
8. Accordion items collapse/expand

---

## Verification Checklist

- [ ] `/faq` loads in <200ms (no JS required for content)
- [ ] `/mi` loads with LaTeX rendered (MathJax async)
- [ ] `/faq.N` and `/mi.N` scroll to correct item
- [ ] Images enlarge on click
- [ ] Accordions collapse/expand
- [ ] `/8`, `/9`, `/8.N`, `/9.N` unchanged
- [ ] `Cache-Control: public, max-age=86400` header present
- [ ] No React/Dash/Plotly JS loaded on static pages
