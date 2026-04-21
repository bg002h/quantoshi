"""Static HTML pages for deep-link routes (/faq, /mi).

Renders Dash component trees to plain HTML at startup, serves via Flask.
No React, no Dash renderer, no Plotly.js — single HTTP round-trip.
"""
import re
from html import escape as _esc
from colors import SPLASH_BRAND_DARK

_CAMEL_RE = re.compile(r"([a-z])([A-Z])")

def _style_to_css(style_dict):
    if not style_dict:
        return ""
    parts = []
    for k, v in style_dict.items():
        css_prop = _CAMEL_RE.sub(r"\1-\2", k).lower()
        parts.append(f"{css_prop}:{v}")
    return ";".join(parts)

_VOID_TAGS = frozenset(["br", "hr", "img", "input", "meta", "link"])
_TAG_MAP = {"I": "i"}

def _dash_to_html(component):
    if component is None:
        return ""
    if isinstance(component, str):
        return _esc(component)
    if isinstance(component, (int, float)):
        return _esc(str(component))
    if isinstance(component, list):
        return "".join(_dash_to_html(c) for c in component)

    type_name = type(component).__name__
    if type_name == "Markdown":
        return _render_markdown(component)
    if type_name == "Accordion":
        return _render_accordion(component)
    if type_name == "AccordionItem":
        return _render_accordion_item(component, 0)
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
    if type_name == "Modal":
        return ""

    tag = _TAG_MAP.get(type_name, type_name.lower())
    attrs = []
    comp_id = getattr(component, "id", None)
    if isinstance(comp_id, str):
        attrs.append(f'id="{_esc(comp_id)}"')
    cls = getattr(component, "className", None)
    if cls:
        attrs.append(f'class="{_esc(cls)}"')
    style = getattr(component, "style", None)
    if style and isinstance(style, dict):
        attrs.append(f'style="{_esc(_style_to_css(style))}"')
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
    import markdown as md
    text = getattr(component, "children", "") or ""
    if isinstance(text, list):
        text = "\n".join(str(t) for t in text)
    cls = getattr(component, "className", "") or ""
    html_content = md.markdown(text, extensions=["tables", "fenced_code"])
    return f'<div class="{_esc(cls)}">{html_content}</div>'

def _render_accordion(component):
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
    <nav class="navbar navbar-dark" style="background:{nav_bg};padding:6px 16px">
        <div class="container-fluid">
            <a class="navbar-brand d-flex align-items-center" href="/" style="font-family:'Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif">
                <img src="/assets/quantoshi_logo_nav.png" height="40" class="me-2"> Quantoshi
            </a>
            <a href="/{app_tab}" class="btn btn-outline-light btn-sm">Open full app &rarr;</a>
        </div>
    </nav>
    {tab_bar}
    <div class="container mt-3">
        {content}
    </div>
    <script src="/assets/bootstrap.bundle.min.js"></script>
    <script src="/assets/faq_lightbox.js"></script>
    {foot_extra}
</body>
</html>"""


# Fake tab bar matching the live Dash app's chrome. Each tab is a plain
# anchor to its corresponding route — in-app, Dash's clientside URL router
# catches these via pushState; from the static page, a click is a real
# browser navigation into the Dash SPA. Uses stable named routes where
# available (/leverage, /mi, /faq) so links don't break on tab reorder.
from colors import BOOTSTRAP_BORDER as _TABBAR_BORDER, WHITE as _TABBAR_BG

_TAB_BAR_TEMPLATE = f"""
<ul class="nav nav-tabs static-tab-bar" style="padding:0 12px;margin:0;border-bottom:1px solid {_TABBAR_BORDER};background:{_TABBAR_BG};flex-wrap:nowrap;overflow-x:auto;">
  <li class="nav-item"><a class="nav-link {{cls_bubble}}"      href="/1">Price Models</a></li>
  <li class="nav-item"><a class="nav-link {{cls_heatmap}}"     href="/2">Heatmap</a></li>
  <li class="nav-item"><a class="nav-link {{cls_dca}}"         href="/3">Accumulator</a></li>
  <li class="nav-item"><a class="nav-link {{cls_retire}}"      href="/4">RetireMentator</a></li>
  <li class="nav-item"><a class="nav-link {{cls_supercharge}}" href="/5">Supercharger</a></li>
  <li class="nav-item"><a class="nav-link {{cls_citadel}}"     href="/6">Citadel</a></li>
  <li class="nav-item"><a class="nav-link {{cls_leverage}}"    href="/leverage">Max Pay-Price</a></li>
  <li class="nav-item"><a class="nav-link {{cls_stack}}"       href="/8">Stack</a></li>
  <li class="nav-item"><a class="nav-link {{cls_model_info}}"  href="/mi">Model Info</a></li>
  <li class="nav-item"><a class="nav-link {{cls_faq}}"         href="/faq">FAQ</a></li>
</ul>
"""


def _render_tab_bar(current_tab: str) -> str:
    """Return the fake tab-bar HTML with `current_tab` marked active."""
    keys = ("bubble", "heatmap", "dca", "retire", "supercharge",
            "citadel", "leverage", "stack", "model_info", "faq")
    return _TAB_BAR_TEMPLATE.format(
        **{f"cls_{k}": ("active" if k == current_tab else "") for k in keys}
    )

# MathJax has no acceptable same-origin replacement (the es5 bundle loads a
# dozen more scripts by itself). Only include it on clearnet; strip from onion
# so Tor users don't leak an IP/fingerprint to jsdelivr.  Onion visitors miss
# rendered LaTeX on /mi but the text content is still readable -- acceptable
# tradeoff per no-user-data-ever policy.
_MATHJAX_HEAD_CLEARNET = """<script>MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>"""
_MATHJAX_HEAD_ONION = ""  # strip remote script on onion

_SCROLL_SCRIPT = """<script>
document.addEventListener('DOMContentLoaded',function(){{
    var el=document.getElementById('{item_id}');
    if(el)el.scrollIntoView({{block:'start'}});
}});
</script>"""

# Cached rendered HTML (populated by render_static_pages())
_STATIC_FAQ_HTML = None
_STATIC_MI_HTML = None


def render_static_faq():
    """Render the FAQ accordion to a complete static HTML page."""
    from layout.faq import _faq_tab
    content_component = _faq_tab()
    content_html = _dash_to_html(content_component)
    return _PAGE_TEMPLATE.format(
        title="FAQ",
        head_extra="",
        app_tab="9",
        tab_bar=_render_tab_bar("faq"),
        content=content_html,
        foot_extra="",
        nav_bg=SPLASH_BRAND_DARK,
    )


def render_static_model_info(mathjax_head: str = ""):
    """Render the Model Info accordion to a complete static HTML page.

    mathjax_head: pass _MATHJAX_HEAD_CLEARNET for clearnet, "" for onion.
    """
    from layout.model_info import _model_info_tab
    content_component = _model_info_tab()
    content_html = _dash_to_html(content_component)
    return _PAGE_TEMPLATE.format(
        title="Model Info",
        head_extra=mathjax_head,
        app_tab="8",
        tab_bar=_render_tab_bar("model_info"),
        content=content_html,
        foot_extra="",
        nav_bg=SPLASH_BRAND_DARK,
    )


# Two rendered copies per page: one with MathJax for clearnet, one without
# for onion. Picked at request time by Host header.
_STATIC_MI_HTML_ONION = None  # no MathJax


def render_static_pages():
    """Render and cache both static pages. Call after models are loaded."""
    global _STATIC_FAQ_HTML, _STATIC_MI_HTML, _STATIC_MI_HTML_ONION
    _STATIC_FAQ_HTML = render_static_faq()
    _STATIC_MI_HTML = render_static_model_info(_MATHJAX_HEAD_CLEARNET)
    _STATIC_MI_HTML_ONION = render_static_model_info(_MATHJAX_HEAD_ONION)


def _open_accordion_item(html_str, item_id):
    """Patch the HTML to open a specific accordion item + scroll to it.

    Modifies the server-rendered HTML directly:
    1. Add 'show' class to the target collapse div
    2. Remove 'collapsed' from the button, set aria-expanded=true
    3. Append a scroll script
    """
    collapse_id = f"collapse-{item_id}"
    heading_id = f"heading-{item_id}"
    # Open the collapse div
    html_str = html_str.replace(
        f'id="{collapse_id}" class="accordion-collapse collapse"',
        f'id="{collapse_id}" class="accordion-collapse collapse show"',
    )
    # Fix button: remove 'collapsed' class and set aria-expanded=true
    html_str = html_str.replace(
        f'id="{heading_id}"><button class="accordion-button collapsed" '
        f'type="button" data-bs-toggle="collapse" '
        f'data-bs-target="#{collapse_id}" aria-expanded="false"',
        f'id="{heading_id}"><button class="accordion-button" '
        f'type="button" data-bs-toggle="collapse" '
        f'data-bs-target="#{collapse_id}" aria-expanded="true"',
    )
    # Append scroll script
    scroll = _SCROLL_SCRIPT.format(item_id=item_id)
    html_str = html_str.replace("</body>", scroll + "</body>")
    return html_str


# Clearnet CSP: allow jsdelivr for MathJax only (clearnet /mi). Bootstrap is
# now self-hosted so we don't allow script/style from the CDN anymore.
_STATIC_CSP_CLEARNET = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline'; "
    "font-src 'self'; "
    "connect-src 'self' https://cdn.jsdelivr.net; "
    "img-src 'self' data: blob:; "
    "frame-ancestors 'none'; "
    "base-uri 'self'"
)
# Onion CSP: strict 'self'. No CDN, no remote connections, no fingerprint
# leak. MathJax has been stripped from the onion HTML above.
_STATIC_CSP_ONION = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "img-src 'self' data: blob:; "
    "frame-ancestors 'none'; "
    "base-uri 'self'"
)


def _is_onion_request():
    """True when the current Flask request arrived via the .onion hostname."""
    from flask import request
    host = (request.host or "").lower()
    return host.endswith(".onion")


def _static_headers(extra=None):
    h = {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "public, max-age=86400",
        "Content-Security-Policy": _STATIC_CSP_ONION if _is_onion_request()
                                    else _STATIC_CSP_CLEARNET,
    }
    if extra:
        h.update(extra)
    return h


def _ensure_rendered():
    """On-demand render if the background warm thread hasn't completed yet.

    Eliminates the "Static pages not yet rendered" race during the first
    few seconds after a restart. First visitor pays the ~2s render cost;
    subsequent requests hit the cache.
    """
    if _STATIC_FAQ_HTML is None or _STATIC_MI_HTML is None or _STATIC_MI_HTML_ONION is None:
        render_static_pages()


_RESEARCH_IMAGES = [
    ("blocksweep.jpg",                   "BM floor + band model sweep — R² and exponent vs block offset"),
    ("bandmodel_fits.jpg",               "Band model fits — offset = 0"),
    ("bandmodel_fits_offset18750.jpg",   "Band model fits — offset = 18,750 blocks"),
    ("temporal_sweep.jpg",               "Temporal sweep — A2 consistency and A1 floor scores vs offset"),
    ("temporal_fits.jpg",                "Best temporal fits at optimal (Q-level, offset)"),
    ("qstar_sweep.jpg",                  "q*(offset) sweep — convergence-quantile method"),
    ("qstar_fits.jpg",                   "q*-crossing fits at optimal offset per weight mode"),
    ("qstar_3d.jpg",                     "3D cvar surface — log-density weighted"),
    ("qstar_fine_3d.jpg",                "Fine-resolution 3D cvar surface"),
    ("qstar_slices.jpg",                 "cvar vs offset — fixed Q-level slices"),
    ("qstar_valley_width.jpg",           "Valley width analysis across quantile levels"),
    ("qstar_crossing_score.jpg",         "Crossing-symmetry score vs offset"),
    ("qstar_crossing_fan.jpg",           "Quantile fan — offset=0 vs optimal offset"),
]

_RESEARCH_HTML = None

def _build_research_html():
    imgs = "".join(
        f'<figure style="margin:0 0 2.5rem 0">'
        f'<img src="/assets/research/{fname}" style="width:100%;border-radius:6px;display:block">'
        f'<figcaption style="margin-top:.5rem;font-size:.8rem;color:#aaa;line-height:1.4">{_esc(caption)}</figcaption>'
        f'</figure>'
        for fname, caption in _RESEARCH_IMAGES
    )
    return (
        "<!doctype html><html><head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Block-offset research — Quantoshi</title>"
        "<style>"
        "body{margin:0;padding:1rem;background:#111;color:#ddd;font-family:sans-serif;max-width:900px;margin:0 auto}"
        "h1{font-size:1.2rem;color:#FFB300;margin-bottom:.3rem}"
        "p.sub{font-size:.8rem;color:#888;margin-bottom:2rem}"
        "</style></head><body>"
        "<h1>Block-offset exploration — April 2026</h1>"
        "<p class='sub'>Bitcoin power law floor model research. "
        "<a href='/' style='color:#FFB300'>← Quantoshi</a></p>"
        + imgs +
        "</body></html>"
    )


def register_static_routes(server):
    """Register /faq, /faq.N, /mi, /mi.N, /Z Flask routes."""

    @server.route("/faq")
    def _static_faq():
        _ensure_rendered()
        return _STATIC_FAQ_HTML, 200, _static_headers()

    @server.route("/faq.<int:item>")
    def _static_faq_item(item):
        _ensure_rendered()
        item_id = f"faq-{item - 1}"  # URL is 1-indexed, internal is 0-indexed
        page = _open_accordion_item(_STATIC_FAQ_HTML, item_id)
        return page, 200, _static_headers()

    @server.route("/mi")
    def _static_mi():
        _ensure_rendered()
        html_cache = _STATIC_MI_HTML_ONION if _is_onion_request() else _STATIC_MI_HTML
        return html_cache, 200, _static_headers()

    @server.route("/mi.<int:item>")
    def _static_mi_item(item):
        _ensure_rendered()
        html_cache = _STATIC_MI_HTML_ONION if _is_onion_request() else _STATIC_MI_HTML
        from layout.model_info import _MODEL_INFO_ITEM_IDS
        if 1 <= item <= len(_MODEL_INFO_ITEM_IDS):
            item_id = _MODEL_INFO_ITEM_IDS[item - 1]
        else:
            item_id = f"item-{item}"
        page = _open_accordion_item(html_cache, item_id)
        return page, 200, _static_headers()

    @server.route("/Z")
    def _research_gallery():
        global _RESEARCH_HTML
        if _RESEARCH_HTML is None:
            _RESEARCH_HTML = _build_research_html()
        return _RESEARCH_HTML, 200, _static_headers()
