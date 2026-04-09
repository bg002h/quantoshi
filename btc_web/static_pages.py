"""Static HTML pages for deep-link routes (/faq, /mi).

Renders Dash component trees to plain HTML at startup, serves via Flask.
No React, no Dash renderer, no Plotly.js — single HTTP round-trip.
"""
import re
from html import escape as _esc

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


def register_static_routes(server):
    """Register /faq, /faq.N, /mi, /mi.N Flask routes."""

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
        item_id = f"faq-{item - 1}"  # URL is 1-indexed, internal is 0-indexed
        page = _open_accordion_item(_STATIC_FAQ_HTML, item_id)
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
        from layout.model_info import _MODEL_INFO_ITEM_IDS
        if 1 <= item <= len(_MODEL_INFO_ITEM_IDS):
            item_id = _MODEL_INFO_ITEM_IDS[item - 1]
        else:
            item_id = f"item-{item}"
        page = _open_accordion_item(_STATIC_MI_HTML, item_id)
        return page, 200, {
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "public, max-age=86400",
        }
