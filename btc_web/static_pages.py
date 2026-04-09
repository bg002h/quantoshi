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
