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
    assert "<div>" in result and "content" in result and "</div>" in result

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
    assert "<table>" in result and "<th>" in result and "<td>1</td>" in result

def test_html_list():
    from static_pages import _dash_to_html
    result = _dash_to_html(html.Ul([html.Li("a"), html.Li("b")]))
    assert "<ul>" in result and "<li>a</li>" in result

def test_nested_children():
    from static_pages import _dash_to_html
    comp = html.Span(["text ", html.Strong("bold"), " more"])
    result = _dash_to_html(comp)
    assert "text <strong>bold</strong> more" in result

def test_list_children():
    from static_pages import _dash_to_html
    result = _dash_to_html([html.P("a"), html.P("b")])
    assert "<p>a</p>" in result and "<p>b</p>" in result

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
    assert "col-8" in result and "offset-2" in result

def test_col_with_int_width():
    from static_pages import _dash_to_html
    result = _dash_to_html(dbc.Col("x", width=6))
    assert "col-6" in result

def test_modal_skipped():
    from static_pages import _dash_to_html
    result = _dash_to_html(dbc.Modal("hidden content", is_open=False))
    assert result == ""


def test_render_static_faq_returns_html():
    from static_pages import render_static_faq
    html_str = render_static_faq()
    assert "<!DOCTYPE html>" in html_str
    assert "Frequently Asked Questions" in html_str
    assert "quantoshi_logo_nav.png" in html_str
    assert "bootstrap_flatly.min.css" in html_str
    assert "faq_lightbox.js" in html_str
    assert "accordion" in html_str
    assert "mathjax" not in html_str.lower()


def test_faq_has_all_items():
    from static_pages import render_static_faq
    from layout.faq import _FAQ
    html_str = render_static_faq()
    for entry in _FAQ:
        assert entry["q"][:30] in html_str, f"Missing FAQ: {entry['q'][:40]}"


def test_faq_item_ids():
    from static_pages import render_static_faq
    html_str = render_static_faq()
    assert 'id="faq-0"' in html_str
    assert 'id="faq-1"' in html_str
