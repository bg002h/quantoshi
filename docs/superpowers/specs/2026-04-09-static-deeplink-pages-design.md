# Static Deep-Link Pages for FAQ and Model Info

**Date:** 2026-04-09
**Status:** Draft

## Problem

Visiting `/faq` or `/9.4` in the Quantoshi Dash app shows a 1-2 second flash of tab 1 (bubble chart) before switching to the requested tab. Root cause: Dash does zero server-side rendering. The HTML page is an empty `<div>Loading...</div>` shell. The DashRenderer JS fetches the layout JSON via a second XHR to `/_dash-layout`, then React mounts the full component tree. During the ~1-2s between page load and React mounting, the browser shows the loading placeholder, then React renders all tab panes — briefly showing the first (bubble) tab before Bootstrap applies the `active_tab` class.

Multiple CSS/JS overlay approaches failed because they target server-rendered HTML that doesn't exist — React creates the DOM from scratch client-side.

## Solution

Serve `/faq`, `/faq.N`, `/mi`, and `/mi.N` as **standalone Flask routes** returning pre-rendered static HTML. No React, no Dash renderer, no Plotly.js. Content renders in a single HTTP round-trip (~50ms).

The existing `/8`, `/8.N`, `/9`, `/9.N` routes continue to load the full Dash SPA as before.

**Breaking change:** `/faq` and `/faq.N` currently load the full Dash SPA with the FAQ tab active. After this change, they serve a static HTML page instead. Users with bookmarked `/faq` links will get the static page (faster, but no price ticker or tab switching). `/9` and `/9.N` remain unchanged for users who want the full SPA experience.

## Routes

| Route | Content | Scroll behavior |
|-------|---------|----------------|
| `/faq` | Full FAQ page, all items collapsible | Top of page |
| `/faq.N` | Same page | Auto-scroll to item N |
| `/mi` | Full Model Info page, all items collapsible | Top of page |
| `/mi.N` | Same page | Auto-scroll to item N |

## Architecture

### Rendering pipeline

1. **At app startup** (after models load in `app.py`), call `render_static_faq()` and `render_static_model_info()`.
2. These functions walk the existing Dash component trees (`_FAQ` list in `layout/faq.py`, accordion builder in `layout/model_info.py`) and serialize them to plain HTML using a recursive `_dash_to_html()` serializer.
3. The rendered HTML strings are cached in module-level variables (`_STATIC_FAQ_HTML`, `_STATIC_MI_HTML`).
4. Flask routes (`/faq`, `/faq.N`, `/mi`, `/mi.N`) return the cached HTML wrapped in a minimal page template.

### Serializer: `_dash_to_html(component) → str`

Located in a new module `btc_web/static_pages.py`.

Recursive function that converts Dash component trees to HTML strings:

| Dash component | HTML output |
|---------------|-------------|
| `html.Div`, `html.Span`, `html.P`, `html.H5`, `html.H6` | `<div>`, `<span>`, `<p>`, `<h5>`, `<h6>` |
| `html.Strong`, `html.I`, `html.Em`, `html.B`, `html.Br`, `html.Hr` | `<strong>`, `<i>`, `<em>`, `<b>`, `<br>`, `<hr>` |
| `html.Code`, `html.Sub` | `<code>`, `<sub>` |
| `html.Ul`, `html.Ol`, `html.Li` | `<ul>`, `<ol>`, `<li>` |
| `html.Table`, `html.Thead`, `html.Tbody`, `html.Tr`, `html.Th`, `html.Td` | Direct HTML table tags |
| `html.Img` | `<img>` with `src`, `style` attributes |
| `html.A` | `<a>` with `href`, `target` attributes |
| `dbc.Accordion` | `<div class="accordion" id="...">` |
| `dbc.AccordionItem` | Bootstrap 5 accordion-item markup (header + collapsible body) |
| `dbc.Row`, `dbc.Col` | `<div class="row">`, `<div class="col">` |
| `dcc.Markdown(mathjax=True)` | Render markdown to HTML via Python `markdown` library; LaTeX `$$..$$` blocks left intact for client-side MathJax |
| `str` (plain text) | HTML-escaped text node |
| `list` | Concatenate serialized children |

**Generic fallback:** Any `html.*` component not explicitly listed is serialized as `<tagname>...</tagname>` using the lowercase class name. This covers edge cases without maintaining an exhaustive list.

**Prop handling:**
- `style` dicts converted to CSS strings: `{"fontSize": "13px", "marginBottom": "12px"}` → `font-size:13px;margin-bottom:12px`
- `className` prop maps to HTML `class` attribute (critical for Bootstrap utility classes like `mb-3`, `text-muted`, etc.)
- `id` prop: string IDs map directly to `id="..."`. Dict IDs (pattern-matching, e.g., `{"type": "mi-img", "src": ...}`) are serialized as `data-dash-id` and ignored for anchor linking.
- `href`, `target`, `src`, `alt` pass through directly.

### Page template

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Quantoshi — {page_title}</title>
    <link rel="icon" type="image/png" href="/assets/quantoshi_favicon.png">
    <link rel="stylesheet" href="/assets/bootstrap_flatly.min.css">
    <link rel="stylesheet" href="/assets/style.css">
    {mathjax_script if needed}
</head>
<body>
    <nav class="navbar navbar-dark" style="background:#2c3e50">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">
                <img src="/assets/quantoshi_logo_nav.png" height="40"> Quantoshi
            </a>
            <a href="/{full_app_tab}" class="btn btn-outline-light btn-sm">Open full app →</a>
        </div>
    </nav>
    <div class="container mt-3">
        {rendered_content}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="/assets/faq_lightbox.js"></script>
    {scroll_script if .N route}
</body>
</html>
```

### MathJax (Model Info only)

Model Info uses `dcc.Markdown(mathjax=True)` for LaTeX formulas (21 occurrences). Include MathJax 3 via CDN:

```html
<script>
MathJax = {tex: {inlineMath: [['$','$'], ['\\(','\\)']]}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
```

FAQ has no LaTeX — MathJax is not loaded on FAQ pages.

### Live model coefficients (Model Info only)

Model Info pulls live coefficient values from `_app_ctx.PRICE_MODELS` (e.g., `model._A`, `model._sigma`). The static page must be rendered **after models load**. The render call goes in `app.py` after model registration, inside the existing `_prewarm_caches()` flow or immediately after it.

### Image lightbox

Reuse `btc_web/assets/faq_lightbox.js` (already written). It attaches click handlers to `<img>` elements inside `.accordion-body` for click-to-enlarge. Works with plain HTML — no Dash callbacks needed.

Model Info's `_clickable_img()` uses Dash pattern-matching IDs (`{"type": "mi-img", "src": src}`) and a `dbc.Modal` callback for lightbox in the SPA. On static pages, these are rendered as plain `<img>` tags with `cursor: zoom-in` style. The `faq_lightbox.js` handles click-to-enlarge via CSS class toggle — it targets any `<img>` inside `.accordion-body`, so it works identically for both FAQ and Model Info images without the Dash callback.

### Scroll-to-item for `.N` routes

For `/faq.3` or `/mi.5`, append a small inline script:

```html
<script>
document.addEventListener('DOMContentLoaded', function() {
    var el = document.getElementById('item-3');
    if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
});
</script>
```

Each accordion item gets `id="item-N"` (1-indexed, matching existing URL convention).

### Caching

- Flask response headers: `Cache-Control: public, max-age=86400` (24h)
- Browser caches the page for 24h — repeat visits are instant from disk cache
- nginx is a reverse proxy without `proxy_cache_path` configured, so it does not cache at the proxy level. The `Cache-Control` header is passed through to the browser. If nginx proxy caching is added later, these pages will benefit automatically.
- Content regenerates on each app restart (deploy). The 24h browser cache TTL is conservative enough that stale content after a deploy is acceptable (FAQ/Model Info change infrequently).

### Bootstrap CSS accordion markup

Each `dbc.AccordionItem` serializes to:

```html
<div class="accordion-item" id="item-N">
    <h2 class="accordion-header">
        <button class="accordion-button" type="button"
                data-bs-toggle="collapse" data-bs-target="#collapse-N">
            {title}
        </button>
    </h2>
    <div id="collapse-N" class="accordion-collapse collapse show">
        <div class="accordion-body">
            {body content}
        </div>
    </div>
</div>
```

Bootstrap 5 JS (collapse plugin) is included in the page template for accordion toggle functionality.

## Files

| File | Action | Purpose |
|------|--------|---------|
| `btc_web/static_pages.py` | Create | `_dash_to_html()` serializer, `render_static_faq()`, `render_static_model_info()`, page template, Flask route registration |
| `btc_web/api.py` | Modify | Register `/faq`, `/faq.N`, `/mi`, `/mi.N` routes (or delegate to static_pages) |
| `btc_web/app.py` | Modify | Call static page rendering after model load |
| `btc_web/assets/faq_lightbox.js` | No change | Already works with plain HTML |
| `btc_web/assets/style.css` | Minor | Ensure `.accordion a` color rule applies outside Dash context |

## What these pages DON'T include

- No React, Dash renderer, or Plotly.js (~5MB JS saved)
- No tabs, no callbacks, no price ticker, no snapshot/share
- No `dcc.Store`, `dcc.Interval`, or any client-side state

## Navigation

- Static page navbar: logo → `/`, "Open full app →" → `/9` (FAQ) or `/8` (Model Info)
- Each accordion item has a copyable anchor `#item-N`
- Existing `/8`, `/8.N`, `/9`, `/9.N` routes unchanged — load full Dash app as before

## Success criteria

1. `/faq` renders FAQ content in <200ms (single round-trip, no JS required)
2. `/faq.N` scrolls to the correct accordion item
3. `/mi` renders Model Info with LaTeX formulas (MathJax loads async)
4. `/mi.N` scrolls to the correct accordion item
5. All images are clickable to enlarge
6. Accordion items collapse/expand via Bootstrap JS
7. `/8`, `/9`, `/8.N`, `/9.N` continue to work as before
8. nginx caches the static pages (verified via response headers)
