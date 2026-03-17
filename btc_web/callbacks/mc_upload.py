"""MC simulation save/load callbacks — upload JSON, download, modal dismiss."""

import json
import base64

import dash
from dash import Input, Output, State, callback

import _app_ctx


# ── MC simulation save (clientside JSON download) ────────────────────────────
_MC_FILENAME_JS = """
    function _mcFilename(tab, mc_data) {
        var pk = mc_data.path_key || {};
        var ok = mc_data.overlay_key || {};
        var parts = ['mc', tab];
        if (pk.mc_start_yr || pk.start_yr)
            parts.push('yr' + (pk.mc_start_yr || pk.start_yr));
        if (pk.mc_years) parts.push(pk.mc_years + 'y');
        var eq = pk.mc_entry_q || pk.entry_q;
        if (eq) parts.push('q' + Math.round(eq));
        if (pk.mc_bins) parts.push(pk.mc_bins + 'b');
        if (pk.mc_sims) parts.push(pk.mc_sims + 's');
        if (pk.mc_freq) parts.push(pk.mc_freq.charAt(0).toLowerCase());
        if (ok.mc_amount) parts.push('$' + ok.mc_amount);
        if (ok.mc_infl) parts.push(ok.mc_infl + 'pctInfl');
        if (ok.start_stack) parts.push(ok.start_stack + 'btc');
        var blocked = pk.mc_blocked_bins || [];
        if (blocked.length) parts.push('no' + blocked.join('_'));
        return parts.join('_') + '.json';
    }
"""

for _mc_prefix in ("dca", "ret", "hm", "sc"):
    _app_ctx.app.clientside_callback(
        """
        function(n_clicks, mc_data) {
            """ + _MC_FILENAME_JS + """
            if (!n_clicks || !mc_data) return window.dash_clientside.no_update;
            var json = JSON.stringify(mc_data, null, 2);
            var blob = new Blob([json], {type: 'application/json'});
            var url  = URL.createObjectURL(blob);
            var a    = document.createElement('a');
            a.href     = url;
            a.download = _mcFilename('""" + _mc_prefix + """', mc_data);
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            return window.dash_clientside.no_update;
        }
        """,
        Output(f"{_mc_prefix}-mc-dl-dummy", "data"),
        Input(f"{_mc_prefix}-mc-dl-btn",    "n_clicks"),
        State(f"{_mc_prefix}-mc-results",   "data"),
        prevent_initial_call=True,
    )


# ── MC save-prompt modal: dismiss closes modal ──────────────────────────────
@callback(
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Input("mc-save-modal-dismiss", "n_clicks"),
    prevent_initial_call=True,
)
def _mc_modal_dismiss(n):
    return False


# ── MC save-prompt modal: save triggers download then closes ────────────────
_app_ctx.app.clientside_callback(
    """
    function(n_clicks, tab, dca_data, ret_data, hm_data, sc_data) {
        """ + _MC_FILENAME_JS + """
        if (!n_clicks) return [window.dash_clientside.no_update, true];
        var map = {dca: dca_data, ret: ret_data, hm: hm_data, sc: sc_data};
        var mc_data = map[tab];
        if (!mc_data) return [window.dash_clientside.no_update, false];
        var json = JSON.stringify(mc_data, null, 2);
        var blob = new Blob([json], {type: 'application/json'});
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement('a');
        a.href     = url;
        a.download = _mcFilename(tab, mc_data);
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return [window.dash_clientside.no_update, false];
    }
    """,
    Output("mc-save-modal-dummy", "data"),
    Output("mc-save-modal", "is_open", allow_duplicate=True),
    Input("mc-save-modal-dl", "n_clicks"),
    State("mc-save-tab", "data"),
    State("dca-mc-results", "data"),
    State("ret-mc-results", "data"),
    State("hm-mc-results", "data"),
    State("sc-mc-results", "data"),
    prevent_initial_call=True,
)


# ── MC simulation load (upload JSON → store + set UI controls) ───────────────

_TAB_LABELS = {"dca": "DCA (tab 3)", "ret": "Retire (tab 4)",
               "hm": "Heatmap (tab 2)", "sc": "Supercharger (tab 5)"}

def _parse_mc_upload(contents, expected_tab=None):
    """Decode uploaded MC JSON, return (data, error_msg).

    If expected_tab is given, reject files saved from a different tab.
    Validates metadata.app == "Quantoshi" when metadata is present.
    """
    if not contents:
        return None, None
    _, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    data = json.loads(raw)
    # Reject files with metadata from a different app
    meta = data.get("metadata")
    if meta and meta.get("app") and meta["app"] != "Quantoshi":
        return None, f"Not a Quantoshi file (app: {meta['app']})."
    if "path_key" not in data and "params" not in data:
        return None, "Invalid MC file (missing path data)."
    file_tab = data.get("tab")
    if expected_tab and file_tab and file_tab != expected_tab:
        src = _TAB_LABELS.get(file_tab, file_tab)
        dst = _TAB_LABELS.get(expected_tab, expected_tab)
        return None, f"Wrong tab: this is a {src} simulation. Upload it on {dst}."
    return data, None


def _extract_mc_key_val(data, key, default=None):
    """Get a value from loaded data's path_key or overlay_key."""
    pk = data.get("path_key", {})
    ok = data.get("overlay_key", {})
    return pk.get(key, ok.get(key, default))


# ── MC upload callback factory ───────────────────────────────────────────────
# Each entry: (output_suffix, data_key_or_fallback_tuple, default, cast_int)
_MC_UPLOAD_FIELDS = {
    "dca": [
        ("years",    "mc_years",                None, False),
        ("start-yr", "mc_start_yr",             None, False),
        ("entry-q",  "mc_entry_q",              50,   True),
        ("window",   "mc_window",               None, False),
    ],
    "ret": [
        ("years",    "mc_years",                None, False),
        ("start-yr", "mc_start_yr",             None, False),
        ("entry-q",  "mc_entry_q",              50,   True),
        ("window",   "mc_window",               None, False),
    ],
    "hm": [
        ("years",    "mc_years",                None, False),
        ("start-yr", "mc_start_yr",             None, False),
        ("entry-q",  "mc_entry_q",              50,   True),
        ("window",   "mc_window",               None, False),
    ],
    "sc": [
        ("years",    "mc_years",                None, False),
        ("start-yr", "mc_start_yr",             None, False),
        ("entry-q",  "mc_entry_q",              50,   True),
        ("window",   "mc_window",               None, False),
    ],
}


def _register_mc_upload(prefix, fields):
    """Register an MC file upload callback for a given tab prefix."""
    outputs = [
        Output(f"{prefix}-mc-upload", "contents"),
        Output(f"{prefix}-mc-results", "data", allow_duplicate=True),
        Output(f"{prefix}-mc-upload-status", "children"),
        Output(f"{prefix}-mc-loaded", "data", allow_duplicate=True),
        Output(f"{prefix}-mc-enable", "value", allow_duplicate=True),
    ] + [
        Output(f"{prefix}-mc-{suffix}", "value", allow_duplicate=True)
        for suffix, _, _, _ in fields
    ]
    n_total = len(outputs)

    @callback(
        *outputs,
        Input(f"{prefix}-mc-upload", "contents"),
        State(f"{prefix}-mc-loaded", "data"),
        prevent_initial_call=True,
    )
    def _load_mc(contents, loaded_count):
        if not contents:
            raise dash.exceptions.PreventUpdate
        nu = dash.no_update
        err_tuple = (dash.no_update,) * n_total
        try:
            data, err = _parse_mc_upload(contents, expected_tab=prefix)
            if err:
                return err_tuple[:1] + (err,) + err_tuple[2:]
            extra = []
            for _, data_key, default, cast_int in fields:
                val = _extract_mc_key_val(data, data_key, default if default is not None else nu)
                if cast_int and val is not nu:
                    val = int(val)
                extra.append(val)
            return (None,  # clear upload so re-uploading same file works
                    data,
                    f"Loaded: {data.get('created', '?')[:19]}Z",
                    (loaded_count or 0) + 1,
                    ["yes"],
                    *extra)
        except Exception as e:
            return err_tuple[:1] + (f"Error: {e}",) + err_tuple[2:]


for _prefix, _fields in _MC_UPLOAD_FIELDS.items():
    _register_mc_upload(_prefix, _fields)
