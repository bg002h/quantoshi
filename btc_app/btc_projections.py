"""btc_projections.py — Standalone Bitcoin Projections interactive app.

Two tabs:
  • Bubble + QR Overlay  (replaces notebook cell 4)
  • CAGR Heatmap         (replaces notebook cell 5)

Usage:
    python3 btc_projections.py [path/to/model_data.pkl]
"""

import os, sys, pickle, ast, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.regression.quantile_regression import QuantReg

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavToolbar,
)
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator, StrMethodFormatter
import matplotlib.colors as mcolors

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QSplitter,
    QHBoxLayout, QVBoxLayout, QFormLayout, QScrollArea, QGroupBox,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QPushButton, QLineEdit, QFileDialog, QSizePolicy, QColorDialog,
    QMessageBox, QListWidget, QListWidgetItem, QAbstractItemView,
    QButtonGroup, QRadioButton, QStatusBar, QFrame, QFontDialog,
    QTableWidget, QTableWidgetItem, QDateEdit,
)
from PyQt5.QtCore import Qt, pyqtSignal, QDate
from PyQt5.QtGui import QColor, QIcon, QPalette, QFont


# ── linestyle catalogue ───────────────────────────────────────────────────────
LS_OPTS = [
    ("-",            "Solid"),
    ("--",           "Dashed"),
    (":",            "Dotted"),
    ("-.",           "Dash-Dot"),
    ((0, (5, 2)),    "Long Dash"),
    ((0, (1, 2)),    "Dense Dot"),
    ((0, (3, 2, 1, 2)), "Dash-Dot-Dot"),
    ((0, (5, 2, 1, 2)), "Long-Dash-Dot"),
]
LS_SPECS  = [s for s, _ in LS_OPTS]
LS_NAMES  = [n for _, n in LS_OPTS]

_CB_COLORS = [
    "#1855B0", "#167050", "#B01020", "#6B3090",
    "#A06000", "#006878", "#A02060", "#404040",
]
_CB_STYLES = [s for s, _ in LS_OPTS]

# Factory-default quantiles shown on charts / CAGR heatmap after "Reset to
# Factory Defaults".  Must be a subset of QR_QUANTILES in the notebook (so
# they are precomputed in model_data.pkl).  Any entry not found in the loaded
# model is silently skipped.
_DEFAULT_QS = [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# ── persistent settings ───────────────────────────────────────────────────────
_SETTINGS_PATH = Path.home() / ".config" / "btc-projections" / "ui_settings.json"

def _load_ui_settings():
    if _SETTINGS_PATH.exists():
        try:
            with open(_SETTINGS_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_ui_settings(d):
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(d, f, indent=2)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_ls(s):
    """Restore a linestyle spec that was stored as repr()."""
    if isinstance(s, str) and s.startswith("("):
        try:
            return ast.literal_eval(s)
        except Exception:
            return "-"
    return s


def _ls_index(spec):
    for i, s in enumerate(LS_SPECS):
        if s == spec:
            return i
    return 0


def qr_price(q, t, qr_fits):
    f = qr_fits[q]
    return 10.0 ** (f["intercept"] + f["slope"] * np.log10(np.asarray(t, float)))


def yr_to_t(cal_year, genesis=pd.Timestamp("2009-01-03")):
    return (pd.Timestamp(f"{int(cal_year)}-01-01") - genesis).days / 365.25


def today_t(genesis=pd.Timestamp("2009-01-03")):
    return (pd.Timestamp.today() - genesis).days / 365.25


def fmt_price(p):
    if p >= 1e6:
        return f"${p/1e6:.2f}M"
    if p >= 1e3:
        return f"${p/1e3:.1f}K"
    return f"${p:.0f}"


def _fmt_btc(v):
    """Format a BTC quantity for axis labels."""
    if v >= 1000: return f"{v:.0f} BTC"
    if v >= 1:    return f"{v:.2f} BTC"
    if v >= 0.01: return f"{v:.4f} BTC"
    return f"{v:.6f} BTC"


def make_seg_cmap(mat, b1, b2, c_lo, c_mid1, c_mid2, c_hi, n_disc):
    mn, mx = float(mat.min()), float(mat.max())
    if mx - mn < 1e-9:
        return mcolors.ListedColormap([c_mid1] * max(2, n_disc)), mcolors.Normalize(mn, mn + 1)
    c0 = c_lo   if mn <= b1 else (c_mid1 if mn <= b2 else c_mid2)
    c1 = c_mid1 if mx <= b1 else (c_mid2 if mx <= b2 else c_hi)
    anc = [(0.0, c0)]
    if mn < b1 < mx:
        anc.append(((b1 - mn) / (mx - mn), c_mid1))
    if mn < b2 < mx:
        anc.append(((b2 - mn) / (mx - mn), c_mid2))
    anc.append((1.0, c1))
    base = mcolors.LinearSegmentedColormap.from_list("_seg", anc, N=512)
    nd   = max(2, n_disc)
    cm   = mcolors.ListedColormap([base(k / (nd - 1)) for k in range(nd)])
    nm   = mcolors.BoundaryNorm(np.linspace(mn, mx, nd + 1), cm.N)
    return cm, nm


def _desktop_save_path(parent, filename, fmt):
    """Return a full save path under ~/Desktop, or open a file picker if unavailable."""
    desktop = Path.home() / "Desktop"
    ext_map = {"svg": "SVG files (*.svg)", "jpg": "JPEG files (*.jpg *.jpeg)"}
    if desktop.is_dir():
        return str(desktop / f"{filename}.{fmt}")
    path, _ = QFileDialog.getSaveFileName(
        parent, "Save Image",
        str(Path.home() / f"{filename}.{fmt}"),
        ext_map.get(fmt, "All files (*)"))
    return path or None


def _fit_one_qr(model, q):
    """Fit a single quantile regression from in-memory model price data."""
    years  = model.price_years
    prices = model.price_prices
    mask   = (years >= 1.0) & (prices > 0)
    ly = np.log10(years[mask])
    lp = np.log10(prices[mask])
    X  = np.column_stack([np.ones(len(ly)), ly])
    res = QuantReg(lp, X).fit(q=q, max_iter=2000)
    return {"intercept": float(res.params[0]), "slope": float(res.params[1]), "r2": 0.0}


def _find_lot_percentile(t, price, qr_fits):
    """Interpolate the QR percentile (0-1) for a given time t and price."""
    if not qr_fits:
        return 0.5
    sorted_qs = sorted(qr_fits.keys())
    t_safe = max(float(t), 0.5)
    log_p  = np.log10(max(float(price), 1e-10))
    log_ps = [np.log10(max(float(qr_price(q, t_safe, qr_fits)), 1e-10)) for q in sorted_qs]
    if log_p <= log_ps[0]:
        return sorted_qs[0]
    if log_p >= log_ps[-1]:
        return sorted_qs[-1]
    for i in range(len(sorted_qs) - 1):
        if log_ps[i] <= log_p <= log_ps[i + 1]:
            frac = (log_p - log_ps[i]) / (log_ps[i + 1] - log_ps[i] + 1e-30)
            return sorted_qs[i] + frac * (sorted_qs[i + 1] - sorted_qs[i])
    return sorted_qs[-1]


def fit_qr_from_csv(csv_path, quantiles, genesis="2009-01-03", fit_min="2010-01-01"):
    """Re-fit QR model from a price CSV.  Returns (df, qr_fits, ols_intercept, ols_slope)."""
    df = pd.read_csv(csv_path)
    df.columns = ["Date", "Price"]
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = df[df["Price"] > 0].copy()
    gen = pd.Timestamp(genesis)
    df["years"]     = (df["date"] - gen).dt.days / 365.25
    df["log_years"] = np.log10(df["years"])
    df["log_price"] = np.log10(df["Price"])
    df = df.rename(columns={"Price": "price"})
    mask = df["date"] >= pd.Timestamp(fit_min)
    dfit = df[mask].copy()
    X    = np.column_stack([np.ones(len(dfit)), dfit["log_years"].values])
    ols_slope, ols_int, *_ = linregress(dfit["log_years"].values, dfit["log_price"].values)
    qr = {}
    for q in quantiles:
        res = QuantReg(dfit["log_price"].values, X).fit(q=q, max_iter=2000)
        qr[q] = {"intercept": float(res.params[0]), "slope": float(res.params[1]), "r2": 0.0}
    return df, qr, float(ols_int), float(ols_slope)


# ── model data ────────────────────────────────────────────────────────────────

def _find_model_data():
    """Search for model_data.pkl: CLI arg > user config > app bundle dir > cwd."""
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        return sys.argv[1]
    cfg = Path.home() / ".config" / "btc-projections" / "model_data.pkl"
    if cfg.exists():
        return str(cfg)
    # PyInstaller: _MEIPASS
    base = getattr(sys, "_MEIPASS", None) or Path(__file__).parent
    bundled = Path(base) / "model_data.pkl"
    if bundled.exists():
        return str(bundled)
    local = Path("model_data.pkl")
    if local.exists():
        return str(local)
    return None


class ModelData:
    def __init__(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._path = path
        self.qr_fits       = {float(k): v for k, v in d["qr_fits"].items()}
        self.QR_QUANTILES  = [float(q) for q in d["QR_QUANTILES"]]
        self.ols_intercept = d["ols_intercept"]
        self.ols_slope     = d["ols_slope"]
        self.genesis       = pd.Timestamp(d.get("GENESIS_DATE", "2009-01-03"))
        self.years_plot_bm = np.array(d["years_plot_bm"])
        self.support_bm    = np.array(d["support_plot_bm"])
        self.comp_by_n     = [np.array(c) for c in d["bm_comp_by_n"]]
        self.bm_r2         = d["bm_r2_comp"]
        self.n_future_max  = d["bm_n_future_max"]
        self.price_dates   = d["price_dates"]
        self.price_years   = np.array(d["price_years"])
        self.price_prices  = np.array(d["price_prices"])
        self.qr_colors     = {float(k): v for k, v in d["qr_colors"].items()}
        raw_ls = d["QR_LINESTYLES"]
        self.qr_linestyles = {float(k): _parse_ls(v) for k, v in raw_ls.items()}
        # Visual config
        for key in ("PLOT_BG_COLOR", "TEXT_COLOR", "TITLE_COLOR", "SPINE_COLOR",
                    "GRID_MAJOR_COLOR", "GRID_MINOR_COLOR", "DATA_COLOR",
                    "CAGR_SEG_C_LO", "CAGR_SEG_C_MID1", "CAGR_SEG_C_MID2", "CAGR_SEG_C_HI"):
            setattr(self, key, d.get(key, "#888888"))
        for key in ("DATA_PT_SIZE", "DATA_PT_SIZE_ZOOM", "ZOOM_YEAR_LO", "ZOOM_YEAR_HI",
                    "CAGR_GRAD_STEPS", "CAGR_HEATMAP_FONTSIZE"):
            setattr(self, key, int(d.get(key, 8)))
        for key in ("ZOOM_PRICE_LO", "ZOOM_PRICE_HI", "CAGR_SEG_B1", "CAGR_SEG_B2"):
            setattr(self, key, float(d.get(key, 0)))
        self.TABLE_YEARS = d.get("TABLE_YEARS", list(range(2025, 2041)))

    def update_from_csv(self, csv_path):
        df, qr, ols_int, ols_sl = fit_qr_from_csv(
            csv_path, self.QR_QUANTILES, str(self.genesis.date()))
        self.qr_fits       = qr
        self.ols_intercept = ols_int
        self.ols_slope     = ols_sl
        self.price_years   = df["years"].values
        self.price_prices  = df["price"].values
        self.price_dates   = df["date"].dt.strftime("%Y-%m-%d").tolist()

    def save_user_override(self):
        cfg_dir = Path.home() / ".config" / "btc-projections"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        dst = cfg_dir / "model_data.pkl"
        with open(self._path, "rb") as f:
            d = pickle.load(f)
        d["qr_fits"]       = {str(k): v for k, v in self.qr_fits.items()}
        d["ols_intercept"] = self.ols_intercept
        d["ols_slope"]     = self.ols_slope
        d["price_dates"]   = list(self.price_dates)
        d["price_years"]   = list(self.price_years)
        d["price_prices"]  = list(self.price_prices)
        with open(dst, "wb") as f:
            pickle.dump(d, f, protocol=4)
        return str(dst)


# ── ColorBtn ──────────────────────────────────────────────────────────────────

class ColorBtn(QPushButton):
    color_changed = pyqtSignal(str)

    def __init__(self, color="#888888", parent=None):
        super().__init__(parent)
        self.setFixedSize(28, 22)
        self.set_color(color)
        self.clicked.connect(self._pick)

    def set_color(self, hex_color):
        self._color = hex_color
        self.setStyleSheet(
            f"background:{hex_color}; border:1px solid #555; border-radius:3px;")

    def color(self):
        return self._color

    def _pick(self):
        c = QColorDialog.getColor(QColor(self._color), self)
        if c.isValid():
            self.set_color(c.name())
            self.color_changed.emit(self._color)


# ── FontPicker ────────────────────────────────────────────────────────────────

class FontPicker(QWidget):
    """Editable font-family field + size spinbox + '…' button."""
    font_changed = pyqtSignal(str)
    size_changed = pyqtSignal(int)

    def __init__(self, family="sans-serif", size=10, parent=None):
        super().__init__(parent)
        self._family = family
        self._size   = size
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._edit = QLineEdit(family)
        self._edit.setMinimumWidth(80)
        self._edit.editingFinished.connect(self._on_edit)
        self._size_sp = QSpinBox()
        self._size_sp.setRange(5, 36)
        self._size_sp.setValue(size)
        self._size_sp.setSuffix("pt")
        self._size_sp.setFixedWidth(52)
        self._size_sp.valueChanged.connect(self._on_size_changed)
        btn = QPushButton("…")
        btn.setFixedSize(24, 22)
        btn.clicked.connect(self._pick)
        lay.addWidget(self._edit)
        lay.addWidget(self._size_sp)
        lay.addWidget(btn)

    def _pick(self):
        font, ok = QFontDialog.getFont(QFont(self._family, self._size), self)
        if ok:
            self._family = font.family()
            self._edit.setText(self._family)
            sz = max(5, min(36, font.pointSize()))
            self._size = sz
            self._size_sp.blockSignals(True)
            self._size_sp.setValue(sz)
            self._size_sp.blockSignals(False)
            self.font_changed.emit(self._family)
            self.size_changed.emit(self._size)

    def _on_edit(self):
        self._family = self._edit.text().strip() or "sans-serif"
        self.font_changed.emit(self._family)

    def _on_size_changed(self, val):
        self._size = val
        self.size_changed.emit(val)

    def family(self):
        return self._family

    def size(self):
        return self._size

    def set_family(self, name):
        self._family = name or "sans-serif"
        self._edit.setText(self._family)

    def set_size(self, n):
        n = max(5, min(36, int(n)))
        self._size = n
        self._size_sp.blockSignals(True)
        self._size_sp.setValue(n)
        self._size_sp.blockSignals(False)


# ── BubbleTab ─────────────────────────────────────────────────────────────────

class BubbleTab(QWidget):
    q_state_changed = pyqtSignal(list)
    all_fonts_applied = pyqtSignal(str, int)

    def __init__(self, model: ModelData):
        super().__init__()
        self.m = model
        self._busy = False
        # quantile state list
        self._q_state = []
        for i, q in enumerate(model.QR_QUANTILES):
            pct = q * 100
            lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
            self._q_state.append({
                "q":   q,
                "lbl": lbl,
                "vis":   True,
                "color": model.qr_colors.get(q, _CB_COLORS[i % len(_CB_COLORS)]),
                "ls":    model.qr_linestyles.get(q, _CB_STYLES[i % len(_CB_STYLES)]),
                "lw":    2.0 if abs(q - 0.5) < 1e-6 else 1.5,
            })
        self._lot_source = None
        self._build_ui()

    # ── build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ── left: plot area ──────────────────────────────────────────────────
        plot_w = QWidget()
        plot_l = QVBoxLayout(plot_w)
        plot_l.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(10, 6.5), facecolor=self.m.PLOT_BG_COLOR)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        toolbar = NavToolbar(self.canvas, plot_w)
        plot_l.addWidget(toolbar)
        plot_l.addWidget(self.canvas)

        # ── right: controls ──────────────────────────────────────────────────
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(295)
        ctrl_w = QWidget()
        ctrl_l = QVBoxLayout(ctrl_w)
        ctrl_l.setSpacing(6)

        fac_btn = QPushButton("Reset to Factory Defaults")
        fac_btn.clicked.connect(self._factory_reset)
        ctrl_l.addWidget(fac_btn)

        # Axes group
        ax_grp = QGroupBox("Axes")
        ax_l   = QVBoxLayout(ax_grp)
        ax_l.setSpacing(4)

        ax_l.addWidget(QLabel("X scale:"))
        self._xscale_grp = QButtonGroup(self)
        xrow = QHBoxLayout()
        for i, lbl in enumerate(["Linear", "Log", "Exp₁₀"]):
            rb = QRadioButton(lbl)
            if i == 1:
                rb.setChecked(True)
            self._xscale_grp.addButton(rb, i)
            xrow.addWidget(rb)
        ax_l.addLayout(xrow)

        ax_l.addWidget(QLabel("Y scale:"))
        self._yscale_grp = QButtonGroup(self)
        yrow = QHBoxLayout()
        for i, lbl in enumerate(["Log", "Linear"]):
            rb = QRadioButton(lbl)
            if i == 0:
                rb.setChecked(True)
            self._yscale_grp.addButton(rb, i)
            yrow.addWidget(rb)
        ax_l.addLayout(yrow)

        rng_form = QFormLayout()
        rng_form.setSpacing(3)
        self._xmin = QSpinBox(); self._xmin.setRange(2009, 2055); self._xmin.setValue(2009)
        self._xmax = QSpinBox(); self._xmax.setRange(2010, 2060); self._xmax.setValue(2047)
        self._ymin = QDoubleSpinBox()
        self._ymin.setRange(0.001, 1e6); self._ymin.setValue(0.01)
        self._ymin.setDecimals(3); self._ymin.setSingleStep(0.1)
        self._ymax = QDoubleSpinBox()
        self._ymax.setRange(1, 1e9); self._ymax.setValue(1e8)
        self._ymax.setDecimals(0); self._ymax.setSingleStep(1e6)
        rng_form.addRow("X min (year):", self._xmin)
        rng_form.addRow("X max (year):", self._xmax)
        rng_form.addRow("Y min ($):",    self._ymin)
        rng_form.addRow("Y max ($):",    self._ymax)
        ax_l.addLayout(rng_form)

        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self._reset_axes)
        ax_l.addWidget(reset_btn)

        self._shade_cb = QCheckBox("Shade channels")
        self._shade_cb.setChecked(True)
        ax_l.addWidget(self._shade_cb)
        self._minor_ticks_chk = QCheckBox("Minor log ticks (×0.1 dec)")
        self._minor_ticks_chk.setChecked(False)
        ax_l.addWidget(self._minor_ticks_chk)
        ctrl_l.addWidget(ax_grp)

        # Quantile lines group
        q_grp = QGroupBox("Quantile Lines")
        q_outer = QVBoxLayout(q_grp)
        q_outer.setSpacing(4)

        self._q_rows_w = QWidget()
        self._q_rows_l = QVBoxLayout(self._q_rows_w)
        self._q_rows_l.setSpacing(2)
        self._q_rows_l.setContentsMargins(0, 0, 0, 0)
        q_outer.addWidget(self._q_rows_w)
        self._rebuild_q_rows()

        add_row = QHBoxLayout()
        add_row.setSpacing(4)
        add_row.addWidget(QLabel("Add Q:"))
        self._add_q_sp = QDoubleSpinBox()
        self._add_q_sp.setRange(0.001, 99.999)
        self._add_q_sp.setValue(75.0)
        self._add_q_sp.setDecimals(3)
        self._add_q_sp.setSingleStep(5.0)
        self._add_q_sp.setSuffix(" %")
        add_btn = QPushButton("Add")
        add_btn.setFixedWidth(48)
        add_btn.clicked.connect(self._q_add)
        add_row.addWidget(self._add_q_sp)
        add_row.addWidget(add_btn)
        q_outer.addLayout(add_row)
        ctrl_l.addWidget(q_grp)

        # Bubble model group
        bm_grp  = QGroupBox("Bubble Model")
        bm_form = QFormLayout(bm_grp)
        bm_form.setSpacing(4)

        comp_row = QHBoxLayout()
        self._show_comp = QCheckBox("Show composite")
        self._show_comp.setChecked(True)
        self._comp_color = ColorBtn("#DC2626")
        comp_row.addWidget(self._show_comp)
        comp_row.addWidget(self._comp_color)
        bm_form.addRow(comp_row)

        sup_row = QHBoxLayout()
        self._show_sup = QCheckBox("Show support")
        self._show_sup.setChecked(True)
        self._sup_color = ColorBtn("#3B82F6")
        sup_row.addWidget(self._show_sup)
        sup_row.addWidget(self._sup_color)
        bm_form.addRow(sup_row)

        self._comp_lw = QDoubleSpinBox()
        self._comp_lw.setRange(0.5, 6.0); self._comp_lw.setValue(2.2); self._comp_lw.setSingleStep(0.2)
        bm_form.addRow("Composite lw:", self._comp_lw)

        self._sup_lw = QDoubleSpinBox()
        self._sup_lw.setRange(0.5, 4.0); self._sup_lw.setValue(1.3); self._sup_lw.setSingleStep(0.1)
        bm_form.addRow("Support lw:", self._sup_lw)

        n_row = QHBoxLayout()
        self._n_slider = QSlider(Qt.Horizontal)
        self._n_slider.setRange(0, self.m.n_future_max)
        self._n_slider.setValue(self.m.n_future_max)
        self._n_label  = QLabel(str(self.m.n_future_max))
        self._n_label.setMinimumWidth(18)
        n_row.addWidget(self._n_slider)
        n_row.addWidget(self._n_label)
        bm_form.addRow("Future bubbles:", n_row)

        self._r2_label = QLabel(f"R² = {self.m.bm_r2:.4f}")
        bm_form.addRow(self._r2_label)
        ctrl_l.addWidget(bm_grp)

        # Display group
        disp_grp  = QGroupBox("Display")
        disp_form = QFormLayout(disp_grp)
        disp_form.setSpacing(4)
        self._show_ols  = QCheckBox("OLS line");   self._show_ols.setChecked(True)
        self._show_data = QCheckBox("Price data"); self._show_data.setChecked(True)
        self._show_today = QCheckBox("Today line"); self._show_today.setChecked(True)
        disp_form.addRow(self._show_ols)
        disp_form.addRow(self._show_data)
        disp_form.addRow(self._show_today)
        self._pt_size = QSpinBox(); self._pt_size.setRange(1, 80); self._pt_size.setValue(self.m.DATA_PT_SIZE)
        self._pt_alpha = QDoubleSpinBox()
        self._pt_alpha.setRange(0.05, 1.0); self._pt_alpha.setValue(0.4); self._pt_alpha.setSingleStep(0.05)
        disp_form.addRow("Pt size:", self._pt_size)
        disp_form.addRow("Pt alpha:", self._pt_alpha)
        self._bg_color_btn = ColorBtn(self.m.PLOT_BG_COLOR)
        self._bg_color_btn.setToolTip("Plot background colour")
        disp_form.addRow("Plot bg:", self._bg_color_btn)
        ctrl_l.addWidget(disp_grp)

        # Bitcoin Stack group
        stack_grp  = QGroupBox("Bitcoin Stack")
        stack_form = QFormLayout(stack_grp)
        stack_form.setSpacing(4)
        self._stack_sp = QDoubleSpinBox()
        self._stack_sp.setRange(0, 10_000_000)
        self._stack_sp.setDecimals(8)
        self._stack_sp.setValue(0.0)
        self._stack_sp.setSingleStep(0.1)
        stack_form.addRow("BTC owned:", self._stack_sp)
        self._show_stack_cb = QCheckBox("Relabel Y-axis as stack value")
        self._show_stack_cb.setChecked(False)
        stack_form.addRow(self._show_stack_cb)
        ctrl_l.addWidget(stack_grp)

        # LEO group
        leo_grp = QGroupBox("Lot Entry Override (LEO)")
        leo_l   = QVBoxLayout(leo_grp)
        leo_l.setSpacing(4)
        self._use_lots_chk = QCheckBox("Plot lots on chart")
        self._use_lots_chk.setChecked(False)
        leo_l.addWidget(self._use_lots_chk)
        self._lot_list = QListWidget()
        self._lot_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._lot_list.setMinimumHeight(60)
        leo_l.addWidget(self._lot_list)
        _leo_row1 = QHBoxLayout()
        _leo_all1 = QPushButton("All");  _leo_all1.setFixedWidth(38)
        _leo_none1 = QPushButton("None"); _leo_none1.setFixedWidth(44)
        _leo_all1.clicked.connect(lambda: [self._lot_list.item(_i).setSelected(True) for _i in range(self._lot_list.count())])
        _leo_none1.clicked.connect(lambda: [self._lot_list.item(_i).setSelected(False) for _i in range(self._lot_list.count())])
        _leo_row1.addWidget(_leo_all1); _leo_row1.addWidget(_leo_none1); _leo_row1.addStretch()
        leo_l.addLayout(_leo_row1)
        self._leo_lbl = QLabel("")
        self._leo_lbl.setStyleSheet("font-size:9px;")
        self._leo_lbl.setWordWrap(True)
        leo_l.addWidget(self._leo_lbl)
        ctrl_l.addWidget(leo_grp)

        # Fonts group
        font_grp  = QGroupBox("Fonts")
        font_form = QFormLayout(font_grp)
        font_form.setSpacing(4)
        all_row = QHBoxLayout()
        self._font_all_b = FontPicker("sans-serif")
        all_btn = QPushButton("Apply to all")
        all_btn.setFixedWidth(90)
        all_btn.clicked.connect(self._apply_font_to_all)
        all_tabs_btn = QPushButton("All tabs"); all_tabs_btn.setFixedWidth(65)
        all_tabs_btn.clicked.connect(
            lambda: self.all_fonts_applied.emit(
                self._font_all_b.family(), self._font_all_b.size()))
        all_row.addWidget(self._font_all_b)
        all_row.addWidget(all_btn)
        all_row.addWidget(all_tabs_btn)
        font_form.addRow("All fields:", all_row)
        self._font_title  = FontPicker("sans-serif")
        self._font_axis_t = FontPicker("sans-serif")
        self._font_ticks        = FontPicker("sans-serif")
        self._font_ticks_minor  = FontPicker("sans-serif")
        self._font_legend = FontPicker("sans-serif")
        font_form.addRow("Chart title:", self._font_title)
        font_form.addRow("Axis titles:", self._font_axis_t)
        font_form.addRow("Major ticks:", self._font_ticks)
        font_form.addRow("Minor ticks:", self._font_ticks_minor)
        font_form.addRow("Legend:",      self._font_legend)
        ctrl_l.addWidget(font_grp)

        # Save group
        save_grp  = QGroupBox("Save")
        save_form = QFormLayout(save_grp)
        save_form.setSpacing(4)
        self._fn_edit = QLineEdit("btc_bubble_overlay")
        self._dpi_sp  = QSpinBox(); self._dpi_sp.setRange(72, 600); self._dpi_sp.setValue(150)
        save_form.addRow("Filename:", self._fn_edit)
        save_form.addRow("DPI:", self._dpi_sp)
        btn_row = QHBoxLayout()
        svg_btn = QPushButton("Save SVG"); svg_btn.clicked.connect(lambda: self._save("svg"))
        jpg_btn = QPushButton("Save JPG"); jpg_btn.clicked.connect(lambda: self._save("jpg"))
        btn_row.addWidget(svg_btn); btn_row.addWidget(jpg_btn)
        save_form.addRow(btn_row)
        self._save_lbl = QLabel("")
        save_form.addRow(self._save_lbl)
        ctrl_l.addWidget(save_grp)

        ctrl_l.addStretch()
        ctrl_scroll.setWidget(ctrl_w)

        self._splitter = splitter
        self._splitter.addWidget(plot_w)
        self._splitter.addWidget(ctrl_scroll)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

        main_l = QHBoxLayout(self)
        main_l.setContentsMargins(0, 0, 0, 0)
        main_l.addWidget(self._splitter)

        # Connect signals
        for btn in self._xscale_grp.buttons():
            btn.toggled.connect(self.redraw)
        for btn in self._yscale_grp.buttons():
            btn.toggled.connect(self.redraw)
        for w in (self._xmin, self._xmax, self._ymin, self._ymax):
            w.valueChanged.connect(self.redraw)
        self._shade_cb.toggled.connect(self.redraw)
        self._minor_ticks_chk.toggled.connect(self.redraw)
        self._show_comp.toggled.connect(self.redraw)
        self._comp_color.color_changed.connect(self.redraw)
        self._show_sup.toggled.connect(self.redraw)
        self._sup_color.color_changed.connect(self.redraw)
        self._comp_lw.valueChanged.connect(self.redraw)
        self._sup_lw.valueChanged.connect(self.redraw)
        self._n_slider.valueChanged.connect(self._on_n_changed)
        self._show_ols.toggled.connect(self.redraw)
        self._show_data.toggled.connect(self.redraw)
        self._show_today.toggled.connect(self.redraw)
        self._pt_size.valueChanged.connect(self.redraw)
        self._pt_alpha.valueChanged.connect(self.redraw)
        self._stack_sp.valueChanged.connect(self.redraw)
        self._show_stack_cb.toggled.connect(self.redraw)
        self._use_lots_chk.toggled.connect(self.redraw)
        self._lot_list.itemSelectionChanged.connect(self.redraw)
        for fp in (self._font_title, self._font_axis_t, self._font_ticks,
                   self._font_ticks_minor, self._font_legend):
            fp.font_changed.connect(self.redraw)
            fp.size_changed.connect(self.redraw)
        self._bg_color_btn.color_changed.connect(self.redraw)

    # ── quantile rows ─────────────────────────────────────────────────────────
    def _rebuild_q_rows(self):
        # Clear existing rows
        while self._q_rows_l.count():
            w = self._q_rows_l.takeAt(0).widget()
            if w:
                w.deleteLater()
        for i, qs in enumerate(self._q_state):
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(3)

            cb = QCheckBox(qs["lbl"])
            cb.setChecked(qs["vis"])
            cb.setMinimumWidth(85)
            cb.toggled.connect(lambda v, idx=i: self._q_vis(idx, v))

            clr_btn = ColorBtn(qs["color"])
            clr_btn.color_changed.connect(lambda c, idx=i: self._q_color(idx, c))

            ls_cb = QComboBox()
            ls_cb.addItems(LS_NAMES)
            ls_cb.setCurrentIndex(_ls_index(qs["ls"]))
            ls_cb.setFixedWidth(105)
            ls_cb.currentIndexChanged.connect(lambda v, idx=i: self._q_ls(idx, v))

            lw_sp = QDoubleSpinBox()
            lw_sp.setRange(0.3, 6.0); lw_sp.setValue(qs["lw"]); lw_sp.setSingleStep(0.25)
            lw_sp.setFixedWidth(56)
            lw_sp.valueChanged.connect(lambda v, idx=i: self._q_lw(idx, v))

            rm_btn = QPushButton("×")
            rm_btn.setFixedSize(22, 22)
            rm_btn.clicked.connect(lambda _, idx=i: self._q_remove(idx))

            row_l.addWidget(cb)
            row_l.addWidget(clr_btn)
            row_l.addWidget(ls_cb)
            row_l.addWidget(lw_sp)
            row_l.addWidget(rm_btn)
            self._q_rows_l.addWidget(row_w)

    def _emit_q_changed(self):
        self.q_state_changed.emit(list(self._q_state))

    def _q_vis(self, idx, v):   self._q_state[idx]["vis"] = v;   self._emit_q_changed(); self.redraw()
    def _q_color(self, idx, c): self._q_state[idx]["color"] = c; self._emit_q_changed(); self.redraw()
    def _q_ls(self, idx, v):    self._q_state[idx]["ls"] = LS_SPECS[v]; self._emit_q_changed(); self.redraw()
    def _q_lw(self, idx, v):    self._q_state[idx]["lw"] = v;   self._emit_q_changed(); self.redraw()

    def _q_remove(self, idx):
        self._q_state.pop(idx)
        self._rebuild_q_rows()
        self._emit_q_changed()
        self.redraw()

    def _q_add(self):
        q = self._add_q_sp.value() / 100.0
        # Ignore duplicates
        for s in self._q_state:
            if abs(s["q"] - q) < 1e-6:
                return
        # Fit if not already in model
        if q not in self.m.qr_fits:
            try:
                self.m.qr_fits[q] = _fit_one_qr(self.m, q)
            except Exception as e:
                QMessageBox.warning(self, "Fit error",
                                    f"Could not fit Q{q*100:.4g}%:\n{e}")
                return
        n   = len(self._q_state)
        pct = q * 100
        lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
        self._q_state.append({
            "q":     q,
            "lbl":   lbl,
            "vis":   True,
            "color": _CB_COLORS[n % len(_CB_COLORS)],
            "ls":    _CB_STYLES[n % len(_CB_STYLES)],
            "lw":    2.0 if abs(q - 0.5) < 1e-6 else 1.5,
        })
        self._q_state.sort(key=lambda s: s["q"])
        self._rebuild_q_rows()
        self._emit_q_changed()
        self.redraw()

    def _factory_q_state(self):
        m = self.m
        out = []
        for i, q in enumerate(q for q in _DEFAULT_QS if q in m.qr_fits):
            pct = q * 100
            lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
            out.append({
                "q":     q,
                "lbl":   lbl,
                "vis":   True,
                "color": m.qr_colors.get(q, _CB_COLORS[i % len(_CB_COLORS)]),
                "ls":    m.qr_linestyles.get(q, _CB_STYLES[i % len(_CB_STYLES)]),
                "lw":    2.0 if abs(q - 0.5) < 1e-6 else 1.5,
            })
        return out

    def _factory_reset(self):
        self._busy = True
        self._xscale_grp.button(1).setChecked(True)   # Log
        self._yscale_grp.button(0).setChecked(True)   # Log
        self._xmin.setValue(2010)
        self._xmax.setValue(2047)
        self._ymin.setValue(0.01)
        self._ymax.setValue(1e8)
        self._shade_cb.setChecked(True)
        self._minor_ticks_chk.setChecked(False)
        self._show_comp.setChecked(True)
        self._comp_color.set_color("#DC2626")
        self._comp_lw.setValue(2.2)
        self._show_sup.setChecked(True)
        self._sup_color.set_color("#3B82F6")
        self._sup_lw.setValue(1.3)
        self._n_slider.setValue(self.m.n_future_max)
        self._n_label.setText(str(self.m.n_future_max))
        self._show_ols.setChecked(True)
        self._show_data.setChecked(True)
        self._show_today.setChecked(True)
        self._pt_size.setValue(self.m.DATA_PT_SIZE)
        self._pt_alpha.setValue(0.4)
        self._q_state = self._factory_q_state()
        self._rebuild_q_rows()
        self._stack_sp.setValue(0.0)
        self._show_stack_cb.setChecked(False)
        for fp in (self._font_title, self._font_axis_t, self._font_ticks, self._font_legend,
                   self._font_all_b):
            fp.set_family("sans-serif")
        self._busy = False
        self.redraw()

    def showEvent(self, event):
        super().showEvent(event)
        sizes = getattr(self, '_pending_splitter', None)
        if sizes is not None:
            self._splitter.setSizes(sizes)
            self._pending_splitter = None

    def _collect_settings(self):
        def _ls_str(ls):
            return repr(ls) if isinstance(ls, tuple) else str(ls)
        return {
            "xscale":     self._xscale_grp.checkedId(),
            "yscale":     self._yscale_grp.checkedId(),
            "xmin":       self._xmin.value(),
            "xmax":       self._xmax.value(),
            "ymin":       self._ymin.value(),
            "ymax":       self._ymax.value(),
            "shade":        self._shade_cb.isChecked(),
            "minor_ticks":  self._minor_ticks_chk.isChecked(),
            "show_comp":    self._show_comp.isChecked(),
            "comp_color": self._comp_color.color(),
            "comp_lw":    self._comp_lw.value(),
            "show_sup":   self._show_sup.isChecked(),
            "sup_color":  self._sup_color.color(),
            "sup_lw":     self._sup_lw.value(),
            "n_future":   self._n_slider.value(),
            "show_ols":   self._show_ols.isChecked(),
            "show_data":  self._show_data.isChecked(),
            "show_today": self._show_today.isChecked(),
            "pt_size":    self._pt_size.value(),
            "pt_alpha":   self._pt_alpha.value(),
            "q_state": [
                {"q": s["q"], "vis": s["vis"], "color": s["color"],
                 "ls": _ls_str(s["ls"]), "lw": s["lw"]}
                for s in self._q_state
            ],
            "stack":        self._stack_sp.value(),
            "show_stack":   self._show_stack_cb.isChecked(),
            "font_title":   self._font_title.family(),
            "font_axis_t":  self._font_axis_t.family(),
            "font_ticks":        self._font_ticks.family(),
            "font_ticks_minor":  self._font_ticks_minor.family(),
            "font_legend":       self._font_legend.family(),
            "font_title_sz":       self._font_title.size(),
            "font_axis_t_sz":      self._font_axis_t.size(),
            "font_ticks_sz":       self._font_ticks.size(),
            "font_ticks_minor_sz": self._font_ticks_minor.size(),
            "font_legend_sz":      self._font_legend.size(),
            "bg_color":          self._bg_color_btn.color(),
            "splitter_sizes": list(self._splitter.sizes()),
        }

    def _apply_settings(self, d):
        self._busy = True
        try:
            btn = self._xscale_grp.button(d.get("xscale", 1))
            if btn: btn.setChecked(True)
            btn = self._yscale_grp.button(d.get("yscale", 0))
            if btn: btn.setChecked(True)
            self._xmin.setValue(d.get("xmin", 2010))
            self._xmax.setValue(d.get("xmax", 2047))
            self._ymin.setValue(d.get("ymin", 0.01))
            self._ymax.setValue(d.get("ymax", 1e8))
            self._shade_cb.setChecked(d.get("shade", True))
            self._minor_ticks_chk.setChecked(d.get("minor_ticks", False))
            self._show_comp.setChecked(d.get("show_comp", True))
            self._comp_color.set_color(d.get("comp_color", "#DC2626"))
            self._comp_lw.setValue(d.get("comp_lw", 2.2))
            self._show_sup.setChecked(d.get("show_sup", True))
            self._sup_color.set_color(d.get("sup_color", "#3B82F6"))
            self._sup_lw.setValue(d.get("sup_lw", 1.3))
            nf = min(int(d.get("n_future", self.m.n_future_max)), self.m.n_future_max)
            self._n_slider.setValue(nf)
            self._n_label.setText(str(nf))
            self._show_ols.setChecked(d.get("show_ols", True))
            self._show_data.setChecked(d.get("show_data", True))
            self._show_today.setChecked(d.get("show_today", True))
            self._pt_size.setValue(d.get("pt_size", self.m.DATA_PT_SIZE))
            self._pt_alpha.setValue(d.get("pt_alpha", 0.4))
            raw_qs = d.get("q_state")
            if raw_qs:
                new_qs = []
                for i, qs in enumerate(raw_qs):
                    q = float(qs["q"])
                    if q not in self.m.qr_fits:
                        try:
                            self.m.qr_fits[q] = _fit_one_qr(self.m, q)
                        except Exception:
                            continue
                    pct = q * 100
                    lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
                    new_qs.append({
                        "q":     q,
                        "lbl":   lbl,
                        "vis":   bool(qs.get("vis", True)),
                        "color": qs.get("color", _CB_COLORS[i % len(_CB_COLORS)]),
                        "ls":    _parse_ls(qs.get("ls", "-")),
                        "lw":    float(qs.get("lw", 1.5)),
                    })
                if new_qs:
                    self._q_state = new_qs
            self._rebuild_q_rows()
            self._stack_sp.setValue(float(d.get("stack", 0.0)))
            self._show_stack_cb.setChecked(bool(d.get("show_stack", False)))
            self._font_title.set_family(d.get("font_title",  "sans-serif"))
            self._font_axis_t.set_family(d.get("font_axis_t", "sans-serif"))
            self._font_ticks.set_family(d.get("font_ticks",  "sans-serif"))
            self._font_ticks_minor.set_family(d.get("font_ticks_minor", "sans-serif"))
            self._font_legend.set_family(d.get("font_legend", "sans-serif"))
            self._font_title.set_size(d.get("font_title_sz", 11))
            self._font_axis_t.set_size(d.get("font_axis_t_sz", 10))
            self._font_ticks.set_size(d.get("font_ticks_sz", 10))
            self._font_ticks_minor.set_size(d.get("font_ticks_minor_sz", 6))
            self._font_legend.set_size(d.get("font_legend_sz", 7))
            self._bg_color_btn.set_color(d.get("bg_color", self.m.PLOT_BG_COLOR))
            if "splitter_sizes" in d:
                self._pending_splitter = [int(x) for x in d["splitter_sizes"]]
        finally:
            self._busy = False
        self.redraw()

    def _apply_font_to_all(self):
        fam = self._font_all_b.family()
        sz  = self._font_all_b.size()
        for fp in (self._font_title, self._font_axis_t, self._font_ticks,
                   self._font_ticks_minor, self._font_legend):
            fp.set_family(fam)
            fp.set_size(sz)
        self.redraw()

    def set_lot_source(self, tracker_tab):
        self._lot_source = tracker_tab

    def refresh_lot_list(self):
        self._lot_list.clear()
        if self._lot_source is None:
            return
        for i, lot in enumerate(self._lot_source.get_lots()):
            pct_str = f"Q{lot['pct_q']*100:.2f}%"
            lbl = f"{lot['date']}  {lot['btc']:.4f} BTC @ {fmt_price(lot['price'])}  ({pct_str})"
            item = QListWidgetItem(lbl)
            item.setData(Qt.UserRole, i)
            self._lot_list.addItem(item)
            item.setSelected(True)

    def _on_n_changed(self, v):
        self._n_label.setText(str(v))
        self.redraw()

    def _reset_axes(self):
        self._busy = True
        self._xmin.setValue(2009); self._xmax.setValue(2047)
        self._ymin.setValue(0.01); self._ymax.setValue(1e8)
        self._busy = False
        self.redraw()

    # ── redraw ────────────────────────────────────────────────────────────────
    def redraw(self, *_):
        if self._busy:
            return
        m  = self.m
        ax = self.ax
        bg_color = self._bg_color_btn.color()
        ax.clear()
        ax.set_facecolor(bg_color)
        self.fig.patch.set_facecolor(bg_color)
        for sp in ax.spines.values():
            sp.set_edgecolor(m.SPINE_COLOR)
        ax.tick_params(colors=m.TEXT_COLOR)
        ax.grid(True, which="major", color=m.GRID_MAJOR_COLOR, lw=0.6, alpha=0.8, zorder=0)

        font_title       = self._font_title.family()
        font_axis_t      = self._font_axis_t.family()
        font_ticks       = self._font_ticks.family()
        font_ticks_minor = self._font_ticks_minor.family()
        font_legend      = self._font_legend.family()
        font_title_sz    = self._font_title.size()
        font_axis_sz     = self._font_axis_t.size()
        font_ticks_sz       = self._font_ticks.size()
        font_ticks_minor_sz = self._font_ticks_minor.size()
        font_legend_sz      = self._font_legend.size()
        _stack = self._stack_sp.value()
        stack  = _stack if (self._show_stack_cb.isChecked() and _stack > 0) else 0

        # axis ranges
        t_lo = yr_to_t(self._xmin.value(), m.genesis)
        t_hi = yr_to_t(self._xmax.value(), m.genesis)
        y_lo = self._ymin.value()
        y_hi = self._ymax.value()
        t_lo = max(t_lo, 0.01)

        # x scale
        xmode = self._xscale_grp.checkedId()  # 0=linear, 1=log, 2=exp10
        if xmode == 1:
            ax.set_xscale("log")
        elif xmode == 2:
            _mt = float(m.price_years.max()) if len(m.price_years) else t_hi
            ax.set_xscale("function", functions=(
                lambda x, _m=_mt: np.power(10.0, np.asarray(x, float) / _m),
                lambda q, _m=_mt: _m * np.log10(np.maximum(np.asarray(q, float), 1e-10))))

        # y scale
        if self._yscale_grp.checkedId() == 0:
            ax.set_yscale("log")

        # ── plot time arrays ──────────────────────────────────────────────────
        t_arr  = np.linspace(max(t_lo, 0.1), t_hi, 2000)
        vis_qs = [s for s in self._q_state if s["vis"]]

        # shading between adjacent visible quantiles
        if self._shade_cb.isChecked() and len(vis_qs) >= 2:
            for j in range(len(vis_qs) - 1):
                lo_p = qr_price(vis_qs[j]["q"],     t_arr, m.qr_fits)
                hi_p = qr_price(vis_qs[j+1]["q"],   t_arr, m.qr_fits)
                ax.fill_between(t_arr, lo_p, hi_p,
                                color=vis_qs[j]["color"], alpha=0.07, zorder=1)

        # quantile lines
        for qs in vis_qs:
            p = qr_price(qs["q"], t_arr, m.qr_fits)
            pct = qs["q"] * 100
            lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
            ax.plot(t_arr, p, color=qs["color"], ls=qs["ls"], lw=qs["lw"],
                    label=lbl, zorder=3)

        # OLS line
        if self._show_ols.isChecked():
            p_ols = 10.0 ** (m.ols_intercept + m.ols_slope * np.log10(t_arr))
            ax.plot(t_arr, p_ols, color="#888888", ls="--", lw=1.3,
                    alpha=0.8, label="OLS", zorder=2)

        # bubble support
        if self._show_sup.isChecked():
            bm_mask = (m.years_plot_bm >= t_lo) & (m.years_plot_bm <= t_hi)
            ax.plot(m.years_plot_bm[bm_mask], m.support_bm[bm_mask],
                    color=self._sup_color.color(), lw=float(self._sup_lw.value()),
                    ls="--", alpha=0.9, zorder=4, label="Bubble support")

        # bubble composite
        if self._show_comp.isChecked():
            n   = self._n_slider.value()
            bm_mask = (m.years_plot_bm >= t_lo) & (m.years_plot_bm <= t_hi)
            ax.plot(m.years_plot_bm[bm_mask], m.comp_by_n[n][bm_mask],
                    color=self._comp_color.color(),
                    lw=float(self._comp_lw.value()),
                    zorder=6,
                    label=f"Bubble composite (N={n})  R²={m.bm_r2:.4f}")

        # historical price
        if self._show_data.isChecked():
            mask = (m.price_years >= t_lo) & (m.price_years <= t_hi)
            ax.scatter(m.price_years[mask], m.price_prices[mask],
                       s=self._pt_size.value(), c=m.DATA_COLOR,
                       alpha=float(self._pt_alpha.value()),
                       edgecolors="none", zorder=5)

        # today line
        if self._show_today.isChecked():
            td = today_t(m.genesis)
            if t_lo <= td <= t_hi:
                ax.axvline(td, color="#FF6600", ls="--", lw=1.5, alpha=0.85,
                           label="Today", zorder=7)

        # LEO lot markers
        if (self._use_lots_chk.isChecked() and self._lot_source is not None
                and self._lot_list.count() > 0):
            _sel_leo = [self._lot_list.item(_i).data(Qt.UserRole)
                        for _i in range(self._lot_list.count())
                        if self._lot_list.item(_i).isSelected()]
            _all_leo = self._lot_source.get_lots()
            _lots_to_plot = [_all_leo[_j] for _j in _sel_leo if _j < len(_all_leo)]
            _plotted = 0
            for _lot in _lots_to_plot:
                try:
                    _lt = (pd.Timestamp(_lot["date"]) - m.genesis).days / 365.25
                    _lp = _lot["price"]
                    if t_lo <= _lt <= t_hi and y_lo <= _lp <= y_hi:
                        _lbl_lot = f"Q{_lot['pct_q']*100:.1f}%" if _plotted == 0 else None
                        ax.scatter([_lt], [_lp], s=55, c="#FFD700", zorder=11,
                                   edgecolors="#333333", linewidths=0.7,
                                   label="Lot" if _plotted == 0 else None)
                        ax.annotate(
                            f"Q{_lot['pct_q']*100:.1f}%  {_lot['btc']:.4f}\u20bf",
                            xy=(_lt, _lp), xytext=(5, 4),
                            textcoords="offset points", fontsize=7,
                            color=m.TEXT_COLOR, fontfamily=font_ticks,
                            zorder=12)
                        _plotted += 1
                except Exception:
                    pass
            n_lots = len(_lots_to_plot)
            self._leo_lbl.setText(f"{_plotted}/{n_lots} lot(s) in view" if n_lots else "")
        else:
            self._leo_lbl.setText("")

        # axes limits & ticks
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(y_lo, y_hi)

        # year tick labels on x-axis
        step = 1 if (self._xmax.value() - self._xmin.value()) <= 15 else (
               2 if (self._xmax.value() - self._xmin.value()) <= 30 else 5)
        tick_yrs = range(self._xmin.value(), self._xmax.value() + 1, step)
        tick_ts  = [yr_to_t(y, m.genesis) for y in tick_yrs]
        tick_ts  = [t for t, y in zip(tick_ts, tick_yrs) if t_lo <= t <= t_hi]
        tick_yrs = [y for t, y in zip(
            [yr_to_t(y, m.genesis) for y in range(self._xmin.value(), self._xmax.value() + 1, step)],
            range(self._xmin.value(), self._xmax.value() + 1, step))
            if yr_to_t(y, m.genesis) in tick_ts]
        if tick_ts:
            ax.set_xticks(tick_ts)
            ax.set_xticklabels([str(y) for y in tick_yrs], rotation=45, ha="right",
                               fontfamily=font_ticks, fontsize=font_ticks_sz)
            ax.xaxis.set_minor_locator(NullLocator())

        # price y-axis ticks (log scale)
        if self._yscale_grp.checkedId() == 0:
            def _fmt_y(p):
                v = p * stack if stack > 0 else p
                if v >= 1e6: return f"${v/1e6:.1f}M"
                if v >= 1e3: return f"${v/1e3:.0f}K"
                return f"${v:.3g}"
            maj = [p for p in (0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
                   if y_lo <= p <= y_hi]
            if maj:
                ax.yaxis.set_major_locator(FixedLocator(maj))
                if self._minor_ticks_chk.isChecked():
                    import math as _mt
                    _lo_e = int(_mt.floor(_mt.log10(max(y_lo, 1e-100))))
                    _hi_e = int(_mt.ceil(_mt.log10(max(y_hi, 1e-100))))
                    _mj_d = [10.0**e for e in range(_lo_e, _hi_e + 1)]
                    _minors_b = [float(_p * _k)
                                 for _p in _mj_d for _k in range(2, 10)
                                 if y_lo <= _p * _k <= y_hi]
                    if _minors_b:
                        ax.yaxis.set_minor_locator(FixedLocator(_minors_b))
                        ax.tick_params(axis='y', which='minor', length=3,
                                       color=m.SPINE_COLOR,
                                       labelsize=font_ticks_minor_sz, labelcolor=m.TEXT_COLOR,
                                       labelfontfamily=font_ticks_minor)
                        def _mfmt_b(v, pos, _m=_mt):
                            if v <= 0: return ''
                            exp = int(_m.floor(_m.log10(max(v, 1e-100))))
                            mult = round(v / 10.0**exp)
                            return fmt_price(v) if mult % 2 == 0 else ''
                        ax.yaxis.set_minor_formatter(FuncFormatter(_mfmt_b))
                        ax.grid(True, which='minor',
                                color=m.GRID_MINOR_COLOR,
                                linewidth=0.4, linestyle=':')
                else:
                    ax.yaxis.set_minor_locator(NullLocator())
                    ax.yaxis.set_minor_formatter(NullFormatter())
                    ax.grid(False, which='minor')
                ax.set_yticklabels([_fmt_y(p) for p in maj], fontfamily=font_ticks,
                                  fontsize=font_ticks_sz)

        ax.set_xlabel("Years since genesis (2009-01-03)", color=m.TEXT_COLOR,
                      fontfamily=font_axis_t, fontsize=font_axis_sz)
        ax.set_ylabel("Stack Value (USD)" if stack > 0 else "Bitcoin Price (USD)",
                      color=m.TEXT_COLOR, fontfamily=font_axis_t, fontsize=font_axis_sz)
        ax.set_title("Bitcoin Bubble Model + Quantile Regression Channels",
                     color=m.TITLE_COLOR, fontsize=font_title_sz, fontfamily=font_title)
        if vis_qs or self._show_comp.isChecked():
            ax.legend(framealpha=0.9, ncol=2, edgecolor=m.GRID_MAJOR_COLOR,
                      loc="upper left", prop={'family': font_legend, 'size': font_legend_sz})
        if _stack > 0:
            ax.text(0.99, 0.01, f"Stack: {_stack:.8g} BTC",
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8, color=m.TEXT_COLOR, fontfamily=font_ticks,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=m.PLOT_BG_COLOR,
                              edgecolor=m.SPINE_COLOR, alpha=0.8))
        self.canvas.draw_idle()

    def _save(self, fmt):
        fn   = self._fn_edit.text().strip() or "btc_bubble_overlay"
        path = _desktop_save_path(self, fn, fmt)
        if not path:
            return
        try:
            self.fig.savefig(path, format=fmt,
                             bbox_inches="tight", dpi=self._dpi_sp.value())
            self._save_lbl.setText(f"✓ Saved {Path(path).name}")
        except Exception as e:
            self._save_lbl.setText(f"Error: {e}")


# ── HeatmapTab ────────────────────────────────────────────────────────────────

class HeatmapTab(QWidget):
    all_fonts_applied = pyqtSignal(str, int)

    def __init__(self, model: ModelData):
        super().__init__()
        self.m = model
        self._busy = False
        self._lot_source = None
        self._build_ui()

    def set_lot_source(self, tracker_tab):
        self._lot_source = tracker_tab

    def refresh_lot_list(self):
        """Rebuild the lot selection list from StackTrackerTab data."""
        self._lot_list.clear()
        if self._lot_source is None:
            return
        for i, lot in enumerate(self._lot_source.get_lots()):
            pct_str = f"Q{lot['pct_q']*100:.2f}%"
            lbl = f"{lot['date']}  {lot['btc']:.4f} BTC @ {fmt_price(lot['price'])}  ({pct_str})"
            item = QListWidgetItem(lbl)
            item.setData(Qt.UserRole, i)
            self._lot_list.addItem(item)
            item.setSelected(True)
        if not self._busy:
            self.redraw()

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ── left: plot area ──────────────────────────────────────────────────
        plot_w = QWidget()
        plot_l = QVBoxLayout(plot_w)
        plot_l.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(10, 7), facecolor=self.m.PLOT_BG_COLOR)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavToolbar(self.canvas, plot_w)
        plot_l.addWidget(toolbar)
        plot_l.addWidget(self.canvas)

        # ── right: controls ──────────────────────────────────────────────────
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(295)
        ctrl_w = QWidget()
        ctrl_l = QVBoxLayout(ctrl_w)
        ctrl_l.setSpacing(6)

        fac_btn = QPushButton("Reset to Factory Defaults")
        fac_btn.clicked.connect(self._factory_reset)
        ctrl_l.addWidget(fac_btn)

        # Entry group
        entry_grp  = QGroupBox("Entry")
        entry_form = QFormLayout(entry_grp)
        entry_form.setSpacing(4)

        eyr_row = QHBoxLayout()
        self._eyr_sl = QSlider(Qt.Horizontal)
        self._eyr_sl.setRange(2010, 2039); self._eyr_sl.setValue(2025)
        self._eyr_lbl = QLabel("2025")
        self._eyr_lbl.setMinimumWidth(36)
        eyr_row.addWidget(self._eyr_sl); eyr_row.addWidget(self._eyr_lbl)
        entry_form.addRow("Entry Year:", eyr_row)

        self._eq_cb  = QComboBox()
        _qq = self.m.QR_QUANTILES
        for q in _qq:
            p = q * 100
            self._eq_cb.addItem(f"Q{p:.4g}%" if p >= 1 else f"Q{p:.3g}%", q)
        self._eq_cb.setCurrentIndex(2)  # default Q0.1%
        entry_form.addRow("Entry Q:", self._eq_cb)

        eq_add_row = QHBoxLayout()
        eq_add_row.setSpacing(4)
        self._eq_add_sp = QDoubleSpinBox()
        self._eq_add_sp.setRange(0.001, 99.999); self._eq_add_sp.setValue(75.0)
        self._eq_add_sp.setDecimals(3); self._eq_add_sp.setSingleStep(5.0)
        self._eq_add_sp.setSuffix(" %")
        eq_add_btn = QPushButton("Add")
        eq_add_btn.setFixedWidth(48)
        eq_add_btn.clicked.connect(self._hm_add_entry_q)
        eq_add_row.addWidget(self._eq_add_sp); eq_add_row.addWidget(eq_add_btn)
        entry_form.addRow("Add entry Q:", eq_add_row)

        self._ep_lbl = QLabel("")
        self._ep_lbl.setStyleSheet("font-size:14px; font-weight:bold; color:#1A3060;")
        entry_form.addRow("Entry Price:", self._ep_lbl)
        ctrl_l.addWidget(entry_grp)

        # Lot Entry Override group
        lot_grp  = QGroupBox("Lot Entry Override")
        lot_l    = QVBoxLayout(lot_grp)
        lot_l.setSpacing(4)
        self._use_lots_chk = QCheckBox("Use StackTracker lots as entry")
        self._use_lots_chk.setChecked(False)
        lot_l.addWidget(self._use_lots_chk)
        self._lot_list = QListWidget()
        self._lot_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._lot_list.setMinimumHeight(60)
        lot_l.addWidget(self._lot_list)
        _lot_sel_row = QHBoxLayout()
        _lot_sel_all = QPushButton("All")
        _lot_sel_all.setFixedWidth(42)
        _lot_sel_none = QPushButton("None")
        _lot_sel_none.setFixedWidth(46)
        _lot_sel_all.clicked.connect(lambda: [self._lot_list.item(i).setSelected(True) for i in range(self._lot_list.count())])
        _lot_sel_none.clicked.connect(lambda: [self._lot_list.item(i).setSelected(False) for i in range(self._lot_list.count())])
        _lot_sel_row.addWidget(_lot_sel_all)
        _lot_sel_row.addWidget(_lot_sel_none)
        _lot_sel_row.addStretch()
        lot_l.addLayout(_lot_sel_row)
        self._lot_entry_lbl = QLabel("")
        self._lot_entry_lbl.setStyleSheet("font-size:10px; color:#1A3060;")
        self._lot_entry_lbl.setWordWrap(True)
        lot_l.addWidget(self._lot_entry_lbl)
        ctrl_l.addWidget(lot_grp)

        # Exit group
        exit_grp  = QGroupBox("Exit")
        exit_form = QFormLayout(exit_grp)
        exit_form.setSpacing(4)

        xlo_row = QHBoxLayout()
        self._xlo_sl = QSlider(Qt.Horizontal)
        self._xlo_sl.setRange(2010, 2044); self._xlo_sl.setValue(2027)
        self._xlo_lbl = QLabel("2027"); self._xlo_lbl.setMinimumWidth(36)
        xlo_row.addWidget(self._xlo_sl); xlo_row.addWidget(self._xlo_lbl)
        exit_form.addRow("Exit Yr Min:", xlo_row)

        xhi_row = QHBoxLayout()
        self._xhi_sl = QSlider(Qt.Horizontal)
        self._xhi_sl.setRange(2010, 2044); self._xhi_sl.setValue(2040)
        self._xhi_lbl = QLabel("2040"); self._xhi_lbl.setMinimumWidth(36)
        xhi_row.addWidget(self._xhi_sl); xhi_row.addWidget(self._xhi_lbl)
        exit_form.addRow("Exit Yr Max:", xhi_row)

        exit_form.addRow(QLabel("Exit Quantiles (Ctrl+click):"))
        self._xq_list = QListWidget()
        self._xq_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._xq_list.setMinimumHeight(80)
        for q in _qq:
            p = q * 100
            item = QListWidgetItem(f"Q{p:.4g}%" if p >= 1 else f"Q{p:.3g}%")
            item.setData(Qt.UserRole, q)
            self._xq_list.addItem(item)
            item.setSelected(True)
        exit_form.addRow(self._xq_list)

        xq_add_row = QHBoxLayout()
        xq_add_row.setSpacing(4)
        self._xq_add_sp = QDoubleSpinBox()
        self._xq_add_sp.setRange(0.001, 99.999); self._xq_add_sp.setValue(75.0)
        self._xq_add_sp.setDecimals(3); self._xq_add_sp.setSingleStep(5.0)
        self._xq_add_sp.setSuffix(" %")
        xq_add_btn = QPushButton("Add")
        xq_add_btn.setFixedWidth(48)
        xq_add_btn.clicked.connect(self._hm_add_exit_q)
        xq_rm_btn  = QPushButton("Remove")
        xq_rm_btn.setFixedWidth(62)
        xq_rm_btn.clicked.connect(self._hm_remove_exit_q)
        xq_add_row.addWidget(self._xq_add_sp)
        xq_add_row.addWidget(xq_add_btn)
        xq_add_row.addWidget(xq_rm_btn)
        exit_form.addRow(xq_add_row)
        ctrl_l.addWidget(exit_grp)

        # Colors group
        col_grp = QGroupBox("Colors")
        col_l   = QVBoxLayout(col_grp)
        col_l.setSpacing(4)

        mode_grp = QButtonGroup(self)
        mode_row = QHBoxLayout()
        for i, lbl in enumerate(["Segmented", "Data-Scaled", "Diverging"]):
            rb = QRadioButton(lbl)
            if i == 0:
                rb.setChecked(True)
            mode_grp.addButton(rb, i)
            mode_row.addWidget(rb)
        self._mode_grp = mode_grp
        col_l.addLayout(mode_row)

        col_form = QFormLayout()
        col_form.setSpacing(4)
        self._b1_sp = QDoubleSpinBox()
        self._b1_sp.setRange(-50, 200); self._b1_sp.setValue(self.m.CAGR_SEG_B1)
        self._b1_sp.setSingleStep(0.5); self._b1_sp.setDecimals(1)
        self._b2_sp = QDoubleSpinBox()
        self._b2_sp.setRange(-50, 500); self._b2_sp.setValue(self.m.CAGR_SEG_B2)
        self._b2_sp.setSingleStep(0.5); self._b2_sp.setDecimals(1)
        col_form.addRow("Break 1 (%):", self._b1_sp)
        col_form.addRow("Break 2 (%):", self._b2_sp)
        col_l.addLayout(col_form)

        clr_row = QHBoxLayout()
        self._c_lo  = ColorBtn(self.m.CAGR_SEG_C_LO)
        self._c_m1  = ColorBtn(self.m.CAGR_SEG_C_MID1)
        self._c_m2  = ColorBtn(self.m.CAGR_SEG_C_MID2)
        self._c_hi  = ColorBtn(self.m.CAGR_SEG_C_HI)
        for cb, lbl in ((self._c_lo,"Lo"),(self._c_m1,"M1"),(self._c_m2,"M2"),(self._c_hi,"Hi")):
            sub = QVBoxLayout()
            sub.setSpacing(1)
            sub.addWidget(cb)
            sub.addWidget(QLabel(lbl, alignment=Qt.AlignHCenter))
            clr_row.addLayout(sub)
        col_l.addLayout(clr_row)

        grad_form = QFormLayout()
        self._grad_sp = QSpinBox(); self._grad_sp.setRange(4, 64)
        self._grad_sp.setValue(self.m.CAGR_GRAD_STEPS)
        grad_form.addRow("Grad Steps:", self._grad_sp)
        col_l.addLayout(grad_form)
        ctrl_l.addWidget(col_grp)

        # Display group
        disp_grp  = QGroupBox("Display")
        disp_form = QFormLayout(disp_grp)
        disp_form.setSpacing(4)
        self._vfmt_cb = QComboBox()
        for lbl, val in [("CAGR %","cagr"),("Exit Price","price"),("Both","both"),
                         ("CAGR % + Stack","stack"),("Portfolio Value","port_only"),
                         ("Multiple (×)","mult_only"),("CAGR % + Multiple","cagr_mult"),
                         ("Multiple + Portfolio","mult_port"),("None","none")]:
            self._vfmt_cb.addItem(lbl, val)
        self._show_cb_chk = QCheckBox("Show colorbar"); self._show_cb_chk.setChecked(True)
        self._tight_chk   = QCheckBox("Tight layout");  self._tight_chk.setChecked(True)
        disp_form.addRow("Cell Text:", self._vfmt_cb)
        disp_form.addRow(self._show_cb_chk)
        disp_form.addRow(self._tight_chk)
        self._bg_color_btn_hm = ColorBtn(self.m.PLOT_BG_COLOR)
        self._bg_color_btn_hm.setToolTip("Plot background colour")
        disp_form.addRow("Plot bg:", self._bg_color_btn_hm)
        ctrl_l.addWidget(disp_grp)

        # Bitcoin Stack group
        hstack_grp  = QGroupBox("Bitcoin Stack")
        hstack_form = QFormLayout(hstack_grp)
        hstack_form.setSpacing(4)
        self._stack_sp_hm = QDoubleSpinBox()
        self._stack_sp_hm.setRange(0, 10_000_000)
        self._stack_sp_hm.setDecimals(8)
        self._stack_sp_hm.setValue(0.0)
        self._stack_sp_hm.setSingleStep(0.1)
        hstack_form.addRow("BTC owned:", self._stack_sp_hm)
        ctrl_l.addWidget(hstack_grp)

        # Fonts group
        hfont_grp  = QGroupBox("Fonts")
        hfont_form = QFormLayout(hfont_grp)
        hfont_form.setSpacing(4)
        hall_row = QHBoxLayout()
        self._font_all_hm = FontPicker("sans-serif")
        hall_btn = QPushButton("Apply to all")
        hall_btn.setFixedWidth(90)
        hall_btn.clicked.connect(self._apply_font_to_all)
        all_tabs_btn_hm = QPushButton("All tabs"); all_tabs_btn_hm.setFixedWidth(65)
        all_tabs_btn_hm.clicked.connect(
            lambda: self.all_fonts_applied.emit(
                self._font_all_hm.family(), self._font_all_hm.size()))
        hall_row.addWidget(self._font_all_hm)
        hall_row.addWidget(hall_btn)
        hall_row.addWidget(all_tabs_btn_hm)
        hfont_form.addRow("All fields:", hall_row)
        self._hfont_title  = FontPicker("sans-serif")
        self._hfont_axis_t = FontPicker("sans-serif")
        self._hfont_ticks  = FontPicker("sans-serif")
        self._hfont_cells  = FontPicker("sans-serif")
        hfont_form.addRow("Chart title:", self._hfont_title)
        hfont_form.addRow("Axis titles:", self._hfont_axis_t)
        hfont_form.addRow("Tick labels:", self._hfont_ticks)
        hfont_form.addRow("Cell text:",   self._hfont_cells)
        ctrl_l.addWidget(hfont_grp)

        # Save group
        save_grp  = QGroupBox("Save")
        save_form = QFormLayout(save_grp)
        save_form.setSpacing(4)
        self._fn_edit = QLineEdit("cagr_heatmap")
        self._dpi_sp  = QSpinBox(); self._dpi_sp.setRange(72, 600); self._dpi_sp.setValue(150)
        save_form.addRow("Filename:", self._fn_edit)
        save_form.addRow("DPI:", self._dpi_sp)
        btn_row = QHBoxLayout()
        svg_btn = QPushButton("Save SVG"); svg_btn.clicked.connect(lambda: self._save("svg"))
        jpg_btn = QPushButton("Save JPG"); jpg_btn.clicked.connect(lambda: self._save("jpg"))
        btn_row.addWidget(svg_btn); btn_row.addWidget(jpg_btn)
        save_form.addRow(btn_row)
        self._save_lbl = QLabel("")
        save_form.addRow(self._save_lbl)
        ctrl_l.addWidget(save_grp)

        ctrl_l.addStretch()
        ctrl_scroll.setWidget(ctrl_w)

        self._splitter = splitter
        self._splitter.addWidget(plot_w)
        self._splitter.addWidget(ctrl_scroll)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

        main_l = QHBoxLayout(self)
        main_l.setContentsMargins(0, 0, 0, 0)
        main_l.addWidget(self._splitter)

        # signals
        self._eyr_sl.valueChanged.connect(lambda v: (self._eyr_lbl.setText(str(v)), self.redraw()))
        self._eq_cb.currentIndexChanged.connect(self.redraw)
        self._xlo_sl.valueChanged.connect(lambda v: (self._xlo_lbl.setText(str(v)), self.redraw()))
        self._xhi_sl.valueChanged.connect(lambda v: (self._xhi_lbl.setText(str(v)), self.redraw()))
        self._xq_list.itemSelectionChanged.connect(self.redraw)
        for rb in self._mode_grp.buttons():
            rb.toggled.connect(self.redraw)
        for w in (self._b1_sp, self._b2_sp, self._grad_sp):
            w.valueChanged.connect(self.redraw)
        for cb in (self._c_lo, self._c_m1, self._c_m2, self._c_hi):
            cb.color_changed.connect(self.redraw)
        self._vfmt_cb.currentIndexChanged.connect(self.redraw)
        self._show_cb_chk.toggled.connect(self.redraw)
        self._tight_chk.toggled.connect(self.redraw)
        self._stack_sp_hm.valueChanged.connect(self.redraw)
        self._use_lots_chk.toggled.connect(self.redraw)
        self._lot_list.itemSelectionChanged.connect(self.redraw)
        for fp in (self._hfont_title, self._hfont_axis_t, self._hfont_ticks, self._hfont_cells):
            fp.font_changed.connect(self.redraw)
            fp.size_changed.connect(self.redraw)
        self._bg_color_btn_hm.color_changed.connect(self.redraw)

    # ── redraw ────────────────────────────────────────────────────────────────
    def redraw(self, *_):
        if self._busy:
            return
        m      = self.m
        font_title  = self._hfont_title.family()
        font_axis_t = self._hfont_axis_t.family()
        font_ticks  = self._hfont_ticks.family()
        font_cells  = self._hfont_cells.family()
        font_title_sz = self._hfont_title.size()
        font_axis_sz  = self._hfont_axis_t.size()
        font_ticks_sz = self._hfont_ticks.size()
        hm_stack    = self._stack_sp_hm.value()
        eyr    = self._eyr_sl.value()
        eq     = self._eq_cb.currentData()
        ep     = float(qr_price(eq, yr_to_t(eyr, m.genesis), m.qr_fits))
        self._ep_lbl.setText(fmt_price(ep))

        # Lot entry override
        _use_lots   = False
        _sel_lots   = []
        _entry_t    = yr_to_t(eyr, m.genesis)
        if (self._use_lots_chk.isChecked()
                and self._lot_source is not None
                and self._lot_list.count() > 0):
            _sel_idxs = [self._lot_list.item(i).data(Qt.UserRole)
                         for i in range(self._lot_list.count())
                         if self._lot_list.item(i).isSelected()]
            _all_lots = self._lot_source.get_lots()
            _sel_lots = [_all_lots[j] for j in _sel_idxs if j < len(_all_lots)]
            if _sel_lots:
                _total_w = sum(l["btc"] for l in _sel_lots)
                ep       = sum(l["price"] * l["btc"] for l in _sel_lots) / _total_w
                _entry_t = sum(
                    (pd.Timestamp(l["date"]) - m.genesis).days / 365.25 * l["btc"]
                    for l in _sel_lots) / _total_w
                _avg_pct = sum(l["pct_q"] * l["btc"] for l in _sel_lots) / _total_w
                self._ep_lbl.setText(fmt_price(ep))
                self._lot_entry_lbl.setText(
                    f"Wtd avg: {fmt_price(ep)}/BTC  |  {_total_w:.6g} BTC  |  Q{_avg_pct*100:.2f}%")
                _use_lots = True
            else:
                self._lot_entry_lbl.setText("(no lots selected)")
        else:
            self._lot_entry_lbl.setText("")

        xlo   = self._xlo_sl.value()
        xhi   = self._xhi_sl.value()
        eyrs  = [y for y in range(xlo, xhi + 1) if y >= eyr]
        xqs   = sorted([self._xq_list.item(i).data(Qt.UserRole)
                        for i in range(self._xq_list.count())
                        if self._xq_list.item(i).isSelected()],
                       reverse=True)
        if not eyrs or not xqs:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_title("No data — adjust Entry / Exit settings", color=m.TITLE_COLOR)
            self.canvas.draw_idle()
            return

        mc = np.zeros((len(xqs), len(eyrs)))
        mp = np.zeros((len(xqs), len(eyrs)))
        mm = np.zeros((len(xqs), len(eyrs)))
        for ci, ey in enumerate(eyrs):
            et = yr_to_t(ey, m.genesis)
            nyr_real = et - _entry_t if _use_lots else float(ey - eyr)
            for ri, xq in enumerate(xqs):
                xp           = float(qr_price(xq, et, m.qr_fits))
                mp[ri, ci]   = xp
                mm[ri, ci]   = xp / ep if ep > 0 else 0.0
                if nyr_real <= 0:
                    mc[ri, ci] = (xp / ep - 1.0) * 100.0
                else:
                    mc[ri, ci] = ((xp / ep) ** (1.0 / nyr_real) - 1.0) * 100.0

        nd   = max(2, 2 * self._grad_sp.value())
        mode = self._mode_grp.checkedId()
        if mode == 0:  # Segmented
            cm, nm = make_seg_cmap(mc, self._b1_sp.value(), self._b2_sp.value(),
                                   self._c_lo.color(), self._c_m1.color(),
                                   self._c_m2.color(), self._c_hi.color(), nd)
        elif mode == 1:  # Data-Scaled
            base = mcolors.LinearSegmentedColormap.from_list(
                "_ds", [self._c_lo.color(), self._c_m1.color(), self._c_hi.color()], N=512)
            cm = mcolors.ListedColormap([base(k / (nd - 1)) for k in range(nd)])
            nm = mcolors.BoundaryNorm(np.linspace(float(mc.min()), float(mc.max()), nd + 1), cm.N)
        else:  # Diverging
            vn  = float(mc.min()); vx = float(mc.max())
            rng = max(vx - vn, 1e-6)
            vc  = min(max(0.0, vn + rng * 0.01), vx - rng * 0.01)
            base2 = mcolors.LinearSegmentedColormap.from_list(
                "_dv", [self._c_lo.color(), self._c_m1.color(), self._c_hi.color()], N=512)
            cm = mcolors.ListedColormap([base2(k / (nd - 1)) for k in range(nd)])
            nm = mcolors.TwoSlopeNorm(vmin=vn, vcenter=vc, vmax=vx)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(self._bg_color_btn_hm.color())
        for sp in ax.spines.values():
            sp.set_edgecolor(m.SPINE_COLOR)
        ax.tick_params(colors=m.TEXT_COLOR)

        im = ax.imshow(mc, aspect="auto", cmap=cm, norm=nm, origin="upper")

        vfmt = self._vfmt_cb.currentData()
        fs   = self._hfont_cells.size()
        for ri in range(len(xqs)):
            for ci in range(len(eyrs)):
                vc2 = mc[ri, ci]; vp2 = mp[ri, ci]
                if vfmt == "cagr":
                    tx = f"{vc2:+.0f}%"
                elif vfmt == "price":
                    tx = fmt_price(vp2)
                elif vfmt == "both":
                    tx = f"{vc2:+.0f}%\n{fmt_price(vp2)}"
                elif vfmt == "stack":
                    sv = fmt_price(vp2 * hm_stack) if hm_stack > 0 else fmt_price(vp2)
                    tx = f"{vc2:+.0f}%\n{sv}"
                elif vfmt == "port_only":
                    tx = fmt_price(vp2 * hm_stack) if hm_stack > 0 else fmt_price(vp2)
                elif vfmt == "mult_only":
                    tx = f"{mm[ri, ci]:.2f}×"
                elif vfmt == "cagr_mult":
                    tx = f"{vc2:+.0f}%\n{mm[ri, ci]:.2f}×"
                elif vfmt == "mult_port":
                    _pv = fmt_price(vp2 * hm_stack) if hm_stack > 0 else fmt_price(vp2)
                    tx = f"{mm[ri, ci]:.2f}×\n{_pv}"
                else:
                    tx = ""
                if tx:
                    ax.text(ci, ri, tx, ha="center", va="center",
                            fontsize=fs, color="#1A1A1A", fontweight="bold",
                            fontfamily=font_cells)

        ax.set_xticks(range(len(eyrs)))
        ax.set_xticklabels([str(y) for y in eyrs], rotation=45, ha="right",
                           fontsize=font_ticks_sz, fontfamily=font_ticks)
        ax.set_yticks(range(len(xqs)))
        ax.set_yticklabels(
            [f"Q{q*100:.4g}%" if q*100 >= 1 else f"Q{q*100:.3g}%" for q in xqs],
            fontsize=font_ticks_sz, fontfamily=font_ticks)
        ax.set_xlabel("Exit Year", color=m.TEXT_COLOR, fontfamily=font_axis_t,
                      fontsize=font_axis_sz)
        ax.set_ylabel("Exit Quantile", color=m.TEXT_COLOR, fontfamily=font_axis_t,
                      fontsize=font_axis_sz)
        pct_str = f"Q{eq*100:.4g}%" if eq*100 >= 1 else f"Q{eq*100:.3g}%"
        _stk_str = f"   |  Stack: {hm_stack:.8g} BTC" if hm_stack > 0 else ""
        if _use_lots and _sel_lots:
            _dstr = _sel_lots[0]["date"][:7] if len(_sel_lots) == 1 else f"{len(_sel_lots)} lots"
            ax.set_title(
                f"CAGR Heatmap  —  Entry: {_dstr} @ {fmt_price(ep)}{_stk_str}",
                fontsize=font_title_sz, color=m.TITLE_COLOR, fontfamily=font_title)
        else:
            ax.set_title(
                f"CAGR Heatmap  —  Entry: Jan {eyr} @ {pct_str}  ≈  {fmt_price(ep)}{_stk_str}",
                fontsize=font_title_sz, color=m.TITLE_COLOR, fontfamily=font_title)

        if self._show_cb_chk.isChecked():
            if mode == 0:  # Segmented: annotated ticks
                tks = [float(mc.min())]
                if float(mc.min()) < self._b1_sp.value() < float(mc.max()):
                    tks.append(self._b1_sp.value())
                if float(mc.min()) < self._b2_sp.value() < float(mc.max()):
                    tks.append(self._b2_sp.value())
                tks.append(float(mc.max()))
                cbar = self.fig.colorbar(im, ax=ax, label="CAGR (%)", shrink=0.85)
                cbar.set_ticks(tks)
                cbar.set_ticklabels([f"{t:+.0f}%" for t in tks])
            else:
                self.fig.colorbar(im, ax=ax, label="CAGR (%)", shrink=0.85)

        if self._tight_chk.isChecked():
            try:
                self.fig.tight_layout()
            except Exception:
                pass
        self.canvas.draw_idle()

    def _save(self, fmt):
        fn   = self._fn_edit.text().strip() or "cagr_heatmap"
        path = _desktop_save_path(self, fn, fmt)
        if not path:
            return
        try:
            self.fig.savefig(path, format=fmt,
                             bbox_inches="tight", dpi=self._dpi_sp.value())
            self._save_lbl.setText(f"✓ Saved {Path(path).name}")
        except Exception as e:
            self._save_lbl.setText(f"Error: {e}")

    def _apply_font_to_all(self):
        fam = self._font_all_hm.family()
        sz  = self._font_all_hm.size()
        for fp in (self._hfont_title, self._hfont_axis_t, self._hfont_ticks, self._hfont_cells):
            fp.set_family(fam)
            fp.set_size(sz)
        self.redraw()

    def _hm_add_entry_q(self):
        q = self._eq_add_sp.value() / 100.0
        for i in range(self._eq_cb.count()):
            if abs(float(self._eq_cb.itemData(i)) - q) < 1e-6:
                self._eq_cb.setCurrentIndex(i)
                return
        if q not in self.m.qr_fits:
            try:
                self.m.qr_fits[q] = _fit_one_qr(self.m, q)
            except Exception as e:
                QMessageBox.warning(self, "Fit error", f"Could not fit Q{q*100:.4g}%:\n{e}")
                return
        pct = q * 100
        lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
        self._eq_cb.addItem(lbl, q)
        self._eq_cb.setCurrentIndex(self._eq_cb.count() - 1)

    def _hm_add_exit_q(self):
        q = self._xq_add_sp.value() / 100.0
        for i in range(self._xq_list.count()):
            if abs(float(self._xq_list.item(i).data(Qt.UserRole)) - q) < 1e-6:
                self._xq_list.item(i).setSelected(True)
                return
        if q not in self.m.qr_fits:
            try:
                self.m.qr_fits[q] = _fit_one_qr(self.m, q)
            except Exception as e:
                QMessageBox.warning(self, "Fit error", f"Could not fit Q{q*100:.4g}%:\n{e}")
                return
        pct = q * 100
        lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
        item = QListWidgetItem(lbl)
        item.setData(Qt.UserRole, q)
        self._xq_list.addItem(item)
        item.setSelected(True)
        self.redraw()

    def _hm_remove_exit_q(self):
        base_qs = set(self.m.QR_QUANTILES)
        for i in reversed(range(self._xq_list.count())):
            item = self._xq_list.item(i)
            if item.isSelected() and float(item.data(Qt.UserRole)) not in base_qs:
                self._xq_list.takeItem(i)
        self.redraw()

    def _factory_reset(self):
        self._busy = True
        self._eyr_sl.setValue(2025)
        self._eyr_lbl.setText("2025")
        # Set Q86% as default entry quantile (fit if needed)
        _q86 = 0.86
        if _q86 not in self.m.qr_fits:
            try:
                self.m.qr_fits[_q86] = _fit_one_qr(self.m, _q86)
            except Exception:
                pass
        _q86_idx = -1
        for _i in range(self._eq_cb.count()):
            if abs(float(self._eq_cb.itemData(_i)) - _q86) < 1e-6:
                _q86_idx = _i
                break
        if _q86_idx == -1 and _q86 in self.m.qr_fits:
            _pct = _q86 * 100
            self._eq_cb.addItem(f"Q{_pct:.4g}%", _q86)
            _q86_idx = self._eq_cb.count() - 1
        if _q86_idx >= 0:
            self._eq_cb.setCurrentIndex(_q86_idx)
        else:
            self._eq_cb.setCurrentIndex(min(2, self._eq_cb.count() - 1))
        self._xlo_sl.setValue(2027)
        self._xlo_lbl.setText("2027")
        self._xhi_sl.setValue(2040)
        self._xhi_lbl.setText("2040")
        _dq_set = set(_DEFAULT_QS)
        for i in range(self._xq_list.count()):
            item = self._xq_list.item(i)
            item.setSelected(item.data(Qt.UserRole) in _dq_set)
        self._mode_grp.button(0).setChecked(True)
        self._b1_sp.setValue(self.m.CAGR_SEG_B1)
        self._b2_sp.setValue(self.m.CAGR_SEG_B2)
        self._c_lo.set_color(self.m.CAGR_SEG_C_LO)
        self._c_m1.set_color(self.m.CAGR_SEG_C_MID1)
        self._c_m2.set_color(self.m.CAGR_SEG_C_MID2)
        self._c_hi.set_color(self.m.CAGR_SEG_C_HI)
        self._grad_sp.setValue(self.m.CAGR_GRAD_STEPS)
        self._vfmt_cb.setCurrentIndex(0)
        self._show_cb_chk.setChecked(True)
        self._tight_chk.setChecked(True)
        self._stack_sp_hm.setValue(0.0)
        self._use_lots_chk.setChecked(False)
        for fp in (self._hfont_title, self._hfont_axis_t, self._hfont_ticks, self._hfont_cells,
                   self._font_all_hm):
            fp.set_family("sans-serif")
        self._busy = False
        self.redraw()

    def showEvent(self, event):
        super().showEvent(event)
        sizes = getattr(self, '_pending_splitter', None)
        if sizes is not None:
            self._splitter.setSizes(sizes)
            self._pending_splitter = None

    def _collect_settings(self):
        sel_qs = [
            self._xq_list.item(i).data(Qt.UserRole)
            for i in range(self._xq_list.count())
            if self._xq_list.item(i).isSelected()
        ]
        # Save full lists so custom quantiles survive reload
        entry_q_all = [self._eq_cb.itemData(i) for i in range(self._eq_cb.count())]
        exit_q_all  = [self._xq_list.item(i).data(Qt.UserRole)
                       for i in range(self._xq_list.count())]
        return {
            "entry_yr":      self._eyr_sl.value(),
            "entry_q_idx":   self._eq_cb.currentIndex(),
            "entry_q_all":   entry_q_all,
            "exit_yr_lo":    self._xlo_sl.value(),
            "exit_yr_hi":    self._xhi_sl.value(),
            "exit_q_sel":    sel_qs,
            "exit_q_all":    exit_q_all,
            "color_mode":    self._mode_grp.checkedId(),
            "break1":        self._b1_sp.value(),
            "break2":        self._b2_sp.value(),
            "c_lo":          self._c_lo.color(),
            "c_m1":          self._c_m1.color(),
            "c_m2":          self._c_m2.color(),
            "c_hi":          self._c_hi.color(),
            "grad_steps":    self._grad_sp.value(),
            "cell_text_idx": self._vfmt_cb.currentIndex(),
            "show_colorbar":  self._show_cb_chk.isChecked(),
            "tight_layout":   self._tight_chk.isChecked(),
            "stack_hm":       self._stack_sp_hm.value(),
            "use_lots":       self._use_lots_chk.isChecked(),
            "hfont_title":    self._hfont_title.family(),
            "hfont_axis_t":   self._hfont_axis_t.family(),
            "hfont_ticks":    self._hfont_ticks.family(),
            "hfont_cells":    self._hfont_cells.family(),
            "hfont_title_sz":  self._hfont_title.size(),
            "hfont_axis_t_sz": self._hfont_axis_t.size(),
            "hfont_ticks_sz":  self._hfont_ticks.size(),
            "hfont_cells_sz":  self._hfont_cells.size(),
            "bg_color_hm":    self._bg_color_btn_hm.color(),
            "splitter_sizes": list(self._splitter.sizes()),
        }

    def _apply_settings(self, d):
        self._busy = True
        try:
            self._eyr_sl.setValue(d.get("entry_yr", 2025))
            self._eyr_lbl.setText(str(self._eyr_sl.value()))

            # Restore any custom entry quantiles before setting the index
            base_qs = set(self.m.QR_QUANTILES)
            existing_entry = {float(self._eq_cb.itemData(i))
                              for i in range(self._eq_cb.count())}
            for q in d.get("entry_q_all", []):
                q = float(q)
                if q not in base_qs and q not in existing_entry:
                    if q not in self.m.qr_fits:
                        try: self.m.qr_fits[q] = _fit_one_qr(self.m, q)
                        except Exception: continue
                    pct = q * 100
                    lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
                    self._eq_cb.addItem(lbl, q)
                    existing_entry.add(q)

            self._eq_cb.setCurrentIndex(d.get("entry_q_idx", min(2, self._eq_cb.count() - 1)))
            self._xlo_sl.setValue(d.get("exit_yr_lo", 2027))
            self._xlo_lbl.setText(str(self._xlo_sl.value()))
            self._xhi_sl.setValue(d.get("exit_yr_hi", 2040))
            self._xhi_lbl.setText(str(self._xhi_sl.value()))

            # Restore any custom exit quantiles before setting selection
            existing_exit = {float(self._xq_list.item(i).data(Qt.UserRole))
                             for i in range(self._xq_list.count())}
            for q in d.get("exit_q_all", []):
                q = float(q)
                if q not in base_qs and q not in existing_exit:
                    if q not in self.m.qr_fits:
                        try: self.m.qr_fits[q] = _fit_one_qr(self.m, q)
                        except Exception: continue
                    pct = q * 100
                    lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
                    item = QListWidgetItem(lbl)
                    item.setData(Qt.UserRole, q)
                    self._xq_list.addItem(item)
                    existing_exit.add(q)

            sel_qs = set(float(q) for q in d.get("exit_q_sel", []))
            for i in range(self._xq_list.count()):
                q = float(self._xq_list.item(i).data(Qt.UserRole))
                self._xq_list.item(i).setSelected(not sel_qs or q in sel_qs)
            btn = self._mode_grp.button(d.get("color_mode", 0))
            if btn: btn.setChecked(True)
            self._b1_sp.setValue(d.get("break1", self.m.CAGR_SEG_B1))
            self._b2_sp.setValue(d.get("break2", self.m.CAGR_SEG_B2))
            self._c_lo.set_color(d.get("c_lo", self.m.CAGR_SEG_C_LO))
            self._c_m1.set_color(d.get("c_m1", self.m.CAGR_SEG_C_MID1))
            self._c_m2.set_color(d.get("c_m2", self.m.CAGR_SEG_C_MID2))
            self._c_hi.set_color(d.get("c_hi", self.m.CAGR_SEG_C_HI))
            self._grad_sp.setValue(d.get("grad_steps", self.m.CAGR_GRAD_STEPS))
            self._vfmt_cb.setCurrentIndex(d.get("cell_text_idx", 0))
            self._show_cb_chk.setChecked(d.get("show_colorbar", True))
            self._tight_chk.setChecked(d.get("tight_layout", True))
            self._stack_sp_hm.setValue(float(d.get("stack_hm", 0.0)))
            self._use_lots_chk.setChecked(bool(d.get("use_lots", False)))
            self._hfont_title.set_family(d.get("hfont_title",  "sans-serif"))
            self._hfont_axis_t.set_family(d.get("hfont_axis_t", "sans-serif"))
            self._hfont_ticks.set_family(d.get("hfont_ticks",  "sans-serif"))
            self._hfont_cells.set_family(d.get("hfont_cells",  "sans-serif"))
            self._hfont_title.set_size(d.get("hfont_title_sz", 11))
            self._hfont_axis_t.set_size(d.get("hfont_axis_t_sz", 10))
            self._hfont_ticks.set_size(d.get("hfont_ticks_sz", 9))
            self._hfont_cells.set_size(d.get("hfont_cells_sz", 9))
            self._bg_color_btn_hm.set_color(d.get("bg_color_hm", self.m.PLOT_BG_COLOR))
            if "splitter_sizes" in d:
                self._pending_splitter = [int(x) for x in d["splitter_sizes"]]
        finally:
            self._busy = False
        self.redraw()



# ── DCATab ────────────────────────────────────────────────────────────────────

class DCATab(QWidget):
    """Tab 3 — Bitcoin DCA Accumulator: simulate dollar-cost-averaging into BTC."""
    q_state_changed = pyqtSignal(list)
    all_fonts_applied = pyqtSignal(str, int)

    def __init__(self, model: ModelData):
        super().__init__()
        self.m = model
        self._busy = False
        self._q_state = []
        for i, q in enumerate(model.QR_QUANTILES):
            pct = q * 100
            lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
            self._q_state.append({
                "q": q, "lbl": lbl, "vis": True,
                "color": model.qr_colors.get(q, _CB_COLORS[i % len(_CB_COLORS)]),
                "ls":    model.qr_linestyles.get(q, _CB_STYLES[i % len(_CB_STYLES)]),
                "lw":    2.0 if abs(q - 0.5) < 1e-6 else 1.5,
            })
        self._lot_source = None
        self._build_ui()

    # ── build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        plot_w = QWidget()
        plot_l = QVBoxLayout(plot_w)
        plot_l.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(10, 6.5), facecolor=self.m.PLOT_BG_COLOR)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavToolbar(self.canvas, plot_w)
        plot_l.addWidget(toolbar)
        plot_l.addWidget(self.canvas)

        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(295)
        ctrl_w = QWidget()
        ctrl_l = QVBoxLayout(ctrl_w)
        ctrl_l.setSpacing(6)

        fac_btn = QPushButton("Reset to Factory Defaults")
        fac_btn.clicked.connect(self._factory_reset)
        ctrl_l.addWidget(fac_btn)

        # Bitcoin Stack
        stk_grp  = QGroupBox("Bitcoin Stack")
        stk_form = QFormLayout(stk_grp)
        stk_form.setSpacing(4)
        self._stack_sp = QDoubleSpinBox()
        self._stack_sp.setRange(0, 10_000_000); self._stack_sp.setDecimals(8)
        self._stack_sp.setValue(0.0); self._stack_sp.setSingleStep(0.1)
        stk_form.addRow("BTC owned:", self._stack_sp)
        ctrl_l.addWidget(stk_grp)

        # LEO group
        leo_grp3 = QGroupBox("Lot Entry Override (LEO)")
        leo_l3   = QVBoxLayout(leo_grp3)
        leo_l3.setSpacing(4)
        self._use_lots_chk = QCheckBox("Use lots as starting stack")
        self._use_lots_chk.setChecked(False)
        leo_l3.addWidget(self._use_lots_chk)
        self._lot_list = QListWidget()
        self._lot_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._lot_list.setMinimumHeight(60)
        leo_l3.addWidget(self._lot_list)
        _leo_row3 = QHBoxLayout()
        _leo_all3 = QPushButton("All");  _leo_all3.setFixedWidth(38)
        _leo_none3 = QPushButton("None"); _leo_none3.setFixedWidth(44)
        _leo_all3.clicked.connect(lambda: [self._lot_list.item(_i).setSelected(True) for _i in range(self._lot_list.count())])
        _leo_none3.clicked.connect(lambda: [self._lot_list.item(_i).setSelected(False) for _i in range(self._lot_list.count())])
        _leo_row3.addWidget(_leo_all3); _leo_row3.addWidget(_leo_none3); _leo_row3.addStretch()
        leo_l3.addLayout(_leo_row3)
        self._leo_lbl = QLabel("")
        self._leo_lbl.setStyleSheet("font-size:9px;")
        self._leo_lbl.setWordWrap(True)
        leo_l3.addWidget(self._leo_lbl)
        ctrl_l.addWidget(leo_grp3)

        # DCA Investment
        inv_grp  = QGroupBox("DCA Investment")
        inv_form = QFormLayout(inv_grp)
        inv_form.setSpacing(4)
        self._amount_sp = QDoubleSpinBox()
        self._amount_sp.setRange(1, 1_000_000); self._amount_sp.setValue(500)
        self._amount_sp.setSingleStep(100); self._amount_sp.setDecimals(2)
        self._amount_sp.setPrefix("$")
        inv_form.addRow("Per period:", self._amount_sp)
        self._freq_cb = QComboBox()
        for lbl, val in [("Monthly","Monthly"),("Weekly","Weekly"),
                         ("Quarterly","Quarterly"),("Annually","Annually")]:
            self._freq_cb.addItem(lbl, val)
        inv_form.addRow("Frequency:", self._freq_cb)

        syr_row = QHBoxLayout()
        self._syr_sl = QSlider(Qt.Horizontal)
        self._syr_sl.setRange(2010, 2039); self._syr_sl.setValue(2020)
        self._syr_lbl = QLabel("2020"); self._syr_lbl.setMinimumWidth(36)
        syr_row.addWidget(self._syr_sl); syr_row.addWidget(self._syr_lbl)
        inv_form.addRow("Start Year:", syr_row)

        eyr_row = QHBoxLayout()
        self._eyr_sl = QSlider(Qt.Horizontal)
        self._eyr_sl.setRange(2015, 2060); self._eyr_sl.setValue(2040)
        self._eyr_lbl = QLabel("2040"); self._eyr_lbl.setMinimumWidth(36)
        eyr_row.addWidget(self._eyr_sl); eyr_row.addWidget(self._eyr_lbl)
        inv_form.addRow("End Year:", eyr_row)
        ctrl_l.addWidget(inv_grp)

        # Price Scenarios
        scen_grp   = QGroupBox("Price Scenarios")
        scen_outer = QVBoxLayout(scen_grp)
        scen_outer.setSpacing(4)
        self._q_rows_w = QWidget()
        self._q_rows_l = QVBoxLayout(self._q_rows_w)
        self._q_rows_l.setSpacing(2)
        self._q_rows_l.setContentsMargins(0, 0, 0, 0)
        scen_outer.addWidget(self._q_rows_w)
        self._rebuild_q_rows()
        add_row = QHBoxLayout(); add_row.setSpacing(4)
        add_row.addWidget(QLabel("Add Q:"))
        self._add_q_sp = QDoubleSpinBox()
        self._add_q_sp.setRange(0.001, 99.999); self._add_q_sp.setValue(75.0)
        self._add_q_sp.setDecimals(3); self._add_q_sp.setSingleStep(5.0)
        self._add_q_sp.setSuffix(" %")
        add_btn = QPushButton("Add"); add_btn.setFixedWidth(48)
        add_btn.clicked.connect(self._q_add)
        add_row.addWidget(self._add_q_sp); add_row.addWidget(add_btn)
        scen_outer.addLayout(add_row)
        ctrl_l.addWidget(scen_grp)

        # Display
        disp_grp  = QGroupBox("Display")
        disp_form = QFormLayout(disp_grp)
        disp_form.setSpacing(4)
        self._disp_cb = QComboBox()
        for lbl, val in [("BTC Balance","btc"),("USD Value","usd")]:
            self._disp_cb.addItem(lbl, val)
        disp_form.addRow("Y-axis:", self._disp_cb)
        self._log_y_chk = QCheckBox("Log Y scale"); self._log_y_chk.setChecked(False)
        self._today_chk = QCheckBox("Show today"); self._today_chk.setChecked(True)
        self._dual_y_chk = QCheckBox("Dual Y-axis"); self._dual_y_chk.setChecked(False)
        self._minor_ticks_chk = QCheckBox("Minor log ticks (×0.1 dec)")
        self._minor_ticks_chk.setChecked(False)
        disp_form.addRow(self._log_y_chk)
        disp_form.addRow(self._today_chk)
        disp_form.addRow(self._dual_y_chk)
        disp_form.addRow(self._minor_ticks_chk)
        self._bg_color_btn = ColorBtn(self.m.PLOT_BG_COLOR)
        self._bg_color_btn.setToolTip("Plot background colour")
        disp_form.addRow("Plot bg:", self._bg_color_btn)
        ctrl_l.addWidget(disp_grp)

        # Fonts
        font_grp  = QGroupBox("Fonts")
        font_form = QFormLayout(font_grp)
        font_form.setSpacing(4)
        all_row = QHBoxLayout()
        self._font_all_b = FontPicker("sans-serif")
        all_btn = QPushButton("Apply to all"); all_btn.setFixedWidth(90)
        all_btn.clicked.connect(self._apply_font_to_all)
        all_tabs_btn = QPushButton("All tabs"); all_tabs_btn.setFixedWidth(65)
        all_tabs_btn.clicked.connect(
            lambda: self.all_fonts_applied.emit(
                self._font_all_b.family(), self._font_all_b.size()))
        all_row.addWidget(self._font_all_b); all_row.addWidget(all_btn)
        all_row.addWidget(all_tabs_btn)
        font_form.addRow("All fields:", all_row)
        self._font_title       = FontPicker("sans-serif")
        self._font_axis_t      = FontPicker("sans-serif")
        self._font_ticks       = FontPicker("sans-serif")
        self._font_ticks_minor = FontPicker("sans-serif")
        self._font_legend      = FontPicker("sans-serif")
        font_form.addRow("Chart title:", self._font_title)
        font_form.addRow("Axis titles:", self._font_axis_t)
        font_form.addRow("Major ticks:", self._font_ticks)
        font_form.addRow("Minor ticks:", self._font_ticks_minor)
        font_form.addRow("Legend:",      self._font_legend)
        for fp in (self._font_title, self._font_axis_t, self._font_ticks,
                   self._font_ticks_minor, self._font_legend):
            fp.font_changed.connect(self.redraw)
            fp.size_changed.connect(self.redraw)
        self._bg_color_btn.color_changed.connect(self.redraw)
        ctrl_l.addWidget(font_grp)

        # Save
        save_grp  = QGroupBox("Save")
        save_form = QFormLayout(save_grp)
        save_form.setSpacing(4)
        self._fn_edit = QLineEdit("btc_dca")
        self._dpi_sp  = QSpinBox(); self._dpi_sp.setRange(72, 600); self._dpi_sp.setValue(150)
        save_form.addRow("Filename:", self._fn_edit)
        save_form.addRow("DPI:", self._dpi_sp)
        btn_row = QHBoxLayout()
        svg_btn = QPushButton("Save SVG"); svg_btn.clicked.connect(lambda: self._save("svg"))
        jpg_btn = QPushButton("Save JPG"); jpg_btn.clicked.connect(lambda: self._save("jpg"))
        btn_row.addWidget(svg_btn); btn_row.addWidget(jpg_btn)
        save_form.addRow(btn_row)
        self._save_lbl = QLabel("")
        save_form.addRow(self._save_lbl)
        ctrl_l.addWidget(save_grp)

        ctrl_l.addStretch()
        ctrl_scroll.setWidget(ctrl_w)
        self._splitter = splitter
        self._splitter.addWidget(plot_w); self._splitter.addWidget(ctrl_scroll)
        self._splitter.setStretchFactor(0, 1); self._splitter.setStretchFactor(1, 0)
        main_l = QHBoxLayout(self)
        main_l.setContentsMargins(0, 0, 0, 0)
        main_l.addWidget(self._splitter)

        # Signals
        self._syr_sl.valueChanged.connect(lambda v: (self._syr_lbl.setText(str(v)), self.redraw()))
        self._eyr_sl.valueChanged.connect(lambda v: (self._eyr_lbl.setText(str(v)), self.redraw()))
        self._amount_sp.valueChanged.connect(self.redraw)
        self._freq_cb.currentIndexChanged.connect(self.redraw)
        self._disp_cb.currentIndexChanged.connect(self.redraw)
        self._log_y_chk.toggled.connect(self.redraw)
        self._today_chk.toggled.connect(self.redraw)
        self._stack_sp.valueChanged.connect(self.redraw)
        self._use_lots_chk.toggled.connect(self.redraw)
        self._lot_list.itemSelectionChanged.connect(self.redraw)
        self._dual_y_chk.toggled.connect(self.redraw)
        self._minor_ticks_chk.toggled.connect(self.redraw)

    # ── quantile rows ─────────────────────────────────────────────────────────
    def _rebuild_q_rows(self):
        while self._q_rows_l.count():
            w = self._q_rows_l.takeAt(0).widget()
            if w: w.deleteLater()
        for i, qs in enumerate(self._q_state):
            row_w = QWidget(); row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0); row_l.setSpacing(3)
            cb = QCheckBox(qs["lbl"]); cb.setChecked(qs["vis"]); cb.setMinimumWidth(85)
            cb.toggled.connect(lambda v, idx=i: self._q_vis(idx, v))
            clr_btn = ColorBtn(qs["color"])
            clr_btn.color_changed.connect(lambda c, idx=i: self._q_color(idx, c))
            ls_cb = QComboBox(); ls_cb.addItems(LS_NAMES); ls_cb.setCurrentIndex(_ls_index(qs["ls"]))
            ls_cb.setFixedWidth(105)
            ls_cb.currentIndexChanged.connect(lambda v, idx=i: self._q_ls(idx, v))
            lw_sp = QDoubleSpinBox(); lw_sp.setRange(0.3, 6.0); lw_sp.setValue(qs["lw"])
            lw_sp.setSingleStep(0.25); lw_sp.setFixedWidth(56)
            lw_sp.valueChanged.connect(lambda v, idx=i: self._q_lw(idx, v))
            rm_btn = QPushButton("×"); rm_btn.setFixedSize(22, 22)
            rm_btn.clicked.connect(lambda _, idx=i: self._q_remove(idx))
            row_l.addWidget(cb); row_l.addWidget(clr_btn)
            row_l.addWidget(ls_cb); row_l.addWidget(lw_sp); row_l.addWidget(rm_btn)
            self._q_rows_l.addWidget(row_w)

    def _emit_q_changed(self):
        self.q_state_changed.emit(list(self._q_state))

    def _q_vis(self, idx, v):   self._q_state[idx]["vis"] = v;   self._emit_q_changed(); self.redraw()
    def _q_color(self, idx, c): self._q_state[idx]["color"] = c; self._emit_q_changed(); self.redraw()
    def _q_ls(self, idx, v):    self._q_state[idx]["ls"] = LS_SPECS[v]; self._emit_q_changed(); self.redraw()
    def _q_lw(self, idx, v):    self._q_state[idx]["lw"] = v;   self._emit_q_changed(); self.redraw()

    def _q_remove(self, idx):
        self._q_state.pop(idx); self._rebuild_q_rows(); self._emit_q_changed(); self.redraw()

    def _q_add(self):
        q = self._add_q_sp.value() / 100.0
        for s in self._q_state:
            if abs(s["q"] - q) < 1e-6: return
        if q not in self.m.qr_fits:
            try: self.m.qr_fits[q] = _fit_one_qr(self.m, q)
            except Exception as e:
                QMessageBox.warning(self, "Fit error", f"Could not fit Q{q*100:.4g}%:\n{e}")
                return
        n = len(self._q_state); pct = q * 100
        lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
        self._q_state.append({"q": q, "lbl": lbl, "vis": True,
                              "color": _CB_COLORS[n % len(_CB_COLORS)],
                              "ls": _CB_STYLES[n % len(_CB_STYLES)], "lw": 1.5})
        self._q_state.sort(key=lambda s: s["q"])
        self._rebuild_q_rows(); self._emit_q_changed(); self.redraw()

    def set_lot_source(self, tracker_tab):
        self._lot_source = tracker_tab

    def refresh_lot_list(self):
        self._lot_list.clear()
        if self._lot_source is None:
            return
        for i, lot in enumerate(self._lot_source.get_lots()):
            pct_str = f"Q{lot['pct_q']*100:.2f}%"
            lbl = f"{lot['date']}  {lot['btc']:.4f} BTC @ {fmt_price(lot['price'])}  ({pct_str})"
            item = QListWidgetItem(lbl)
            item.setData(Qt.UserRole, i)
            self._lot_list.addItem(item)
            item.setSelected(True)

    def _apply_font_to_all(self):
        fam = self._font_all_b.family()
        sz  = self._font_all_b.size()
        for fp in (self._font_title, self._font_axis_t, self._font_ticks,
                   self._font_ticks_minor, self._font_legend):
            fp.set_family(fam)
            fp.set_size(sz)
        self.redraw()

    # ── redraw ────────────────────────────────────────────────────────────────
    def redraw(self, *_):
        if self._busy: return
        m = self.m
        bg_color = self._bg_color_btn.color()
        FREQ_PPY = {"Weekly": 52, "Monthly": 12, "Quarterly": 4, "Annually": 1}
        freq_str = self._freq_cb.currentData()
        ppy = FREQ_PPY[freq_str]
        dt  = 1.0 / ppy
        start_yr = self._syr_sl.value()
        end_yr   = self._eyr_sl.value()
        if end_yr <= start_yr:
            return
        t_start = max(yr_to_t(start_yr, m.genesis), 1.0)
        t_end   = yr_to_t(end_yr, m.genesis)
        ts      = np.arange(t_start, t_end + dt * 0.5, dt)
        if len(ts) == 0:
            return

        start_stack = self._stack_sp.value()
        # LEO override: use selected lots total as starting stack
        if (self._use_lots_chk.isChecked() and self._lot_source is not None
                and self._lot_list.count() > 0):
            _sel3 = [self._lot_list.item(_i).data(Qt.UserRole)
                     for _i in range(self._lot_list.count())
                     if self._lot_list.item(_i).isSelected()]
            _allL3 = self._lot_source.get_lots()
            _sL3   = [_allL3[_j] for _j in _sel3 if _j < len(_allL3)]
            if _sL3:
                start_stack = sum(l["btc"] for l in _sL3)
                self._leo_lbl.setText(f"LEO: {start_stack:.8g} BTC from {len(_sL3)} lot(s)")
            else:
                self._leo_lbl.setText("(no lots selected)")
        else:
            self._leo_lbl.setText("")
        amount      = self._amount_sp.value()
        disp_mode   = self._disp_cb.currentData()
        font_title       = self._font_title.family()
        font_axis_t      = self._font_axis_t.family()
        font_ticks       = self._font_ticks.family()
        font_ticks_minor = self._font_ticks_minor.family()
        font_legend      = self._font_legend.family()
        font_title_sz    = self._font_title.size()
        font_axis_sz     = self._font_axis_t.size()
        font_ticks_sz       = self._font_ticks.size()
        font_ticks_minor_sz = self._font_ticks_minor.size()
        font_legend_sz      = self._font_legend.size()

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(bg_color)
        self.fig.patch.set_facecolor(bg_color)
        for sp in ax.spines.values(): sp.set_edgecolor(m.SPINE_COLOR)
        ax.tick_params(colors=m.TEXT_COLOR)
        ax.grid(True, which="major", color=m.GRID_MAJOR_COLOR, lw=0.6, alpha=0.8)

        vis_qs = [s for s in self._q_state if s["vis"]]
        for qs in vis_qs:
            q = qs["q"]
            stack = start_stack
            vals  = np.empty(len(ts))
            for i, t in enumerate(ts):
                t_safe = max(t, 0.5)
                price  = float(qr_price(q, t_safe, m.qr_fits))
                stack += amount / price
                vals[i] = stack
            if disp_mode == "usd":
                prices = np.array([float(qr_price(q, max(t, 0.5), m.qr_fits)) for t in ts])
                y_vals = vals * prices
                final_lbl = fmt_price(float(y_vals[-1]))
            else:
                y_vals = vals
                final_lbl = f"{float(vals[-1]):.4f} BTC"
            pct = q * 100
            line_lbl = (f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%") + f"  →  {final_lbl}"
            ax.plot(ts, y_vals, color=qs["color"], ls=qs["ls"], lw=qs["lw"], label=line_lbl)

        if self._today_chk.isChecked():
            td = today_t(m.genesis)
            if t_start <= td <= t_end:
                ax.axvline(td, color="#FF6600", ls="--", lw=1.5, alpha=0.85, label="Today")

        if self._log_y_chk.isChecked():
            ax.set_yscale("log")

        # X axis year ticks
        span = end_yr - start_yr
        step = 1 if span <= 15 else (2 if span <= 30 else 5)
        tick_yrs = [y for y in range(start_yr, end_yr + 1, step)]
        tick_ts  = [yr_to_t(y, m.genesis) for y in tick_yrs]
        valid = [(t, y) for t, y in zip(tick_ts, tick_yrs) if t_start <= t <= t_end]
        if valid:
            vts, vys = zip(*valid)
            ax.set_xticks(list(vts))
            ax.set_xticklabels([str(y) for y in vys], rotation=45, ha="right",
                               fontfamily=font_ticks, fontsize=font_ticks_sz)
        ax.xaxis.set_minor_locator(NullLocator())

        # Y-axis formatting (Tab-1 style decade ticks)
        if disp_mode == "usd":
            y_lo2, y_hi2 = ax.get_ylim()
            _usd_decades = [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
            maj2 = [p for p in _usd_decades if y_lo2 * 0.9 <= p <= y_hi2 * 1.1]
            if maj2:
                ax.yaxis.set_major_locator(FixedLocator(maj2))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_yticklabels([fmt_price(p) for p in maj2], fontfamily=font_ticks,
                                  fontsize=font_ticks_sz)
        else:
            y_lo2, y_hi2 = ax.get_ylim()
            _btc_ticks = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 50, 100, 500, 1000,
                          5000, 10000, 100000]
            majb = [p for p in _btc_ticks if y_lo2 * 0.9 <= p <= y_hi2 * 1.1]
            if majb:
                ax.yaxis.set_major_locator(FixedLocator(majb))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_yticklabels([_fmt_btc(p) for p in majb], fontfamily=font_ticks,
                                  fontsize=font_ticks_sz)

        # Minor log ticks
        if self._log_y_chk.isChecked() and self._minor_ticks_chk.isChecked():
            import math as _mt2
            _yl2, _yh2 = ax.get_ylim()
            _lo_e2 = int(_mt2.floor(_mt2.log10(max(_yl2, 1e-100))))
            _hi_e2 = int(_mt2.ceil(_mt2.log10(max(_yh2, 1e-100))))
            _mj_d2 = [10.0**e for e in range(_lo_e2, _hi_e2 + 1)]
            _minors_d = [float(_p * _k)
                         for _p in _mj_d2 for _k in range(2, 10)
                         if _yl2 <= _p * _k <= _yh2]
            if _minors_d:
                ax.yaxis.set_minor_locator(FixedLocator(_minors_d))
                ax.tick_params(axis='y', which='minor', length=3,
                               color=m.SPINE_COLOR,
                               labelsize=font_ticks_minor_sz, labelcolor=m.TEXT_COLOR,
                               labelfontfamily=font_ticks_minor)
                def _mfmt_d(v, pos, _m=_mt2):
                    if v <= 0: return ''
                    exp = int(_m.floor(_m.log10(max(v, 1e-100))))
                    mult = round(v / 10.0**exp)
                    return fmt_price(v) if mult % 2 == 0 else ''
                ax.yaxis.set_minor_formatter(FuncFormatter(_mfmt_d))
                ax.grid(True, which='minor',
                        color=m.GRID_MINOR_COLOR,
                        linewidth=0.4, linestyle=':')
        else:
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(False, which='minor')

        # Dual Y-axis (secondary axis showing the other unit)
        if self._dual_y_chk.isChecked() and vis_qs:
            _ref_q = vis_qs[0]["q"]
            _ref_price = float(qr_price(_ref_q, max(t_start, 0.5), m.qr_fits))
            if _ref_price > 0:
                ax2 = ax.twinx()
                ax2.set_facecolor("none")
                ax2.tick_params(colors=m.TEXT_COLOR, labelsize=7)
                for sp2 in ax2.spines.values():
                    sp2.set_edgecolor(m.SPINE_COLOR)
                y_lo2, y_hi2 = ax.get_ylim()
                if disp_mode == "btc":
                    ax2.set_ylim(y_lo2 * _ref_price, y_hi2 * _ref_price)
                    _usd_dec2 = [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
                    _maj_usd = [p for p in _usd_dec2 if y_lo2 * _ref_price * 0.9 <= p <= y_hi2 * _ref_price * 1.1]
                    if _maj_usd:
                        ax2.yaxis.set_major_locator(FixedLocator([p / _ref_price for p in _maj_usd]))
                        ax2.set_yticklabels([fmt_price(p) for p in _maj_usd], fontfamily=font_ticks, fontsize=font_ticks_sz)
                    pct2 = _ref_q * 100
                    ax2.set_ylabel(f"USD @ {start_yr} {f'Q{pct2:.4g}%' if pct2>=1 else f'Q{pct2:.3g}%'}",
                                   color=m.TEXT_COLOR, fontfamily=font_axis_t, fontsize=font_axis_sz)
                else:
                    ax2.set_ylim(y_lo2 / _ref_price, y_hi2 / _ref_price)
                    _btc_dec2 = [0.0001,0.001,0.01,0.1,0.25,0.5,1,2,5,10,50,100,500,1000]
                    _maj_btc = [p for p in _btc_dec2 if (y_lo2/_ref_price)*0.9 <= p <= (y_hi2/_ref_price)*1.1]
                    if _maj_btc:
                        ax2.yaxis.set_major_locator(FixedLocator([p * _ref_price for p in _maj_btc]))
                        ax2.set_yticklabels([_fmt_btc(p) for p in _maj_btc], fontfamily=font_ticks, fontsize=font_ticks_sz)
                    pct2 = _ref_q * 100
                    ax2.set_ylabel(f"BTC @ {start_yr} {f'Q{pct2:.4g}%' if pct2>=1 else f'Q{pct2:.3g}%'}",
                                   color=m.TEXT_COLOR, fontfamily=font_axis_t, fontsize=font_axis_sz)

        amount_str = fmt_price(amount)
        stk_str = f"  (stack: {start_stack:.8g} BTC)" if start_stack > 0 else ""
        ax.set_xlabel("Year", color=m.TEXT_COLOR, fontfamily=font_axis_t,
                      fontsize=font_axis_sz)
        ax.set_ylabel("BTC Balance" if disp_mode == "btc" else "Stack Value (USD)",
                      color=m.TEXT_COLOR, fontfamily=font_axis_t, fontsize=font_axis_sz)
        ax.set_title(f"Bitcoin DCA — {amount_str}/{freq_str} from {start_yr} to {end_yr}{stk_str}",
                     color=m.TITLE_COLOR, fontsize=font_title_sz, fontfamily=font_title)
        if vis_qs:
            ax.legend(framealpha=0.9, edgecolor=m.GRID_MAJOR_COLOR,
                      loc="upper left", prop={"family": font_legend, "size": font_legend_sz})
        self.canvas.draw_idle()

    # ── factory reset / settings ───────────────────────────────────────────────
    def _factory_q_state(self):
        m = self.m; out = []
        for i, q in enumerate(q for q in _DEFAULT_QS if q in m.qr_fits):
            pct = q * 100
            lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
            out.append({"q": q, "lbl": lbl, "vis": True,
                        "color": m.qr_colors.get(q, _CB_COLORS[i % len(_CB_COLORS)]),
                        "ls": m.qr_linestyles.get(q, _CB_STYLES[i % len(_CB_STYLES)]),
                        "lw": 2.0 if abs(q - 0.5) < 1e-6 else 1.5})
        return out

    def _factory_reset(self):
        self._busy = True
        self._stack_sp.setValue(0.0)
        self._amount_sp.setValue(500.0)
        self._freq_cb.setCurrentIndex(0)
        self._syr_sl.setValue(2020); self._syr_lbl.setText("2020")
        self._eyr_sl.setValue(2040); self._eyr_lbl.setText("2040")
        self._disp_cb.setCurrentIndex(0)
        self._log_y_chk.setChecked(False)
        self._today_chk.setChecked(True)
        self._dual_y_chk.setChecked(False)
        self._minor_ticks_chk.setChecked(False)
        self._q_state = self._factory_q_state()
        self._rebuild_q_rows()
        for fp in (self._font_title, self._font_axis_t, self._font_ticks, self._font_legend,
                   self._font_all_b):
            fp.set_family("sans-serif")
        self._busy = False
        self.redraw()

    def showEvent(self, event):
        super().showEvent(event)
        sizes = getattr(self, '_pending_splitter', None)
        if sizes is not None:
            self._splitter.setSizes(sizes)
            self._pending_splitter = None

    def _collect_settings(self):
        def _ls_str(ls): return repr(ls) if isinstance(ls, tuple) else str(ls)
        return {
            "stack":      self._stack_sp.value(),
            "amount":     self._amount_sp.value(),
            "freq_idx":   self._freq_cb.currentIndex(),
            "start_yr":   self._syr_sl.value(),
            "end_yr":     self._eyr_sl.value(),
            "disp_idx":   self._disp_cb.currentIndex(),
            "log_y":      self._log_y_chk.isChecked(),
            "show_today": self._today_chk.isChecked(),
            "dual_y":       self._dual_y_chk.isChecked(),
            "minor_ticks":  self._minor_ticks_chk.isChecked(),
            "q_state": [{"q": s["q"], "vis": s["vis"], "color": s["color"],
                         "ls": _ls_str(s["ls"]), "lw": s["lw"]} for s in self._q_state],
            "font_title":  self._font_title.family(),
            "font_axis_t": self._font_axis_t.family(),
            "font_ticks":        self._font_ticks.family(),
            "font_ticks_minor":  self._font_ticks_minor.family(),
            "font_legend":       self._font_legend.family(),
            "font_title_sz":       self._font_title.size(),
            "font_axis_t_sz":      self._font_axis_t.size(),
            "font_ticks_sz":       self._font_ticks.size(),
            "font_ticks_minor_sz": self._font_ticks_minor.size(),
            "font_legend_sz":      self._font_legend.size(),
            "bg_color":          self._bg_color_btn.color(),
            "splitter_sizes": list(self._splitter.sizes()),
        }

    def _apply_settings(self, d):
        self._busy = True
        try:
            self._stack_sp.setValue(float(d.get("stack", 0.0)))
            self._amount_sp.setValue(float(d.get("amount", 500.0)))
            self._freq_cb.setCurrentIndex(int(d.get("freq_idx", 0)))
            self._syr_sl.setValue(int(d.get("start_yr", 2020)))
            self._syr_lbl.setText(str(self._syr_sl.value()))
            self._eyr_sl.setValue(int(d.get("end_yr", 2040)))
            self._eyr_lbl.setText(str(self._eyr_sl.value()))
            self._disp_cb.setCurrentIndex(int(d.get("disp_idx", 0)))
            self._log_y_chk.setChecked(bool(d.get("log_y", False)))
            self._today_chk.setChecked(bool(d.get("show_today", True)))
            self._dual_y_chk.setChecked(bool(d.get("dual_y", False)))
            self._minor_ticks_chk.setChecked(bool(d.get("minor_ticks", False)))
            raw_qs = d.get("q_state")
            if raw_qs:
                new_qs = []
                for i, qs in enumerate(raw_qs):
                    q = float(qs["q"])
                    if q not in self.m.qr_fits:
                        try: self.m.qr_fits[q] = _fit_one_qr(self.m, q)
                        except Exception: continue
                    pct = q * 100; lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
                    new_qs.append({"q": q, "lbl": lbl, "vis": bool(qs.get("vis", True)),
                                   "color": qs.get("color", _CB_COLORS[i % len(_CB_COLORS)]),
                                   "ls": _parse_ls(qs.get("ls", "-")), "lw": float(qs.get("lw", 1.5))})
                if new_qs: self._q_state = new_qs
            self._rebuild_q_rows()
            self._font_title.set_family(d.get("font_title",  "sans-serif"))
            self._font_axis_t.set_family(d.get("font_axis_t", "sans-serif"))
            self._font_ticks.set_family(d.get("font_ticks",  "sans-serif"))
            self._font_ticks_minor.set_family(d.get("font_ticks_minor", "sans-serif"))
            self._font_legend.set_family(d.get("font_legend", "sans-serif"))
            self._font_title.set_size(d.get("font_title_sz", 11))
            self._font_axis_t.set_size(d.get("font_axis_t_sz", 10))
            self._font_ticks.set_size(d.get("font_ticks_sz", 10))
            self._font_ticks_minor.set_size(d.get("font_ticks_minor_sz", 6))
            self._font_legend.set_size(d.get("font_legend_sz", 7))
            self._bg_color_btn.set_color(d.get("bg_color", self.m.PLOT_BG_COLOR))
            if "splitter_sizes" in d:
                self._pending_splitter = [int(x) for x in d["splitter_sizes"]]
        finally:
            self._busy = False
        self.redraw()

    def _save(self, fmt):
        fn   = self._fn_edit.text().strip() or "btc_dca"
        path = _desktop_save_path(self, fn, fmt)
        if not path: return
        try:
            self.fig.savefig(path, format=fmt, bbox_inches="tight", dpi=self._dpi_sp.value())
            self._save_lbl.setText(f"✓ Saved {Path(path).name}")
        except Exception as e:
            self._save_lbl.setText(f"Error: {e}")


# ── RetireTab ──────────────────────────────────────────────────────────────────

class RetireTab(QWidget):
    """Tab 4 — Bitcoin Retireator: simulate periodic USD withdrawals from BTC stack."""
    q_state_changed = pyqtSignal(list)
    all_fonts_applied = pyqtSignal(str, int)

    def __init__(self, model: ModelData):
        super().__init__()
        self.m = model
        self._busy = False
        self._q_state = []
        for i, q in enumerate(model.QR_QUANTILES):
            pct = q * 100
            lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
            self._q_state.append({
                "q": q, "lbl": lbl, "vis": True,
                "color": model.qr_colors.get(q, _CB_COLORS[i % len(_CB_COLORS)]),
                "ls":    model.qr_linestyles.get(q, _CB_STYLES[i % len(_CB_STYLES)]),
                "lw":    2.0 if abs(q - 0.5) < 1e-6 else 1.5,
            })
        self._lot_source = None
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        plot_w = QWidget()
        plot_l = QVBoxLayout(plot_w)
        plot_l.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(10, 6.5), facecolor=self.m.PLOT_BG_COLOR)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavToolbar(self.canvas, plot_w)
        plot_l.addWidget(toolbar)
        plot_l.addWidget(self.canvas)

        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(295)
        ctrl_w = QWidget()
        ctrl_l = QVBoxLayout(ctrl_w)
        ctrl_l.setSpacing(6)

        fac_btn = QPushButton("Reset to Factory Defaults")
        fac_btn.clicked.connect(self._factory_reset)
        ctrl_l.addWidget(fac_btn)

        # Bitcoin Stack
        stk_grp  = QGroupBox("Bitcoin Stack")
        stk_form = QFormLayout(stk_grp)
        stk_form.setSpacing(4)
        self._stack_sp = QDoubleSpinBox()
        self._stack_sp.setRange(0, 10_000_000); self._stack_sp.setDecimals(8)
        self._stack_sp.setValue(0.0); self._stack_sp.setSingleStep(0.1)
        stk_form.addRow("BTC owned:", self._stack_sp)
        ctrl_l.addWidget(stk_grp)

        # LEO group
        leo_grp4 = QGroupBox("Lot Entry Override (LEO)")
        leo_l4   = QVBoxLayout(leo_grp4)
        leo_l4.setSpacing(4)
        self._use_lots_chk = QCheckBox("Use lots as starting stack")
        self._use_lots_chk.setChecked(False)
        leo_l4.addWidget(self._use_lots_chk)
        self._lot_list = QListWidget()
        self._lot_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._lot_list.setMinimumHeight(60)
        leo_l4.addWidget(self._lot_list)
        _leo_row4 = QHBoxLayout()
        _leo_all4 = QPushButton("All");  _leo_all4.setFixedWidth(38)
        _leo_none4 = QPushButton("None"); _leo_none4.setFixedWidth(44)
        _leo_all4.clicked.connect(lambda: [self._lot_list.item(_i).setSelected(True) for _i in range(self._lot_list.count())])
        _leo_none4.clicked.connect(lambda: [self._lot_list.item(_i).setSelected(False) for _i in range(self._lot_list.count())])
        _leo_row4.addWidget(_leo_all4); _leo_row4.addWidget(_leo_none4); _leo_row4.addStretch()
        leo_l4.addLayout(_leo_row4)
        self._leo_lbl = QLabel("")
        self._leo_lbl.setStyleSheet("font-size:9px;")
        self._leo_lbl.setWordWrap(True)
        leo_l4.addWidget(self._leo_lbl)
        ctrl_l.addWidget(leo_grp4)

        # Withdrawal
        wd_grp  = QGroupBox("Withdrawals")
        wd_form = QFormLayout(wd_grp)
        wd_form.setSpacing(4)
        self._wd_sp = QDoubleSpinBox()
        self._wd_sp.setRange(1, 10_000_000); self._wd_sp.setValue(5000)
        self._wd_sp.setSingleStep(500); self._wd_sp.setDecimals(2)
        self._wd_sp.setPrefix("$")
        wd_form.addRow("Per period:", self._wd_sp)
        self._freq_cb = QComboBox()
        for lbl, val in [("Monthly","Monthly"),("Weekly","Weekly"),
                         ("Quarterly","Quarterly"),("Annually","Annually")]:
            self._freq_cb.addItem(lbl, val)
        wd_form.addRow("Frequency:", self._freq_cb)

        syr_row = QHBoxLayout()
        self._syr_sl = QSlider(Qt.Horizontal)
        self._syr_sl.setRange(2010, 2050); self._syr_sl.setValue(2025)
        self._syr_lbl = QLabel("2025"); self._syr_lbl.setMinimumWidth(36)
        syr_row.addWidget(self._syr_sl); syr_row.addWidget(self._syr_lbl)
        wd_form.addRow("Start Year:", syr_row)

        eyr_row = QHBoxLayout()
        self._eyr_sl = QSlider(Qt.Horizontal)
        self._eyr_sl.setRange(2015, 2075); self._eyr_sl.setValue(2055)
        self._eyr_lbl = QLabel("2055"); self._eyr_lbl.setMinimumWidth(36)
        eyr_row.addWidget(self._eyr_sl); eyr_row.addWidget(self._eyr_lbl)
        wd_form.addRow("End Year:", eyr_row)

        self._infl_sp = QDoubleSpinBox()
        self._infl_sp.setRange(0.0, 50.0); self._infl_sp.setValue(3.0)
        self._infl_sp.setSingleStep(0.5); self._infl_sp.setDecimals(2)
        self._infl_sp.setSuffix(" %")
        wd_form.addRow("USD Inflation:", self._infl_sp)
        ctrl_l.addWidget(wd_grp)

        # Price Scenarios
        scen_grp   = QGroupBox("Price Scenarios")
        scen_outer = QVBoxLayout(scen_grp)
        scen_outer.setSpacing(4)
        self._q_rows_w = QWidget()
        self._q_rows_l = QVBoxLayout(self._q_rows_w)
        self._q_rows_l.setSpacing(2)
        self._q_rows_l.setContentsMargins(0, 0, 0, 0)
        scen_outer.addWidget(self._q_rows_w)
        self._rebuild_q_rows()
        add_row = QHBoxLayout(); add_row.setSpacing(4)
        add_row.addWidget(QLabel("Add Q:"))
        self._add_q_sp = QDoubleSpinBox()
        self._add_q_sp.setRange(0.001, 99.999); self._add_q_sp.setValue(75.0)
        self._add_q_sp.setDecimals(3); self._add_q_sp.setSingleStep(5.0)
        self._add_q_sp.setSuffix(" %")
        add_btn = QPushButton("Add"); add_btn.setFixedWidth(48)
        add_btn.clicked.connect(self._q_add)
        add_row.addWidget(self._add_q_sp); add_row.addWidget(add_btn)
        scen_outer.addLayout(add_row)
        ctrl_l.addWidget(scen_grp)

        # Display
        disp_grp  = QGroupBox("Display")
        disp_form = QFormLayout(disp_grp)
        disp_form.setSpacing(4)
        self._disp_cb = QComboBox()
        for lbl, val in [("BTC Remaining","btc"),("USD Value","usd")]:
            self._disp_cb.addItem(lbl, val)
        disp_form.addRow("Y-axis:", self._disp_cb)
        self._log_y_chk = QCheckBox("Log Y scale"); self._log_y_chk.setChecked(False)
        self._today_chk = QCheckBox("Show today"); self._today_chk.setChecked(True)
        self._annot_chk = QCheckBox("Annotate depletion"); self._annot_chk.setChecked(True)
        self._dual_y_chk = QCheckBox("Dual Y-axis"); self._dual_y_chk.setChecked(False)
        self._minor_ticks_chk = QCheckBox("Minor log ticks (×0.1 dec)")
        self._minor_ticks_chk.setChecked(False)
        disp_form.addRow(self._log_y_chk)
        disp_form.addRow(self._today_chk)
        disp_form.addRow(self._annot_chk)
        disp_form.addRow(self._dual_y_chk)
        disp_form.addRow(self._minor_ticks_chk)
        self._bg_color_btn = ColorBtn(self.m.PLOT_BG_COLOR)
        self._bg_color_btn.setToolTip("Plot background colour")
        disp_form.addRow("Plot bg:", self._bg_color_btn)
        ctrl_l.addWidget(disp_grp)

        # Fonts
        font_grp  = QGroupBox("Fonts")
        font_form = QFormLayout(font_grp)
        font_form.setSpacing(4)
        all_row = QHBoxLayout()
        self._font_all_b = FontPicker("sans-serif")
        all_btn = QPushButton("Apply to all"); all_btn.setFixedWidth(90)
        all_btn.clicked.connect(self._apply_font_to_all)
        all_tabs_btn = QPushButton("All tabs"); all_tabs_btn.setFixedWidth(65)
        all_tabs_btn.clicked.connect(
            lambda: self.all_fonts_applied.emit(
                self._font_all_b.family(), self._font_all_b.size()))
        all_row.addWidget(self._font_all_b); all_row.addWidget(all_btn)
        all_row.addWidget(all_tabs_btn)
        font_form.addRow("All fields:", all_row)
        self._font_title       = FontPicker("sans-serif")
        self._font_axis_t      = FontPicker("sans-serif")
        self._font_ticks       = FontPicker("sans-serif")
        self._font_ticks_minor = FontPicker("sans-serif")
        self._font_legend      = FontPicker("sans-serif")
        font_form.addRow("Chart title:", self._font_title)
        font_form.addRow("Axis titles:", self._font_axis_t)
        font_form.addRow("Major ticks:", self._font_ticks)
        font_form.addRow("Minor ticks:", self._font_ticks_minor)
        font_form.addRow("Legend:",      self._font_legend)
        for fp in (self._font_title, self._font_axis_t, self._font_ticks,
                   self._font_ticks_minor, self._font_legend):
            fp.font_changed.connect(self.redraw)
            fp.size_changed.connect(self.redraw)
        self._bg_color_btn.color_changed.connect(self.redraw)
        ctrl_l.addWidget(font_grp)

        # Save
        save_grp  = QGroupBox("Save")
        save_form = QFormLayout(save_grp)
        save_form.setSpacing(4)
        self._fn_edit = QLineEdit("btc_retire")
        self._dpi_sp  = QSpinBox(); self._dpi_sp.setRange(72, 600); self._dpi_sp.setValue(150)
        save_form.addRow("Filename:", self._fn_edit)
        save_form.addRow("DPI:", self._dpi_sp)
        btn_row = QHBoxLayout()
        svg_btn = QPushButton("Save SVG"); svg_btn.clicked.connect(lambda: self._save("svg"))
        jpg_btn = QPushButton("Save JPG"); jpg_btn.clicked.connect(lambda: self._save("jpg"))
        btn_row.addWidget(svg_btn); btn_row.addWidget(jpg_btn)
        save_form.addRow(btn_row)
        self._save_lbl = QLabel("")
        save_form.addRow(self._save_lbl)
        ctrl_l.addWidget(save_grp)

        ctrl_l.addStretch()
        ctrl_scroll.setWidget(ctrl_w)
        self._splitter = splitter
        self._splitter.addWidget(plot_w); self._splitter.addWidget(ctrl_scroll)
        self._splitter.setStretchFactor(0, 1); self._splitter.setStretchFactor(1, 0)
        main_l = QHBoxLayout(self)
        main_l.setContentsMargins(0, 0, 0, 0)
        main_l.addWidget(self._splitter)

        # Signals
        self._syr_sl.valueChanged.connect(lambda v: (self._syr_lbl.setText(str(v)), self.redraw()))
        self._eyr_sl.valueChanged.connect(lambda v: (self._eyr_lbl.setText(str(v)), self.redraw()))
        self._wd_sp.valueChanged.connect(self.redraw)
        self._freq_cb.currentIndexChanged.connect(self.redraw)
        self._infl_sp.valueChanged.connect(self.redraw)
        self._disp_cb.currentIndexChanged.connect(self.redraw)
        self._log_y_chk.toggled.connect(self.redraw)
        self._today_chk.toggled.connect(self.redraw)
        self._annot_chk.toggled.connect(self.redraw)
        self._dual_y_chk.toggled.connect(self.redraw)
        self._minor_ticks_chk.toggled.connect(self.redraw)
        self._use_lots_chk.toggled.connect(self.redraw)
        self._lot_list.itemSelectionChanged.connect(self.redraw)
        self._stack_sp.valueChanged.connect(self.redraw)

    # ── quantile rows (identical pattern to DCATab) ───────────────────────────
    def _rebuild_q_rows(self):
        while self._q_rows_l.count():
            w = self._q_rows_l.takeAt(0).widget()
            if w: w.deleteLater()
        for i, qs in enumerate(self._q_state):
            row_w = QWidget(); row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0); row_l.setSpacing(3)
            cb = QCheckBox(qs["lbl"]); cb.setChecked(qs["vis"]); cb.setMinimumWidth(85)
            cb.toggled.connect(lambda v, idx=i: self._q_vis(idx, v))
            clr_btn = ColorBtn(qs["color"])
            clr_btn.color_changed.connect(lambda c, idx=i: self._q_color(idx, c))
            ls_cb = QComboBox(); ls_cb.addItems(LS_NAMES); ls_cb.setCurrentIndex(_ls_index(qs["ls"]))
            ls_cb.setFixedWidth(105)
            ls_cb.currentIndexChanged.connect(lambda v, idx=i: self._q_ls(idx, v))
            lw_sp = QDoubleSpinBox(); lw_sp.setRange(0.3, 6.0); lw_sp.setValue(qs["lw"])
            lw_sp.setSingleStep(0.25); lw_sp.setFixedWidth(56)
            lw_sp.valueChanged.connect(lambda v, idx=i: self._q_lw(idx, v))
            rm_btn = QPushButton("×"); rm_btn.setFixedSize(22, 22)
            rm_btn.clicked.connect(lambda _, idx=i: self._q_remove(idx))
            row_l.addWidget(cb); row_l.addWidget(clr_btn)
            row_l.addWidget(ls_cb); row_l.addWidget(lw_sp); row_l.addWidget(rm_btn)
            self._q_rows_l.addWidget(row_w)

    def _emit_q_changed(self):
        self.q_state_changed.emit(list(self._q_state))

    def _q_vis(self, idx, v):   self._q_state[idx]["vis"] = v;   self._emit_q_changed(); self.redraw()
    def _q_color(self, idx, c): self._q_state[idx]["color"] = c; self._emit_q_changed(); self.redraw()
    def _q_ls(self, idx, v):    self._q_state[idx]["ls"] = LS_SPECS[v]; self._emit_q_changed(); self.redraw()
    def _q_lw(self, idx, v):    self._q_state[idx]["lw"] = v;   self._emit_q_changed(); self.redraw()

    def _q_remove(self, idx):
        self._q_state.pop(idx); self._rebuild_q_rows(); self._emit_q_changed(); self.redraw()

    def _q_add(self):
        q = self._add_q_sp.value() / 100.0
        for s in self._q_state:
            if abs(s["q"] - q) < 1e-6: return
        if q not in self.m.qr_fits:
            try: self.m.qr_fits[q] = _fit_one_qr(self.m, q)
            except Exception as e:
                QMessageBox.warning(self, "Fit error", f"Could not fit Q{q*100:.4g}%:\n{e}")
                return
        n = len(self._q_state); pct = q * 100
        lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
        self._q_state.append({"q": q, "lbl": lbl, "vis": True,
                              "color": _CB_COLORS[n % len(_CB_COLORS)],
                              "ls": _CB_STYLES[n % len(_CB_STYLES)], "lw": 1.5})
        self._q_state.sort(key=lambda s: s["q"])
        self._rebuild_q_rows(); self._emit_q_changed(); self.redraw()

    def set_lot_source(self, tracker_tab):
        self._lot_source = tracker_tab

    def refresh_lot_list(self):
        self._lot_list.clear()
        if self._lot_source is None:
            return
        for i, lot in enumerate(self._lot_source.get_lots()):
            pct_str = f"Q{lot['pct_q']*100:.2f}%"
            lbl = f"{lot['date']}  {lot['btc']:.4f} BTC @ {fmt_price(lot['price'])}  ({pct_str})"
            item = QListWidgetItem(lbl)
            item.setData(Qt.UserRole, i)
            self._lot_list.addItem(item)
            item.setSelected(True)

    def _apply_font_to_all(self):
        fam = self._font_all_b.family()
        sz  = self._font_all_b.size()
        for fp in (self._font_title, self._font_axis_t, self._font_ticks,
                   self._font_ticks_minor, self._font_legend):
            fp.set_family(fam)
            fp.set_size(sz)
        self.redraw()

    # ── redraw ────────────────────────────────────────────────────────────────
    def redraw(self, *_):
        if self._busy: return
        m = self.m
        bg_color = self._bg_color_btn.color()
        FREQ_PPY = {"Weekly": 52, "Monthly": 12, "Quarterly": 4, "Annually": 1}
        freq_str = self._freq_cb.currentData()
        ppy = FREQ_PPY[freq_str]
        dt  = 1.0 / ppy
        start_yr   = self._syr_sl.value()
        end_yr     = self._eyr_sl.value()
        if end_yr <= start_yr: return
        t_start = max(yr_to_t(start_yr, m.genesis), 1.0)
        t_end   = yr_to_t(end_yr, m.genesis)
        ts      = np.arange(t_start, t_end + dt * 0.5, dt)
        if len(ts) == 0: return

        start_stack  = self._stack_sp.value()
        # LEO override: use selected lots total as starting stack
        if (self._use_lots_chk.isChecked() and self._lot_source is not None
                and self._lot_list.count() > 0):
            _sel4 = [self._lot_list.item(_i).data(Qt.UserRole)
                     for _i in range(self._lot_list.count())
                     if self._lot_list.item(_i).isSelected()]
            _allL4 = self._lot_source.get_lots()
            _sL4   = [_allL4[_j] for _j in _sel4 if _j < len(_allL4)]
            if _sL4:
                start_stack = sum(l["btc"] for l in _sL4)
                self._leo_lbl.setText(f"LEO: {start_stack:.8g} BTC from {len(_sL4)} lot(s)")
            else:
                self._leo_lbl.setText("(no lots selected)")
        else:
            self._leo_lbl.setText("")
        base_wd      = self._wd_sp.value()
        infl_rate    = self._infl_sp.value() / 100.0   # inflation ONLY affects withdrawal $
        disp_mode    = self._disp_cb.currentData()
        font_title       = self._font_title.family()
        font_axis_t      = self._font_axis_t.family()
        font_ticks       = self._font_ticks.family()
        font_ticks_minor = self._font_ticks_minor.family()
        font_legend      = self._font_legend.family()
        font_title_sz    = self._font_title.size()
        font_axis_sz     = self._font_axis_t.size()
        font_ticks_sz       = self._font_ticks.size()
        font_ticks_minor_sz = self._font_ticks_minor.size()
        font_legend_sz      = self._font_legend.size()

        # Inflation-adjusted withdrawal amounts (only $ changes, not BTC price)
        years_elapsed = (ts - ts[0])
        wd_amounts    = base_wd * (1.0 + infl_rate) ** years_elapsed

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(bg_color)
        self.fig.patch.set_facecolor(bg_color)
        for sp in ax.spines.values(): sp.set_edgecolor(m.SPINE_COLOR)
        ax.tick_params(colors=m.TEXT_COLOR)
        ax.grid(True, which="major", color=m.GRID_MAJOR_COLOR, lw=0.6, alpha=0.8)

        vis_qs = [s for s in self._q_state if s["vis"]]
        for qs in vis_qs:
            q = qs["q"]
            stack = start_stack
            vals  = np.empty(len(ts))
            for i, t in enumerate(ts):
                if stack <= 0.0:
                    vals[i:] = 0.0
                    break
                t_safe = max(t, 0.5)
                price  = float(qr_price(q, t_safe, m.qr_fits))
                btc_sold = wd_amounts[i] / price
                stack = max(0.0, stack - btc_sold)
                vals[i] = stack
            else:
                vals[-1] = max(0.0, vals[-1])

            if disp_mode == "usd":
                prices = np.array([float(qr_price(q, max(t, 0.5), m.qr_fits)) for t in ts])
                y_vals = vals * prices
            else:
                y_vals = vals

            pct = q * 100
            line_lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"

            # Find depletion point
            deplete_idx = None
            zero_idx = np.where(vals <= 0)[0]
            if len(zero_idx) > 0:
                deplete_idx = zero_idx[0]

            ax.plot(ts, y_vals, color=qs["color"], ls=qs["ls"], lw=qs["lw"], label=line_lbl)

            if self._annot_chk.isChecked() and deplete_idx is not None:
                dep_t = ts[deplete_idx]
                dep_yr = start_yr + (dep_t - t_start)
                ax.axvline(dep_t, color=qs["color"], ls=":", lw=1.0, alpha=0.7)
                ax.annotate(
                    f"{dep_yr:.1f}",
                    xy=(dep_t, 0), xytext=(dep_t, float(ax.get_ylim()[1]) * 0.05 if ax.get_ylim()[1] > 0 else 1),
                    color=qs["color"], fontsize=7, ha="center",
                    arrowprops=dict(arrowstyle="->", color=qs["color"], lw=0.8))

        if self._today_chk.isChecked():
            td = today_t(m.genesis)
            if t_start <= td <= t_end:
                ax.axvline(td, color="#FF6600", ls="--", lw=1.5, alpha=0.85, label="Today")

        if self._log_y_chk.isChecked():
            try: ax.set_yscale("log")
            except Exception: pass

        # X axis year ticks
        span = end_yr - start_yr
        step = 1 if span <= 15 else (2 if span <= 30 else 5)
        tick_yrs = [y for y in range(start_yr, end_yr + 1, step)]
        tick_ts  = [yr_to_t(y, m.genesis) for y in tick_yrs]
        valid = [(t, y) for t, y in zip(tick_ts, tick_yrs) if t_start <= t <= t_end]
        if valid:
            vts, vys = zip(*valid)
            ax.set_xticks(list(vts))
            ax.set_xticklabels([str(y) for y in vys], rotation=45, ha="right",
                               fontfamily=font_ticks, fontsize=font_ticks_sz)
        ax.xaxis.set_minor_locator(NullLocator())

        # Y-axis formatting (Tab-1 style decade ticks)
        if disp_mode == "usd":
            _y_lo_r, _y_hi_r = ax.get_ylim()
            _usd_dec_r = [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
            _maj_r = [p for p in _usd_dec_r if _y_lo_r * 0.9 <= p <= _y_hi_r * 1.1]
            if _maj_r:
                ax.yaxis.set_major_locator(FixedLocator(_maj_r))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_yticklabels([fmt_price(p) for p in _maj_r], fontfamily=font_ticks,
                                  fontsize=font_ticks_sz)
        else:
            _y_lo_r, _y_hi_r = ax.get_ylim()
            _btc_ticks_r = [0.0001,0.001,0.01,0.1,0.25,0.5,1,2,5,10,50,100,500,1000,5000,10000]
            _majb_r = [p for p in _btc_ticks_r if _y_lo_r * 0.9 <= p <= _y_hi_r * 1.1]
            if _majb_r:
                ax.yaxis.set_major_locator(FixedLocator(_majb_r))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_yticklabels([_fmt_btc(p) for p in _majb_r], fontfamily=font_ticks,
                                  fontsize=font_ticks_sz)

        # Minor log ticks
        if self._log_y_chk.isChecked() and self._minor_ticks_chk.isChecked():
            import math as _mt3
            _yl3, _yh3 = ax.get_ylim()
            _lo_e3 = int(_mt3.floor(_mt3.log10(max(_yl3, 1e-100))))
            _hi_e3 = int(_mt3.ceil(_mt3.log10(max(_yh3, 1e-100))))
            _mj_d3 = [10.0**e for e in range(_lo_e3, _hi_e3 + 1)]
            _minors_r = [float(_p * _k)
                         for _p in _mj_d3 for _k in range(2, 10)
                         if _yl3 <= _p * _k <= _yh3]
            if _minors_r:
                ax.yaxis.set_minor_locator(FixedLocator(_minors_r))
                ax.tick_params(axis='y', which='minor', length=3,
                               color=m.SPINE_COLOR,
                               labelsize=font_ticks_minor_sz, labelcolor=m.TEXT_COLOR,
                               labelfontfamily=font_ticks_minor)
                def _mfmt_r(v, pos, _m=_mt3):
                    if v <= 0: return ''
                    exp = int(_m.floor(_m.log10(max(v, 1e-100))))
                    mult = round(v / 10.0**exp)
                    return fmt_price(v) if mult % 2 == 0 else ''
                ax.yaxis.set_minor_formatter(FuncFormatter(_mfmt_r))
                ax.grid(True, which='minor',
                        color=m.GRID_MINOR_COLOR,
                        linewidth=0.4, linestyle=':')
        else:
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(False, which='minor')

        # Dual Y-axis
        if self._dual_y_chk.isChecked() and vis_qs:
            _ref_q_r = vis_qs[0]["q"]
            _ref_price_r = float(qr_price(_ref_q_r, max(t_start, 0.5), m.qr_fits))
            if _ref_price_r > 0:
                ax2r = ax.twinx()
                ax2r.set_facecolor("none")
                ax2r.tick_params(colors=m.TEXT_COLOR, labelsize=7)
                for sp2 in ax2r.spines.values():
                    sp2.set_edgecolor(m.SPINE_COLOR)
                _yl_r, _yh_r = ax.get_ylim()
                if disp_mode == "btc":
                    ax2r.set_ylim(_yl_r * _ref_price_r, _yh_r * _ref_price_r)
                    _usd_d2 = [0.01,0.1,1,10,100,1e3,1e4,1e5,1e6,1e7,1e8,1e9]
                    _m_u = [p for p in _usd_d2 if _yl_r*_ref_price_r*0.9 <= p <= _yh_r*_ref_price_r*1.1]
                    if _m_u:
                        ax2r.yaxis.set_major_locator(FixedLocator([p/_ref_price_r for p in _m_u]))
                        ax2r.set_yticklabels([fmt_price(p) for p in _m_u], fontfamily=font_ticks, fontsize=font_ticks_sz)
                    _pct2r = _ref_q_r * 100
                    ax2r.set_ylabel(f"USD @ {start_yr} {f'Q{_pct2r:.4g}%' if _pct2r>=1 else f'Q{_pct2r:.3g}%'}",
                                    color=m.TEXT_COLOR, fontfamily=font_axis_t, fontsize=font_axis_sz)
                else:
                    ax2r.set_ylim(_yl_r/_ref_price_r, _yh_r/_ref_price_r)
                    _btc_d2 = [0.0001,0.001,0.01,0.1,0.25,0.5,1,2,5,10,50,100,500,1000]
                    _m_b = [p for p in _btc_d2 if (_yl_r/_ref_price_r)*0.9 <= p <= (_yh_r/_ref_price_r)*1.1]
                    if _m_b:
                        ax2r.yaxis.set_major_locator(FixedLocator([p*_ref_price_r for p in _m_b]))
                        ax2r.set_yticklabels([_fmt_btc(p) for p in _m_b], fontfamily=font_ticks, fontsize=font_ticks_sz)
                    _pct2r = _ref_q_r * 100
                    ax2r.set_ylabel(f"BTC @ {start_yr} {f'Q{_pct2r:.4g}%' if _pct2r>=1 else f'Q{_pct2r:.3g}%'}",
                                    color=m.TEXT_COLOR, fontfamily=font_axis_t, fontsize=font_axis_sz)

        infl_str = f"{self._infl_sp.value():.2g}% inflation"
        wd_str   = fmt_price(base_wd)
        stk_str_r = f"  (stack: {start_stack:.8g} BTC)" if start_stack > 0 else ""
        ax.set_xlabel("Year", color=m.TEXT_COLOR, fontfamily=font_axis_t,
                      fontsize=font_axis_sz)
        ax.set_ylabel("BTC Remaining" if disp_mode == "btc" else "Stack Value (USD)",
                      color=m.TEXT_COLOR, fontfamily=font_axis_t, fontsize=font_axis_sz)
        ax.set_title(
            f"Bitcoin Retireator — {wd_str}/{freq_str} from {start_yr}{stk_str_r}  ({infl_str})",
            color=m.TITLE_COLOR, fontsize=font_title_sz, fontfamily=font_title)
        if vis_qs:
            ax.legend(framealpha=0.9, edgecolor=m.GRID_MAJOR_COLOR,
                      loc="upper right", prop={"family": font_legend, "size": font_legend_sz})
        self.canvas.draw_idle()

    # ── factory reset / settings ───────────────────────────────────────────────
    def _factory_q_state(self):
        m = self.m; out = []
        for i, q in enumerate(q for q in _DEFAULT_QS if q in m.qr_fits):
            pct = q * 100; lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
            out.append({"q": q, "lbl": lbl, "vis": True,
                        "color": m.qr_colors.get(q, _CB_COLORS[i % len(_CB_COLORS)]),
                        "ls": m.qr_linestyles.get(q, _CB_STYLES[i % len(_CB_STYLES)]),
                        "lw": 2.0 if abs(q - 0.5) < 1e-6 else 1.5})
        return out

    def _factory_reset(self):
        self._busy = True
        self._stack_sp.setValue(0.0)
        self._wd_sp.setValue(5000.0)
        self._freq_cb.setCurrentIndex(0)
        self._syr_sl.setValue(2025); self._syr_lbl.setText("2025")
        self._eyr_sl.setValue(2055); self._eyr_lbl.setText("2055")
        self._infl_sp.setValue(3.0)
        self._disp_cb.setCurrentIndex(0)
        self._log_y_chk.setChecked(False)
        self._today_chk.setChecked(True)
        self._annot_chk.setChecked(True)
        self._dual_y_chk.setChecked(False)
        self._minor_ticks_chk.setChecked(False)
        self._q_state = self._factory_q_state()
        self._rebuild_q_rows()
        for fp in (self._font_title, self._font_axis_t, self._font_ticks, self._font_legend,
                   self._font_all_b):
            fp.set_family("sans-serif")
        self._busy = False
        self.redraw()

    def showEvent(self, event):
        super().showEvent(event)
        sizes = getattr(self, '_pending_splitter', None)
        if sizes is not None:
            self._splitter.setSizes(sizes)
            self._pending_splitter = None

    def _collect_settings(self):
        def _ls_str(ls): return repr(ls) if isinstance(ls, tuple) else str(ls)
        return {
            "stack":      self._stack_sp.value(),
            "wd_amount":  self._wd_sp.value(),
            "freq_idx":   self._freq_cb.currentIndex(),
            "start_yr":   self._syr_sl.value(),
            "end_yr":     self._eyr_sl.value(),
            "inflation":  self._infl_sp.value(),
            "disp_idx":   self._disp_cb.currentIndex(),
            "log_y":      self._log_y_chk.isChecked(),
            "show_today": self._today_chk.isChecked(),
            "annotate":   self._annot_chk.isChecked(),
            "dual_y":       self._dual_y_chk.isChecked(),
            "minor_ticks":  self._minor_ticks_chk.isChecked(),
            "q_state": [{"q": s["q"], "vis": s["vis"], "color": s["color"],
                         "ls": _ls_str(s["ls"]), "lw": s["lw"]} for s in self._q_state],
            "font_title":  self._font_title.family(),
            "font_axis_t": self._font_axis_t.family(),
            "font_ticks":        self._font_ticks.family(),
            "font_ticks_minor":  self._font_ticks_minor.family(),
            "font_legend":       self._font_legend.family(),
            "font_title_sz":       self._font_title.size(),
            "font_axis_t_sz":      self._font_axis_t.size(),
            "font_ticks_sz":       self._font_ticks.size(),
            "font_ticks_minor_sz": self._font_ticks_minor.size(),
            "font_legend_sz":      self._font_legend.size(),
            "bg_color":          self._bg_color_btn.color(),
            "splitter_sizes": list(self._splitter.sizes()),
        }

    def _apply_settings(self, d):
        self._busy = True
        try:
            self._stack_sp.setValue(float(d.get("stack", 0.0)))
            self._wd_sp.setValue(float(d.get("wd_amount", 5000.0)))
            self._freq_cb.setCurrentIndex(int(d.get("freq_idx", 0)))
            self._syr_sl.setValue(int(d.get("start_yr", 2025)))
            self._syr_lbl.setText(str(self._syr_sl.value()))
            self._eyr_sl.setValue(int(d.get("end_yr", 2055)))
            self._eyr_lbl.setText(str(self._eyr_sl.value()))
            self._infl_sp.setValue(float(d.get("inflation", 3.0)))
            self._disp_cb.setCurrentIndex(int(d.get("disp_idx", 0)))
            self._log_y_chk.setChecked(bool(d.get("log_y", False)))
            self._today_chk.setChecked(bool(d.get("show_today", True)))
            self._annot_chk.setChecked(bool(d.get("annotate", True)))
            self._dual_y_chk.setChecked(bool(d.get("dual_y", False)))
            self._minor_ticks_chk.setChecked(bool(d.get("minor_ticks", False)))
            raw_qs = d.get("q_state")
            if raw_qs:
                new_qs = []
                for i, qs in enumerate(raw_qs):
                    q = float(qs["q"])
                    if q not in self.m.qr_fits:
                        try: self.m.qr_fits[q] = _fit_one_qr(self.m, q)
                        except Exception: continue
                    pct = q * 100; lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
                    new_qs.append({"q": q, "lbl": lbl, "vis": bool(qs.get("vis", True)),
                                   "color": qs.get("color", _CB_COLORS[i % len(_CB_COLORS)]),
                                   "ls": _parse_ls(qs.get("ls", "-")), "lw": float(qs.get("lw", 1.5))})
                if new_qs: self._q_state = new_qs
            self._rebuild_q_rows()
            self._font_title.set_family(d.get("font_title",  "sans-serif"))
            self._font_axis_t.set_family(d.get("font_axis_t", "sans-serif"))
            self._font_ticks.set_family(d.get("font_ticks",  "sans-serif"))
            self._font_ticks_minor.set_family(d.get("font_ticks_minor", "sans-serif"))
            self._font_legend.set_family(d.get("font_legend", "sans-serif"))
            self._font_title.set_size(d.get("font_title_sz", 11))
            self._font_axis_t.set_size(d.get("font_axis_t_sz", 10))
            self._font_ticks.set_size(d.get("font_ticks_sz", 10))
            self._font_ticks_minor.set_size(d.get("font_ticks_minor_sz", 6))
            self._font_legend.set_size(d.get("font_legend_sz", 7))
            self._bg_color_btn.set_color(d.get("bg_color", self.m.PLOT_BG_COLOR))
            if "splitter_sizes" in d:
                self._pending_splitter = [int(x) for x in d["splitter_sizes"]]
        finally:
            self._busy = False
        self.redraw()

    def _save(self, fmt):
        fn   = self._fn_edit.text().strip() or "btc_retire"
        path = _desktop_save_path(self, fn, fmt)
        if not path: return
        try:
            self.fig.savefig(path, format=fmt, bbox_inches="tight", dpi=self._dpi_sp.value())
            self._save_lbl.setText(f"✓ Saved {Path(path).name}")
        except Exception as e:
            self._save_lbl.setText(f"Error: {e}")

# ── StackTrackerTab ─────────────────────────────────────────────────────────────

class StackTrackerTab(QWidget):
    """Tab 5 — Stack Tracker: record bitcoin lots and compute percentiles."""
    lots_changed = pyqtSignal(float)   # emits total_btc

    def __init__(self, model: ModelData):
        super().__init__()
        self.m = model
        self._lots = []
        self._busy = False
        self._build_ui()
        self._load_lots()

    def _build_ui(self):
        main_l = QHBoxLayout(self)
        main_l.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        main_l.addWidget(splitter)

        # LEFT: table
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(0, 0, 0, 0)
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Date", "BTC", "Price ($/BTC)", "Total Paid ($)", "Percentile", "Notes"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSortingEnabled(True)
        for col, w in enumerate([100, 110, 120, 120, 80]):
            self._table.setColumnWidth(col, w)
        left_l.addWidget(self._table)

        # RIGHT: controls
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(240)
        ctrl_w  = QWidget()
        ctrl_l  = QVBoxLayout(ctrl_w)
        ctrl_l.setSpacing(6)

        # Add Lot group
        add_grp  = QGroupBox("Add Lot")
        add_form = QFormLayout(add_grp)
        add_form.setSpacing(4)
        self._date_edit = QDateEdit()
        self._date_edit.setCalendarPopup(True)
        self._date_edit.setDate(QDate.currentDate())
        self._date_edit.setDisplayFormat("yyyy-MM-dd")
        self._btc_sp = QDoubleSpinBox()
        self._btc_sp.setRange(1e-8, 21_000_000)
        self._btc_sp.setDecimals(8)
        self._btc_sp.setValue(0.1)
        self._btc_sp.setSingleStep(0.01)
        self._price_sp = QDoubleSpinBox()
        self._price_sp.setRange(1, 10_000_000)
        self._price_sp.setDecimals(2)
        self._price_sp.setValue(50000.0)
        self._price_sp.setSingleStep(1000)
        self._price_sp.setPrefix("$")
        self._notes_edit = QLineEdit()
        self._notes_edit.setPlaceholderText("optional")
        add_btn = QPushButton("Add Lot")
        add_btn.clicked.connect(self._add_lot)
        add_form.addRow("Date:", self._date_edit)
        add_form.addRow("BTC:", self._btc_sp)
        add_form.addRow("Price/BTC:", self._price_sp)
        add_form.addRow("Notes:", self._notes_edit)
        add_form.addRow(add_btn)
        ctrl_l.addWidget(add_grp)

        # Summary group
        sum_grp  = QGroupBox("Summary")
        sum_form = QFormLayout(sum_grp)
        sum_form.setSpacing(4)
        self._total_btc_lbl = QLabel("0.00000000 BTC")
        self._num_lots_lbl  = QLabel("0 lots")
        self._avg_price_lbl = QLabel("—")
        self._total_usd_lbl = QLabel("—")
        self._avg_pct_lbl   = QLabel("—")
        self._total_btc_lbl.setStyleSheet("font-weight:bold; font-size:13px;")
        sum_form.addRow("Total BTC:", self._total_btc_lbl)
        sum_form.addRow("# Lots:", self._num_lots_lbl)
        sum_form.addRow("Wtd Avg Price:", self._avg_price_lbl)
        sum_form.addRow("Total Paid:", self._total_usd_lbl)
        sum_form.addRow("Avg Percentile:", self._avg_pct_lbl)
        ctrl_l.addWidget(sum_grp)

        # Actions group
        act_grp = QGroupBox("Actions")
        act_l   = QVBoxLayout(act_grp)
        act_l.setSpacing(4)
        del_btn = QPushButton("Delete Selected Row(s)")
        del_btn.clicked.connect(self._delete_selected)
        clr_btn = QPushButton("Clear All Lots")
        clr_btn.clicked.connect(self._clear_all)
        csv_btn = QPushButton("Import CSV…")
        csv_btn.clicked.connect(self._import_csv)
        act_l.addWidget(del_btn)
        act_l.addWidget(clr_btn)
        act_l.addWidget(csv_btn)
        ctrl_l.addWidget(act_grp)

        ctrl_l.addStretch()
        ctrl_scroll.setWidget(ctrl_w)
        self._splitter = splitter
        self._splitter.addWidget(left_w)
        self._splitter.addWidget(ctrl_scroll)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

    # ── lot operations ────────────────────────────────────────────────────
    def _add_lot(self):
        date_str = self._date_edit.date().toString("yyyy-MM-dd")
        btc   = self._btc_sp.value()
        price = self._price_sp.value()
        notes = self._notes_edit.text().strip()
        try:
            t = (pd.Timestamp(date_str) - self.m.genesis).days / 365.25
        except Exception:
            return
        pct_q = _find_lot_percentile(t, price, self.m.qr_fits)
        self._lots.append({"date": date_str, "btc": btc, "price": price,
                           "total_usd": btc * price, "pct_q": pct_q, "notes": notes})
        self._lots.sort(key=lambda l: l["date"])
        self._refresh_table()
        self._save_lots()
        self.lots_changed.emit(self.total_btc())

    def _delete_selected(self):
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < self._table.rowCount():
                date_val = self._table.item(r, 0).text()
                btc_val  = float(self._table.item(r, 1).text())
                # Remove first matching lot
                for j, lot in enumerate(self._lots):
                    if lot["date"] == date_val and abs(lot["btc"] - btc_val) < 1e-10:
                        self._lots.pop(j)
                        break
        self._refresh_table()
        self._save_lots()
        self.lots_changed.emit(self.total_btc())

    def _clear_all(self):
        if self._lots:
            if QMessageBox.question(self, "Clear All", "Remove all lots?",
                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
        self._lots.clear()
        self._refresh_table()
        self._save_lots()
        self.lots_changed.emit(self.total_btc())

    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Lots CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]
            added = 0
            for _, row in df.iterrows():
                try:
                    date_str = str(row.get("date", "")).strip()
                    btc      = float(row.get("btc", row.get("amount", 0)))
                    price    = float(row.get("price", row.get("price_usd", 0)))
                    notes    = str(row.get("notes", "")).strip()
                    if not date_str or btc <= 0 or price <= 0:
                        continue
                    t = (pd.Timestamp(date_str) - self.m.genesis).days / 365.25
                    pct_q = _find_lot_percentile(t, price, self.m.qr_fits)
                    self._lots.append({"date": date_str, "btc": btc, "price": price,
                                       "total_usd": btc * price, "pct_q": pct_q, "notes": notes})
                    added += 1
                except Exception:
                    continue
            self._lots.sort(key=lambda l: l["date"])
            self._refresh_table()
            self._save_lots()
            self.lots_changed.emit(self.total_btc())
            QMessageBox.information(self, "Import", f"Imported {added} lot(s).")
        except Exception as e:
            QMessageBox.warning(self, "Import error", str(e))

    def _refresh_table(self):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        for lot in self._lots:
            r = self._table.rowCount()
            self._table.insertRow(r)
            pct_str = f"Q{lot['pct_q']*100:.2f}%"
            for c, val in enumerate([
                lot["date"],
                f"{lot['btc']:.8f}",
                f"${lot['price']:,.2f}",
                f"${lot['total_usd']:,.2f}",
                pct_str,
                lot.get("notes", ""),
            ]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(r, c, item)
        self._table.setSortingEnabled(True)
        self._update_summary()

    def _update_summary(self):
        total_btc = self.total_btc()
        total_usd = sum(l["total_usd"] for l in self._lots)
        n = len(self._lots)
        self._total_btc_lbl.setText(f"{total_btc:.8f} BTC")
        self._num_lots_lbl.setText(f"{n} lot{'s' if n != 1 else ''}")
        if total_btc > 0:
            self._avg_price_lbl.setText(fmt_price(total_usd / total_btc))
            self._total_usd_lbl.setText(fmt_price(total_usd))
            avg_pct = sum(l["pct_q"] * l["btc"] for l in self._lots) / total_btc
            self._avg_pct_lbl.setText(f"Q{avg_pct*100:.2f}%")
        else:
            self._avg_price_lbl.setText("—")
            self._total_usd_lbl.setText("—")
            self._avg_pct_lbl.setText("—")

    def total_btc(self):
        return sum(l["btc"] for l in self._lots)

    def get_lots(self):
        return list(self._lots)

    def _save_lots(self):
        lots_path = _SETTINGS_PATH.parent / "lots.json"
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(lots_path, "w") as f:
                json.dump(self._lots, f, indent=2)
        except Exception:
            pass

    def _load_lots(self):
        lots_path = _SETTINGS_PATH.parent / "lots.json"
        if not lots_path.exists():
            return
        try:
            with open(lots_path) as f:
                data = json.load(f)
            for lot in data:
                try:
                    t = (pd.Timestamp(lot["date"]) - self.m.genesis).days / 365.25
                    lot["pct_q"] = _find_lot_percentile(t, lot["price"], self.m.qr_fits)
                    lot.setdefault("notes", "")
                except Exception:
                    pass
            self._lots = data
            self._refresh_table()
        except Exception:
            pass


# ── MainWindow ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, model: ModelData):
        super().__init__()
        self.m = model
        self.setWindowTitle("Bitcoin Projections")
        self.resize(1280, 780)

        self.bubble_tab        = BubbleTab(model)
        self.heatmap_tab       = HeatmapTab(model)
        self.dca_tab           = DCATab(model)
        self.retire_tab        = RetireTab(model)
        self.stack_tracker_tab = StackTrackerTab(model)

        # Wire StackTracker → all tabs lot lists
        for _tab in (self.bubble_tab, self.heatmap_tab, self.dca_tab, self.retire_tab):
            _tab.set_lot_source(self.stack_tracker_tab)

        tabs = QTabWidget()
        tabs.addTab(self.bubble_tab,        "Bubble + QR Overlay")
        tabs.addTab(self.heatmap_tab,       "CAGR Heatmap")
        tabs.addTab(self.dca_tab,           "BTC Accumulator")
        tabs.addTab(self.retire_tab,        "BTC Retireator")
        tabs.addTab(self.stack_tracker_tab, "Stack Tracker")

        # Link all stack spinboxes so changing one updates all others
        self._stack_spinboxes = [
            self.bubble_tab._stack_sp,
            self.heatmap_tab._stack_sp_hm,
            self.dca_tab._stack_sp,
            self.retire_tab._stack_sp,
        ]
        for sp in self._stack_spinboxes:
            sp.valueChanged.connect(self._sync_stack)

        # Link quantile state across tabs
        self._syncing_qs = False
        self.bubble_tab.q_state_changed.connect(
            lambda s: self._on_q_state_changed(s, self.bubble_tab))
        self.dca_tab.q_state_changed.connect(
            lambda s: self._on_q_state_changed(s, self.dca_tab))
        self.retire_tab.q_state_changed.connect(
            lambda s: self._on_q_state_changed(s, self.retire_tab))

        # Minor ticks sync across chart tabs
        self._syncing_minor_ticks = False
        for _mt_chk in (self.bubble_tab._minor_ticks_chk,
                        self.dca_tab._minor_ticks_chk,
                        self.retire_tab._minor_ticks_chk):
            _mt_chk.toggled.connect(
                lambda checked, chk=_mt_chk: self._on_minor_ticks_changed(checked, chk))

        # Font sync across all tabs (individual role pickers)
        self._syncing_fonts = False
        self._font_role_map = {
            'title':  [self.bubble_tab._font_title,  self.heatmap_tab._hfont_title,
                       self.dca_tab._font_title,    self.retire_tab._font_title],
            'axis_t': [self.bubble_tab._font_axis_t, self.heatmap_tab._hfont_axis_t,
                       self.dca_tab._font_axis_t,   self.retire_tab._font_axis_t],
            'ticks':       [self.bubble_tab._font_ticks,       self.heatmap_tab._hfont_ticks,
                            self.dca_tab._font_ticks,         self.retire_tab._font_ticks],
            'ticks_minor': [self.bubble_tab._font_ticks_minor,
                            self.dca_tab._font_ticks_minor,   self.retire_tab._font_ticks_minor],
            'legend':      [self.bubble_tab._font_legend, self.heatmap_tab._hfont_cells,
                            self.dca_tab._font_legend,    self.retire_tab._font_legend],
        }
        self._tab_font_pickers = {
            self.bubble_tab:  (self.bubble_tab._font_title,  self.bubble_tab._font_axis_t,
                               self.bubble_tab._font_ticks,  self.bubble_tab._font_ticks_minor,
                               self.bubble_tab._font_legend),
            self.heatmap_tab: (self.heatmap_tab._hfont_title, self.heatmap_tab._hfont_axis_t,
                               self.heatmap_tab._hfont_ticks, self.heatmap_tab._hfont_cells),
            self.dca_tab:     (self.dca_tab._font_title,    self.dca_tab._font_axis_t,
                               self.dca_tab._font_ticks,    self.dca_tab._font_ticks_minor,
                               self.dca_tab._font_legend),
            self.retire_tab:  (self.retire_tab._font_title,  self.retire_tab._font_axis_t,
                               self.retire_tab._font_ticks,  self.retire_tab._font_ticks_minor,
                               self.retire_tab._font_legend),
        }
        for _role, _fps in self._font_role_map.items():
            for _fp in _fps:
                _fp.font_changed.connect(
                    lambda fam, r=_role, sfp=_fp: self._on_font_changed(r, fam, sfp))

        # Wire font size changes across tabs
        self._syncing_font_sizes = False
        for _role, _fps in self._font_role_map.items():
            for _fp in _fps:
                _fp.size_changed.connect(
                    lambda sz, r=_role, sfp=_fp: self._on_font_size_changed(r, sz, sfp))

        # Wire 'All tabs' font button from each tab
        for _tab in (self.bubble_tab, self.heatmap_tab, self.dca_tab, self.retire_tab):
            _tab.all_fonts_applied.connect(self._apply_font_to_all_tabs)

        # Wire StackTracker lots → BTC Owned spinboxes (one-way)
        self.stack_tracker_tab.lots_changed.connect(self._on_lots_changed)

        # ── in-app button bar ─────────────────────────────────────────────
        btn_bar = QWidget()
        btn_bar.setFixedHeight(36)
        btn_l = QHBoxLayout(btn_bar)
        btn_l.setContentsMargins(6, 3, 6, 3)
        btn_l.setSpacing(6)
        for label, slot in [
            ("Update Price Data…", self._update_price),
            ("Save Model Override", self._save_override),
            ("Save Settings",       self._save_settings),
            ("About",               self._about),
            ("Quit",                self.close),
        ]:
            b = QPushButton(label)
            b.setFixedHeight(28)
            b.clicked.connect(slot)
            btn_l.addWidget(b)
        btn_l.addStretch()
        self._status_lbl = QLabel(
            f"Loaded: {Path(model._path).name}  |  "
            f"{len(model.price_dates)} price rows  |  "
            f"{model.n_future_max} future bubbles")
        self._status_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        btn_l.addWidget(self._status_lbl)

        central = QWidget()
        central_l = QVBoxLayout(central)
        central_l.setContentsMargins(0, 0, 0, 0)
        central_l.setSpacing(0)
        central_l.addWidget(btn_bar)
        central_l.addWidget(tabs)
        self.setCentralWidget(central)

        # Load saved settings (if any) before first draw
        saved = _load_ui_settings()
        if saved.get("bubble"):
            self.bubble_tab._apply_settings(saved["bubble"])
        else:
            self.bubble_tab.redraw()
        if saved.get("heatmap"):
            self.heatmap_tab._apply_settings(saved["heatmap"])
        else:
            self.heatmap_tab.redraw()
        if saved.get("dca"):
            self.dca_tab._apply_settings(saved["dca"])
        else:
            self.dca_tab.redraw()
        if saved.get("retire"):
            self.retire_tab._apply_settings(saved["retire"])
        else:
            self.retire_tab.redraw()

        # Populate lot lists in all tabs from loaded StackTracker data
        for _tab in (self.bubble_tab, self.heatmap_tab, self.dca_tab, self.retire_tab):
            if hasattr(_tab, 'refresh_lot_list'):
                _tab.refresh_lot_list()

    def _update_price(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Price CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            self.m.update_from_csv(path)
            self.bubble_tab.redraw()
            self.heatmap_tab.redraw()
            self._status_lbl.setText(
                f"Updated from {Path(path).name}  |  {len(self.m.price_dates)} rows")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV:\n{e}")

    def _save_override(self):
        try:
            dst = self.m.save_user_override()
            QMessageBox.information(self, "Saved", f"Model override saved to:\n{dst}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _save_settings(self):
        try:
            _save_ui_settings({
                "bubble":  self.bubble_tab._collect_settings(),
                "heatmap": self.heatmap_tab._collect_settings(),
                "dca":     self.dca_tab._collect_settings(),
                "retire":  self.retire_tab._collect_settings(),
            })
            self._status_lbl.setText(f"Settings saved to {_SETTINGS_PATH}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")

    def _sync_stack(self, value):
        """Propagate stack spinbox change to all other tabs."""
        for sp in self._stack_spinboxes:
            if abs(sp.value() - value) > 1e-9:
                sp.blockSignals(True)
                sp.setValue(value)
                sp.blockSignals(False)

    # ── quantile sync ─────────────────────────────────────────────────────────
    def _on_q_state_changed(self, new_state, source_tab):
        """Sync quantile state from source_tab to all other tabs."""
        if getattr(self, "_syncing_qs", False):
            return
        self._syncing_qs = True
        try:
            for tab in (self.bubble_tab, self.dca_tab, self.retire_tab):
                if tab is source_tab:
                    continue
                self._sync_q_to_tab(tab, new_state)
            self._sync_q_to_heatmap(new_state)
        finally:
            self._syncing_qs = False

    def _sync_q_to_tab(self, tab, new_state):
        """Update tab._q_state to match new_state (add/remove/vis; keep per-tab style)."""
        old_map = {qs["q"]: qs for qs in tab._q_state}
        result = []
        for qs in new_state:
            q = qs["q"]
            if q not in self.m.qr_fits:
                try:
                    self.m.qr_fits[q] = _fit_one_qr(self.m, q)
                except Exception:
                    continue
            if q in old_map:
                entry = dict(old_map[q])
                entry["vis"] = qs["vis"]
            else:
                pct = q * 100
                lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
                entry = {"q": q, "lbl": lbl, "vis": qs["vis"],
                         "color": qs["color"], "ls": qs["ls"], "lw": qs["lw"]}
            result.append(entry)
        tab._busy = True
        tab._q_state = result
        tab._rebuild_q_rows()
        tab._busy = False
        tab.redraw()

    def _sync_q_to_heatmap(self, new_state):
        """Sync quantile list (add/remove) and visibility (selection) in heatmap exit list."""
        hm = self.heatmap_tab
        hm._busy = True
        new_qs = {qs["q"] for qs in new_state}
        vis_map = {qs["q"]: qs["vis"] for qs in new_state}
        # Add missing quantiles
        existing = {float(hm._xq_list.item(i).data(Qt.UserRole))
                    for i in range(hm._xq_list.count())}
        for q in sorted(new_qs):
            if q not in existing:
                if q not in self.m.qr_fits:
                    try:
                        self.m.qr_fits[q] = _fit_one_qr(self.m, q)
                    except Exception:
                        continue
                pct = q * 100
                lbl = f"Q{pct:.4g}%" if pct >= 1 else f"Q{pct:.3g}%"
                item = QListWidgetItem(lbl)
                item.setData(Qt.UserRole, q)
                hm._xq_list.addItem(item)
        # Sync selection
        for i in range(hm._xq_list.count()):
            q = float(hm._xq_list.item(i).data(Qt.UserRole))
            if q in vis_map:
                hm._xq_list.item(i).setSelected(vis_map[q])
        hm._busy = False
        hm.redraw()

    def _on_minor_ticks_changed(self, checked, source_chk):
        """Sync minor-ticks checkbox state to all other chart tabs."""
        if getattr(self, '_syncing_minor_ticks', False):
            return
        self._syncing_minor_ticks = True
        try:
            for chk in (self.bubble_tab._minor_ticks_chk,
                        self.dca_tab._minor_ticks_chk,
                        self.retire_tab._minor_ticks_chk):
                if chk is not source_chk and chk.isChecked() != checked:
                    chk.blockSignals(True)
                    chk.setChecked(checked)
                    chk.blockSignals(False)
                    # Trigger redraw on the peer tab manually
                    if chk is self.bubble_tab._minor_ticks_chk:
                        self.bubble_tab.redraw()
                    elif chk is self.dca_tab._minor_ticks_chk:
                        self.dca_tab.redraw()
                    elif chk is self.retire_tab._minor_ticks_chk:
                        self.retire_tab.redraw()
        finally:
            self._syncing_minor_ticks = False

    def _on_font_changed(self, role, family, source_fp):
        """Propagate font family change from source_fp to all other tabs."""
        if getattr(self, '_syncing_fonts', False):
            return
        self._syncing_fonts = True
        try:
            # Update peer pickers (set_family does NOT emit font_changed)
            for fp in self._font_role_map.get(role, []):
                if fp is not source_fp:
                    fp.set_family(family)
            # Redraw each tab whose picker was updated
            source_tab = next(
                (tab for tab, fps in self._tab_font_pickers.items()
                 if source_fp in fps), None)
            for tab in (self.bubble_tab, self.heatmap_tab,
                        self.dca_tab, self.retire_tab):
                if tab is not source_tab:
                    tab.redraw()
        finally:
            self._syncing_fonts = False

    def _on_font_size_changed(self, role, size, source_fp):
        """Sync font size change to all peer pickers for the same role."""
        if getattr(self, '_syncing_font_sizes', False):
            return
        self._syncing_font_sizes = True
        try:
            for fp in self._font_role_map.get(role, []):
                if fp is not source_fp:
                    fp.set_size(size)
            source_tab = next(
                (tab for tab, fps in self._tab_font_pickers.items()
                 if source_fp in fps), None)
            for tab in (self.bubble_tab, self.heatmap_tab,
                        self.dca_tab, self.retire_tab):
                if tab is not source_tab:
                    tab.redraw()
        finally:
            self._syncing_font_sizes = False

    def _apply_font_to_all_tabs(self, family, size):
        """Apply one font family+size to every role picker on every tab."""
        for fps in self._font_role_map.values():
            for fp in fps:
                fp.set_family(family)
                fp.set_size(size)
        for _all_fp in (self.bubble_tab._font_all_b,
                        self.heatmap_tab._font_all_hm,
                        self.dca_tab._font_all_b,
                        self.retire_tab._font_all_b):
            _all_fp.set_family(family)
            _all_fp.set_size(size)
        for tab in (self.bubble_tab, self.heatmap_tab, self.dca_tab, self.retire_tab):
            tab.redraw()

    def _on_lots_changed(self, total_btc):
        """Sync total BTC from StackTracker to all stack spinboxes (one-way)."""
        for sp in self._stack_spinboxes:
            if abs(sp.value() - total_btc) > 1e-9:
                sp.blockSignals(True)
                sp.setValue(total_btc)
                sp.blockSignals(False)
        # Refresh lot lists on all tabs then redraw
        for _tab in (self.bubble_tab, self.heatmap_tab, self.dca_tab, self.retire_tab):
            _tab.refresh_lot_list()
        self.bubble_tab.redraw()
        self.heatmap_tab.redraw()
        self.dca_tab.redraw()
        self.retire_tab.redraw()

    def _about(self):
        QMessageBox.about(self, "About",
            "Bitcoin Projections\n\n"
            "Interactive standalone app built from SP.ipynb.\n\n"
            "• Tab 1: Bubble model + quantile regression overlay\n"
            "• Tab 2: CAGR heatmap (entry → exit scenarios)\n\n"
            "File > Update Price Data to load a new BitcoinPricesDaily.csv.\n"
            "File > Save Model Override to persist QR updates.")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    path = _find_model_data()
    if path is None:
        # Minimal Qt error before app starts — use print
        print("ERROR: model_data.pkl not found.\n"
              "Run the SP.ipynb export cell first, or pass the path as an argument.")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("Bitcoin Projections")

    try:
        model = ModelData(path)
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        sys.exit(1)

    win = MainWindow(model)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
