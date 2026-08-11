#!/usr/bin/env python3
"""Task 4 (grid assembly + parallel driver) for the "Time Machine" feature.

Builds the as-of grid consumed by the Time Machine loader/UI: for a list of
frame dates, cold-fits every EPPL config (Task 2, ``fit_config_asof``) and
optionally the Bubble Model (Task 3, ``fit_bm_asof``) on data truncated to
each frame's as-of horizon, then writes the whole thing as one gzip-
compressed JSON file (no pickle, no ``dtype=object``).

Grid shape::

    {"frames": ["YYYY-MM-DD", ...],
     "models": {"bub": [bm_frame_dict, ...],
                "ecfg_1d_1u": [ecfg_frame_dict, ...], ...}}

``model_key`` is ``"bub"`` (Bubble Model) or one of the 36 ``"ecfg_*"`` keys
from ``fit_all_eppl_configs.config_key``. There is intentionally no
flagship ``"eppl"`` frame series -- it never draws on Tab 1
(FLAGSHIP_POLICY = "own_frame", Task 0).

Run (documented, NOT exercised by the test suite -- multi-hour build):

    btc_venv/bin/python3 tools/build_timemachine_grid.py --full

Cold-fits each frame independently (no warm-starting from the previous
frame's params) and parallelizes ``(config, frame)`` fit tasks over a
``ProcessPoolExecutor`` when ``workers > 1``. ``workers=1`` (used by the
unit test) takes a plain sequential path -- multiprocessing is exercised
only by ``--full``.
"""
import argparse
import gzip
import json
import os
import sys

# Pin single-threaded BLAS BEFORE numpy is imported -- with ~24 worker
# processes each running differential_evolution, an unpinned OpenBLAS/MKL
# thread pool per process oversubscribes the box (24 processes x N BLAS
# threads each) and the build slows down instead of speeding up. Must be
# set before `import numpy` -- OpenBLAS/MKL read these env vars once, at
# first use, not on every call.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), ROOT]

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

from tools.timemachine.fit_eppl_asof import fit_config_asof  # noqa: E402
from tools.timemachine.fit_bm_asof import fit_bm_asof  # noqa: E402
from tools.timemachine.fit_qr_asof import fit_qr_asof  # noqa: E402
from tools.timemachine.fit_spl_asof import fit_spl_asof  # noqa: E402
from tools.timemachine.fit_logi_asof import fit_logi_asof  # noqa: E402
from tools.model_toolkit.data import load_prices  # noqa: E402
from tools.timemachine.frames import frame_dates  # noqa: E402
from fit_all_eppl_configs import build_model_fn, config_key, all_configs  # noqa: E402

GENESIS = pd.Timestamp("2009-07-25")

# Task 0 constants (tools/timemachine/spike.py CONSTANTS).
EPPL_MAXITER = 600
LEFT_EDGE_DATE = "2012-01-24"

DEFAULT_OUT_PATH = os.path.join(ROOT, "timemachine_grid.json.gz")

# continuity_scan() thresholds (Step 3b).
_JUMP_THRESHOLD = 0.5
_R2_THRESHOLD = 0.85

# BM downsample cap (post-build size fix). BM frames carry the FULL
# PLOT_GRID_POINTS plotting grid (3000 pts, tools/model_toolkit/composite.py)
# x 4 comp_by_n rows x 142 frames -- that alone is most of the ~19MB grid
# file, well over the ~5MB budget every gunicorn worker pays to load this
# (ALARA). EPPL frames are params-only (a handful of floats) and are never
# touched by this. Task 7 interpolates comp_by_n/t_grid onto its own display
# grid anyway, so shipping the full 3000-point resolution buys nothing.
MAX_BM_POINTS = 512


# ─────────────────────────────────────────────────────────────────────────
# JSON coercion — numpy scalars are NOT all natively JSON-serializable
# (np.float64 subclasses float and serializes fine; np.int64 does NOT).
# Recursively convert to plain Python so json.dump never chokes.
# ─────────────────────────────────────────────────────────────────────────
def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    return obj


def _frame_ymax(frame_date):
    """As-of horizon in years since the calendar genesis, for one frame date."""
    return (pd.Timestamp(frame_date) - GENESIS).days / 365.25


def _downsample_bm_frame(frame, max_pts=MAX_BM_POINTS):
    """Downsample one BM frame's ``t_grid`` + every ``comp_by_n`` row to
    ``<= max_pts`` points, keeping every row aligned to the SAME (shorter)
    ``t_grid`` -- ``btc_web/timemachine.py::asof_bm`` reconstructs
    ``support_bm`` from ``t_grid`` at load time, so alignment has to hold
    after this runs, not be re-derived by the loader.

    A no-op (returns ``frame`` unchanged) when it's already <= max_pts --
    safe to call unconditionally, and idempotent if re-run on an
    already-downsampled frame.

    Even index-stride subsampling (not interpolation): ``t_grid`` is
    ``np.linspace`` in ``build_composite`` (uniform in linear years, not
    log), so a stride keeps points evenly spaced in the same space the
    original grid used; Task 7 interpolates onto its own display grid
    regardless, so exact point placement here isn't load-bearing.

    Parameters
    ----------
    frame : dict or None
        One ``models["bub"][i]`` entry (``fit_bm_asof`` output). ``None``
        passes through unchanged (null/failed frame).
    max_pts : int
        Point cap (default ``MAX_BM_POINTS`` = 512).

    Returns
    -------
    dict or None
    """
    if frame is None:
        return None
    t_grid = frame["t_grid"]
    n = len(t_grid)
    if n <= max_pts:
        return frame

    idx = np.unique(np.round(np.linspace(0, n - 1, max_pts)).astype(int))
    t_arr = np.asarray(t_grid)

    out = dict(frame)  # shallow copy -- never mutate the caller's dict
    out["t_grid"] = t_arr[idx].tolist()
    out["comp_by_n"] = [np.asarray(row)[idx].tolist() for row in frame["comp_by_n"]]
    return out


def downsample_existing_grid(path, max_pts=MAX_BM_POINTS):
    """Post-process an already-built grid file IN PLACE: downsample every
    ``"bub"`` frame's ``comp_by_n``/``t_grid`` to ``<= max_pts`` points.
    Every ``ecfg_*`` (params-only) frame is left untouched.

    Avoids a full rebuild (~1h40m on 24 cores) when only the BM
    plotting-grid resolution needs to shrink -- no re-fitting happens here,
    this only reshapes already-fitted data.

    Parameters
    ----------
    path : str
        Path to the gzip-compressed JSON grid to rewrite in place.
    max_pts : int
        Point cap passed to ``_downsample_bm_frame`` (default
        ``MAX_BM_POINTS`` = 512).

    Returns
    -------
    dict
        The rewritten (JSON-safe) grid object, same as what was written.
    """
    with gzip.open(path, "rt") as f:
        grid = json.load(f)

    bub = grid["models"].get("bub")
    if bub is None:
        print(f"downsample_existing_grid: no 'bub' key in {path}, nothing to do")
        return grid

    grid["models"]["bub"] = [_downsample_bm_frame(rec, max_pts=max_pts) for rec in bub]

    with gzip.open(path, "wt") as f:
        json.dump(grid, f)

    return grid


# ─────────────────────────────────────────────────────────────────────────
# Module-level worker wrappers — required so ProcessPoolExecutor can pickle
# them (bound methods / closures are not picklable).
# ─────────────────────────────────────────────────────────────────────────
def _fit_ecfg_task(cfg, t, lp, ymax, maxiter):
    return fit_config_asof(cfg, t, lp, ymax, maxiter)


def _fit_bm_task(prices, ymax):
    # Downsampled INSIDE the worker (not after fut.result()) so the
    # oversized (3000-pt x 4-row) array never has to round-trip the
    # ProcessPoolExecutor pickle boundary -- smaller IPC payload, not just
    # a smaller file.
    return _downsample_bm_frame(fit_bm_asof(prices, ymax))


def _fit_qr_task(prices, ymax):
    # QR frames are params-only (27 quantiles x 3 floats) -- no downsampling.
    return fit_qr_asof(prices, ymax)


def _fit_spl_task(prices, ymax):
    # spl frames are params-only (log10_L/t0/beta) -- no downsampling.
    return fit_spl_asof(prices, ymax)


def _fit_logi_task(prices, ymax):
    # logi frames are params-only (K/r/t0) -- no downsampling.
    return fit_logi_asof(prices, ymax)


# Kinds that share the (prices, ymax) -> {"params": ..., "sigma": ..., "r2": ...}
# call shape -- qr/spl/logi, but NOT "ecfg" (extra cfg/t/lp/maxiter args) or
# "bm" (downsampled + a different return shape). Used by build_grid's dispatch
# AND add_series_to_grid (the DRY incremental-add path, Task 3).
_ASOF_FITTERS = {"qr": fit_qr_asof, "spl": fit_spl_asof, "logi": fit_logi_asof}
_ASOF_TASKS = {"qr": _fit_qr_task, "spl": _fit_spl_task, "logi": _fit_logi_task}


def _pool_worker_init():
    """ProcessPoolExecutor initializer -- pin single-threaded BLAS in each
    worker process too. Belt-and-suspenders alongside the module-level
    ``os.environ.setdefault`` calls above: those cover the common case
    (fork inherits parent env; spawn re-executes this module from the top
    before importing numpy), this covers a worker whose numpy/BLAS was
    already imported/initialized by some other path before the env vars
    it inherited took effect.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def build_grid(frames, configs, include_bm, out_path, maxiter, workers,
               include_qr=True, include_spl=False, include_logi=False):
    """Build the as-of grid and write it as gzipped JSON.

    Parameters
    ----------
    frames : list[str]
        As-of frame dates ("YYYY-MM-DD"), e.g. from ``frame_dates()``.
    configs : list[tuple]
        EPPL configs ``(n_log, n_cal, log_damps, cal_damps)``, e.g. from
        ``fit_all_eppl_configs.all_configs()``.
    include_bm : bool
        Whether to also cold-fit the Bubble Model per frame (key ``"bub"``).
    out_path : str
        Destination path for the gzip-compressed JSON grid.
    maxiter : int
        ``differential_evolution`` maxiter passed to every EPPL fit.
    workers : int
        ``1`` -> simple sequential loop. ``>1`` -> ``ProcessPoolExecutor``
        with ``max_workers = max(1, workers)`` (caller-controlled -- pass
        the desired core count directly, e.g. ``os.cpu_count()``).
    include_qr, include_spl, include_logi : bool
        Whether to also cold-fit QR / spl / logi per frame (keys ``"qr"``,
        ``"spl"``, ``"logi"``) -- all params-only, no downsampling. ``qr``
        defaults on (cheap, no DE); ``spl``/``logi`` default off (DE-based,
        like EPPL) -- the real grid gets them via the incremental
        ``add_series_to_grid`` path (Task 3 / ``--add-models``) instead of a
        full rebuild, so a full rebuild doesn't need them on by default.

    Returns
    -------
    dict
        The assembled (JSON-safe) grid object, same as what was written.
    """
    prices = load_prices("BitcoinPricesDaily.csv")
    t = prices.df_full["years"].values
    lp = prices.df_full["log_price"].values

    ymaxes = {frame: _frame_ymax(frame) for frame in frames}

    # configs carry lists (log_damps/cal_damps) so they're unhashable --
    # keep the key alongside the config instead of using it as a dict key.
    ecfg_key_list = [config_key(*cfg) for cfg in configs]
    models = {key: [None] * len(frames) for key in ecfg_key_list}
    if include_bm:
        models["bub"] = [None] * len(frames)
    if include_qr:
        models["qr"] = [None] * len(frames)
    if include_spl:
        models["spl"] = [None] * len(frames)
    if include_logi:
        models["logi"] = [None] * len(frames)

    # Flat job list: (kind, key, frame_idx, frame_date, cfg_or_None)
    jobs = []
    for cfg, key in zip(configs, ecfg_key_list):
        for i, frame in enumerate(frames):
            jobs.append(("ecfg", key, i, frame, cfg))
    if include_bm:
        for i, frame in enumerate(frames):
            jobs.append(("bm", "bub", i, frame, None))
    if include_qr:
        for i, frame in enumerate(frames):
            jobs.append(("qr", "qr", i, frame, None))
    if include_spl:
        for i, frame in enumerate(frames):
            jobs.append(("spl", "spl", i, frame, None))
    if include_logi:
        for i, frame in enumerate(frames):
            jobs.append(("logi", "logi", i, frame, None))

    failed = []

    if workers <= 1:
        for kind, key, i, frame, cfg in jobs:
            ymax = ymaxes[frame]
            try:
                if kind == "ecfg":
                    result = fit_config_asof(cfg, t, lp, ymax, maxiter)
                elif kind in _ASOF_FITTERS:
                    result = _ASOF_FITTERS[kind](prices, ymax)
                else:
                    result = _downsample_bm_frame(fit_bm_asof(prices, ymax))
                models[key][i] = result
            except Exception as e:  # noqa: BLE001 - log and keep building
                print(f"build_grid: FAILED frame={frame} key={key}: {e!r}")
                failed.append((frame, key, repr(e)))
    else:
        max_workers = max(1, workers)
        with ProcessPoolExecutor(
            max_workers=max_workers, initializer=_pool_worker_init,
        ) as ex:
            futures = {}
            for kind, key, i, frame, cfg in jobs:
                ymax = ymaxes[frame]
                if kind == "ecfg":
                    fut = ex.submit(_fit_ecfg_task, cfg, t, lp, ymax, maxiter)
                elif kind in _ASOF_TASKS:
                    fut = ex.submit(_ASOF_TASKS[kind], prices, ymax)
                else:
                    fut = ex.submit(_fit_bm_task, prices, ymax)
                futures[fut] = (kind, key, i, frame)
            for fut in as_completed(futures):
                kind, key, i, frame = futures[fut]
                try:
                    models[key][i] = fut.result()
                except Exception as e:  # noqa: BLE001 - log and keep building
                    print(f"build_grid: FAILED frame={frame} key={key}: {e!r}")
                    failed.append((frame, key, repr(e)))

    if failed:
        print(f"build_grid: {len(failed)} frame/model fit(s) FAILED "
              "(logged above) -- corresponding grid entries are null, not "
              "silently dropped.")

    grid = _to_jsonable({"frames": list(frames), "models": models})

    suspects = continuity_scan(grid)
    for s in suspects:
        print(s)

    with gzip.open(out_path, "wt") as f:
        json.dump(grid, f)

    return grid


def add_series_to_grid(grid_path, kinds, workers):
    """Incrementally add one or more model series to an EXISTING grid file.

    ALARA path: load the grid, fit each ``kind`` (``qr``/``spl``/``logi``, see
    ``_ASOF_FITTERS``) for its EXACT ``frames`` (via ``_frame_ymax``), inject
    ``models[kind]``, and rewrite the file in place -- WITHOUT recomputing the
    expensive BM/EPPL series. Frame alignment is exact because each frame's
    as-of horizon depends only on data <= D (immutable history), so fitting on
    the current CSV reproduces the same window the grid was originally built
    with. All jobs across all ``kinds`` share ONE ``ProcessPoolExecutor`` (not
    one pool per kind) to keep pool-spinup overhead flat regardless of how
    many series are being added at once.

    Generalizes the original QR-only ``add_qr_to_grid`` (still present as a
    thin ``add_series_to_grid(path, ["qr"], workers)`` wrapper below, for
    back-compat with the ``--add-qr`` CLI and its existing test).

    Parameters
    ----------
    grid_path : str
        Path to the gzip-compressed JSON grid to rewrite in place.
    kinds : list[str]
        Model series to (re)fit, each one of ``sorted(_ASOF_FITTERS)`` --
        currently ``{"logi", "qr", "spl"}``.
    workers : int
        ``ProcessPoolExecutor`` max_workers for the fits. Pass the desired
        core count (e.g. ``os.cpu_count()`` capped at 24). ``<= 1`` runs a
        simple sequential loop.

    Returns
    -------
    dict
        The updated grid object (also written back to ``grid_path``).
    """
    invalid = sorted(set(kinds) - set(_ASOF_FITTERS))
    if invalid:
        raise ValueError(
            f"add_series_to_grid: unknown kind(s) {invalid}; choose from "
            f"{sorted(_ASOF_FITTERS)}")

    prices = load_prices("BitcoinPricesDaily.csv")
    with gzip.open(grid_path, "rt") as f:
        grid = json.load(f)
    frames = grid["frames"]
    ymaxes = [_frame_ymax(fr) for fr in frames]
    series = {kind: [None] * len(frames) for kind in kinds}
    failed = []

    max_workers = max(1, workers)
    if max_workers <= 1:
        for kind in kinds:
            fit_fn = _ASOF_FITTERS[kind]
            for i, ymax in enumerate(ymaxes):
                try:
                    series[kind][i] = fit_fn(prices, ymax)
                except Exception as e:  # noqa: BLE001 - log and keep going
                    print(f"add_series_to_grid: FAILED kind={kind} "
                          f"frame={frames[i]}: {e!r}")
                    failed.append((kind, frames[i], repr(e)))
    else:
        with ProcessPoolExecutor(
            max_workers=max_workers, initializer=_pool_worker_init,
        ) as ex:
            futures = {}
            for kind in kinds:
                task_fn = _ASOF_TASKS[kind]
                for i, ymax in enumerate(ymaxes):
                    futures[ex.submit(task_fn, prices, ymax)] = (kind, i)
            for fut in as_completed(futures):
                i = futures[fut]
                kind, i = futures[fut]
                try:
                    series[kind][i] = fut.result()
                except Exception as e:  # noqa: BLE001 - log and keep going
                    print(f"add_series_to_grid: FAILED kind={kind} "
                          f"frame={frames[i]}: {e!r}")
                    failed.append((kind, frames[i], repr(e)))

    for kind in kinds:
        grid["models"][kind] = _to_jsonable(series[kind])
        n_ok = sum(1 for x in series[kind] if x is not None)
        n_failed = sum(1 for f in failed if f[0] == kind)
        print(f"add_series_to_grid: kind={kind}: {n_ok}/{len(frames)} frames "
              f"fit ({n_failed} failed, stored as null).")

    with gzip.open(grid_path, "wt") as f:
        json.dump(grid, f)
    return grid


def add_qr_to_grid(grid_path, workers):
    """Back-compat wrapper -- see ``add_series_to_grid`` (the QR-only path
    this used to implement directly is now one call into the generalized
    incremental-add function shared with spl/logi). Keeps the ``--add-qr``
    CLI and its existing test working unchanged."""
    return add_series_to_grid(grid_path, ["qr"], workers)


def continuity_scan(grid):
    """Report-only scan for overfit/aliased frames (Task-0 concerns C2/C3).

    For each ``ecfg_*`` model key, walks adjacent frames and evaluates each
    frame's median (via ``build_model_fn`` + the frame's own ``params``) on
    a shared reference t-grid. Flags a frame as suspect when its own
    ``r2 < 0.85`` OR its max abs median-log10 change vs the previous
    (non-null) frame exceeds 0.5.

    ``"bub"`` (BM) gets the SAME two checks. BM's ``comp_by_n[-1]``
    (composite USD price with all future bubbles) lives on a per-frame
    plotting grid (``t_grid``, which varies frame to frame), so it is
    first interpolated (``np.interp``) onto the shared reference t-grid
    and converted to log10 before the same adjacent-frame jump
    comparison is applied. BM's bubble decomposition (major/minor
    classification) can be just as degenerate near the sparse-data left
    edge as EPPL's oscillators -- this is the same C2/C3 concern, not a
    separate one, so BM is not exempt from the jump check.

    Never drops frames or raises -- purely informational, for a human to
    eyeball the printed lines.

    Parameters
    ----------
    grid : dict
        A grid object as produced by ``build_grid`` (or loaded back from
        the gzipped JSON file).

    Returns
    -------
    list[str]
        ``"SUSPECT <date> <key>: <reason>"`` lines, possibly empty.
    """
    frames = grid["frames"]
    models = grid["models"]
    ref_t_grid = np.logspace(np.log10(1.0), np.log10(20.0), 200)

    suspects = []

    for key, frame_list in models.items():
        if not key.startswith("ecfg_"):
            continue

        prev_vals = None
        prev_date = None
        for date, rec in zip(frames, frame_list):
            if rec is None:
                prev_vals = None
                prev_date = None
                continue

            r2 = rec.get("r2")
            if r2 is not None and r2 < _R2_THRESHOLD:
                suspects.append(
                    f"SUSPECT {date} {key}: r2={r2:.4f} < {_R2_THRESHOLD}")

            try:
                model_fn, param_names, _ = build_model_fn(
                    rec["n_log"], rec["n_cal"], rec["log_damps"], rec["cal_damps"])
                params_arr = [rec["params"][pn] for pn in param_names]
                vals = model_fn(ref_t_grid, *params_arr)
            except Exception as e:  # noqa: BLE001 - report, don't crash the scan
                suspects.append(
                    f"SUSPECT {date} {key}: median evaluation failed ({e!r})")
                prev_vals = None
                prev_date = None
                continue

            if prev_vals is not None:
                max_delta = float(np.max(np.abs(vals - prev_vals)))
                if max_delta > _JUMP_THRESHOLD:
                    suspects.append(
                        f"SUSPECT {date} {key}: max median-log10 change vs "
                        f"{prev_date} = {max_delta:.3f} > {_JUMP_THRESHOLD}")

            prev_vals = vals
            prev_date = date

    if "bub" in models:
        prev_vals = None
        prev_date = None
        for date, rec in zip(frames, models["bub"]):
            if rec is None:
                prev_vals = None
                prev_date = None
                continue

            r2 = rec.get("bm_r2")
            if r2 is not None and r2 < _R2_THRESHOLD:
                suspects.append(
                    f"SUSPECT {date} bub: bm_r2={r2:.4f} < {_R2_THRESHOLD}")

            try:
                median_usd = np.interp(ref_t_grid, rec["t_grid"], rec["comp_by_n"][-1])
                vals = np.log10(np.maximum(median_usd, 1e-300))
            except Exception as e:  # noqa: BLE001 - report, don't crash the scan
                suspects.append(
                    f"SUSPECT {date} bub: median evaluation failed ({e!r})")
                prev_vals = None
                prev_date = None
                continue

            if prev_vals is not None:
                max_delta = float(np.max(np.abs(vals - prev_vals)))
                if max_delta > _JUMP_THRESHOLD:
                    suspects.append(
                        f"SUSPECT {date} bub: max median-log10 change vs "
                        f"{prev_date} = {max_delta:.3f} > {_JUMP_THRESHOLD}")

            prev_vals = vals
            prev_date = date

    return suspects


def _last_full_month():
    """Start-of-month date for the most recently fully-elapsed month."""
    today = pd.Timestamp.today().normalize()
    current_month_start = today.replace(day=1)
    return (current_month_start - pd.Timedelta(days=1)).replace(day=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--full", action="store_true",
        help="Build the real grid: frame_dates(LEFT_EDGE_DATE, last full "
             "month), all_configs() (36 EPPL configs), include_bm=True, "
             "maxiter=EPPL_MAXITER. Long-running (documented, not run in "
             "CI/tests) -- lands in Task 5's integration.")
    ap.add_argument("--out", default=DEFAULT_OUT_PATH,
                     help=f"Output path (default: {DEFAULT_OUT_PATH})")
    ap.add_argument("--workers", type=int, default=None,
                     help="ProcessPoolExecutor workers "
                          "(default: os.cpu_count(), saturate the box -- "
                          "BLAS is pinned to 1 thread/process so this "
                          "doesn't oversubscribe)")
    ap.add_argument(
        "--downsample-existing", metavar="PATH", default=None,
        help="Post-process an ALREADY-BUILT grid file in place: downsample "
             "every 'bub' frame's comp_by_n/t_grid to --max-bm-points "
             "(no re-fitting, seconds not hours). Use instead of a full "
             "rebuild when only the BM plotting-grid resolution changed. "
             "Ignores --full/--out/--workers.")
    ap.add_argument("--max-bm-points", type=int, default=MAX_BM_POINTS,
                     help=f"Downsample cap for BM comp_by_n/t_grid arrays "
                          f"(default {MAX_BM_POINTS}).")
    ap.add_argument(
        "--add-qr", metavar="PATH", default=None,
        help="Incrementally add a 'qr' model series to an ALREADY-BUILT grid "
             "file in place (fits QR channels for its exact frames, parallel "
             "over --workers cores; minutes, not hours -- does NOT redo BM/"
             "EPPL). Use to add QR to a grid built before QR support. Honours "
             "--workers; ignores --full/--out. Equivalent to "
             "--add-models qr --out PATH (generalized below).")
    ap.add_argument(
        "--add-models", metavar="KINDS", default=None,
        help=f"Incrementally add one or more model series (comma-separated, "
             f"each in {sorted(_ASOF_FITTERS)}) to an ALREADY-BUILT grid file "
             f"(--out, default {DEFAULT_OUT_PATH}) in place -- parallel over "
             f"--workers cores; minutes, not hours -- does NOT redo BM/EPPL. "
             f"e.g. --add-models spl,logi. Honours --workers/--out; ignores "
             f"--full.")
    args = ap.parse_args()

    if args.downsample_existing:
        downsample_existing_grid(args.downsample_existing, max_pts=args.max_bm_points)
        return

    if args.add_qr:
        workers = args.workers or os.cpu_count() or 4
        print(f"--add-qr: fitting QR for {args.add_qr}, workers={workers}")
        add_qr_to_grid(args.add_qr, workers)
        return

    if args.add_models:
        kinds = [k.strip() for k in args.add_models.split(",") if k.strip()]
        invalid = sorted(set(kinds) - set(_ASOF_FITTERS))
        if invalid:
            ap.error(f"--add-models: unknown kind(s) {invalid}; choose from "
                      f"{sorted(_ASOF_FITTERS)}")
        workers = args.workers or os.cpu_count() or 4
        print(f"--add-models: fitting {kinds} for {args.out}, workers={workers}")
        add_series_to_grid(args.out, kinds, workers)
        return

    if not args.full:
        ap.print_help()
        print("\nNothing to do without --full or --downsample-existing. Use "
              "build_grid()/continuity_scan() directly (see btc_web/test_"
              "timemachine_grid_build.py) for a quick smoke build.")
        return

    last_full_month = _last_full_month().strftime("%Y-%m-%d")
    frames = frame_dates(LEFT_EDGE_DATE, last_full_month)
    configs = all_configs()
    workers = args.workers or os.cpu_count() or 4

    print(f"--full: {len(frames)} frames x {len(configs)} EPPL configs + BM + "
          f"QR + spl + logi, maxiter={EPPL_MAXITER}, workers={workers}, "
          f"out={args.out}")
    build_grid(frames=frames, configs=configs, include_bm=True, include_qr=True,
               include_spl=True, include_logi=True,
               out_path=args.out, maxiter=EPPL_MAXITER, workers=workers)


if __name__ == "__main__":
    main()
