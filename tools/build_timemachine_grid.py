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

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), ROOT]

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

from tools.timemachine.fit_eppl_asof import fit_config_asof  # noqa: E402
from tools.timemachine.fit_bm_asof import fit_bm_asof  # noqa: E402
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


# ─────────────────────────────────────────────────────────────────────────
# Module-level worker wrappers — required so ProcessPoolExecutor can pickle
# them (bound methods / closures are not picklable).
# ─────────────────────────────────────────────────────────────────────────
def _fit_ecfg_task(cfg, t, lp, ymax, maxiter):
    return fit_config_asof(cfg, t, lp, ymax, maxiter)


def _fit_bm_task(prices, ymax):
    return fit_bm_asof(prices, ymax)


def build_grid(frames, configs, include_bm, out_path, maxiter, workers):
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
        with ``max_workers = min(max(1, nproc - 2), 22)``.

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

    # Flat job list: (kind, key, frame_idx, frame_date, cfg_or_None)
    jobs = []
    for cfg, key in zip(configs, ecfg_key_list):
        for i, frame in enumerate(frames):
            jobs.append(("ecfg", key, i, frame, cfg))
    if include_bm:
        for i, frame in enumerate(frames):
            jobs.append(("bm", "bub", i, frame, None))

    failed = []

    if workers <= 1:
        for kind, key, i, frame, cfg in jobs:
            ymax = ymaxes[frame]
            try:
                if kind == "ecfg":
                    result = fit_config_asof(cfg, t, lp, ymax, maxiter)
                else:
                    result = fit_bm_asof(prices, ymax)
                models[key][i] = result
            except Exception as e:  # noqa: BLE001 - log and keep building
                print(f"build_grid: FAILED frame={frame} key={key}: {e!r}")
                failed.append((frame, key, repr(e)))
    else:
        max_workers = min(max(1, (os.cpu_count() or 4) - 2), 22)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for kind, key, i, frame, cfg in jobs:
                ymax = ymaxes[frame]
                if kind == "ecfg":
                    fut = ex.submit(_fit_ecfg_task, cfg, t, lp, ymax, maxiter)
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
                          "(default: min(max(1, nproc-2), 22))")
    args = ap.parse_args()

    if not args.full:
        ap.print_help()
        print("\nNothing to do without --full. Use build_grid()/"
              "continuity_scan() directly (see btc_web/test_timemachine_"
              "grid_build.py) for a quick smoke build.")
        return

    last_full_month = _last_full_month().strftime("%Y-%m-%d")
    frames = frame_dates(LEFT_EDGE_DATE, last_full_month)
    configs = all_configs()
    workers = args.workers or min(max(1, (os.cpu_count() or 4) - 2), 22)

    print(f"--full: {len(frames)} frames x {len(configs)} EPPL configs + BM, "
          f"maxiter={EPPL_MAXITER}, workers={workers}, out={args.out}")
    build_grid(frames=frames, configs=configs, include_bm=True,
               out_path=args.out, maxiter=EPPL_MAXITER, workers=workers)


if __name__ == "__main__":
    main()
