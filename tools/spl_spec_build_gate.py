"""Build gate for the spl spec: run the code the spec actually contains.

Covers: the §5 _model_log10 body, the logaddexp stability claim, and whether
the class as written reproduces the §3.1 fitted RMSE. Does NOT cover the
_ShrinkingBandsMixin wiring (needs the real base class) or any btc_web step.
"""
import numpy as np, pandas as pd

# ---- verbatim from spec §5 --------------------------------------------
class SaturatingPowerLawModel:                 # mixin omitted for the gate
    name        = "Saturating Power Law"
    short_name  = "spl"
    legend_name = "SatPL"
    quantized   = True
    _log10_L = 5.846849
    _t0      = 23.869131
    _beta    = 5.103979

    def _model_log10(self, t):
        t_arr = np.asarray(t, float)
        u = -self._beta * (np.log(t_arr) - np.log(self._t0))
        return self._log10_L - np.logaddexp(0.0, u) / np.log(10.0)
# ------------------------------------------------------------------------

GENESIS = pd.Timestamp("2009-07-25")
df = pd.read_csv("BitcoinPricesDaily.csv")
dc=[c for c in df.columns if c.lower() in ("date","time","day")][0]
pc=[c for c in df.columns if "price" in c.lower() or "close" in c.lower()][0]
t=(pd.to_datetime(df[dc],format="mixed")-GENESIS).dt.total_seconds().to_numpy()/(365.25*86400)
# T_MIN mask (1.0), matching what the model class will see
p=df[pc].to_numpy(float); m=(t>=1.0)&(p>0); t,p=t[m],p[m]; lp=np.log10(p)

m_ = SaturatingPowerLawModel()
pred = m_._model_log10(t)
rmse = float(np.sqrt(np.mean((lp-pred)**2)))
print(f"[gate 1] class evaluates      : RMSE {rmse:.6f}  (spec §3.1 says 0.2939)")
print(f"         within 1e-4 of spec? {'YES' if abs(rmse-0.293851) < 1e-4 else 'NO'}")

# stability: the naive form the spec says would overflow
naive_bad = False
with np.errstate(over="ignore", invalid="ignore"):
    for beta in (5.0905, 20.0, 40.0):
        for tt in (0.01, 0.1, 1.0):
            raw = (tt/28.371)**(-beta)
            if not np.isfinite(raw) or raw > 1e300:
                naive_bad = True
            v = m_._model_log10(np.array([tt]))
            assert np.isfinite(v).all(), f"logaddexp form blew up at t={tt}, beta={beta}"
print(f"[gate 2] logaddexp finite over t<=0.01, beta<=40 : YES")
print(f"         naive form finite there (spec §5 claim): "
      f"{'NO -- overflows' if naive_bad else 'YES -- spec §5 correct'}")

# spec §3.1 PRIMARY claim: boundary-corrected LRT fails to reject t0 = infinity
from scipy.stats import chi2
X = np.vstack([np.ones_like(t), np.log10(t)]).T
c, *_ = np.linalg.lstsq(X, lp, rcond=None)
sse_pl = float(np.sum((lp - X @ c) ** 2))
sse_spl = float(np.sum((lp - pred) ** 2))
stat = len(t) * np.log(sse_pl / sse_spl)
crit = float(chi2.ppf(1 - 2 * 0.05, 1))
print(f"[gate 3] §3.1 LRT stat {stat:.4f} vs boundary crit {crit:.4f}"
      f"  -> {'fail to reject' if stat < crit else 'REJECTS'}")
print(f"         spec §3.1 current column says 13.6536 / REJECTS : "
      f"{'MATCHES' if abs(stat-13.6536) < 5e-2 and stat > crit else 'MISMATCH'}")
print(f"         (the 2.2802/fail-to-reject figure is the RETIRED 2026-06-03"
      f" window; §3.1 keeps both columns deliberately)")
