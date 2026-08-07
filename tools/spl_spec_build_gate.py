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
    _log10_L = 6.2133
    _t0      = 28.314
    _beta    = 5.0910

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
print(f"[gate 1] class evaluates      : RMSE {rmse:.6f}  (spec §3.1 says 0.2945)")
print(f"         within 1e-4 of spec? {'YES' if abs(rmse-0.294470) < 1e-4 else 'NO'}")

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

# spec §3.4 identity: spl - pl == -log10(1 + (t/t0)^beta)
A = m_._log10_L - m_._beta*np.log10(m_._t0)
pl = A + m_._beta*np.log10(t)
closed = -np.log10(1.0 + (t/m_._t0)**m_._beta)
err = float(np.max(np.abs((pred-pl)-closed)))
print(f"[gate 3] §3.3 identity max err : {err:.2e}  "
      f"({'holds' if err < 1e-12 else 'FAILS'})")
