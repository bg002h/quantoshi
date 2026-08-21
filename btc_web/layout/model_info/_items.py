"""Accordion item builders for Model Info tab.

Returns the full list of 30 AccordionItem blocks in the documented display
order. Each item references the live coefficient-table helpers in ._helpers
so the rendered HTML always reflects the currently loaded model fits.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

from ._helpers import (
    _clickable_img,
    _coeff_table,
    _bm_rows,
    _ef_rows,
    _s2f_coeff_table,
    _hybppl_dd_rows,
    _hyb2l_coeff_table,
    _hyb2c_coeff_table,
    _hyb2b_coeff_table,
    _hyb4d_coeff_table,
    _pca_coeff_table,
    _pca_formula_table,
    _pca_variance_table,
    _pca_expanded_formula,
    _pca_basis_listing,
    _eppl_coeff_table,
    _gompertz_coeff_table,
    _plo_coeff_table,
    _sexp_coeff_table,
    _logi_coeff_table,
    _bpl_coeff_table,
    _grdy_coeff_table,
    _grdy_basis_table,
    _pl_coeff_table,
    _lppl_coeff_table,
    _lp2_coeff_table,
    _linppl_coeff_table,
    _hybppl_coeff_table,
    _exp_coeff_table,
    _spl_coeff_table,
    _spl_two_window_table,
    _spl_by_cutoff_table,
    _spl_cycle_phase_table,
    _spl_profile_table,
    _qr_table,
    _comparison_table,
    _regime_data_tables,
)
from colors import FALLBACK_MODEL_GRAY, UI_FONT_BASE, UI_FONT_LG


def _build_accordion_items():
    """Return the ordered list of 30 AccordionItems for the Model Info accordion."""
    return [

                        # ── 1. Bubble Model ──
                        dbc.AccordionItem([
                            html.H6("Overview"),
                            html.P([
                                "The Bubble Model combines quantile regression power-law channels "
                                "with a bubble composite overlay. The QR channels provide the "
                                "long-term structural framework; the bubble composite captures "
                                "cyclical deviations above support.",
                            ]),

                            html.H6("Bubble Composite"),
                            html.P([
                                "The gold composite line on Tab 1 fits a parameterized trapezoid shape "
                                "(rise, plateau, decay) to each historical Bitcoin bubble (2013, 2017, 2021, 2025) "
                                "in log-residual space above a power-law support line (bottom ~25% of data). "
                                "The composite is:"
                            ]),
                            dcc.Markdown(r"""
$$\text{composite}(t) = 10^{\,\log_{10}(\text{support}(t)) \;+\; \sum_{i} \text{bubble}_i(t)}$$

Future bubbles are extrapolated from the trend in historical bubble parameters (amplitude decreasing, width increasing).
                            """, mathjax=True),

                            html.H6("Support Line Fitting"),
                            html.P([
                                "The support (floor) line is fitted in three steps: ",
                                "(1) OLS regression on all price data, ",
                                "(2) filter to the bottom 20% of OLS residuals (the floor points), ",
                                "(3) quantile regression at Q50% (median) on those floor points. ",
                                "This yields slope=5.125, intercept=\u22121.559 in log\u2081\u2080 space.",
                            ]),

                            html.H6("Bubble Timing Extrapolation"),
                            html.P([
                                "Future bubble onset times are predicted by extrapolating the ",
                                html.Strong("linear trend in historical t\u1d63\u1d62\u209b\u2091 intervals"),
                                ". The BM model finds 5 major historical bubbles with rise-onset intervals of "
                                "2.1, 4.3, 2.3, and 4.0 years. The trend slope is +0.37 yr/cycle "
                                "(intervals lengthening), extrapolating to a 5.7-year gap and a next "
                                "onset around ",
                                html.Strong("~2029"),
                                ". Bubble amplitudes are simultaneously shrinking (K declining from "
                                "1.18 to 0.40), consistent with a maturing asset.",
                            ]),

                            html.H6("Sensitivity to Floor Choice"),
                            html.P([
                                "The support slope and intercept are the upstream parameters that determine "
                                "everything downstream: bubble amplitudes, intervals, and predicted onset. "
                                "A sensitivity sweep across slope (4\u20137) and intercept "
                                "(\u00b12.5) shows: ",
                            ]),
                            html.Ul([
                                html.Li([
                                    html.Strong("Broad stability region: "),
                                    "Below and left of the optimal R\u00b2 ridge, bubble timing and amplitude "
                                    "change smoothly and gradually. The BM fit sits in this stable zone.",
                                ]),
                                html.Li([
                                    html.Strong("Sharp instability above the ridge: "),
                                    "When the support line rises above most prices, log-excess goes negative, "
                                    "peak detection becomes noise-sensitive, and results are chaotic.",
                                ]),
                                html.Li([
                                    html.Strong("Onset timing is the most sensitive parameter: "),
                                    "Small shifts in support can change predicted onset by \u00b12 years, "
                                    "while amplitude and interval are more robust. "
                                    "This is because onset prediction depends on the ",
                                    html.Em("trend"),
                                    " in intervals (linear extrapolation), not the mean. "
                                    "BM and EF have nearly identical mean intervals (~3.2 yr) "
                                    "but their trends diverge: BM sees intervals lengthening "
                                    "(+0.37 yr/cycle \u2192 5.7 yr next), while EF sees them "
                                    "flattening (+0.20 yr/cycle \u2192 3.7 yr next). "
                                    "A single bubble\u2019s fitted t\u1d63\u1d62\u209b\u2091 shifting by "
                                    "0.8 years is enough to swing the prediction by ~2 years. "
                                    "The mean is robust; the trend (and therefore the prediction) is not.",
                                ]),
                                html.Li([
                                    html.Strong("The R\u00b2-optimal support is NOT the floor: "),
                                    "Maximum R\u00b2 occurs where the line fits the data mean, not the floor. "
                                    "Fitting the floor deliberately sacrifices R\u00b2 for a robust, stable baseline.",
                                ]),
                            ]),
                            html.P(
                                "A separate parameter sweep tests 49 combinations of the floor "
                                "percentile (5\u201335%) and quantile regression target (5\u201395%), "
                                "running the full bubble pipeline at each point. Results are broadly "
                                "stable across the grid, reinforcing that the model is not over-fitted "
                                "to one specific parameter choice."
                            ),
                            html.H6("Fitted Coefficients"),
                            _coeff_table(_bm_rows()),
                        ], title="Bubble Model", item_id="mi-bub"),

                        # ── 1b. Quantile Regression ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = \alpha_q + \beta_q \cdot \log_{10}(t)$$

Solved for price:

$$\text{price}(q,\, t) = 10^{\alpha_q} \cdot t^{\,\beta_q}$$

where $t$ = years since the optimal time origin (2009-07-25), and each quantile $q$ has its own independently fitted intercept $\alpha_q$ and slope $\beta_q$.
                            """, mathjax=True, className="mb-3"),

                            html.H6("Method"),
                            html.P([
                                "Quantile regression (",
                                html.Code("statsmodels.QuantReg"),
                                ") fits a separate power law at each percentile of the historical price "
                                "distribution. Unlike OLS (which minimizes squared residuals to find the mean), "
                                "QR minimizes asymmetrically weighted absolute residuals to find the conditional "
                                "quantile. Each line has its own slope \u2014 the lines are ",
                                html.Strong("not"),
                                " parallel in log-log space.",
                            ]),

                            html.H6("Fitted Coefficients"),
                            html.P("27 quantiles fitted to daily BTC prices from 2010-07-17 onward:"),
                            _qr_table(),
                        ], title="Quantile Regression", item_id="mi-qr"),

                        # ── 2. Power Law ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = (\alpha + z_q \cdot \sigma) + \beta \cdot \log_{10}(t)$$

Solved for price:

$$\text{price}(q,\, t) = 10^{\,\alpha + z_q \sigma} \cdot t^{\,\beta}$$

where $t$ = years since the optimal time origin (2009-07-25), $\alpha$ and $\beta$ are the OLS regression coefficients, $\sigma$ is the residual standard deviation, and $z_q = \Phi^{-1}(q)$ is the inverse normal CDF at quantile $q$.
                            """, mathjax=True, className="mb-3"),

                            html.H6("Method"),
                            html.P(
                                "Ordinary Least Squares regression fits a single line to the mean of "
                                "log\u2081\u2080(price) vs log\u2081\u2080(t). Quantile bands are created by "
                                "shifting this line by z\u00b7\u03c3, assuming Gaussian residuals. All bands "
                                "share the same slope \u2014 they are parallel lines in log-log space."
                            ),

                            html.H6("Fitted Coefficients"),
                            _pl_coeff_table(),
                            html.P(
                                "The slope means Bitcoin\u2019s price has historically grown as "
                                "t\u2075\u00b7\u2077 \u2014 roughly 5.7 orders of magnitude per order "
                                "of magnitude in time.",
                                className="text-muted small mt-2",
                            ),
                        ], title="Power Law (OLS)", item_id="mi-pl"),

                        # ── 3. LPPL ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A + B \cdot \log_{10}(t) + C \cdot t^{-D} \cdot \cos(\omega \cdot \ln(t) + \varphi) + z_q \cdot \sigma$$

Solved for price:

$$\text{price}(q,\, t) = 10^{\,A + z_q \sigma} \;\cdot\; t^{\,B} \;\cdot\; 10^{\,C \,\cdot\, t^{-D} \,\cdot\, \cos(\omega \ln t + \varphi)}$$

The first two terms are a standard power law. The third term adds **log-periodic oscillations** — sinusoidal in $\ln(t)$ — with amplitude that decays as $t^{-D}$. Quantile bands use Gaussian shift like PL.
                            """, mathjax=True, className="mb-3"),

                            html.H6("Method"),
                            html.P([
                                "Log-Periodic Power Law models were introduced by ",
                                html.A("Didier Sornette",
                                       href="https://en.wikipedia.org/wiki/Didier_Sornette",
                                       target="_blank", rel="noopener noreferrer"),
                                " for modeling financial bubbles. The classic LPPL uses a critical time "
                                "t\u2099 (singularity), but Quantoshi uses a ",
                                html.Strong("damped growth variant"),
                                " without a singularity \u2014 the oscillations decay naturally as t increases, "
                                "matching Bitcoin\u2019s empirically decreasing volatility over time. "
                                "Parameters are fitted via differential evolution (global optimization) "
                                "on all daily BTC prices from t \u2265 1 year.",
                            ]),

                            html.H6("Fitted Coefficients"),
                            _lppl_coeff_table(),

                            html.H6("Interpretation"),
                            html.Ul([
                                html.Li([
                                    html.Strong("\u03c9 \u2248 7.56: "),
                                    "The oscillation period in ln(t) space corresponds to ~4-year "
                                    "cycles, aligning with Bitcoin\u2019s halving schedule.",
                                ]),
                                html.Li([
                                    html.Strong("D \u2248 0.61: "),
                                    "Oscillation amplitude decays as t\u207b\u2070\u00b7\u2076. "
                                    "Each successive bubble is smaller relative to the trend \u2014 "
                                    "Bitcoin is getting less volatile over time.",
                                ]),
                                html.Li([
                                    html.Strong("B \u2248 5.08: "),
                                    "The underlying power-law slope matches QR and BM closely.",
                                ]),
                                html.Li([
                                    html.Strong("\u03c3 = 0.227: "),
                                    "Smaller than PL\u2019s residual \u03c3 because the oscillatory term "
                                    "absorbs variance that PL attributes to noise. This means LPPL\u2019s "
                                    "quantile bands are narrower.",
                                ]),
                            ]),
                        ], title="Log-Periodic Power Law (LPPL)", item_id="mi-lppl"),

                        # ── 3b. LPPL₂ ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A + B \cdot \log_{10}(t) + C_1 \cdot t^{-D} \cdot \cos(\omega_1 \cdot \ln(t) + \varphi_1) + C_2 \cdot \cos(\omega_2 \cdot \ln(t) + \varphi_2)$$

The first oscillation ($\omega_1$) is **damped** by $t^{-D}$ — bubble cycles that decay over time.
The second oscillation ($\omega_2$) is **undamped** — a permanent structural feature that persists
even as bubble amplitude shrinks. $\omega_2$ is **not constrained** to be a harmonic of $\omega_1$.
                            """, mathjax=True, className="mb-3"),

                            html.H6("Motivation"),
                            html.P([
                                "Bitcoin\u2019s price history shows evidence of two distinct oscillation "
                                "frequencies: the well-known ~4-year halving cycle (captured by LPPL\u2019s "
                                "single cosine term) and a secondary oscillation that doesn\u2019t fit "
                                "neatly as a harmonic. An initial attempt with the second frequency "
                                "locked to 2\u00d7\u03c9\u2081 (Weierstrass-type) showed minimal improvement "
                                "(R\u00b2 \u0394=+0.0009). Releasing \u03c9\u2082 as a free parameter and "
                                "moving the second term outside the damping envelope yielded a much "
                                "larger improvement (R\u00b2 \u0394=+0.0060), confirming the secondary "
                                "oscillation is real, not a perfect harmonic, and not decaying.",
                            ]),

                            html.H6("Fitted Coefficients"),
                            _lp2_coeff_table(),

                            html.H6("Key Findings"),
                            html.Ul([
                                html.Li([
                                    html.Strong("\u03c9\u2082/\u03c9\u2081 = 2.83: "),
                                    "The secondary frequency is ~2.83\u00d7 the primary \u2014 close to "
                                    "but not exactly a 3\u00d7 harmonic. When constrained to exactly "
                                    "2\u00d7, the fit barely improved, confirming the real frequency "
                                    "isn\u2019t a simple integer multiple.",
                                ]),
                                html.Li([
                                    html.Strong("C\u2082/C\u2081 = 24%: "),
                                    "The secondary oscillation has about a quarter of the primary\u2019s "
                                    "amplitude, but because it\u2019s undamped it becomes increasingly "
                                    "dominant in future projections as the primary decays.",
                                ]),
                                html.Li([
                                    html.Strong("\u03c3 = 0.193 (vs LPPL 0.227): "),
                                    "15% tighter residuals. The secondary term absorbs real structure "
                                    "that single-frequency LPPL treats as noise.",
                                ]),
                                html.Li([
                                    html.Strong("D = 0.57 (vs LPPL 0.61): "),
                                    "Less damping needed for the primary when the secondary carries "
                                    "some of the oscillatory load.",
                                ]),
                            ]),

                            html.H6("Physical Interpretation"),
                            html.P([
                                "The damped primary cycle captures the halving-driven boom/bust pattern "
                                "that is empirically shrinking with each cycle. The undamped secondary "
                                "oscillation may reflect a persistent structural feature of Bitcoin\u2019s "
                                "market microstructure \u2014 perhaps related to mid-cycle rallies, "
                                "accumulation phases, or a different class of market participants "
                                "operating on a different timescale. Its persistence implies this "
                                "oscillation will continue even as the major bubble cycles converge "
                                "toward the power-law trend.",
                            ]),

                            html.H6("Refitting"),
                            html.P(
                                "All 9 parameters are refitted daily via differential evolution "
                                "as part of the price update pipeline. The primary parameters "
                                "(A, B, C\u2081, \u03c9\u2081, \u03c6\u2081, D) are inherited from LPPL "
                                "as starting bounds; \u03c9\u2082 is free to find any frequency between "
                                "1 and 30 in log-time."
                            ),
                        ], title="LPPL\u2082 (Two-Frequency)", item_id="mi-lp2"),

                        # ── 3c. LPPL weighting / regime shift story ──
                        dbc.AccordionItem([
                            html.H6("The Non-Uniform Sampling Problem"),
                            html.P([
                                "Bitcoin trades 24/7 and we have exactly one price point per calendar day. "
                                "But LPPL models oscillations in ",
                                html.Em("log-time"),
                                ", and daily-uniform sampling is ",
                                html.Strong("non-uniform in log-time"),
                                ": at t=1 yr we have ~730 samples per unit of ln(t); at t=16 yr we have "
                                "~5,840 — about 8\u00d7 denser. The standard least-squares fit therefore "
                                "over-weights recent years by roughly that ratio.",
                            ]),

                            html.H6("Systematic Effect on the Damping Exponent D"),
                            html.P([
                                "Refitting each LPPL variant with log-time-uniform weighting "
                                "(residuals weighted by 1/t) shifts the damping exponent dramatically:"
                            ]),
                            _coeff_table([
                                ("LPPL D (unweighted)", "0.61"),
                                ("LPPL D (log-time weighted)", "0.36"),
                                ("LPPL\u2082 D (unweighted)", "0.57"),
                                ("LPPL\u2082 D (log-time weighted)", "0.30"),
                                ("LPPL\u2083 D (unweighted)", "0.37"),
                                ("LPPL\u2083 D (log-time weighted)", "0.25"),
                            ]),
                            html.P([
                                "Every variant shows D reduced by 30\u201347% under weighted fitting. "
                                "The narrative that ",
                                html.Em("\u201cBitcoin's bubble cycles are dramatically shrinking\u201d"),
                                " is ",
                                html.Strong("partly an artifact"),
                                " of the 2020\u20132024 cycle being larger in absolute log-price terms AND "
                                "over-weighted by the uniform-in-calendar-time sampling. Under a uniform-in-log-time "
                                "weighting each successive cycle is smaller than the previous, but by less than "
                                "the unweighted fit suggests.",
                            ]),

                            html.H6("Regime Shift: the Secondary Frequency Flips"),
                            html.P([
                                "For LPPL\u2082, the fitted secondary frequency ",
                                html.Strong("depends strongly on weighting"),
                                ":"
                            ]),
                            _coeff_table([
                                ("LPPL\u2082 \u03c9\u2082 unweighted", "20.90 (ratio 2.83 \u2248 3\u00d7 W\u2081)"),
                                ("LPPL\u2082 \u03c9\u2082 weighted",  "9.26 (ratio 1.35, non-harmonic)"),
                            ]),
                            html.P([
                                "When LPPL\u2082 is forced to pick only ONE secondary oscillation, "
                                "the two weightings choose completely different frequencies. The data "
                                "actually contains BOTH oscillations \u2014 the weighting just determines "
                                "which one dominates the fit.",
                            ]),

                            html.H6("What This Means: A Regime Shift Around 2017\u20132020"),
                            html.P([
                                "Bitcoin's market structure changed materially during 2017 (retail boom, "
                                "CME futures launch) and 2020\u20132021 (institutional adoption, spot ETFs). "
                                "The 2010\u20132019 era and the 2020\u20132025 era have ",
                                html.Strong("different oscillation spectra"),
                                ". LPPL\u2083 is rich enough to capture both simultaneously "
                                "(\u03c9\u22489, \u03c9\u224821 are both present regardless of weighting), but "
                                "LPPL and LPPL\u2082 are forced to pick the dominant frequency for whichever "
                                "era the fit weights more heavily. ",
                                html.Strong("Neither weighting is \u201cwrong\u201d"),
                                " \u2014 they answer different questions:",
                            ]),
                            html.Ul([
                                html.Li([
                                    html.Strong("Unweighted fit: "),
                                    "\u201cWhat's the dominant structure in the data that I have?\u201d "
                                    "(recent-era focused, good for near-term prediction)"
                                ]),
                                html.Li([
                                    html.Strong("Weighted fit: "),
                                    "\u201cWhat's the universal log-periodic structure across Bitcoin's entire history?\u201d "
                                    "(matches LPPL's theoretical framing, treats all cycles equally)"
                                ]),
                            ]),

                            html.H6("Current Quantoshi Fits"),
                            html.P([
                                "Quantoshi currently uses ",
                                html.Strong("unweighted fits"),
                                " for all LPPL variants because they match the recent market "
                                "regime that users are most likely to project from. The displayed "
                                "damping D values, secondary frequencies, and forward projections "
                                "should be read with this caveat in mind. The LPPL\u2083 fit captures "
                                "both regime structures simultaneously and is the least sensitive "
                                "to this weighting choice.",
                            ]),

                            html.H6("Related: Regime Shift Detection"),
                            html.P([
                                "Sliding-window LPPL fits (e.g., 5-year windows stepped monthly) "
                                "can detect regime shifts by tracking how W, D, and residual "
                                "variance evolve over time. Sudden parameter jumps flag structural "
                                "breaks. Complements existing Bai-Perron / CUSUM analyses "
                                "(see FAQ) by extending the idea from power-law slope to "
                                "oscillation parameters.",
                            ]),

                            html.Hr(),
                            html.H6("Why LPPL\u2083 (ω\u22487, 9, 21) is the most physically honest model"),
                            html.P([
                                "Across ALL fitting configurations we tested (weighted/unweighted, "
                                "with/without \u03c9=13 excluded, LPPL\u2081 through LPPL\u2084), exactly ",
                                html.Strong("three"),
                                " oscillation frequencies appear consistently:",
                            ]),
                            _coeff_table([
                                ("\u03c9 \u2248 7",  "Primary halving cycle (damped, W\u2081)"),
                                ("\u03c9 \u2248 9",  "Genuine non-harmonic secondary (W\u2083 in LPPL\u2083)"),
                                ("\u03c9 \u2248 21", "Either 3\u00d7W\u2081 harmonic or distinct structural mode (W\u2082)"),
                            ]),
                            html.P([
                                "These three frequencies ARE present in LPPL\u2083's fit regardless of "
                                "weighting choice. They're not artifacts of recent-data over-weighting. "
                                "They're not intermodulation products of each other. They represent the ",
                                html.Strong("genuine log-periodic structure"),
                                " in Bitcoin's price history.",
                            ]),
                            html.P([
                                "LPPL\u2082 (2 frequencies) has to discard one of these — the weighting "
                                "choice determines which. LPPL\u2084 (4 frequencies) adds a 4th that is ",
                                html.Em("not"),
                                " robust across constraints (see below). LPPL\u2083 is the Goldilocks fit: "
                                "rich enough to capture all genuine structure, disciplined enough to "
                                "not chase intermod artifacts.",
                            ]),
                            html.P([
                                html.Strong("Default recommendation: "),
                                "LPPL\u2083 is now the default in the LPPL Models config panel.",
                            ]),

                            html.Hr(),
                            html.H6("\u26A0 Why LPPL\u2084 is probably NOT that smart"),
                            html.P([
                                "When you enable LPPL\u2084 in the config panel, a warning fires. "
                                "Here's why: only ",
                                html.Strong("three"),
                                " frequencies are consistently present across ALL fitting constraints "
                                "(weighted, unweighted, with/without \u03c9=13 excluded):",
                            ]),
                            html.Ul([
                                html.Li("\u03c9 \u2248 7  — primary halving cycle (damped)"),
                                html.Li("\u03c9 \u2248 9  — genuine non-harmonic secondary"),
                                html.Li("\u03c9 \u2248 21 — either 3\u00d7W\u2081 harmonic or genuine 3rd frequency"),
                            ]),
                            html.P([
                                "LPPL\u2084's fourth frequency isn't stable. Under different weightings "
                                "it appears at ~13 or ~17, and each of these can be explained as an ",
                                html.Strong("intermodulation product"),
                                " of the 3 stable frequencies:",
                            ]),
                            html.Ul([
                                html.Li("\u03c9\u224813 \u2248 W\u2082 − W\u2081 = 20.9 − 7.4 = 13.5"),
                                html.Li("\u03c9\u224817 \u2248 W\u2081 + W\u2083 = 7.1 + 9.9 = 17.0"),
                            ]),
                            html.P([
                                "When we exclude the \u03c9=13 band via the \u201cExclude \u03c9\u224813 "
                                "intermod\u201d toggle, the optimizer just migrates to the \u03c9\u224817 "
                                "pocket instead \u2014 suggesting LPPL\u2084's 4th frequency is ",
                                html.Strong("fundamentally an artifact"),
                                " rather than a real structural oscillation in Bitcoin's price data. ",
                                "LPPL\u2083 captures the genuine signal; LPPL\u2084 adds cosmetic complexity.",
                            ]),
                            html.P([
                                html.Strong("Bottom line: "),
                                "Use LPPL\u2083 (or LPPL\u2082) for physically defensible fits. "
                                "LPPL\u2084 is available in the config panel for comparison but "
                                "should be read as \u201cdemonstrates overfitting\u201d not "
                                "\u201creveals 4th oscillation.\u201d",
                            ]),
                        ], title="LPPL Weighting & Regime Shifts", item_id="mi-lppl-weighting"),

                        # ── 3d. LinPPL ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A + B\log_{10}(t) + C \cdot t^{-D}\cos(\omega_{\text{cal}} \cdot t + \varphi)$$

The oscillation is in **calendar time** ($\omega_{\text{cal}} \cdot t$), not log-time. The period
$T = 2\pi/\omega_{\text{cal}}$ stays constant in calendar years.
                            """, mathjax=True, className="mb-3"),

                            html.H6("Motivation"),
                            html.P([
                                "LPPL assumes log-periodicity \u2014 successive cycles scale by "
                                "a constant ratio in ln(t). But Bitcoin's halving cycle is "
                                "approximately constant at 4 calendar years, which is ",
                                html.Strong("linear-periodic"),
                                ", not log-periodic. LinPPL replaces LPPL's ",
                                html.Code("cos(\u03c9\u00b7ln t)"),
                                " with ",
                                html.Code("cos(\u03c9_cal\u00b7t)"),
                                " \u2014 a direct calendar-time oscillation matching the "
                                "halving rhythm.",
                            ]),

                            html.H6("Fitted Coefficients (full history)"),
                            _linppl_coeff_table(),

                            html.H6("Key Findings"),
                            html.Ul([
                                html.Li([
                                    html.Strong("T = 3.56 years: "),
                                    "close to Bitcoin's ~4-year halving cycle, "
                                    "confirming the calendar-periodic hypothesis.",
                                ]),
                                html.Li([
                                    html.Strong("D hits the lower bound (0.01): "),
                                    "the fit wants NO damping on the calendar cycle. "
                                    "Bitcoin's halving-driven oscillations don't shrink "
                                    "over time (in log-price terms).",
                                ]),
                                html.Li([
                                    html.Strong("R\u00b2 barely beats LPPL (+0.0009): "),
                                    "globally, LinPPL is nearly equivalent to LPPL. The "
                                    "calendar-time vs log-time choice doesn't resolve "
                                    "the fit quality question in isolation.",
                                ]),
                                html.Li([
                                    html.Strong("Rolling-window T stays at 3-4.5 yr: "),
                                    "across all 5-year rolling windows, T anchors to the "
                                    "halving cycle more stably than LPPL\u2081's W does "
                                    "(see /E for the evolution).",
                                ]),
                            ]),

                            html.H6("Limits"),
                            html.P([
                                "LinPPL doesn't capture Bitcoin's early-era (2010-2015) ",
                                html.Em("self-similarity"),
                                " \u2014 the pattern that motivated LPPL in the first place. "
                                "The next model, HybPPL, combines both oscillation types.",
                            ]),
                        ], title="LinPPL (Linear-Periodic PPL)", item_id="mi-linppl"),

                        # ── 3e. HybPPL ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A + B\log_{10}(t) + C_1 t^{-D} \cos(\omega_{\text{log}} \ln t + \varphi_1) + C_2 \cos(\omega_{\text{cal}} t + \varphi_2)$$

Two oscillation terms:
- **Log-periodic damped** ($\omega_{\text{log}} \cdot \ln t$) — inherits LPPL's framing, captures early-era self-similarity
- **Linear-periodic undamped** ($\omega_{\text{cal}} \cdot t$) — captures the halving cycle in calendar time
                            """, mathjax=True, className="mb-3"),

                            html.H6("Motivation"),
                            html.P([
                                "Neither LPPL nor LinPPL fits Bitcoin cleanly \u2014 LPPL's "
                                "log-periodicity misrepresents the halving cycle's calendar-time "
                                "consistency, while LinPPL ignores the early-era self-similar "
                                "scaling. HybPPL combines both: one term for each type of "
                                "oscillation.",
                            ]),

                            html.H6("Fitted Coefficients (full history)"),
                            _hybppl_coeff_table(),

                            html.H6("Why HybPPL is the current best-fit"),
                            html.Ul([
                                html.Li([
                                    html.Strong("R\u00b2 = 0.9889 beats LPPL\u2082 (0.9840) "
                                                "at the same 9-param count"),
                                    " \u2014 \u0394 R\u00b2 = +0.005, 2\u00d7 the gain "
                                    "LPPL\u2082 had over LPPL.",
                                ]),
                                html.Li([
                                    html.Strong("Ties LPPL\u2083's R\u00b2 (0.9889) with 3 "
                                                "fewer parameters"),
                                    " \u2014 decomposing Bitcoin's structure into one log-periodic "
                                    "+ one calendar-periodic term is more efficient than three "
                                    "log-periodic terms.",
                                ]),
                                html.Li([
                                    html.Strong("Both oscillations are meaningful: "),
                                    "C\u2081 \u2248 0.69 (log), C\u2082 \u2248 0.23 (calendar). "
                                    "The calendar cycle contributes ~1/3 the amplitude of the "
                                    "log-periodic cycle \u2014 not dominant, not negligible.",
                                ]),
                                html.Li([
                                    html.Strong("Rolling-window T_cal is rock stable: "),
                                    "3.0-3.9 yr across all windows, vs LPPL\u2082's W\u2081 which "
                                    "swings from 15 to 40. Confirms the halving cycle as a "
                                    "persistent feature independent of fitting window.",
                                ]),
                            ]),

                            html.H6("Interpretation"),
                            html.P([
                                "Bitcoin's price dynamics appear to have ",
                                html.Strong("two distinct oscillation mechanisms"),
                                ":",
                            ]),
                            html.Ol([
                                html.Li([
                                    html.Strong("Log-periodic market structure cycle"),
                                    " \u2014 self-similar scaling in early Bitcoin "
                                    "(2010-2015), possibly driven by adoption dynamics or "
                                    "market-maturing effects. This term is damped "
                                    "(D=0.71) \u2014 it shrinks over time."
                                ]),
                                html.Li([
                                    html.Strong("Linear-periodic halving cycle"),
                                    " \u2014 locked to Bitcoin's 4-year block reward schedule. "
                                    "This term is undamped \u2014 halving cycles don't "
                                    "diminish as the asset matures.",
                                ]),
                            ]),
                            html.P([
                                "The damped log-periodic component is fading, but the "
                                "linear-periodic halving cycle persists. HybPPL captures "
                                "this distinction cleanly, which is why it outperforms "
                                "LPPL variants at equal parameter counts.",
                            ]),

                            html.H6("Caveats"),
                            html.Ul([
                                html.Li("Fit per-window, HybPPL is roughly tied with LPPL\u2082 "
                                        "(54% of 129 windows). Its advantage comes from the "
                                        "full-history fit."),
                                html.Li("Like all LPPL-family models, parameters drift daily "
                                        "with new price data. Refitted in update_prices.py."),
                                html.Li("The log-periodic damping D=0.71 is strong \u2014 by "
                                        "2035 (t\u224826), C\u2081\u00b7t\u207b\u1d30 will be "
                                        "~0.08 (vs C\u2082=0.23). The calendar cycle will "
                                        "dominate forward projections."),
                            ]),
                        ], title="Log-Periodic + Halving Cycle", item_id="mi-hybppl"),

                        # ── 3f. HybPPL (DD — Double Damped) ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A + B\log_{10}(t)
  + C_1\,t^{-D_1}\cos\!\big(\omega_{\log}\ln t + \phi_1\big)
  + C_2\,t^{-D_2}\cos\!\big(\omega_{\text{cal}}\,t + \phi_2\big)$$

Solved for price:

$$\text{price}(t) = 10^{A} \cdot t^{B}
  \cdot 10^{\,C_1 t^{-D_1} \cos(\omega_{\log} \ln t + \varphi_1)}
  \cdot 10^{\,C_2 t^{-D_2} \cos(\omega_{\text{cal}} t + \varphi_2)}$$
                            """, mathjax=True, className="mb-3"),

                            html.H6("Motivation"),
                            html.P(
                                "HybPPL (DD) is the double-damped variant of HybPPL. Like HybPPL it "
                                "co-fits the power-law trend (A, B) alongside both oscillators, but "
                                "gives the calendar-periodic (halving cycle) oscillator its own damping "
                                "exponent D\u2082. This tests whether the ~4-year halving "
                                "cycle is a permanent feature of Bitcoin\u2019s price dynamics or whether "
                                "it too is decaying over time as Bitcoin matures. If D\u2082 converges "
                                "near zero, the data does not support calendar damping \u2014 the halving "
                                "cycle appears permanent."
                            ),

                            html.H6("Fitted Coefficients"),
                            html.P(
                                "All 10 parameters are co-fit via differential evolution "
                                "and refit daily via fit_hybppl_dd.py.",
                                className="text-muted small",
                            ),
                            _coeff_table(_hybppl_dd_rows()),

                            html.H6("Interpretation"),
                            html.P([
                                "D\u2082 \u2248 0.001 suggests the calendar oscillator is effectively "
                                "undamped \u2014 the halving cycle appears permanent. The extra parameter "
                                "does not meaningfully improve the fit, confirming that HybPPL "
                                "with its undamped calendar term is the more parsimonious choice. This "
                                "model exists primarily as a diagnostic: if D\u2082 ever drifts "
                                "significantly above zero in future refits, it would signal that the "
                                "halving cycle is beginning to fade.",
                            ]),

                            html.H6("What the two D\u2019s tell us"),
                            html.P(
                                "Early Bitcoin\u2019s price didn\u2019t follow a clean power law \u2014 "
                                "it oscillated wildly around the trend as the market discovered what "
                                "Bitcoin was worth. The damped log-periodic term (D\u2081 \u2248 0.71) "
                                "absorbs those early deviations. By t \u2248 25 (~2035), the damped "
                                "amplitude shrinks to C\u2081\u00b725\u207b\u2070\u00b7\u2077\u2071 "
                                "\u2248 0.07 in log\u2081\u2080 \u2014 a \u00b117% modulation, down "
                                "from \u00b1400% in the early years. The log-periodic oscillation is "
                                "essentially gone."
                            ),
                            html.P(
                                "What remains is the calendar cycle (D\u2082 \u2248 0, undamped): a "
                                "fixed-period ~3.6-year oscillation tied to the halving schedule. This "
                                "one doesn\u2019t decay because it has a physical cause (supply shock "
                                "every ~4 years) rather than being a transient artifact of price "
                                "discovery in a nascent market."
                            ),
                            html.P([
                                html.Strong("Nuance: "),
                                "the exponent B itself is not unstable early on \u2014 B is constant "
                                "in the model. Rather, the ",
                                html.Em("residuals around B"),
                                " were large and oscillatory in the early era. The damped term "
                                "explains why they were large (self-similar bubble dynamics in a new "
                                "asset class) and why they shrink (market maturation). The DD model\u2019s "
                                "D\u2082 \u2248 0 result is the quantitative confirmation that the halving "
                                "cycle is a different beast \u2014 structural, not transient.",
                            ]),

                            html.P(
                                "Stop staring at the Double D\u2019s \u2014 there\u2019s nothing to see. "
                                "D\u2082 is basically zero.",
                                style={"fontStyle": "italic", "color": FALLBACK_MODEL_GRAY,
                                       "fontSize": UI_FONT_BASE},
                            ),

                            html.P(
                                "Refitted daily via tools/fit_hybppl_dd.py along with the rest of "
                                "the model pipeline in update_prices.py.",
                                className="text-muted small",
                            ),
                        ], title="Log-Periodic + Halving Cycle (both decaying)", item_id="mi-hybppl-dd"),

                        # ── 3f. HybPPL +2nd Log ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A + B \log_{10}(t)
+ C_1 t^{-D_1} \cos(\omega_1 \ln t + \varphi_1)
+ C_2 \cos(\omega_c t + \varphi_2)
+ C_3 t^{-D_2} \cos(\omega_2 \ln t + \varphi_3)$$
                            """, mathjax=True, className="mb-3"),
                            html.H6("Description"),
                            html.P("Adds a second damped log-periodic harmonic to HybPPL. "
                                   "The second frequency (\u03c9\u2082 \u2248 16) is roughly "
                                   "double the primary (\u03c9\u2081 \u2248 7.5). 13 parameters."),
                            html.H6("Damping Behavior"),
                            html.P([
                                html.Strong("Log osc 1"), " (D\u2081 \u2248 0.77): ",
                                "Currently producing \u00b10.09 log\u2081\u2080 swings (\u00b123% price). "
                                "By 2035 this drops to \u00b10.06 (\u00b115%) \u2014 losing ~31% of its "
                                "amplitude per decade. Still visible but fading.",
                            ]),
                            html.P([
                                html.Strong("Cal osc"), " (undamped): ",
                                "Fixed \u00b10.26 log\u2081\u2080 swing (\u00b180% price) on a 3.6-year cycle. "
                                "This is the halving cycle \u2014 it persists indefinitely and will "
                                "dominate forward projections as the log-periodic terms decay.",
                            ]),
                            html.P([
                                html.Strong("Log osc 2"), " (D\u2082 \u2248 0.93): ",
                                "Currently only \u00b10.03 log\u2081\u2080 (\u00b17% price) and fading 36% "
                                "by 2035. This higher harmonic is already near-negligible \u2014 "
                                "it captures fine structure in early Bitcoin but contributes little "
                                "to forward projections.",
                            ]),
                            html.H6("Fitted Coefficients"),
                            _hyb2l_coeff_table(),
                        ], title="Log-Periodic + Halving + 2nd Log Harmonic", item_id="mi-hyb2l"),

                        # ── 3g. HybPPL +2nd Cal ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A + B \log_{10}(t)
+ C_1 t^{-D} \cos(\omega_1 \ln t + \varphi_1)
+ C_2 \cos(\omega_{c1} t + \varphi_2)
+ C_3 \cos(\omega_{c2} t + \varphi_3)$$
                            """, mathjax=True, className="mb-3"),
                            html.H6("Description"),
                            html.P("Adds a second undamped calendar-periodic term to HybPPL. "
                                   "The second frequency (T \u2248 1.9yr) is roughly half the "
                                   "halving cycle \u2014 may capture sub-halving market structure. "
                                   "Best R\u00b2 improvement per added parameter. 12 parameters."),
                            html.H6("Damping Behavior"),
                            html.P([
                                html.Strong("Log osc"), " (D \u2248 0.73): ",
                                "Currently \u00b10.10 log\u2081\u2080 (\u00b126% price). "
                                "By 2035 this drops to \u00b10.07 (\u00b117%) \u2014 losing ~30% per decade. "
                                "The slowest-decaying log-periodic term in the family.",
                            ]),
                            html.P([
                                html.Strong("Cal osc 1"), " (T \u2248 3.6yr, undamped): ",
                                "Fixed \u00b10.24 log\u2081\u2080 (\u00b170% price). The halving cycle \u2014 "
                                "persists indefinitely.",
                            ]),
                            html.P([
                                html.Strong("Cal osc 2"), " (T \u2248 1.9yr, undamped): ",
                                "Fixed \u00b10.11 log\u2081\u2080 (\u00b130% price). A sub-halving rhythm "
                                "at roughly half the halving period. Also persists indefinitely. "
                                "Together with cal osc 1, these two undamped terms fully determine "
                                "the model\u2019s long-term oscillatory behavior.",
                            ]),
                            html.H6("Fitted Coefficients"),
                            _hyb2c_coeff_table(),
                        ], title="Log-Periodic + Halving + Sub-Halving", item_id="mi-hyb2c"),

                        # ── 3h. HybPPL +Both ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A + B \log_{10}(t)
+ C_1 t^{-D_1} \cos(\omega_{l1} \ln t + \varphi_1)
+ C_2 \cos(\omega_{c1} t + \varphi_2)
+ C_3 t^{-D_2} \cos(\omega_{l2} \ln t + \varphi_3)
+ C_4 \cos(\omega_{c2} t + \varphi_4)$$
                            """, mathjax=True, className="mb-3"),
                            html.H6("Description"),
                            html.P("Full second-frequency model: both log-periodic and "
                                   "calendar-periodic get a second harmonic. Highest R\u00b2 "
                                   "in the family (0.993). BIC still improves despite 16 "
                                   "parameters \u2014 both frequencies are statistically justified."),
                            html.H6("Damping Behavior"),
                            html.P([
                                html.Strong("Log osc 1"), " (D\u2081 \u2248 0.83): ",
                                "Currently \u00b10.09 log\u2081\u2080 (\u00b123% price). "
                                "By 2035 drops to \u00b10.06 (\u00b115%) \u2014 losing ~33% per decade.",
                            ]),
                            html.P([
                                html.Strong("Cal osc 1"), " (T \u2248 3.6yr, undamped): ",
                                "Fixed \u00b10.24 log\u2081\u2080 (\u00b170% price). The halving cycle.",
                            ]),
                            html.P([
                                html.Strong("Log osc 2"), " (D\u2082 \u2248 1.17): ",
                                "Currently only \u00b10.017 log\u2081\u2080 (\u00b14% price) and fading 43% "
                                "by 2035. The fastest-decaying term in the model \u2014 essentially "
                                "extinct by the 2040s. Captures early-era fractal structure only.",
                            ]),
                            html.P([
                                html.Strong("Cal osc 2"), " (T \u2248 1.9yr, undamped): ",
                                "Fixed \u00b10.11 log\u2081\u2080 (\u00b126% price). Sub-halving rhythm. "
                                "Together with the halving cycle, these two undamped terms "
                                "dominate all forward projections beyond ~2035.",
                            ]),
                            html.H6("Fitted Coefficients"),
                            _hyb2b_coeff_table(),
                        ], title="Full Hybrid (2 log + 2 calendar)", item_id="mi-hyb2b"),

                        # ── 3i. HybPPL 4D (all damped) ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A + B \log_{10}(t)
+ C_1 t^{-D_1} \cos(\omega_{l1} \ln t + \varphi_1)
+ C_2 t^{-D_{c1}} \cos(\omega_{c1} t + \varphi_2)
+ C_3 t^{-D_2} \cos(\omega_{l2} \ln t + \varphi_3)
+ C_4 t^{-D_{c2}} \cos(\omega_{c2} t + \varphi_4)$$
                            """, mathjax=True, className="mb-3"),
                            html.H6("Description"),
                            html.P(
                                "The \u201call damped\u201d variant of HybPPL +2B: every oscillatory "
                                "component carries its own damping exponent t^(-D). "
                                "18 parameters (2 more than Hyb2B). This model tests whether "
                                "the calendar-periodic terms should also decay over time."
                            ),
                            html.H6("Why This Model Is Informative But Not Good", className="text-danger"),
                            html.Ul([
                                html.Li([
                                    html.Strong("R\u00b2 = 0.992 vs Hyb2B\u2019s 0.993"), " \u2014 adding 2 extra D "
                                    "parameters made the fit ", html.Em("worse"), ", not better."
                                ]),
                                html.Li([
                                    html.Strong("BIC = \u221222,624 vs Hyb2B\u2019s \u221223,203"), " \u2014 BIC penalizes "
                                    "the extra complexity, and the penalty far exceeds any SSE improvement."
                                ]),
                                html.Li([
                                    html.Strong("D"), html.Sub("c2"), html.Strong(" \u2248 0.076 (near zero)"),
                                    " \u2014 the 2nd calendar term ", html.Em("wants"), " to be undamped. "
                                    "The optimizer can\u2019t push D", html.Sub("c2"), " to zero because "
                                    "the parameterization forces D > 0, so it gets as close as possible."
                                ]),
                                html.Li([
                                    html.Strong("D"), html.Sub("c1"), html.Strong(" \u2248 1.06 distorted the halving cycle"),
                                    " \u2014 to accommodate damping on the primary calendar term, "
                                    "the optimizer shifted \u03c9", html.Sub("c1"),
                                    " from 1.74 rad/yr (T \u2248 3.6 yr) to 1.12 rad/yr (T \u2248 5.6 yr). "
                                    "This is no longer the halving cycle \u2014 it\u2019s a distorted artifact."
                                ]),
                            ]),
                            html.P([
                                html.Strong("Conclusion: "),
                                "Calendar-periodic terms should remain undamped. The Bitcoin halving "
                                "cycle is a fundamental protocol feature that does not decay. Forcing "
                                "damping on calendar terms degrades both fit quality and physical "
                                "interpretability. Hyb2B (undamped calendar terms) remains the "
                                "superior specification."
                            ]),
                            html.H6("Damping Behavior"),
                            html.P([
                                html.Strong("Log osc 1"), " (D\u2081 \u2248 0.59): ",
                                "Currently \u00b10.08 log\u2081\u2080 (\u00b120% price). "
                                "The slowest-decaying log-periodic term \u2014 loses ~25% per decade. "
                                "Still contributes meaningfully through the 2040s.",
                            ]),
                            html.P([
                                html.Strong("Cal osc 1"), " (D", html.Sub("c1"), " \u2248 1.06, T \u2248 5.6 yr): ",
                                "Currently \u00b10.03 log\u2081\u2080 (\u00b17% price) and decaying rapidly. "
                                "Compare Hyb2B\u2019s undamped cal osc 1: fixed \u00b10.24 log\u2081\u2080 (\u00b170% price) "
                                "at T \u2248 3.6 yr. The damping killed both the amplitude and the period.",
                            ]),
                            html.P([
                                html.Strong("Log osc 2"), " (D\u2082 \u2248 0.75): ",
                                "Currently \u00b10.03 log\u2081\u2080 (\u00b17% price). "
                                "Fading ~30% per decade. Near-negligible by the 2030s.",
                            ]),
                            html.P([
                                html.Strong("Cal osc 2"), " (D", html.Sub("c2"), " \u2248 0.076, T \u2248 3.7 yr): ",
                                "Currently \u00b10.30 log\u2081\u2080 (\u00b1100% price) \u2014 barely damped. "
                                "This is effectively the \u201creal\u201d halving cycle that the optimizer "
                                "moved here from cal osc 1 to avoid the heavy D", html.Sub("c1"),
                                " damping. The near-zero exponent confirms calendar terms should "
                                "not be damped.",
                            ]),
                            html.H6("Fitted Coefficients"),
                            _hyb4d_coeff_table(),
                        ], title="Full Hybrid (all decaying)", item_id="mi-hyb4d"),

                        # ── 3j. PCA Model ──
                        dbc.AccordionItem([
                            html.H6("Method"),
                            html.P([
                                "Takes the ~30 component time series from all HybPPL-family models "
                                "(HybPPL, HybPPL DD, Hyb2L, Hyb2C, Hyb2B, Hyb4D), runs PCA via SVD "
                                "to find orthogonal directions of maximum variance, then OLS-regresses "
                                "log\u2081\u2080(price) on the top ", html.Em("k"), " principal components."
                            ]),
                            html.P([
                                "At prediction time, evaluates all 30 source basis functions at ", html.Em("t"),
                                ", applies a pre-computed 30-dimensional weight vector (collapsed PCA + OLS). "
                                "No matrix operations needed \u2014 equivalent to a weighted sum of known functions."
                            ]),

                            html.H6("What are Principal Components (PCs)?"),
                            html.P(
                                "The 30 component curves from 6 models are highly correlated \u2014 every model "
                                "has a power law trend, every model has something like a halving cycle. "
                                "PCA asks: what are the independent patterns hiding in these 30 curves?"
                            ),
                            html.Ul([
                                html.Li([
                                    html.Strong("PC1"), " = the pattern that explains the most variation. "
                                    "It turns out to be the power law trend (B\u00b7log\u2081\u2080(t)), because "
                                    "that's the biggest signal \u2014 all 6 models agree on it. Explains 97.2% of the variation."
                                ]),
                                html.Li([
                                    html.Strong("PC2"), " = the next most important pattern, ",
                                    html.Em("orthogonal"), " to PC1 (independent of it). "
                                    "It's the halving cycle \u2014 the ~3.6yr calendar oscillation. 1.5%."
                                ]),
                                html.Li([
                                    html.Strong("PC3"), " = the log-periodic oscillation. 1.0%."
                                ]),
                                html.Li([
                                    html.Strong("PC4\u20136"), " = higher harmonics and fine structure. <0.3% combined."
                                ]),
                            ]),
                            html.P(
                                "Each PC is a specific weighted combination of the 30 source curves. "
                                "It's not a single model's component \u2014 it's the \"consensus direction\" "
                                "across all models for that pattern."
                            ),

                            html.H6("What does k mean?"),
                            html.P([
                                html.Em("k"), " = how many PCs we use in the model. More PCs capture finer "
                                "patterns but add parameters:"
                            ]),
                            html.Ul([
                                html.Li([html.Strong("k=1"), " = only the power law trend. Basically a power law. R\u00b2=0.963."]),
                                html.Li([html.Strong("k=2"), " = trend + halving cycle. R\u00b2=0.987."]),
                                html.Li([html.Strong("k=6"), " = all 6 PCs. R\u00b2=0.993. ", html.Strong("This is our registered model.")]),
                            ]),
                            html.P(
                                "6 numbers (the 6 PC weights) capture the same information that individual "
                                "models need 9\u201316 parameters to express, because PCA eliminates the "
                                "redundancy between models."
                            ),

                            html.H6("What is SVD?"),
                            html.P([
                                "SVD (Singular Value Decomposition) is the math behind PCA. Any matrix "
                                "can be factored into three pieces: ", html.Strong("X = U \u00d7 S \u00d7 V\u1d40"),
                            ]),
                            html.Ul([
                                html.Li([html.Strong("X"), " = data matrix (5,727 days \u00d7 30 component curves)"]),
                                html.Li([html.Strong("U"), " = how much of each pattern is present at each time point"]),
                                html.Li([html.Strong("S"), " = importance scores \u2014 how strong each pattern is (ranked largest to smallest)"]),
                                html.Li([html.Strong("V\u1d40"), " = what each pattern is made of \u2014 the recipe for combining the 30 curves into each PC"]),
                            ]),
                            html.P(
                                "S tells you the rank order. S\u2081 is huge (the power law), S\u2082 is much smaller "
                                "(halving cycle), S\u2083 even smaller... by S\u2087 you're in the noise. "
                                "That's why k=6 is enough. SVD finds the few percent of signal that's ",
                            ),
                            html.P(
                                html.Em("not"),
                                style={"display": "inline"},
                            ),
                            html.P(
                                " redundant \u2014 the genuine independent patterns \u2014 and throws away the rest. "
                                "The result is a model that's both simpler (fewer params) and more robust "
                                "(averaged across 6 models' estimates of each pattern)."
                            ),

                            html.H6("Basis Set"),
                            html.P("30 additive component functions from 6 HybPPL-family models:"),
                            _pca_basis_listing(),

                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = \beta_0 + \sum_{i=1}^{k} \beta_i \cdot \text{PC}_i(t) + z_q \cdot \sigma$$

where each $\text{PC}_i(t)$ is a linear combination of the 30 source basis functions,
computed via SVD of the centered component matrix.
                            """, mathjax=True, className="mb-3"),

                            html.H6("Registered Model (k=6)"),
                            _pca_formula_table(),

                            html.H6("PCA Variance Decomposition"),
                            html.P("30 components from 6 models collapse into orthogonal directions:"),
                            _pca_variance_table(),

                            html.H6("Model Comparison by Number of PCs"),
                            html.P([
                                "Adding more PCs improves fit but adds parameters. ",
                                html.Strong("k=6 (7 params) is the registered model"),
                                " \u2014 best BIC in the family:"
                            ]),
                            _coeff_table([
                                ("k=1 (2p)", "R\u00b2=0.963  \u03c3=0.295  BIC=\u221213,974  \u2248 Power Law (3p)"),
                                ("k=2 (3p)", "R\u00b2=0.987  \u03c3=0.171  BIC=\u221220,179  beats LPPL (9p) with 3 params"),
                                ("k=4 (5p)", "R\u00b2=0.992  \u03c3=0.136  BIC=\u221222,781  \u2248 Hyb2B (16p) with 5 params"),
                                ("k=6 (7p) \u2190", "R\u00b2=0.993  \u03c3=0.125  BIC=\u221223,776  \u2605 registered model"),
                                ("k=8 (9p)", "R\u00b2=0.993  \u03c3=0.124  BIC=\u221223,852  diminishing returns"),
                            ]),

                            html.H6("Why This Works"),
                            html.P(
                                "The 30 components from 6 models are highly correlated \u2014 every model "
                                "has nearly the same A, B, and similar oscillatory terms. PCA collapses "
                                "this redundancy into ~6 orthogonal directions. The result: R\u00b2=0.993 "
                                "with 7 total parameters, beating Hyb2B (16 params) on BIC (\u221223,776 vs "
                                "\u221223,203). The extra parameters in individual models are spent on "
                                "correlated structure that PCA captures more efficiently."
                            ),

                            html.H6("Fitted Coefficients"),
                            _pca_coeff_table(),

                            html.H6("Full Expanded Formula (for replication)"),
                            html.P([
                                "The model collapses to: ",
                                html.Code("log\u2081\u2080(price) = \u03b1 + \u03b2\u00b7log\u2081\u2080(t) + \u03a3 w\u2c7c\u00b7f\u2c7c(t)"),
                                ", a weighted sum of known oscillatory functions. "
                                "Each f\u2c7c is a component from a source model with fixed parameters."
                            ]),
                            _pca_expanded_formula(),

                            html.H6("Caveats"),
                            html.Ul([
                                html.Li("PCA directions change when source models refit \u2014 the weight "
                                        "vector is recomputed at startup, not stored as fixed constants."),
                                html.Li("Depends on all 6 source models being loaded. If any is missing, "
                                        "the basis is reduced and fit quality may change."),
                                html.Li("Not as interpretable as individual models \u2014 each PC is a "
                                        "linear combination of all 30 source components."),
                            ]),

                            html.Hr(),
                            html.H6("Addendum: Synthetic Basis Search"),
                            html.P([
                                "We tested whether a ", html.Strong("generic dictionary"),
                                " of 224 candidate basis functions could beat the curated "
                                "HybPPL basis. The dictionary included:"
                            ]),
                            html.Ul([
                                html.Li("Power law terms: log\u2081\u2080(t) at powers 0.5\u20132.0, t\u1d45 for \u03b1 = 0.25\u20131.5"),
                                html.Li("Log-periodic: cos/sin(\u03c9\u00b7ln(t)) for \u03c9 = 2\u201340 (20 frequencies \u00d7 2 phases)"),
                                html.Li("Damped log-periodic: t^(\u2212D)\u00b7cos/sin(\u03c9\u00b7ln(t)) for D = 0.3/0.7/1.0/1.5, \u03c9 = 7\u201320"),
                                html.Li("Calendar periodic: cos/sin(2\u03c0/T\u00b7t) for T = 1\u201310 years (13 periods)"),
                                html.Li("Damped calendar: t^(\u2212D)\u00b7cos/sin(2\u03c0/T\u00b7t) for D = 0.3/0.7/1.0, T = 1.9\u20136yr"),
                                html.Li("Gompertz saturation curves: 27 variants (K \u00d7 r \u00d7 t\u2080 grid)"),
                                html.Li("Broken power law pieces: piecewise log\u2081\u2080(t) at 4 breakpoints"),
                            ]),

                            html.H6("Result: curated basis wins"),
                            _coeff_table([
                                ("HybPPL basis (30 functions, k=6)", "R\u00b2=0.993  \u03c3=0.125  BIC=\u221223,776  \u2605"),
                                ("Full dictionary (224 functions, k=6)", "R\u00b2=0.951  \u03c3=0.340  BIC=\u221212,301"),
                                ("Full dictionary (224 functions, k=24)", "R\u00b2=0.995  \u03c3=0.106  BIC=\u221225,515"),
                                ("Filtered dictionary (139 functions, k=19)", "R\u00b2=0.991  \u03c3=0.142  BIC=\u221222,179"),
                            ]),
                            html.P([
                                "The generic dictionary needs ", html.Strong("k=24 (25 params)"),
                                " to beat what the curated basis achieves with k=6 (7 params). "
                                "At equal k=6, the dictionary scores R\u00b2=0.951 vs the curated "
                                "basis\u2019s 0.993 \u2014 dramatically worse."
                            ]),

                            html.H6("Why the curated basis wins"),
                            html.Ul([
                                html.Li([
                                    html.Strong("DE-fitted parameters: "),
                                    "The HybPPL models\u2019 \u03c9\u22487.42 and D\u22480.71 were found by "
                                    "differential evolution against actual price data. A grid can\u2019t "
                                    "match this precision \u2014 nearby-but-wrong frequencies waste PCA dimensions."
                                ]),
                                html.Li([
                                    html.Strong("Pre-optimized combinations: "),
                                    "Each source model\u2019s components are already tuned to work together. "
                                    "The dictionary\u2019s cos/sin pairs are generic \u2014 PCA must discover the "
                                    "right phase and amplitude from scratch."
                                ]),
                                html.Li([
                                    html.Strong("Variance concentration: "),
                                    "In the curated basis, PC1 captures 97.2% of variance (power law). "
                                    "In the generic dictionary, PC1 only captures 91.2% \u2014 the remaining "
                                    "variance is spread across many irrelevant directions."
                                ]),
                            ]),

                            html.H6("What the search confirmed"),
                            html.P(
                                "Correlation analysis of 224 functions against price residuals (after "
                                "removing the power law trend) independently recovered the same two "
                                "dominant signals that the HybPPL models were designed around:"
                            ),
                            _coeff_table([
                                ("Strongest residual signal", "t^(\u22120.7)\u00b7sin(7.4\u00b7ln(t))  |corr|=0.636"),
                                ("Second strongest", "sin(T=3.63yr)  |corr|=0.598"),
                            ]),
                            html.P([
                                "These are precisely \u03c9\u22487.4 in log-time (the LPPL frequency) and "
                                "T\u22483.6yr in calendar time (the halving cycle) \u2014 the two pillars "
                                "of the entire HybPPL family. The synthetic basis search provides "
                                "independent confirmation that these are the real signals in the data, "
                                "not artifacts of the fitting procedure."
                            ]),

                            html.Hr(),
                            html.H6("Addendum: EMD (Empirical Mode Decomposition)"),
                            html.P([
                                "EMD decomposes log\u2081\u2080(price) into ",
                                html.Strong("Intrinsic Mode Functions (IMFs)"),
                                " \u2014 oscillatory components extracted directly from the data "
                                "with ", html.Em("no assumed frequencies"), ". It works by iteratively "
                                "finding local extrema and peeling off oscillations from fastest to slowest."
                            ]),
                            html.P([
                                "EMD found 8 IMFs. Period estimates use two methods: ",
                                html.Strong("zero-crossing"), " (counting oscillation cycles \u2014 crude) and ",
                                html.Strong("FFT"), " (Fourier transform peak frequency \u2014 more precise, "
                                "gives the dominant average period):"
                            ]),
                            _coeff_table([
                                ("IMF 8 (trend)", "R\u00b2=0.925 alone \u2014 the power law backbone"),
                                ("IMF 7 (T\u22483.9yr by FFT)", "+R\u00b2\u21920.961 \u2014 the halving cycle"),
                                ("IMF 6 (T\u22482.0yr by FFT)", "+R\u00b2\u21920.992 \u2014 sub-halving rhythm"),
                                ("IMF 5 (T\u22480.8yr by FFT)", "+R\u00b2\u21920.997 \u2014 sub-annual oscillation"),
                                ("IMFs 1\u20134 (<3 months)", "noise \u2014 negligible contribution"),
                            ]),
                            html.P([
                                html.Strong("Key result: "),
                                "just 3 IMFs (trend + two cycles) give R\u00b2=0.992 with zero "
                                "assumed functional form. EMD independently discovers the same "
                                "two dominant cycles as the HybPPL models: the halving cycle "
                                "(~3.5\u20134.5yr) and a sub-halving period (~2yr)."
                            ]),

                            html.Hr(),
                            html.H6("Addendum: DMD (Dynamic Mode Decomposition)"),
                            html.P([
                                "DMD finds the ", html.Strong("eigenmodes of the system\u2019s dynamics"),
                                " \u2014 not just patterns in the signal (like PCA/EMD), but the "
                                "linear operator that best maps x(t) \u2192 x(t+1). Each eigenvalue "
                                "encodes a growth/decay rate and an oscillation frequency."
                            ]),
                            html.P([
                                "Applied to power-law-detrended residuals with time-delay embedding "
                                "(730-day window), DMD discovers these modes:"
                            ]),
                            _coeff_table([
                                ("Mode 1: T=3.97yr", "growth=\u22120.04/yr (nearly stable) \u2014 halving cycle"),
                                ("Mode 2: T=1.98yr", "growth=\u22120.24/yr (slowly decaying) \u2014 sub-halving"),
                                ("Mode 3: T=0.91yr", "growth=\u22120.29/yr (decaying) \u2014 annual cycle"),
                                ("Mode 4: T=0.66yr", "growth=\u22120.14/yr \u2014 ~8-month cycle"),
                                ("Mode 5: T=0.54yr", "growth=\u22120.51/yr (fast decay) \u2014 ~6-month cycle"),
                            ]),
                            html.P([
                                html.Strong("Key results: "),
                                "(1) The ~4yr halving cycle is the most stable mode (barely decaying), "
                                "confirming it persists indefinitely. "
                                "(2) The ~2yr sub-halving mode decays slowly \u2014 it\u2019s real but fading. "
                                "(3) DMD operates in calendar time so it cannot see log-periodic "
                                "structure (\u03c9\u22487.4); that requires the LPPL framework."
                            ]),

                            html.Hr(),
                            html.H6("Summary: Three Independent Methods Agree"),
                            _coeff_table([
                                ("PCA (basis search)", "\u03c9\u22487.4 log-periodic + T\u22483.6yr calendar"),
                                ("EMD (assumption-free)", "T\u22483.9yr + T\u22482.0yr (IMFs 7+6)"),
                                ("DMD (eigenmode analysis)", "T\u22484.0yr (stable) + T\u22482.0yr (decaying)"),
                            ]),
                            html.P(
                                "All three methods \u2014 using fundamentally different mathematics \u2014 "
                                "converge on the same two dominant oscillatory structures in Bitcoin\u2019s "
                                "price: a ~4yr halving cycle and a ~2yr sub-halving rhythm. The HybPPL "
                                "model family was designed around these signals. The PCA model captures "
                                "them optimally in 7 parameters."
                            ),
                        ], title="PCA (HybPPL Basis)", item_id="mi-pca"),

                        # ── Greedy Select ──
                        dbc.AccordionItem([
                            html.H6("Method"),
                            html.P([
                                html.Strong("Greedy forward BIC minimisation (v3, 2026-04-17)"),
                                " over a compact 36-function dictionary. At each step the single "
                                "candidate that maximally reduces BIC is added; the algorithm "
                                "stops at 5 oscillatory terms plus the α + β\u00b7log\u2081\u2080(t) "
                                "support. See ", html.Code("tools/fit_grdy.py"),
                                " for the search code."
                            ]),

                            html.H6("Candidate dictionary (36 terms)"),
                            html.Ul([
                                html.Li([html.Strong("3 log-oscillation frequencies"),
                                         " from the best-fit LPPL\u2083 triplet."]),
                                html.Li([html.Strong("3 calendar-oscillation frequencies"),
                                         " from a separate 3-freq DE fit of log\u2081\u2080(p) = "
                                         "\u03b1 + \u03b2\u00b7log\u2081\u2080(t) + \u03a3 A\u1d62"
                                         "\u00b7cos(w\u1d62\u00b7t + \u03c6\u1d62)."]),
                                html.Li([html.Strong("3 damping flavours"), " per frequency:"]),
                            ]),
                            _coeff_table([
                                ("undamped", "bare sin / cos"),
                                ("hybrid-damped", "t^(\u2212D) \u00b7 sin / cos   [LPPL-style power-law envelope]"),
                                ("entropy-damped", "E(w_e\u00b7t) \u00b7 sin / cos  [Shannon entropy envelope]"),
                            ]),
                            html.P([
                                "Each frequency gets both sin and cos parts, so phase is recovered by "
                                "picking both variants at the same \u03c9 when beneficial. ",
                                "Total: 6 freqs \u00d7 3 dampings \u00d7 2 phases = 36 candidates."
                            ]),

                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = \alpha + \beta \cdot \log_{10}(t) + \sum_i w_i \cdot D_i(t) \cdot \varphi_i(t) + z_q \cdot \sigma$$

where $D_i$ is the damping envelope (1 for undamped, $t^{-D}$ for hybrid,
$E(w_e \cdot t) = \max(-w_e t \ln (w_e t), 0)/(1/e)$ for entropy) and
$\varphi_i(t)$ is either $\sin(\omega \ln t)$, $\cos(\omega \ln t)$ (log
space) or $\sin(\omega t)$, $\cos(\omega t)$ (calendar space).
                            """, mathjax=True, className="mb-3"),

                            html.H6("Trend coefficients"),
                            _grdy_coeff_table(),

                            html.H6("Selected basis terms"),
                            _grdy_basis_table(),

                            html.H6("Selection order (grid-mode BIC trace)"),
                            html.P([
                                "The five terms below are chosen one at a time by forward-greedy BIC "
                                "minimisation over the 36-entry dictionary with frequencies frozen at "
                                "the grid values. A subsequent DE pass (", html.Code("--mode=de"),
                                ") then jointly refines frequencies and damping parameters around "
                                "those seeds; the numbers below are the fixed-grid snapshot."
                            ]),
                            html.Table([
                                html.Thead(html.Tr([
                                    html.Th("Step", style={"paddingRight": "10px"}),
                                    html.Th("Component added", style={"paddingRight": "10px"}),
                                    html.Th("R\u00b2", style={"paddingRight": "10px"}),
                                    html.Th("BIC", style={"paddingRight": "10px"}),
                                    html.Th("Params"),
                                ])),
                                html.Tbody([
                                    html.Tr([html.Td("0"), html.Td("\u03b1 + \u03b2\u00b7log\u2081\u2080(t)"),
                                             html.Td("0.9632"), html.Td("\u221214,001"), html.Td("2")]),
                                    html.Tr([html.Td("1"), html.Td("+ log-entropy sin \u03c9\u22487.12"),
                                             html.Td("0.9788"), html.Td("\u221217,163"), html.Td("3")]),
                                    html.Tr([html.Td("2"), html.Td("+ cal-undamped sin \u03c9\u22481.76"),
                                             html.Td("0.9843"), html.Td("\u221218,875"), html.Td("4")]),
                                    html.Tr([html.Td("3"), html.Td("+ log-undamped sin \u03c9\u224820.81"),
                                             html.Td("0.9870"), html.Td("\u221219,980"), html.Td("5")]),
                                    html.Tr([html.Td("4"), html.Td("+ log-undamped cos \u03c9\u22487.12"),
                                             html.Td("0.9884"), html.Td("\u221220,617"), html.Td("6")]),
                                    html.Tr([html.Td("5"), html.Td("+ cal-hybrid sin \u03c9\u22483.15"),
                                             html.Td("0.9895"), html.Td("\u221221,156"), html.Td("7")]),
                                ]),
                            ], style={"marginBottom": "12px", "fontSize": UI_FONT_LG}),

                            html.H6("Comparison (7 params)"),
                            _coeff_table([
                                ("Greedy v3 (DE)", "R\u00b2=0.9927  \u03c3=0.131  BIC=\u221223,293  \u2605 chosen fit"),
                                ("Greedy v3 (grid)", "R\u00b2=0.9895  \u03c3=0.158  BIC=\u221221,156"),
                                ("PCA (7p)", "R\u00b2=0.9933  \u03c3=0.125  BIC=\u221223,776"),
                                ("EPPL 2+2 (16p)", "R\u00b2=0.9933  \u03c3=0.125  BIC=\u221223,681"),
                            ]),

                            html.Hr(),
                            html.H6("What changed from v2"),
                            html.P([
                                "v2 hardcoded 5 entropy-damped term shapes in the class body. v3 stores "
                                "the selected basis as a generic ",
                                html.Code("_BASIS = ((space, damping, freq, phase, weight, d_param), ...)"),
                                " tuple, so future refits can drop any term mix in without touching "
                                "the class \u2014 ", html.Code("tools/fit_grdy.py --update"),
                                " regex-patches just the coefficient lines. The DE-refined v3 fit "
                                "selects a mix of undamped + entropy + hybrid terms rather than v2's "
                                "entropy-only choice, and reaches ~v2 quality (R\u00b2=0.9927) with a "
                                "cleaner programmatic design."
                            ]),

                            html.Hr(),
                            html.H6("Beyond sinusoidal: evolving bubble shapes"),
                            html.P(
                                "The greedy and HybPPL models use cosine waves for the calendar "
                                "oscillation. But Bitcoin bubbles aren\u2019t smooth sinusoids \u2014 "
                                "early bubbles have sharp blow-off tops (more like triangles) while "
                                "later bubbles develop extended plateaus (more like flat-topped waves). "
                                "We tested whether a shape-evolving periodic function fits better."
                            ),
                            html.H6("Evolving shape model"),
                            html.P([
                                "Replace ", html.Code("cos(\u03c9t+\u03c6)"), " with ",
                                html.Code("tanh(k(t)\u00b7sin(\u03c9t+\u03c6))"),
                                ", where ", html.Code("k(t) = k\u2080 + k\u2081\u00b7t"),
                                " controls the peak shape:"
                            ]),
                            html.Ul([
                                html.Li("k small \u2192 sinusoidal peaks (sharp, early Bitcoin)"),
                                html.Li("k large \u2192 flat-topped plateaus (mature Bitcoin)"),
                            ]),
                            html.P("Fitted result (11 params, +2 over HybPPL):"),
                            _coeff_table([
                                ("k(2011) = 0.37", "nearly sinusoidal \u2014 sharp bubble peaks"),
                                ("k(2025) = 2.28", "noticeably flat-topped \u2014 extended plateaus"),
                                ("k(2034) = 3.51", "near square wave (projected)"),
                                ("R\u00b2 = 0.9893", "\u03c3 = 0.158, BIC = \u221221,028"),
                            ]),

                            html.H6("What stays the same across all models"),
                            html.Ul([
                                html.Li([
                                    html.Strong("Log-periodic: "),
                                    "\u03c9\u22487.4, D\u22480.65\u20130.83 \u2014 identical whether you use "
                                    "cosine, triangle, or flat-top. This signal is rock solid and doesn\u2019t "
                                    "need shape evolution. Its natural t^(\u2212D) damping already handles "
                                    "the peak sharpening as amplitude decays."
                                ]),
                                html.Li([
                                    html.Strong("Calendar period: "),
                                    "T\u22483.6yr \u2014 the halving cycle is present in every model."
                                ]),
                                html.Li([
                                    html.Strong("Power law slope: "),
                                    "B\u22485.0\u20135.1 \u2014 consistent everywhere."
                                ]),
                            ]),

                            html.H6("The key insight: two types of oscillation"),
                            html.P([
                                html.Strong("Log-periodic oscillation "),
                                "(\u03c9\u22487.4 in ln(t)) is driven by ",
                                html.Em("fractal self-similarity"),
                                " \u2014 scale-invariant, no shape change needed. "
                                "The damping exponent D naturally sharpens peaks as amplitude decays."
                            ]),
                            html.P([
                                html.Strong("Calendar oscillation "),
                                "(T\u22483.6yr) is driven by ",
                                html.Em("human behavior"),
                                " \u2014 the halving cycle, FOMO dynamics, and market participation. "
                                "This is where shape evolution matters: as Bitcoin\u2019s participant base "
                                "grows, bubble peaks flatten from sharp blow-off tops to extended "
                                "distribution plateaus. The tanh(k(t)\u00b7sin) formulation captures this "
                                "with a single linearly-evolving parameter."
                            ]),

                            html.H6("Non-sinusoidal basis search"),
                            html.P(
                                "We also tested triangle, flat-top, and trapezoid waves as alternatives "
                                "to cosine in the greedy component selection:"
                            ),
                            _coeff_table([
                                ("Sinusoidal only (16p)", "R\u00b2=0.9943  BIC=\u221224,579"),
                                ("Triangle (17p)", "R\u00b2=0.9953  BIC=\u221225,714"),
                                ("Flat-top (17p)", "R\u00b2=0.9954  BIC=\u221225,751"),
                                ("Mixed shapes (27p)", "R\u00b2=0.9970  BIC=\u221228,213"),
                            ]),
                            html.P(
                                "Non-sinusoidal waves beat cosine by ~1,000 BIC. The mixed-shape "
                                "dictionary (all 5 wave types \u00d7 both time bases) achieves R\u00b2=0.997. "
                                "Flat-top waves dominate (10 of 25 selected terms), followed by triangle (6)."
                            ),

                            html.H6("Bubble shape evolution across cycles"),
                            html.P(
                                "Asymmetric trapezoid fits to individual bubbles (excess above BM support) "
                                "show two statistically significant trends:"
                            ),
                            _coeff_table([
                                ("Duration widening (p=0.012)", "0.87yr (2011) \u2192 5.39yr (2024)"),
                                ("Amplitude shrinking (p=0.016)", "1.38 \u2192 0.47 log\u2081\u2080 (each cycle ~20% smaller)"),
                            ]),
                            html.P(
                                "Early bubbles (2011\u20132017) are tall narrow triangles with no plateau. "
                                "Later bubbles (2021\u20132024) are shorter, wider, and developing plateau "
                                "structure. As the participant base grows, speculative excess distributes "
                                "over longer periods with reduced peak amplitude \u2014 consistent with "
                                "market maturation."
                            ),
                        ], title="Greedy Select", item_id="mi-grdy"),

                        # ── Entropy PPL ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A + B \log_{10}(t) + C_1 \cdot E(w_1 t) \cos(\omega_1 \ln t + \varphi_1) + C_3 \cdot E(w_2 t) \cos(\omega_2 \ln t + \varphi_3) + C_2 \cos(\omega_{c1} t + \varphi_2) + C_4 \cos(\omega_{c2} t + \varphi_4)$$

where $E(x) = \max(-x \ln x,\; 0)\, / \,(1/e)$ is the **normalized Shannon entropy envelope**.
                            """, mathjax=True, className="mb-3"),

                            html.H6("Shannon Entropy Interpretation"),
                            html.P([
                                "The function ", html.Code("-x\u00b7ln(x)"),
                                " is the core of Shannon\u2019s information entropy. If you think of ",
                                html.Code("w\u00b7t"), " as the ",
                                html.Em("fraction of Bitcoin\u2019s adoption lifecycle completed"),
                                ", then E(w\u00b7t) measures the ",
                                html.Strong("information uncertainty"),
                                " about whether adoption is complete."
                            ]),
                            html.Ul([
                                html.Li([
                                    html.Strong("Peak at w\u00b7t = 1/e \u2248 37%: "),
                                    "The moment of maximum uncertainty about whether Bitcoin "
                                    "will succeed or fail. Before that, it\u2019s \u201cprobably too "
                                    "early to tell.\u201d After that, the outcome is increasingly determined."
                                ]),
                                html.Li([
                                    html.Strong("Zero at w\u00b7t = 1 (100%): "),
                                    "Adoption is \u201cresolved\u201d \u2014 the speculative question is "
                                    "answered. There\u2019s nothing left to speculate about."
                                ]),
                            ]),
                            html.P([
                                "The oscillations multiplied by this envelope mean: ",
                                html.Strong("speculative cycles are strongest when uncertainty "
                                            "about Bitcoin\u2019s future is highest."),
                                " As adoption becomes more certain (w\u00b7t \u2192 1, envelope \u2192 0), "
                                "the oscillations die out."
                            ]),

                            html.H6("What the fitted parameters say"),
                            html.P("With the primary log-periodic\u2019s fitted w = 0.107:"),
                            _coeff_table([
                                ("w\u00b7t = 0.37 at t=3.5yr (2013)",
                                 "Exactly 1/e \u2014 maximum entropy, peak speculation"),
                                ("w\u00b7t = 1.0 at t=9.5yr (2019)",
                                 "\u201cAdoption resolved\u201d \u2014 oscillations extinct"),
                                ("w\u00b7t = 1.68 at t=16yr (now)",
                                 "Well past the zero crossing"),
                            ]),
                            html.P([
                                html.Strong("This is saying: "),
                                "Bitcoin\u2019s speculative uncertainty was resolved by ~2019. "
                                "After that, price dynamics are driven by the halving cycle "
                                "(calendar-periodic), not by speculative bubble behavior "
                                "(log-periodic). The log-periodic signal was an artifact of "
                                "the \u201cwill it survive?\u201d era."
                            ]),
                            html.P([
                                "Whether this is deep or coincidental depends on whether you "
                                "believe the binary adoption framing. But the mathematical form "
                                "is ", html.Em("literally"), " the entropy function, and the "
                                "fitted parameters put maximum entropy right at the peak of "
                                "early Bitcoin speculation."
                            ]),

                            html.H6("Why entropy damping beats power-law damping"),
                            html.P(
                                "HybPPL uses t^(\u2212D), which decays monotonically from "
                                "infinity but never reaches zero. The entropy envelope "
                                "provides a physically motivated alternative: oscillations "
                                "are born, peak, and die. This gives the model a natural "
                                "cutoff where log-periodic oscillations cease entirely, "
                                "rather than persisting as ever-smaller ripples. "
                                "At 9 params: EPPL BIC=\u221221,065 vs HybPPL BIC=\u221220,814."
                            ),

                            html.H6("Fitted Coefficients"),
                            _eppl_coeff_table(),

                            html.H6("Model Comparison"),
                            _coeff_table([
                                ("EPPL 2+2 (16p)", "R\u00b2=0.9933  BIC=\u221223,681"),
                                ("Hyb2B (16p)", "R\u00b2=0.9927  BIC=\u221223,203"),
                                ("PCA (7p)", "R\u00b2=0.9928  BIC=\u221223,776"),
                            ]),

                            html.H6("Caveats"),
                            html.Ul([
                                html.Li(
                                    "The entropy envelope introduces the w parameter which "
                                    "controls when oscillations peak and die. This is a strong "
                                    "assumption about adoption timing."
                                ),
                                html.Li(
                                    "Once w\u00b7t > 1, the log-periodic terms are permanently "
                                    "zero. Long-horizon projections rely entirely on the power "
                                    "law trend and the undamped halving-cycle oscillations."
                                ),
                            ]),
                        ], title="Entropy PPL", item_id="mi-eppl"),

                        # ── 4. Exponential ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = \alpha + \beta \cdot t + z_q \cdot \sigma$$

Solved for price:

$$\text{price}(q,\, t) = 10^{\,\alpha + z_q \sigma} \cdot 10^{\,\beta \, t}$$

where $t$ = years since optimal time origin (2009-07-25, linear, not log-transformed), and $z_q = \Phi^{-1}(q)$.
                            """, mathjax=True, className="mb-3"),

                            html.H6("Method"),
                            html.P(
                                "OLS regression of log\u2081\u2080(price) against time (not log-time). "
                                "This assumes constant percentage growth per year \u2014 exponential "
                                "in price. Quantile bands use Gaussian shift like PL. Included for "
                                "comparison to demonstrate why power-law models are preferred: "
                                "Bitcoin\u2019s growth rate is not constant but decelerates over time, "
                                "which exponential models cannot capture."
                            ),

                            html.H6("Fitted Coefficients"),
                            _exp_coeff_table(),

                            html.H6("Why it fails"),
                            html.Ul([
                                html.Li([
                                    html.Strong("R\u00b2 = 0.871:"),
                                    " Substantially worse than QR (0.98), PL (0.97), or LPPL (0.98).",
                                ]),
                                html.Li([
                                    html.Strong("\u03c3 = 0.553:"),
                                    " Nearly double PL\u2019s 0.302 \u2014 the model explains much less "
                                    "variance, so quantile bands are very wide.",
                                ]),
                                html.Li([
                                    html.Strong("Constant growth rate:"),
                                    " The model assumes Bitcoin grows at the same percentage rate "
                                    "forever. In reality, growth decelerates \u2014 early years saw "
                                    "1000\u00d7 gains while recent years see 2\u20135\u00d7. Power laws "
                                    "naturally capture this deceleration; exponentials cannot.",
                                ]),
                                html.Li([
                                    html.Strong("Extreme projections:"),
                                    " By 2040 the median hits $12B, by 2050 $18T. The exponential "
                                    "overshoots every other model by orders of magnitude at long horizons.",
                                ]),
                            ]),
                        ], title="Exponential (included for comparison)", item_id="mi-exp"),

                        # ── 4b. Gompertz ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = K \cdot \exp\!\big(-\exp(-r \cdot (t - t_0))\big) + z_q \cdot \sigma$$

The **Gompertz** curve is an asymmetric S-curve that saturates more gradually
than the symmetric logistic. $K$ is the carrying capacity (log$_{10}$ of the
maximum price Bitcoin can reach), $r$ controls the growth rate, and $t_0$ is
the inflection point where growth rate is maximized.
                            """, mathjax=True, className="mb-3"),
                            html.H6("Method"),
                            html.P(
                                "Fitted via scipy.optimize.curve_fit on historical log\u2081\u2080(price) "
                                "for t \u2265 1 year. Quantile bands use Gaussian shift (z\u00b7\u03c3) like PL."
                            ),
                            html.H6("Motivation"),
                            html.P(
                                "All other Quantoshi power-law models (PL, LPPL, etc.) are unbounded \u2014 "
                                "they predict infinite price given enough time. The Gompertz model "
                                "asks: what if Bitcoin adoption saturates? It provides an upper "
                                "bound that no power law model can."
                            ),
                            html.H6("Fitted Coefficients"),
                            _gompertz_coeff_table(),
                            html.H6("Caveats"),
                            html.Ul([
                                html.Li(
                                    "The carrying capacity K is poorly constrained by current data \u2014 "
                                    "Bitcoin may still be in early growth phase where Gompertz and "
                                    "power law are indistinguishable."
                                ),
                                html.Li(
                                    "The model assumes a single S-curve. Real adoption may follow "
                                    "multiple S-curves (e.g., retail \u2192 institutional \u2192 sovereign)."
                                ),
                                html.Li(
                                    "R\u00b2 is lower than power law (~0.93 vs ~0.96) because the "
                                    "Gompertz curve cannot capture the early explosive growth as well."
                                ),
                            ]),
                        ], title="Gompertz", item_id="mi-gomp"),

                        # ── 4c. Broken Power Law ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
For $t < t_{\text{break}}$:
$$\log_{10}(\text{price}) = a_1 + b_1 \cdot \log_{10}(t) + z_q \cdot \sigma$$

For $t \geq t_{\text{break}}$:
$$\log_{10}(\text{price}) = a_2 + b_2 \cdot \log_{10}(t) + z_q \cdot \sigma$$

with continuity constraint: $a_2 = a_1 + (b_1 - b_2) \cdot \log_{10}(t_{\text{break}})$.
                            """, mathjax=True, className="mb-3"),
                            html.H6("Method"),
                            html.P(
                                "Fitted via differential evolution (4 free parameters: a\u2081, b\u2081, "
                                "t_break, b\u2082; a\u2082 derived from continuity). Quantile bands use "
                                "Gaussian shift like PL."
                            ),
                            html.H6("Motivation"),
                            html.P(
                                "Tests whether Bitcoin's growth rate has structurally changed. "
                                "If b\u2082 < b\u2081, growth has slowed (consistent with maturation). "
                                "If b\u2082 \u2248 b\u2081, a single power law suffices."
                            ),
                            html.H6("Fitted Coefficients"),
                            _bpl_coeff_table(),
                            html.H6("Interpretation"),
                            html.Ul([
                                html.Li(
                                    "The breakpoint date indicates when the structural shift occurred. "
                                    "A mid-2010s breakpoint may reflect the transition from early-adopter "
                                    "to retail phases."
                                ),
                                html.Li(
                                    "If the two slopes are nearly equal, the single power law is sufficient "
                                    "and the break adds no value."
                                ),
                            ]),
                        ], title="Broken Power Law", item_id="mi-bpl"),

                        # ── 4d. Offset Power Law ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A + m \cdot \log_{10}(t + c) + z_q \cdot \sigma$$

A three-parameter power law that lets the model pick its own effective
time-zero via the offset $c$. When $c \approx 0$ the model degenerates to
plain PL; non-zero $c$ implies the data prefer a different time-origin
than the 2009-07-25 genesis.
                            """, mathjax=True, className="mb-3"),
                            html.H6("Method"),
                            html.P(
                                "Fitted via differential evolution on log\u2081\u2080(price) for t \u2265 1 year, "
                                "then polished with curve_fit. Bounds keep t+c positive across the dataset. "
                                "Quantile bands use _ShrinkingBandsMixin (same as Gompertz / Exp)."
                            ),
                            html.H6("Motivation"),
                            html.P(
                                "Tests whether the 2009-07-25 time origin is statistically optimal. "
                                "If the optimizer settles on c \u2248 0 (as it currently does), that's "
                                "empirical confirmation that genesis is a reasonable choice."
                            ),
                            html.H6("Fitted Coefficients"),
                            _plo_coeff_table(),
                            html.H6("Caveats"),
                            html.Ul([
                                html.Li(
                                    "Parameter identifiability is weak: (A, m, c) trade off against "
                                    "each other, so fitted values are less interpretable than plain PL's."
                                ),
                                html.Li(
                                    "Comparison-only model \u2014 R\u00b2 is similar to plain PL. "
                                    "Included as a sanity check on our genesis choice."
                                ),
                                html.Li(
                                    "Keep t+c > 0 for prediction. Extrapolating to t \u2248 0 or negative "
                                    "would push log\u2081\u2080 arg into the tiny-positive regime."
                                ),
                            ]),
                        ], title="Offset Power Law", item_id="mi-plo"),

                        # ── 4e. Stretched Exponential ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A + B \cdot t^{\beta} + z_q \cdot \sigma$$

A three-parameter family that interpolates smoothly between pure
exponential ($\beta=1$) and log-like ($\beta \to 0$). For BTC the
optimiser wants $\beta \to 0$ (the power law is a better fit than
any stretched exponential), so $\beta$ is floored at 0.25 to keep the
model in a genuinely stretched-exp regime.
                            """, mathjax=True, className="mb-3"),
                            html.H6("Method"),
                            html.P(
                                "Differential evolution on log\u2081\u2080(price) for t \u2265 1 year, "
                                "polished with curve_fit. β bounded to [0.25, 1.5] to prevent "
                                "degeneracy toward the logarithm (β → 0) or super-exponential blow-up (β > 1.5). "
                                "Quantile bands use _ShrinkingBandsMixin."
                            ),
                            html.H6("Motivation"),
                            html.P(
                                "Stretched exponentials are common in physics (relaxation phenomena) "
                                "and sometimes fitted to adoption curves. This model shows that BTC "
                                "is sub-exponential but not in a stretched-exp way \u2014 the optimiser "
                                "wants to collapse β to zero (i.e., logarithm), which just recovers "
                                "the power law. Included as a sanity check."
                            ),
                            html.H6("Fitted Coefficients"),
                            _sexp_coeff_table(),
                            html.H6("Caveats"),
                            html.Ul([
                                html.Li(
                                    "β hits the 0.25 floor. Economically, the data prefer power-law "
                                    "to stretched-exp."
                                ),
                                html.Li(
                                    "(A, B) trade off via the t\u1d2c exponent; fitted values are not "
                                    "independently interpretable."
                                ),
                                html.Li(
                                    "R\u00b2 ~ 0.956 is similar to plain PL. Keep as diagnostic only."
                                ),
                            ]),
                        ], title="Stretched Exponential", item_id="mi-sexp"),

                        # ── 4f. Logistic (true S-curve) ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = \dfrac{K}{1 + e^{-r(t - t_0)}} + z_q \cdot \sigma$$

The **symmetric** logistic S-curve in log-price space \u2014 saturates at $K$
(log$_{10}$ of max price), inflects at $t_0$. Distinct from Gompertz:

* Logistic spends equal time below and above its inflection point.
* Gompertz decelerates more gradually on the upper side.

At $t_0$, Logistic sits at $K/2$ while Gompertz sits at $K/e \\approx 0.37 K$.
                            """, mathjax=True, className="mb-3"),
                            html.H6("Method"),
                            html.P(
                                "Differential evolution on log\u2081\u2080(price) for t \u2265 1 year, "
                                "polished with curve_fit. Quantile bands use _ShrinkingBandsMixin."
                            ),
                            html.H6("Motivation"),
                            html.P(
                                "The contrast case for Gompertz: same saturation concept, symmetric "
                                "dynamics. Fitting both and showing their divergent extrapolations "
                                "illustrates how sensitive long-horizon BTC forecasts are to the "
                                "shape of the saturation curve."
                            ),
                            html.H6("Fitted Coefficients"),
                            _logi_coeff_table(),
                            html.H6("Caveats"),
                            html.Ul([
                                html.Li(
                                    "The fitted K currently sits below observed recent-year prices \u2014 "
                                    "BTC has already blown through the saturation ceiling the symmetric "
                                    "logistic wants to impose. Instructive: this is evidence that a "
                                    "symmetric S-curve is a poor model for BTC adoption dynamics."
                                ),
                                html.Li(
                                    "R\u00b2 is noticeably lower than Gompertz and PL because the "
                                    "symmetric saturation fits recent data badly."
                                ),
                                html.Li(
                                    "Don't treat the K value as a price target \u2014 it's an artefact "
                                    "of the model shape, not a forecast."
                                ),
                            ]),
                        ], title="Logistic (S-curve)", item_id="mi-logi"),

                        # ── 5. Stock-to-Flow ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = \alpha + \beta \cdot \log_{10}(\text{S2F})$$

Solved for price:

$$\text{price}(t) = 10^{\,\alpha} \cdot \text{S2F}(t)^{\,\beta}$$

$$\text{S2F}(t) = \frac{\text{stock}(t)}{\text{flow}(t)}$$

where stock = cumulative BTC mined and flow = the **trailing 365-day**
issuance, stock(t) - stock(t-1yr) -- Plan B's original definition.
                            """, mathjax=True, className="mb-3"),

                            html.H6("Method"),
                            html.P([
                                "The Stock-to-Flow model, popularized by ",
                                html.A("Plan B",
                                       href="https://twitter.com/100trillionUSD",
                                       target="_blank", rel="noopener noreferrer"),
                                ", posits that Bitcoin\u2019s scarcity (measured by the S2F ratio) drives "
                                "its price. This default uses the trailing 365-day flow, which blends the "
                                "block reward smoothly across each halving rather than stepping. An "
                                "instantaneous-flow variant (S2F inst) is available separately. "
                                "The model fits a linear regression in "
                                "log\u2081\u2080(S2F) vs log\u2081\u2080(price) space.",
                            ]),

                            html.H6("Fitted Coefficients"),
                            _s2f_coeff_table("s2f"),

                            html.H6("Bitcoin Halving Schedule Constants"),
                            _coeff_table([
                                ("Halving interval", "210,000 blocks"),
                                ("Blocks per day", "144"),
                                ("Initial reward", "50 BTC"),
                                ("Current reward (2024+)", "3.125 BTC"),
                            ]),

                            html.H6("Characteristics"),
                            html.Ul([
                                html.Li(
                                    "Single trajectory (non-quantized) \u2014 no percentile bands."
                                ),
                                html.Li(
                                    "Trailing flow keeps S2F continuous across each halving (no step-function jump)."
                                ),
                                html.Li(
                                    "Tends to produce extremely high projections at long time horizons "
                                    "because S2F grows exponentially while BTC issuance approaches zero."
                                ),
                                html.Li(
                                    "The model is widely criticized for assuming a causal "
                                    "scarcity\u2013price relationship that may be correlational."
                                ),
                            ]),
                        ], title="Stock-to-Flow (S2F)", item_id="mi-s2f"),

                        # ── 5. Monte Carlo Simulation ──
                        dbc.AccordionItem([
                            html.H6("Overview"),
                            html.P(
                                "The Monte Carlo engine generates thousands of possible future Bitcoin "
                                "price paths using a Markov chain trained on historical price transitions. "
                                "It produces fan-shaped probability bands rather than single deterministic lines."
                            ),

                            html.H6("Transition Matrix"),
                            dcc.Markdown(r"""
Historical prices are discretized into **5 bins** (price regimes) based on their quantile position:

| Bin | Regime | Quantile Range |
|-----|--------|---------------|
| 0 | Bargain | 0\%–20\% |
| 1 | Cheap | 20\%–40\% |
| 2 | Fair | 40\%–60\% |
| 3 | Pricey | 60\%–80\% |
| 4 | Bubble | 80\%–100\% |

For each consecutive 30-day interval, the bin-to-bin transition is recorded. The resulting 5×5 matrix $\mathbf{T}$ is row-normalized so each row sums to 1:

$$T_{ij} = P(\text{bin}_{t+1} = j \mid \text{bin}_t = i)$$
                            """, mathjax=True),

                            _clickable_img("/assets/markov_states.png", "700px"),

                            html.H6("Transition Counts by Year"),
                            html.P(
                                "The grid below shows how many times each transition occurred in each "
                                "calendar year. This reveals the temporal structure behind the matrix: "
                                "Bargain\u2192Bargain dominates in early years, Bubble\u2192Bubble clusters "
                                "around 2017 and 2021, and rare transitions like Bubble\u2192Fair have "
                                "only occurred twice in Bitcoin\u2019s history.",
                                className="small text-muted",
                            ),
                            _clickable_img("/assets/markov_histograms.png", "900px"),

                            html.H6("Forward Simulation"),
                            html.Ol([
                                html.Li("Start all paths in the bin corresponding to the user\u2019s entry percentile."),
                                html.Li("At each time step, sample the next bin from the transition probability row."),
                                html.Li("Convert bin indices back to log-prices (uniform sampling within bin)."),
                                html.Li("Repeat for 800 simulations (100 in free tier)."),
                            ]),

                            html.H6("Fan Percentiles"),
                            html.P(
                                "Six percentile bands are computed across all simulated paths:"
                            ),
                            _coeff_table([
                                ("P1%", "Extreme low (near worst case)"),
                                ("P5%", "Lower bound of 90% confidence interval"),
                                ("P25%", "Lower quartile"),
                                ("P50%", "Median outcome"),
                                ("P75%", "Upper quartile"),
                                ("P95%", "Upper bound of 90% confidence interval"),
                            ]),

                            html.H6("Simulation Parameters"),
                            _coeff_table([
                                ("Simulations", "800 (paid) / 100 (free)"),
                                ("Frequency", "Monthly (30-day steps)"),
                                ("Training window", "2010\u2013present"),
                                ("Step size", "30 days"),
                                ("Bins", "5 (Bargain / Cheap / Fair / Pricey / Bubble)"),
                            ]),

                            html.H6("Regime Filter"),
                            html.P(
                                "Users can block specific bins to model constrained scenarios "
                                "(e.g., \u201cwhat if we never see another extreme bubble?\u201d). "
                                "Blocked bins have their columns zeroed in the transition matrix. "
                                "A ghost overlay shows the unfiltered simulation for comparison."
                            ),

                            html.H6("Pre-Computed Cache"),
                            _coeff_table([
                                ("Cache size", "~834 MB (RAM at startup)"),
                                ("Scenarios", "~45,000"),
                                ("Start years", "2026, 2028, 2031, 2035, 2040"),
                                ("Entry percentiles", "10% steps (10\u201390%)"),
                                ("Durations", "10, 20, 30, 40 years"),
                                ("Withdrawal amounts", "$5K, $7.5K, $12.5K, $20K, $32.5K, $69,420"),
                                ("Inflation rates", "2%, 3%, 4%, 6%, 8%, 10%, 12%"),
                                ("Stack sizes", "0.1, 0.5, 1.0, 2.0, 5.0, 10.0 BTC"),
                            ]),
                        ], title="Monte Carlo (Markov Chain) Simulation", item_id="mi-mc"),

                        # ── BM Empirical Floor ──
                        dbc.AccordionItem([
                            html.H6("Overview"),
                            html.P(
                                "The Empirical Floor uses a power law support line drawn through "
                                "two observed bear-market lows: 2010-10-05 ($0.06) and "
                                "2026-02-05 ($58,000). The second anchor balances temporal "
                                "uniformity of below-line data points with the breadth of the "
                                "time window considered, ensuring the support is relevant across "
                                "all eras of Bitcoin\u2019s history rather than being an artifact "
                                "of one crash."
                            ),
                            html.H6("Parameters"),
                            html.P([
                                "Support slope: 5.3106 (vs 5.13 standard). ",
                                "Support intercept: \u22121.6246. ",
                                "R\u00b2 with bubble fitting: 0.9932. ",
                                "Quantile bands: Gaussian z-shifted from the bubble composite median."
                            ]),
                            html.H6("Bubble Timing: EF vs BM"),
                            html.P([
                                "The steeper EF floor sits higher under recent data, which shifts "
                                "the fitted t\u1d63\u1d62\u209b\u2091 of the 4th major bubble from "
                                "~2019.5 (BM) to ~2020.3 (EF). This compresses the last interval "
                                "from 4.0 to 3.4 years, flattening the interval trend. "
                                "The extrapolated next onset is ",
                                html.Strong("~2027"),
                                " for EF vs ",
                                html.Strong("~2029"),
                                " for BM \u2014 a ~2 year difference driven by a 0.8-year shift "
                                "in a single historical bubble fit.",
                            ]),
                            html.P(
                                "EF interval trend: +0.20 yr/cycle (nearly flat, ~3.7 yr next). "
                                "BM interval trend: +0.37 yr/cycle (lengthening, ~5.7 yr next). "
                                "This is visible when comparing the EF and BM sensitivity sweeps."
                            ),

                            html.H6("Convergence Narrative"),
                            html.P(
                                "The steeper support line means bubble amplitudes decay faster "
                                "(K declining from 1.21 to 0.32 vs BM\u2019s 1.18 to 0.40). "
                                "Predicted future bubbles converge on the support rapidly \u2014 "
                                "implying that the classic 4-year halving-driven boom/bust cycle "
                                "is approaching its end, with Bitcoin transitioning to a more "
                                "mature, lower-volatility asset."
                            ),

                            html.H6("Sensitivity"),
                            html.P([
                                "Both models sit in the ",
                                html.Strong("stable region"),
                                " below the R\u00b2 ridge on the sensitivity sweep \u2014 "
                                "moderate perturbations to slope/intercept produce smooth, "
                                "predictable changes in bubble parameters. However, onset timing "
                                "is inherently the most sensitive output: predicting an exact "
                                "moment in the future is fundamentally uncertain even when other "
                                "parameters (amplitude, interval trend) are robust.",
                            ]),
                            html.H6("Fitted Coefficients"),
                            _coeff_table(_ef_rows()),
                        ], title="BM Empirical Floor", item_id="mi-ef"),

                        # ── User Model (U₁) ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = \alpha + \beta \cdot \log_{10}(t)$$

where $\alpha$ (intercept) and $\beta$ (slope) are derived from **two user-selected data points** in log-log space.
                            """, mathjax=True, className="mb-3"),

                            html.H6("Method"),
                            html.P([
                                "The user clicks two historical data points on the Bubble chart. "
                                "The slope and intercept define a power law line through those points. "
                                "Parallel quantile bands are derived from the ",
                                html.Strong("empirical residual distribution"),
                                " \u2014 for each standard quantile, the shift equals the corresponding "
                                "percentile of residuals between the user's line and all historical prices. "
                                "This captures the asymmetric spread (bubbles push upper quantiles "
                                "farther from the median than lower quantiles).",
                            ]),

                            html.H6("Properties"),
                            html.Ul([
                                html.Li("Fully quantized: all standard quantile bands available"),
                                html.Li("Same slope for all quantile lines (parallel in log-log)"),
                                html.Li("The user's drawn line has shift = 0 (passes exactly through both points)"),
                                html.Li("Own quantile = fraction of historical data below the line"),
                                html.Li("Session-only \u2014 disappears on page refresh"),
                                html.Li([html.Strong("Color: "), html.Span("orange (#e67e22), 3px for the drawn line")]),
                            ]),
                        ], title="User Model (U\u2081)", item_id="mi-u1"),

                        # ── 6. Model Comparison ──
                        dbc.AccordionItem([
                            html.H6("At a Glance"),
                            _comparison_table(),

                            html.H6("Key Differences"),
                            html.Ul([
                                html.Li([
                                    html.Strong("QR vs PL: "),
                                    "QR fits each percentile independently (different slopes); PL shifts "
                                    "a single OLS line (same slope, parallel bands). QR captures the fact that "
                                    "extreme percentiles have different growth rates.",
                                ]),
                                html.Li([
                                    html.Strong("LPPL vs PL: "),
                                    "Both are power laws with Gaussian bands, but LPPL adds damped "
                                    "log-periodic oscillations. LPPL\u2019s \u03c3 is smaller (0.217 vs 0.302) "
                                    "because the oscillatory term absorbs bubble/bust variance.",
                                ]),
                                html.Li([
                                    html.Strong("Exponential vs power law: "),
                                    "Exponential assumes constant percentage growth (straight line in "
                                    "log-price vs time). Power law assumes decelerating growth (straight "
                                    "line in log-price vs log-time). Bitcoin\u2019s empirical deceleration "
                                    "makes power law the better fit (R\u00b2 0.97 vs 0.87).",
                                ]),
                                html.Li([
                                    html.Strong("S2F vs all others: "),
                                    "S2F is driven by supply mechanics (halvings), not time. It produces "
                                    "step-function jumps and extreme far-future projections. Single "
                                    "trajectory only \u2014 no uncertainty bands.",
                                ]),
                                html.Li([
                                    html.Strong("MC vs deterministic models: "),
                                    "MC generates stochastic paths from empirical transitions, producing "
                                    "probability distributions rather than fixed curves. It can model "
                                    "path-dependent scenarios (DCA, withdrawals) that deterministic "
                                    "models cannot.",
                                ]),
                            ]),
                        ], title="Model Comparison", item_id="mi-compare"),

                        # ── Historical Regimes (Dollar Assets) ──
                        dbc.AccordionItem([
                            html.H6("Overview"),
                            html.P([
                                "The Historical Regimes model replaces user-input rates for dollar "
                                "assets (equities, bonds, treasuries) with a Markov chain that "
                                "transitions between regimes based on historical data. Used in the "
                                "Citadel Planner's ",
                                html.Strong("Dollar Asset Returns"),
                                " dropdown.",
                            ]),

                            html.H6("Data Sources"),
                            html.Ul([
                                html.Li("Equities: S&P 500 monthly total returns (Yahoo Finance)"),
                                html.Li("Bonds: AGG Bond ETF monthly total returns"),
                                html.Li("Short Treasuries: 3-month T-bill yield \u2192 total return via duration approximation"),
                                html.Li("Medium Treasuries: 5-year T-note yield \u2192 total return"),
                                html.Li("Long Treasuries: 20-year T-bond yield \u2192 total return"),
                            ]),

                            html.H6("Method"),
                            html.P([
                                "Monthly returns are discretized into ",
                                html.Code("n_bins"),
                                " (default 5) regimes via percentile binning. A row-stochastic "
                                "transition matrix records the probability of moving from one regime "
                                "to another. At each simulation step, the current regime determines "
                                "the return distribution, and a random draw selects the next regime.",
                            ]),

                            html.H6("Key Properties"),
                            html.Ul([
                                html.Li("5 independent Markov chains (one per asset class)"),
                                html.Li("Regime transitions are independent of BTC price paths"),
                                html.Li("When selected, user-input rates/volatility are ignored"),
                                html.Li("Available in both deterministic (\u25b6) and MC (\u26a1) modes"),
                            ]),

                            html.H6("Assumptions"),
                            html.P(
                                "Investment gains in the Citadel Planner are classified as "
                                "long-term capital gains. Individual equity and bond lot "
                                "tracking is not modeled."
                            ),

                            html.H6("Historical Data Summary"),
                            _regime_data_tables(),

                        ], title="Historical Regimes (Dollar Assets)", item_id="mi-regimes"),

                        # ── Citadel Planner ──
                        dbc.AccordionItem([
                            html.H6("Overview"),
                            html.P(
                                "The Citadel Planner is a multi-asset retirement simulation "
                                "engine. It combines BTC price projections with dollar assets "
                                "(cash, treasuries, equities, bonds) and an optional US federal "
                                "+ state tax layer."
                            ),

                            html.H6("Withdrawal Cost Ranking"),
                            html.P(
                                "Each withdrawal period, assets are ranked by their true "
                                "opportunity cost before any funds are drawn. The cheapest "
                                "source is consumed first, preserving higher-growth assets "
                                "for as long as possible."
                            ),
                            html.H6("Opportunity Cost Horizons"),
                            html.P(
                                "The Citadel Planner computes withdrawal cost as immediate "
                                "tax plus forgone compounding. Bitcoin uses a 10-year horizon "
                                "(twice the historical 5-year break-even). Equities and bonds "
                                "use 15 years. Treasuries use the holder\u2019s remaining "
                                "lifetime (capped at 40 years). Tax-deferred (401k/IRA) "
                                "horizons shorten as RMDs approach: before RMD age, capped "
                                "at years until forced distributions begin; at RMD age and "
                                "beyond, the IRS Uniform Lifetime Table factor sets the "
                                "horizon directly. This makes TD progressively cheaper to "
                                "withdraw voluntarily, shifting spending toward TD before "
                                "RMDs force it out at potentially higher marginal rates."
                            ),

                            html.H6("Roth Ordering"),
                            html.P(
                                "Roth (tax-free) accounts have zero tax cost but full opportunity "
                                "cost from tax-free compounding, so the cost function naturally "
                                "ranks them as expensive to withdraw. No special rule forces Roth "
                                "last \u2014 the math handles it."
                            ),
                        ], title="Citadel Planner", item_id="mi-citadel"),

                        # ── Saturating Power Law ──
                        # APPENDED LAST, and it must stay last: /mi.N and /9.N
                        # resolve positionally against _MODEL_INFO_ITEM_IDS.
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\text{price}(t) = \frac{L}{1 + (t/t_0)^{-\beta}}
\qquad\Longleftrightarrow\qquad
\log_{10} p = \log_{10} L - \log_{10}\left(1 + (t/t_0)^{-\beta}\right)$$

A **logistic in log-time** — the same S-curve as the Logistic model, with
$\ln t$ in place of $t$ — so it bends at *both* ends. While $t \ll t_0$ it is
an ordinary power law, $p \propto t^{\beta}$; as $t$ grows it rolls over
toward the ceiling $L$, reaching exactly $L/2$ at $t = t_0$.

As $t_0 \to \infty$ the second term vanishes and the model reduces **exactly**
to the Power Law. That nesting is the point: it turns "is there a ceiling?"
into "is $t_0$ finite?", a question the data can be asked directly.
                            """, mathjax=True, className="mb-3"),

                            # Everything below is prose and tables full of
                            # dollar amounts. The static /mi page loads MathJax
                            # with '$'..'$' as inline-math delimiters and
                            # typesets the whole document, which eats the text
                            # between any two of them. "mathjax_ignore" is the
                            # v3 default ignoreHtmlClass -- NOT "tex2jax_ignore",
                            # which is the v2 name and is inert here, and
                            # static_pages.py sets no options.ignoreHtmlClass of
                            # its own. Inert in the Dash app, which only typesets
                            # mathjax=True components; the onion /mi ships no
                            # MathJax at all. So this only ever shows up on the
                            # clearnet /mi -- verify there, not on /9.
                            # The formula block above stays outside.
                            html.Div([
                                html.P(html.Em(
                                    "Every figure on this card is pinned to two data windows — "
                                    "3 June 2026 and 6 August 2026 — and does not move as new "
                                    "prices arrive. That fixed comparison is the point being "
                                    "made. The plotted SatPL curve is a different matter: it "
                                    "always uses the latest fit, so it can differ from the "
                                    "numbers below."
                                ), style={"marginBottom": "16px"}),

                                html.H6("The headline: the ceiling moves when you add data"),
                                html.P(
                                    "Refitting this model on data through 3 June 2026 puts the "
                                    "ceiling at $34 trillion. Refitting it nine weeks later — "
                                    "same code, same test, 1.1% more rows — puts it at "
                                    "$15 trillion. The two windows end at prices within 1% of each "
                                    "other. The two ceilings differ by 2.3×. So the estimate is "
                                    "not tracking the price level; it is tracking the recent "
                                    "trajectory relative to trend."
                                ),
                                _spl_two_window_table(),
                                html.P(
                                    "Across the last two years the same fit has returned anything "
                                    "from $11 trillion to no ceiling at all — a 93× range — "
                                    "and the verdict flips non-monotonically: it rejects, then does "
                                    "not reject for four windows running, then rejects again. A "
                                    "number that behaves this way is not being estimated."
                                ),
                                _spl_by_cutoff_table(),
                                html.P([
                                    html.Strong("Reading the two capped rows. "),
                                    "A ceiling of infinity ", html.Em("is"), " the pure power law, "
                                    "so a window whose fit pins at the $1,000T fitting bound is data "
                                    "asking for no ceiling at all. It is also why the statistic goes "
                                    "slightly negative there: this model nests the power law, so the "
                                    "statistic cannot be negative at a true optimum, and −0.11 / "
                                    "−0.14 are the bound stopping the optimiser just short of "
                                    "that limit. Reported rather than clipped to zero, because the "
                                    "sign is informative.",
                                ]),
                                html.Div(html.Em(
                                    "So we don’t claim Bitcoin has no ceiling. We claim that "
                                    "seventeen years of data can put a floor under one but cannot "
                                    "put a number on it."
                                ), style={"borderLeft": f"3px solid {FALLBACK_MODEL_GRAY}",
                                          "paddingLeft": "12px", "marginBottom": "16px"}),

                                html.H6("Why: the upper asymptote is not in the data yet"),
                                html.P(
                                    "This is the pre-inflection sigmoid problem, familiar from "
                                    "carrying-capacity estimates in ecology and from ceiling fits "
                                    "early in an epidemic: the upper asymptote of an S-curve cannot "
                                    "be identified until the bend is inside the sample. On a log-log "
                                    "chart Bitcoin has not visibly bent."
                                ),
                                html.P(
                                    "Holding t₀ at a series of fixed values and re-optimising "
                                    "everything else shows the shape of the problem. A low ceiling "
                                    "fits visibly worse. Everything above roughly $100 trillion fits "
                                    "equally well — and “no ceiling at all” fits "
                                    "essentially as well as the best fit. The RMSE column stops "
                                    "discriminating in the fourth decimal:"
                                ),
                                _spl_profile_table(),
                                html.P(
                                    "The ceiling is therefore estimable from one side only. The data "
                                    "put a floor under it, near the current trend level, and no upper "
                                    "bound whatsoever: they are consistent with a ceiling anywhere "
                                    "above that floor, and equally consistent with none."
                                ),
                                html.P(
                                    "The same profile as a picture, for both pinned windows: the best "
                                    "fit attainable with the ceiling held at each value, with the "
                                    "shaded region marking the ceilings that window’s data cannot "
                                    "tell apart from its own best fit.",
                                    className="small text-muted",
                                ),
                                _clickable_img("/assets/spl_profile.png", "700px"),

                                html.H6("Method"),
                                html.P(
                                    "Differential evolution on log₁₀(price) for t ≥ 1 "
                                    "year, with the ceiling bounded below by the highest observed "
                                    "price and above at a $1,000T market cap. Quantile bands use the "
                                    "shared shrinking-σ mixin, as for PL, Gompertz and the other "
                                    "simple models."
                                ),

                                html.H6("Fitted Coefficients"),
                                _spl_coeff_table(),
                                # Pinned to the 2026-08-06 window, like every
                                # other figure on this card. See the refresh
                                # note above _SPL_WINDOW_LABEL in ._helpers.
                                html.P(
                                    "Read those with the caveat attached. The fitted roll-over at "
                                    "t₀ ≈ 23.9 years sits 1.4× beyond the last "
                                    "observation — on that window the price history reaches "
                                    "t = 17.0 years — so the part of the curve that would fix "
                                    "the ceiling lies entirely outside the data. Refits on other "
                                    "windows put t₀ anywhere from 22 to 55 years, and twice "
                                    "the fit stopped asking for a ceiling at all."
                                ),

                                html.H6("The fitted ceiling tracks the cycle, not a ceiling"),
                                html.P(
                                    "Line each window’s ceiling up against where that window "
                                    "happened to end relative to its own power-law trend. The "
                                    "residual column is the mean log₁₀ residual over the "
                                    "window’s final year:"
                                ),
                                _spl_cycle_phase_table(),
                                html.P(
                                    "The three windows that ended above trend produced the two “no "
                                    "ceiling at all” fits and the largest finite one; the three "
                                    "that ended below trend produced the three smallest, including "
                                    "both rejections. t₀ is behaving as a cycle-phase detector."
                                ),

                                html.H6("The four-year cycle, and why it doesn’t rescue the ceiling"),
                                html.P(
                                    "On the current window the plain test does reject a ceiling of "
                                    "infinity — 13.6536 against a boundary-corrected 5% critical "
                                    "value of 2.7055. That is not evidence the ceiling is real. The "
                                    "same test, on the same code, did not reject on four of the last "
                                    "six windows: the verdict is unstable across windows, and a "
                                    "statistic that flips on 1.1% more data is a symptom of that "
                                    "instability rather than a finding."
                                ),
                                html.P(
                                    "The obvious objection is that the four-year cycle masks the "
                                    "curvature, and it deserves to be taken seriously because it "
                                    "half-works. Adding a calendar sinusoid to the mean removes 44.3% "
                                    "of the variance — against the saturation term’s 0.23% — "
                                    "and lifts the statistic to 151.5 assuming independent residuals, "
                                    "which would reject outright. Three things say not to believe it:"
                                ),
                                html.Ul([
                                    html.Li(
                                        "79% of the improvement accrues in the final two years. "
                                        "That is a terminal excursion, not a systematic bend."
                                    ),
                                    html.Li(
                                        "The cycle-conditional t₀ sat at its 100-year bound for "
                                        "every cutoff through 2024, became finite only in 2025 "
                                        "(33 yr), and is still moving fast (≈19 yr now). That is "
                                        "the opposite of convergence."
                                    ),
                                    html.Li(
                                        "The statistic assumes independent residuals, and these are "
                                        "not. Significance at that level needs the residuals to "
                                        "decorrelate within about 56 days; the measured 30-day "
                                        "residual autocorrelation is 0.90. Scaled to the ≈4.5 "
                                        "cycles of data actually present, 151.5 becomes 0.116 against "
                                        "a critical value of 2.706."
                                    ),
                                ]),

                                html.H6("What that says about the other models"),
                                html.P(
                                    "That comparison is the one actionable thing this model teaches. "
                                    "The cycle carries 44.3% of the variance in log-price; the "
                                    "saturation term carries 0.23% — roughly 190× less. The "
                                    "dominant feature of the residuals is cyclical, not curvature. "
                                    "The log-periodic family (LPPL, HybPPL, EPPL) models that "
                                    "structure explicitly and is where to look for it; SatPL is here "
                                    "to show what the data cannot support, not to add a forecast."
                                ),

                                html.H6("Saturation in log-time, not calendar time"),
                                html.P(
                                    "Substituting t for ln t — the ordinary logistic in years, "
                                    "i.e. saturation on a linear timescale — fits far worse: RMSE "
                                    "0.4452, against 0.2939 for this model and 0.2942 for the plain "
                                    "power law. Whatever bend the data contain is a bend in log-time."
                                ),
                                _clickable_img("/assets/spl_linear_vs_log.png", "700px"),

                                html.H6("In short"),
                                html.P(html.Em(
                                    "Even models that account for Bitcoin’s four-year cycle "
                                    "cannot pin the ceiling down — the apparent bend they find "
                                    "sits right at the edge of the data, where it cannot be "
                                    "distinguished from the current cycle’s swing."
                                )),

                                html.H6("Caveats"),
                                html.Ul([
                                    html.Li(
                                        "No dollar figure on this card is a forecast. Every one is "
                                        "conditional on a saturating shape the data cannot confirm, "
                                        "and every one moves when the data window moves."
                                    ),
                                    html.Li(
                                        "The right summary is not “the power law wins”. It "
                                        "is that L is unidentified, and PL is the parsimonious "
                                        "representative of a family of ceilings this data cannot tell "
                                        "apart — absence of evidence, not evidence of absence."
                                    ),
                                    html.Li(
                                        "The quantile bands are the usual shrinking-σ spread "
                                        "about the fitted median. They carry none of the uncertainty "
                                        "in L, which dominates everything else at long horizons."
                                    ),
                                ]),
                            ], className="mathjax_ignore"),
                        ], title="Saturating Power Law (diagnostic)", item_id="mi-spl"),
                        # -- Stock-to-Flow (instantaneous flow -- variant) --
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
Same fit as S2F, but flow is the **instantaneous** issuance rate:

$$\text{flow}(t) = \text{reward}(t) \times \text{blocks/day} \times 365.25$$

the current block reward annualized. Because the reward halves at a single
block, this flow steps discontinuously at each halving.
                            """, mathjax=True, className="mb-3"),

                            html.H6("Method"),
                            html.P(
                                "Identical to the default S2F model except for the flow definition. "
                                "The instantaneous flow annualizes the block reward in effect at time t, "
                                "so S2F roughly doubles the moment a halving occurs and the fitted price "
                                "jumps as a step. This was the original S2F behaviour before the "
                                "trailing-flow default; it is kept as a variant for comparison."
                            ),

                            html.H6("Fitted Coefficients"),
                            _s2f_coeff_table("s2f_inst"),

                            html.H6("Characteristics"),
                            html.Ul([
                                html.Li("Single trajectory (non-quantized) -- no percentile bands."),
                                html.Li("Step-function jumps at each halving as the annualized reward halves."),
                                html.Li("Projects higher than the trailing variant at long horizons "
                                        "because the post-halving flow drop is applied instantly."),
                            ]),
                        ], title="Stock-to-Flow (instantaneous flow)", item_id="mi-s2f-inst"),
    ]
