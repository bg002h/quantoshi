"""Tab 7 — Model Info: detailed documentation of all price models."""

from dash import html, dcc
import dash_bootstrap_components as dbc

import _app_ctx


def _clickable_img(src, max_width="700px"):
    """Image that opens in a lightbox modal on click."""
    return html.Img(
        src=src,
        className="model-info-img",
        n_clicks=0,
        id={"type": "mi-img", "src": src},
        style={"width": "100%", "maxWidth": max_width,
               "borderRadius": "8px", "marginBottom": "16px",
               "cursor": "zoom-in"},
    )


def _model_info_tab():
    return html.Div([
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H5("Price Models & Simulation Reference",
                            className="mb-3 mt-2"),
                    html.P(
                        "Quantoshi uses several price models and a Monte Carlo simulation engine. "
                        "All models operate in log\u2081\u2080 space where t is years since "
                        "July 25, 2009 \u2014 the statistically optimal time origin for the "
                        "power law fit, confirmed independently by multiple researchers. "
                        "This page documents the mathematics, fitted coefficients, and "
                        "methodology behind each.",
                        className="text-muted mb-4",
                    ),
                    dbc.Accordion([

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
                                "A ",
                                html.A("sensitivity sweep", href="/B", style={"color": "#1a6fa8"}),
                                " across slope (4\u20137) and intercept (\u00b12.5) shows: ",
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
                            html.P([
                                "A separate ",
                                html.A("parameter sweep", href="/C", style={"color": "#1a6fa8"}),
                                " tests 49 combinations of the floor percentile (5\u201335%) and "
                                "quantile regression target (5\u201395%), running the full bubble pipeline "
                                "at each point. Results are broadly stable across the grid, reinforcing "
                                "that the model is not over-fitted to one specific parameter choice.",
                            ]),
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
                            _coeff_table([
                                ("\u03b1 (intercept)", "\u22121.175443"),
                                ("\u03b2 (slope)", "5.084045"),
                                ("\u03c3 (residual std)", "~0.302"),
                            ]),
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
                            _coeff_table([
                                ("A (intercept, log\u2081\u2080 USD)", "\u22121.154"),
                                ("B (slope)", "5.080"),
                                ("C (osc. amplitude, log\u2081\u2080)", "0.734"),
                                ("\u03c9 (log-time freq, rad)", "7.559"),
                                ("\u03c6 (phase, rad)", "1.376"),
                                ("D (damping exponent)", "0.608"),
                                ("\u03c3 (residual, log\u2081\u2080)", "0.227"),
                                ("R\u00b2", "0.9780"),
                            ]),

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
                            _coeff_table([
                                ("A (intercept, log\u2081\u2080 USD)", "\u22121.131"),
                                ("B (slope)", "5.039"),
                                ("C\u2081 (primary amp, log\u2081\u2080)", "0.706"),
                                ("\u03c9\u2081 (log-time freq, rad)", "7.378"),
                                ("\u03c6\u2081 (phase, rad)", "1.582"),
                                ("D (damping, primary only)", "0.566"),
                                ("C\u2082 (secondary amp, log\u2081\u2080)", "0.169"),
                                ("\u03c9\u2082 (log-time freq, rad)", "20.902"),
                                ("\u03c6\u2082 (phase, rad)", "\u22121.154"),
                                ("\u03c3 (residual, log\u2081\u2080)", "0.193"),
                                ("R\u00b2", "0.9840"),
                            ]),

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
                            _coeff_table([
                                ("A (intercept, log\u2081\u2080 USD)", "\u22121.213"),
                                ("B (slope)", "5.111"),
                                ("C (amp, log\u2081\u2080)", "0.282"),
                                ("\u03c9_cal (calendar freq)", "1.766 rad/yr"),
                                ("T (= 2\u03c0/\u03c9)", "3.56 yr"),
                                ("\u03c6 (phase, rad)", "\u22122.283"),
                                ("D (damping)", "0.010 (at lower bound)"),
                                ("\u03c3 (residual, log\u2081\u2080)", "0.222"),
                                ("R\u00b2", "0.9789"),
                            ]),

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
                            _coeff_table([
                                ("A", "\u22121.147"),
                                ("B", "5.052"),
                                ("C\u2081 (log-periodic amplitude)", "0.690"),
                                ("\u03c9_log", "7.420"),
                                ("\u03c6\u2081", "1.453"),
                                ("D (damping on log term)", "0.708"),
                                ("C\u2082 (linear-periodic amplitude)", "0.233"),
                                ("\u03c9_cal", "1.733 rad/yr"),
                                ("T (= 2\u03c0/\u03c9_cal)", "3.63 years"),
                                ("\u03c6\u2082", "\u22121.922"),
                                ("R\u00b2", "0.9889"),
                                ("\u03c3", "0.161"),
                                ("C\u2082/C\u2081 ratio", "0.34"),
                            ]),

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
                        ], title="HybPPL (Hybrid Log+Linear PPL)", item_id="mi-hybppl"),

                        # ── 3f. HybPPL (excess) ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A_{\text{sup}} + B_{\text{sup}}\log_{10}(t)
  + a_0 + C_1\,t^{-D}\cos\!\big(\omega_{\log}\ln t + \phi_1\big)
  + C_2\cos\!\big(\omega_{\text{cal}}\,t + \phi_2\big)$$
                            """, mathjax=True, className="mb-3"),

                            html.H6("Motivation"),
                            html.P(
                                "HybPPL (excess) decouples the trend from the oscillation. Instead of "
                                "co-fitting a power-law support with the log- and calendar-periodic terms "
                                "(as HybPPL does), this variant fixes the support to the BM power-law "
                                "(A_sup, B_sup) and fits only the 8 oscillation parameters against the "
                                "BM-excess residual. This eliminates cross-contamination between the "
                                "support fit and the oscillation fit."
                            ),

                            html.H6("Fitted Coefficients"),
                            html.P(
                                "A_sup and B_sup are inherited from the BM support line; the 8 "
                                "oscillation parameters are refit daily via fit_hybppl_excess.py.",
                                className="text-muted small",
                            ),
                            _coeff_table(_hybppl_ex_rows()),

                            html.H6("Comparison to HybPPL"),
                            html.P(
                                "HybPPL and HybPPL (excess) share the same functional form for the "
                                "oscillation terms. They differ in the fitting procedure: HybPPL co-fits "
                                "all parameters (including the power-law trend), while HybPPL (excess) "
                                "fixes the trend to the BM support line first, then fits oscillations on "
                                "the residual. See the /F experimental page for side-by-side diagnostics."
                            ),

                            html.P(
                                "Refitted daily via tools/fit_hybppl_excess.py along with the rest of "
                                "the model pipeline in update_prices.py.",
                                className="text-muted small",
                            ),
                        ], title="HybPPL (excess)", item_id="mi-hybppl-ex"),

                        # ── 3g. HybPPL (excess DD) ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = A_{\text{sup}} + B_{\text{sup}}\log_{10}(t)
  + a_0 + C_1\,t^{-D_1}\cos\!\big(\omega_{\log}\ln t + \phi_1\big)
  + C_2\,t^{-D_2}\cos\!\big(\omega_{\text{cal}}\,t + \phi_2\big)$$

Solved for price:

$$\text{price}(t) = 10^{A_{\text{sup}}} \cdot t^{B_{\text{sup}}} \cdot 10^{a_0}
  \cdot 10^{\,C_1 t^{-D_1} \cos(\omega_{\log} \ln t + \varphi_1)}
  \cdot 10^{\,C_2 t^{-D_2} \cos(\omega_{\text{cal}} t + \varphi_2)}$$
                            """, mathjax=True, className="mb-3"),

                            html.H6("Motivation"),
                            html.P(
                                "HybPPL (excess DD) is the double-damped variant of HybPPL (excess). "
                                "It adds a separate damping exponent D\u2082 to the calendar-periodic "
                                "(halving cycle) oscillator. This tests whether the ~4-year halving "
                                "cycle is a permanent feature of Bitcoin\u2019s price dynamics or whether "
                                "it too is decaying over time as Bitcoin matures. If D\u2082 converges "
                                "near zero, the data does not support calendar damping \u2014 the halving "
                                "cycle appears permanent."
                            ),

                            html.H6("Fitted Coefficients"),
                            html.P(
                                "A_sup and B_sup are inherited from the BM support line; the 9 "
                                "oscillation parameters (including D\u2081, D\u2082) are refit daily "
                                "via fit_hybppl_excess_dd.py.",
                                className="text-muted small",
                            ),
                            _coeff_table(_hybppl_ex_dd_rows()),

                            html.H6("Interpretation"),
                            html.P([
                                "D\u2082 \u2248 0.001 suggests the calendar oscillator is effectively "
                                "undamped \u2014 the halving cycle appears permanent. The extra parameter "
                                "does not meaningfully improve the fit, confirming that HybPPL (excess) "
                                "with its undamped calendar term is the more parsimonious choice. This "
                                "model exists primarily as a diagnostic: if D\u2082 ever drifts "
                                "significantly above zero in future refits, it would signal that the "
                                "halving cycle is beginning to fade.",
                            ]),

                            html.P(
                                "Refitted daily via tools/fit_hybppl_excess_dd.py along with the rest of "
                                "the model pipeline in update_prices.py.",
                                className="text-muted small",
                            ),
                        ], title="HybPPL (excess DD \u2014 Double Damped)", item_id="mi-hybppl-ex-dd"),

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
                            _coeff_table([
                                ("\u03b1 (intercept)", "0.240277"),
                                ("\u03b2 (slope)", "0.317792 per year"),
                                ("\u03c3 (residual std)", "0.553180"),
                                ("R\u00b2", "0.871"),
                            ]),

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

                        # ── 5. Stock-to-Flow ──
                        dbc.AccordionItem([
                            html.H6("Formula"),
                            dcc.Markdown(r"""
$$\log_{10}(\text{price}) = \alpha + \beta \cdot \log_{10}(\text{S2F})$$

Solved for price:

$$\text{price}(t) = 10^{\,\alpha} \cdot \text{S2F}(t)^{\,\beta}$$

$$\text{S2F}(t) = \frac{\text{stock}(t)}{\text{annual flow}(t)}$$

where stock = cumulative BTC mined, flow = annual new BTC issuance.
                            """, mathjax=True, className="mb-3"),

                            html.H6("Method"),
                            html.P([
                                "The Stock-to-Flow model, popularized by ",
                                html.A("Plan B",
                                       href="https://twitter.com/100trillionUSD",
                                       target="_blank", rel="noopener noreferrer"),
                                ", posits that Bitcoin\u2019s scarcity (measured by the S2F ratio) drives "
                                "its price. S2F doubles at each halving as the flow is cut in half while "
                                "the stock continues growing. The model fits a linear regression in "
                                "log\u2081\u2080(S2F) vs log\u2081\u2080(price) space.",
                            ]),

                            html.H6("Fitted Coefficients"),
                            _coeff_table([
                                ("\u03b1 (intercept)", "\u22120.631492"),
                                ("\u03b2 (slope)", "2.974994"),
                            ]),

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
                                    "Produces step-function jumps at each halving as S2F doubles abruptly."
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
                            html.P([
                                "EF interval trend: +0.20 yr/cycle (nearly flat, ~3.7 yr next). ",
                                "BM interval trend: +0.37 yr/cycle (lengthening, ~5.7 yr next). ",
                                "This is visible in the ",
                                html.A("EF sensitivity sweep", href="/BB", style={"color": "#1a6fa8"}),
                                " compared to the ",
                                html.A("BM sweep", href="/B", style={"color": "#1a6fa8"}),
                                ".",
                            ]),

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

                    ], id="model-info-accordion", start_collapsed=True, flush=True),
                ]),
                width={"size": 10, "offset": 1},
            )
        ),
        # Lightbox modal for enlarged images
        dbc.Modal([
            dbc.ModalBody(
                html.Img(id="mi-lightbox-img", style={"width": "100%"}),
                style={"padding": "0", "backgroundColor": "#1a1a2e"},
            ),
        ], id="mi-lightbox", size="xl", centered=True, is_open=False),
    ], className="p-3")


def _hybppl_ex_rows():
    """Live coefficient table for HybPPL (excess) — pulls from model class."""
    m = _app_ctx.M
    mdl = _app_ctx.PRICE_MODELS.get("hybppl_ex")
    if mdl is None:
        return [("(model not loaded)", "\u2014")]
    T_cal = 2 * 3.14159265358979 / mdl._W_cal
    return [
        ("A_sup (BM intercept, log\u2081\u2080 USD)",  f"{m.support_intercept:.4f}"),
        ("B_sup (BM slope)",                           f"{m.support_slope:.4f}"),
        ("a\u2080 (constant offset, log\u2081\u2080)", f"{mdl._a0:.4f}"),
        ("C\u2081 (damped amplitude, log\u2081\u2080)", f"{mdl._C1:.4f}"),
        ("\u03c9_log (log-time freq, rad)",            f"{mdl._W_log:.4f}"),
        ("\u03c6\u2081 (phase, rad)",                   f"{mdl._PHI1:.4f}"),
        ("D (damping exponent)",                        f"{mdl._D:.4f}"),
        ("C\u2082 (undamped amplitude, log\u2081\u2080)", f"{mdl._C2:.4f}"),
        ("\u03c9_cal (calendar freq, rad/yr)",          f"{mdl._W_cal:.4f}"),
        ("T_cal (calendar period)",                    f"{T_cal:.2f} yr"),
        ("\u03c6\u2082 (phase, rad)",                   f"{mdl._PHI2:.4f}"),
    ]


def _hybppl_ex_dd_rows():
    """Live coefficient table for HybPPL (excess DD) — pulls from model class."""
    m = _app_ctx.M
    mdl = _app_ctx.PRICE_MODELS.get("hybppl_ex_dd")
    if mdl is None:
        return [("(model not loaded)", "\u2014")]
    T_cal = 2 * 3.14159265358979 / mdl._W_cal
    return [
        ("A_sup (BM intercept, log\u2081\u2080 USD)",  f"{m.support_intercept:.4f}"),
        ("B_sup (BM slope)",                           f"{m.support_slope:.4f}"),
        ("a\u2080 (constant offset, log\u2081\u2080)", f"{mdl._a0:.4f}"),
        ("C\u2081 (damped amplitude, log\u2081\u2080)", f"{mdl._C1:.4f}"),
        ("\u03c9_log (log-time freq, rad)",            f"{mdl._W_log:.4f}"),
        ("\u03c6\u2081 (phase, rad)",                   f"{mdl._PHI1:.4f}"),
        ("D\u2081 (log damping exponent)",              f"{mdl._D1:.4f}"),
        ("C\u2082 (cal amplitude, log\u2081\u2080)",   f"{mdl._C2:.4f}"),
        ("\u03c9_cal (calendar freq, rad/yr)",          f"{mdl._W_cal:.4f}"),
        ("T_cal (calendar period)",                    f"{T_cal:.2f} yr"),
        ("\u03c6\u2082 (phase, rad)",                   f"{mdl._PHI2:.4f}"),
        ("D\u2082 (cal damping exponent)",              f"{mdl._D2:.6f}"),
    ]


def _bm_rows():
    """Live coefficient table for Bubble Model — support + band parameters."""
    m = _app_ctx.M
    if m is None:
        return [("(model not loaded)", "\u2014")]
    return [
        ("A_sup (support intercept, log\u2081\u2080 USD)", f"{m.support_intercept:.4f}"),
        ("B_sup (support slope)",                         f"{m.support_slope:.4f}"),
        ("\u03c3\u2080 up (base vol, upper bands)",
            f"{getattr(m, 'bm_sigma0_up', 0.085):.4f}"),
        ("\u03b1 up (vol shrinkage, upper)",
            f"{getattr(m, 'bm_alpha_up', 0.132):.4f}"),
        ("\u03c3\u2080 down (base vol, lower bands)",
            f"{getattr(m, 'bm_sigma0_down', 0.075):.4f}"),
        ("\u03b1 down (vol shrinkage, lower)",
            f"{getattr(m, 'bm_alpha_down', 0.218):.4f}"),
        ("R\u00b2 (composite on full history)",          f"{float(m.bm_r2):.4f}"),
        ("N future bubbles (max)",                       f"{int(m.n_future_max)}"),
    ]


def _ef_rows():
    """Live coefficient table for Empirical Floor Model."""
    ef = _app_ctx.PRICE_MODELS.get("ef")
    if ef is None:
        return [("(EF model not loaded \u2014 model_data_ef.pkl missing)", "\u2014")]
    return [
        ("Support intercept (log\u2081\u2080 USD)",  f"{ef._intercept:.4f}"),
        ("Support slope",                             f"{ef._slope:.4f}"),
        ("\u03c3\u2080 up (base vol, upper)",        f"{ef._sigma0_up:.4f}"),
        ("\u03b1 up (vol shrinkage, upper)",         f"{ef._alpha_up:.4f}"),
        ("\u03c3\u2080 down (base vol, lower)",      f"{ef._sigma0_down:.4f}"),
        ("\u03b1 down (vol shrinkage, lower)",       f"{ef._alpha_down:.4f}"),
        ("R\u00b2 (composite)",                       f"{float(ef._bm_r2):.4f}"),
        ("N future bubbles (max)",                   f"{int(ef._n_future_max)}"),
    ]


def _coeff_table(rows):
    """Small two-column coefficient table."""
    return html.Table([
        html.Tbody([
            html.Tr([
                html.Td(html.Strong(label), style={"paddingRight": "20px",
                         "paddingBottom": "4px", "whiteSpace": "nowrap"}),
                html.Td(html.Code(value) if not isinstance(value, str) or
                         any(c in value for c in "0123456789.\u2212") else value,
                         style={"paddingBottom": "4px"}),
            ]) for label, value in rows
        ])
    ], style={"marginBottom": "12px", "fontSize": "13px"})


def _qr_table():
    """Quantile regression coefficient table from live model data."""
    m = _app_ctx.M
    if m is None:
        return html.P("Model data not loaded.", className="text-muted")
    # Show a representative subset
    show_qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    rows = []
    for q in show_qs:
        if q in m.qr_fits:
            f = m.qr_fits[q]
            rows.append(html.Tr([
                html.Td(f"Q{q*100:.0f}%", style={"paddingRight": "12px"}),
                html.Td(html.Code(f"{f['intercept']:.6f}"),
                         style={"paddingRight": "12px"}),
                html.Td(html.Code(f"{f['slope']:.6f}")),
            ]))
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Quantile", style={"paddingRight": "12px"}),
            html.Th("\u03b1 (intercept)", style={"paddingRight": "12px"}),
            html.Th("\u03b2 (slope)"),
        ])),
        html.Tbody(rows),
    ], style={"marginBottom": "12px", "fontSize": "13px"})


def _comparison_table():
    """Model comparison summary table."""
    hdr_style = {"paddingRight": "12px", "paddingBottom": "6px",
                 "borderBottom": "1px solid #555", "fontSize": "12px"}
    cell_style = {"paddingRight": "12px", "paddingBottom": "4px",
                  "paddingTop": "4px", "fontSize": "12px",
                  "borderBottom": "1px solid #333"}
    return html.Table([
        html.Thead(html.Tr([
            html.Th("", style=hdr_style),
            html.Th("QR (Bubble)", style=hdr_style),
            html.Th("Power Law", style=hdr_style),
            html.Th("LPPL", style=hdr_style),
            html.Th("Exponential", style=hdr_style),
            html.Th("S2F", style=hdr_style),
            html.Th("Monte Carlo", style=hdr_style),
        ])),
        html.Tbody([
            html.Tr([
                html.Td(html.Strong("Type"), style=cell_style),
                html.Td("Quantile regression", style=cell_style),
                html.Td("OLS + Gaussian shift", style=cell_style),
                html.Td("Damped log-periodic + Gaussian", style=cell_style),
                html.Td("OLS (linear time) + Gaussian", style=cell_style),
                html.Td("Supply-driven regression", style=cell_style),
                html.Td("Stochastic simulation", style=cell_style),
            ]),
            html.Tr([
                html.Td(html.Strong("Bands"), style=cell_style),
                html.Td("Independent slopes", style=cell_style),
                html.Td("Parallel (same slope)", style=cell_style),
                html.Td("Parallel + oscillating", style=cell_style),
                html.Td("Parallel (very wide)", style=cell_style),
                html.Td("None (single line)", style=cell_style),
                html.Td("Fan (P1\u2013P95)", style=cell_style),
            ]),
            html.Tr([
                html.Td(html.Strong("Captures cycles"), style=cell_style),
                html.Td("No", style=cell_style),
                html.Td("No", style=cell_style),
                html.Td("Yes (damped)", style=cell_style),
                html.Td("No", style=cell_style),
                html.Td("Halvings only", style=cell_style),
                html.Td("Empirically", style=cell_style),
            ]),
            html.Tr([
                html.Td(html.Strong("Parameters"), style=cell_style),
                html.Td("2 per quantile", style=cell_style),
                html.Td("3 (\u03b1, \u03b2, \u03c3)", style=cell_style),
                html.Td("7 (A,B,C,\u03c9,\u03c6,D,\u03c3)", style=cell_style),
                html.Td("3 (\u03b1, \u03b2, \u03c3)", style=cell_style),
                html.Td("2 (\u03b1, \u03b2)", style=cell_style),
                html.Td("5\u00d75 matrix", style=cell_style),
            ]),
            html.Tr([
                html.Td(html.Strong("Dash style"), style=cell_style),
                html.Td("Solid", style=cell_style),
                html.Td("Dotted", style=cell_style),
                html.Td("Dash-dot", style=cell_style),
                html.Td("Long dash-dot", style=cell_style),
                html.Td("Long dash", style=cell_style),
                html.Td("Fan shading", style=cell_style),
            ]),
        ]),
    ], style={"marginBottom": "16px", "width": "100%", "borderCollapse": "collapse"})


def _regime_data_tables():
    """Build summary tables + transition matrices for all asset classes."""
    try:
        from data.asset_matrices import load_asset_matrices
        matrices = load_asset_matrices()
    except Exception as e:
        return html.P(f"Data not available: {e}", className="text-muted")

    sections = []
    _cell = {"fontSize": "11px", "padding": "2px 6px", "border": "1px solid #ddd",
             "textAlign": "right"}
    _hdr = {**_cell, "fontWeight": "bold", "backgroundColor": "#f5f5f0", "textAlign": "center"}

    for key in ("equity", "bond", "tres_short", "tres_med", "tres_long"):
        m = matrices.get(key)
        if not m:
            continue

        label = m.get("label", key)
        n_bins = len(m["bin_means"])

        # Summary stats
        sections.append(html.H6(f"{label}", style={"marginTop": "12px"}))
        sections.append(html.Table([
            html.Tbody([
                html.Tr([
                    html.Td("Observations", style={**_cell, "fontWeight": "bold"}),
                    html.Td(f"{m['n_obs']} months", style=_cell),
                    html.Td("Ann. Return", style={**_cell, "fontWeight": "bold"}),
                    html.Td(f"{m['ann_return']*100:.1f}%", style=_cell),
                    html.Td("Ann. Vol", style={**_cell, "fontWeight": "bold"}),
                    html.Td(f"{m['ann_vol']*100:.1f}%", style=_cell),
                ]),
            ]),
        ], style={"marginBottom": "4px", "borderCollapse": "collapse"}))

        # Regime bins: mean return + volatility per bin
        bin_header = [html.Th("Regime", style=_hdr)] + [
            html.Th(f"Bin {i+1}", style=_hdr) for i in range(n_bins)
        ]
        bin_means_row = [html.Td("Mean mo. return", style={**_cell, "fontWeight": "bold"})] + [
            html.Td(f"{m['bin_means'][i]*100:+.2f}%", style=_cell) for i in range(n_bins)
        ]
        bin_vols_row = [html.Td("Mo. volatility", style={**_cell, "fontWeight": "bold"})] + [
            html.Td(f"{m['bin_vols'][i]*100:.2f}%", style=_cell) for i in range(n_bins)
        ]
        bin_edges_row = [html.Td("Return range", style={**_cell, "fontWeight": "bold"})] + [
            html.Td(f"{m['bin_edges'][i]*100:+.1f} to {m['bin_edges'][i+1]*100:+.1f}%",
                     style={**_cell, "fontSize": "10px"})
            for i in range(n_bins)
        ]

        sections.append(html.Table([
            html.Thead(html.Tr(bin_header)),
            html.Tbody([
                html.Tr(bin_means_row),
                html.Tr(bin_vols_row),
                html.Tr(bin_edges_row),
            ]),
        ], style={"marginBottom": "4px", "borderCollapse": "collapse", "width": "100%"}))

        # Transition matrix
        trans = m["trans"]
        t_header = [html.Th("From \u2193 To \u2192", style=_hdr)] + [
            html.Th(f"Bin {j+1}", style=_hdr) for j in range(n_bins)
        ]
        t_rows = []
        for i in range(n_bins):
            cells = [html.Td(f"Bin {i+1}", style={**_cell, "fontWeight": "bold"})]
            for j in range(n_bins):
                p = trans[i, j]
                # Color-code: darker for higher probability
                bg = f"rgba(230,126,34,{min(p * 1.5, 0.4):.2f})" if p > 0.1 else "transparent"
                cells.append(html.Td(f"{p:.0%}", style={**_cell, "backgroundColor": bg}))
            t_rows.append(html.Tr(cells))

        sections.append(html.Details([
            html.Summary("Transition matrix", style={"fontSize": "12px", "cursor": "pointer",
                                                      "color": "#888", "marginBottom": "4px"}),
            html.Table([
                html.Thead(html.Tr(t_header)),
                html.Tbody(t_rows),
            ], style={"borderCollapse": "collapse", "width": "100%"}),
        ], style={"marginBottom": "12px"}))

    return html.Div(sections)
