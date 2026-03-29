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

                        # ── 1. Quantile Regression (Bubble Model) ──
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

                            html.H6("Bubble Model Overlay"),
                            html.P([
                                "The gold composite line on Tab 1 is a separate model layered on top of QR. "
                                "It fits a parameterized trapezoid shape (rise, plateau, decay) to each "
                                "historical Bitcoin bubble (2013, 2017, 2021, 2025) in log-residual space "
                                "above a power-law support line (bottom ~25% of data). The composite is:"
                            ]),
                            dcc.Markdown(r"""
$$\text{composite}(t) = 10^{\,\log_{10}(\text{support}(t)) \;+\; \sum_{i} \text{bubble}_i(t)}$$

Future bubbles are extrapolated from the trend in historical bubble parameters (amplitude decreasing, width increasing).
                            """, mathjax=True),
                        ], title="Quantile Regression (Bubble Model)", item_id="mi-qr"),

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
                                ("A (intercept)", "\u22121.155084"),
                                ("B (slope)", "5.081303"),
                                ("C (oscillation amplitude)", "0.734286"),
                                ("\u03c9 (log-frequency)", "7.563897"),
                                ("\u03c6 (phase)", "1.371053"),
                                ("D (damping exponent)", "0.608874"),
                                ("\u03c3 (residual std)", "0.226960"),
                                ("R\u00b2", "0.9801"),
                            ]),

                            html.H6("Interpretation"),
                            html.Ul([
                                html.Li([
                                    html.Strong("\u03c9 \u2248 8.89:"),
                                    " The oscillation period in ln(t) space corresponds to ~4-year "
                                    "cycles, aligning with Bitcoin\u2019s halving schedule.",
                                ]),
                                html.Li([
                                    html.Strong("D \u2248 0.70:"),
                                    " Oscillation amplitude decays as t\u207b\u2070\u00b7\u2077. "
                                    "Each successive bubble is smaller relative to the trend \u2014 "
                                    "Bitcoin is getting less volatile over time.",
                                ]),
                                html.Li([
                                    html.Strong("B \u2248 5.70:"),
                                    " The underlying power-law slope matches QR and PL closely.",
                                ]),
                                html.Li([
                                    html.Strong("\u03c3 = 0.217:"),
                                    " Smaller than PL\u2019s 0.302 because the oscillatory term "
                                    "absorbs variance that PL attributes to noise. This means LPPL\u2019s "
                                    "quantile bands are narrower.",
                                ]),
                            ]),
                        ], title="Log-Periodic Power Law (LPPL)", item_id="mi-lppl"),

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
                            html.H6("Convergence Narrative"),
                            html.P(
                                "The steeper support line means bubble amplitudes decay faster. "
                                "Predicted future bubbles converge on the support rapidly \u2014 "
                                "implying that the classic 4-year halving-driven boom/bust cycle "
                                "is approaching its end, with Bitcoin transitioning to a more "
                                "mature, lower-volatility asset. See the FAQ for the full "
                                "derivation."
                            ),
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
