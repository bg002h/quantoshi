"""Tab 8 — FAQ layout."""

from dash import html
import dash_bootstrap_components as dbc

from layout.common import _STYLE_ADDR_CELL, _STYLE_ADDR_CODE
from colors import CODE_BG, DANGER_HIGHLIGHT, _hex_alpha

_FAQ = [
    {
        "q": "What does the Share button do?",
        "a": (
            "It's cooler than you might expect. Suppose you've customized a graph and want to "
            "show someone else your plot. You could save an image — or you could send them a "
            "link that takes them directly to Quantoshi with all of your customized configuration "
            "already applied. Every control across all tabs is encoded into the URL, so the "
            "recipient sees exactly what you see. You can optionally include your Stack Tracker "
            "lots in the link too. Your link history is saved in your browser so you can revisit "
            "or re-share any configuration you've generated."
        ),
    },
    {
        "q": "What is quantile regression?",
        "a": (
            "Ordinary regression finds the line that best fits the average of your data. "
            "Quantile regression does something more powerful: it fits a separate line for any "
            "percentile you choose. The 50th percentile line (the median) splits the data in half "
            "— as many points above as below. The 5th percentile line fits the bottom 5% of the "
            "data, and the 95th fits the top 5%. On Quantoshi, quantile regression is applied to "
            "the historical log-log relationship between time and Bitcoin price, giving you a "
            "family of curves that describe not just where Bitcoin has typically been, but how "
            "extreme the highs and lows have historically gotten. The percentile of your purchase "
            "price tells you how cheap or expensive that entry was relative to all historical "
            "prices at that point in Bitcoin's life."
        ),
    },
    {
        "q": "Is Quantoshi predicting future Bitcoin price? What guarantees do I have this will be true?",
        "a": (
            "No. Quantoshi is extrapolating by quantile regression of a power law model. This is math. "
            "Prediction is what YOU do with this math. And as far as a guarantee, there is none (any "
            "guarantee would be worth how much you were required to pay to use this software, which is "
            "free!). Now, if I would have made Quantoshi in 2016, I would have been surprisingly "
            "accurate in 2026, but I would caution anyone working with any dataset against extrapolating "
            "much beyond 1/3 of the dataset. Bitcoin is 17 years old. 17/3 is 5 to a physicist, 6 to a "
            "mathematician, 5.67 to an engineer, and 5-2/3 in US Army Mixed Number Format. Use "
            "caution extrapolating beyond 6 or so years."
        ),
    },
    {
        "q": "What time origin does Quantoshi use, and why?",
        "a": html.Span([
            "All models need a Day Zero \u2014 the point where t=0 in the power law. "
            "Several candidates were considered:",
            html.Br(), html.Br(),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Date", style={"paddingRight": "12px"}),
                    html.Th("Event", style={"paddingRight": "12px"}),
                    html.Th("Slope"),
                    html.Th("R\u00b2", style={"paddingLeft": "12px"}),
                ])),
                html.Tbody([
                    html.Tr([html.Td("Jan 3, 2009"), html.Td("Genesis block mined"), html.Td("5.69"), html.Td("0.9615")]),
                    html.Tr([html.Td("Jan 9, 2009"), html.Td("Bitcoin v0.1 released"), html.Td("5.68"), html.Td("0.9616")]),
                    html.Tr([html.Td("Jul 25, 2009"), html.Td("Optimal (see below)"), html.Td("5.08"), html.Td("0.9630")]),
                    html.Tr([html.Td("Nov 18, 2009"), html.Td("Power law: 1 block reward = $0.01"), html.Td("4.69"), html.Td("0.9620")]),
                    html.Tr([html.Td("Apr 2, 2010"), html.Td("Power law: 1 BTC = $0.01"), html.Td("4.23"), html.Td("0.9521")]),
                    html.Tr([html.Td("Aug 24, 2010"), html.Td("Power law: 1 BTC = $0.10"), html.Td("3.85"), html.Td("0.9324")]),
                ]),
            ], style={"fontSize": "13px", "marginBottom": "12px", "borderCollapse": "collapse"}),
            html.Br(),
            html.Strong("Why does it matter? A practical example."),
            " It is currently 2026 and Bitcoin trades at ~$70,000. Ten years ago, "
            "in 2016, the price was ~$700 \u2014 exactly 100\u00d7 less. You might think "
            "two points a decade apart are enough to pin down the power law slope. "
            "They are not.",
            html.Br(), html.Br(),
            "In log\u2081\u2080 space, 100\u00d7 is always a ",
            html.Strong("rise"),
            " of 2.0 units (because 10\u00b2 = 100). But the ",
            html.Strong("run"),
            " \u2014 the horizontal distance those 10 years span \u2014 depends entirely "
            "on where t = 0 is:",
            html.Br(), html.Br(),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Genesis date", style={"paddingRight": "12px"}),
                    html.Th("t in 2016", style={"paddingRight": "12px"}),
                    html.Th("t in 2026", style={"paddingRight": "12px"}),
                    html.Th("Run (\u0394log\u2081\u2080t)", style={"paddingRight": "12px"}),
                    html.Th("Slope (rise/run)"),
                ])),
                html.Tbody([
                    html.Tr([html.Td("2015"), html.Td("1 yr"), html.Td("11 yr"),
                             html.Td("1.04"), html.Td(html.Strong("1.9"))]),
                    html.Tr([html.Td("2009.6 (optimal)"), html.Td("6.4 yr"), html.Td("16.4 yr"),
                             html.Td("0.41"), html.Td(html.Strong("4.9"))]),
                    html.Tr([html.Td("2000"), html.Td("16 yr"), html.Td("26 yr"),
                             html.Td("0.21"), html.Td(html.Strong("9.5"))]),
                ]),
            ], style={"fontSize": "13px", "marginBottom": "12px", "borderCollapse": "collapse"}),
            html.Br(),
            "The same 100\u00d7 price increase over the same 10 years implies "
            "\u03b2 = 1.9, 4.9, or 9.5 depending on Day Zero. This is because "
            "logarithms compress large numbers: the interval t = 1\u219211 yr spans "
            "1.04 log-units, but t = 16\u219226 yr spans only 0.21 log-units \u2014 "
            "five times narrower. A narrower run with the same rise means a steeper slope.",
            html.Br(),
            html.Img(src="/assets/genesis_slope_example.png",
                     style={"width": "100%", "maxWidth": "900px", "borderRadius": "8px",
                            "marginTop": "8px", "marginBottom": "8px"}),
            html.Br(),
            "This is why Quantoshi invests heavily in determining the optimal "
            "genesis date \u2014 a shift of even a few months changes the implied slope "
            "and all downstream projections.",
            html.Br(), html.Br(),
            html.Strong("How the optimal date is determined."),
            " The same quantile regression lines, evaluated from two different "
            "time origins, produce visibly different fits \u2014 especially at the "
            "earliest price points where the time offset is a larger fraction of t:",
            html.Br(),
            html.Img(src="/assets/genesis_qr_comparison.jpg",
                     style={"width": "100%", "maxWidth": "800px", "borderRadius": "8px",
                            "marginTop": "8px", "marginBottom": "8px"}),
            html.Br(),
            "The Q50% line crosses $1 million roughly 7 months earlier with the "
            "genesis-block origin (Aug 2034) vs. the optimal origin (Mar 2035) \u2014 "
            "a meaningful shift from an identical set of fitted coefficients. "
            "The orange-circled region shows where the choice of Day Zero has "
            "the largest impact: the earliest prices sit noticeably farther from "
            "the regression lines when the time origin is wrong.",
            html.Br(), html.Br(),
            "Quantoshi uses ",
            html.Strong("July 25, 2009"),
            " as the date of t=0 \u2014 the date when people first started using "
            "Bitcoin like money \u2014 throughout the site.",
            html.Br(), html.Br(),
            "It should come as no surprise that there was a period of time during "
            "which Bitcoin was treated unlike money; initially, bitcoin wasn\u2019t "
            "treated like money at all. In fact, block 0 (the Genesis block) created "
            "50 unspendable bitcoins due to a bug and block 1 wasn\u2019t mined until "
            "5 days later \u2014 clearly Satoshi wasn\u2019t treating bitcoin as money at "
            "this point. I suspect it wasn\u2019t until summer break that people started "
            "using or at least thinking of Bitcoin as money, but there is simply no "
            "price data available at this time.",
            html.Br(), html.Br(),
            "Five independent lines of evidence converge on this date:",
            html.Br(), html.Br(),
            html.Strong("1. Residual autocorrelation (Durbin-Watson): "),
            "Sweeping 546 candidate genesis dates, July 25 produces the least "
            "autocorrelated residuals (DW = 0.004, ranking #1 of 546 candidates). "
            "While still far from the ideal 2.0 \u2014 due to bubble cycle structure, "
            "not wrong model choice \u2014 this date minimizes the systematic bias.",
            html.Br(), html.Br(),
            html.Strong("2. Out-of-sample prediction: "),
            "Fitting on partial data (train up to 2016, 2019, or 2022) and predicting "
            "3 years forward, July 20\u201325 minimizes forward prediction error, ranking "
            "#11 of 546. The model trained with this origin generalizes best to unseen data.",
            html.Br(), html.Br(),
            html.Strong("3. Slope stability across time windows: "),
            "Fitting OLS on 6 expanding windows (2010\u20132014 through 2010\u20132024), "
            "early September 2009 gives the most consistent slope (\u03c3 = 0.099). "
            "July 25 ranks #80 \u2014 top 15%. A stable slope means the power law "
            "relationship holds regardless of how much data you include.",
            html.Br(), html.Br(),
            html.Strong("4. Windowed support-line fit (bubble-free): "),
            "Extracting only the bottom 5% of prices in each 4-year sliding window "
            "removes all bubble effects. A power law fit to these support-only points "
            "finds an optimal origin of July 28, 2009 with R\u00b2 = 0.9903 \u2014 higher "
            "than the full-data fit (0.963). Repeating with 2-year and 3-year windows, "
            "and with bottom 10% and 20%, all converge on late June to early August 2009. "
            "This confirms the date is where Bitcoin\u2019s ",
            html.I("floor price"),
            " begins following a power law, independent of bubble peaks.",
            # ── BEGIN JUL25 ANALYSIS (change_origin.py toggles this block) ──
            html.Br(), html.Br(),
            html.Strong("4b. Log-time binned minimum support: "),
            "To address over-representation of recent data in log-time, the time axis "
            "is divided into 34 equal-width bins in log\u2081\u2080(t) and the minimum price "
            "in each bin is taken. This gives each era equal weight regardless of data "
            "density. The optimal genesis is June 25, 2009 (R\u00b2 = 0.984); July 28 "
            "ranks #3 of 944 candidates.",
            html.Br(),
            html.Img(src="/assets/support_logbin34.png",
                     style={"width": "100%", "maxWidth": "700px", "borderRadius": "8px",
                            "marginTop": "8px", "marginBottom": "8px"}),
            html.Br(),
            html.Strong("5. Hand-drawn Q10 support line: "),
            "A line drawn by eye through two support points \u2014 the end of the "
            "~$0.065 flat price run (Oct 2010, t\u22481.18 yr) and the Dec 2022 bear "
            "market bottom ($16,500, t\u224813.4 yr) \u2014 produces slope = 5.11 with "
            "exactly 10% of all historical prices below it. Sweeping genesis dates to "
            "find which origin keeps this line at Q10%, the answer is July 26, 2009 "
            "\u2014 confirming the same week as all other methods.",
            html.Br(),
            html.Img(src="/assets/support_handdrawn_q10.png",
                     style={"width": "100%", "maxWidth": "700px", "borderRadius": "8px",
                            "marginTop": "8px", "marginBottom": "8px"}),
            html.Br(), html.Br(),
            html.Strong("6. Floor-to-peaks convergence: "),
            "Dividing the time axis into 34 equal log-time bins and fitting power laws "
            "to the minimum, P5, median, P95, and maximum price in each bin reveals that "
            "all five percentiles follow nearly the same slope: 5.02 (floor) to 5.14 (P95). "
            "The gap between peaks and floor narrows over time \u2014 Bitcoin\u2019s volatility "
            "is decreasing while the underlying power law growth rate (~5.08) remains "
            "constant across the entire price distribution.",
            html.Br(),
            html.Img(src="/assets/floor_to_peaks.png",
                     style={"width": "100%", "maxWidth": "700px", "borderRadius": "8px",
                            "marginTop": "8px", "marginBottom": "8px"}),
            html.Br(),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Percentile", style={"paddingRight": "12px"}),
                    html.Th("Slope", style={"paddingRight": "12px"}),
                    html.Th("Optimal Genesis", style={"paddingRight": "12px"}),
                    html.Th("R\u00b2"),
                ])),
                html.Tbody([
                    html.Tr([html.Td("Bin max (peaks)"), html.Td("5.09"), html.Td("Nov 23, 2009"), html.Td("0.949")]),
                    html.Tr([html.Td("Bin P95"), html.Td("5.14"), html.Td("Nov 17, 2009"), html.Td("0.955")]),
                    html.Tr([html.Td("Bin median"), html.Td("5.11"), html.Td("Oct 16, 2009"), html.Td("0.972")]),
                    html.Tr([html.Td("Bin P5"), html.Td("5.03"), html.Td("Jun 1, 2009"), html.Td("0.978")]),
                    html.Tr([html.Td("Bin min (floor)"), html.Td("5.02"), html.Td("Jun 25, 2009"), html.Td("0.984")]),
                ]),
            ], style={"fontSize": "13px", "marginBottom": "12px", "borderCollapse": "collapse"}),
            "The optimal genesis shifts from June 2009 (floor) to November 2009 (peaks) "
            "\u2014 about 5 months apart. The floor \u201cstarts\u201d earlier because it is "
            "anchored by the very first prices; the peaks start later because the first "
            "real bubble (2011) didn\u2019t happen until later. July 25 sits in the middle "
            "of this range, balancing floor and peak behavior.",
            html.Br(), html.Br(),
            html.Strong("7. Q5% temporal uniformity: "),
            "A true 5th-percentile support line should have 5% of prices below it "
            "at every point in time, not just 5% overall. Sweeping 870 candidate "
            "genesis dates from 2008 through mid-2010 and optimizing the support "
            "line slope and intercept to distribute the below-line points as "
            "uniformly as possible across 15 log-time bins, the optimal genesis "
            "is June 13, 2009. The top candidates cluster in June 2009 \u2014 "
            "converging on the same window as the floor-support analysis above "
            "from a completely independent angle. Perfect uniformity is "
            "unachievable with a single power law because bubble cycles create "
            "structural clustering: during bull runs zero prices touch the floor, "
            "and after crashes many do.",
            html.Br(),
            html.Img(src="/assets/genesis_uniform_q5.jpg",
                     style={"width": "100%", "maxWidth": "700px", "borderRadius": "8px",
                            "marginTop": "8px", "marginBottom": "8px"}),
            # ── END JUL25 ANALYSIS ──
            html.Br(), html.Br(),
            "Blockchain analysis of 2009 shows virtually zero economic "
            "transactions \u2014 only ~62 blocks/day of mining (below the 144/day target) "
            "with near-zero on-chain transfers. The first dollar-denominated transactions "
            "appear in late 2009, with real economic usage beginning mid-2010. "
            "The July 2009 date is not an economic event but the mathematical inflection "
            "where Bitcoin\u2019s price behavior begins following a power law.",
        ]),
    },
    {
        "q": "Is the power law a single regime, or are there multiple phases?",
        "a": html.Span([
            "A natural question: if residuals are highly autocorrelated "
            "(Durbin-Watson \u2248 0.004, far from the ideal 2.0), does that imply "
            "Bitcoin has undergone structural regime changes \u2014 different power "
            "law slopes in different eras?",
            html.Br(), html.Br(),
            "We investigated this with four standard econometric tests, using "
            "all available price data (2010\u20132026, N = 5,714) with no fit-window "
            "filter:",
            html.Br(), html.Br(),
            html.Strong("Box-Cox \u03bb sweep: "),
            "Before testing regimes, we checked whether log-log is even the "
            "right coordinate system. Sweeping the Box-Cox parameter \u03bb from "
            "\u22123 to +3, the optimal transformation is \u03bb = 0.012 \u2014 "
            "essentially zero (pure log-log). RMSE improves by only 0.03% over "
            "\u03bb = 0. The power law is the correct functional form; no "
            "stretched-exponential or other transformation fits better.",
            html.Br(), html.Br(),
            html.Strong("Rolling Regression: "),
            "A 2-year sliding OLS window shows the power law slope oscillating "
            "between roughly \u22128 and +15, but it mean-reverts toward ~5.0 after "
            "every cycle. If there were permanent regime changes, the slope would "
            "shift and stay shifted. Instead, it always comes back \u2014 consistent "
            "with one regime plus bubble cycles.",
            html.Br(), html.Br(),
            html.Strong("Bai-Perron structural breakpoints: "),
            "The BIC-optimal segmentation finds 6 breakpoints (2011-01, 2013-03, "
            "2013-11, 2014-10, 2017-05, 2020-11). But these all land at bubble "
            "peaks and troughs. Segment slopes range from 2.1 to 5.2, bracketing "
            "the full-sample slope of 5.08. These are cycle boundaries, not "
            "permanent regime shifts.",
            html.Br(), html.Br(),
            html.Strong("Chow Test: "),
            "F-tests at each breakpoint are highly significant (p \u2248 0), but "
            "this is expected whenever bubble cycles temporarily alter the slope. "
            "Statistical significance does not imply permanence.",
            html.Br(), html.Br(),
            html.Strong("CUSUM (cumulative sum of residuals): "),
            "The CUSUM wanders above and below zero in a wave pattern \u2014 "
            "classic bubble oscillation. There is no sharp one-way departure "
            "that would indicate a true structural break.",
            html.Br(), html.Br(),
            html.Img(src="/assets/regime_analysis.jpg",
                     style={"width": "100%", "maxWidth": "800px", "borderRadius": "8px",
                            "marginTop": "4px", "marginBottom": "8px"}),
            html.Br(),
            html.Img(src="/assets/regime_analysis_table.jpg",
                     style={"width": "100%", "maxWidth": "800px", "borderRadius": "8px",
                            "marginTop": "4px", "marginBottom": "8px"}),
            html.Br(), html.Br(),
            html.Strong("Conclusion: "),
            "Bitcoin follows a single power law regime (\u03bb = 0, slope \u2248 5.1). "
            "The DW \u2248 0 and structural breaks reflect cyclic bubble deviations, "
            "not permanent regime changes. The slope oscillates with market "
            "cycles but always reverts to the same long-term trend.",
        ]),
    },
    {
        "q": "What do additional robustness tests reveal about the power law?",
        "a": html.Span([
            "We conducted twelve additional investigations to stress-test the power law "
            "and its oscillatory structure:",
            html.Br(), html.Br(),
            html.Strong("1. Time-varying frequency: "),
            "We tested whether the log-periodic frequency \u03c9\u22487.4 drifts over time "
            "by fitting \u03c9(t) = \u03c9\u2080 + \u03c9\u2081\u00b7ln(t). Result: \u03c9\u2081 = \u22120.056 \u2248 0. "
            "The frequency is constant \u2014 not drifting. This validates the fixed-\u03c9 assumption "
            "used in all LPPL/HybPPL models.",
            html.Br(), html.Br(),
            html.Strong("2. Heteroscedastic volatility (now implemented): "),
            "Residual analysis shows \u03c3(t) = \u03c3\u2080\u00b7t^(\u2212\u03b1) \u2014 volatility shrinks "
            "over time. Windowed \u03c3: 0.150 (2010\u201313) \u2192 0.109 (2013\u201317) \u2192 0.139 "
            "(2017\u201321) \u2192 0.092 (2021\u201326). All models now use ",
            html.Strong("shrinking, asymmetric \u03c3 bands"),
            ": \u03c3_up(t) and \u03c3_down(t) are fitted separately, so quantile bands are "
            "wider in the early era (high volatility) and narrower in the current era "
            "(low volatility). Typical \u03b1 \u2248 0.35\u20130.50 across models, meaning bands "
            "shrink 2\u20133\u00d7 from t=2yr to t=16yr.",
            html.Br(), html.Br(),
            html.Strong("3. Cross-validation: "),
            "Training on data before a cutoff date and testing on data after reveals "
            "that all models overfit (train R\u00b2\u22480.99, test R\u00b2\u22480.64). However, "
            "the Entropy PPL (EPPL 1d+1u) generalizes most consistently across 5 different "
            "cutoff dates \u2014 the entropy envelope\u2019s zero-crossing prevents dead oscillations "
            "from propagating forward. HybPPL collapses at the 2021 cutoff (test R\u00b2=0.19) "
            "while EPPL holds at 0.64.",
            html.Br(), html.Br(),
            html.Strong("4. Symbolic regression: "),
            "PySR (genetic programming) was given raw data with no assumptions and "
            "independently discovered log\u2081\u2080(price) = 2.174\u00b7ln(t) \u2212 1.097 = "
            "5.01\u00b7log\u2081\u2080(t) \u2212 1.10 \u2014 the power law with \u03b2\u22485.0, "
            "from scratch. Higher-complexity equations found cosine terms but with "
            "minimal improvement, confirming oscillations are secondary to the trend. "
            "The genesis date is an input assumption, not something the algorithm discovers.",
            html.Br(), html.Br(),
            html.Strong("5. Wavelet scalogram: "),
            "Continuous wavelet transform of detrended log\u2081\u2080(price) shows the "
            "halving cycle (T\u22483.6yr) as the overwhelmingly dominant signal across "
            "all scales and all time periods. Early-era power is 2\u00d7 the late era, "
            "confirming oscillations are fading. The log-periodic signal is invisible "
            "in wavelets because it\u2019s a chirp in calendar time \u2014 at t=2yr it looks "
            "like T\u22481.7yr, at t=16yr it looks like T\u224813.6yr.",
            html.Br(), html.Br(),
            html.Strong("6. Asymmetric oscillations: "),
            "Fitting separate damping exponents for rising vs falling phases reveals "
            "that falls damp 3.5\u00d7 faster than rises (D_rise=0.56, D_fall=2.0). "
            "FOMO builds slowly; panic resolves quickly. BIC improves by 120 over "
            "symmetric damping.",
            html.Br(), html.Br(),
            html.Strong("7. Quantile-specific oscillations: "),
            "Quantile regression with oscillatory terms shows the log-periodic amplitude "
            "scales with quantile: Q5% (floor) has C=0.07, Q50% has C=0.15, Q95% (peaks) "
            "has C=0.27. Upper quantiles oscillate 4\u00d7 larger than the floor. The power "
            "law slope also varies: \u03b2=5.41 at Q5% vs \u03b2=4.87 at Q95% \u2014 the floor "
            "grows steeper than the peaks, confirming volatility compression over time.",
            html.Br(), html.Br(),
            html.Strong("8. Fourier extrapolation: "),
            "Top 20 FFT modes of the detrended signal project: $724K (2027), $85K (2028), "
            "$294K (2029), $677K (2030), $157K (2031). These are unreliable because Fourier "
            "extrapolation has no damping \u2014 it projects historical amplitudes unchanged. "
            "Useful for showing cyclical structure, not for price targets.",
            html.Br(), html.Br(),
            html.Strong("9. Multi-scale decomposition: "),
            "Fitting the power law at daily, weekly, monthly, quarterly, and yearly "
            "resolution gives \u03b2 = 5.08, 5.09, 5.12, 5.18, 5.34 with R\u00b2 > 0.96 "
            "at every scale. The power law is self-similar. The slight \u03b2 increase at "
            "coarser scales reflects aggregation removing downside volatility.",
            html.Br(), html.Br(),
            html.Strong("10. Multivariate analysis: "),
            "We tested whether on-chain network metrics improve the power law, "
            "using daily hash rate, active addresses, transaction count, UTXO set "
            "size, and difficulty from blockchain.com. Result: even with real "
            "network data, all 5 metrics combined add only \u0394R\u00b2 = +0.006 "
            "(0.963 \u2192 0.969). Hash rate is the strongest individual predictor "
            "(+0.0017), followed by difficulty (+0.0014). Metcalfe\u2019s law is "
            "close: price \u221d addresses\u00b9\u00b7\u2079\u00b9 (expect 2.0), but "
            "addresses alone (R\u00b2=0.845) explain far less than time (R\u00b2=0.963). "
            "When controlling for time, addresses add only +0.0006. The fundamental "
            "problem is collinearity: hash rate, addresses, and difficulty all grow "
            "as approximate power laws of time, making it impossible to distinguish "
            "\u2018price follows time\u2019 from \u2018price follows adoption which "
            "grows with time.\u2019 The power law is predominantly temporal.",
            html.Br(), html.Br(),
            html.Strong("11. Phase-space reconstruction: "),
            "Takens embedding of the detrended log-price with optimal delay "
            "\u03c4 = 256 days (0.70 yr). False nearest neighbors drop to 4.2% at "
            "embedding dimension 3, meaning the attractor resolves in 3D. "
            "Correlation dimension \u2248 2.0 (computed at embedding dimensions 4\u20135). "
            "Bitcoin\u2019s detrended price dynamics live on a roughly 2-dimensional "
            "attractor \u2014 consistent with two dominant oscillatory modes "
            "(log-periodic + halving cycle) found by PCA, EMD, and DMD.",
            html.Br(), html.Br(),
            html.Strong("12. Neural network comparison: "),
            "MLPs of varying depth (10 to 100-50-20 nodes) were trained on the "
            "same data. All catastrophically overfit: train R\u00b2 = 0.993\u20130.999, "
            "test R\u00b2 = \u221220.3 to \u22123.1. With 91\u20136,891 parameters, neural "
            "networks achieve higher training R\u00b2 but wildly negative test R\u00b2 \u2014 "
            "far worse than EPPL\u2019s 16 parameters (test R\u00b2 \u2248 0.64). "
            "Parametric models with domain-informed structure are vastly more "
            "efficient than unconstrained function approximators on this dataset.",
        ]),
    },
    {
        "q": "What is the BM Empirical Floor model?",
        "a": html.Span([
            "The BM Empirical Floor is an alternate bubble model with a steeper "
            "power law support line, anchored to two observed bear-market "
            "floor prices:",
            html.Br(), html.Br(),
            html.Strong("Anchor 1: "),
            "October 5, 2010 ($0.06) \u2014 the end of Bitcoin\u2019s initial "
            "flat-price run, the earliest observable floor.",
            html.Br(),
            html.Strong("Anchor 2: "),
            "February 5, 2026 ($58,000) \u2014 selected to balance temporal "
            "uniformity of below-line data points with the width of the "
            "time window considered.",
            html.Br(), html.Br(),
            "The second anchor reflects a trade-off: a higher price captures "
            "more data points below the line (better statistics) but clusters "
            "them in recent bear markets; a lower price distributes the "
            "below-line points more evenly across eras but with fewer total "
            "points. The Kolmogorov-Smirnov (KS) statistic measures this "
            "uniformity \u2014 lower is better. The standard bubble model "
            "support (KS = 0.581) clusters its below-line points in "
            "2\u20133 bear markets; the Empirical Floor (KS = 0.425) "
            "achieves a more balanced distribution while keeping ~12.6% "
            "of prices below the line.",
            html.Br(), html.Br(),
            html.Img(src="/assets/support_4way_loglog.jpg",
                     style={"width": "100%", "maxWidth": "800px",
                            "borderRadius": "8px",
                            "marginTop": "4px", "marginBottom": "8px"}),
            html.Br(),
            html.Strong("End of the 4-year cycle: "),
            "The steeper support (slope 5.31 vs 5.13) means each successive "
            "bubble sits lower above the floor. When bubble shapes are fitted "
            "and extrapolated, future bubbles converge on the support line "
            "much faster than in the standard model \u2014 implying that the "
            "classic halving-driven boom/bust cycle is approaching its end. "
            "Bitcoin would transition from a volatile, cycle-driven asset to "
            "one with steadily diminishing oscillations around a steep but "
            "smooth power law growth path.",
        ]),
    },
    {
        "q": "What is the Unfairly Cheap Line?",
        "a": html.Span([
            "Only 7 days in Bitcoin\u2019s entire history have ever fallen below "
            "the Q0.1% quantile regression line. Of those 7, only two are "
            "separated by more than a year: ",
            html.Strong("September 21, 2015 ($229)"),
            " and ",
            html.Strong("January 1, 2023 ($16,905)"),
            ".",
            html.Br(), html.Br(),
            "These two points, spanning 7.3 years across two completely different "
            "bear markets, constrain the power law slope to within 0.05% \u2014 "
            "effectively defining a ",
            html.Strong("unique"),
            " power law floor for Bitcoin (slope = 5.51, intercept = \u22121.99). "
            "No other pair of widely-separated points in Bitcoin\u2019s history "
            "can serve as a two-point floor.",
            html.Br(), html.Br(),
            html.Strong("What made these prices unfairly cheap?"),
            " Both breaches were caused by cascading failures of trust, not "
            "failures of Bitcoin itself:",
            html.Br(), html.Br(),
            html.Strong("September 2015 ($229): "),
            "The 2013 bubble had peaked on Mt. Gox, which at the time handled "
            "roughly 70% of all Bitcoin trading worldwide. In February 2014, "
            "Mt. Gox halted withdrawals and filed for bankruptcy, revealing "
            "that 850,000 BTC (\u223c$450M at the time) had been lost or stolen "
            "\u2014 the largest exchange failure in Bitcoin\u2019s history until "
            "FTX. For many early adopters, Mt. Gox ",
            html.I("was"),
            " Bitcoin; its collapse shattered confidence and triggered a "
            "prolonged bear market. By September 2015 \u2014 19 months after "
            "the Mt. Gox bankruptcy \u2014 the community was further divided by "
            "the block size debate (the \u201cscaling wars\u201d), which had "
            "fractured developer consensus. Media coverage was overwhelmingly "
            "negative, pronouncing Bitcoin dead for the 89th time. Trading "
            "volume had collapsed with no obvious successor exchange filling "
            "the void Mt. Gox left. Yet the network kept mining blocks every "
            "10 minutes, and adoption quietly grew beneath the noise.",
            html.Br(), html.Br(),
            html.Strong("November 2022 \u2013 January 2023 ($16,169\u2013$16,905): "),
            "A chain reaction of centralized failures. Terra/Luna\u2019s "
            "algorithmic stablecoin imploded in May 2022, wiping out $40B and "
            "taking down Three Arrows Capital (a $10B hedge fund) in June. "
            "The contagion spread to lenders Celsius and Voyager, then "
            "culminated in the FTX collapse in November 2022 \u2014 the "
            "third-largest crypto exchange, revealed as an $8B fraud. Each "
            "failure was a failure of people and institutions built on top of "
            "Bitcoin, not of the protocol itself. The network processed every "
            "transaction without interruption.",
            html.Br(), html.Br(),
            "In both cases, the price recovered to new all-time highs within "
            "2\u20133 years. Anyone who bought at the Unfairly Cheap Line "
            "captured the deepest value Bitcoin has ever offered.",
            html.Br(), html.Br(),
            "The line is available as a toggle "
            "(\u201cUnfairly Cheap Line\u201d) in the Bubble tab\u2019s Display "
            "section. The Model Scanner also shows how far above the UCL the "
            "current price sits.",
        ]),
    },
    {
        "q": "What is the difference between Quantile Regression and Markov Chain Monte Carlo?",
        "a": html.Span([
            "Quantile Regression (QR) fits smooth percentile curves to the historical log-log "
            "relationship between time and Bitcoin price. It is deterministic: given a percentile "
            "and a date, it returns exactly one price. The computation is a single matrix solve — "
            "fast enough to run in your browser in milliseconds. QR tells you ",
            html.I("where"),
            " a given percentile has historically fallen, but it assumes the future follows the "
            "same smooth power-law trend.",
            html.Br(), html.Br(),
            "Markov Chain Monte Carlo (MCMC) simulation is fundamentally different. It models "
            "Bitcoin's price as a random walk governed by a transition matrix estimated from "
            "historical returns. At each time step the simulation draws a random move from the "
            "learned distribution, so every run produces a different price path — and each path "
            "consists of hundreds of sequential transitions. To get stable "
            "statistics (median, percentile bands), we need to repeat this hundreds of times — "
            "Quantoshi runs several hundred simulations per scenario.",
            html.Br(), html.Br(),
            "This is why MCMC is computationally expensive: each path requires stepping through "
            "hundreds of time periods in sequence, and we repeat this hundreds of times. A single QR "
            "lookup is O(1); a single MCMC fan requires O(paths \u00d7 periods) floating-point "
            "operations — hundreds of paths times hundreds of steps per path.",
            html.Br(), html.Br(),
            "To keep this tractable, Quantoshi pre-computes a cache of over 45,000 scenarios "
            "covering different entry percentiles, time horizons, withdrawal amounts, inflation "
            "rates, and stack sizes. The full cache occupies roughly 834 MB of RAM, loaded at "
            "server startup from compressed arrays on disk. A compiled Cython engine generates "
            "the cache offline; at runtime, lookups are instantaneous.",
            html.Br(), html.Br(),
            "When you choose parameters outside the pre-computed cache — a custom withdrawal "
            "amount, a different inflation rate, or a start year we haven't cached — Quantoshi "
            "has to run a fresh simulation on the server in real time. This means generating "
            "hundreds of full price paths from scratch, each stepping through hundreds of "
            "transitions, and then computing withdrawal overlays on top of them. That is why "
            "custom simulations cost a small lightning payment and may take a few seconds to "
            "return: you are paying for dedicated compute time that cannot be amortized across "
            "other users.",
        ]),
    },
    {
        "q": "Why does this look so bad on [my device / browser]?",
        "a": (
            "I have only been able to test this on Linux and iPhone. I'm a physician, not a "
            "programmer, so I really couldn't tell you anyhow! But if anyone can send me a "
            "screenshot I'd love to fix it for their platform!"
        ),
    },
    {
        "q": "Why do some high-percentile extrapolated quantile projection lines cross in the future?",
        "a": (
            "When a higher-percentile line crosses a lower one (or vice versa), it indicates "
            "a trend in the dataset that renders extrapolation unreliable. For example, because "
            "subsequent Bitcoin bubble peaks have been getting less extreme over time, the 99th "
            "percentile line rises less steeply than the 95th — so much so that these two lines "
            "cross around 2034. It should be noted that the lower-percentile extrapolations "
            "(e.g. the 30th percentile) remain more or less parallel well beyond any reasonable "
            "planning horizon."
        ),
    },
    {
        "q": "Why a Power Law?",
        "a": html.Span([
            "Great question. I did my undergrad in astrophysics and math, so the real question "
            "is: how do you value money? We can't use cash flow analysis on money itself, and "
            "ideally we'd want a scale-invariant model that works at small and large times — "
            "excluding exponential models like the first popular model, the Stock-to-Flow by "
            "the venerable Plan B. Power laws are observed everywhere in nature (literally "
            "everywhere space exists), but for a more detailed discussion please see the ",
            html.A("Scientific Bitcoin Institute",
                   href="https://scientificbitcoininstitute.org/",
                   target="_blank", rel="noopener noreferrer"),
            ". Giovanni Santostasi had the first Bitcoin price model — a Power Law model — "
            "before Plan B's S2F... but as physics people speak differently, it took a while "
            "to catch on :)",
        ]),
    },
    {
        "q": "What is the difference between Bubble Model quantiles and Power Law quantiles?",
        "a": html.Span([
            "Both are Q-percentile power-law fits to Bitcoin\u2019s historical price data, "
            "but they use different regression approaches.",
            html.Br(), html.Br(),
            html.Strong("Bubble Model (QR): "),
            "Quantile regression fits each percentile independently \u2014 every line has "
            "its own slope and intercept. Q25% is positioned so that exactly 25% of "
            "historical data falls below it. Because each percentile is fitted separately, "
            "the spacing between quantile lines can vary and the lines are not necessarily "
            "parallel.",
            html.Br(), html.Br(),
            html.Strong("Power Law (PL): "),
            "Ordinary Least Squares (OLS) regression fits a single line to the mean of "
            "the data, then shifts that line up or down using the residual distribution "
            "to create percentile bands. Q25% is computed as OLS intercept + z\u2080.\u2082\u2085 "
            "\u00d7 \u03c3, where z is the normal distribution\u2019s 25th percentile z-score and "
            "\u03c3 is the standard deviation of the residuals. All PL percentile lines are "
            "parallel in log-log space because they share the same slope.",
            html.Br(), html.Br(),
            "In practice, the two models diverge most at extreme percentiles (Q1%, Q99%) "
            "where the actual price distribution is not symmetric or normally distributed.",
        ]),
    },
    {
        "q": "Can I send you a tip?",
        "a": html.Table([
            html.Tbody([
                html.Tr([html.Td("Bitcoin", style=_STYLE_ADDR_CELL),
                         html.Td(html.Code("bc1qgh6kfnf02uvplq490nyslc7768tnvzftlrw5fe", style=_STYLE_ADDR_CODE))]),
                html.Tr([html.Td("Lightning", style=_STYLE_ADDR_CELL),
                         html.Td(html.Code("lno1pgjrzv34xscxyvrp94jrvdej956rgdnp95ukydt9943rxdpkxucrqvpsv5ury93pqgfffll4jmjf0tffqtx47xt886gzp9fajp3966xz96gm2xj9cqedx", style=_STYLE_ADDR_CODE))]),
                html.Tr([html.Td("Ecash", style=_STYLE_ADDR_CELL),
                         html.Td(html.Code("creqApGF0gaNhdGRwb3N0YWF4QGh0dHBzOi8vY29pbm9zLmlvL2FwaS9lY2FzaC8xMjU0MGIwYS1kNjcyLTQ0NmEtOWI1ZS1iMzQ2NzAwMDBlODJhZ/dhaXgkMTI1NDBiMGEtZDY3M:", style=_STYLE_ADDR_CODE))]),
                html.Tr([html.Td("Liquid BTC", style=_STYLE_ADDR_CELL),
                         html.Td(html.Code("lq1qqfztsa6ffjkspk3qxp4ft8kn2sxu5ja9prn5d9vwuqjjut5g2tzc8rpsgz2pysayplrgemf9dt3vpkqhvsvtkfxvdyk9mlsel", style=_STYLE_ADDR_CODE))]),
                html.Tr([html.Td("Liquid USDt", style=_STYLE_ADDR_CELL),
                         html.Td(html.Code("liquidnetwork:lq1qqfjgl0fvv7a5prqd7d0k4x80kq2v0cngzxuj7hz3pdhuj0xg57tuzk9q0knrsuevsrywqys92ttefak83xzqq6uqmngkkaa74?assetid=ce091c998b83c78bb71a632313ba3760f176", style=_STYLE_ADDR_CODE))]),
            ])
        ], style={"width":"100%","borderCollapse":"collapse","marginTop":"4px"}),
    },
    {
        "q": "I see you modeled up to 3 future bubbles in the first tab... What is your model / how did you model it?",
        "a": (
            "Interesting question. I modeled Bitcoin price as a power law running through the "
            "bottom roughly 30% of the data, and then modeled each bubble separately in "
            "log-log space as something like a trapezoid. I then looked at the shape of each "
            "of the trapezoids and noticed how they changed from a tall triangle, to a "
            "medium-height trapezoid, and then to a very long, short, almost table-like "
            "trapezoid (kinda like a Japanese low table, a chabudai)... Anyhow, in "
            "mathematical terms, I parameterized each bubble and took the trend through time "
            "on each part of each shape and extrapolated that trend (along with the timing "
            "trend) to up to three future bubbles. The result is underwhelming — the bubbles "
            "converge somewhat rapidly on the support trendline... which is part of what "
            "everyone means when they say Bitcoin is getting less volatile over time. I only use "
            "the last three bubbles to extrapolate over; adding the very first bubble massively "
            "screws up the trend, and we were just kids back then, so it shouldn't really "
            "count :)"
        ),
    },
    {
        "q": "Why did you make this?",
        "a": (
            "Everyone needs bitcoin... and Bitcoin is for everyone. The more clearly people "
            "can see the past, the more accurately they can model the future. I'm just doing "
            "a tiny part in helping people see the bright orange future we are racing towards."
        ),
    },
    {
        "q": "What is the Stack-celerator on the DCA tab?",
        "a": html.Span([
            html.Span(
                "It's \u201cActivate Saylor Mode\u201d \u2014 a strategy popularized by Michael Saylor and "
                "MicroStrategy: instead of only dollar-cost averaging, you also borrow money "
                "and use the loan proceeds to buy a lump sum of Bitcoin up front. You then "
                "service the loan from your regular DCA contributions. If Bitcoin appreciates "
                "faster than your interest rate — historically a very safe bet — you end up "
                "with significantly more Bitcoin than plain DCA would have gotten you. "
                "The dashed overlay lines on the chart show your projected stack with the loan "
                "versus without (solid lines). The Stack-celeration factor in the chart title "
                "tells you how many times better the loan strategy performs versus plain DCA "
                "at the median."
            ),
            html.Br(), html.Br(),
            html.Span(
                "Two loan types are available. "
                "Interest-only: you pay just the interest each period and repay the full "
                "principal at the end of each term by selling some Bitcoin — subject to capital "
                "gains tax. "
                "Amortizing: like a standard mortgage, each payment covers both interest and a "
                "slice of the principal. No Bitcoin needs to be sold; the loan is paid off "
                "entirely in fiat from your DCA contributions."
            ),
            html.Br(), html.Br(),
            html.Span(
                "For interest-only loans you can also enable Roll over, which is the more "
                "realistic HODLer approach: instead of selling Bitcoin to repay at term end, "
                "you refinance into a new loan. Your Bitcoin is never sold mid-simulation — "
                "only once, at the very end of the final term."
            ),
            html.Br(), html.Br(),
            html.Span(
                "Please note: it is possible to compound losses by using a loan if you buy "
                "Bitcoin at a high percentile and sell at a lower percentile \u2014 sometimes "
                "even many years in the future, according to quantile regression extrapolations. "
                "Be careful when you choose to predict Bitcoin price from historical price data. "
                "Past performance is not a guarantee of future returns."
            ),
        ]),
    },
    {
        "q": "Do you have a podcast?",
        "a": html.Span([
            "No. I'm ugly and I work in the dark all day (only half of this is true)... "
            "if you are looking for a podcast recommendation, see ",
            html.A("porkopolis.io",
                   href="https://www.porkopolis.io/youtube/",
                   target="_blank", rel="noopener noreferrer"),
            ". Mezenskis has way nicer charts... in fact, I'm waiting to subscribe to "
            "his charts myself!",
        ]),
    },
    {
        "q": "I see you have a Bitcoin price ticker in the header... does this reveal my IP address to a third party?",
        "a": "No. It reveals the IP address of Quantoshi to Binance.",
    },
    {
        "q": "Can I link directly to a tab?",
        "a": html.Span([
            "YES! Just add a /1 to the URL to get to the first tab, a /2 to get to the "
            "second, and so on. For example, ",
            html.A("quantoshi.xyz/4",
                   href="https://quantoshi.xyz/4",
                   target="_blank", rel="noopener noreferrer"),
            " will take you directly to the retirement extrapolator.",
        ]),
    },
    {
        "q": "Can I run my own Quantoshi? Is it Open Source?",
        "a": html.Span([
            "Yes and yes. Quantoshi is free as in beer and free as in speech — BSD-2 licensed "
            "open source code available at ",
            html.A("github.com/bg002h/quantoshi",
                   href="https://github.com/bg002h/quantoshi",
                   target="_blank", rel="noopener noreferrer"),
            ". You are welcome to do anything with the code... or nothing. "
            "There's also a native Linux app compiled as an x86 AppImage there too, "
            "but it's a few iterations behind.",
        ]),
    },
    {
        "q": "If I enter my purchases in Quantoshi as Stack Tracker lots, where does that data go?",
        "a": html.Span([
            html.Strong("Zero server-side storage."),
            " Stack Tracker lots, journey stats, and all settings live in your browser's "
            "localStorage — never on the server. Charts are rendered server-side but no "
            "user-specific data is retained.",
            html.Br(), html.Br(),
            html.Strong("Logging: "),
            "IP addresses are never stored. The nginx log format hardcodes "
            "0.0.0.0 in place of the client IP \u2014 your real address never "
            "touches disk. The template is:",
            html.Br(),
            html.Code(
                "0.0.0.0 - $remote_user [$time_local] \"$request\" "
                "$status $body_bytes_sent \"$http_referer\" \"$http_user_agent\"",
                style={"fontSize": "11px", "display": "block",
                       "margin": "6px 0", "padding": "4px 8px",
                       "background": CODE_BG, "borderRadius": "4px",
                       "wordBreak": "break-all"},
            ),
            "Logs are deleted every 27 days. "
            "No cookies are used. No analytics. No third-party scripts.",
            html.Br(), html.Br(),
            html.Strong("Onion site: "),
            "The ",
            html.A(".onion version",
                   href="http://u5dprelc4ti7xoczb5sbtye6qidlji2l6psmkx35anvxgjyqrkmu32ad.onion",
                   target="_blank", rel="noopener noreferrer"),
            " goes further — all resources (CSS, fonts, block height data) are self-hosted "
            "or routed through .onion endpoints. Your browser makes zero clearnet requests. "
            "The Content-Security-Policy is tightened to block clearnet connections entirely. "
            "Stay dark, Anon.",
        ]),
    },
    {
        "q": "Is there any way to contact someone about this app?",
        "a": html.Span([
            "Email: ",
            html.A("bcg@pm.me", href="mailto:bcg@pm.me"),
            " or Nostr: ",
            html.A("npub1fa8c9pr\u2026qanthnd",
                   href="https://nostr.com/npub1fa8c9prxnrlkdtjl48adfsxyaduz8tas075l2n4f6903y9awjmxqanthnd",
                   target="_blank", rel="noopener noreferrer"),
        ]),
    },
    {
        "q": "Was the 2013\u20132015 bear market unusually bad?",
        "a": html.Span([
            "It felt that way at the time, but statistically, the ",
            html.Strong("depth was completely normal"),
            ". The \u221282.5% drawdown from the November 2013 peak ($1,131) "
            "to the January 2015 trough ($198) sits dead center of "
            "the distribution \u2014 all four major BTC bear markets have "
            "crashed between \u221276% and \u221293%.",
            html.Br(), html.Br(),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Bear Market", style={"paddingRight": "12px"}),
                    html.Th("Peak", style={"paddingRight": "12px"}),
                    html.Th("Trough", style={"paddingRight": "12px"}),
                    html.Th("Drawdown", style={"paddingRight": "12px"}),
                    html.Th("Days to trough", style={"paddingRight": "12px"}),
                    html.Th("Days to new ATH"),
                ])),
                html.Tbody([
                    html.Tr([html.Td("2011"), html.Td("$30"), html.Td("$2"),
                             html.Td("\u221292%"), html.Td("164"), html.Td("623")]),
                    html.Tr([html.Td(html.Strong("2013\u20132015")),
                             html.Td(html.Strong("$1,131")), html.Td(html.Strong("$198")),
                             html.Td(html.Strong("\u221283%")), html.Td(html.Strong("410")),
                             html.Td(html.Strong("1,181"))],
                            style={"background": _hex_alpha(DANGER_HIGHLIGHT, 0.1)}),
                    html.Tr([html.Td("2017\u20132018"), html.Td("$19,389"), html.Td("$3,212"),
                             html.Td("\u221283%"), html.Td("363"), html.Td("1,095")]),
                    html.Tr([html.Td("2021\u20132022"), html.Td("$66,847"), html.Td("$16,238"),
                             html.Td("\u221276%"), html.Td("376"), html.Td("848")]),
                ]),
            ], style={"fontSize": "13px", "marginBottom": "12px", "borderCollapse": "collapse"}),
            html.Br(),
            "What made it ",
            html.Em("feel"),
            " awful was the ",
            html.Strong("duration"),
            ". At 1,181 days (3.2 years) from peak to new all-time high, "
            "it was the longest recovery in Bitcoin\u2019s history \u2014 "
            "roughly twice as long as the 2011 bear and 40% longer than "
            "2021\u20132022. It also spent 346 days below \u221270% drawdown, "
            "more than any other cycle.",
            html.Br(), html.Br(),
            "However, with only four major bear markets in Bitcoin\u2019s "
            "history, none of these durations are statistically significant "
            "outliers. The z-score for recovery time is 1.12 \u2014 about one "
            "standard deviation above the mean, which is longer than average "
            "but well within the range you\u2019d expect from normal variation "
            "in a sample of four. You\u2019d need z > 2 (and a larger sample) "
            "to call it truly unusual.",
            html.Br(), html.Br(),
            html.Strong("Bottom line: "),
            "the 2013\u20132015 bear was the longest but not the deepest, "
            "and with N=4 bear markets there simply isn\u2019t enough data to "
            "label any single cycle as a statistical anomaly. Every BTC bear "
            "market has been brutal by traditional-finance standards.",
        ]),
    },
    {
        "q": "Why are chart downloads corrupted on Tor Browser?",
        "a": html.Span([
            "Tor Browser's canvas fingerprinting protection adds noise to PNG image exports, "
            "producing garbled results. To download a clean chart, click the camera icon in "
            "the chart toolbar and select ",
            html.Strong("SVG"),
            " from the format dropdown. SVG files are vector graphics that scale to any size "
            "and can be opened in any browser or converted to PNG with tools like Inkscape.",
        ]),
    },
]


def _faq_tab():
    items = []
    # Note: item_id uses loop index -- reordering _FAQ entries will break
    # direct links (/9.N) since they reference items by position.
    for i, entry in enumerate(_FAQ):
        items.append(
            dbc.AccordionItem(
                html.Div(entry["a"], className="mb-0"),
                title=entry["q"],
                item_id=f"faq-{i}",
            )
        )
    return html.Div([
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H5("Frequently Asked Questions", className="mb-3 mt-2"),
                    dbc.Accordion(items, id="faq-accordion", start_collapsed=True, flush=True),
                ]),
                width={"size": 8, "offset": 2},
            )
        ),
    ], className="p-3")
