"""Tab 8 — FAQ layout."""

from dash import html
import dash_bootstrap_components as dbc

from layout.common import _STYLE_ADDR_CELL, _STYLE_ADDR_CODE

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
        "q": "What time origin does the model use, and why?",
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
            "The current model uses ",
            html.Strong("July 25, 2009"),
            ". Five independent lines of evidence converge on this date:",
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
            "February 9, 2026 ($70,339) \u2014 selected from several candidates "
            "to maximize the temporal uniformity of below-line data points.",
            html.Br(), html.Br(),
            "The second anchor was chosen using the Kolmogorov-Smirnov (KS) "
            "statistic, which measures how evenly the below-line points are "
            "distributed across time. A good support line should have a "
            "consistent fraction of prices below it in every era, not just "
            "during one or two crashes. The standard bubble model support "
            "(KS = 0.581) clusters its below-line points in 2\u20133 bear "
            "markets; the Empirical Floor (KS = 0.247) distributes them "
            "across 8 of 10 time bins.",
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
            "Bitcoin was 22 months into its post-2013 bear market. The community "
            "was consumed by the block size debate (the \u201cscaling wars\u201d), "
            "which had fractured developer consensus and stalled progress. Media "
            "coverage was overwhelmingly negative, pronouncing Bitcoin dead for "
            "the 89th time. Trading volume had collapsed. Yet the network kept "
            "mining blocks every 10 minutes, and adoption quietly grew beneath "
            "the noise.",
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
            "Server logs are stripped of User-Agent strings and referrer headers. "
            "IP addresses are anonymized daily \u2014 a nightly cron job replaces all IPs "
            "with 0.0.0.0 and saves only aggregate counts (unique visitors, page loads) "
            "to a summary CSV. Logs are deleted every 27 days to prevent aggregation by "
            "authorities, some of whom can demand data thirty days or older without a warrant. "
            "No cookies are tracked. No analytics. No third-party scripts.",
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
    # direct links (/8.N) since they reference items by position.
    for i, entry in enumerate(_FAQ):
        items.append(
            dbc.AccordionItem(
                html.P(entry["a"], className="mb-0"),
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
