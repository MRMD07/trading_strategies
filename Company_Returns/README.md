Post-Earnings-Announcement Drift (PEAD) Strategy Research

A self-directed research project testing whether stocks that report earnings surprises keep drifting in the direction of the surprise over the following weeks — and whether that drift is large and consistent enough to trade.

Background

Fast-moving, high-profile news (Fed announcements, major macro headlines) is already efficiently priced by low-latency systematic desks — not a realistic edge to chase without their infrastructure. PEAD is a slower, well-documented academic phenomenon that fits a "hold for days, not milliseconds" style of research instead: markets appear to underreact to earnings surprises at the moment of announcement, with the price continuing to drift toward its "correct" level over the following days to weeks.

This repo is an end-to-end walkthrough of testing that hypothesis: from raw event study, to a manual rules-based backtest with real trading costs, to early (so far inconclusive) attempts at improving trade selection with ML.

Repository Structure
File	Purpose
prices.py	Core pipeline. Auto-fetches historical earnings dates and surprise percentages per ticker, downloads price history, computes benchmark-adjusted abnormal returns (AR) against SPY, aligns events by trading day relative to announcement, and produces Cumulative Abnormal Return (CAR) per event.
backtest.py / backtest2.py	Manual, rules-based backtest: long after a positive surprise / short after a negative surprise, holding 20 trading days, net of an assumed round-trip fee/slippage cost.
ml_dataset.csv	One row per earnings event (Ticker, Event_Date, Surprise_Pct, CAR_20d, Profitable_20d), exported from the event study for use in modeling.
binary_classification.py	Logistic regression predicting whether a trade is profitable at +20 trading days, using Surprise_Pct as the sole feature.
regression.py	Linear regression predicting CAR_20d directly, using Surprise_Pct as the sole feature.
Figure_2.png	Mean CAR by event day, split by surprise direction — the core event-study chart.
Regression plot.png	Scatter of Surprise_Pct vs. CAR_20d with the fitted regression line.
Universe

6 mega-cap tickers (AAPL, META, MSFT, TSLA, QCOM, LII) plus 14 mid/small-cap tickers (CROX, FIVE, WING, CAKE, TXRH, SFM, OLLI, RH, DECK, ONON, CHDN, FND, DUOL, APPF), chosen to include names with thinner analyst coverage than the mega-caps — PEAD is better documented in less-followed names.

Method
Pull each ticker's last several quarters of reported EPS estimate, actual EPS, and surprise % automatically (no manual data entry).
Exclude any event whose 20-trading-day post-announcement window hasn't fully elapsed yet.
Download price history once per ticker (not once per event) to avoid hitting exchange data-provider rate limits.
Compute daily abnormal return: AR = Stock Return − SPY Return.
Align every event on trading days relative to its own announcement date (event_day = 0), so events on different calendar dates become comparable.
Cumulative-sum AR within each event to get CAR.
Average CAR by event_day across events, split by surprise direction.
Key Findings

Event study (n=435 events, +20 trading day horizon):

Group	n	Mean CAR	p-value	Result
All events	435	+1.21%	0.0750	Not significant
Positive surprise	339	+2.28%	0.0025	Significant
Negative surprise	96	-2.57%	0.0966	Not significant

Manual backtest (20-day hold, fees included): 54.7% win rate, +0.99% mean net return per trade overall. The long side alone (+1.86% mean) is responsible for essentially all of the edge — the short side lost money on average (-1.90% mean), because CAR measures return relative to SPY; an unhedged short can lose even when the stock underperforms the market, if the market itself is rising.

Out-of-sample check: splitting trades chronologically (not randomly), the long-only edge held up and was, if anything, stronger in the more recent period (p=0.0070 vs. p=0.1284 in the earlier period) — though the later period also includes a broader mix of tickers, which may partly explain the apparent strengthening.

ML modeling (in progress, inconclusive so far): both a logistic classifier and a linear regressor, using Surprise_Pct as the only feature, failed to show a statistically significant ability to rank trades beyond the already-confirmed average effect (top-quartile edges of +1.78pp and -2.17pp respectively, neither significant, p > 0.45 in both cases). A data quality issue was also identified during this stage: a small number of extreme Surprise_Pct outliers (into the thousands of percent, caused by near-zero EPS estimates in the denominator of the surprise calculation) likely distorted both models and should be filtered before re-running.

Current Status / Limitations
Only the long-only rule (buy after a positive surprise, hold 20 days) is currently supported as tradeable by the data.
The negative-surprise signal is real in relative terms but needs a market-neutral redesign (short the stock, hedge with a long SPY position) before it can actually be captured as profit.
All results are backtested, not live — no paper trading has been run yet.
The extreme-outlier Surprise_Pct rows have not yet been filtered out and re-tested; current ML conclusions should be treated as provisional pending that fix.
Return/stdev per trade (~0.09) is low — a real but noisy edge, meaning it would need to be spread across many concurrent positions rather than relied on for any single trade.
