# Post-Earnings-Announcement Drift (PEAD) Strategy Research

A self-directed quantitative research project investigating whether equities that report earnings surprises continue to drift in the direction of the surprise over subsequent weeks — and evaluating whether this drift is large, persistent, and consistent enough to trade net of costs.

---

## Background

Fast-moving, high-profile macro headlines and central bank announcements are priced near-instantaneously by low-latency systematic infrastructure. Generating alpha on those horizons requires specialized co-location and execution capabilities.

In contrast, **Post-Earnings-Announcement Drift (PEAD)** is a well-documented academic market anomaly operating on a multi-day to multi-week horizon:
* **The Premise:** Markets systematically underreact to quarterly earnings surprises at the moment of release.
* **The Mechanism:** Prices tend to drift incrementally toward their fundamental post-earnings value over days or weeks following the announcement.

This repository provides an end-to-end framework evaluating this hypothesis: spanning raw event study construction, rules-based backtesting under realistic transaction friction, out-of-sample validation, and machine learning trade selection models.

---

## Repository Structure

| File / Asset | Description |
| :--- | :--- |
| `prices.py` | Core data pipeline. Automatically retrieves historical earnings dates and surprise percentages, downloads price histories, computes benchmark-adjusted abnormal returns ($AR$) against SPY, aligns events to a relative trading-day index ($t_0$), and computes Cumulative Abnormal Return ($CAR$) series. |
| `backtest.py` / `backtest2.py` | Rules-based backtesting engine: enters long positions post-positive surprise and short positions post-negative surprise over a 20-day holding horizon, net of round-trip slippage and execution costs. |
| `ml_dataset.csv` | Structured dataset containing one record per earnings event (`Ticker`, `Event_Date`, `Surprise_Pct`, `CAR_20d`, `Profitable_20d`) exported from the event pipeline. |
| `binary_classification.py` | Logistic regression model evaluating trade probability of profit at +20 trading days using `Surprise_Pct`. |
| `regression.py` | OLS linear regression model estimating expected `CAR_20d` directly from `Surprise_Pct`. |
| `Figure_2.png` | Event study visualization showing mean CAR trajectory across relative event days, segmented by surprise polarity. |
| `Regression plot.png` | Scatter plot of `Surprise_Pct` vs. realized `CAR_20d` overlaid with the fitted regression curve. |

---

## Universe Selection

The backtest universe spans 20 tickers balancing liquid large caps with less-covered mid/small-cap equities (where PEAD effects are traditionally more pronounced due to lower analyst coverage):

* **Mega-Cap Baseline (6):** `AAPL`, `META`, `MSFT`, `TSLA`, `QCOM`, `LII`
* **Mid/Small-Cap Focus (14):** `CROX`, `FIVE`, `WING`, `CAKE`, `TXRH`, `SFM`, `OLLI`, `RH`, `DECK`, `ONON`, `CHDN`, `FND`, `DUOL`, `APPF`

---

## Methodology

1. **Automated Ingestion:** Historical consensus EPS estimates, reported EPS, and surprise percentages are ingested programmatically across multiple historical quarters per ticker.
2. **Window Censoring:** Any recent earnings event lacking a full +20 trading-day forward window is excluded from the sample.
3. **Optimized Market Data Pipeline:** Adjusted closing prices are fetched per ticker in bulk to avoid API rate constraints.
4. **Abnormal Return Calculation:** Daily abnormal return is calculated against the market benchmark:
   $$AR_{i,t} = R_{i,t} - R_{SPY,t}$$
5. **Event-Time Alignment:** Event timestamps are normalized such that the announcement date is defined as $t = 0$.
6. **CAR Aggregation:** Daily abnormal returns are cumulatively summed across the 20-day post-announcement window:
   $$CAR_{i} = \sum_{t=1}^{20} AR_{i,t}$$
7. **Cohort Evaluation:** Mean CAR paths are grouped and evaluated across positive vs. negative surprise cohorts.

---

## Key Findings

### 1. Event Study Results ($n = 435$, 20-Day Horizon)

| Group | $n$ | Mean CAR | $p$-value | Statistical Significance |
| :--- | :--- | :--- | :--- | :--- |
| **All Events** | 435 | +1.21% | 0.0750 | Not Significant ($\alpha = 0.05$) |
| **Positive Surprise** | 339 | **+2.28%** | **0.0025** | **Statistically Significant** |
| **Negative Surprise** | 96 | -2.57% | 0.0966 | Not Significant ($\alpha = 0.05$) |

### 2. Backtest Performance (20-Day Fixed Holding, Net of Fees)
* **Overall Metrics:** 54.7% win rate, **+0.99% mean net return** per trade.
* **Long vs. Short Asymmetry:** The long-only leg delivered **+1.86% mean return**, driving the entire statistical edge. The unhedged short leg averaged **-1.90%**. Because returns were unhedged in an upward-trending market regime, short positions lost money in absolute terms despite relative underperformance vs. SPY.

### 3. Chronological Out-of-Sample Validation
Splitting trades chronologically (rather than randomly) demonstrated that the long-only drift persisted into the out-of-sample period ($p = 0.0070$ in the recent cohort vs. $p = 0.1284$ in the baseline historical cohort).

### 4. Machine Learning & Feature Exploration
* **Current Status:** Inconclusive. Single-feature models (`Surprise_Pct`) using Logistic Regression and OLS Regression showed no statistically significant rank-ordering ability beyond the baseline categorical positive/negative split ($p > 0.45$).
* **Data Anomaly Identified:** Outliers in `Surprise_Pct` (reaching thousands of percent due to near-zero denominators in EPS consensus estimates) skewed parameter estimates and require winsorization/trimming.

---

## Current Status & Limitations

* **Execution Scope:** Only the **long-only** post-positive-surprise strategy (+20 trading days) is currently viable under unhedged conditions.
* **Market-Neutral Redesign Required:** To monetize negative surprises, the short leg must be implemented as a market-neutral pair (Short Equity + Long SPY beta-adjusted hedge).
* **Signal-to-Noise Ratio:** The mean return-to-volatility ratio per trade is low (~0.09). Capturing this edge requires broad position diversification across a multi-ticker portfolio rather than concentrated allocations.
* **Provisional Modeling:** ML results are pending outlier remediation on denominator-distorted EPS surprise metrics. All figures reflect simulated historical backtests without live or paper execution.
