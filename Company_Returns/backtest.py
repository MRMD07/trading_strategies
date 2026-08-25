import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta
import sys

# =====================================================================
# 1. AUTO-BUILD EVENT TABLE (replaces manual data.csv)
# =====================================================================
before_days = 5
after_days = 20
quarters_back = 12  # how many past earnings reports to pull per ticker

# Kept your original 6 mega-caps, added a batch of mid/small-caps.
# These are chosen for: (a) still being liquid enough that yfinance has
# clean price history, (b) more likely to actually miss/beat by a wide
# margin than a mega-cap, since analyst coverage is thinner.
mega_cap_tickers = ['AAPL', 'META', 'MSFT', 'TSLA', 'QCOM', 'LII']
mid_small_cap_tickers = [
    'CROX', 'FIVE', 'WING', 'CAKE', 'TXRH', 'SFM',
    'OLLI', 'RH', 'DECK', 'ONON', 'CHDN', 'FND', 'DUOL', 'APPF'
]
tickers = mega_cap_tickers + mid_small_cap_tickers

event_rows = []
for ticker in tickers:
    try:
        # This single call replaces your whole data.csv: for a given
        # ticker it returns its last N earnings dates along with EPS
        # estimate, reported EPS, and surprise % already computed.
        edates = yf.Ticker(ticker).get_earnings_dates(limit=quarters_back)
    except Exception as e:
        print(f" -> {ticker}: could not fetch earnings dates ({e})")
        continue

    if edates is None or edates.empty:
        print(f" -> {ticker}: no earnings-date data returned")
        continue

    # This table also lists FUTURE scheduled earnings (with no actual
    # reported yet) - those rows have NaN in 'Reported EPS'/'Surprise(%)'.
    # Drop them; we only want events that have already happened.
    edates = edates.dropna(subset=['Reported EPS', 'Surprise(%)'])

    for date, row in edates.iterrows():
        event_rows.append({
            'Ticker': ticker,
            'Event_Date': pd.Timestamp(date).tz_localize(None),
            'Surprise_Pct': row['Surprise(%)']
        })

df_events = pd.DataFrame(event_rows)
print(f"Auto-built {len(df_events)} historical earnings events across "
      f"{df_events['Ticker'].nunique()} tickers\n" + "=" * 50)

events = df_events.to_dict('records')

# =====================================================================
# 2a. DOWNLOAD PRICE HISTORY ONCE PER TICKER (not once per event!)
# =====================================================================
# Your version called yf.Ticker(x).history() twice per event (stock + SPY).
# With 473 events that's ~950 network calls in a burst, which is what
# made Yahoo start throttling and return fake "possibly delisted" errors.
# Fix: figure out the full date span each ticker actually needs across
# ALL its events, download that ONE time, then slice each event's window
# out of the cached DataFrame below. Total calls drops to ~21.
today = pd.Timestamp.now()

ticker_ranges = {}
for event in events:
    t = str(event['Ticker']).strip()
    d = pd.to_datetime(event['Event_Date'])
    lo, hi = d - timedelta(days=35), d + timedelta(days=45)
    if t not in ticker_ranges:
        ticker_ranges[t] = [lo, hi]
    else:
        ticker_ranges[t][0] = min(ticker_ranges[t][0], lo)
        ticker_ranges[t][1] = max(ticker_ranges[t][1], hi)

spy_lo = min(lo for lo, hi in ticker_ranges.values())
spy_hi = min(max(hi for lo, hi in ticker_ranges.values()), today)

print(f"Downloading price history for {len(ticker_ranges)} tickers (one call each)...")
price_cache = {}
for t, (lo, hi) in ticker_ranges.items():
    try:
        hist = yf.Ticker(t).history(start=lo.strftime('%Y-%m-%d'), end=min(hi, today).strftime('%Y-%m-%d'))
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        price_cache[t] = hist
    except Exception as e:
        print(f" -> {t}: failed to download history ({e})")

try:
    spy_hist_full = yf.Ticker('SPY').history(start=spy_lo.strftime('%Y-%m-%d'), end=spy_hi.strftime('%Y-%m-%d'))
    spy_hist_full.index = pd.to_datetime(spy_hist_full.index).tz_localize(None)
except Exception as e:
    print(f"CRITICAL: could not download SPY history ({e})")
    sys.exit()

print("Done downloading. Processing events...\n" + "=" * 50)

# =====================================================================
# 2b. PROCESS EACH EVENT (slices from the cache above - no new downloads)
# =====================================================================
all_event_data = []

for event in events:
    ticker = str(event['Ticker']).strip()
    event_date = pd.to_datetime(event['Event_Date'])
    surprise = float(event['Surprise_Pct'])

    if today < (event_date + timedelta(days=28)):
        continue  # post-earnings window hasn't fully happened yet

    if ticker not in price_cache:
        continue

    start_date = (event_date - timedelta(days=35))
    end_date = (event_date + timedelta(days=45))

    try:
        stock_hist = price_cache[ticker].loc[start_date:end_date]
        spy_hist = spy_hist_full.loc[start_date:end_date]

        if stock_hist.empty or spy_hist.empty:
            continue

        merged_chunk = pd.DataFrame({
            'Stock_Close': stock_hist['Close'],
            'SPY_Close': spy_hist['Close']
        }).dropna()

        if merged_chunk.empty:
            continue

        merged_chunk['Stock_Return'] = merged_chunk['Stock_Close'].pct_change()
        merged_chunk['SPX_Return'] = merged_chunk['SPY_Close'].pct_change()
        merged_chunk['AR'] = merged_chunk['Stock_Return'] - merged_chunk['SPX_Return']

        merged_chunk['Ticker'] = ticker
        merged_chunk['Event_Date'] = event_date
        merged_chunk['Surprise_Pct'] = surprise
        merged_chunk['Date_Window'] = f"{start_date.date()}_to_{end_date.date()}"

        merged_chunk = merged_chunk.reset_index().rename(columns={'index': 'Date'})
        merged_chunk = merged_chunk.sort_values('Date').reset_index(drop=True)

        match = merged_chunk[merged_chunk['Date'] >= event_date]
        if match.empty:
            continue

        anchor_idx = match.index[0]
        merged_chunk['event_day'] = merged_chunk.index - anchor_idx
        merged_chunk = merged_chunk[
            (merged_chunk['event_day'] >= -before_days) & (merged_chunk['event_day'] <= after_days)
        ]

        all_event_data.append(merged_chunk)

    except Exception as e:
        print(f" -> Error on {ticker} {event_date.date()}: {e}")

if not all_event_data:
    print("CRITICAL: no valid events processed.")
    sys.exit()

merged_df = pd.concat(all_event_data, ignore_index=True)
merged_df['CAR'] = merged_df.groupby(['Ticker', 'Date_Window'])['AR'].cumsum()

# =====================================================================
# 3. SAMPLE SIZE CHECK - this is what last chart was missing
# =====================================================================
# Count unique EVENTS (not rows) per group. Always look at this number
# before trusting a curve - a smooth line from n=3 means nothing.
n_pos = merged_df.loc[merged_df['Surprise_Pct'] > 0, ['Ticker', 'Date_Window']].drop_duplicates().shape[0]
n_neg = merged_df.loc[merged_df['Surprise_Pct'] < 0, ['Ticker', 'Date_Window']].drop_duplicates().shape[0]
print(f"Positive-surprise events: {n_pos}")
print(f"Negative-surprise events: {n_neg}")

# =====================================================================
# 4. AGGREGATION & CHART
# =====================================================================
avg_car_all = merged_df.groupby('event_day')['CAR'].mean().reset_index()
avg_car_pos = merged_df[merged_df['Surprise_Pct'] > 0].groupby('event_day')['CAR'].mean().reset_index()
avg_car_neg = merged_df[merged_df['Surprise_Pct'] < 0].groupby('event_day')['CAR'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(avg_car_all['event_day'], avg_car_all['CAR'],
         label=f'All Events (n={n_pos + n_neg})', color='black', linewidth=2)
if not avg_car_pos.empty:
    plt.plot(avg_car_pos['event_day'], avg_car_pos['CAR'],
              label=f'Positive Surprise (n={n_pos})', color='green', linewidth=2)
if not avg_car_neg.empty:
    plt.plot(avg_car_neg['event_day'], avg_car_neg['CAR'],
              label=f'Negative Surprise (n={n_neg})', color='red', linewidth=2)

plt.axvline(x=0, color='grey', linestyle='--', label='Earnings Day (0)')
plt.axhline(y=0, color='grey', linestyle='-')
plt.title('Post-Earnings-Announcement Drift (PEAD)')
plt.xlabel('Trading Days Relative to Earnings Announcement')
plt.ylabel('Cumulative Abnormal Return (CAR)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('car_plot.png')
plt.show()

# =====================================================================
# 5. IS THE DRIFT STATISTICALLY REAL, OR JUST A CLEAN-LOOKING AVERAGE?
# =====================================================================
# One-sample t-test: is mean CAR at the +20 day horizon significantly
# different from 0? With n=360/97 this is now a meaningful test, not
# just directional guesswork like it was with n=3.
from scipy import stats

horizon = merged_df[merged_df['event_day'] == after_days]
groups = {
    'All events': horizon,
    'Positive surprise': horizon[horizon['Surprise_Pct'] > 0],
    'Negative surprise': horizon[horizon['Surprise_Pct'] < 0],
}
print("\n" + "=" * 50 + "\nSignificance check at +20 trading days:")
for label, subset in groups.items():
    vals = subset['CAR'].dropna()
    if len(vals) > 1:
        t_stat, p_val = stats.ttest_1samp(vals, 0)
        print(f"{label}: n={len(vals)}, mean CAR@+20d={vals.mean():.4f}, t={t_stat:.2f}, p={p_val:.4f}")
    else:
        print(f"{label}: not enough events for a t-test (n={len(vals)})")

# =====================================================================
# 6. MANUAL BACKTEST: enter at reaction day, hold 20 trading days, fees included
# =====================================================================
# Direction: LONG after a positive surprise (ride the drift up),
# SHORT after a negative surprise (ride the drift down) - what the CAR
# chart suggests should work if the drift is real and tradeable.
#
# Entry price: Close on event_day == 0. Caveat: if the company reports
# AFTER market close (most do), you can't actually trade at that same
# day's close - the honest entry is event_day == 1's open. Treat this
# version as an optimistic upper bound; swap to event_day==1 open below
# once you want a stricter number.
# Exit price: Close on event_day == after_days.
ROUND_TRIP_COST_PCT = 0.001  # 0.10% total, both legs combined - EDIT to match your real costs

trades = []
for (ticker, window), grp in merged_df.groupby(['Ticker', 'Date_Window']):
    entry_row = grp[grp['event_day'] == 0]
    exit_row = grp[grp['event_day'] == after_days]
    if entry_row.empty or exit_row.empty:
        continue

    entry_price = entry_row['Stock_Close'].iloc[0]
    exit_price = exit_row['Stock_Close'].iloc[0]
    surprise = entry_row['Surprise_Pct'].iloc[0]
    direction = 1 if surprise > 0 else -1  # long on beat, short on miss

    raw_return = direction * (exit_price / entry_price - 1)
    net_return = raw_return - ROUND_TRIP_COST_PCT

    trades.append({
        'Ticker': ticker, 'Date_Window': window, 'Surprise_Pct': surprise,
        'Direction': 'Long' if direction == 1 else 'Short',
        'Raw_Return': raw_return, 'Net_Return': net_return
    })

trades_df = pd.DataFrame(trades)

print("\n" + "=" * 50 + "\nMANUAL BACKTEST RESULTS (hold 20 trading days)")
print(f"Total trades: {len(trades_df)}")
print(f"Win rate (net of fees): {(trades_df['Net_Return'] > 0).mean():.1%}")
print(f"Mean net return per trade: {trades_df['Net_Return'].mean():.4%}")
print(f"Median net return per trade: {trades_df['Net_Return'].median():.4%}")
print(f"Total compounded return (equal-weighted, sequential trades): "
      f"{(np.prod(1 + trades_df['Net_Return']) - 1):.2%}")

# crude signal-to-noise ratio across trades - not an annualized Sharpe,
# just a first-pass check of consistency vs how much returns bounce around
sharpe_like = trades_df['Net_Return'].mean() / trades_df['Net_Return'].std()
print(f"Return / stdev across trades (crude signal-to-noise): {sharpe_like:.3f}")

print("\nBy direction:")
print(trades_df.groupby('Direction')['Net_Return'].agg(['count', 'mean', 'median']))