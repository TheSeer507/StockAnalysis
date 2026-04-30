# Kraken Trend Bot — Backtest Technicals & X Post Guide

> Your cheat-sheet for understanding the numbers and turning them into
> compelling social-media content.

---

## What This Bot Does (Elevator Pitch)

A fully autonomous crypto trading system that scans **100+ coins** on
Kraken every 2 minutes and picks the best 5 opportunities using:

| Layer | Technology |
|-------|-----------|
| Trend Detection | EMA crossover (21/55) with RSI & ATR filters |
| Price Forecasting | GRU neural network (short-term 15 min + long-term daily) |
| News Intelligence | 25+ RSS feeds with keyword sentiment scoring |
| Risk Management | Stop-loss, tiered take-profit, trailing stop |

**Backtesting** replays the strategy against historical market data so
you can measure how it *would have* performed — without risking real
money.

---

## Key Performance Metrics Explained

### Total Return (%)
- **What:** Overall portfolio gain/loss from start to finish.
- **Formula:** `(Final Equity / Initial Capital − 1) × 100`
- **Good benchmark:** Anything that beats BTC buy-and-hold for the same period is noteworthy.

### Win Rate (%)
- **What:** Percentage of closed trades that were profitable.
- **Formula:** `Winning Trades / Total Trades × 100`
- **Context:** A 40-60 % win rate is common for trend-following strategies — they win big on a few trades and cut losers quickly. A high win rate alone is *not* enough; pair it with Profit Factor.

### Profit Factor
- **What:** Gross profit divided by gross loss.
- **Formula:** `Σ Winning PnL / |Σ Losing PnL|`
- **Interpretation:**
  - **> 1.0** → strategy is net profitable
  - **> 1.5** → solid edge
  - **> 2.0** → very strong
  - **< 1.0** → losing money overall

### Max Drawdown (%)
- **What:** The largest peak-to-trough drop in equity during the backtest.
- **Why it matters:** Shows worst-case pain. Investors and followers want to know the *risk*, not just the return.
- **Rule of thumb:** If max DD is greater than the total return, the risk-reward is questionable.

### Sharpe Ratio
- **What:** Risk-adjusted return — how much return per unit of volatility.
- **Formula:** `Mean(returns) / Std(returns) × √(annualisation factor)`
- **Benchmarks:**
  - **< 0** → negative returns
  - **0–1** → subpar risk-adjusted
  - **1–2** → good
  - **> 2** → excellent

### Sortino Ratio
- **What:** Like Sharpe but only penalises *downside* volatility. More relevant for trading strategies because upside volatility is desirable.
- **Interpretation:** Same scale as Sharpe; higher is better.

### Calmar Ratio
- **What:** Return divided by max drawdown.
- **Formula:** `Total Return % / Max Drawdown %`
- **Interpretation:** A Calmar > 1 means return exceeded worst drawdown.

### Expectancy ($ per trade)
- **What:** Average profit/loss per trade.
- **Formula:** `Net PnL / Total Trades`
- **Why it matters:** If expectancy is positive, every trade is a +EV bet.

### ML Accuracy
- **What:** Percentage of ML TP80 predictions that actually hit within the 24 h forecast horizon.
- **Context:** Even 30-40 % TP80 accuracy adds edge because the wins are sized larger than the losses.

---

## Strategy Components (Technical Detail)

### EMA Crossover (21 / 55)
Two Exponential Moving Averages smooth out price noise. When the **fast
EMA (21)** is above the **slow EMA (55)**, the trend is classified as
**BULL**. Entries are only allowed during bull trends.

### RSI Momentum Filter (14-period, min 52)
Relative Strength Index must be above 52 to confirm upward momentum.
This prevents buying into weak rallies inside a downtrend.

### ATR Extension Guard (14-period, max 1.8×)
Average True Range measures volatility. If price has already moved
**> 1.8 × ATR** above the EMA, the entry is skipped because the asset
is *over-extended* and likely to mean-revert.

### ML Neural Network — GRU Model
A **Gated Recurrent Unit** (a type of recurrent neural network)
processes 64 bars of history with **20+ engineered features** including
multi-timeframe momentum, candle micro-structure, volume surges, and
temporal encoding. It outputs **quantile take-profit predictions**
(TP50, TP80, TP90) for a 24-hour horizon.

### Sentiment Analysis
Headlines from **25+ RSS feeds** (CoinDesk, Cointelegraph, CryptoSlate,
etc.) are scored with a keyword-based system from **-1** (very bearish)
to **+1** (very bullish). Trades with sentiment below **-0.3** are
filtered out.

---

## Understanding the Charts

### Equity Curve (`equity_curve.png`)
- **Blue line:** Your bot's portfolio value over time (normalised to %).
- **Orange dashed line:** BTC buy-and-hold for the same period.
- **Green/red shading:** Periods of profit vs loss.
- **What to look for:** A steadily rising blue line that stays above the
  orange line = your strategy beats passive BTC holding.

### Drawdown Chart (`drawdown.png`)
- **Red area:** How far below the all-time high the portfolio is at each
  point in time.
- **Annotated arrow:** Marks the maximum drawdown.
- **What to look for:** Shallow, short drawdowns = good risk management.

### Trade P&L Distribution (`trade_pnl.png`)
- **Left panel:** Bar chart of each trade's profit or loss.
- **Right panel:** Win/loss count with win-rate percentage.
- **What to look for:** More green bars than red; a few large green bars
  (trend-following profits) offset many small red bars (stopped-out).

### Symbol Breakdown (`symbol_breakdown.png`)
- Horizontal bars showing net P&L per traded asset.
- Labels show trade count and win rate per symbol.
- **What to look for:** Diversification — you want profits spread
  across multiple assets, not concentrated in one lucky pick.

### Summary Card (`summary_card.png`)
- All-in-one social-media-ready image (1200 × 675 px, Twitter/X card).
- Contains every key metric, strategy info, and a disclaimer.
- **Just screenshot or share directly.**

---

## X Post Templates

### Template 1 — Backtest Results Drop

```
My AI crypto trading bot just finished a backtest

Results over [X] days, [N] assets on Kraken:

Total Return: [+X.XX%]
Win Rate: [XX.X%]
Profit Factor: [X.XX]
Max Drawdown: [X.XX%]
Sharpe Ratio: [X.XX]

Strategy: Dual GRU neural nets + EMA crossover + news sentiment

Simulated results only — not financial advice

#AlgoTrading #CryptoAI #QuantFinance
```

### Template 2 — Strategy Breakdown

```
How my AI trading bot decides what to buy

1. Scans 100+ coins on Kraken every 2 min
2. Filters by EMA trend + RSI momentum + ATR volatility
3. Runs a GRU neural network for 24h price targets
4. Checks news sentiment from 25+ RSS feeds
5. Picks Top 5 → executes with stop-loss + trailing stop

ML model predicts take-profit levels at 50th/80th/90th percentiles

Thread on the architecture below

#MachineLearning #CryptoTrading #Python
```

### Template 3 — ML vs No-ML Comparison

```
Does adding machine learning actually help?

I ran my crypto bot twice on the same data:

           With ML    Without ML
Return:    [+X.XX%]   [+X.XX%]
Win Rate:  [XX.X%]    [XX.X%]
Max DD:    [X.XX%]    [X.XX%]
Trades:    [N]        [N]

ML adds +[X.XX%] edge by filtering bad entries

The GRU model was trained on 20+ features including
momentum, volume surges, and candle micro-structure

#MLTrading #DataScience #CryptoBot
```

### Template 4 — Chart Post (attach chart image)

```
Equity curve for my AI crypto bot

Blue = bot performance
Orange = BTC buy-and-hold

Key stats:
- Return: [+X.XX%]
- Sharpe: [X.XX]
- Max DD: [X.XX%]

The goal isn't to beat BTC in every period — it's to
do it with lower drawdowns and systematic risk management

Full backtest: [X] days, [N] assets

#QuantTrading #AlgoTrading #CryptoAI
```

### Template 5 — Thread Opener (multi-tweet)

```
TWEET 1 (Hook):
I built an AI-powered crypto trading bot from scratch in Python

It uses neural networks, news sentiment, and trend analysis
to trade autonomously on Kraken

Here's exactly how it works and how it performed

A thread 🧵

---

TWEET 2 (Architecture):
The bot runs a pipeline every 2 minutes:

1. Pull all 700+ Kraken tickers
2. Filter to top 100 by volume
3. Run EMA/RSI/ATR trend strategy on each
4. Run GRU neural network for 24h price targets
5. Check news sentiment (25+ RSS feeds)
6. Pick top 5, execute with risk management

---

TWEET 3 (ML Detail):
The ML model is a bidirectional GRU neural network

Trained on 20+ features:
- Multi-timeframe momentum (3/7/14 period)
- Volume surge detection
- Candle micro-structure (wick ratios)
- BTC correlation (beta)
- Temporal encoding (hour-of-day)

Outputs quantile TP predictions: 50th, 80th, 90th %ile

---

TWEET 4 (Results + Chart):
Backtest results over [X] days:

Return: [+X.XX%]
Win Rate: [XX.X%]
Max DD: [X.XX%]
Sharpe: [X.XX]

[ATTACH: summary_card.png]

---

TWEET 5 (CTA):
The full bot is ~3,000 lines of Python

Tech stack:
- ccxt for exchange API
- PyTorch for ML models
- feedparser for news
- matplotlib for reporting

All backtested with realistic fees (0.26%) and slippage

Would you want me to open-source this?
```

---

## Hashtags (Mix & Match)

**Core:** `#AlgoTrading` `#CryptoTrading` `#CryptoAI` `#QuantFinance`

**Technical:** `#MachineLearning` `#DeepLearning` `#Python` `#PyTorch`
`#DataScience` `#NeuralNetworks`

**Engagement:** `#BuildInPublic` `#CryptoTwitter` `#TradingBot`
`#FinTech` `#Automation`

**Niche:** `#GRU` `#TimeSeries` `#SentimentAnalysis` `#Kraken`
`#QuantDev`

---

## Important Disclaimers

Always include one of these in your posts to maintain credibility and
avoid regulatory issues:

> **Short:** "Simulated results only — not financial advice."

> **Medium:** "Backtested on historical data with simulated trades.
> Past performance does not guarantee future results. Not financial
> advice."

> **Full:** "These results are from a backtest using historical market
> data and simulated order execution. They include realistic fees
> (0.26%) but do not account for slippage, market impact, or liquidity
> constraints. Past performance is not indicative of future results.
> This is a personal project — not financial advice."

---

## Quick Reference — Running the Report

```bash
# Run a fresh backtest and generate charts
python -m backtest.backtester
python -m backtest.backtest_report

# Or re-run + report in one command
python -m backtest.backtest_report --rerun

# Report for a specific result file
python -m backtest.backtest_report --file backtest_ml_20260323_100258.json

# Generate reports for ALL saved results
python -m backtest.backtest_report --all
```

Charts are saved to `backtest/results/charts/<tag>_<timestamp>/`.

The **summary_card.png** is optimised for Twitter/X (1200 × 675 px) —
attach it directly to your post.
