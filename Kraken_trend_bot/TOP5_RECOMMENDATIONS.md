# 🏆 Top-5 Recommendation Engine — How It Works

> The brain behind every trade: how the bot scans 100+ coins and picks
> the best 5 opportunities — in under 60 seconds.

---

## 🔄 The Pipeline at a Glance

```
  Every 2 minutes the bot runs this cycle:

  ┌─────────────────────────────────────────────────┐
  │  1. UNIVERSE SCAN                               │
  │     Pull all Kraken tickers → 700+ markets      │
  ├─────────────────────────────────────────────────┤
  │  2. VOLUME FILTER                               │
  │     Sort by 24h quote volume → keep top 100     │
  ├─────────────────────────────────────────────────┤
  │  3. PER-COIN DEEP ANALYSIS                      │
  │     For each of the 100 candidates:             │
  │       • Fetch 15m OHLCV candles                 │
  │       • Run trend strategy (EMA + RSI + ATR)    │
  │       • Run ML model (GRU neural network)       │
  │       • Run sentiment analysis (25+ RSS feeds)  │
  ├─────────────────────────────────────────────────┤
  │  4. MULTI-LAYER FILTERING                       │
  │     Must pass ALL gates to get a BUY signal     │
  ├─────────────────────────────────────────────────┤
  │  5. SMART RANKING                               │
  │     Score & rank → pick top 5                   │
  ├─────────────────────────────────────────────────┤
  │  6. RISK-CHECKED EXECUTION                      │
  │     Budget caps, position limits, then buy      │
  └─────────────────────────────────────────────────┘
```

---

## 📡 Step 1 — Universe Scan

The bot calls `fetch_tickers()` on Kraken and receives real-time data
for **every listed market** (~700+ pairs). It immediately filters to:

- **Spot markets only** (no futures, no swaps)
- **USD-quoted pairs** (configurable base currency)
- **Real crypto assets** (excludes stablecoins and fiat-like tokens)

> 💡 This runs every cycle so the bot always sees newly listed coins.

---

## 📊 Step 2 — Volume Filter

From all valid pairs, the bot sorts by **24-hour quote volume** (USD
traded) and keeps the **top 100** most liquid coins.

Why volume matters:
- High volume = tight spreads = less slippage
- High volume = genuine market interest, not thin manipulation
- The bot never touches illiquid micro-caps

```
  700+ pairs → filter → sort by volume → Top 100
```

---

## 🔬 Step 3 — Deep Analysis (Per Coin)

For each of the 100 candidates, **three independent systems** evaluate
the coin simultaneously:

### 🟢 A. Trend Strategy (Technical Analysis)

```
  EMA 21 vs EMA 55 Crossover
  ├── EMA fast > EMA slow → BULL trend
  ├── RSI ≥ 52          → Momentum confirmed
  ├── Price not extended → Not chasing (< 1.8× ATR above EMA)
  └── Price > EMA fast  → Uptrend intact
       ✅ → LONG signal
```

| Indicator | What it does |
|-----------|-------------|
| **EMA 21/55** | Identifies trend direction (fast above slow = bullish) |
| **RSI 14** | Confirms momentum isn't exhausted (must be ≥ 52) |
| **ATR extension guard** | Prevents buying parabolic moves that are about to reverse |

### 🧠 B. ML Model (AI Price Prediction)

A **GRU neural network** trained on 24 features analyzes the last
64 candles and predicts where price could go in the next 24 hours:

```
  Input: 64 bars × 24 features
     ├── Price returns, volume changes
     ├── EMA gaps, RSI, ATR
     ├── Candle patterns (body %, wick ratios)
     ├── Volume profile & momentum
     └── BTC cross-asset beta
         ↓
  ┌──────────────────────────┐
  │  GRU Neural Network      │
  │  Quantile Regression     │
  └──────────────────────────┘
         ↓
  Output: TP predictions at 3 confidence levels
     ├── TP50 (50% chance price reaches this)
     ├── TP80 (80% chance — the key decision metric)
     └── TP90 (90% chance — conservative target)
```

The bot requires **TP80 return ≥ 6%** — meaning the model is 80%
confident the coin can rise at least 6% within 24 hours.

### 📰 C. Sentiment Analysis (News Intelligence)

The bot monitors **25+ cryptocurrency RSS feeds** in real-time and
scores each coin's news sentiment:

```
  RSS Feeds (CoinDesk, CoinTelegraph, Decrypt, ...)
       ↓
  Keyword Matching (coin name, ticker)
       ↓
  Sentiment Scoring (-1.0 to +1.0)
       ↓
  Must be ≥ -0.3 (not bearish)
```

> 📰 If the news says "SEC sues XYZ" → sentiment drops → bot skips it.

---

## 🚦 Step 4 — Multi-Layer Filter Gates

A coin only gets a **BUY** label if it passes **ALL** gates:

```
  ┌──────────────────────────────────────────────┐
  │  Gate 1: Trend = BULL          ✅ or ❌      │
  │  Gate 2: Signal = LONG         ✅ or ❌      │
  │  Gate 3: ML TP80 ≥ 6%         ✅ or ❌      │
  │  Gate 4: Sentiment ≥ -0.3     ✅ or ❌      │
  │                                               │
  │  ALL ✅ → BUY                                 │
  │  ANY ❌ → HOLD (skip)                         │
  └──────────────────────────────────────────────┘
```

If the coin is **already held**, the same gates apply but the action
becomes **COMPOUND** (add to existing position) instead of BUY.

---

## 🏅 Step 5 — Smart Ranking

Coins that pass all gates are scored and ranked:

```
  Score = (ML TP80 return × 100)
        + (Sentiment score × weight)
        + (Volume as tiebreaker)
```

**ML prediction dominates** — a coin predicted to gain +12% ranks higher
than one predicted at +7%, regardless of volume. Sentiment provides a
small boost or penalty. Volume only breaks ties.

The top 5 are selected from this ranked list.

---

## 💰 Step 6 — Risk-Checked Execution

Before buying, each recommendation goes through **strict budget controls**:

```
  ┌─────────────────────────────────────┐
  │  Check 1: Max open positions (12)   │
  │  Check 2: Position size cap (12%)   │
  │  Check 3: Max trade size ($10)      │
  │  Check 4: Explore budget remaining  │
  │  Check 5: Cash above stable floor   │
  │  Check 6: Min trade size ($5)       │
  └─────────────────────────────────────┘
```

| Control | Value | Purpose |
|---------|-------|---------|
| **Max positions** | 12 | Diversification limit |
| **Max per trade** | $10 | Risk control per entry |
| **Stable floor** | 10% of equity | Always keep cash reserve |
| **Explore budget** | 30% of equity | Cap on non-core holdings |
| **Core vs Explore** | Core has no explore cap | BTC/ETH/SOL treated differently |

---

## 🎯 After The Buy — Position Management

Once a position is opened, the bot manages it every cycle:

```
  ┌─────────── STOP LOSS ──────────────┐
  │  -15% from entry → SELL ALL        │
  ├─────────── TAKE PROFIT ────────────┤
  │  +10%  → sell 15% of position      │
  │  +25%  → sell 20%                  │
  │  +50%  → sell 25%                  │
  │  +100% → sell 40%                  │
  ├─────────── TRAILING STOP ──────────┤
  │  After +8% gain:                   │
  │  Track peak price                  │
  │  If drops 10% from peak → SELL ALL │
  └────────────────────────────────────┘
```

---

## ⚡ Real Output Example

```
[SCREENER] Evaluating 100 high-volume symbols...
[SCREENER] Found 5 BUY/COMPOUND opportunities out of 100 evaluated

[ENGINE] Top 5 recommendations (quote=USD):
   1. XMR/USD     price=213.40  24h=+3.2% trend=BULL sig=LONG -> BUY
      ML TP50=220.18 TP80=227.45 TP90=231.02
      SENTIMENT: Positive (+0.42, 3 news)
      📰 Recent News for XMR:
         1. Monero privacy features gain institutional interest
         2. XMR network hashrate reaches new all-time high

   2. SOL/USD     price=148.50  24h=+1.8% trend=BULL sig=LONG -> BUY
      ML TP50=153.20 TP80=159.80 TP90=163.10
      SENTIMENT: Neutral (+0.12, 5 news)

   3. ATOM/USD    price=7.85    24h=+2.1% trend=BULL sig=LONG -> BUY
      ...
```

---

## 🧮 By The Numbers

| Metric | Value |
|--------|-------|
| Universe scanned | 700+ Kraken pairs |
| Volume filter | Top 100 most liquid |
| Final picks | Top 5 ranked |
| Analysis per coin | 3 systems (TA + ML + Sentiment) |
| ML features | 24 per candle × 64 candles |
| News feeds monitored | 25+ RSS sources |
| Cycle frequency | Every ~2 minutes |
| Decision gates | 4 independent filters |
| Total time per scan | ~30-60 seconds |

---

## 🔑 Key Design Principles

1. **Consensus-driven** — No single system decides. All three (TA, ML,
   Sentiment) must agree before buying.

2. **Volume-first** — Only trades liquid coins. No illiquid traps.

3. **ML as gatekeeper** — The neural network must see ≥6% upside at 80%
   confidence. This filters out ~70% of false signals.

4. **News-aware** — Won't buy into bearish headlines. The sentiment
   layer catches regulatory FUD, hacks, and negative events.

5. **Conservative sizing** — Small position sizes ($5-$10) with strict
   caps prevent any single trade from damaging the portfolio.

6. **Layered exits** — Partial take-profits lock in gains progressively.
   Trailing stop captures trend continuation. Stop-loss limits damage.

---

*Built with Python, PyTorch, ccxt, and 25+ live news feeds.
Running on Kraken exchange.*
