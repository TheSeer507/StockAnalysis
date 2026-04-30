# 🤖 Kraken Trend Bot — Architecture & System Overview

> **A fully autonomous crypto trading system powered by dual AI models, real-time news intelligence, and multi-layer risk management.**

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        KRAKEN TREND BOT                              │
│                                                                      │
│  ┌────────────┐   ┌────────────────┐   ┌─────────────────────────┐  │
│  │  Kraken     │   │  25+ RSS News  │   │   Configuration Layer   │  │
│  │  Exchange   │   │  Feeds (Global)│   │   (YAML-driven)         │  │
│  │  (via ccxt) │   │  🇺🇸🇬🇧🇩🇪🇯🇵🇰🇷  │   │                         │  │
│  └─────┬──────┘   └───────┬────────┘   └─────────────────────────┘  │
│        │                  │                                          │
│        ▼                  ▼                                          │
│  ┌──────────────────────────────────────────┐                       │
│  │           📊 DATA & INTELLIGENCE LAYER    │                       │
│  │                                          │                       │
│  │  • OHLCV candles (15m + daily)           │                       │
│  │  • Volume rankings & market screening    │                       │
│  │  • News aggregation & deduplication      │                       │
│  │  • Keyword sentiment scoring [-1, +1]    │                       │
│  └──────────────┬───────────────────────────┘                       │
│                 │                                                     │
│                 ▼                                                     │
│  ┌──────────────────────────────────────────┐                       │
│  │           🧠 AI / ML DECISION LAYER       │                       │
│  │                                          │                       │
│  │  ┌──────────────────────────────────┐    │                       │
│  │  │  Short-Term GRU Model (15m)      │    │                       │
│  │  │  • 24-feature vector per bar     │    │                       │
│  │  │  • Quantile regression (q50/80/90)│   │                       │
│  │  │  • Predicts 24h price targets    │    │                       │
│  │  └──────────────────────────────────┘    │                       │
│  │  ┌──────────────────────────────────┐    │                       │
│  │  │  Long-Term GRU Model (Daily)     │    │                       │
│  │  │  • 28-feature vector per day     │    │                       │
│  │  │  • Multi-horizon: 7d / 14d / 30d │    │                       │
│  │  │  • Conviction scoring system     │    │                       │
│  │  └──────────────────────────────────┘    │                       │
│  │  ┌──────────────────────────────────┐    │                       │
│  │  │  EMA Crossover Strategy          │    │                       │
│  │  │  • Fast/Slow EMA trend detection │    │                       │
│  │  │  • RSI momentum guard            │    │                       │
│  │  │  • ATR extension filter          │    │                       │
│  │  └──────────────────────────────────┘    │                       │
│  └──────────────┬───────────────────────────┘                       │
│                 │                                                     │
│                 ▼                                                     │
│  ┌──────────────────────────────────────────┐                       │
│  │        ⚙️ EXECUTION & RISK LAYER          │                       │
│  │                                          │                       │
│  │  • Position sizing (% of equity)         │                       │
│  │  • Max open positions cap                │                       │
│  │  • Multi-level take-profit (partial)     │                       │
│  │  • Trailing stop (arm after +X%)         │                       │
│  │  • Stop-loss (% + absolute levels)       │                       │
│  │  • Core portfolio rebalancing            │                       │
│  │  • Stable cash floor enforcement         │                       │
│  │  • Explore budget cap for altcoins       │                       │
│  │  • Paper mode / Live mode toggle         │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                      │
│  ┌──────────────────────────────────────────┐                       │
│  │        📈 MONITORING & TRACKING           │                       │
│  │                                          │                       │
│  │  • ML prediction accuracy tracking       │                       │
│  │  • Directional accuracy, MAE, RMSE       │                       │
│  │  • Per-symbol performance breakdown      │                       │
│  │  • Automated model degradation detection │                       │
│  │  • Portfolio PnL & trade history         │                       │
│  └──────────────────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 How the AI Models Work

### Short-Term Forecaster (15-minute candles)

The short-term model is a **GRU (Gated Recurrent Unit) neural network** trained with **quantile regression** to predict the maximum price upside over the next ~24 hours.

**24 Input Features per candle:**

| # | Feature | Description |
|---|---------|-------------|
| 0 | Log Return | Log price change between bars |
| 1 | High-Low Range | Normalized volatility measure |
| 2 | Open-Close Return | Bar body direction & size |
| 3 | Volume Delta | Volume change rate |
| 4 | EMA Spread | Fast vs slow EMA distance |
| 5 | RSI (14) | Relative Strength Index |
| 6 | ATR Ratio | Average True Range normalized |
| 7-9 | Momentum (3/7/14) | Multi-window momentum |
| 10-11 | Upper/Lower Wicks | Rejection signal strength |
| 12 | Volume Ratio | Current vs moving avg volume |
| 13 | Bollinger %B | Position within Bollinger Bands |
| 14 | VWAP Distance | Distance from volume-weighted avg |
| 15-16 | Order Flow | Buy vs sell pressure proxy |
| 17-22 | Time Encoding | Cyclical hour/day-of-week |
| 23 | BTC Beta | Cross-asset correlation slot |

**Architecture:** `Input → GRU (multi-layer) → Attention → LayerNorm → MLP → 3 quantile outputs`

**Output:** Three confidence levels for price targets:
- **q50 (Median):** Conservative target — 50% chance of being reached
- **q80 (Optimistic):** Stretch target — 20% chance of being exceeded
- **q90 (Aggressive):** Moon target — 10% chance of being exceeded

### Long-Term Forecaster (Daily candles)

The long-term model uses the same GRU architecture but trained on **daily** timeframes with **28 features** and **multi-horizon** predictions:

- **7-day outlook** — Short swing trade horizon
- **14-day outlook** — Medium hold period
- **30-day outlook** — Portfolio position horizon

**Conviction System:** Based on multi-horizon agreement:
- 🟢 **STRONG_HOLD** — All horizons bullish, trend accelerating
- 🟡 **HOLD** — Mixed signals, moderate upside
- 🟠 **WEAK** — Low confidence, limited upside expected
- 🔴 **EXIT** — Negative outlook across horizons

---

## 📊 Trading Decision Flow

### Entry (New Position)

```
1. SCREEN     → Top 100 symbols by 24h volume on Kraken
2. TREND      → EMA crossover must show BULL trend
3. MOMENTUM   → RSI must be above minimum threshold
4. EXTENSION  → Price can't be over-extended from EMAs (ATR guard)
5. ML PREDICT → Short-term model must show >X% upside (q80)
6. SENTIMENT  → News sentiment must be ≥ -0.3 (not deeply negative)
7. RANK       → Score = ML_predicted_return + sentiment_weight
8. RISK CHECK → Max positions? Within budget? Enough equity?
9. SIZE       → Position size = min(risk%, max_position%, max_notional)
10. EXECUTE   → Market order via paper portfolio or live Kraken API
```

### Exit (Position Management)

Every 60 seconds, for each open position:

```
STOP LOSS       → Price dropped X% from entry? → SELL ALL
TAKE PROFIT     → Hit TP level 1/2/3/4? → SELL partial %
TRAILING STOP   → Profit > arm_threshold? → Track peak → Sell if drops X% from peak
SIGNAL EXIT     → EMA crossed bearish? → Close position
```

**Multi-Level Take Profit Example (default config):**
| Level | Trigger | Action |
|-------|---------|--------|
| TP1 | +10% | Sell 15% of position |
| TP2 | +25% | Sell 20% of position |
| TP3 | +50% | Sell 25% of position |
| TP4 | +100% | Sell 40% of position |

---

## 🌐 News Intelligence System

The bot monitors **25+ RSS feeds** from global crypto news sources:

| Region | Sources |
|--------|---------|
| 🇺🇸 Global/English | CoinDesk, Cointelegraph, Decrypt, Bitcoin Magazine, The Block, CryptoSlate, CryptoPotato, The Defiant, and more |
| 🇯🇵 Japan | CoinPost, Cointelegraph Japan |
| 🇰🇷 South Korea | TokenPost Korea, Cointelegraph Korea, CoinReaders |
| 🇬🇧🇩🇪 Europe | CityAM Crypto, BTC-ECHO, Bitcoin-Bude, CryptoNews UK |

**How it works:**
1. Fetches all feeds every ~10 minutes
2. Deduplicates via SHA1 hashing of URLs
3. Matches headlines against held portfolio assets
4. Scores by asset mentions (×2.5) + keyword matches (×1.0)
5. High-impact keywords: `ETF`, `SEC`, `hack`, `exploit`, `listing`, `fork`, `airdrop`
6. Feeds into **sentiment scoring** (keyword-based, scale: -1 to +1)

---

## 🛡️ Risk Management Layers

| Layer | Protection |
|-------|-----------|
| **Position Sizing** | Each trade capped at X% of equity |
| **Max Positions** | Hard cap on total concurrent positions (default: 12) |
| **Max Trade Notional** | Dollar cap per single trade |
| **Stop Loss** | Automatic sell at X% loss from entry |
| **Hard Stops** | Absolute price floors per core asset (e.g. BTC < $80K) |
| **Trailing Stop** | Locks in profits by tracking highest price |
| **Stable Floor** | Always maintains X% in stablecoins (USD/USDT/USDC) |
| **Explore Budget** | Caps altcoin allocation to X% of total portfolio |
| **Min Equity Guard** | Won't trade if account below minimum threshold |
| **Staked Asset Protection** | Never sells staked/locked balances |
| **Safety Toggles** | `allow_live_trading` + `execute_trades` dual safety switch |

---

## 📐 Portfolio Structure

```
Portfolio Allocation (configurable):
┌────────────────────────────────────────────┐
│  CORE TARGETS (weighted allocation)        │
│  ├── BTC ────────── 45%                    │
│  ├── ETH ────────── 20%                    │
│  ├── SOL ────────── 10%                    │
│  └── XMR ────────── 10%                    │
│                                            │
│  STABLE FLOOR ────── 10% (USD/USDT/USDC)  │
│                                            │
│  EXPLORE BUDGET ──── 30% (altcoin trades)  │
│  └── Dynamic: XRP, ATOM, screened assets   │
└────────────────────────────────────────────┘
```

The **rebalancer** runs periodically to bring core holdings back toward target weights when they drift beyond the configured band (default: ±2%).

---

## 🔄 Bot Loop (Main Cycle)

```
Every 60 seconds:
  ├── Fetch main pair price & strategy signal
  ├── Check for new crypto news
  ├── Manage all open positions (stops/TP/trailing)
  │
  Every 2 loops:
  ├── Run market screener → generate buy recommendations
  ├── Execute buys within risk parameters
  │
  Every 2 loops:
  ├── Deep portfolio scan (short-term + long-term ML)
  │
  Every 10 loops:
  ├── Rebalance core portfolio targets
  │
  Every 120 loops (~2 hours):
  └── Print ML performance report
```

---

## 🔬 ML Model Training Pipeline

Both models auto-retrain when stale (7 days short-term, 14 days long-term):

```
1. Connect to Kraken → fetch top N symbols by volume
2. Download historical OHLCV (120 days / 365 days)
3. Engineer feature vectors (24 / 28 features)
4. Compute MFE targets (max favorable price excursion)
5. Time-based train/validation split (80/20)
6. Feature normalization (z-score standardization)
7. Train GRU with:
   ├── Quantile pinball loss + crossing penalty
   ├── Gradient clipping (0.5)
   ├── Mixed precision training (AMP)
   ├── OneCycle learning rate schedule
   └── Early stopping (patience: 10-12 epochs)
8. Save model weights + metadata + normalization stats
```

**Hyperparameter highlights:**
- Hidden size: 384 (short-term) / 192 (long-term)
- Layers: 4 / 3
- Lookback: 64 bars / 90 days
- Learning rate: ~0.0006 / 0.0005
- Batch size: 512 / 256
- Quantiles: q50, q80, q90

---

## 📊 ML Performance Monitoring

The bot continuously validates its own predictions:

- **Logs every prediction** (symbol, price, predicted targets, timestamp)
- **Matches predictions to actual outcomes** when positions close
- **Tracks metrics:** TP hit rates (q80/q90), MAE, RMSE, directional accuracy, calibration
- **Per-symbol breakdown** — identifies which assets the model predicts best
- **Degradation detection** — compares recent vs. historical accuracy to flag model staleness

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Exchange API | ccxt (Kraken) |
| ML Framework | PyTorch (GRU, quantile regression) |
| Configuration | YAML |
| Data Format | JSON (state), CSV (exports) |
| News | RSS/Atom feed parsing |
| Scheduling | Built-in polling loop |
| Mode Support | Paper trading + Live trading |

---

## 📝 Key Configuration (config.yaml)

The entire system is driven by a rich YAML configuration covering:

- Exchange settings (pair, timeframe, base currency)
- Risk parameters (per-trade %, max positions, max notional, stops)
- Take-profit levels (multi-tier partial sells)
- Trailing stop settings (arm threshold, trail percentage)
- Portfolio structure (core targets, stable floor, explore budget)
- ML model hyperparameters (architecture, training, horizons)
- News sources (25+ RSS feeds, high-impact keywords)
- Sentiment thresholds
- Safety switches

---

*Built with ❤️ and Python — Autonomous crypto trading powered by AI.*
