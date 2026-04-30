# Coin Prediction Accuracy Analysis Guide

## Overview
This system tracks which coins/cryptos have the most accurate ML predictions, helping you focus on assets with reliable forecasts.

## What are TP50, TP80, TP90?

These are **Target Prices** representing different confidence levels:
- **TP50** (50th percentile): 50% chance the price reaches or exceeds this level
- **TP80** (80th percentile): 80% chance the price reaches or exceeds this level  
- **TP90** (90th percentile): 90% chance the price reaches or exceeds this level

When a prediction "hits" its target, it means the actual price peak met or exceeded the predicted level.

## How Tracking Works

1. **Predictions are logged** when the bot evaluates a coin (in `ml_predictions.json`)
2. **Outcomes are recorded** when the position reaches a peak or closes (24-48 hours later)
3. **Hit rates are calculated** for each coin showing prediction accuracy:
   - Hit rate = (# of times target was hit) / (# of total predictions)
   - Well-calibrated model: TP80 should hit ~80% of the time

## Analysis Tools

### 1. Quick Overview: `track_live_predictions.py`
Shows overall prediction performance plus per-coin breakdown.

```bash
python track_live_predictions.py
```

**Output:**
- Per-coin accuracy table ranked by quality score
- Best/Acceptable/Poor performing coins
- Summary statistics

### 2. Detailed Analysis: `analyze_coin_accuracy.py`  
Comprehensive breakdown of which coins hit which targets.

```bash
# Basic report
python analyze_coin_accuracy.py

# With detailed trade history
python analyze_coin_accuracy.py --details
```

**Output:**
- Complete accuracy table with TP50/TP80/TP90 hit counts
- Quality ratings (EXCELLENT/GOOD/FAIR/POOR)
- Specific recommendations on which coins to trade
- Optional: Individual trade details for top performers

### 3. ML Performance Tracker (in bot)
The `MLPerformanceTracker` class in `bot/ml_performance_tracker.py` provides:

```python
from bot.ml_performance_tracker import MLPerformanceTracker

tracker = MLPerformanceTracker(Path("data/ml_predictions.json"))

# Get per-symbol performance
symbol_perf = tracker.get_symbol_performance(min_predictions=3)

for symbol, stats in symbol_perf.items():
    print(f"{symbol}: TP80 hit rate = {stats['tp80_rate']:.1%}")
```

## Understanding the Metrics

### Hit Rates
- **TP50 Hit Rate**: Should be around 50% (by design)
- **TP80 Hit Rate**: Should be around 80% (by design)  
- **TP90 Hit Rate**: Should be around 90% (by design)

### Quality Score
Weighted combination: `TP80×50% + TP90×30% + TP50×20%`

**Rating Scale:**
- 🌟 **EXCELLENT**: Score ≥ 75, TP80 ≥ 75% → **FOCUS HERE**
- ✅ **GOOD**: Score ≥ 60, TP80 ≥ 65% → Safe to trade
- ⚠️ **FAIR**: Score ≥ 45, TP80 ≥ 50% → Use caution
- ❌ **POOR**: Score < 45 → Avoid or needs more data

## Example Output

```
PREDICTION ACCURACY BY COIN
═══════════════════════════════════════════════════════════════════════════════════
Rank  Symbol       Total   TP50     TP80     TP90     Quality Rating
───────────────────────────────────────────────────────────────────────────────────
1     BTC/USD        15   14 (93%)  13 (87%)  11 (73%)   84.5    🌟 EXCELLENT
2     ETH/USD        12   11 (92%)  10 (83%)   8 (67%)   81.2    🌟 EXCELLENT  
3     SOL/USD        10    9 (90%)   8 (80%)   6 (60%)   76.0    🌟 EXCELLENT
4     ADA/USD         8    7 (88%)   6 (75%)   4 (50%)   72.5    ✅ GOOD
5     DOGE/USD        6    5 (83%)   4 (67%)   2 (33%)   63.5    ✅ GOOD
6     XRP/USD         5    4 (80%)   3 (60%)   1 (20%)   58.0    ⚠️  FAIR
───────────────────────────────────────────────────────────────────────────────────

✅ EXCELLENT COINS (3) - FOCUS HERE!
   These coins consistently hit TP80+ targets. Prioritize trading these:
   • BTC/USD    → TP80: 87%, TP90: 73% (15 predictions)
   • ETH/USD    → TP80: 83%, TP90: 67% (12 predictions)
   • SOL/USD    → TP80: 80%, TP90: 60% (10 predictions)
```

## Trading Strategy

### Focus on Winners
1. **Prioritize EXCELLENT coins** (score ≥ 75)
   - Allocate 60-70% of capital here
   - Use larger position sizes
   - Higher confidence in predictions

2. **Include GOOD coins** (score 60-75)
   - Allocate 20-30% of capital
   - Standard position sizes
   - Reliable but not exceptional

3. **Minimize FAIR coins** (score 45-60)
   - Use small positions or skip
   - Watch for improvement

4. **Avoid POOR coins** (score < 45)
   - Don't trade until more data or model improves
   - May indicate data quality issues

### Position Sizing Example
If you have $10,000 to allocate:

- **EXCELLENT coins** (BTC, ETH, SOL): $6,000-7,000 total
  - BTC: $2,500
  - ETH: $2,500  
  - SOL: $1,500-2,500

- **GOOD coins** (ADA, DOGE): $2,000-3,000 total
  - ADA: $1,000-1,500
  - DOGE: $1,000-1,500

- **FAIR/POOR coins**: $0-1,000 (or skip entirely)

## Data Requirements

For reliable statistics, each coin needs:
- **Minimum**: 3 predictions (appears in reports)
- **Reliable**: 5-10 predictions (confident assessment)
- **Solid**: 15+ predictions (highly confident)

Newly added coins will show "N/A" or be excluded until enough data accumulates.

## Monitoring Schedule

Run these analyses regularly:
- **Daily**: Quick check with `track_live_predictions.py`
- **Weekly**: Detailed review with `analyze_coin_accuracy.py --details`
- **Monthly**: Adjust coin focus based on latest accuracy trends

## Files Location

All tracking data stored in:
```
Kraken_trend_bot/data/
  ├── ml_predictions.json      # Main prediction records
  ├── live_positions.json      # Current positions
  └── trades.csv               # Historical trades
```

## Troubleshooting

**No predictions showing?**
- Ensure ML is enabled in `config.yaml`: `enable_ml_forecaster: true`
- Check model file exists: `data/torch_tp_forecaster.pt`
- Bot needs to be running with positions

**No outcomes recorded?**
- Wait 24-48 hours for positions to move
- Check `manage_existing_positions: true` in config
- Ensure positions are being monitored

**All coins show POOR?**
- May need model retraining
- Check data quality (sufficient historical data)
- Review feature engineering

## Advanced: Using in Code

```python
from pathlib import Path
from bot.ml_performance_tracker import MLPerformanceTracker

# Initialize tracker
tracker = MLPerformanceTracker(Path("data/ml_predictions.json"))

# Get best performing symbols
symbol_perf = tracker.get_symbol_performance(min_predictions=5)
best_symbols = sorted(
    symbol_perf.items(), 
    key=lambda x: x[1]['accuracy_score'], 
    reverse=True
)[:5]

print("Top 5 coins to trade:")
for symbol, stats in best_symbols:
    print(f"{symbol}: {stats['tp80_rate']:.1%} TP80 hit rate")
```

## Summary

The tracking system helps you:
1. ✅ Identify which coins have the most accurate predictions
2. ✅ Focus capital on reliable coins (EXCELLENT/GOOD)
3. ✅ Avoid coins with poor prediction accuracy
4. ✅ Make data-driven decisions about position sizing
5. ✅ Monitor model performance over time

Run `python analyze_coin_accuracy.py` weekly to stay informed!
