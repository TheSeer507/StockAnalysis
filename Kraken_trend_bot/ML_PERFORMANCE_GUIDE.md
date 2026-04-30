# ML Performance Tracking & Improvement Guide

## Overview

Your trading bot now has comprehensive **ML performance tracking and monitoring** capabilities! The system tracks every ML prediction against actual market outcomes, calculates accuracy metrics, and provides detailed performance analysis.

---

## 🎯 What Was Added

### 1. **Performance Tracking System** ([bot/ml_performance_tracker.py](bot/ml_performance_tracker.py))
   - Logs every ML prediction with timestamp and predicted values
   - Tracks actual outcomes (peak prices, returns, time to peak)
   - Calculates comprehensive accuracy metrics
   - Stores historical prediction data for analysis
   - Recommends when to retrain the model

### 2. **Performance Analyzer** ([analyze_ml_performance.py](analyze_ml_performance.py))
   - Detailed performance reports by symbol
   - Performance trends over time
   - Hour-of-day analysis
   - Export reports to text files

### 3. **Integrated Tracking in Bot**
   - Automatic prediction logging during recommendations
   - Automatic outcome tracking during position management
   - Periodic performance reports (every 2 hours by default)
   - Retraining alerts when performance degrades

---

## 📊 Key Metrics Tracked

### Accuracy Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **TP80 Hit Rate** | % of times actual peak ≥ predicted TP80 | ~80% |
| **TP90 Hit Rate** | % of times actual peak ≥ predicted TP90 | ~90% |
| **Directional Accuracy** | % correct price direction predictions | >60% |
| **MAE** | Mean Absolute Error in price predictions | Lower is better |
| **RMSE** | Root Mean Squared Error | Lower is better |
| **MAPE** | Mean Absolute Percentage Error | <15% is good |

### Calibration Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **TP80 Coverage** | Actual coverage of 80th quantile | ~80% |
| **TP90 Coverage** | Actual coverage of 90th quantile | ~90% |

**Well-calibrated models**: Quantile predictions match reality (80% quantile should be reached ~80% of the time)

### Performance Statistics

- **Avg Predicted Return**: Average ML predicted return
- **Avg Actual Return**: Average actual return achieved
- **Avg Time to Peak**: How long it takes to reach peak price
- **Recent vs. Historical**: 7-day vs 30-day performance comparison

---

## 🚀 How It Works

### 1. **Prediction Logging**
When the bot evaluates a trading opportunity:

```python
# ML makes prediction
ml_tp50 = 95500.00   # 50th quantile price
ml_tp80 = 96800.00   # 80th quantile price  
ml_tp90 = 97500.00   # 90th quantile price
ml_ret80 = 0.0215    # Expected 2.15% return

# System logs this prediction
tracker.log_prediction(
    symbol="BTC/USD",
    entry_price=95234.50,
    pred_tp50=95500.00,
    pred_tp80=96800.00,
    pred_tp90=97500.00,
    pred_ret80=0.0215
)
```

### 2. **Outcome Tracking**
As positions are managed, the system tracks actual results:

```python
# Position reaches new peak
current_peak = 97234.50  # Higher than entry

# System updates outcome
tracker.update_outcome(
    symbol="BTC/USD",
    entry_price=95234.50,
    actual_peak_price=97234.50
)

# Calculates: Did we hit TP80? TP90? What was actual return?
```

### 3. **Metrics Calculation**
Periodically (every 2 hours by default), the system:
- Aggregates all predictions with outcomes
- Calculates accuracy metrics
- Identifies performance trends
- Checks if retraining is needed

---

## 📈 Performance Reports

### Automatic Reports (Every 2 Hours)

```
======================================================================
ML Performance Report - Last 30 Days
======================================================================

Dataset Size:
  Total Predictions:      243
  With Outcomes:          187
  Coverage:               76.9%

Prediction Accuracy:
  TP80 Hit Rate:          73.8% (target: 80%)
  TP90 Hit Rate:          85.2% (target: 90%)
  Directional Accuracy:   67.4%

Error Metrics (TP80):
  MAE (Mean Absolute):    $145.23
  RMSE (Root MSE):        $213.45
  MAPE (% Error):         12.3%

Model Calibration:
  TP80 Coverage:          73.8% (should be ~80%)
  TP90 Coverage:          85.2% (should be ~90%)

Return Statistics:
  Avg Predicted Return:   +2.34%
  Avg Actual Return:      +1.89%
  Avg Time to Peak:       18.3 hours

Recent Performance (7 days):
  TP80 Hit Rate:          68.5%
  Directional Accuracy:   64.2%

Performance Assessment:
  ⚠️  TP80 accuracy is ACCEPTABLE (73.8%)
  ✅ Directional accuracy is GOOD (67.4% ≥ 60%)
  ✅ Model is well-calibrated (coverage 73.8% ≈ 80%)

======================================================================
```

### Detailed Analysis Script

Run the analyzer for comprehensive insights:

```powershell
python analyze_ml_performance.py
```

This generates:
- **Performance by Symbol**: Which assets the model predicts best
- **Performance by Hour**: Best/worst times for predictions
- **Trend Analysis**: Is performance improving or degrading?
- **Recommendations**: When to retrain

---

## 🔧 Configuration

Add this to your `config/config.yaml` (already added):

```yaml
ml_tracking:
  enabled: true                          # Enable tracking
  data_path: data/ml_predictions.json    # Where to store predictions
  report_every_loops: 120                # Report frequency (every 2hrs if poll=60s)
  max_records: 10000                     # Max records to keep
  auto_retrain_on_poor_performance: false  # Auto-retrain on poor performance
```

### Configuration Options

**enabled**: Turn tracking on/off
- `true`: Track all predictions (recommended)
- `false`: Disable tracking

**report_every_loops**: How often to print reports
- `60`: Every hour (if poll_interval_sec=60)
- `120`: Every 2 hours (default)
- `240`: Every 4 hours

**max_records**: Maximum predictions to store
- `5000`: ~2-3 months of data
- `10000`: ~6 months (default)
- `20000`: ~1 year

**auto_retrain_on_poor_performance**: Automatically retrain when performance drops
- `false`: Manual retraining only (default)
- `true`: Auto-retrain when metrics drop below thresholds

---

## 📋 ML Performance Best Practices

### 1. **Monitor Regularly**
- Check performance reports every few days
- Look for degrading trends
- Pay attention to recent (7-day) metrics

### 2. **Retrain When Needed**

Retrain if:
- ✅ TP80 hit rate < 65% (below acceptable)
- ✅ Directional accuracy < 55% (barely better than random)
- ✅ Recent performance significantly worse than historical
- ✅ Major market regime change (bull → bear or vice versa)

**How to retrain**:
```powershell
cd D:\PythonProjects\StockAnalysis\Kraken_trend_bot
D:\PythonProjects\StockAnalysis\.venv\Scripts\python.exe -m ml.train_torch_forecaster
```

### 3. **Understand Metrics**

**Good Performance**:
- TP80 Hit Rate: 75-85%
- TP90 Hit Rate: 85-95%
- Directional Accuracy: >60%
- MAPE: <15%

**Acceptable Performance**:
- TP80 Hit Rate: 65-75%
- TP90 Hit Rate: 80-85%
- Directional Accuracy: 55-60%
- MAPE: 15-20%

**Poor Performance** (retrain needed):
- TP80 Hit Rate: <65%
- TP90 Hit Rate: <80%
- Directional Accuracy: <55%
- MAPE: >20%

### 4. **Interpret Calibration**

**Well-Calibrated Model**:
- TP80 coverage ≈ 80% (within 5-10%)
- TP90 coverage ≈ 90% (within 5-10%)

**Under-calibrated** (too optimistic):
- TP80 coverage < 70%
- Predictions too aggressive

**Over-calibrated** (too conservative):
- TP80 coverage > 90%
- Predictions too cautious

---

## 🎓 Advanced Features

### 1. **Performance by Symbol**

Identifies which assets the model predicts best/worst:

```
Symbol       Count    TP80 Rate    Dir Acc    Avg Pred    Avg Actual
BTC/USD      45       78.5%        72.3%      +2.45%      +2.12%
ETH/USD      38       71.2%        65.8%      +3.12%      +2.67%
SOL/USD      29       82.4%        75.9%      +4.23%      +3.98%
```

**Use this to**:
- Focus on assets where ML works best
- Increase confidence thresholds for poorly-predicted assets
- Understand model strengths/weaknesses

### 2. **Time-of-Day Analysis**

Shows when predictions are most accurate:

```
Hour Block      Predictions    Hit Rate
00:00 - 04:00   23            68.5%
04:00 - 08:00   31            72.3%
08:00 - 12:00   45            79.1%  ← Best time
12:00 - 16:00   38            74.2%
16:00 - 20:00   29            71.5%
20:00 - 24:00   21            69.8%
```

**Use this to**:
- Adjust confidence during low-accuracy hours
- Understand market microstructure effects
- Time entries/exits better

### 3. **Trend Detection**

Monitors if performance is improving/degrading:

```
⚠️  Performance degrading: Recent accuracy significantly worse
    7-day: 68.5%  vs  30-day: 76.3%
    Action: Retrain model soon
```

**Use this to**:
- Catch model decay early
- Trigger retraining before major losses
- Adapt to market regime changes

---

## 🛠️ Troubleshooting

### No Predictions Being Logged

**Problem**: `predictions_with_outcomes = 0`

**Solutions**:
1. Check `ml_tracking.enabled: true` in config
2. Verify ML model is loaded (check [ML] messages on startup)
3. Ensure bot is making trades or generating recommendations
4. Wait 24-48 hours for outcomes to be recorded

### Low TP80 Hit Rate

**Problem**: TP80 hit rate < 65%

**Root Causes**:
- Model trained on different market conditions
- Market regime changed (bull → bear)
- Training data too old
- Model overfitting

**Solutions**:
1. Retrain with more recent data
2. Increase `history_days` in ml config (120 → 180)
3. Add more training symbols (`train_symbols: 80 → 120`)
4. Reduce model complexity (decrease `hidden_size` or `num_layers`)

### Poor Directional Accuracy

**Problem**: Directional accuracy < 55%

**Root Causes**:
- Weak feature engineering
- Market is range-bound (not trending)
- Model not capturing momentum

**Solutions**:
1. Add more features to model (volume, orderbook, etc.)
2. Adjust strategy to work better in range-bound markets
3. Increase `lookback` period (64 → 96)
4. Use ensemble of models

### Model Under-Calibrated

**Problem**: TP80 coverage < 70%

**Meaning**: Model is too optimistic (predictions too aggressive)

**Solutions**:
1. Increase quantile values in training
2. Add regularization (`dropout: 0.15 → 0.20`)
3. Use more conservative thresholds (`min_tp80_pct: 0.06 → 0.08`)

### Model Over-Calibrated

**Problem**: TP80 coverage > 90%

**Meaning**: Model is too conservative (predictions too cautious)

**Solutions**:
1. Decrease quantile uncertainty
2. Reduce regularization (`dropout: 0.15 → 0.10`)
3. Train longer (`epochs: 60 → 80`)

---

## 📊 Example Workflow

### Day 1-7: Initial Period
```
- Bot makes predictions
- No outcomes yet (need time for trades to play out)
- Build prediction history
```

### Day 7-14: First Results
```
- Some predictions now have outcomes
- First performance metrics available
- Review initial TP80/TP90 hit rates
```

### Week 3: First Report
```
[ML-TRACKER] ML Performance Report:
  Predictions: 45
  With Outcomes: 28
  TP80 Hit Rate: 71.4%
  
Status: Acceptable, continue monitoring
```

### Week 6: Performance Check
```
[ML-TRACKER] ML Performance Report:
  Predictions: 156
  With Outcomes: 118
  TP80 Hit Rate: 68.2%
  Recent (7d): 62.5%
  
⚠️  Performance degrading - Consider retraining
```

### Week 7: Retrain
```
Run: python -m ml.train_torch_forecaster
```

### Week 8: Improved
```
[ML-TRACKER] ML Performance Report:
  TP80 Hit Rate: 76.8%
  Recent (7d): 78.2%
  
✅ Performance improved after retraining
```

---

## 🎁 Benefits

### 1. **Data-Driven Decisions**
- Know when your ML model is working
- Quantify prediction accuracy
- Make informed retraining decisions

### 2. **Continuous Improvement**
- Track performance over time
- Identify weaknesses
- Optimize strategy based on data

### 3. **Risk Management**
- Reduce reliance on poor predictions
- Adjust position sizes based on ML confidence
- Avoid trading during low-accuracy periods

### 4. **Transparency**
- Understand ML model behavior
- Build trust in automated decisions
- Debugging and troubleshooting

---

## 🚀 Next Steps

1. **Run the bot for a week** to build initial prediction history
2. **Check first performance report** after 7-10 days
3. **Run detailed analyzer** monthly: `python analyze_ml_performance.py`
4. **Retrain when needed** based on metrics
5. **Iterate and improve** based on insights

---

## 📚 Files Reference

| File | Purpose |
|------|---------|
| [bot/ml_performance_tracker.py](bot/ml_performance_tracker.py) | Core tracking system |
| [analyze_ml_performance.py](analyze_ml_performance.py) | Detailed analysis tool |
| data/ml_predictions.json | Prediction history storage |
| data/ml_performance_report_*.txt | Saved reports |

---

## 💡 Pro Tips

1. **Save Reports**: `analyze_ml_performance.py` saves dated reports for comparison
2. **Compare Models**: Keep old model files to compare performance
3. **A/B Testing**: Run two models in paper mode and compare metrics
4. **Feature Importance**: Track which features (EMA, RSI, etc.) correlate with accuracy
5. **Market Conditions**: Note market conditions (bull/bear/sideways) in reports

---

**Your ML model now has a comprehensive feedback loop for continuous improvement!** 📈

