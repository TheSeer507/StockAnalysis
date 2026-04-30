# ML Improvements - Quick Start

## What's New? 🎉

Your trading bot now tracks ML prediction accuracy and provides detailed performance metrics!

---

## Key Features Added

### 1. **Automatic Performance Tracking**
✅ Every ML prediction is logged  
✅ Actual outcomes are tracked automatically  
✅ Comprehensive metrics calculated  

### 2. **Performance Reports**
✅ Auto-generated every 2 hours  
✅ Shows TP80/TP90 hit rates  
✅ Directional accuracy  
✅ Alerts when retraining needed  

### 3. **Detailed Analysis**
✅ Performance by symbol  
✅ Time-of-day analysis  
✅ Trend detection  
✅ Export reports  

---

## Quick Configuration

Already added to `config.yaml`:

```yaml
ml_tracking:
  enabled: true
  data_path: data/ml_predictions.json
  report_every_loops: 120  # Every 2 hours
  max_records: 10000
```

---

## What You'll See

### In Bot Output (Every 2 Hours):

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

Performance Assessment:
  ✅ TP80 accuracy is ACCEPTABLE (73.8%)
  ✅ Directional accuracy is GOOD (67.4% ≥ 60%)

======================================================================
```

### Retraining Alerts:

```
[ML-TRACKER][ALERT] Model retraining recommended: Recent accuracy degraded (62.5% < 60%)
```

---

## Key Metrics Explained

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **TP80 Hit Rate** | How often price reaches predicted TP80 | 75-85% |
| **TP90 Hit Rate** | How often price reaches predicted TP90 | 85-95% |
| **Directional Accuracy** | % of correct up/down predictions | >60% |
| **Calibration** | Do quantiles match reality? | ±10% of target |

---

## How To Use

### 1. **Monitor Automatically**
Just run your bot normally. Performance reports appear every 2 hours.

### 2. **Run Detailed Analysis**
```powershell
cd D:\PythonProjects\StockAnalysis\Kraken_trend_bot
python analyze_ml_performance.py
```

Generates comprehensive report with:
- Performance by symbol
- Hour-of-day analysis  
- Trend detection
- Detailed recommendations

### 3. **Check Data File**
```powershell
cat data/ml_predictions.json
```

See raw prediction history and outcomes.

---

## When to Retrain

Retrain your ML model if:

✅ **TP80 Hit Rate < 65%** (poor accuracy)  
✅ **Directional Accuracy < 55%** (barely better than random)  
✅ **Recent performance worse** (7-day << 30-day)  
✅ **Major market change** (bull → bear transition)  

### How to Retrain:
```powershell
cd D:\PythonProjects\StockAnalysis\Kraken_trend_bot
D:\PythonProjects\StockAnalysis\.venv\Scripts\python.exe -m ml.train_torch_forecaster
```

---

## Performance Targets

### 🎯 Good Performance
- TP80 Hit Rate: **75-85%**
- TP90 Hit Rate: **85-95%**
- Directional Accuracy: **>65%**

### ⚠️ Acceptable Performance  
- TP80 Hit Rate: **65-75%**
- TP90 Hit Rate: **80-85%**
- Directional Accuracy: **55-65%**

### ❌ Poor Performance (Retrain!)
- TP80 Hit Rate: **<65%**
- TP90 Hit Rate: **<80%**
- Directional Accuracy: **<55%**

---

## Files Created

| File | Purpose |
|------|---------|
| **bot/ml_performance_tracker.py** | Core tracking system |
| **analyze_ml_performance.py** | Detailed analyzer tool |
| **data/ml_predictions.json** | Prediction history (auto-created) |
| **ML_PERFORMANCE_GUIDE.md** | Complete documentation |

---

## Timeline

**Week 1**: Build prediction history  
**Week 2**: First metrics available  
**Week 3**: First full performance report  
**Week 4+**: Regular monitoring and optimization  

---

## Quick Checks

### Is Tracking Working?
```powershell
# Check if predictions are being logged
cat data/ml_predictions.json
# Should see "records" array with predictions
```

### Are Outcomes Being Tracked?
```powershell
# Look for "outcome_recorded": true in the JSON
# Should appear after positions hit peaks/close
```

### Manual Performance Check
```powershell
python analyze_ml_performance.py
# Generates detailed report immediately
```

---

## Troubleshooting

**No predictions logged?**
- Check `ml_tracking.enabled: true` in config
- Verify ML model is loaded (check startup logs)
- Ensure bot is running and making recommendations

**All outcomes are false?**
- Normal in first 24-48 hours (positions need time)
- Outcomes update when peaks are reached
- Be patient, it takes a few days

**TP80 rate is 0%?**
- Not enough time has passed
- Need at least 10-20 completed predictions
- Run bot for at least a week

---

## Pro Tips

1. **Save Historical Reports**: `analyze_ml_performance.py` creates dated reports
2. **Compare Before/After**: Keep reports before and after retraining
3. **Track Manually**: Note market conditions when performance changes
4. **Iterate**: Use insights to improve training parameters
5. **Be Patient**: Need 2-4 weeks of data for meaningful metrics

---

## Next Steps

1. ✅ **Run bot normally** - Tracking is automatic
2. ⏰ **Wait 7-10 days** - Let prediction history build
3. 📊 **Check first report** - Review initial metrics
4. 🔄 **Retrain if needed** - Based on performance data
5. 📈 **Continuous improvement** - Monitor and optimize

---

**Your ML model now has continuous feedback for improvement!** 🚀

For complete details, see **[ML_PERFORMANCE_GUIDE.md](ML_PERFORMANCE_GUIDE.md)**
