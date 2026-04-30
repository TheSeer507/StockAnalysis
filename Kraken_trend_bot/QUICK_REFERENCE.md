# Hyperparameter Optimization - Quick Reference

## ✅ COMPLETED

- **Config Updated**: Optimized hyperparameters applied to `config/config.yaml`
- **Dependencies Installed**: torch, optuna, scikit-learn, pandas, numpy
- **Scripts Created**: 
  - `hyperparameter_tuning.py` - Automated tuning system
  - `compare_hyperparameters.py` - Before/after comparison
  - `setup_optimization.py` - Setup verification

## 🚀 QUICK START

### Option 1: Train with Optimized Settings (Fastest)
```bash
python ml/train_torch_forecaster.py
```

### Option 2: Run Automated Tuning (Best)
```bash
# Quick test (20 minutes)
python hyperparameter_tuning.py --trials 20 --timeout 1200 --symbols 20

# Standard (1 hour)
python hyperparameter_tuning.py --trials 50 --timeout 3600

# Apply results
python hyperparameter_tuning.py --apply
```

### Option 3: Compare Changes
```bash
python compare_hyperparameters.py
```

## 📊 KEY IMPROVEMENTS

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Data** |
| Symbols | 100 | 150 | +50% |
| History | 180d | 365d | +103% |
| Lookback | 96 | 128 | +33% |
| Samples | ~10K | ~21K | +103% |
| **Model** |
| Hidden Size | 192 | 256 | +33% |
| Layers | 3 | 4 | +33% |
| Parameters | 733K | 1.7M | +131% |
| **Training** |
| Epochs | 80 | 120 | +50% |
| Batch Size | 256 | 128 | 2× updates |
| LR | 0.0003 | 0.0001 | Conservative |
| Scheduler | ReduceLR | OneCycle | Better |

## 🎯 EXPECTED RESULTS

- **Validation Loss**: 5-15% better
- **TP80 Accuracy**: +3-8 points
- **TP90 Accuracy**: +2-5 points
- **Directional**: +5-10 points
- **Calibration**: Improved coverage

## 📖 DOCUMENTATION

- `OPTIMIZATION_SUMMARY.md` - Complete guide
- `HYPERPARAMETER_TUNING.md` - Tuning details
- `ML_PERFORMANCE_GUIDE.md` - Performance tracking

## 🔧 TROUBLESHOOTING

### Out of Memory
```yaml
# Reduce in config.yaml:
batch_size: 64
hidden_size: 192
```

### Training Too Slow
```yaml
# Reduce in config.yaml:
train_symbols: 100
epochs: 80
```

### Poor Calibration
```bash
# Run longer training:
python ml/train_torch_forecaster.py
# Increase in config: epochs: 150
```

## ✨ MAIN OPTIMIZATIONS

1. **More Data**: 365 days, 150 symbols → Better generalization
2. **Bigger Model**: 4 layers, 256 hidden → More capacity
3. **OneCycleLR**: Better convergence than step decay
4. **Better Regularization**: Dropout 0.25, weight decay 0.001
5. **Automated Tuning**: Optuna-based hyperparameter search

## 📞 NEXT STEPS

1. ✅ Train model with new settings
2. ✅ Compare performance metrics
3. ✅ Monitor live trading results
4. ⏰ Retrain every 7-30 days

---

**Ready to train!** 🚀

```bash
python ml/train_torch_forecaster.py
```
