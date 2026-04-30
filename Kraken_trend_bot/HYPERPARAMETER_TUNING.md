# Hyperparameter Tuning Guide

## Overview

The hyperparameter tuning system uses **Optuna** to systematically search for optimal model parameters. This replaces manual trial-and-error with automated optimization.

## What Gets Optimized

The tuning process searches for optimal values of:

### Model Architecture
- `hidden_size`: [128, 192, 256, 320, 384] - Model capacity
- `num_layers`: [2, 3, 4, 5] - Network depth
- `dropout`: [0.1 - 0.4] - Regularization strength

### Training Parameters
- `lr`: [1e-5 - 1e-3] - Learning rate (log scale)
- `batch_size`: [64, 128, 256, 512] - Training batch size
- `weight_decay`: [1e-5 - 1e-2] - L2 regularization
- `grad_clip`: [0.3 - 2.0] - Gradient clipping threshold

### Data Parameters
- `lookback`: [64, 96, 128, 160] - Input sequence length
- `stride`: [1, 2, 3, 4] - Sample stride for dataset

## Installation

Install Optuna first:

```bash
pip install optuna
```

## Usage

### Quick Start (Fast Tuning)

For rapid testing (20 trials, 30 minutes):

```bash
python hyperparameter_tuning.py --trials 20 --timeout 1800 --symbols 20 --epochs 20
```

### Standard Tuning (Recommended)

Balanced tuning (50 trials, 1 hour):

```bash
python hyperparameter_tuning.py --trials 50 --timeout 3600 --symbols 30 --epochs 30
```

### Thorough Tuning (Best Results)

Comprehensive search (100 trials, 3 hours):

```bash
python hyperparameter_tuning.py --trials 100 --timeout 10800 --symbols 50 --epochs 40
```

### Command Line Options

- `--trials N`: Number of hyperparameter combinations to try (default: 50)
- `--timeout S`: Maximum seconds to run (default: 3600)
- `--symbols N`: Number of symbols to use per trial (default: 30)
- `--epochs N`: Training epochs per trial (default: 30)
- `--apply`: Apply best parameters to config.yaml

## How It Works

### 1. Smart Search Strategy

The tuning uses **TPE (Tree-structured Parzen Estimator)** sampling:
- Learns from previous trials
- Focuses on promising parameter regions
- More efficient than random or grid search

### 2. Early Stopping & Pruning

- **Trial Pruning**: Stops unpromising trials early (saves time)
- **Early Stopping**: Each trial stops if no improvement after 5 epochs

### 3. Optimization Metric

Optimizes a **composite score** that balances:
- Validation loss (prediction accuracy)
- Calibration quality (quantile coverage at 80% and 90%)

Formula:
```
score = val_loss + calibration_penalty
```

Where calibration_penalty penalizes:
- TP80 coverage outside [70%, 90%] range
- TP90 coverage outside [80%, 95%] range

## Results

### During Tuning

You'll see real-time progress:
```
[I 2026-01-08 10:15:32] Trial 5 finished with value: 0.0234
[I 2026-01-08 10:18:45] Trial 6 pruned at epoch 12
[I 2026-01-08 10:21:15] Trial 7 finished with value: 0.0189 (best)
```

### After Completion

Results are saved to: `data/hyperparameter_tuning_results.json`

Example output:
```
================================================================================
HYPERPARAMETER TUNING RESULTS
================================================================================

Best Trial: 23
Best Value (composite score): 0.018654

Best Hyperparameters:
  hidden_size: 256
  num_layers: 4
  dropout: 0.25
  lr: 0.00012
  batch_size: 128
  weight_decay: 0.0008
  lookback: 128
  stride: 2
  grad_clip: 0.6
```

### Applying Results

Apply the best parameters to your config:

```bash
python hyperparameter_tuning.py --apply
```

This will:
1. Backup your current config to `config.yaml.backup`
2. Update `config.yaml` with optimized parameters
3. Preserve all other settings

## Interpretation

### Good Results
- **Composite score < 0.02**: Excellent model performance
- **TP80 coverage 75-85%**: Well-calibrated predictions
- **TP90 coverage 85-92%**: Good confidence intervals

### Warning Signs
- **Composite score > 0.05**: Poor predictions, may need more data
- **TP80 coverage < 70% or > 90%**: Miscalibrated, adjust dropout/weight_decay
- **Trials pruned early**: May need longer training, increase `--epochs`

## Advanced Usage

### Manual Parameter Ranges

Edit `hyperparameter_tuning.py` to customize search space:

```python
# In objective_function():
hidden_size = trial.suggest_categorical("hidden_size", [256, 384, 512])  # Larger models
dropout = trial.suggest_float("dropout", 0.2, 0.5)  # More regularization
```

### Different Optimization Goals

Modify the composite score calculation:

```python
# Focus more on calibration than loss
composite_score = final_val_loss * 0.5 + calibration_penalty * 2.0

# Or optimize for directional accuracy
directional_bonus = calculate_directional_accuracy(model, val_loader)
composite_score = final_val_loss - directional_bonus * 0.1
```

### Parallel Tuning

For multiple GPUs or machines:

```python
# Use SQLite storage for distributed tuning
import optuna

storage = "sqlite:///hyperparameter_tuning.db"
study = optuna.create_study(storage=storage, study_name="ml_tuning")
```

Then run multiple instances:
```bash
# Terminal 1
python hyperparameter_tuning.py --trials 25

# Terminal 2
python hyperparameter_tuning.py --trials 25
```

## Performance Tips

### Speed Up Tuning

1. **Reduce data**: Use fewer symbols and shorter history
   ```bash
   python hyperparameter_tuning.py --symbols 15 --epochs 20
   ```

2. **Cache OHLCV data**: Set `cache_ohlcv: true` in config.yaml

3. **Use GPU**: Automatic if CUDA available

4. **Increase pruning aggressiveness**:
   ```python
   pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=5)
   ```

### Improve Results

1. **More trials**: Use `--trials 100+` for thorough search

2. **Longer training**: Use `--epochs 50` for better convergence

3. **More data**: Use `--symbols 50+` for generalization

4. **Multiple runs**: Run tuning several times, compare results

## Integration with Training

After tuning, the regular training script will use the optimized parameters:

```bash
# Apply best parameters
python hyperparameter_tuning.py --apply

# Train with optimized settings
python ml/train_torch_forecaster.py
```

## Troubleshooting

### "Insufficient training data" Error
- Increase `--symbols`
- Check if exchange API is rate-limiting
- Verify cache directory has data

### All Trials Pruned
- Increase `--epochs` (trials need more time to show improvement)
- Reduce pruning aggressiveness
- Check if data quality is poor

### Out of Memory
- Reduce `batch_size` search space
- Reduce `hidden_size` options
- Use fewer symbols per trial

### Poor Convergence
- Increase `tuning_epochs` in the script
- Adjust learning rate range
- Check feature normalization

## Example Workflow

Complete tuning workflow:

```bash
# 1. Quick test (5 minutes)
python hyperparameter_tuning.py --trials 10 --timeout 300 --symbols 10 --epochs 15

# 2. Review results
cat data/hyperparameter_tuning_results.json

# 3. If promising, do thorough tuning (2 hours)
python hyperparameter_tuning.py --trials 80 --timeout 7200 --symbols 40 --epochs 35

# 4. Apply best parameters
python hyperparameter_tuning.py --apply

# 5. Full training with optimized settings
python ml/train_torch_forecaster.py

# 6. Evaluate performance
python analyze_ml_performance.py
```

## Expected Improvements

After hyperparameter optimization, you should see:

- **5-15% better validation loss**
- **Improved calibration**: TP80/TP90 coverage closer to targets
- **Better generalization**: Performance more consistent across symbols
- **Faster convergence**: Training reaches optimum in fewer epochs

## Best Practices

1. **Tune periodically**: Re-run every 3-6 months as markets change
2. **Version results**: Keep history of tuning runs
3. **Validate on new data**: Test on unseen time periods
4. **Document changes**: Note performance before/after tuning
5. **Start conservative**: Begin with small search space, expand if needed

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Hyperparameter Tuning Best Practices](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
