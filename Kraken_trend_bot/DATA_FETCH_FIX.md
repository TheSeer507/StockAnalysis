# Data Fetching Fix Summary

## Problem Identified

The ML model was only training on **7.5 days** of data instead of the requested 365 days, resulting in:
- **Severely miscalibrated predictions** (21% coverage vs 80% target)
- Model predicting take-profit levels too high
- Missing trading opportunities

## Root Cause

**Kraken API Limitation**: The public API only provides ~720 candles per request, which means:
- 15m timeframe: 720 candles = 7.5 days
- 1h timeframe: 720 candles = 30 days  
- 4h timeframe: 720 candles = 120 days
- 1d timeframe: 720 candles = 720 days

## Solutions Implemented

### 1. Improved Data Fetching Logic
Enhanced `fetch_ohlcv_paged()` function in `ml/train_torch_forecaster.py`:
- Better error handling and logging
- Progress tracking for large fetches
- Verbose mode to show fetch details
- Improved cache validation (checks freshness AND data span coverage)

### 2. Configuration Changes
Updated `config/config.yaml`:
```yaml
ml:
  training_timeframe: 4h    # Changed from using 15m (exchange default)
  history_days: 120         # Reduced from 365 (more realistic for Kraken limits)
  horizon_bars: 6           # Reduced from 96 (6 bars @ 4h = 24 hours forecast)
  cache_ohlcv: false        # Disabled to force fresh data fetch
  rate_limit_sleep_sec: 1.0 # Increased for API respect
```

### 3. Training Script Updates
- Added `training_timeframe` config option (separate from live trading timeframe)
- Better logging to track data quality during training
- Clear error messages for insufficient data

## Expected Results

With 4-hour timeframe:
- **120 days of historical data** per symbol (vs 7.5 days before)
- **~720 candles** providing rich market patterns
- **16x more data** for model training
- Better calibration and coverage metrics

## Horizon Adjustment

- **Old**: 96 bars @ 15m = 24 hours forecast
- **New**: 6 bars @ 4h = 24 hours forecast (same prediction window)

## Next Steps

1. **Delete old cache**: `rm -rf data/cache_ohlcv/*`
2. **Retrain model**: `python ml/train_torch_forecaster.py`
3. **Verify results**:
   - Check data span in logs
   - Target: TP80 coverage 75-85%, TP90 coverage 85-92%
   - Monitor for ~120 days per symbol

## Testing

Run test script to verify:
```bash
python test_data_fetch.py
```

Expected output:
- BTC/USD: ~720 candles (120 days) @ 4h timeframe
- Coverage: >95% of requested history

## Trade-offs

**Advantages:**
- Much more historical data
- Better model calibration
- Captures longer-term patterns
- Still maintains 24h prediction horizon

**Considerations:**
- 4h bars are less granular than 15m
- May miss very short-term patterns
- But this is appropriate for swing trading strategy

## Kraken API Limits Summary

| Timeframe | Candles | History | Best For |
|-----------|---------|---------|----------|
| 15m | 720 | 7.5 days | ❌ Too short for training |
| 1h | 720 | 30 days | ⚠️ Minimal but usable |
| 4h | 720 | 120 days | ✅ Recommended |
| 1d | 720 | 720 days | ✅ Maximum history |

## Alternative: Mix Timeframes

For future enhancement, consider:
1. Train on 4h data for pattern recognition
2. Use 15m data for entry/exit timing in live trading
3. Multi-timeframe feature engineering
