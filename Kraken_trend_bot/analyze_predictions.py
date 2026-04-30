#!/usr/bin/env python3
"""
Analyze ML model predictions against actual outcomes.
Shows coverage statistics and per-symbol accuracy.
"""
import json
import torch
from pathlib import Path
from bot.torch_tp_forecaster import QuantileGRU
from ml.train_torch_forecaster import (
    load_config, build_samples_for_symbol, split_train_val_timewise, 
    fetch_ohlcv_paged
)
import ccxt
import time


def analyze_model_predictions():
    """Load model and analyze prediction accuracy."""
    
    # Load config and model
    cfg = load_config()
    ml_cfg = cfg.get("ml", {})
    model_path = Path(ml_cfg.get("model_path", "data/torch_tp_forecaster.pt"))
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    meta = checkpoint["meta"]
    state_dict = checkpoint["state_dict"]
    feat_mean = checkpoint["feat_mean"]
    feat_std = checkpoint["feat_std"]
    
    print(f"Model trained: {meta['trained_at']}")
    print(f"Timeframe: {meta['timeframe']}, History: {meta['history_days']} days")
    print(f"Quantiles: {meta['quantiles']}")
    print(f"\nOverall Validation Metrics:")
    print(f"  TP50 Coverage: {meta['metrics'].get('val_cov50', 0):.2%} (target: ~50%)")
    print(f"  TP80 Coverage: {meta['metrics'].get('val_cov80', 0):.2%} (target: ~80%)")
    print(f"  TP90 Coverage: {meta['metrics'].get('val_cov90', 0):.2%} (target: ~90%)")
    print(f"  Avg Width (q90-q50): {meta['metrics'].get('avg_width', 0):.4f}")
    
    # Initialize model
    model = QuantileGRU(
        input_size=meta['input_size'],
        hidden_size=meta['hidden_size'],
        num_layers=meta['num_layers'],
        dropout=meta['dropout'],
        quantiles=tuple(meta['quantiles'])
    )
    model.load_state_dict(state_dict)
    model.eval()
    
    # Test on a few symbols
    exchange_cfg = cfg.get("exchange", {})
    exchange_name = exchange_cfg.get("name", "kraken")
    api_key = exchange_cfg.get("api_key", "")
    secret = exchange_cfg.get("secret", "")
    
    exchange_cls = getattr(ccxt, exchange_name)
    exchange = exchange_cls({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
    })
    
    timeframe = meta.get('timeframe', '15m')
    history_days = meta.get('history_days', 7)
    lookback = meta['lookback']
    horizon_bars = meta['horizon_bars']
    
    # Test symbols
    test_symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD"]
    
    print(f"\n{'='*80}")
    print(f"PER-SYMBOL PREDICTION ANALYSIS (using validation set)")
    print(f"{'='*80}\n")
    
    for symbol in test_symbols:
        try:
            print(f"Analyzing {symbol}...")
            
            # Fetch data
            now_ms = int(time.time() * 1000)
            since_ms = now_ms - (history_days * 86400 * 1000)
            
            rows = fetch_ohlcv_paged(
                exchange, symbol, timeframe,
                since_ms=since_ms,
                until_ms=now_ms,
                page_limit=720,
                sleep_sec=1.0,
                verbose=False
            )
            
            if len(rows) < lookback + horizon_bars:
                print(f"  ⚠️  Insufficient data ({len(rows)} candles)\n")
                continue
            
            # Build features
            X, y = build_samples_for_symbol(
                rows,
                lookback=lookback,
                horizon_bars=horizon_bars,
                stride=1,
                ema_fast=meta['ema_fast'],
                ema_slow=meta['ema_slow'],
                rsi_period=meta['rsi_period'],
                atr_period=meta['atr_period']
            )
            
            if len(y) < 20:
                print(f"  ⚠️  Insufficient samples ({len(y)})\n")
                continue
            
            # Use only validation set (last 20%)
            _, _, X_val, y_val = split_train_val_timewise(X, y, val_frac=0.2)
            
            # Normalize
            X_tensor = torch.tensor(X_val, dtype=torch.float32)
            X_tensor = (X_tensor - torch.tensor(feat_mean)) / (torch.tensor(feat_std) + 1e-8)
            y_tensor = torch.tensor(y_val, dtype=torch.float32)
            
            # Predict
            with torch.no_grad():
                preds = model(X_tensor)  # Shape: (N, 3) for [q50, q80, q90]
            
            # Calculate coverage for each quantile
            q50_pred = preds[:, 0]
            q80_pred = preds[:, 1]
            q90_pred = preds[:, 2]
            
            cov50 = (y_tensor >= q50_pred).float().mean().item()
            cov80 = (y_tensor >= q80_pred).float().mean().item()
            cov90 = (y_tensor >= q90_pred).float().mean().item()
            
            # Calculate mean absolute error for median
            mae50 = (y_tensor - q50_pred).abs().mean().item()
            
            # Calculate prediction ranges
            avg_pred_q50 = q50_pred.mean().item()
            avg_pred_q80 = q80_pred.mean().item()
            avg_pred_q90 = q90_pred.mean().item()
            avg_actual = y_tensor.mean().item()
            
            print(f"  Samples: {len(y_val)}")
            print(f"  TP50 Coverage: {cov50:.2%} {'✓' if 0.40 <= cov50 <= 0.60 else '✗'}  (target: ~50%)")
            print(f"  TP80 Coverage: {cov80:.2%} {'✓' if 0.15 <= cov80 <= 0.30 else '✗'}  (target: ~20%)")
            print(f"  TP90 Coverage: {cov90:.2%} {'✓' if 0.05 <= cov90 <= 0.15 else '✗'}  (target: ~10%)")
            print(f"  Median MAE: {mae50:.4f}")
            print(f"  Avg Predictions: TP50={avg_pred_q50:+.4f}, TP80={avg_pred_q80:+.4f}, TP90={avg_pred_q90:+.4f}")
            print(f"  Avg Actual Return: {avg_actual:+.4f}")
            print()
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            continue
    
    print(f"\n{'='*80}")
    print("INTERPRETATION GUIDE:")
    print("=" * 80)
    print("Coverage = % of actual returns that EXCEEDED the predicted quantile")
    print("  • TP50: Should be ~50% (half above, half below the median prediction)")
    print("  • TP80: Should be ~20% (only 20% of actuals should exceed this high target)")
    print("  • TP90: Should be ~10% (only 10% of actuals should exceed this very high target)")
    print("")
    print("If TP80 coverage is near 0%:  Model is TOO OPTIMISTIC (targets never reached)")
    print("If TP80 coverage is >> 20%:   Model is TOO CONSERVATIVE (easy targets)")
    print("If TP80 coverage is ~20%:     Model is WELL CALIBRATED ✓")
    print("=" * 80)


if __name__ == "__main__":
    analyze_model_predictions()
