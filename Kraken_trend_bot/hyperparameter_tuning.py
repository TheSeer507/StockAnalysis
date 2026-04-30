#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for Torch TP Forecaster

This script performs systematic hyperparameter optimization using Optuna.
It searches for optimal combinations of:
- Model architecture (hidden_size, num_layers, dropout)
- Training parameters (lr, batch_size, weight_decay)
- Data parameters (lookback, stride)

Usage:
    python hyperparameter_tuning.py --trials 50 --timeout 3600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add bot directory to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from ml.train_torch_forecaster import (
    load_config,
    create_exchange,
    load_or_fetch_ohlcv,
    build_samples_for_symbol,
    split_train_val_timewise,
    evaluate_with_coverage,
    canonical_asset,
    should_train_base,
    is_spot_market,
    _safe_float,
)
from bot.torch_tp_forecaster import QuantileGRU, quantile_loss


def objective_function(
    trial,
    symbols: list,
    ex,
    cfg: dict,
    device: torch.device,
    fixed_params: dict,
) -> float:
    """
    Objective function for hyperparameter optimization.
    Returns validation loss (lower is better).
    """
    
    # Hyperparameters to tune
    hidden_size = trial.suggest_categorical("hidden_size", [128, 192, 256, 320, 384])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    lookback = trial.suggest_categorical("lookback", [32, 48, 64, 96])
    stride = trial.suggest_int("stride", 1, 4)
    grad_clip = trial.suggest_float("grad_clip", 0.3, 2.0, step=0.1)
    
    ml = cfg.get("ml", {}) or {}
    ex_cfg = cfg.get("exchange", {}) or {}
    
    timeframe = str(ex_cfg.get("timeframe", "15m"))
    history_days = int(ml.get("history_days", 365))
    horizon_bars = int(ml.get("horizon_bars", 96))
    val_frac = float(ml.get("val_frac", 0.20))
    
    ema_fast = int(ml.get("ema_fast", 20))
    ema_slow = int(ml.get("ema_slow", 50))
    rsi_period = int(ml.get("rsi_period", 14))
    atr_period = int(ml.get("atr_period", 14))
    
    page_limit = int(ml.get("ohlcv_page_limit", 720))
    sleep_sec = float(ml.get("rate_limit_sleep_sec", 0.3))
    # Always use cache during tuning to avoid hitting Kraken rate limits
    # (30 symbols × 50 trials = 1500 fetches otherwise)
    use_cache = True
    
    quantiles = tuple(float(x) for x in (ml.get("quantiles") or [0.5, 0.8, 0.9]))
    input_size = 24  # 20 original + 4 new features (order_flow, bollinger_%B, vwap_dist, btc_beta)
    
    # Enforce minimum stride to reduce sample overlap
    min_stride = max(1, horizon_bars // 4)
    stride = max(min_stride, stride)
    
    # Reduced epochs for faster tuning
    epochs = fixed_params.get("tuning_epochs", 30)
    use_amp = bool(ml.get("use_amp", True))
    
    # Build dataset - collect all samples, then split the pool
    X_all = []
    y_all = []
    
    # Use subset of symbols for faster tuning
    max_symbols = fixed_params.get("tuning_symbols", min(30, len(symbols)))
    tuning_symbols = symbols[:max_symbols]
    
    # Use smaller stride in tuning to get more samples from limited candles
    # Kraken often only returns ~720 candles for 15m, so be aggressive
    tuning_stride = max(1, min(stride, 4))
    
    for sym in tuning_symbols:
        try:
            ohlcv = load_or_fetch_ohlcv(
                ex=ex,
                symbol=sym,
                timeframe=timeframe,
                history_days=history_days,
                page_limit=page_limit,
                sleep_sec=sleep_sec,
                use_cache=use_cache,
            )
            if len(ohlcv) < (lookback + horizon_bars + 10):
                continue

            # Adaptive stride for limited data
            n_available = len(ohlcv) - lookback - horizon_bars
            sym_stride = tuning_stride
            if n_available > 0 and n_available // sym_stride < 20 and sym_stride > 1:
                sym_stride = max(1, n_available // 20)
            
            X, y = build_samples_for_symbol(
                ohlcv=ohlcv,
                lookback=lookback,
                horizon_bars=horizon_bars,
                stride=sym_stride,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                rsi_period=rsi_period,
                atr_period=atr_period,
            )
            
            if X and y:
                X_all.extend(X)
                y_all.extend(y)
                
        except Exception as e:
            continue
    
    if len(X_all) < 60:
        raise ValueError(f"Insufficient training data: {len(X_all)} total samples")
    
    # Split the pooled dataset into train/val (time-wise per the overall pool)
    n_total = len(y_all)
    split_idx = int(n_total * (1.0 - val_frac))
    split_idx = max(1, min(split_idx, n_total - 1))
    
    X_train_all = X_all[:split_idx]
    y_train_all = y_all[:split_idx]
    X_val_all = X_all[split_idx:]
    y_val_all = y_all[split_idx:]
    
    # Normalize features
    X_train_tensor = torch.tensor(X_train_all, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_all, dtype=torch.float32)
    
    feat_mean = X_train_tensor.view(-1, input_size).mean(dim=0)
    feat_std = X_train_tensor.view(-1, input_size).std(dim=0) + 1e-8
    
    X_train_tensor = (X_train_tensor - feat_mean) / feat_std
    X_val_tensor = (X_val_tensor - feat_mean) / feat_std
    
    y_train_tensor = torch.tensor(y_train_all, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_all, dtype=torch.float32)
    
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    val_ds = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model
    model = QuantileGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        quantiles=quantiles,
    ).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Use OneCycleLR for better convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr * 10,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 5  # Reduced for tuning
    
    for ep in range(1, epochs + 1):
        model.train()
        
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            opt.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                pred = model(xb)
                loss = quantile_loss(pred, yb, quantiles)
            
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
        
        # Validation (returns: val_loss, cov50, cov80, cov90, avg_width)
        val_loss, _cov50, cov80, cov90, avg_width = evaluate_with_coverage(model, val_loader, device, quantiles)
        
        # Report intermediate value for pruning
        trial.report(val_loss, ep)
        
        # Check for pruning
        if trial.should_prune():
            import optuna
            raise optuna.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break
    
    # Calculate final metrics (returns: val_loss, cov50, cov80, cov90, avg_width)
    final_val_loss, _final_cov50, final_cov80, final_cov90, final_width = evaluate_with_coverage(
        model, val_loader, device, quantiles
    )
    
    # Composite metric: balance loss with calibration
    # Coverage = P(actual >= predicted_quantile)
    # For q0.8 coverage should be ~20%, for q0.9 should be ~10%
    calibration_penalty = 0.0
    if final_cov80 < 0.12 or final_cov80 > 0.30:
        calibration_penalty += abs(final_cov80 - 0.20) * 2.0
    if final_cov90 < 0.05 or final_cov90 > 0.18:
        calibration_penalty += abs(final_cov90 - 0.10) * 1.0
    
    composite_score = final_val_loss + calibration_penalty
    
    return composite_score


def run_hyperparameter_tuning(
    n_trials: int = 50,
    timeout: int = 3600,
    tuning_symbols: int = 30,
    tuning_epochs: int = 30,
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        n_trials: Number of trials to run
        timeout: Maximum time in seconds
        tuning_symbols: Number of symbols to use for tuning (subset for speed)
        tuning_epochs: Number of epochs per trial (reduced for speed)
    
    Returns:
        Dictionary with best parameters and study results
    """
    
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except ImportError:
        print("ERROR: Optuna not installed. Install with: pip install optuna")
        sys.exit(1)
    
    cfg = load_config()
    ml = cfg.get("ml", {}) or {}
    ex_cfg = cfg.get("exchange", {}) or {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TUNING] Using device: {device}")
    
    # Create exchange and get symbols
    ex = create_exchange(cfg)
    markets = ex.markets or {}
    base_currency = canonical_asset(ex_cfg.get("base_currency", "USD"))
    stable_assets = (cfg.get("portfolio", {}) or {}).get("stable_assets", ["USD", "USDT", "USDC"])
    
    tickers = ex.fetch_tickers() or {}
    candidates = []
    for sym, t in tickers.items():
        if "/" not in sym:
            continue
        base, quote = sym.split("/", 1)
        base = canonical_asset(base)
        quote = canonical_asset(quote)
        if quote != base_currency:
            continue
        m = markets.get(sym) or {}
        if not is_spot_market(m):
            continue
        if not should_train_base(base, stable_assets):
            continue
        qv = _safe_float(t.get("quoteVolume") or t.get("quote_volume") or 0.0, 0.0)
        if qv <= 0:
            continue
        candidates.append((sym, qv))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    max_symbols = int(ml.get("train_symbols", 150))
    symbols = [s for s, _ in candidates[:max_symbols]]
    
    print(f"[TUNING] Available symbols: {len(symbols)}")
    print(f"[TUNING] Will use {tuning_symbols} symbols per trial for speed")
    
    fixed_params = {
        "tuning_symbols": tuning_symbols,
        "tuning_epochs": tuning_epochs,
    }
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    
    print(f"[TUNING] Starting hyperparameter search...")
    print(f"[TUNING] Trials: {n_trials}, Timeout: {timeout}s")
    
    # Pre-fetch and cache data for all symbols before tuning starts
    print(f"[TUNING] Pre-fetching OHLCV data for {len(symbols[:tuning_symbols])} symbols...")
    _prefetch_ml = cfg.get("ml", {}) or {}
    _prefetch_tf = str(ex_cfg.get("timeframe", "15m"))
    _prefetch_hd = int(_prefetch_ml.get("history_days", 120))
    _prefetch_pl = int(_prefetch_ml.get("ohlcv_page_limit", 720))
    _prefetch_sl = float(_prefetch_ml.get("rate_limit_sleep_sec", 0.3))
    for _i, _sym in enumerate(symbols[:tuning_symbols], 1):
        try:
            _data = load_or_fetch_ohlcv(ex=ex, symbol=_sym, timeframe=_prefetch_tf,
                                         history_days=_prefetch_hd, page_limit=_prefetch_pl,
                                         sleep_sec=_prefetch_sl, use_cache=True, verbose=False)
            if _i % 10 == 0 or _i == 1:
                print(f"  [{_i}/{tuning_symbols}] {_sym}: {len(_data)} candles")
        except Exception:
            pass
    print(f"[TUNING] Pre-fetch complete.\n")

    study.optimize(
        lambda trial: objective_function(
            trial, symbols, ex, cfg, device, fixed_params
        ),
        n_trials=n_trials,
        timeout=timeout,
        catch=(ValueError, RuntimeError),
        show_progress_bar=True,
    )
    
    # Results
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*80)
    
    completed_trials = [t for t in study.trials if t.value is not None]
    if not completed_trials:
        print("\n❌ All trials failed! No valid results.")
        print("   Check that symbols have enough candle data.")
        print("   Try increasing --epochs or reducing lookback range.")
        return {"error": "all_trials_failed", "n_trials": len(study.trials)}
    
    print(f"\nCompleted Trials: {len(completed_trials)}/{len(study.trials)}")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Value (composite score): {study.best_trial.value:.6f}")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }
    
    results_path = REPO_ROOT / "data" / "hyperparameter_tuning_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, indent=2, fp=f)
    
    print(f"\n[TUNING] Results saved to: {results_path}")
    
    # Generate updated config snippet
    print("\n" + "="*80)
    print("SUGGESTED CONFIG.YAML UPDATES:")
    print("="*80)
    print("\nml:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return results


def apply_best_params_to_config(results_path: Path = None) -> None:
    """
    Apply best hyperparameters from tuning results to config.yaml
    """
    if results_path is None:
        results_path = REPO_ROOT / "data" / "hyperparameter_tuning_results.json"
    
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        return
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    config_path = REPO_ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Update ML section
    ml = cfg.get("ml", {}) or {}
    best_params = results["best_params"]
    
    for key, value in best_params.items():
        ml[key] = value
    
    cfg["ml"] = ml
    
    # Backup original config
    backup_path = config_path.with_suffix(".yaml.backup")
    with open(backup_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    
    # Write updated config
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    
    print(f"[TUNING] Config updated: {config_path}")
    print(f"[TUNING] Backup saved: {backup_path}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for ML forecaster")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
    parser.add_argument("--symbols", type=int, default=30, help="Number of symbols per trial")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per trial")
    parser.add_argument("--apply", action="store_true", help="Apply best params to config.yaml")
    
    args = parser.parse_args()
    
    if args.apply:
        apply_best_params_to_config()
    else:
        results = run_hyperparameter_tuning(
            n_trials=args.trials,
            timeout=args.timeout,
            tuning_symbols=args.symbols,
            tuning_epochs=args.epochs,
        )
        
        print("\n[TUNING] To apply these parameters to config.yaml, run:")
        print(f"  python {__file__} --apply")


if __name__ == "__main__":
    main()
