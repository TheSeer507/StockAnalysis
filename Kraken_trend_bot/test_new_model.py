#!/usr/bin/env python3
"""
Test the newly trained model on recent data to see prediction quality
"""

import sys
from pathlib import Path
from datetime import datetime
import yaml
import ccxt
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from bot.torch_tp_forecaster import QuantileGRU, build_seq_features

CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
MODEL_PATH = REPO_ROOT / "data" / "torch_tp_forecaster.pt"
META_PATH = REPO_ROOT / "data" / "torch_tp_forecaster.meta.json"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def test_model_predictions():
    """Test model on recent data for a few popular symbols"""
    
    print("="*80)
    print("TESTING NEWLY TRAINED MODEL")
    print("="*80)
    print()
    
    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print("❌ Model file not found!")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Reconstruct model from state dict and meta
    meta = bundle["meta"]
    
    model = QuantileGRU(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
        quantiles=tuple(meta["quantiles"])
    ).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    
    feat_mean = torch.tensor(bundle["feat_mean"], device=device)
    feat_std = torch.tensor(bundle["feat_std"], device=device)
    quantiles = meta["quantiles"]
    lookback = meta["lookback"]
    horizon_bars = meta["horizon_bars"]
    
    print(f"✅ Model loaded (device: {device})")
    print(f"   Lookback: {lookback} bars")
    print(f"   Horizon: {horizon_bars} bars (~{horizon_bars*15/60:.1f} hours)")
    print(f"   Quantiles: {quantiles}")
    print()
    
    # Setup exchange
    cfg = load_config()
    exch_cfg = cfg.get("exchange", {})
    exchange_name = exch_cfg.get("name", "kraken")
    timeframe = exch_cfg.get("timeframe", "15m")
    
    ex_cls = getattr(ccxt, exchange_name)
    exchange = ex_cls({"enableRateLimit": True})
    exchange.load_markets()
    
    print(f"Connected to {exchange_name}")
    print()
    
    # Test on popular symbols
    test_symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "BNB/USD"]
    
    print("PREDICTIONS ON RECENT DATA:")
    print("-"*80)
    
    for symbol in test_symbols:
        try:
            if symbol not in exchange.markets:
                print(f"⚠️  {symbol:12s} - Not available on {exchange_name}")
                continue
            
            # Fetch recent data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=lookback)
            
            if len(ohlcv) < lookback:
                print(f"⚠️  {symbol:12s} - Insufficient data")
                continue
            
            # Build features
            features = build_seq_features(ohlcv)
            if len(features) < lookback:
                print(f"⚠️  {symbol:12s} - Feature building failed")
                continue
            
            # Normalize
            X = torch.tensor(features[-lookback:], dtype=torch.float32, device=device)
            X = (X - feat_mean) / (feat_std + 1e-9)
            X = X.unsqueeze(0)  # batch dimension
            
            # Predict
            with torch.no_grad():
                preds = model(X).cpu().numpy()[0]
            
            current_price = float(ohlcv[-1][4])  # close price
            
            # Calculate TP prices
            tp50_price = current_price * (1 + preds[0])
            tp80_price = current_price * (1 + preds[1])
            tp90_price = current_price * (1 + preds[2])
            
            # Format output
            print(f"{symbol:12s} @ ${current_price:>10,.2f}")
            print(f"             TP50: ${tp50_price:>10,.2f} ({preds[0]:+6.2%})")
            print(f"             TP80: ${tp80_price:>10,.2f} ({preds[1]:+6.2%})")
            print(f"             TP90: ${tp90_price:>10,.2f} ({preds[2]:+6.2%})")
            
            # Check if quantiles are properly ordered
            if preds[0] <= preds[1] <= preds[2]:
                print(f"             ✅ Quantiles properly ordered")
            else:
                print(f"             ⚠️  Quantiles NOT ordered correctly")
            
            print()
            
        except Exception as e:
            print(f"❌ {symbol:12s} - Error: {e}")
            print()
    
    print("="*80)
    print("Test completed. Compare these predictions with actual price movements")
    print("over the next ~24 hours to evaluate model quality.")
    print("="*80)


if __name__ == "__main__":
    test_model_predictions()
