#!/usr/bin/env python3
"""
Test script to verify data fetching works correctly.
Fetches data for a few symbols to validate the pagination logic.
"""

import sys
from pathlib import Path
import yaml
import ccxt

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from ml.train_torch_forecaster import (
    load_or_fetch_ohlcv,
    canonical_asset,
    create_exchange,
    load_config
)

def test_fetch():
    """Test data fetching for a few symbols."""
    
    print("=" * 70)
    print("DATA FETCH TEST")
    print("=" * 70)
    
    cfg = load_config()
    ex = create_exchange(cfg)
    
    ml_cfg = cfg.get("ml", {}) or {}
    ex_cfg = cfg.get("exchange", {}) or {}
    
    timeframe = str(ex_cfg.get("timeframe", "15m"))
    history_days = int(ml_cfg.get("history_days", 365))
    page_limit = int(ml_cfg.get("ohlcv_page_limit", 720))
    sleep_sec = float(ml_cfg.get("rate_limit_sleep_sec", 1.0))
    
    print(f"\nSettings:")
    print(f"  Exchange: {ex.__class__.__name__}")
    print(f"  Timeframe: {timeframe}")
    print(f"  History Days: {history_days}")
    print(f"  Page Limit: {page_limit}")
    print(f"  Sleep Between Pages: {sleep_sec}s")
    
    # Test with a few major coins
    test_symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    
    for symbol in test_symbols:
        print(f"\n{'=' * 70}")
        print(f"Testing: {symbol}")
        print('=' * 70)
        
        try:
            ohlcv = load_or_fetch_ohlcv(
                ex=ex,
                symbol=symbol,
                timeframe=timeframe,
                history_days=history_days,
                page_limit=page_limit,
                sleep_sec=sleep_sec,
                use_cache=False,  # Force fresh fetch for testing
                verbose=True,
            )
            
            if ohlcv:
                first_ts = ohlcv[0][0]
                last_ts = ohlcv[-1][0]
                span_days = (last_ts - first_ts) / (1000 * 86400)
                coverage_pct = (span_days / history_days) * 100
                
                print(f"\n✓ SUCCESS:")
                print(f"  Total Candles: {len(ohlcv):,}")
                print(f"  Actual Span: {span_days:.1f} days")
                print(f"  Coverage: {coverage_pct:.1f}% of {history_days} days requested")
                
                if coverage_pct >= 95:
                    print(f"  ✓ EXCELLENT - Near complete data")
                elif coverage_pct >= 80:
                    print(f"  ✓ GOOD - Adequate coverage")
                elif coverage_pct >= 50:
                    print(f"  ⚠ FAIR - Partial coverage")
                else:
                    print(f"  ✗ POOR - Insufficient data")
                    
            else:
                print(f"\n✗ FAILED: No data returned")
                
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 70}")
    print("TEST COMPLETE")
    print('=' * 70)

if __name__ == "__main__":
    test_fetch()
