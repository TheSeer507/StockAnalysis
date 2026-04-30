#!/usr/bin/env python3
"""
Quick diagnostic script to check bot configuration and identify issues.
"""

import sys
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
MODEL_PATH = REPO_ROOT / "data" / "torch_tp_forecaster.pt"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    print("="*70)
    print("BOT DIAGNOSTICS")
    print("="*70)
    
    # Check config
    print("\n1. CONFIG FILE")
    if CONFIG_PATH.exists():
        print(f"   ✅ Config exists: {CONFIG_PATH}")
        cfg = load_config()
        
        # Check screener settings
        scr = cfg.get("screener", {}) or {}
        reco_every = int(scr.get("reco_every_loops", 60))
        reco_count = int(scr.get("recommendations_count", 5))
        
        print(f"   📊 Recommendations: {reco_count} every {reco_every} loops")
        
        poll = int((cfg.get("bot", {}) or {}).get("poll_interval_sec", 60))
        print(f"   ⏱️  Poll interval: {poll}s")
        print(f"   ⏱️  Recommendations every: {reco_every * poll}s = {reco_every * poll / 60:.1f} min")
        
        # Check portfolio scan
        portfolio_scan_every = int((cfg.get("bot", {}) or {}).get("portfolio_scan_every_loops", 10))
        print(f"   📈 Portfolio scan every: {portfolio_scan_every} loops = {portfolio_scan_every * poll}s")
        
        # Check ML settings
        ml = cfg.get("ml", {}) or {}
        ml_enabled = bool(ml.get("enabled", False))
        print(f"\n   ML Status: {'✅ ENABLED' if ml_enabled else '❌ DISABLED'}")
        
        if ml_enabled:
            min_tp80_pct = float(ml.get("min_tp80_pct", 0.06))
            print(f"   ML min_tp80_pct: {min_tp80_pct*100:.1f}% (filters out low predictions)")
        
        # Check strategy settings
        pm = cfg.get("position_management", {}) or {}
        require_long = bool(pm.get("require_long_for_entry", True))
        print(f"\n   Strategy filters:")
        print(f"   - require_long_for_entry: {require_long}")
        
    else:
        print(f"   ❌ Config NOT found: {CONFIG_PATH}")
    
    # Check ML model
    print("\n2. ML MODEL")
    if MODEL_PATH.exists():
        print(f"   ✅ Model exists: {MODEL_PATH}")
        
        # Try to load and check
        try:
            import torch
            blob = torch.load(MODEL_PATH, map_location="cpu")
            meta = blob.get("meta", {})
            
            input_size = meta.get("input_size", "unknown")
            print(f"   📊 Model input_size: {input_size}")
            
            if input_size == 8:
                print(f"   ⚠️  WARNING: Old model (8 features) - needs retraining!")
                print(f"   🔄 Run: python -m ml.train_torch_forecaster")
            elif input_size == 20:
                print(f"   ⚠️  WARNING: Old model (20 features) - needs retraining for v3 architecture!")
                print(f"   🔄 Run: python -m ml.train_torch_forecaster")
            elif input_size == 24:
                print(f"   ✅ Current model architecture (24 features + attention)")
            
            trained_at = meta.get("trained_at", "unknown")
            print(f"   📅 Trained: {trained_at}")
            
            # Check for normalization params
            has_norm = "feat_mean" in blob and "feat_std" in blob
            print(f"   🔢 Normalization params: {'✅ Present' if has_norm else '❌ Missing (old model)'}")
            
        except Exception as e:
            print(f"   ❌ Failed to inspect model: {e}")
    else:
        print(f"   ❌ Model NOT found: {MODEL_PATH}")
        print(f"   🔄 Train with: python -m ml.train_torch_forecaster")
    
    # Check data directory
    print("\n3. DATA DIRECTORY")
    data_dir = REPO_ROOT / "data"
    if data_dir.exists():
        print(f"   ✅ Data dir exists: {data_dir}")
        
        # Check for portfolio files
        paper_port = data_dir / "paper_portfolio.json"
        live_state = data_dir / "live_positions.json"
        
        if paper_port.exists():
            print(f"   ✅ Paper portfolio: {paper_port}")
        else:
            print(f"   ⚠️  No paper_portfolio.json (will be created)")
        
        if live_state.exists():
            print(f"   ✅ Live state: {live_state}")
        else:
            print(f"   ⚠️  No live_positions.json (will be created)")
    else:
        print(f"   ❌ Data dir NOT found: {data_dir}")
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    if not MODEL_PATH.exists():
        print("❌ ISSUE: ML model missing")
        print("   FIX: Run training: python -m ml.train_torch_forecaster")
    elif MODEL_PATH.exists():
        try:
            import torch
            blob = torch.load(MODEL_PATH, map_location="cpu")
            meta = blob.get("meta", {})
            input_size = meta.get("input_size", 8)
            has_norm = "feat_mean" in blob and "feat_std" in blob
            
            if input_size != 24 or not has_norm:
                print("❌ ISSUE: ML model is from old architecture (incompatible)")
                print(f"   DETAILS: input_size={input_size} (expected 24), norms={'present' if has_norm else 'missing'}")
                print("   IMPACT: Recommendations won't work properly")
                print("   FIX: Retrain model with: python -m ml.train_torch_forecaster")
                print("   TIME: ~45-70 minutes")
            else:
                print("✅ ML model looks good (v3 architecture with attention)")
        except:
            pass
    
    cfg = load_config()
    scr = cfg.get("screener", {}) or {}
    reco_every = int(scr.get("reco_every_loops", 60))
    poll = int((cfg.get("bot", {}) or {}).get("poll_interval_sec", 60))
    
    if reco_every > 60:
        print(f"⚠️  INFO: Recommendations run every {reco_every * poll / 60:.1f} minutes")
        print(f"   If bot just started, wait {reco_every * poll / 60:.1f} min for first recommendations")
    
    print("\n" + "="*70)
    print("\nTo see detailed bot output, check:")
    print("  - Terminal where bot is running")
    print("  - Look for '[SCREENER]' and '[PORTFOLIO]' messages")
    print("="*70)


if __name__ == "__main__":
    main()
