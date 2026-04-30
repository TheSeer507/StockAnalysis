#!/usr/bin/env python3
"""
Backfill ML prediction outcomes from actual Kraken price data.

This script is designed for HOLD-style users who don't actively trade through
the bot.  It fetches real OHLCV data from Kraken and checks what the actual
peak price was within the prediction horizon (24 hours at 15m) after each
prediction was logged.

Usage:
    python backfill_outcomes.py                  # backfill all pending
    python backfill_outcomes.py --days 7         # only last 7 days
    python backfill_outcomes.py --dry-run        # preview without saving
    python backfill_outcomes.py --report         # backfill + print report
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import yaml


def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_exchange(cfg: dict) -> ccxt.Exchange:
    ex_cfg = cfg.get("exchange", {})
    name = ex_cfg.get("name", "kraken")
    cls = getattr(ccxt, name)
    return cls({
        "apiKey": ex_cfg.get("api_key", ""),
        "secret": ex_cfg.get("secret", ""),
        "enableRateLimit": True,
    })


def fetch_ohlcv_window(
    exchange: ccxt.Exchange,
    symbol: str,
    since_ms: int,
    until_ms: int,
    timeframe: str = "15m",
    page_limit: int = 720,
) -> List[List[float]]:
    """Fetch OHLCV candles for a specific time window."""
    all_rows: List[List[float]] = []
    cursor = since_ms
    while cursor < until_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=page_limit)
        except Exception:
            break
        if not batch:
            break
        all_rows.extend(batch)
        cursor = int(batch[-1][0]) + 1
        if len(batch) < page_limit:
            break
        time.sleep(1.0)

    # Deduplicate by timestamp and filter to window
    seen = set()
    result = []
    for row in all_rows:
        ts = int(row[0])
        if ts not in seen and ts <= until_ms:
            seen.add(ts)
            result.append(row)
    result.sort(key=lambda r: r[0])
    return result


def backfill_outcomes(
    predictions_path: Path,
    exchange: ccxt.Exchange,
    horizon_hours: float = 24.0,
    days_filter: Optional[int] = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> Tuple[int, int, int]:
    """
    Backfill prediction outcomes from actual Kraken price data.

    Returns: (total_pending, successfully_filled, skipped_too_recent)
    """
    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both flat list and wrapped dict formats
    if isinstance(data, dict):
        records = data.get("records", [])
        is_wrapped = True
    else:
        records = data
        is_wrapped = False

    now = time.time()
    horizon_sec = horizon_hours * 3600
    cutoff_ts = (now - days_filter * 86400) if days_filter else 0

    # Find records that need outcomes
    pending = [
        (i, r) for i, r in enumerate(records)
        if not r.get("outcome_recorded", False)
        and r.get("timestamp", 0) >= cutoff_ts
    ]

    if verbose:
        print(f"Total records: {len(records)}")
        print(f"Pending (no outcome): {len(pending)}")

    # Group by symbol to minimize API calls
    by_symbol: Dict[str, List[Tuple[int, dict]]] = {}
    too_recent = 0
    for idx, rec in pending:
        # Skip if the prediction is too recent (horizon hasn't elapsed)
        pred_ts = rec.get("timestamp", 0)
        if now - pred_ts < horizon_sec:
            too_recent += 1
            continue
        sym = rec.get("symbol", "")
        if sym:
            by_symbol.setdefault(sym, []).append((idx, rec))

    if verbose:
        print(f"Too recent (horizon not elapsed): {too_recent}")
        print(f"Symbols to fetch: {len(by_symbol)}")
        print()

    filled = 0
    for sym_idx, (symbol, recs) in enumerate(sorted(by_symbol.items()), 1):
        if verbose:
            print(f"[{sym_idx}/{len(by_symbol)}] Fetching {symbol} ({len(recs)} predictions)...")

        # Find the time window we need: earliest prediction to latest + horizon
        earliest_ts = min(r["timestamp"] for _, r in recs)
        latest_ts = max(r["timestamp"] for _, r in recs)
        since_ms = int((earliest_ts - 900) * 1000)  # 1 bar before
        until_ms = int((latest_ts + horizon_sec + 900) * 1000)  # horizon + 1 bar after

        try:
            candles = fetch_ohlcv_window(exchange, symbol, since_ms, until_ms)
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed to fetch: {e}")
            continue

        if not candles:
            if verbose:
                print(f"  ✗ No candle data returned")
            continue

        # Build a quick lookup: list of (timestamp_sec, high)
        highs_timeline = [(int(c[0]) / 1000, float(c[2])) for c in candles]

        for idx, rec in recs:
            pred_ts = rec["timestamp"]
            entry_price = rec["entry_price"]
            if entry_price <= 0:
                continue

            # Find all highs within the horizon window after this prediction
            window_start = pred_ts
            window_end = pred_ts + horizon_sec
            future_highs = [h for ts, h in highs_timeline if window_start < ts <= window_end]

            if not future_highs:
                continue

            peak_price = max(future_highs)
            peak_return = (peak_price / entry_price) - 1.0
            time_to_peak = 0.0

            # Find when the peak occurred
            for ts, h in highs_timeline:
                if window_start < ts <= window_end and h == peak_price:
                    time_to_peak = (ts - pred_ts) / 3600.0
                    break

            # Update the record
            if not dry_run:
                records[idx]["actual_peak_price"] = peak_price
                records[idx]["actual_peak_return"] = peak_return
                records[idx]["time_to_peak_hours"] = time_to_peak
                records[idx]["outcome_recorded"] = True

            filled += 1

        if verbose:
            sym_filled = sum(
                1 for _, r in recs
                if (not dry_run and records[_].get("outcome_recorded"))
                or dry_run  # in dry-run we just count
            )
            print(f"  ✓ Filled {sym_filled}/{len(recs)} predictions")

        time.sleep(1.0)  # Rate limit between symbols

    # Save
    if not dry_run and filled > 0:
        if is_wrapped:
            data["records"] = records
            data["total_records"] = len(records)
            data["last_updated"] = time.time()
            save_data = data
        else:
            save_data = records
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=None)
        if verbose:
            print(f"\n✓ Saved {filled} outcomes to {predictions_path}")

    return len(pending), filled, too_recent


def print_quick_report(predictions_path: Path) -> None:
    """Print a quick accuracy report from backfilled data."""
    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", data) if isinstance(data, dict) else data
    completed = [r for r in records if r.get("outcome_recorded", False)]
    if not completed:
        print("\n⚠️  No completed predictions yet. Run without --dry-run first.")
        return

    total = len(completed)
    tp50_hits = sum(1 for r in completed if r["actual_peak_price"] >= r["pred_tp50"])
    tp80_hits = sum(1 for r in completed if r["actual_peak_price"] >= r["pred_tp80"])
    tp90_hits = sum(1 for r in completed if r["actual_peak_price"] >= r["pred_tp90"])

    avg_pred_ret = sum(r["pred_ret80"] for r in completed) / total
    avg_actual_ret = sum(r.get("actual_peak_return", 0) for r in completed) / total
    avg_time_to_peak = sum(r.get("time_to_peak_hours", 0) for r in completed) / total

    # Per-symbol breakdown
    by_sym: Dict[str, list] = {}
    for r in completed:
        by_sym.setdefault(r["symbol"], []).append(r)

    print(f"\n{'='*70}")
    print(f"ML MODEL ACCURACY REPORT (Backfilled Outcomes)")
    print(f"{'='*70}")
    print(f"Completed predictions: {total}")
    print(f"\n--- Overall Hit Rates ---")
    print(f"  TP50 hit rate: {tp50_hits/total:.1%}  ({tp50_hits}/{total})")
    print(f"  TP80 hit rate: {tp80_hits/total:.1%}  ({tp80_hits}/{total})")
    print(f"  TP90 hit rate: {tp90_hits/total:.1%}  ({tp90_hits}/{total})")
    print(f"\n--- Return Analysis ---")
    print(f"  Avg predicted return (TP80): {avg_pred_ret:+.2%}")
    print(f"  Avg actual peak return:      {avg_actual_ret:+.2%}")
    print(f"  Overshoot ratio:             {avg_pred_ret / avg_actual_ret:.1f}x" if avg_actual_ret > 0 else "  Overshoot ratio: N/A")
    print(f"  Avg time to peak:            {avg_time_to_peak:.1f} hours")

    # How many predictions are directionally correct (positive return)?
    positive_actual = sum(1 for r in completed if r.get("actual_peak_return", 0) > 0)
    print(f"  Positive peak return:        {positive_actual/total:.1%}")

    print(f"\n--- Calibration Check ---")
    # For properly calibrated quantiles:
    # TP50: ~50% of actuals should be >= prediction
    # TP80: ~20% should be >= (it's the 80th percentile of the upside)
    # TP90: ~10% should be >= (it's the 90th percentile)
    print(f"  TP50 exceedance (target ~50%): {tp50_hits/total:.1%}")
    print(f"  TP80 exceedance (target ~20%): {tp80_hits/total:.1%}")
    print(f"  TP90 exceedance (target ~10%): {tp90_hits/total:.1%}")

    if tp80_hits / total < 0.10:
        print(f"\n  ⚠️  TP80 is severely over-optimistic — model predicts returns")
        print(f"     that are rarely achieved. Consider retraining.")
    elif tp80_hits / total > 0.35:
        print(f"\n  ⚠️  TP80 is too conservative — model undersells the upside.")

    # Top/Bottom symbols
    sym_accuracy = []
    for sym, rlist in by_sym.items():
        n = len(rlist)
        if n < 3:
            continue
        hits50 = sum(1 for r in rlist if r["actual_peak_price"] >= r["pred_tp50"]) / n
        avg_ret = sum(r.get("actual_peak_return", 0) for r in rlist) / n
        avg_pred = sum(r["pred_ret80"] for r in rlist) / n
        sym_accuracy.append((sym, n, hits50, avg_ret, avg_pred))

    sym_accuracy.sort(key=lambda x: x[2], reverse=True)

    if sym_accuracy:
        print(f"\n--- Top 10 Best Predicted Symbols ---")
        print(f"  {'Symbol':<18} {'N':>4}  {'TP50 Hit':>8}  {'Avg Actual':>10}  {'Avg Pred':>10}")
        for sym, n, h50, ar, ap in sym_accuracy[:10]:
            print(f"  {sym:<18} {n:>4}  {h50:>7.0%}  {ar:>+10.2%}  {ap:>+10.2%}")

        print(f"\n--- Bottom 10 Worst Predicted Symbols ---")
        for sym, n, h50, ar, ap in sym_accuracy[-10:]:
            print(f"  {sym:<18} {n:>4}  {h50:>7.0%}  {ar:>+10.2%}  {ap:>+10.2%}")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Backfill ML prediction outcomes from Kraken price data")
    parser.add_argument("--days", type=int, default=None,
                        help="Only backfill predictions from last N days")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be filled without saving")
    parser.add_argument("--report", action="store_true",
                        help="Print accuracy report after backfilling")
    parser.add_argument("--horizon", type=float, default=24.0,
                        help="Hours to look ahead for peak price (default: 24)")
    parser.add_argument("--data-path", type=str,
                        default="data/ml_predictions.json",
                        help="Path to predictions JSON file")
    args = parser.parse_args()

    cfg = load_config()
    exchange = get_exchange(cfg)
    predictions_path = Path(args.data_path)

    if not predictions_path.exists():
        print(f"✗ Predictions file not found: {predictions_path}")
        return

    print(f"{'='*70}")
    print(f"BACKFILLING ML PREDICTION OUTCOMES")
    print(f"{'='*70}")
    print(f"Horizon: {args.horizon} hours")
    print(f"Data: {predictions_path}")
    if args.dry_run:
        print("Mode: DRY RUN (no changes will be saved)")
    print()

    total, filled, too_recent = backfill_outcomes(
        predictions_path=predictions_path,
        exchange=exchange,
        horizon_hours=args.horizon,
        days_filter=args.days,
        dry_run=args.dry_run,
        verbose=True,
    )

    print(f"\n--- Summary ---")
    print(f"  Total pending:       {total}")
    print(f"  Successfully filled: {filled}")
    print(f"  Too recent (wait):   {too_recent}")

    if args.report and not args.dry_run:
        print_quick_report(predictions_path)


if __name__ == "__main__":
    main()
