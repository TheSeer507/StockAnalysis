#!/usr/bin/env python3
"""
Track live ML predictions vs actual outcomes.
Run this periodically to check how your model performs in real trading.
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


def analyze_by_coin():
    """Analyze which coins have the best prediction accuracy."""
    
    script_dir = Path(__file__).parent
    predictions_file = script_dir / "data" / "ml_predictions.json"
    
    if not predictions_file.exists():
        print("No predictions file found. The bot will create it when ML is enabled.")
        return
    
    try:
        with open(predictions_file, 'r') as f:
            pred_data = json.load(f)
    except Exception as e:
        print(f"Could not load predictions file: {e}")
        return
    
    records = pred_data.get('records', [])
    
    if not records:
        print("No predictions found yet.")
        return
    
    # Group by symbol
    by_symbol = {}
    for record in records:
        if not record.get('outcome_recorded') or record.get('actual_peak_return') is None:
            continue
        
        symbol = record.get('symbol', '')
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(record)
    
    if not by_symbol:
        print("No completed predictions yet. Wait for outcomes to be recorded.")
        return
    
    # Calculate per-symbol stats
    symbol_stats = []
    for symbol, symbol_records in by_symbol.items():
        total = len(symbol_records)
        if total < 3:  # Need at least 3 predictions
            continue
        
        tp50_hits = 0
        tp80_hits = 0
        tp90_hits = 0
        
        for rec in symbol_records:
            entry_price = rec.get('entry_price', 0)
            if entry_price == 0:
                continue
            
            actual_return = rec.get('actual_peak_return', 0)
            pred_tp50 = rec.get('pred_tp50', entry_price)
            pred_tp80 = rec.get('pred_tp80', entry_price)
            pred_tp90 = rec.get('pred_tp90', entry_price)
            
            pred_ret50 = (pred_tp50 / entry_price - 1)
            pred_ret80 = rec.get('pred_ret80', 0)
            pred_ret90 = (pred_tp90 / entry_price - 1)
            
            if actual_return >= pred_ret50:
                tp50_hits += 1
            if actual_return >= pred_ret80:
                tp80_hits += 1
            if actual_return >= pred_ret90:
                tp90_hits += 1
        
        tp50_rate = tp50_hits / total
        tp80_rate = tp80_hits / total
        tp90_rate = tp90_hits / total
        
        # Accuracy score: weighted combination
        accuracy_score = tp80_rate * 0.5 + tp90_rate * 0.3 + tp50_rate * 0.2
        
        symbol_stats.append({
            'symbol': symbol,
            'total': total,
            'tp50_hits': tp50_hits,
            'tp80_hits': tp80_hits,
            'tp90_hits': tp90_hits,
            'tp50_rate': tp50_rate,
            'tp80_rate': tp80_rate,
            'tp90_rate': tp90_rate,
            'accuracy_score': accuracy_score
        })
    
    # Sort by accuracy score
    symbol_stats.sort(key=lambda x: x['accuracy_score'], reverse=True)
    
    print("=" * 95)
    print("PREDICTION ACCURACY BY COIN/CRYPTO")
    print("=" * 95)
    print(f"{'Symbol':<12} {'Total':>6} {'TP50':>6} {'TP80':>6} {'TP90':>6} {'TP50%':>7} {'TP80%':>7} {'TP90%':>7} {'Score':>7}")
    print("-" * 95)
    
    for stats in symbol_stats:
        symbol = stats['symbol']
        total = stats['total']
        tp50_hits = stats['tp50_hits']
        tp80_hits = stats['tp80_hits']
        tp90_hits = stats['tp90_hits']
        tp50_rate = stats['tp50_rate']
        tp80_rate = stats['tp80_rate']
        tp90_rate = stats['tp90_rate']
        score = stats['accuracy_score']
        
        print(f"{symbol:<12} {total:>6} {tp50_hits:>6} {tp80_hits:>6} {tp90_hits:>6} "
              f"{tp50_rate:>6.1%} {tp80_rate:>6.1%} {tp90_rate:>6.1%} {score:>7.3f}")
    
    print("\n" + "=" * 95)
    print("INTERPRETATION:")
    print("=" * 95)
    print("Score: Weighted accuracy (TP80=50%, TP90=30%, TP50=20%)")
    print("✅ Best coins: Score > 0.75 and TP80 > 75%")
    print("⚠️  Acceptable: Score 0.60-0.75")
    print("❌ Poor: Score < 0.60")
    print()
    
    # Recommend best coins
    best_coins = [s for s in symbol_stats if s['accuracy_score'] > 0.75 and s['tp80_rate'] > 0.75]
    acceptable = [s for s in symbol_stats if 0.60 <= s['accuracy_score'] <= 0.75]
    poor = [s for s in symbol_stats if s['accuracy_score'] < 0.60]
    
    if best_coins:
        print(f"✅ BEST PERFORMING COINS ({len(best_coins)}):")
        print("   Focus your trading on these coins with high prediction accuracy:")
        for s in best_coins[:10]:
            print(f"   • {s['symbol']:<10} (Score: {s['accuracy_score']:.3f}, TP80: {s['tp80_rate']:.1%})")
    
    if acceptable:
        print(f"\n⚠️  ACCEPTABLE COINS ({len(acceptable)}):")
        print("   These coins have moderate prediction accuracy:")
        for s in acceptable[:5]:
            print(f"   • {s['symbol']:<10} (Score: {s['accuracy_score']:.3f}, TP80: {s['tp80_rate']:.1%})")
    
    if poor:
        print(f"\n❌ POOR PERFORMING COINS ({len(poor)}):")
        print("   Consider avoiding these coins or improving their prediction models:")
        for s in poor[:5]:
            print(f"   • {s['symbol']:<10} (Score: {s['accuracy_score']:.3f}, TP80: {s['tp80_rate']:.1%})")
    
    print("\n" + "=" * 95)


def analyze_live_predictions():
    """Analyze predictions from ml_predictions.json against actual trade outcomes."""
    
    # Get the script's directory and construct absolute paths
    script_dir = Path(__file__).parent
    predictions_file = script_dir / "data" / "ml_predictions.json"
    trades_file = script_dir / "data" / "trades.csv"
    
    if not predictions_file.exists():
        print("No predictions file found. The bot will create it when ML is enabled.")
        return
    
    # Load predictions
    try:
        with open(predictions_file, 'r') as f:
            pred_data = json.load(f)
    except Exception as e:
        print(f"Could not load predictions file: {e}")
        return
    
    # Handle new format with 'records' list
    records = pred_data.get('records', [])
    
    if not records:
        print("No predictions found yet. Wait for the bot to run with ML enabled.")
        return
    
    last_updated = pred_data.get('last_updated', 0)
    total_records = pred_data.get('total_records', len(records))
    
    # Convert timestamp to datetime for display
    from datetime import datetime
    if last_updated:
        last_update_time = datetime.fromtimestamp(last_updated).strftime('%Y-%m-%d %H:%M:%S')
    else:
        last_update_time = "Unknown"
    
    # Load trades if available
    trades_df = None
    if trades_file.exists():
        try:
            trades_df = pd.read_csv(trades_file)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        except:
            pass
    
    print("=" * 80)
    print("LIVE PREDICTION TRACKING")
    print("=" * 80)
    print(f"Last updated: {last_update_time}")
    print(f"Total prediction records: {total_records:,}\n")
    
    # Analyze predictions with recorded outcomes
    results = []
    for record in records:
        symbol = record.get('symbol', '')
        timestamp = record.get('timestamp', 0)
        entry_price = record.get('entry_price', 0)
        
        # Get predictions (as returns)
        pred_ret50 = (record.get('pred_tp50', entry_price) / entry_price - 1) if entry_price else 0
        pred_ret80 = record.get('pred_ret80', 0)
        pred_ret90 = (record.get('pred_tp90', entry_price) / entry_price - 1) if entry_price else 0
        
        # Get actual outcome
        actual_return = record.get('actual_peak_return')
        outcome_recorded = record.get('outcome_recorded', False)
        trade_taken = record.get('trade_taken', False)
        
        results.append({
            'symbol': symbol,
            'timestamp': timestamp,
            'pred_ret50': pred_ret50,
            'pred_ret80': pred_ret80,
            'pred_ret90': pred_ret90,
            'actual_return': actual_return,
            'outcome_recorded': outcome_recorded,
            'trade_taken': trade_taken,
            'entry_price': entry_price
        })
    
    # Sort by timestamp (most recent first)
    results.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Separate results with outcomes from pending
    with_outcomes = [r for r in results if r['outcome_recorded'] and r['actual_return'] is not None]
    pending = [r for r in results if not r['outcome_recorded'] or r['actual_return'] is None]
    
    # Calculate statistics
    hit_tp50 = 0
    hit_tp80 = 0
    hit_tp90 = 0
    total_with_outcomes = len(with_outcomes)
    
    # Analyze outcomes
    for r in with_outcomes:
        actual = r['actual_return']
        if actual >= r['pred_ret50']:
            hit_tp50 += 1
        if actual >= r['pred_ret80']:
            hit_tp80 += 1
        if actual >= r['pred_ret90']:
            hit_tp90 += 1
    
    # Display section for predictions WITH outcomes
    if with_outcomes:
        print(f"\nPREDICTIONS WITH RECORDED OUTCOMES ({len(with_outcomes)} total)")
        print("=" * 85)
        print(f"{'Symbol':<12} {'TP50':>8} {'TP80':>8} {'TP90':>8} {'Actual':>8} {'Result':>10} {'Traded':>8}")
        print("-" * 85)
        
        for r in with_outcomes[:20]:  # Show first 20 with outcomes
            symbol = r['symbol']
            pred_ret50 = r['pred_ret50']
            pred_ret80 = r['pred_ret80']
            pred_ret90 = r['pred_ret90']
            actual = r['actual_return']
            trade_taken = r['trade_taken']
            
            traded_str = "✓" if trade_taken else ""
            
            # Determine result
            if actual >= pred_ret90:
                hit_str = "✓✓ TP90"
            elif actual >= pred_ret80:
                hit_str = "✓ TP80"
            elif actual >= pred_ret50:
                hit_str = "✓ TP50"
            else:
                hit_str = "✗ MISS"
            
            print(f"{symbol:<12} {pred_ret50:>7.2%} {pred_ret80:>7.2%} {pred_ret90:>7.2%} {actual:>7.2%} {hit_str:>10} {traded_str:>8}")
    
    # Display section for pending predictions
    if pending:
        print(f"\n\nRECENT PENDING PREDICTIONS ({len(pending)} total, showing latest 15)")
        print("=" * 85)
        print(f"{'Symbol':<12} {'TP50':>8} {'TP80':>8} {'TP90':>8} {'Status':>8}")
        print("-" * 85)
        
        for r in pending[:15]:
            symbol = r['symbol']
            pred_ret50 = r['pred_ret50']
            pred_ret80 = r['pred_ret80']
            pred_ret90 = r['pred_ret90']
            
            print(f"{symbol:<12} {pred_ret50:>7.2%} {pred_ret80:>7.2%} {pred_ret90:>7.2%} {'pending':>8}")
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    if total_with_outcomes > 0:
        tp50_rate = hit_tp50 / total_with_outcomes
        tp80_rate = hit_tp80 / total_with_outcomes
        tp90_rate = hit_tp90 / total_with_outcomes
        
        print(f"Completed trades: {total_with_outcomes}")
        print(f"TP50 Hit Rate: {tp50_rate:.1%} (target: ~50%)")
        print(f"TP80 Hit Rate: {tp80_rate:.1%} (target: ~80%)")
        print(f"TP90 Hit Rate: {tp90_rate:.1%} (target: ~90%)")
        
        print("\nCalibration Assessment:")
        if 0.45 <= tp50_rate <= 0.55:
            print("  ✓ TP50: Well calibrated")
        elif tp50_rate > 0.55:
            print(f"  ✗ TP50: Too conservative (predicting {tp50_rate:.1%} > 50%)")
        else:
            print(f"  ✗ TP50: Too aggressive (predicting {tp50_rate:.1%} < 50%)")
            
        if 0.75 <= tp80_rate <= 0.85:
            print("  ✓ TP80: Well calibrated")
        elif tp80_rate > 0.85:
            print(f"  ✗ TP80: Too conservative (predicting {tp80_rate:.1%} > 80%)")
        else:
            print(f"  ✗ TP80: Too aggressive (predicting {tp80_rate:.1%} < 80%)")
            
        if 0.85 <= tp90_rate <= 0.95:
            print("  ✓ TP90: Well calibrated")
        elif tp90_rate > 0.95:
            print(f"  ✗ TP90: Too conservative (predicting {tp90_rate:.1%} > 90%)")
        else:
            print(f"  ✗ TP90: Too aggressive (predicting {tp90_rate:.1%} < 90%)")
    else:
        print("No completed trades found yet. Keep the bot running to collect data.")
    
    print("\n" + "=" * 80)
    print("NOTES:")
    print("- 'Hit Rate' = % of trades where actual return >= predicted level")
    print("- Well-calibrated TP80 should have ~80% hit rate")
    print("- Run this script periodically to track model performance over time")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "="*95)
    print("ML PREDICTION ANALYSIS")
    print("="*95)
    print()
    
    # First show per-coin accuracy
    analyze_by_coin()
    
    print("\n\n")
    
    # Then show overall predictions
    analyze_live_predictions()
