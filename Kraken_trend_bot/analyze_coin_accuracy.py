#!/usr/bin/env python3
"""
Detailed analysis of which coins hit which prediction targets (TP50, TP80, TP90).
Shows you which cryptos have the most reliable predictions so you can focus on them.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List


def analyze_coin_accuracy(show_details: bool = False):
    """
    Analyze prediction accuracy per coin with detailed breakdown.
    
    Args:
        show_details: If True, show individual trade details per coin
    """
    script_dir = Path(__file__).parent
    predictions_file = script_dir / "data" / "ml_predictions.json"
    
    if not predictions_file.exists():
        print("❌ No predictions file found. Run the bot with ML enabled first.")
        return
    
    try:
        with open(predictions_file, 'r') as f:
            pred_data = json.load(f)
    except Exception as e:
        print(f"❌ Could not load predictions file: {e}")
        return
    
    records = pred_data.get('records', [])
    
    if not records:
        print("❌ No predictions found yet. Wait for the bot to make predictions.")
        return
    
    last_updated = pred_data.get('last_updated', 0)
    if last_updated:
        last_update_time = datetime.fromtimestamp(last_updated).strftime('%Y-%m-%d %H:%M:%S')
    else:
        last_update_time = "Unknown"
    
    # Separate completed vs pending
    completed_records = [r for r in records if r.get('outcome_recorded') and r.get('actual_peak_return') is not None]
    pending_records = [r for r in records if not r.get('outcome_recorded') or r.get('actual_peak_return') is None]
    
    print("=" * 100)
    print("DETAILED COIN ACCURACY ANALYSIS - Which Coins Hit Their Targets?")
    print("=" * 100)
    print(f"Last Updated: {last_update_time}")
    print(f"Total Predictions: {len(records)}")
    print(f"  Completed (with outcomes): {len(completed_records)}")
    print(f"  Pending (waiting for outcomes): {len(pending_records)}")
    print()
    
    if not completed_records:
        print("⚠️  No completed predictions yet. Outcomes are recorded when positions reach peaks.")
        print("   This typically takes 24-48 hours. Keep the bot running and check back later.")
        return
    
    # Group by symbol and analyze
    by_symbol: Dict[str, List[dict]] = {}
    for record in completed_records:
        symbol = record.get('symbol', 'UNKNOWN')
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(record)
    
    # Calculate stats for each symbol
    symbol_analysis = []
    for symbol, symbol_records in by_symbol.items():
        total = len(symbol_records)
        
        tp50_hits = []
        tp80_hits = []
        tp90_hits = []
        trades_taken = 0
        
        for rec in symbol_records:
            entry_price = rec.get('entry_price', 0)
            if entry_price == 0:
                continue
            
            actual_return = rec.get('actual_peak_return', 0)
            pred_tp50 = rec.get('pred_tp50', entry_price)
            pred_tp80 = rec.get('pred_tp80', entry_price)
            pred_tp90 = rec.get('pred_tp90', entry_price)
            
            pred_ret50 = (pred_tp50 / entry_price - 1) if entry_price else 0
            pred_ret80 = rec.get('pred_ret80', 0)
            pred_ret90 = (pred_tp90 / entry_price - 1) if entry_price else 0
            
            # Track which targets were hit
            hit_tp50 = actual_return >= pred_ret50
            hit_tp80 = actual_return >= pred_ret80
            hit_tp90 = actual_return >= pred_ret90
            
            if hit_tp50:
                tp50_hits.append(rec)
            if hit_tp80:
                tp80_hits.append(rec)
            if hit_tp90:
                tp90_hits.append(rec)
            
            if rec.get('trade_taken', False):
                trades_taken += 1
        
        tp50_count = len(tp50_hits)
        tp80_count = len(tp80_hits)
        tp90_count = len(tp90_hits)
        
        tp50_rate = tp50_count / total if total > 0 else 0
        tp80_rate = tp80_count / total if total > 0 else 0
        tp90_rate = tp90_count / total if total > 0 else 0
        
        # Quality score: weighted combination favoring TP80 and TP90
        quality_score = (tp80_rate * 0.5 + tp90_rate * 0.3 + tp50_rate * 0.2) * 100
        
        # Classification
        if tp80_rate >= 0.75 and quality_score >= 75:
            quality = "EXCELLENT"
            emoji = "🌟"
        elif tp80_rate >= 0.65 and quality_score >= 60:
            quality = "GOOD"
            emoji = "✅"
        elif tp80_rate >= 0.50 and quality_score >= 45:
            quality = "FAIR"
            emoji = "⚠️ "
        else:
            quality = "POOR"
            emoji = "❌"
        
        symbol_analysis.append({
            'symbol': symbol,
            'total': total,
            'tp50_count': tp50_count,
            'tp80_count': tp80_count,
            'tp90_count': tp90_count,
            'tp50_rate': tp50_rate,
            'tp80_rate': tp80_rate,
            'tp90_rate': tp90_rate,
            'quality_score': quality_score,
            'quality': quality,
            'emoji': emoji,
            'trades_taken': trades_taken,
            'records': symbol_records
        })
    
    # Sort by quality score
    symbol_analysis.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # Display summary table
    print("=" * 100)
    print(f"{'Rank':<5} {'Symbol':<12} {'Total':>6} {'TP50':>8} {'TP80':>8} {'TP90':>8} {'Quality':>7} {'Rating':>12}")
    print("-" * 100)
    
    for idx, stats in enumerate(symbol_analysis, 1):
        symbol = stats['symbol']
        total = stats['total']
        tp50_str = f"{stats['tp50_count']} ({stats['tp50_rate']:.0%})"
        tp80_str = f"{stats['tp80_count']} ({stats['tp80_rate']:.0%})"
        tp90_str = f"{stats['tp90_count']} ({stats['tp90_rate']:.0%})"
        score = stats['quality_score']
        quality = f"{stats['emoji']} {stats['quality']}"
        
        print(f"{idx:<5} {symbol:<12} {total:>6} {tp50_str:>8} {tp80_str:>8} {tp90_str:>8} {score:>6.1f} {quality:>12}")
    
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    
    # Categorize coins
    excellent = [s for s in symbol_analysis if s['quality'] == 'EXCELLENT']
    good = [s for s in symbol_analysis if s['quality'] == 'GOOD']
    fair = [s for s in symbol_analysis if s['quality'] == 'FAIR']
    poor = [s for s in symbol_analysis if s['quality'] == 'POOR']
    
    print(f"\n🌟 EXCELLENT COINS ({len(excellent)}) - FOCUS HERE!")
    print("   These coins consistently hit TP80+ targets. Prioritize trading these:")
    for s in excellent:
        print(f"   • {s['symbol']:<10} → TP80: {s['tp80_rate']:.0%}, TP90: {s['tp90_rate']:.0%} ({s['total']} predictions)")
    
    if good:
        print(f"\n✅ GOOD COINS ({len(good)}) - Reliable")
        print("   Solid prediction accuracy. Safe to trade:")
        for s in good[:10]:
            print(f"   • {s['symbol']:<10} → TP80: {s['tp80_rate']:.0%}, TP90: {s['tp90_rate']:.0%} ({s['total']} predictions)")
    
    if fair:
        print(f"\n⚠️  FAIR COINS ({len(fair)}) - Use Caution")
        print("   Moderate accuracy. Consider smaller positions:")
        for s in fair[:5]:
            print(f"   • {s['symbol']:<10} → TP80: {s['tp80_rate']:.0%}, TP90: {s['tp90_rate']:.0%} ({s['total']} predictions)")
    
    if poor:
        print(f"\n❌ POOR COINS ({len(poor)}) - AVOID or Need More Data")
        print("   Low accuracy. Avoid trading or wait for more data:")
        for s in poor[:5]:
            print(f"   • {s['symbol']:<10} → TP80: {s['tp80_rate']:.0%}, TP90: {s['tp90_rate']:.0%} ({s['total']} predictions)")
    
    # Show detailed records if requested
    if show_details and excellent:
        print("\n" + "=" * 100)
        print("DETAILED BREAKDOWN - TOP PERFORMING COINS")
        print("=" * 100)
        
        for stats in excellent[:3]:  # Show top 3
            symbol = stats['symbol']
            records = stats['records']
            
            print(f"\n{stats['emoji']} {symbol} - {stats['total']} Predictions")
            print("-" * 80)
            print(f"{'Date':<12} {'Entry $':>10} {'Pred TP80':>10} {'Pred TP90':>10} {'Actual':>10} {'Hit':>8}")
            print("-" * 80)
            
            for rec in records[-10:]:  # Last 10
                ts = rec.get('timestamp', 0)
                date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else 'N/A'
                entry = rec.get('entry_price', 0)
                pred_ret80 = rec.get('pred_ret80', 0)
                actual_ret = rec.get('actual_peak_return', 0)
                
                pred_tp80_price = entry * (1 + pred_ret80)
                pred_tp90_price = rec.get('pred_tp90', 0)
                
                # Determine what was hit
                if actual_ret >= (pred_tp90_price / entry - 1):
                    hit = "✓✓ TP90"
                elif actual_ret >= pred_ret80:
                    hit = "✓ TP80"
                elif actual_ret >= (rec.get('pred_tp50', entry) / entry - 1):
                    hit = "✓ TP50"
                else:
                    hit = "✗ Miss"
                
                actual_str = f"{actual_ret:+.2%}"
                
                print(f"{date_str:<12} ${entry:>9.2f} ${pred_tp80_price:>9.2f} ${pred_tp90_price:>9.2f} {actual_str:>10} {hit:>8}")
    
    print("\n" + "=" * 100)
    print("STRATEGY TIPS:")
    print("=" * 100)
    print("1. Focus your capital on EXCELLENT coins for highest success rate")
    print("2. Use larger position sizes for coins with TP80 > 80%")
    print("3. Avoid or minimize exposure to POOR performing coins")
    print("4. Coins need at least 5-10 predictions for reliable statistics")
    print("5. Review this report weekly to track which coins are performing well")
    print("=" * 100)


if __name__ == "__main__":
    import sys
    
    # Check for --details flag
    show_details = '--details' in sys.argv or '-d' in sys.argv
    
    analyze_coin_accuracy(show_details=show_details)
