#!/usr/bin/env python3
"""
ML Performance Analyzer and Visualizer

Analyzes ML prediction history and generates visualizations and detailed reports.
Run this periodically to understand your ML model's performance.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import statistics


def load_ml_predictions(data_path: Path) -> List[Dict]:
    """Load ML prediction records from JSON file."""
    if not data_path.exists():
        print(f"No prediction data found at {data_path}")
        return []
    
    data = json.loads(data_path.read_text(encoding="utf-8"))
    return data.get("records", [])


def analyze_by_symbol(records: List[Dict]) -> Dict[str, Dict]:
    """Analyze performance per symbol."""
    completed = [r for r in records if r.get("outcome_recorded")]
    
    symbol_stats = {}
    
    for record in completed:
        symbol = record["symbol"]
        if symbol not in symbol_stats:
            symbol_stats[symbol] = {
                "count": 0,
                "tp80_hits": 0,
                "correct_direction": 0,
                "avg_pred_return": [],
                "avg_actual_return": []
            }
        
        stats = symbol_stats[symbol]
        stats["count"] += 1
        
        if record.get("actual_peak_price") and record.get("pred_tp80"):
            if record["actual_peak_price"] >= record["pred_tp80"]:
                stats["tp80_hits"] += 1
        
        if record.get("actual_peak_return") and record.get("pred_ret80"):
            if record["pred_ret80"] > 0 and record["actual_peak_return"] > 0:
                stats["correct_direction"] += 1
            stats["avg_pred_return"].append(record["pred_ret80"])
            stats["avg_actual_return"].append(record["actual_peak_return"])
    
    # Calculate averages
    for symbol, stats in symbol_stats.items():
        if stats["count"] > 0:
            stats["tp80_hit_rate"] = stats["tp80_hits"] / stats["count"]
            stats["directional_accuracy"] = stats["correct_direction"] / stats["count"]
            stats["avg_pred_return"] = statistics.mean(stats["avg_pred_return"]) if stats["avg_pred_return"] else 0.0
            stats["avg_actual_return"] = statistics.mean(stats["avg_actual_return"]) if stats["avg_actual_return"] else 0.0
    
    return symbol_stats


def analyze_by_time_of_day(records: List[Dict]) -> Dict[int, Dict]:
    """Analyze performance by hour of day."""
    completed = [r for r in records if r.get("outcome_recorded")]
    
    hour_stats = {hour: {"count": 0, "tp80_hits": 0} for hour in range(24)}
    
    for record in completed:
        ts = record.get("timestamp")
        if not ts:
            continue
        
        dt = datetime.fromtimestamp(ts)
        hour = dt.hour
        
        hour_stats[hour]["count"] += 1
        
        if record.get("actual_peak_price") and record.get("pred_tp80"):
            if record["actual_peak_price"] >= record["pred_tp80"]:
                hour_stats[hour]["tp80_hits"] += 1
    
    # Calculate hit rates
    for hour in hour_stats:
        if hour_stats[hour]["count"] > 0:
            hour_stats[hour]["hit_rate"] = hour_stats[hour]["tp80_hits"] / hour_stats[hour]["count"]
        else:
            hour_stats[hour]["hit_rate"] = 0.0
    
    return hour_stats


def analyze_performance_over_time(records: List[Dict], window_days: int = 7) -> List[Tuple[str, float]]:
    """Analyze performance trends over time."""
    completed = [r for r in records if r.get("outcome_recorded")]
    completed.sort(key=lambda r: r["timestamp"])
    
    if not completed:
        return []
    
    window_seconds = window_days * 24 * 3600
    trends = []
    
    # Create rolling windows
    for i, record in enumerate(completed):
        window_start = record["timestamp"]
        window_end = window_start + window_seconds
        
        # Get all records in this window
        window_records = [
            r for r in completed
            if window_start <= r["timestamp"] < window_end
        ]
        
        if len(window_records) < 5:  # Need minimum samples
            continue
        
        # Calculate hit rate for this window
        hits = sum(
            1 for r in window_records
            if r.get("actual_peak_price") and r.get("pred_tp80") and r["actual_peak_price"] >= r["pred_tp80"]
        )
        hit_rate = hits / len(window_records)
        
        date_str = datetime.fromtimestamp(window_start).strftime("%Y-%m-%d")
        trends.append((date_str, hit_rate))
        
        # Only sample every N records to avoid too many data points
        if i % 10 != 0:
            continue
    
    return trends


def generate_detailed_report(data_path: Path) -> str:
    """Generate comprehensive ML performance report."""
    records = load_ml_predictions(data_path)
    
    if not records:
        return "\nNo ML prediction data available yet.\n"
    
    completed = [r for r in records if r.get("outcome_recorded")]
    
    report = "\n" + "="*80 + "\n"
    report += "ML MODEL DETAILED PERFORMANCE ANALYSIS\n"
    report += "="*80 + "\n\n"
    
    # Overall statistics
    report += f"Dataset Overview:\n"
    report += f"  Total Predictions:        {len(records)}\n"
    report += f"  Completed (with outcomes): {len(completed)}\n"
    report += f"  Pending (no outcome yet):  {len(records) - len(completed)}\n"
    report += f"  Coverage:                  {len(completed)/len(records)*100:.1f}%\n\n"
    
    if not completed:
        report += "No completed predictions yet. Keep trading to build history!\n"
        report += "="*80 + "\n"
        return report
    
    # Calculate overall metrics
    tp80_hits = sum(
        1 for r in completed
        if r.get("actual_peak_price") and r.get("pred_tp80") and r["actual_peak_price"] >= r["pred_tp80"]
    )
    tp80_rate = tp80_hits / len(completed)
    
    tp90_hits = sum(
        1 for r in completed
        if r.get("actual_peak_price") and r.get("pred_tp90") and r["actual_peak_price"] >= r["pred_tp90"]
    )
    tp90_rate = tp90_hits / len(completed)
    
    dir_correct = sum(
        1 for r in completed
        if r.get("actual_peak_return") and r.get("pred_ret80")
        and r["pred_ret80"] > 0 and r["actual_peak_return"] > 0
    )
    dir_rate = dir_correct / len(completed)
    
    report += f"Overall Accuracy:\n"
    report += f"  TP80 Hit Rate:            {tp80_rate*100:.1f}% ({tp80_hits}/{len(completed)})\n"
    report += f"  TP90 Hit Rate:            {tp90_rate*100:.1f}% ({tp90_hits}/{len(completed)})\n"
    report += f"  Directional Accuracy:     {dir_rate*100:.1f}% ({dir_correct}/{len(completed)})\n\n"
    
    # Performance by symbol
    report += "="*80 + "\n"
    report += "PERFORMANCE BY SYMBOL (Top 10)\n"
    report += "="*80 + "\n\n"
    
    symbol_stats = analyze_by_symbol(records)
    sorted_symbols = sorted(
        symbol_stats.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )[:10]
    
    report += f"{'Symbol':<12} {'Count':<8} {'TP80 Rate':<12} {'Dir Acc':<12} {'Avg Pred':<12} {'Avg Actual':<12}\n"
    report += "-"*80 + "\n"
    
    for symbol, stats in sorted_symbols:
        if stats["count"] < 3:  # Skip symbols with too few predictions
            continue
        report += f"{symbol:<12} {stats['count']:<8} "
        report += f"{stats['tp80_hit_rate']*100:>6.1f}%      "
        report += f"{stats['directional_accuracy']*100:>6.1f}%      "
        report += f"{stats['avg_pred_return']*100:>+6.2f}%     "
        report += f"{stats['avg_actual_return']*100:>+6.2f}%\n"
    
    report += "\n"
    
    # Performance by time of day
    report += "="*80 + "\n"
    report += "PERFORMANCE BY HOUR OF DAY\n"
    report += "="*80 + "\n\n"
    
    hour_stats = analyze_by_time_of_day(records)
    
    # Group by 4-hour blocks for cleaner display
    blocks = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
    
    for start, end in blocks:
        block_count = sum(hour_stats[h]["count"] for h in range(start, end))
        block_hits = sum(hour_stats[h]["tp80_hits"] for h in range(start, end))
        
        if block_count > 0:
            block_rate = block_hits / block_count
            report += f"  {start:02d}:00 - {end:02d}:00    Predictions: {block_count:<4}  Hit Rate: {block_rate*100:>6.1f}%\n"
    
    report += "\n"
    
    # Recent vs. historical performance
    report += "="*80 + "\n"
    report += "PERFORMANCE TRENDS\n"
    report += "="*80 + "\n\n"
    
    now = datetime.now().timestamp()
    
    # Last 7 days
    recent_7d = [r for r in completed if (now - r["timestamp"]) <= (7 * 24 * 3600)]
    if recent_7d:
        hits = sum(
            1 for r in recent_7d
            if r.get("actual_peak_price") and r.get("pred_tp80") and r["actual_peak_price"] >= r["pred_tp80"]
        )
        rate = hits / len(recent_7d)
        report += f"  Last 7 Days:     {len(recent_7d):<4} predictions, {rate*100:>6.1f}% hit rate\n"
    
    # Last 30 days
    recent_30d = [r for r in completed if (now - r["timestamp"]) <= (30 * 24 * 3600)]
    if recent_30d:
        hits = sum(
            1 for r in recent_30d
            if r.get("actual_peak_price") and r.get("pred_tp80") and r["actual_peak_price"] >= r["pred_tp80"]
        )
        rate = hits / len(recent_30d)
        report += f"  Last 30 Days:    {len(recent_30d):<4} predictions, {rate*100:>6.1f}% hit rate\n"
    
    # Older than 30 days
    older = [r for r in completed if (now - r["timestamp"]) > (30 * 24 * 3600)]
    if older:
        hits = sum(
            1 for r in older
            if r.get("actual_peak_price") and r.get("pred_tp80") and r["actual_peak_price"] >= r["pred_tp80"]
        )
        rate = hits / len(older)
        report += f"  Older (>30d):    {len(older):<4} predictions, {rate*100:>6.1f}% hit rate\n"
    
    report += "\n"
    
    # Recommendations
    report += "="*80 + "\n"
    report += "RECOMMENDATIONS\n"
    report += "="*80 + "\n\n"
    
    if tp80_rate < 0.60:
        report += "  ❌ CRITICAL: Model performance is poor (TP80 < 60%)\n"
        report += "     Action: Retrain model with more recent data immediately\n\n"
    elif tp80_rate < 0.70:
        report += "  ⚠️  WARNING: Model performance is below target (TP80 < 70%)\n"
        report += "     Action: Consider retraining with updated data\n\n"
    else:
        report += "  ✅ Model performance is acceptable (TP80 ≥ 70%)\n\n"
    
    if dir_rate < 0.55:
        report += "  ⚠️  Directional accuracy is poor (< 55%)\n"
        report += "     This suggests the model may not capture market direction well\n\n"
    
    # Check if performance is degrading
    if recent_7d and recent_30d and len(recent_7d) >= 10 and len(recent_30d) >= 20:
        recent_7d_rate = sum(
            1 for r in recent_7d
            if r.get("actual_peak_price") and r.get("pred_tp80") and r["actual_peak_price"] >= r["pred_tp80"]
        ) / len(recent_7d)
        
        recent_30d_rate = sum(
            1 for r in recent_30d
            if r.get("actual_peak_price") and r.get("pred_tp80") and r["actual_peak_price"] >= r["pred_tp80"]
        ) / len(recent_30d)
        
        if recent_7d_rate < recent_30d_rate - 0.15:  # 15% drop
            report += "  ⚠️  Performance degrading: Recent accuracy significantly worse\n"
            report += f"     7-day: {recent_7d_rate*100:.1f}%  vs  30-day: {recent_30d_rate*100:.1f}%\n"
            report += "     Action: Retrain model soon\n\n"
    
    report += "="*80 + "\n"
    
    return report


def main():
    """Main entry point for ML performance analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ML model performance")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ml_predictions.json",
        help="Path to ML predictions JSON file"
    )
    
    args = parser.parse_args()
    data_path = Path(args.data_path)
    
    report = generate_detailed_report(data_path)
    print(report)
    
    # Save report to file
    report_path = data_path.parent / f"ml_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
