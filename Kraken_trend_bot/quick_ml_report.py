#!/usr/bin/env python3
"""
Quick ML Performance Report Generator

Run this anytime to generate a ML performance report without running the full bot.
Much faster than running analyze_ml_performance.py for quick checks.

Usage:
    python quick_ml_report.py                    # Last 30 days
    python quick_ml_report.py --days 7           # Last 7 days
    python quick_ml_report.py --full             # All time
"""

import sys
from pathlib import Path
from datetime import datetime

# Add bot directory to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from bot.ml_performance_tracker import MLPerformanceTracker


def print_quick_report(tracker: MLPerformanceTracker, days: int = None):
    """Print a quick performance report."""
    
    # Get metrics
    metrics = tracker.get_metrics(days_lookback=days)
    
    print("="*80)
    print("ML MODEL PERFORMANCE REPORT")
    print("="*80)
    print()
    
    # Summary
    timeframe = f"Last {days} days" if days else "All time"
    print(f"Timeframe: {timeframe}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Data status
    print(f"Total Predictions Logged:  {metrics.total_predictions}")
    print(f"Predictions with Outcomes: {metrics.predictions_with_outcomes}")
    print()
    
    if metrics.predictions_with_outcomes == 0:
        print("⚠️  No completed predictions yet!")
        print()
        print("Why? Outcomes are recorded when:")
        print("  • A position reaches its peak price")
        print("  • The bot updates positions (every 60s by default)")
        print("  • Typically takes 24-48 hours for first results")
        print()
        print("The bot IS logging predictions (you have {:,} logged).".format(metrics.total_predictions))
        print("Keep the bot running to see outcomes populate!")
        print("="*80)
        return
    
    # Performance metrics
    print("ACCURACY METRICS")
    print("-"*80)
    print(f"TP80 Hit Rate:          {metrics.tp80_hit_rate*100:>6.1f}%  {'✅' if metrics.tp80_hit_rate >= 0.70 else '⚠️' if metrics.tp80_hit_rate >= 0.60 else '❌'}")
    print(f"TP90 Hit Rate:          {metrics.tp90_hit_rate*100:>6.1f}%")
    print(f"Directional Accuracy:   {metrics.directional_accuracy*100:>6.1f}%  {'✅' if metrics.directional_accuracy >= 0.55 else '⚠️'}")
    print()
    
    # Error metrics
    print("ERROR METRICS")
    print("-"*80)
    print(f"Mean Absolute Error:    {metrics.mae_tp80:.6f}")
    print(f"Mean Abs % Error:       {metrics.mape_tp80:>6.2f}%")
    print()
    
    # Returns
    print("RETURNS")
    print("-"*80)
    print(f"Avg Predicted Return:   {metrics.avg_predicted_return*100:>+6.2f}%")
    print(f"Avg Actual Return:      {metrics.avg_actual_return*100:>+6.2f}%")
    print(f"Avg Time to Peak:       {metrics.avg_time_to_peak_hours:>6.1f} hours")
    print()
    
    # Recent performance
    if days is None or days >= 7:
        print("RECENT PERFORMANCE (Last 7 Days)")
        print("-"*80)
        print(f"Recent TP80 Hit Rate:   {metrics.recent_tp80_hit_rate*100:>6.1f}%")
        print(f"Recent Dir Accuracy:    {metrics.recent_directional_accuracy*100:>6.1f}%")
        print()
    
    # Recommendations
    print("RECOMMENDATIONS")
    print("-"*80)
    
    if metrics.tp80_hit_rate < 0.60:
        print("❌ CRITICAL: Model performance is poor (TP80 < 60%)")
        print("   → Retrain model immediately with recent data")
    elif metrics.tp80_hit_rate < 0.70:
        print("⚠️  WARNING: Model performance below target (TP80 < 70%)")
        print("   → Consider retraining with updated data")
    else:
        print("✅ Model performance is good (TP80 ≥ 70%)")
    
    if metrics.directional_accuracy < 0.55:
        print("⚠️  Directional accuracy is weak (< 55%)")
        print("   → Model may not capture market direction well")
    
    # Check retraining recommendation
    should_retrain, reason = tracker.should_retrain(min_samples=50)
    if should_retrain:
        print(f"⚠️  {reason}")
    
    print()
    print("="*80)
    
    # Check for detailed report
    print()
    print("For more detailed analysis, run:")
    print("  python analyze_ml_performance.py")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate quick ML performance report",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days to look back (default: 30)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show all-time statistics"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ml_predictions.json",
        help="Path to ML predictions file"
    )
    
    args = parser.parse_args()
    
    data_path = REPO_ROOT / args.data_path
    
    if not data_path.exists():
        print(f"❌ Error: ML predictions file not found: {data_path}")
        print()
        print("Make sure:")
        print("  1. The bot has been running with ML enabled")
        print("  2. The path is correct")
        sys.exit(1)
    
    # Load tracker
    tracker = MLPerformanceTracker(data_path)
    
    # Generate report
    days = None if args.full else args.days
    print_quick_report(tracker, days=days)


if __name__ == "__main__":
    main()
