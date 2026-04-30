# bot/ml_performance_tracker.py
"""
ML Performance Tracking System

Tracks ML predictions vs actual outcomes to measure model accuracy and performance over time.
Logs predictions, actual results, and calculates various performance metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics


@dataclass
class MLPredictionRecord:
    """Record of a single ML prediction and its eventual outcome."""
    timestamp: float
    symbol: str
    entry_price: float
    
    # Predictions
    pred_tp50: float
    pred_tp80: float
    pred_tp90: float
    pred_ret80: float
    
    # Actual outcomes (filled in later)
    actual_peak_price: Optional[float] = None
    actual_peak_return: Optional[float] = None
    actual_realized_return: Optional[float] = None
    time_to_peak_hours: Optional[float] = None
    
    # Status
    outcome_recorded: bool = False
    trade_taken: bool = False
    exit_reason: Optional[str] = None


@dataclass
class MLMetrics:
    """Aggregated ML performance metrics."""
    total_predictions: int
    predictions_with_outcomes: int
    
    # Accuracy metrics
    tp80_hit_rate: float  # % of times actual peak >= pred_tp80
    tp90_hit_rate: float
    directional_accuracy: float  # % correct price direction
    
    # Error metrics
    mae_tp80: float  # Mean absolute error for TP80 predictions
    rmse_tp80: float  # Root mean squared error
    mape_tp80: float  # Mean absolute percentage error
    
    # Calibration (how well quantiles match reality)
    tp80_coverage: float  # should be ~80%
    tp90_coverage: float  # should be ~90%
    
    # Performance statistics
    avg_predicted_return: float
    avg_actual_return: float
    avg_time_to_peak_hours: float
    
    # Recent performance (last 7 days)
    recent_tp80_hit_rate: float
    recent_directional_accuracy: float


class MLPerformanceTracker:
    """
    Tracks ML predictions and their outcomes to measure model performance.
    
    Usage:
        1. Call log_prediction() when making a prediction
        2. Call update_outcome() when actual results are known
        3. Call get_metrics() to get performance statistics
        4. Call generate_report() for detailed analysis
    """
    
    def __init__(self, data_path: Path, max_records: int = 10000):
        self.data_path = Path(data_path)
        self.max_records = max_records
        self.records: List[MLPredictionRecord] = []
        self._load()
    
    def _load(self) -> None:
        """Load prediction records from disk."""
        if not self.data_path.exists():
            return
        
        try:
            data = json.loads(self.data_path.read_text(encoding="utf-8"))
            records_data = data.get("records", [])
            
            self.records = []
            for r in records_data:
                self.records.append(MLPredictionRecord(**r))
            
            print(f"[ML-TRACKER] Loaded {len(self.records)} prediction records")
        except Exception as e:
            print(f"[ML-TRACKER][WARN] Failed to load records: {e}")
            self.records = []
    
    def _save(self) -> None:
        """Save prediction records to disk."""
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Keep only max_records most recent
            if len(self.records) > self.max_records:
                self.records = sorted(self.records, key=lambda r: r.timestamp)[-self.max_records:]
            
            data = {
                "last_updated": time.time(),
                "total_records": len(self.records),
                "records": [asdict(r) for r in self.records]
            }
            
            self.data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[ML-TRACKER][WARN] Failed to save records: {e}")
    
    def log_prediction(
        self,
        symbol: str,
        entry_price: float,
        pred_tp50: float,
        pred_tp80: float,
        pred_tp90: float,
        pred_ret80: float,
        trade_taken: bool = False
    ) -> str:
        """
        Log a new ML prediction.
        
        Returns: Prediction ID (timestamp_symbol) for later updating
        """
        record = MLPredictionRecord(
            timestamp=time.time(),
            symbol=symbol,
            entry_price=entry_price,
            pred_tp50=pred_tp50,
            pred_tp80=pred_tp80,
            pred_tp90=pred_tp90,
            pred_ret80=pred_ret80,
            trade_taken=trade_taken
        )
        
        self.records.append(record)
        self._save()
        
        return f"{record.timestamp}_{symbol}"
    
    def update_outcome(
        self,
        symbol: str,
        entry_price: float,
        actual_peak_price: float,
        actual_realized_return: Optional[float] = None,
        exit_reason: Optional[str] = None,
        time_window_hours: float = 48.0
    ) -> bool:
        """
        Update a prediction record with actual outcome.
        
        Finds the most recent prediction for this symbol/price within time_window_hours
        and records the actual results.
        
        Returns: True if a matching prediction was found and updated
        """
        now = time.time()
        cutoff_ts = now - (time_window_hours * 3600)
        
        # Find matching prediction (most recent within window)
        matching_records = [
            r for r in self.records
            if r.symbol == symbol
            and abs(r.entry_price - entry_price) / entry_price < 0.01  # within 1%
            and r.timestamp >= cutoff_ts
            and not r.outcome_recorded
        ]
        
        if not matching_records:
            return False
        
        # Update most recent match
        record = max(matching_records, key=lambda r: r.timestamp)
        
        record.actual_peak_price = actual_peak_price
        record.actual_peak_return = (actual_peak_price / entry_price) - 1.0
        record.actual_realized_return = actual_realized_return
        record.time_to_peak_hours = (now - record.timestamp) / 3600.0
        record.outcome_recorded = True
        record.exit_reason = exit_reason
        
        self._save()
        return True
    
    def get_metrics(self, days_lookback: Optional[int] = None) -> MLMetrics:
        """
        Calculate performance metrics.
        
        Args:
            days_lookback: If provided, only use predictions from last N days
        """
        records = self.records
        
        if days_lookback:
            cutoff_ts = time.time() - (days_lookback * 24 * 3600)
            records = [r for r in records if r.timestamp >= cutoff_ts]
        
        completed = [r for r in records if r.outcome_recorded]
        
        if not completed:
            return self._empty_metrics()
        
        # Hit rates
        tp80_hits = sum(1 for r in completed if r.actual_peak_price and r.actual_peak_price >= r.pred_tp80)
        tp90_hits = sum(1 for r in completed if r.actual_peak_price and r.actual_peak_price >= r.pred_tp90)
        
        tp80_hit_rate = tp80_hits / len(completed) if completed else 0.0
        tp90_hit_rate = tp90_hits / len(completed) if completed else 0.0
        
        # Directional accuracy (predicted up, actually went up)
        correct_direction = sum(
            1 for r in completed
            if r.actual_peak_return and r.pred_ret80 > 0 and r.actual_peak_return > 0
        )
        directional_accuracy = correct_direction / len(completed) if completed else 0.0
        
        # Error metrics for TP80
        errors_tp80 = [
            abs(r.actual_peak_price - r.pred_tp80)
            for r in completed if r.actual_peak_price
        ]
        
        mae_tp80 = statistics.mean(errors_tp80) if errors_tp80 else 0.0
        rmse_tp80 = statistics.stdev(errors_tp80) if len(errors_tp80) > 1 else 0.0
        
        pct_errors = [
            abs(r.actual_peak_price - r.pred_tp80) / r.pred_tp80 * 100
            for r in completed if r.actual_peak_price and r.pred_tp80 > 0
        ]
        mape_tp80 = statistics.mean(pct_errors) if pct_errors else 0.0
        
        # Calibration (coverage)
        tp80_coverage = tp80_hit_rate  # For quantile forecasts, coverage = hit rate
        tp90_coverage = tp90_hit_rate
        
        # Performance stats
        avg_pred = statistics.mean([r.pred_ret80 for r in completed])
        avg_actual = statistics.mean([r.actual_peak_return for r in completed if r.actual_peak_return is not None])
        avg_time = statistics.mean([r.time_to_peak_hours for r in completed if r.time_to_peak_hours is not None])
        
        # Recent performance (last 7 days)
        recent_cutoff = time.time() - (7 * 24 * 3600)
        recent = [r for r in completed if r.timestamp >= recent_cutoff]
        
        if recent:
            recent_tp80 = sum(1 for r in recent if r.actual_peak_price and r.actual_peak_price >= r.pred_tp80) / len(recent)
            recent_dir = sum(1 for r in recent if r.actual_peak_return and r.pred_ret80 > 0 and r.actual_peak_return > 0) / len(recent)
        else:
            recent_tp80 = 0.0
            recent_dir = 0.0
        
        return MLMetrics(
            total_predictions=len(records),
            predictions_with_outcomes=len(completed),
            tp80_hit_rate=tp80_hit_rate,
            tp90_hit_rate=tp90_hit_rate,
            directional_accuracy=directional_accuracy,
            mae_tp80=mae_tp80,
            rmse_tp80=rmse_tp80,
            mape_tp80=mape_tp80,
            tp80_coverage=tp80_coverage,
            tp90_coverage=tp90_coverage,
            avg_predicted_return=avg_pred,
            avg_actual_return=avg_actual,
            avg_time_to_peak_hours=avg_time,
            recent_tp80_hit_rate=recent_tp80,
            recent_directional_accuracy=recent_dir
        )
    
    def _empty_metrics(self) -> MLMetrics:
        """Return empty metrics when no data available."""
        return MLMetrics(
            total_predictions=len(self.records),
            predictions_with_outcomes=0,
            tp80_hit_rate=0.0,
            tp90_hit_rate=0.0,
            directional_accuracy=0.0,
            mae_tp80=0.0,
            rmse_tp80=0.0,
            mape_tp80=0.0,
            tp80_coverage=0.0,
            tp90_coverage=0.0,
            avg_predicted_return=0.0,
            avg_actual_return=0.0,
            avg_time_to_peak_hours=0.0,
            recent_tp80_hit_rate=0.0,
            recent_directional_accuracy=0.0
        )
    
    def generate_report(self, days: Optional[int] = None) -> str:
        """
        Generate a detailed performance report.
        
        Args:
            days: If provided, report on last N days only
        """
        metrics = self.get_metrics(days)
        
        title = f"ML Performance Report - Last {days} Days" if days else "ML Performance Report - All Time"
        
        report = f"\n{'='*70}\n{title}\n{'='*70}\n\n"
        
        report += f"Dataset Size:\n"
        report += f"  Total Predictions:      {metrics.total_predictions}\n"
        report += f"  With Outcomes:          {metrics.predictions_with_outcomes}\n"
        report += f"  Coverage:               {metrics.predictions_with_outcomes/max(1, metrics.total_predictions)*100:.1f}%\n\n"
        
        if metrics.predictions_with_outcomes == 0:
            if metrics.total_predictions > 0:
                report += f"Status: {metrics.total_predictions} predictions logged but no outcomes recorded yet.\n"
                report += "\nOutcomes are recorded when positions reach peaks or close.\n"
                report += "This typically takes 24-48 hours after predictions are made.\n"
                report += "\nTips:\n"
                report += "  - Ensure bot is managing positions (manage_existing_positions: true)\n"
                report += "  - Check that you have open positions to track\n"
                report += "  - Wait 1-2 days for positions to move and record peaks\n"
            else:
                report += "No predictions logged yet. Ensure ML model is enabled and loaded.\n"
            report += f"{'='*70}\n"
            return report
        
        report += f"Prediction Accuracy:\n"
        report += f"  TP80 Hit Rate:          {metrics.tp80_hit_rate*100:.1f}% (target: 80%)\n"
        report += f"  TP90 Hit Rate:          {metrics.tp90_hit_rate*100:.1f}% (target: 90%)\n"
        report += f"  Directional Accuracy:   {metrics.directional_accuracy*100:.1f}%\n\n"
        
        report += f"Error Metrics (TP80):\n"
        report += f"  MAE (Mean Absolute):    ${metrics.mae_tp80:.2f}\n"
        report += f"  RMSE (Root MSE):        ${metrics.rmse_tp80:.2f}\n"
        report += f"  MAPE (% Error):         {metrics.mape_tp80:.1f}%\n\n"
        
        report += f"Model Calibration:\n"
        report += f"  TP80 Coverage:          {metrics.tp80_coverage*100:.1f}% (should be ~80%)\n"
        report += f"  TP90 Coverage:          {metrics.tp90_coverage*100:.1f}% (should be ~90%)\n\n"
        
        report += f"Return Statistics:\n"
        report += f"  Avg Predicted Return:   {metrics.avg_predicted_return*100:+.2f}%\n"
        report += f"  Avg Actual Return:      {metrics.avg_actual_return*100:+.2f}%\n"
        report += f"  Avg Time to Peak:       {metrics.avg_time_to_peak_hours:.1f} hours\n\n"
        
        report += f"Recent Performance (7 days):\n"
        report += f"  TP80 Hit Rate:          {metrics.recent_tp80_hit_rate*100:.1f}%\n"
        report += f"  Directional Accuracy:   {metrics.recent_directional_accuracy*100:.1f}%\n\n"
        
        # Performance assessment
        report += f"Performance Assessment:\n"
        
        if metrics.tp80_hit_rate >= 0.75:
            report += f"  ✅ TP80 accuracy is GOOD ({metrics.tp80_hit_rate*100:.1f}% ≥ 75%)\n"
        elif metrics.tp80_hit_rate >= 0.65:
            report += f"  ⚠️  TP80 accuracy is ACCEPTABLE ({metrics.tp80_hit_rate*100:.1f}%)\n"
        else:
            report += f"  ❌ TP80 accuracy is POOR ({metrics.tp80_hit_rate*100:.1f}% < 65%) - Consider retraining\n"
        
        if metrics.directional_accuracy >= 0.60:
            report += f"  ✅ Directional accuracy is GOOD ({metrics.directional_accuracy*100:.1f}% ≥ 60%)\n"
        else:
            report += f"  ❌ Directional accuracy is POOR ({metrics.directional_accuracy*100:.1f}% < 60%)\n"
        
        if abs(metrics.tp80_coverage - 0.80) <= 0.10:
            report += f"  ✅ Model is well-calibrated (coverage {metrics.tp80_coverage*100:.1f}% ≈ 80%)\n"
        else:
            report += f"  ⚠️  Model calibration off (coverage {metrics.tp80_coverage*100:.1f}% vs target 80%)\n"
        
        report += f"\n{'='*70}\n"
        
        return report
    
    def should_retrain(self, min_samples: int = 50) -> Tuple[bool, str]:
        """
        Determine if model should be retrained based on performance.
        
        Returns: (should_retrain, reason)
        """
        metrics = self.get_metrics(days_lookback=30)  # Last 30 days
        
        if metrics.predictions_with_outcomes < min_samples:
            return False, f"Not enough samples yet ({metrics.predictions_with_outcomes}/{min_samples})"
        
        # Check recent performance (last 7 days)
        recent_metrics = self.get_metrics(days_lookback=7)
        
        if recent_metrics.predictions_with_outcomes < 10:
            return False, "Not enough recent samples for assessment"
        
        # Retrain if recent performance is significantly worse
        if recent_metrics.tp80_hit_rate < 0.60 and metrics.tp80_hit_rate > 0.70:
            return True, f"Recent accuracy degraded ({recent_metrics.tp80_hit_rate*100:.1f}% < 60%)"
        
        if recent_metrics.directional_accuracy < 0.50:
            return True, f"Recent directional accuracy poor ({recent_metrics.directional_accuracy*100:.1f}% < 50%)"
        
        # Check overall performance
        if metrics.tp80_hit_rate < 0.65:
            return True, f"Overall TP80 accuracy poor ({metrics.tp80_hit_rate*100:.1f}% < 65%)"
        
        return False, "Performance is acceptable"
    
    def get_symbol_performance(self, min_predictions: int = 3, days_lookback: Optional[int] = None) -> Dict[str, Dict]:
        """
        Get per-symbol performance statistics.
        
        Args:
            min_predictions: Minimum number of predictions to include symbol
            days_lookback: If provided, only use predictions from last N days
            
        Returns:
            Dictionary mapping symbol to performance stats
        """
        records = self.records
        
        if days_lookback:
            cutoff_ts = time.time() - (days_lookback * 24 * 3600)
            records = [r for r in records if r.timestamp >= cutoff_ts]
        
        completed = [r for r in records if r.outcome_recorded]
        
        # Group by symbol
        by_symbol: Dict[str, List[MLPredictionRecord]] = {}
        for record in completed:
            if record.symbol not in by_symbol:
                by_symbol[record.symbol] = []
            by_symbol[record.symbol].append(record)
        
        # Calculate stats for each symbol
        results = {}
        for symbol, symbol_records in by_symbol.items():
            if len(symbol_records) < min_predictions:
                continue
            
            # Count hits
            tp50_hits = sum(1 for r in symbol_records if r.actual_peak_price and r.actual_peak_price >= r.pred_tp50)
            tp80_hits = sum(1 for r in symbol_records if r.actual_peak_price and r.actual_peak_price >= r.pred_tp80)
            tp90_hits = sum(1 for r in symbol_records if r.actual_peak_price and r.actual_peak_price >= r.pred_tp90)
            
            total = len(symbol_records)
            
            # Average returns
            avg_predicted = statistics.mean([r.pred_ret80 for r in symbol_records])
            avg_actual = statistics.mean([r.actual_peak_return for r in symbol_records if r.actual_peak_return is not None])
            
            # Error metrics
            errors = [abs(r.actual_peak_price - r.pred_tp80) for r in symbol_records if r.actual_peak_price]
            mae = statistics.mean(errors) if errors else 0.0
            
            results[symbol] = {
                'total_predictions': total,
                'tp50_hits': tp50_hits,
                'tp80_hits': tp80_hits,
                'tp90_hits': tp90_hits,
                'tp50_rate': tp50_hits / total,
                'tp80_rate': tp80_hits / total,
                'tp90_rate': tp90_hits / total,
                'avg_predicted_return': avg_predicted,
                'avg_actual_return': avg_actual,
                'mae': mae,
                'accuracy_score': (tp80_hits / total) * 0.5 + (tp90_hits / total) * 0.3 + (tp50_hits / total) * 0.2
            }
        
        return results
