# backtest/backtester.py
"""
Backtesting framework for the Kraken Trend Bot.

Validates strategy, ML models, and position management logic against
historical data without touching the exchange. Supports:
  - Multi-symbol backtests using cached or freshly fetched OHLCV data
  - Full strategy + ML signal evaluation
  - Simulated position management (stop-loss, take-profit, trailing stop)
  - Per-symbol and aggregate performance metrics
  - Equity curve tracking and drawdown analysis
  - Comparison runs (e.g. with/without ML, different parameters)

Usage:
    python -m backtest.backtester                       # default config backtest
    python -m backtest.backtester --symbols BTC/USD ETH/USD SOL/USD
    python -m backtest.backtester --no-ml               # strategy-only mode
    python -m backtest.backtester --history-days 365
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bot.top5_trader import (
    TrendStrategy,
    StrategyResult,
    ema_last,
    rsi_last,
    atr_pct_last,
    canonical_asset,
    split_symbol,
    timeframe_to_minutes,
    bars_in_24h,
    safe_float,
)
from bot.torch_tp_forecaster import TorchTPForecaster, TPPrediction, build_seq_features

CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "backtest" / "results"

# Kraken's REST API hard-limits OHLCV to ~720 candles per timeframe.
# This determines max practical history per timeframe:
#   15m → ~8 days  |  1h → ~30 days  |  4h → ~120 days  |  1d → ~720 days
KRAKEN_MAX_CANDLES = 720

# Default model path in config; per-timeframe models use suffix, e.g.
#   data/torch_tp_forecaster.pt      ← default (15m)
#   data/torch_tp_forecaster_1d.pt   ← daily model
#   data/torch_tp_forecaster_4h.pt   ← 4-hour model
DEFAULT_ML_TIMEFRAME = "15m"


def max_history_days_for_timeframe(timeframe: str) -> int:
    """Max practical history Kraken can serve for a given timeframe."""
    minutes = timeframe_to_minutes(timeframe)
    return int(KRAKEN_MAX_CANDLES * minutes / (60 * 24))


def recommend_timeframe(history_days: int) -> str:
    """Suggest the best timeframe for the requested history."""
    for tf in ["15m", "1h", "4h", "1d"]:
        if max_history_days_for_timeframe(tf) >= history_days:
            return tf
    return "1d"


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """All tuneable backtest parameters in one place."""
    # Data
    symbols: List[str] = field(default_factory=lambda: [])
    timeframe: str = "15m"
    history_days: int = 120
    train_cutoff_pct: float = 0.0   # skip first N% as model warm-up

    # Strategy
    ema_fast: int = 21
    ema_slow: int = 55
    atr_period: int = 14
    rsi_period: int = 14
    rsi_min: float = 52.0
    max_extension_atr: float = 1.8

    # ML
    use_ml: bool = True
    ml_lookback: int = 64
    min_tp80_pct: float = 0.06

    # Risk / position management
    initial_capital: float = 1000.0
    risk_per_trade_pct: float = 15.0
    max_position_pct: float = 12.0
    max_trade_notional: float = 100.0
    max_open_positions: int = 12

    # Stop-loss
    stop_loss_pct: float = 15.0

    # Take-profit levels  [{pct, sell_pct}, ...]
    tp_levels: List[Dict[str, float]] = field(default_factory=lambda: [
        {"pct": 10.0, "sell_pct": 15.0},
        {"pct": 25.0, "sell_pct": 20.0},
        {"pct": 50.0, "sell_pct": 25.0},
        {"pct": 100.0, "sell_pct": 40.0},
    ])

    # Trailing stop
    trailing_arm_pct: float = 8.0
    trailing_trail_pct: float = 10.0

    # Anti-whipsaw
    min_hold_bars: int = 3       # don't exit_signal before N bars
    cooldown_after_stop: int = 5 # bars to wait before re-entry after stop

    # Fees
    fee_pct: float = 0.26   # Kraken taker fee (0.26%)


@dataclass
class Position:
    symbol: str
    entry_bar: int
    entry_price: float
    qty: float
    peak_price: float
    tp_hits: Dict[str, bool] = field(default_factory=dict)
    cost_basis: float = 0.0   # total USD spent (including fees)


@dataclass
class Trade:
    symbol: str
    side: str           # "buy" / "sell"
    bar_index: int
    timestamp: int
    price: float
    qty: float
    notional: float
    fee: float
    pnl: float = 0.0   # realized PnL (sell only)
    reason: str = ""    # "entry" / "stop_loss" / "tp_10" / "trailing" / "exit_signal"


@dataclass
class SymbolResult:
    symbol: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0


@dataclass
class BacktestResult:
    """Aggregate results across all symbols."""
    config: BacktestConfig
    start_time: str = ""
    end_time: str = ""
    duration_sec: float = 0.0

    # Portfolio
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0

    # Trades
    total_trades: int = 0
    total_buys: int = 0
    total_sells: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_trade_pnl: float = 0.0

    # Gross/Net
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0

    # ML specific
    ml_signals_total: int = 0
    ml_signals_correct: int = 0
    ml_accuracy: float = 0.0

    # Time
    total_bars_processed: int = 0
    avg_hold_bars: int = 0

    # Per-symbol breakdown
    symbol_results: Dict[str, SymbolResult] = field(default_factory=dict)

    # Equity curve for charting
    equity_curve: List[float] = field(default_factory=list)
    equity_timestamps: List[int] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)


# ------------------------------------------------------------------
# OHLCV data loading (re-uses training cache or fetches fresh)
# ------------------------------------------------------------------

def _load_cached_ohlcv(symbol: str, timeframe: str, history_days: int) -> Optional[List[List[float]]]:
    """Try to load from ml training cache. Only returns data if it covers
    at least 50% of the requested history span."""
    cache_dir = DATA_DIR / "cache_ohlcv"
    safe = symbol.replace("/", "_")

    # Gather all matching cache files and pick the one with the most data
    candidates: List[Path] = []
    exact = cache_dir / f"{safe}__{timeframe}__{history_days}d.json"
    if exact.exists():
        candidates.append(exact)
    for f in cache_dir.glob(f"{safe}__{timeframe}__*d.json"):
        if f not in candidates:
            candidates.append(f)

    min_span_days = history_days * 0.50  # require at least 50% coverage

    best_rows: Optional[List[List[float]]] = None
    best_span = 0.0

    for p in candidates:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            rows = data.get("ohlcv") or data
            if not isinstance(rows, list) or len(rows) < 100:
                continue
            span_days = (rows[-1][0] - rows[0][0]) / 86_400_000
            if span_days >= min_span_days and span_days > best_span:
                best_rows = rows
                best_span = span_days
        except Exception:
            continue

    if best_rows:
        print(f"  [CACHE] {symbol}: {len(best_rows):,} candles ({best_span:.0f} days) — "
              f"{'✅ covers' if best_span >= history_days * 0.8 else '⚠️  partial'} requested {history_days}d")
    return best_rows


def _fetch_ohlcv_live(symbol: str, timeframe: str, history_days: int) -> Optional[List[List[float]]]:
    """Fetch full history from exchange using pagination (requires internet).

    Uses the same proven pagination approach as the ML training pipeline.
    """
    try:
        import ccxt
        ex = ccxt.kraken({"enableRateLimit": True})
        ex.load_markets()

        if symbol not in ex.markets:
            print(f"  [SKIP] {symbol} not in Kraken markets")
            return None

        tf_ms = timeframe_to_minutes(timeframe) * 60_000
        now_ms = int(time.time() * 1000)
        since_ms = now_ms - history_days * 86_400_000
        page_limit = 720

        all_rows: List[List[float]] = []
        seen: set = set()
        cur = since_ms
        page_count = 0
        expected_candles = int((now_ms - since_ms) / tf_ms)

        while cur < now_ms:
            try:
                rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cur, limit=page_limit)
            except Exception as e:
                print(f"    [FETCH ERR] {symbol} page {page_count}: {e}")
                break

            if not rows:
                break

            added = 0
            for r in rows:
                ts = int(r[0])
                if ts >= now_ms:
                    continue
                if ts in seen:
                    continue
                seen.add(ts)
                all_rows.append(r)
                added += 1

            page_count += 1

            # Progress every 10 pages
            if page_count % 10 == 0:
                progress = len(all_rows) / max(1, expected_candles) * 100
                print(f"    [{symbol}] {len(all_rows):,} candles ({progress:.0f}%) — page {page_count}")

            # Advance cursor past the last received candle
            last_ts = int(rows[-1][0])
            next_cur = last_ts + tf_ms

            if next_cur <= cur:        # stuck
                break
            if last_ts >= now_ms - tf_ms:   # reached present
                break
            if added == 0:              # all duplicates
                break

            cur = next_cur
            time.sleep(1.0)  # respect rate limits

        all_rows.sort(key=lambda x: x[0])

        if all_rows:
            span_days = (all_rows[-1][0] - all_rows[0][0]) / 86_400_000
            print(f"    [{symbol}] Fetch complete: {len(all_rows):,} candles "
                  f"({span_days:.0f} days) in {page_count} pages")
            if span_days < history_days * 0.5:
                print(f"    ⚠️  Kraken only returned {span_days:.0f}d "
                      f"(requested {history_days}d). "
                      f"Try a larger timeframe for more history.")

        if len(all_rows) > 50:
            return all_rows
    except Exception as e:
        print(f"  [ERROR] fetching {symbol}: {e}")
    return None


def load_ohlcv(
    symbol: str,
    timeframe: str,
    history_days: int,
    use_cache: bool = True,
) -> Optional[List[List[float]]]:
    """Load OHLCV data: try cache first, then live fetch."""
    if use_cache:
        data = _load_cached_ohlcv(symbol, timeframe, history_days)
        if data:
            return data

    print(f"  [FETCH] {symbol}: downloading {history_days} days of {timeframe}...")
    data = _fetch_ohlcv_live(symbol, timeframe, history_days)
    if data:
        span_days = (data[-1][0] - data[0][0]) / 86_400_000
        print(f"  [OK] {symbol}: {len(data):,} candles ({span_days:.0f} days)")
    return data


# ------------------------------------------------------------------
# Backtesting engine
# ------------------------------------------------------------------

class BacktestEngine:
    """
    Event-driven backtester that steps through OHLCV bars, evaluates the
    strategy + ML signals, and simulates paper trades with full position
    management (stop-loss, take-profit, trailing stop).
    """

    def __init__(
        self,
        config: BacktestConfig,
        forecaster: Optional[TorchTPForecaster] = None,
    ):
        self.cfg = config
        self.forecaster = forecaster if config.use_ml else None

        self.strategy = TrendStrategy(
            ema_fast=config.ema_fast,
            ema_slow=config.ema_slow,
            atr_period=config.atr_period,
            rsi_period=config.rsi_period,
            rsi_min=config.rsi_min,
            max_extension_atr=config.max_extension_atr,
        )

        # Portfolio state
        self.cash: float = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

        # ML tracking
        self.ml_signals_total: int = 0
        self.ml_signals_correct: int = 0

        # Hold-time tracking
        self._hold_bars: List[int] = []
        self._equity_dict: Dict[int, float] = {}
        self._cooldown: Dict[str, int] = {}  # symbol -> bar when cooldown expires

    # ---------- helpers ----------

    def _equity(self, prices: Dict[str, float]) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            px = prices.get(sym, pos.entry_price)
            eq += pos.qty * px
        return eq

    def _open_position_count(self) -> int:
        return len(self.positions)

    def _compute_size(self, price: float, equity: float) -> float:
        """Compute buy notional respecting risk limits."""
        risk_notional = equity * (self.cfg.risk_per_trade_pct / 100.0)
        max_pos_notional = equity * (self.cfg.max_position_pct / 100.0)
        notional = min(risk_notional, max_pos_notional, self.cfg.max_trade_notional)
        return max(0.0, notional)

    def _fee(self, notional: float) -> float:
        return notional * (self.cfg.fee_pct / 100.0)

    # ---------- trade execution ----------

    def _execute_buy(
        self,
        symbol: str,
        price: float,
        bar_index: int,
        timestamp: int,
        reason: str = "entry",
    ) -> bool:
        if symbol in self.positions:
            return False  # already holding
        if self._open_position_count() >= self.cfg.max_open_positions:
            return False

        equity = self.cash  # simplified — just cash for sizing
        notional = self._compute_size(price, equity)
        if notional <= 0 or notional > self.cash:
            notional = min(notional, self.cash * 0.95)
        if notional <= 1.0:
            return False

        fee = self._fee(notional)
        qty = (notional - fee) / price
        if qty <= 0:
            return False

        self.cash -= notional
        self.positions[symbol] = Position(
            symbol=symbol,
            entry_bar=bar_index,
            entry_price=price,
            qty=qty,
            peak_price=price,
            cost_basis=notional,
        )
        self.trades.append(Trade(
            symbol=symbol, side="buy", bar_index=bar_index,
            timestamp=timestamp, price=price, qty=qty,
            notional=notional, fee=fee, reason=reason,
        ))
        return True

    def _execute_sell(
        self,
        symbol: str,
        price: float,
        qty: float,
        bar_index: int,
        timestamp: int,
        reason: str = "exit",
    ) -> float:
        pos = self.positions.get(symbol)
        if pos is None or pos.qty <= 0:
            return 0.0

        sell_qty = min(qty, pos.qty)
        notional = sell_qty * price
        fee = self._fee(notional)
        proceeds = notional - fee

        cost_frac = (sell_qty / pos.qty) * pos.cost_basis if pos.qty > 0 else 0.0
        pnl = proceeds - cost_frac

        self.cash += proceeds
        pos.qty -= sell_qty
        pos.cost_basis -= cost_frac

        if pos.qty <= 1e-12:
            self._hold_bars.append(bar_index - pos.entry_bar)
            del self.positions[symbol]

        self.trades.append(Trade(
            symbol=symbol, side="sell", bar_index=bar_index,
            timestamp=timestamp, price=price, qty=sell_qty,
            notional=notional, fee=fee, pnl=pnl, reason=reason,
        ))
        return pnl

    # ---------- position management per bar ----------

    def _manage_position(
        self,
        symbol: str,
        price: float,
        bar_index: int,
        timestamp: int,
    ) -> None:
        """Apply stop-loss, take-profit, and trailing stop to an open position."""
        pos = self.positions.get(symbol)
        if pos is None:
            return

        pos.peak_price = max(pos.peak_price, price)
        entry = pos.entry_price
        if entry <= 0:
            return
        ret_pct = (price / entry - 1.0) * 100.0

        # 1. Stop-loss
        if ret_pct <= -self.cfg.stop_loss_pct:
            self._execute_sell(symbol, price, pos.qty, bar_index, timestamp, reason="stop_loss")
            self._cooldown[symbol] = bar_index + self.cfg.cooldown_after_stop
            return

        # 2. Take-profit levels (partial sells)
        for lvl in self.cfg.tp_levels:
            tp_pct = float(lvl["pct"])
            sell_pct = float(lvl["sell_pct"])
            key = str(tp_pct)
            if ret_pct >= tp_pct and not pos.tp_hits.get(key, False):
                part_qty = pos.qty * (sell_pct / 100.0)
                self._execute_sell(symbol, price, part_qty, bar_index, timestamp,
                                   reason=f"tp_{tp_pct:.0f}")
                pos.tp_hits[key] = True
                # re-check position still exists
                if symbol not in self.positions:
                    return

        # 3. Trailing stop
        if ret_pct >= self.cfg.trailing_arm_pct:
            trail_price = pos.peak_price * (1.0 - self.cfg.trailing_trail_pct / 100.0)
            if price <= trail_price:
                remaining = self.positions.get(symbol)
                if remaining:
                    self._execute_sell(symbol, price, remaining.qty, bar_index, timestamp,
                                       reason="trailing_stop")

    # ---------- main backtest loop ----------

    def run_symbol(
        self,
        symbol: str,
        ohlcv: List[List[float]],
        btc_ohlcv: Optional[List[List[float]]] = None,
    ) -> SymbolResult:
        """Run backtest on a single symbol's OHLCV data."""
        cfg = self.cfg
        bars24 = bars_in_24h(cfg.timeframe)
        min_bars = max(cfg.ema_slow * 3, cfg.ml_lookback + 20, 120)

        # Skip warm-up portion
        start_bar = max(min_bars, int(len(ohlcv) * cfg.train_cutoff_pct / 100.0))

        result = SymbolResult(symbol=symbol)
        sym_trades_start = len(self.trades)
        gross_wins = 0.0
        gross_losses = 0.0

        # Build BTC returns map for timestamp matching
        btc_returns: Dict[int, float] = {}
        if btc_ohlcv and len(btc_ohlcv) > 1:
            prev_c = float(btc_ohlcv[0][4])
            for row in btc_ohlcv[1:]:
                ts = int(row[0])
                c = float(row[4])
                if prev_c > 0 and c > 0:
                    btc_returns[ts] = math.log(c / prev_c)
                prev_c = c

        for i in range(start_bar, len(ohlcv)):
            ts = int(ohlcv[i][0])
            price = float(ohlcv[i][4])
            if price <= 0:
                continue

            # Track equity
            prices = {symbol: price}
            for s, p in self.positions.items():
                if s != symbol:
                    prices[s] = p.entry_price  # approximate for other symbols
            eq_val = self._equity(prices)
            self.equity_curve.append(eq_val)
            self._equity_dict[ts] = eq_val

            # Manage existing position first
            self._manage_position(symbol, price, i, ts)

            # Evaluate strategy on history up to current bar
            window = ohlcv[max(0, i - min_bars):i + 1]
            strat_result: StrategyResult = self.strategy.evaluate(window)

            # ML prediction
            ml_ret80: Optional[float] = None
            ml_pred: Optional[TPPrediction] = None
            if self.forecaster is not None and i >= cfg.ml_lookback + 20:
                ml_window = ohlcv[max(0, i - cfg.ml_lookback - 60):i + 1]
                # Slice matching BTC OHLCV window for the beta feature
                btc_window = None
                if btc_ohlcv:
                    bar_ts_start = int(ml_window[0][0]) if ml_window else 0
                    bar_ts_end = int(ml_window[-1][0]) if ml_window else 0
                    btc_window = [r for r in btc_ohlcv if bar_ts_start <= int(r[0]) <= bar_ts_end]
                    if not btc_window or len(btc_window) < 10:
                        btc_window = None

                try:
                    ml_pred = self.forecaster.predict(
                        ml_window,
                        lookback=cfg.ml_lookback,
                        horizon_bars=bars24,
                        btc_ohlcv=btc_window,
                    )
                    if ml_pred:
                        ml_ret80 = ml_pred.mfe_returns.get(0.8)
                except Exception:
                    pass

            # ---- Entry logic ----
            if symbol not in self.positions:
                # Cooldown check: skip entry if recently stopped out
                if i < self._cooldown.get(symbol, 0):
                    continue

                bullish = strat_result.trend == "BULL"
                signal_ok = strat_result.signal == "LONG"  # require full LONG (RSI + ATR guards)

                ml_ok = True
                if cfg.use_ml and ml_ret80 is not None:
                    self.ml_signals_total += 1
                    ml_ok = ml_ret80 >= cfg.min_tp80_pct

                if bullish and signal_ok and ml_ok:
                    bought = self._execute_buy(symbol, price, i, ts, reason="entry")

                    # Track ML accuracy: did price actually reach TP80 within horizon?
                    if bought and ml_pred is not None and ml_ret80 is not None:
                        tp80_price = ml_pred.tp_prices.get(0.8, 0.0)
                        horizon_end = min(i + bars24, len(ohlcv))
                        future_highs = [float(ohlcv[j][2]) for j in range(i + 1, horizon_end)]
                        if future_highs and max(future_highs) >= tp80_price:
                            self.ml_signals_correct += 1

            # ---- Exit signal from strategy ----
            elif strat_result.signal == "EXIT":
                pos = self.positions.get(symbol)
                if pos:
                    # Min-hold guard: don't exit too early (avoids EMA whipsaw)
                    bars_held = i - pos.entry_bar
                    if bars_held < cfg.min_hold_bars:
                        continue
                    self._execute_sell(symbol, price, pos.qty, i, ts, reason="exit_signal")

        # Close any remaining position at last bar
        if symbol in self.positions:
            last_price = float(ohlcv[-1][4])
            last_ts = int(ohlcv[-1][0])
            pos = self.positions[symbol]
            self._execute_sell(symbol, last_price, pos.qty, len(ohlcv) - 1, last_ts,
                               reason="backtest_end")

        # Compute per-symbol stats
        sym_trades = self.trades[sym_trades_start:]
        sells = [t for t in sym_trades if t.side == "sell"]
        result.total_trades = len(sells)
        for t in sells:
            if t.pnl >= 0:
                result.winning_trades += 1
                gross_wins += t.pnl
            else:
                result.losing_trades += 1
                gross_losses += abs(t.pnl)
        result.total_fees = sum(t.fee for t in sym_trades)
        result.gross_pnl = gross_wins - gross_losses
        result.net_pnl = result.gross_pnl
        result.win_rate = (result.winning_trades / result.total_trades * 100.0
                           if result.total_trades > 0 else 0.0)
        result.avg_win = gross_wins / result.winning_trades if result.winning_trades > 0 else 0.0
        result.avg_loss = gross_losses / result.losing_trades if result.losing_trades > 0 else 0.0
        result.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")
        result.expectancy = result.net_pnl / result.total_trades if result.total_trades > 0 else 0.0

        return result

    # ---------- multi-symbol backtest ----------

    def run(
        self,
        symbol_data: Dict[str, List[List[float]]],
        btc_ohlcv: Optional[List[List[float]]] = None,
    ) -> BacktestResult:
        """Run backtest across multiple symbols."""
        t0 = time.time()
        result = BacktestResult(
            config=self.cfg,
            initial_capital=self.cfg.initial_capital,
            start_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        total_bars = 0
        for sym, ohlcv in symbol_data.items():
            print(f"  [BT] Running {sym} ({len(ohlcv):,} bars)...")
            sym_result = self.run_symbol(sym, ohlcv, btc_ohlcv=btc_ohlcv)
            result.symbol_results[sym] = sym_result
            total_bars += len(ohlcv)

        # Aggregate results
        result.total_bars_processed = total_bars
        result.final_equity = self.cash
        # Add any remaining positions (shouldn't be any since we close at end)
        result.total_return_pct = ((result.final_equity / self.cfg.initial_capital) - 1.0) * 100.0

        all_sells = [t for t in self.trades if t.side == "sell"]
        all_buys = [t for t in self.trades if t.side == "buy"]
        result.total_trades = len(all_sells)
        result.total_buys = len(all_buys)
        result.total_sells = len(all_sells)

        gross_profit = sum(t.pnl for t in all_sells if t.pnl >= 0)
        gross_loss = sum(abs(t.pnl) for t in all_sells if t.pnl < 0)
        result.gross_profit = gross_profit
        result.gross_loss = gross_loss
        result.total_fees = sum(t.fee for t in self.trades)
        result.net_pnl = gross_profit - gross_loss

        result.winning_trades = sum(1 for t in all_sells if t.pnl >= 0)
        result.losing_trades = sum(1 for t in all_sells if t.pnl < 0)
        result.win_rate = (result.winning_trades / result.total_trades * 100.0
                           if result.total_trades > 0 else 0.0)
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        result.avg_trade_pnl = result.net_pnl / result.total_trades if result.total_trades > 0 else 0.0
        result.expectancy = result.avg_trade_pnl

        # Max drawdown from equity curve
        if self.equity_curve:
            peak = self.equity_curve[0]
            max_dd = 0.0
            max_dd_usd = 0.0
            for eq in self.equity_curve:
                if eq > peak:
                    peak = eq
                dd_usd = peak - eq
                dd_pct = dd_usd / peak * 100.0 if peak > 0 else 0.0
                if dd_pct > max_dd:
                    max_dd = dd_pct
                    max_dd_usd = dd_usd
            result.max_drawdown_pct = max_dd
            result.max_drawdown_usd = max_dd_usd

        result.equity_curve = self.equity_curve

        # Build timestamped equity series (consolidated by timestamp)
        sorted_eq = sorted(self._equity_dict.items())
        if sorted_eq:
            result.equity_timestamps = [int(t) for t, _ in sorted_eq]
            result.equity_curve = [v for _, v in sorted_eq]

        # Serialize trade log for charting / reporting
        result.trades = [
            {"symbol": t.symbol, "side": t.side, "bar_index": t.bar_index,
             "timestamp": t.timestamp, "price": t.price, "qty": t.qty,
             "notional": t.notional, "fee": t.fee, "pnl": t.pnl,
             "reason": t.reason}
            for t in self.trades
        ]

        # ML stats
        result.ml_signals_total = self.ml_signals_total
        result.ml_signals_correct = self.ml_signals_correct
        result.ml_accuracy = (self.ml_signals_correct / self.ml_signals_total * 100.0
                              if self.ml_signals_total > 0 else 0.0)

        # Average hold time
        result.avg_hold_bars = (int(sum(self._hold_bars) / len(self._hold_bars))
                                if self._hold_bars else 0)

        result.duration_sec = time.time() - t0
        result.end_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return result


# ------------------------------------------------------------------
# Report formatting
# ------------------------------------------------------------------

def format_report(result: BacktestResult) -> str:
    """Generate a human-readable backtest report."""
    lines = []
    w = lines.append

    w("=" * 72)
    w("                    BACKTEST REPORT")
    w("=" * 72)
    w(f"  Run Time:       {result.start_time}  ({result.duration_sec:.1f}s)")
    w(f"  Timeframe:      {result.config.timeframe}")
    w(f"  Symbols:        {len(result.symbol_results)}")
    w(f"  Total Bars:     {result.total_bars_processed:,}")
    w(f"  ML Enabled:     {'YES' if result.config.use_ml else 'NO'}")
    w("")

    w("── PORTFOLIO ──────────────────────────────────────────────")
    w(f"  Initial Capital:  ${result.initial_capital:,.2f}")
    w(f"  Final Equity:     ${result.final_equity:,.2f}")
    w(f"  Net PnL:          ${result.net_pnl:+,.2f}")
    w(f"  Total Return:     {result.total_return_pct:+.2f}%")
    w(f"  Max Drawdown:     {result.max_drawdown_pct:.2f}% (${result.max_drawdown_usd:,.2f})")
    w("")

    w("── TRADES ─────────────────────────────────────────────────")
    w(f"  Total Round-Trips: {result.total_trades}")
    w(f"  Winners:           {result.winning_trades}  ({result.win_rate:.1f}%)")
    w(f"  Losers:            {result.losing_trades}")
    w(f"  Avg Trade PnL:     ${result.avg_trade_pnl:+.2f}")
    w(f"  Profit Factor:     {result.profit_factor:.2f}" if result.profit_factor < 999 else
      f"  Profit Factor:     ∞ (no losses)")
    w(f"  Expectancy:        ${result.expectancy:+.2f}")
    w(f"  Total Fees:        ${result.total_fees:,.2f}")
    w(f"  Avg Hold:          {result.avg_hold_bars} bars")
    w("")

    w("── GROSS BREAKDOWN ────────────────────────────────────────")
    w(f"  Gross Profit:  ${result.gross_profit:,.2f}")
    w(f"  Gross Loss:    ${result.gross_loss:,.2f}")
    w(f"  Net:           ${result.net_pnl:+,.2f}")
    w("")

    if result.config.use_ml:
        w("── ML MODEL PERFORMANCE ───────────────────────────────────")
        w(f"  ML Signals Evaluated:  {result.ml_signals_total}")
        w(f"  TP80 Correct (hit):    {result.ml_signals_correct}")
        w(f"  ML Accuracy:           {result.ml_accuracy:.1f}%")
        w("")

    if result.symbol_results:
        w("── PER-SYMBOL BREAKDOWN ───────────────────────────────────")
        w(f"  {'Symbol':<14s} {'Trades':>6s} {'WinRate':>8s} {'NetPnL':>10s} {'PF':>6s} {'Expectancy':>11s}")
        w("  " + "-" * 57)
        for sym, sr in sorted(result.symbol_results.items(),
                              key=lambda x: x[1].net_pnl, reverse=True):
            pf_str = f"{sr.profit_factor:.2f}" if sr.profit_factor < 999 else "∞"
            w(f"  {sym:<14s} {sr.total_trades:>6d} {sr.win_rate:>7.1f}% "
              f"${sr.net_pnl:>+9.2f} {pf_str:>6s} ${sr.expectancy:>+10.2f}")
        w("")

    w("── STRATEGY PARAMETERS ────────────────────────────────────")
    cfg = result.config
    w(f"  EMA Fast/Slow:     {cfg.ema_fast}/{cfg.ema_slow}")
    w(f"  RSI Period/Min:    {cfg.rsi_period}/{cfg.rsi_min}")
    w(f"  ATR Period:        {cfg.atr_period}")
    w(f"  Extension Guard:   {cfg.max_extension_atr} ATR")
    w(f"  Stop Loss:         {cfg.stop_loss_pct}%")
    w(f"  Trailing:          arm={cfg.trailing_arm_pct}% trail={cfg.trailing_trail_pct}%")
    w(f"  TP Levels:         {', '.join(f'+{l['pct']:.0f}%→sell {l['sell_pct']:.0f}%' for l in cfg.tp_levels)}")
    w(f"  Fee:               {cfg.fee_pct}%")
    if cfg.use_ml:
        w(f"  ML Lookback:       {cfg.ml_lookback} bars")
        w(f"  ML min TP80:       {cfg.min_tp80_pct:.1%}")
    w("")
    w("=" * 72)

    return "\n".join(lines)


def save_results(result: BacktestResult, tag: str = "") -> Path:
    """Save backtest results to JSON + text report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""

    # Save text report
    report = format_report(result)
    report_path = RESULTS_DIR / f"backtest{suffix}_{ts}.txt"
    report_path.write_text(report, encoding="utf-8")

    # Save JSON summary (without full equity curve for size)
    summary = {
        "timestamp": ts,
        "tag": tag,
        "initial_capital": result.initial_capital,
        "final_equity": result.final_equity,
        "total_return_pct": result.total_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor if result.profit_factor < 999 else None,
        "expectancy": result.expectancy,
        "net_pnl": result.net_pnl,
        "total_fees": result.total_fees,
        "ml_accuracy": result.ml_accuracy,
        "duration_sec": result.duration_sec,
        "symbols": list(result.symbol_results.keys()),
        "per_symbol": {
            sym: {
                "trades": sr.total_trades,
                "win_rate": sr.win_rate,
                "net_pnl": sr.net_pnl,
                "profit_factor": sr.profit_factor if sr.profit_factor < 999 else None,
            }
            for sym, sr in result.symbol_results.items()
        },
        "config": {
            "timeframe": result.config.timeframe,
            "ema_fast": result.config.ema_fast,
            "ema_slow": result.config.ema_slow,
            "stop_loss_pct": result.config.stop_loss_pct,
            "use_ml": result.config.use_ml,
            "fee_pct": result.config.fee_pct,
        },
        "trades": result.trades,
        "equity_series": (
            [[t, v] for t, v in zip(result.equity_timestamps, result.equity_curve)]
            if result.equity_timestamps else []
        ),
    }
    json_path = RESULTS_DIR / f"backtest{suffix}_{ts}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return report_path


# ------------------------------------------------------------------
# Comparison runner (with ML vs without ML, or different params)
# ------------------------------------------------------------------

def run_comparison(
    symbol_data: Dict[str, List[List[float]]],
    btc_ohlcv: Optional[List[List[float]]],
    base_config: BacktestConfig,
    forecaster: Optional[TorchTPForecaster],
) -> None:
    """Run two backtests: with ML and without ML, then compare."""

    print("\n╔══════════════════════════════════════════════╗")
    print("║        COMPARISON: ML vs NO-ML               ║")
    print("╚══════════════════════════════════════════════╝\n")

    # Run WITH ML
    if forecaster is not None:
        cfg_ml = BacktestConfig(**{
            k: v for k, v in base_config.__dict__.items()
        })
        cfg_ml.use_ml = True
        engine_ml = BacktestEngine(cfg_ml, forecaster=forecaster)
        result_ml = engine_ml.run(symbol_data, btc_ohlcv=btc_ohlcv)
        report_ml = format_report(result_ml)
        print(report_ml)
        save_results(result_ml, tag="with_ml")
    else:
        print("[WARN] No ML model available — skipping ML comparison")
        result_ml = None

    # Run WITHOUT ML
    cfg_no_ml = BacktestConfig(**{
        k: v for k, v in base_config.__dict__.items()
    })
    cfg_no_ml.use_ml = False
    engine_no_ml = BacktestEngine(cfg_no_ml, forecaster=None)
    result_no_ml = engine_no_ml.run(symbol_data, btc_ohlcv=btc_ohlcv)
    report_no_ml = format_report(result_no_ml)
    print(report_no_ml)
    save_results(result_no_ml, tag="no_ml")

    # Print comparison table
    if result_ml is not None:
        print("\n" + "=" * 60)
        print("           SIDE-BY-SIDE COMPARISON")
        print("=" * 60)
        print(f"  {'Metric':<25s} {'With ML':>15s} {'No ML':>15s}")
        print("  " + "-" * 55)

        def row(label: str, ml_val: str, no_ml_val: str) -> None:
            print(f"  {label:<25s} {ml_val:>15s} {no_ml_val:>15s}")

        row("Total Return",
            f"{result_ml.total_return_pct:+.2f}%",
            f"{result_no_ml.total_return_pct:+.2f}%")
        row("Net PnL",
            f"${result_ml.net_pnl:+,.2f}",
            f"${result_no_ml.net_pnl:+,.2f}")
        row("Max Drawdown",
            f"{result_ml.max_drawdown_pct:.2f}%",
            f"{result_no_ml.max_drawdown_pct:.2f}%")
        row("Total Trades",
            f"{result_ml.total_trades}",
            f"{result_no_ml.total_trades}")
        row("Win Rate",
            f"{result_ml.win_rate:.1f}%",
            f"{result_no_ml.win_rate:.1f}%")
        pf_ml = f"{result_ml.profit_factor:.2f}" if result_ml.profit_factor < 999 else "∞"
        pf_no = f"{result_no_ml.profit_factor:.2f}" if result_no_ml.profit_factor < 999 else "∞"
        row("Profit Factor", pf_ml, pf_no)
        row("Expectancy",
            f"${result_ml.expectancy:+.2f}",
            f"${result_no_ml.expectancy:+.2f}")
        row("Total Fees",
            f"${result_ml.total_fees:,.2f}",
            f"${result_no_ml.total_fees:,.2f}")
        row("Avg Hold (bars)",
            f"{result_ml.avg_hold_bars}",
            f"{result_no_ml.avg_hold_bars}")

        # Edge
        edge = result_ml.total_return_pct - result_no_ml.total_return_pct
        print()
        if edge > 0:
            print(f"  ✅ ML provides +{edge:.2f}% edge over strategy-only")
        elif edge < 0:
            print(f"  ⚠️  ML underperforms strategy-only by {abs(edge):.2f}%")
        else:
            print(f"  ➖ ML and strategy-only perform equally")
        print("=" * 60)


# ------------------------------------------------------------------
# CLI main
# ------------------------------------------------------------------

def load_config_from_yaml() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config_from_yaml(raw: dict, args: argparse.Namespace) -> BacktestConfig:
    """Merge YAML config with CLI overrides."""
    ex_cfg = raw.get("exchange", {}) or {}
    risk_cfg = raw.get("risk", {}) or {}
    ml_cfg = raw.get("ml", {}) or {}
    pm_cfg = raw.get("position_management", {}) or {}
    strat_cfg = raw.get("strategy", {}) or {}

    # Determine timeframe first, then merge per-timeframe ML overrides
    timeframe = args.timeframe or ex_cfg.get("timeframe", "15m")
    tf_key = f"ml_{timeframe}"  # e.g. "ml_1d"
    ml_tf = raw.get(tf_key, {}) or {}
    ml_merged = {**ml_cfg, **ml_tf}  # tf-specific overrides base ml

    cfg = BacktestConfig(
        timeframe=timeframe,
        history_days=args.history_days or int(ml_merged.get("history_days", 120)),

        ema_fast=int(strat_cfg.get("ema_fast", ex_cfg.get("ema_fast", 21))),
        ema_slow=int(strat_cfg.get("ema_slow", ex_cfg.get("ema_slow", 55))),
        atr_period=int(risk_cfg.get("atr_period", 14)),
        rsi_period=14,
        rsi_min=float(strat_cfg.get("rsi_min", 52.0)),
        max_extension_atr=float(strat_cfg.get("max_extension_atr", 1.8)),

        use_ml=not args.no_ml,
        ml_lookback=int(ml_merged.get("lookback", 64)),
        min_tp80_pct=float(ml_merged.get("min_tp80_pct", 0.06)),

        initial_capital=args.capital,
        risk_per_trade_pct=float(risk_cfg.get("risk_per_trade_pct", 15.0)),
        max_position_pct=float(risk_cfg.get("max_position_pct", 12.0)),
        max_trade_notional=float(risk_cfg.get("max_trade_notional", 100.0)),
        max_open_positions=int(risk_cfg.get("max_open_positions", 12)),

        min_hold_bars=int(pm_cfg.get("min_hold_bars", 3)),
        cooldown_after_stop=int(pm_cfg.get("cooldown_bars_after_stop", 5)),

        stop_loss_pct=float((pm_cfg.get("stop_loss") or {}).get("stop_pct", 15.0)),
        tp_levels=(pm_cfg.get("take_profit") or {}).get("levels") or [
            {"pct": 10.0, "sell_pct": 15.0},
            {"pct": 25.0, "sell_pct": 20.0},
            {"pct": 50.0, "sell_pct": 25.0},
            {"pct": 100.0, "sell_pct": 40.0},
        ],
        trailing_arm_pct=float((pm_cfg.get("trailing_stop") or {}).get("arm_after_pct", 8.0)),
        trailing_trail_pct=float((pm_cfg.get("trailing_stop") or {}).get("trail_pct", 10.0)),

        fee_pct=args.fee_pct,
    )

    if args.symbols:
        cfg.symbols = args.symbols

    return cfg


def _resolve_ml_model_path(raw: dict, timeframe: str) -> Optional[Path]:
    """Find the best ML model file for the given timeframe.

    Search order:
      1. Timeframe-specific model: torch_tp_forecaster_{tf}.pt
      2. Default model (only if timeframe matches its training timeframe)
    """
    ml_cfg = raw.get("ml", {}) or {}
    base_path = Path(ml_cfg.get("model_path", str(DATA_DIR / "torch_tp_forecaster.pt")))
    if not base_path.is_absolute():
        base_path = REPO_ROOT / base_path

    # 1. Try timeframe-specific model  e.g. torch_tp_forecaster_1d.pt
    tf_path = base_path.with_name(f"{base_path.stem}_{timeframe}{base_path.suffix}")
    if tf_path.exists():
        return tf_path

    # 2. Fall back to default model — check its training timeframe in meta
    if base_path.exists():
        meta_path = base_path.with_suffix(".meta.json")
        trained_tf = None
        if meta_path.exists():
            try:
                import json as _json
                meta = _json.loads(meta_path.read_text(encoding="utf-8"))
                trained_tf = meta.get("timeframe")
            except Exception:
                pass
        # Use default only when trained on the same timeframe (or unknown)
        if trained_tf is None or trained_tf == timeframe:
            return base_path
        # If trained on a *different* tf, still return it with a warning
        # (the caller decides whether to accept)
        return base_path

    return None


def load_ml_forecaster(raw: dict, timeframe: str = "15m") -> Optional[TorchTPForecaster]:
    ml_cfg = raw.get("ml", {}) or {}
    if not ml_cfg.get("enabled", False):
        return None

    model_path = _resolve_ml_model_path(raw, timeframe)
    if model_path is None or not model_path.exists():
        print(f"[BT] ML model not found for timeframe {timeframe}")
        return None

    f = TorchTPForecaster(model_path)
    if f.load():
        # Warn if the model's training timeframe differs from backtest timeframe
        trained_tf = f.meta.get("timeframe")
        if trained_tf and trained_tf != timeframe:
            print(f"[BT] ⚠️  Model was trained on {trained_tf}, backtesting on {timeframe}.")
            print(f"     Predictions may be less accurate. Train a {timeframe} model with:")
            print(f"     python -m ml.train_torch_forecaster --timeframe {timeframe}")
        print(f"[BT] ✅ ML model loaded: {model_path}")
        return f
    print(f"[BT] ❌ Failed to load ML model")
    return None


def get_default_symbols(raw: dict, base_currency: str) -> List[str]:
    """Get symbols from portfolio config + allowed explores."""
    port = raw.get("portfolio", {}) or {}
    core = list((port.get("core_targets") or {}).keys())
    explore = port.get("allowed_explore_assets") or []

    syms = set()
    for asset in core + explore:
        syms.add(f"{canonical_asset(asset)}/{base_currency}")

    # Always include BTC and ETH
    syms.add(f"BTC/{base_currency}")
    syms.add(f"ETH/{base_currency}")

    return sorted(syms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kraken Trend Bot Backtester")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to backtest (e.g. BTC/USD ETH/USD)")
    parser.add_argument("--timeframe", default=None, help="Candle timeframe (e.g. 15m)")
    parser.add_argument("--history-days", type=int, default=None,
                        help="Days of history to backtest")
    parser.add_argument("--capital", type=float, default=1000.0,
                        help="Initial capital in USD")
    parser.add_argument("--no-ml", action="store_true",
                        help="Run strategy-only (no ML predictions)")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison: ML vs no-ML")
    parser.add_argument("--fee-pct", type=float, default=0.26,
                        help="Trading fee percentage")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip OHLCV cache, always fetch from exchange")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    raw = load_config_from_yaml()
    cfg = build_config_from_yaml(raw, args)

    base_currency = (raw.get("exchange", {}) or {}).get("base_currency", "USD").upper()

    # Determine symbols
    if not cfg.symbols:
        cfg.symbols = get_default_symbols(raw, base_currency)

    # ── Smart timeframe / history validation ────────────────────────
    max_days = max_history_days_for_timeframe(cfg.timeframe)
    if cfg.history_days > max_days and not args.no_cache:
        # Check if cache can satisfy the request before warning
        cache_dir = DATA_DIR / "cache_ohlcv"
        sample_sym = cfg.symbols[0].replace("/", "_") if cfg.symbols else "BTC_USD"
        has_cache = any(cache_dir.glob(f"{sample_sym}__{cfg.timeframe}__*d.json"))
        if not has_cache:
            recommended = recommend_timeframe(cfg.history_days)
            print(f"⚠️  Kraken API only serves ~{KRAKEN_MAX_CANDLES} candles per timeframe.")
            print(f"   {cfg.timeframe} × 720 = ~{max_days} days (you requested {cfg.history_days}).")
            print(f"   Recommended: --timeframe {recommended} --history-days {min(cfg.history_days, max_history_days_for_timeframe(recommended))}")
            print(f"   Or use cached data from ML training (remove --no-cache).")
            print()

    # ML is now supported on any timeframe — per-timeframe models are
    # resolved in load_ml_forecaster().  If no model exists for the
    # chosen timeframe the user is given a training command.

    print("╔══════════════════════════════════════════════╗")
    print("║       KRAKEN TREND BOT — BACKTESTER          ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Timeframe:    {cfg.timeframe}")
    print(f"  History:      {cfg.history_days} days  (Kraken max ~{max_days}d for {cfg.timeframe})")
    print(f"  Capital:      ${cfg.initial_capital:,.2f}")
    print(f"  ML:           {'ON' if cfg.use_ml else 'OFF'}")
    print(f"  Symbols:      {', '.join(cfg.symbols)}")
    print(f"  Fee:          {cfg.fee_pct}%")
    print()

    # Load ML model (per-timeframe model resolution)
    forecaster = None
    if cfg.use_ml:
        forecaster = load_ml_forecaster(raw, timeframe=cfg.timeframe)
        if forecaster is None:
            print(f"[BT] ⚠️  ML enabled but no model available for {cfg.timeframe}.")
            print(f"     Train one with: python -m ml.train_torch_forecaster --timeframe {cfg.timeframe}")
            print(f"     Running strategy-only for now.")
            cfg.use_ml = False

    # Load data
    use_cache = not args.no_cache
    print(f"[BT] Loading OHLCV data... {'(cache enabled)' if use_cache else '(fresh fetch, cache disabled)'}")
    symbol_data: Dict[str, List[List[float]]] = {}
    for sym in cfg.symbols:
        data = load_ohlcv(sym, cfg.timeframe, cfg.history_days, use_cache=use_cache)
        if data and len(data) > 100:
            symbol_data[sym] = data
        else:
            n = len(data) if data else 0
            print(f"  [SKIP] {sym}: insufficient data ({n} candles)")

    if not symbol_data:
        print("[BT] ERROR: No usable data for any symbol. Exiting.")
        return

    # Load BTC data for cross-asset beta feature
    btc_sym = f"BTC/{base_currency}"
    btc_ohlcv = symbol_data.get(btc_sym)
    if btc_ohlcv is None:
        btc_ohlcv_raw = load_ohlcv(btc_sym, cfg.timeframe, cfg.history_days, use_cache=use_cache)
        btc_ohlcv = btc_ohlcv_raw if btc_ohlcv_raw and len(btc_ohlcv_raw) > 100 else None

    # Run backtest
    if args.compare:
        run_comparison(symbol_data, btc_ohlcv, cfg, forecaster)
    else:
        engine = BacktestEngine(cfg, forecaster=forecaster)
        result = engine.run(symbol_data, btc_ohlcv=btc_ohlcv)
        report = format_report(result)
        print(report)

        # Save
        tag = "ml" if cfg.use_ml else "no_ml"
        path = save_results(result, tag=tag)
        print(f"\n[BT] Report saved to: {path}")


if __name__ == "__main__":
    main()
