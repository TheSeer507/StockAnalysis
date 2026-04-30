from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .indicators import ema, atr


@dataclass
class StrategyConfig:
    ema_fast: int
    ema_slow: int
    atr_period: int


@dataclass
class StrategyResult:
    signal: str               # "LONG", "EXIT", "HOLD"
    trend: str                # "BULL", "BEAR", "FLAT"
    ema_fast: Optional[float]
    ema_slow: Optional[float]
    atr: Optional[float]
    price_change_pct: Optional[float]


class TrendStrategy:
    def __init__(self, config: StrategyConfig):
        self.config = config

    def evaluate(self, df: pd.DataFrame) -> StrategyResult:
        """
        Evaluate the trend for a given OHLCV dataframe.

        df index: timestamp
        df columns: open, high, low, close, volume

        Returns a StrategyResult with:
        - signal: event-based ("LONG"/"EXIT"/"HOLD")
        - trend: state-based ("BULL"/"BEAR"/"FLAT")
        - ema_fast / ema_slow: last EMA values
        - atr: last ATR value
        - price_change_pct: % change over the window (close_last vs close_first)
        """
        # Defaults if we don't have enough data
        if df is None or df.empty:
            return StrategyResult(
                signal="HOLD",
                trend="FLAT",
                ema_fast=None,
                ema_slow=None,
                atr=None,
                price_change_pct=None,
            )

        if len(df) < max(self.config.ema_fast, self.config.ema_slow) + 2:
            # Not enough bars to compute EMAs reliably
            close_first = float(df["close"].iloc[0])
            close_last = float(df["close"].iloc[-1])
            price_change_pct = (close_last / close_first - 1.0) * 100.0 if close_first != 0 else None
            return StrategyResult(
                signal="HOLD",
                trend="FLAT",
                ema_fast=None,
                ema_slow=None,
                atr=None,
                price_change_pct=price_change_pct,
            )

        df = df.copy()
        df["ema_fast"] = ema(df["close"], self.config.ema_fast)
        df["ema_slow"] = ema(df["close"], self.config.ema_slow)
        df["atr"] = atr(df, self.config.atr_period)

        fast_prev, fast_curr = df["ema_fast"].iloc[-2], df["ema_fast"].iloc[-1]
        slow_prev, slow_curr = df["ema_slow"].iloc[-2], df["ema_slow"].iloc[-1]
        atr_curr = df["atr"].iloc[-1]

        # Trend state based on current EMAs
        if fast_curr > slow_curr:
            trend = "BULL"
        elif fast_curr < slow_curr:
            trend = "BEAR"
            # otherwise
        else:
            trend = "FLAT"

        # Event-based signal (crossover on the latest bar)
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            signal = "LONG"
        elif fast_prev >= slow_prev and fast_curr < slow_curr:
            signal = "EXIT"
        else:
            signal = "HOLD"

        close_first = float(df["close"].iloc[0])
        close_last = float(df["close"].iloc[-1])
        price_change_pct = (close_last / close_first - 1.0) * 100.0 if close_first != 0 else None

        return StrategyResult(
            signal=signal,
            trend=trend,
            ema_fast=float(fast_curr),
            ema_slow=float(slow_curr),
            atr=float(atr_curr) if pd.notna(atr_curr) else None,
            price_change_pct=price_change_pct,
        )

    def generate_signal(self, df: pd.DataFrame) -> str:
        """
        Backwards-compatible API: only return the event-based signal.
        """
        result = self.evaluate(df)
        return result.signal
