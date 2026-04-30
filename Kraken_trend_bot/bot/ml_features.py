# bot/ml_features.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns: timestamp, open, high, low, close, volume
    Output: feature DataFrame aligned to df index (no future leakage).
    Enhanced with additional predictive features.
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    open_ = df["open"].astype(float)
    vol   = df["volume"].astype(float)

    # Returns / momentum (multiple timeframes)
    out["ret_1"]  = close.pct_change(1)
    out["ret_2"]  = close.pct_change(2)
    out["ret_4"]  = close.pct_change(4)
    out["ret_8"]  = close.pct_change(8)
    out["ret_16"] = close.pct_change(16)
    out["ret_32"] = close.pct_change(32)

    # EMAs (multiple timeframes)
    ema_8  = close.ewm(span=8, adjust=False).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_55 = close.ewm(span=55, adjust=False).mean()
    ema_89 = close.ewm(span=89, adjust=False).mean()

    out["ema8_dist"] = (close - ema_8) / close
    out["ema12_dist"] = (close - ema_12) / close
    out["ema21_dist"] = (close - ema_21) / close
    out["ema55_dist"] = (close - ema_55) / close
    out["ema89_dist"] = (close - ema_89) / close
    out["ema21_over_55"] = (ema_21 / ema_55) - 1.0
    out["ema8_over_21"] = (ema_8 / ema_21) - 1.0

    # ATR / volatility regime
    atr14 = _atr(df, 14)
    atr28 = _atr(df, 28)
    out["atr_pct"] = atr14 / close
    out["atr_ratio"] = atr14 / (atr28 + 1e-8)  # volatility acceleration

    # RSI (multiple timeframes)
    out["rsi14"] = _rsi(close, 14)
    out["rsi28"] = _rsi(close, 28)

    # Candle structure
    rng = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    out["range_pct"] = (high - low) / close
    out["body_to_range"] = (body / rng).fillna(0.0)
    
    # Wick analysis
    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    body_bottom = pd.concat([open_, close], axis=1).min(axis=1)
    out["upper_wick_pct"] = (high - body_top) / close
    out["lower_wick_pct"] = (body_bottom - low) / close

    # Volume features
    vmean = vol.rolling(50).mean()
    vstd  = vol.rolling(50).std().replace(0, np.nan)
    out["vol_z"] = ((vol - vmean) / vstd).fillna(0.0)
    
    vol_ema10 = vol.ewm(span=10, adjust=False).mean()
    out["vol_ema_ratio"] = vol / (vol_ema10 + 1e-8)
    
    # Price position relative to high/low
    high20 = high.rolling(20).max()
    low20 = low.rolling(20).min()
    out["price_position"] = ((close - low20) / (high20 - low20 + 1e-8)).fillna(0.5)

    # Volatility of returns
    out["ret_std_10"] = out["ret_1"].rolling(10).std().fillna(0)
    out["ret_std_20"] = out["ret_1"].rolling(20).std().fillna(0)
    
    # Price acceleration
    out["momentum_3"] = close.pct_change(3)
    out["momentum_7"] = close.pct_change(7)
    out["accel"] = out["momentum_3"] - out["momentum_3"].shift(3)

    # Order flow proxy: (close - low) / (high - low) — 1.0 = strong buying
    hl_range = (high - low).replace(0, np.nan)
    out["order_flow"] = ((close - low) / hl_range).fillna(0.5)

    # Bollinger %B: (close - bb_lower) / (bb_upper - bb_lower)
    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_sma + 2.0 * bb_std
    bb_lower = bb_sma - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    out["bollinger_pctb"] = ((close - bb_lower) / bb_width).fillna(0.5)

    # VWAP distance: (close - vwap20) / close
    cum_pv = (close * vol).rolling(20).sum()
    cum_v = vol.rolling(20).sum().replace(0, np.nan)
    vwap_20 = cum_pv / cum_v
    out["vwap_distance"] = ((close - vwap_20) / close).fillna(0.0)

    # BTC beta slot (placeholder — filled as 0; training pipeline can inject actual BTC returns)
    out["btc_beta"] = 0.0

    # Cleanup
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
