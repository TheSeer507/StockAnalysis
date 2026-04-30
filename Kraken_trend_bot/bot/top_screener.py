# bot/top_screener.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Tuple, List
import time

from data_handler import ohlcv_to_dataframe


@dataclass
class Recommendation:
    symbol: str
    last_price: float
    price_change_pct_24h: float
    trend: str
    signal: str
    quote_volume: float
    already_holding: bool

    # ML outputs (optional)
    ml_tp50: Optional[float] = None
    ml_tp80: Optional[float] = None
    ml_tp90: Optional[float] = None
    ml_ret50: Optional[float] = None
    ml_ret80: Optional[float] = None
    ml_ret90: Optional[float] = None


def _is_spot_symbol(exchange, symbol: str) -> bool:
    m = getattr(exchange, "markets", {}) or {}
    mm = m.get(symbol) or {}
    if "spot" in mm:
        return bool(mm.get("spot"))
    # fallback: if not marked as swap/future/option
    return not (mm.get("swap") or mm.get("future") or mm.get("option"))


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def find_top_recommendations(
    exchange,
    strategy,
    timeframe: str,
    bars_24h: int,
    top_n_volume: int,
    recommendations_count: int,
    quote_filter: Tuple[str, ...],
    held_bases: Iterable[str] = (),
    forecaster=None,
    ml_lookback: int = 64,
    ml_horizon_bars: int = 96,
    min_candles: int = 120,
    max_24h_change_pct: Optional[float] = None,
) -> List[Recommendation]:
    """
    Returns top-N recommendations based on:
      - trend/signal from strategy
      - ML predicted TP targets (if forecaster provided)
    """
    held = {b.upper() for b in held_bases}

    # 1) get tickers and pick by quote volume
    tickers: Dict[str, dict] = exchange.fetch_tickers() or {}
    candidates = []
    for sym, t in tickers.items():
        if "/" not in sym:
            continue
        base, quote = sym.split("/", 1)
        if quote.upper() not in {q.upper() for q in quote_filter}:
            continue
        if not _is_spot_symbol(exchange, sym):
            continue

        qv = _safe_float(t.get("quoteVolume") or t.get("quote_volume") or 0.0, 0.0)
        if qv <= 0:
            continue
        candidates.append((sym, qv, base.upper()))

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:max(10, int(top_n_volume))]

    recos: List[Recommendation] = []

    # 2) evaluate each candidate
    for sym, qv, base in candidates:
        try:
            ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=max(min_candles, bars_24h))
            if not ohlcv or len(ohlcv) < max(30, bars_24h):
                continue

            df = ohlcv_to_dataframe(ohlcv)
            if df is None or df.empty:
                continue

            last = _safe_float(df["close"].iloc[-1], 0.0)
            first = _safe_float(df["close"].iloc[-bars_24h], 0.0) if len(df) >= bars_24h else _safe_float(df["close"].iloc[0], 0.0)
            chg = ((last / first) - 1.0) * 100.0 if first > 0 else 0.0

            if max_24h_change_pct is not None and abs(chg) > float(max_24h_change_pct):
                # optional anti-chase filter
                continue

            analysis = strategy.evaluate(df)
            trend = getattr(analysis, "trend", "UNK")
            signal = getattr(analysis, "signal", "HOLD")

            already_holding = (base in held)

            r = Recommendation(
                symbol=sym,
                last_price=last,
                price_change_pct_24h=chg,
                trend=str(trend),
                signal=str(signal),
                quote_volume=qv,
                already_holding=already_holding,
            )

            # 3) ML TP prediction (optional)
            if forecaster is not None:
                pred = forecaster.predict(ohlcv, lookback=ml_lookback, horizon_bars=ml_horizon_bars)
                if pred is not None:
                    # quantiles typically 0.5/0.8/0.9
                    r.ml_tp50 = pred.tp_prices.get(0.5)
                    r.ml_tp80 = pred.tp_prices.get(0.8)
                    r.ml_tp90 = pred.tp_prices.get(0.9)
                    r.ml_ret50 = pred.mfe_returns.get(0.5)
                    r.ml_ret80 = pred.mfe_returns.get(0.8)
                    r.ml_ret90 = pred.mfe_returns.get(0.9)

            recos.append(r)

        except Exception:
            # keep it resilient; skip symbol
            continue

        # Avoid hammering Kraken too fast
        time.sleep(0.05)

    if not recos:
        return []

    # 4) Rank: prefer higher predicted TP80 return if present; else use 24h change
    def score(x: Recommendation) -> float:
        ml = x.ml_ret80 if x.ml_ret80 is not None else None
        if ml is not None:
            return (ml * 100.0) + (0.000000001 * x.quote_volume)  # ML dominates; volume tie-break
        return (x.price_change_pct_24h) + (0.000000001 * x.quote_volume)

    recos.sort(key=score, reverse=True)
    return recos[: max(1, int(recommendations_count))]
