# bot/portfolio_scanner.py
from __future__ import annotations

import re
from typing import List, Dict, Any, Tuple

from .data_handler import ohlcv_to_dataframe


def _strip_asset_candidates(asset: str) -> List[str]:
    candidates = [asset]

    if "." in asset:
        left = asset.split(".", 1)[0]
        candidates.append(left)
        nodigits = re.sub(r"\d+$", "", left)
        if nodigits and nodigits != left:
            candidates.append(nodigits)

    nodigits2 = re.sub(r"\d+$", "", asset)
    if nodigits2 and nodigits2 != asset:
        candidates.append(nodigits2)

    out = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _fmt_price(p: float) -> str:
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:.4f}"
    if p >= 0.01:
        return f"{p:.6f}"
    return f"{p:.8f}"


def scan_portfolio_signals(
    exchange,
    strategy,
    timeframe: str,
    min_candles: int,
    quote_candidates: Tuple[str, ...] = ("USDT", "USD"),
    min_balance: float = 0.0001,
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts with:
      asset_original, asset_resolved, symbol, qty, last_price, trend, signal, chg24h_pct
    """
    balance = exchange.get_full_balance()
    total = balance.get("total", {})

    results: List[Dict[str, Any]] = []
    skip_assets = {"USD", "USDT", "EUR", "GBP", "USDG", "USDC"}

    # ── Merge staked + free balances under canonical asset names ──
    merged: Dict[str, Dict[str, float]] = {}   # {"SOL": {"total": X, "free": Y}}
    for asset, amount in total.items():
        try:
            qty = float(amount)
        except (TypeError, ValueError):
            continue
        if qty < min_balance:
            continue
        # Determine canonical name (strips .S suffix)
        candidates = _strip_asset_candidates(asset)
        canonical = candidates[-1] if candidates else asset  # last is most stripped
        if canonical in skip_assets:
            continue
        entry = merged.setdefault(canonical, {"total": 0.0, "free": 0.0})
        entry["total"] += qty
        if "." not in asset:       # only count un-staked as free / tradable
            entry["free"] += qty

    for canonical, amounts in merged.items():
        qty = amounts["total"]
        free_qty = amounts["free"]
        if qty < min_balance:
            continue

        symbol = None
        asset_resolved = None

        for cand in _strip_asset_candidates(canonical):
            sym = exchange.resolve_symbol(cand, quote_candidates=quote_candidates)
            if sym:
                symbol = sym
                asset_resolved = cand
                break

        if symbol is None:
            if getattr(exchange, "verbose", False):
                print(f"[PORTFOLIO] No market found for asset '{canonical}', skipping.")
            continue

        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=min_candles)
            df = ohlcv_to_dataframe(ohlcv)
            if df.empty:
                continue

            # Strategy analysis (trend + event signal)
            analysis = strategy.evaluate(df)
            last_close = float(df["close"].iloc[-1])

            # 24h change based on last 24 candles if possible (for 1h timeframe)
            chg24 = None
            if len(df) >= 24:
                start = float(df["close"].iloc[-24])
                end = float(df["close"].iloc[-1])
                if start != 0:
                    chg24 = (end / start - 1.0) * 100.0

            staked_qty = qty - free_qty
            results.append({
                "asset_original": canonical,
                "asset_resolved": asset_resolved,
                "symbol": symbol,
                "qty": qty,
                "free_qty": free_qty,
                "staked_qty": staked_qty,
                "last_price": last_close,
                "last_price_fmt": _fmt_price(last_close),
                "trend": analysis.trend,
                "signal": analysis.signal,
                "chg24h_pct": chg24,
            })

        except Exception as e:
            print(f"[WARN] Could not evaluate {asset} ({symbol}): {e}")

    return results
