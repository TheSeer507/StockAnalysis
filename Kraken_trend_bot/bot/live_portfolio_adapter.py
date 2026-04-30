# bot/live_portfolio_adapter.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Set


STABLE_ALIASES = {"USD", "USDT", "USDC", "USDG", "EUR", "GBP"}


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

    out, seen = [], set()
    for c in candidates:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


@dataclass
class LivePosition:
    asset_original: str
    asset_resolved: str
    symbol: str
    qty_free: float
    qty_total: float
    quote: str


class LivePortfolioAdapter:
    def __init__(
        self,
        exchange,
        base_currency: str = "USD",
        quote_candidates: Tuple[str, ...] = ("USD", "USDT"),
        min_balance: float = 1e-10,
        skip_dot_assets: bool = True,
        only_spot: bool = True,
    ):
        self.exchange = exchange
        self.base_currency = base_currency
        self.quote_candidates = quote_candidates
        self.min_balance = min_balance
        self.skip_dot_assets = skip_dot_assets
        self.only_spot = only_spot

        self._balance_raw: Dict[str, Any] = {}
        self._positions: Dict[str, LivePosition] = {}

    def refresh(self) -> None:
        self._balance_raw = self.exchange.get_full_balance()
        free = self._balance_raw.get("free", {}) or {}
        total = self._balance_raw.get("total", {}) or {}

        markets = getattr(self.exchange, "markets", {}) or {}
        positions: Dict[str, LivePosition] = {}

        for asset, qty_total in total.items():
            try:
                qt = float(qty_total)
            except (TypeError, ValueError):
                continue
            if qt <= self.min_balance:
                continue

            if asset in STABLE_ALIASES:
                continue

            # Avoid staking/earn/bond assets like ATOM21.S, ETH2.S, BTC.B, BTC.M
            if self.skip_dot_assets and "." in asset:
                continue

            qf = float(free.get(asset, 0.0) or 0.0)

            symbol = None
            resolved = None
            for cand in _strip_asset_candidates(asset):
                sym = self.exchange.resolve_symbol(cand, quote_candidates=self.quote_candidates)
                if sym:
                    m = markets.get(sym) or {}
                    if self.only_spot and ("spot" in m) and (not m.get("spot", False)):
                        continue
                    symbol = sym
                    resolved = cand
                    break

            if not symbol or not resolved:
                continue

            quote = symbol.split("/", 1)[1] if "/" in symbol else self.base_currency
            positions[symbol] = LivePosition(
                asset_original=asset,
                asset_resolved=resolved,
                symbol=symbol,
                qty_free=qf,
                qty_total=qt,
                quote=quote,
            )

        self._positions = positions

    def positions(self) -> Dict[str, LivePosition]:
        return dict(self._positions)

    def open_positions_count(self, acceptable_quotes: Optional[Set[str]] = None) -> int:
        if acceptable_quotes is None:
            return len(self._positions)
        return sum(1 for p in self._positions.values() if p.quote in acceptable_quotes)

    def get_cash_available(self) -> float:
        free = self._balance_raw.get("free", {}) or {}
        cash = float(free.get(self.base_currency, 0.0) or 0.0)

        # Treat USD/USDT as near-equivalent cash for sizing if base is USD/USDT
        if self.base_currency == "USD":
            cash += float(free.get("USDT", 0.0) or 0.0)
        elif self.base_currency == "USDT":
            cash += float(free.get("USD", 0.0) or 0.0)

        return cash

    def get_position_qty_free(self, symbol: str) -> float:
        p = self._positions.get(symbol)
        return float(p.qty_free) if p else 0.0

    def estimate_equity_in_base(self, quote_for_pricing: Optional[Tuple[str, ...]] = None) -> float:
        if quote_for_pricing is None:
            quote_for_pricing = (self.base_currency, "USD", "USDT")

        total = (self._balance_raw.get("total", {}) or {})
        equity = 0.0

        for asset, qty_total in total.items():
            try:
                qt = float(qty_total)
            except (TypeError, ValueError):
                continue
            if qt <= 0:
                continue

            if asset == self.base_currency:
                equity += qt
                continue

            if self.base_currency in {"USD", "USDT"} and asset in {"USD", "USDT", "USDC", "USDG"}:
                equity += qt
                continue

            if self.skip_dot_assets and "." in asset:
                continue

            symbol = None
            for cand in _strip_asset_candidates(asset):
                s = self.exchange.resolve_symbol(cand, quote_candidates=quote_for_pricing)
                if s:
                    symbol = s
                    break
            if not symbol:
                continue

            try:
                t = self.exchange.fetch_ticker(symbol)
                last = t.get("last")
                if last is None:
                    continue
                equity += qt * float(last)
            except Exception:
                continue

        return equity
