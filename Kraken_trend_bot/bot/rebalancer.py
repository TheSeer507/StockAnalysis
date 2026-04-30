# bot/rebalancer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import math


@dataclass
class RebalanceAction:
    side: str          # "BUY" or "SELL"
    symbol: str        # e.g. "BTC/USD"
    asset: str         # e.g. "BTC"
    notional: float    # in base currency (USD)
    qty: float
    price: float
    reason: str


class PortfolioRebalancer:
    """
    Enforces:
      - hard stop losses by absolute USD levels (your rules)
      - core portfolio target weights (BTC/ETH/SOL/XMR)
      - stable cash floor (10%)
      - explore bucket cap (5%) for non-core assets

    Works with:
      - paper adapter (must have buy/sell methods)
      - live adapter (no buy/sell -> orders sent via exchange)
    """

    def __init__(
        self,
        exchange,
        portfolio_adapter,
        base_currency: str,
        core_targets: Dict[str, float],
        stable_target_pct: float,
        explore_target_pct: float,
        stable_assets: List[str],
        allowed_explore_assets: List[str],
        stops: Dict[str, float],
        band_pct: float = 0.02,
        max_trade_notional: float = 10.0,
        max_total_notional_per_cycle: float = 20.0,
        mode: str = "paper",
        debug: bool = True,
    ):
        self.exchange = exchange
        self.portfolio = portfolio_adapter
        self.base_currency = base_currency.upper()
        self.core_targets = {k.upper(): float(v) for k, v in core_targets.items()}

        self.stable_target_pct = float(stable_target_pct)
        self.explore_target_pct = float(explore_target_pct)

        self.stable_assets = [x.upper() for x in stable_assets]
        self.allowed_explore_assets = [x.upper() for x in allowed_explore_assets]

        self.stops = {k.upper(): float(v) for k, v in (stops or {}).items()}
        self.band_pct = float(band_pct)
        self.max_trade_notional = float(max_trade_notional)
        self.max_total_notional_per_cycle = float(max_total_notional_per_cycle)

        self.mode = mode.lower()
        self.debug = debug

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(msg)

    # ---------- market helpers ----------

    def _resolve_symbol(self, asset: str) -> Optional[str]:
        # Only trade in base quote
        try:
            return self.exchange.resolve_symbol(asset, quote_candidates=(self.base_currency,))
        except Exception:
            return None

    def _last_price(self, symbol: str) -> Optional[float]:
        try:
            t = self.exchange.fetch_ticker(symbol)
            last = t.get("last")
            return float(last) if last is not None else None
        except Exception as e:
            self._dbg(f"[REBAL][DEBUG] fetch_ticker failed {symbol}: {e}")
            return None

    def _amount_to_precision_down(self, symbol: str, amount: float) -> float:
        if amount <= 0:
            return 0.0
        fn = getattr(self.exchange, "amount_to_precision", None)
        if callable(fn):
            try:
                return float(fn(symbol, amount))
            except Exception:
                pass
        # fallback by market precision if present
        markets = getattr(self.exchange, "markets", {}) or {}
        m = markets.get(symbol) or {}
        dec = (m.get("precision") or {}).get("amount", None)
        if dec is None:
            return amount
        factor = 10 ** int(dec)
        return math.floor(amount * factor) / factor

    def _min_amount(self, symbol: str) -> Optional[float]:
        markets = getattr(self.exchange, "markets", {}) or {}
        m = markets.get(symbol) or {}
        v = ((m.get("limits") or {}).get("amount") or {}).get("min")
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    def _min_cost(self, symbol: str) -> Optional[float]:
        markets = getattr(self.exchange, "markets", {}) or {}
        m = markets.get(symbol) or {}
        v = ((m.get("limits") or {}).get("cost") or {}).get("min")
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    def _normalize_qty(self, symbol: str, qty: float, price: float) -> float:
        qty = float(qty)
        if qty <= 0:
            return 0.0
        qty = self._amount_to_precision_down(symbol, qty)
        mn = self._min_amount(symbol)
        if mn and qty < mn:
            return 0.0
        mc = self._min_cost(symbol)
        if mc and qty * price < mc:
            return 0.0
        return qty

    # ---------- valuation ----------

    def _stable_value(self) -> float:
        """
        Uses available cash from adapter as 'stable'. For LIVE adapter it may include USDT too.
        Good enough for allocation guardrails.
        """
        try:
            return float(self.portfolio.get_cash_available())
        except Exception:
            return 0.0

    def _positions_value_map(self) -> Dict[str, float]:
        """
        Returns map of BASE ASSET -> value in base currency, for non-stable tradable positions.
        Uses FREE qty for live, position qty for paper.
        """
        out: Dict[str, float] = {}

        self.portfolio.refresh()
        pos = self.portfolio.positions()

        for symbol, p in pos.items():
            # symbol looks like "ETH/USD"
            base = symbol.split("/", 1)[0].upper()
            if base in self.stable_assets:
                continue

            qty = float(self.portfolio.get_position_qty_free(symbol))
            if qty <= 0:
                continue

            price = self._last_price(symbol)
            if not price:
                continue

            out[base] = out.get(base, 0.0) + qty * price

        return out

    def equity(self) -> float:
        self.portfolio.refresh()
        return float(self.portfolio.estimate_equity_in_base())

    def current_explore_value(self) -> float:
        """
        Explore = anything not in core_targets and not stable.
        """
        vals = self._positions_value_map()
        explore = 0.0
        for asset, v in vals.items():
            if asset not in self.core_targets:
                explore += v
        return explore

    # ---------- trading ----------

    def _execute_buy(self, symbol: str, qty: float, price: float, reason: str, execute_trades: bool) -> bool:
        qty = self._normalize_qty(symbol, qty, price)
        if qty <= 0:
            self._dbg(f"[REBAL][SKIP] BUY {symbol}: qty normalized to 0 (min/precision/cost)")
            return False

        if not execute_trades:
            self._dbg(f"[REBAL][BLOCK] BUY {symbol}: execute_trades=False | qty={qty:.8f} notional≈{qty*price:.2f}")
            return False

        try:
            if self.mode == "paper":
                self.portfolio.buy(symbol, qty, price)
            else:
                self.exchange.create_market_buy_order(symbol, qty)
            print(f"[REBAL][BUY] {symbol} qty={qty:.8f} price≈{price:.4f} reason={reason}")
            return True
        except Exception as e:
            msg = str(e)
            if "Insufficient funds" in msg or "EOrder:Insufficient funds" in msg:
                print(f"[REBAL][WARN] BUY skipped {symbol}: insufficient funds")
                return False
            print(f"[REBAL][WARN] BUY failed {symbol}: {e}")
            return False

    def _execute_sell(self, symbol: str, qty: float, price: float, reason: str, execute_trades: bool) -> bool:
        # clamp to free qty
        try:
            self.portfolio.refresh()
            avail = float(self.portfolio.get_position_qty_free(symbol))
            qty = min(float(qty), avail) * 0.995
        except Exception:
            qty = float(qty) * 0.995

        qty = self._normalize_qty(symbol, qty, price)
        if qty <= 0:
            self._dbg(f"[REBAL][SKIP] SELL {symbol}: qty normalized to 0")
            return False

        if not execute_trades:
            self._dbg(f"[REBAL][BLOCK] SELL {symbol}: execute_trades=False | qty={qty:.8f} notional≈{qty*price:.2f}")
            return False

        try:
            if self.mode == "paper":
                self.portfolio.sell(symbol, qty, price)
            else:
                self.exchange.create_market_sell_order(symbol, qty)
            print(f"[REBAL][SELL] {symbol} qty={qty:.8f} price≈{price:.4f} reason={reason}")
            return True
        except Exception as e:
            msg = str(e)
            if "Insufficient funds" in msg or "EOrder:Insufficient funds" in msg:
                print(f"[REBAL][WARN] SELL skipped {symbol}: insufficient funds/free qty mismatch")
                return False
            print(f"[REBAL][WARN] SELL failed {symbol}: {e}")
            return False

    # ---------- rules ----------

    def enforce_stops(self, execute_trades: bool) -> int:
        """
        For each held asset with a stop: if last <= stop -> sell full free qty.
        """
        self.portfolio.refresh()
        sold = 0
        for symbol in list(self.portfolio.positions().keys()):
            base = symbol.split("/", 1)[0].upper()
            if base not in self.stops:
                continue
            stop_price = float(self.stops[base])
            last = self._last_price(symbol)
            if not last:
                continue

            if last <= stop_price:
                qty = float(self.portfolio.get_position_qty_free(symbol))
                if qty <= 0:
                    continue
                ok = self._execute_sell(symbol, qty, last, f"STOP {base} <= {stop_price}", execute_trades)
                if ok:
                    sold += 1
        return sold

    def rebalance_core(self, execute_trades: bool) -> List[RebalanceAction]:
        """
        Bring core targets toward desired weights with a band and trade caps.
        Uses available stable cash above the stable floor.
        """
        actions: List[RebalanceAction] = []

        eq = self.equity()
        if eq <= 0:
            return actions

        band_value = eq * self.band_pct
        stable_floor = eq * self.stable_target_pct

        # Current values
        stable_val = self._stable_value()
        vals = self._positions_value_map()  # base_asset -> value

        # compute deltas for core
        deltas: Dict[str, float] = {}
        for asset, w in self.core_targets.items():
            desired = eq * w
            current = vals.get(asset, 0.0)
            deltas[asset] = current - desired  # + overweight, - underweight

        self._dbg(f"[REBAL][DEBUG] equity≈{eq:.2f} stable≈{stable_val:.2f} stable_floor≈{stable_floor:.2f}")

        # 1) SELL overweight core assets first (raise cash)
        total_budget = self.max_total_notional_per_cycle
        for asset, dv in sorted(deltas.items(), key=lambda x: x[1], reverse=True):
            if total_budget <= 0:
                break
            if dv <= band_value:
                continue  # not overweight enough
            symbol = self._resolve_symbol(asset)
            if not symbol:
                continue
            price = self._last_price(symbol)
            if not price:
                continue
            qty_free = float(self.portfolio.get_position_qty_free(symbol))
            if qty_free <= 0:
                continue

            sell_notional = min(dv, self.max_trade_notional, total_budget)
            sell_qty = min(qty_free, sell_notional / price)

            ok = self._execute_sell(symbol, sell_qty, price, f"REB-SELL overweight {asset} (+{dv:.2f})", execute_trades)
            if ok:
                actions.append(RebalanceAction("SELL", symbol, asset, sell_qty * price, sell_qty, price, "overweight"))
                total_budget -= sell_qty * price

        # refresh cash after sells
        self.portfolio.refresh()
        eq = self.equity()
        stable_val = self._stable_value()
        spendable_cash = max(0.0, stable_val - (eq * self.stable_target_pct))

        # 2) BUY underweight core assets using spendable cash
        for asset, dv in sorted(deltas.items(), key=lambda x: x[1]):
            if total_budget <= 0 or spendable_cash <= 0:
                break
            if dv >= -band_value:
                continue  # not underweight enough

            symbol = self._resolve_symbol(asset)
            if not symbol:
                continue
            price = self._last_price(symbol)
            if not price:
                continue

            need = abs(dv)
            buy_notional = min(need, self.max_trade_notional, total_budget, spendable_cash)
            if buy_notional <= 0:
                continue
            buy_qty = buy_notional / price

            ok = self._execute_buy(symbol, buy_qty, price, f"REB-BUY underweight {asset} (-{abs(dv):.2f})", execute_trades)
            if ok:
                actions.append(RebalanceAction("BUY", symbol, asset, buy_notional, buy_qty, price, "underweight"))
                total_budget -= buy_notional
                spendable_cash -= buy_notional

        return actions

    def cap_explore_bucket(self, execute_trades: bool) -> int:
        """
        If explore value exceeds explore_target_pct (+band), sell down non-core, non-allowed assets first.
        """
        eq = self.equity()
        if eq <= 0:
            return 0
        limit = eq * self.explore_target_pct
        band = eq * self.band_pct

        vals = self._positions_value_map()
        explore_assets = [(a, v) for a, v in vals.items() if a not in self.core_targets]
        explore_val = sum(v for _, v in explore_assets)

        if explore_val <= limit + band:
            return 0

        # Sell assets not allowed first (largest value first)
        sold_count = 0
        for asset, v in sorted(explore_assets, key=lambda x: x[1], reverse=True):
            if explore_val <= limit:
                break
            if asset in self.allowed_explore_assets:
                continue
            symbol = self._resolve_symbol(asset)
            if not symbol:
                continue
            price = self._last_price(symbol)
            if not price:
                continue
            qty = float(self.portfolio.get_position_qty_free(symbol))
            if qty <= 0:
                continue

            # sell just enough to bring explore down, cap by trade notional
            excess = explore_val - limit
            sell_notional = min(excess, self.max_trade_notional)
            sell_qty = min(qty, sell_notional / price)

            ok = self._execute_sell(symbol, sell_qty, price, f"EXPLORE-CAP sell {asset} to keep explore≤{limit:.2f}", execute_trades)
            if ok:
                sold_count += 1
                explore_val -= sell_qty * price

        return sold_count
