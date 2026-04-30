# bot/paper_portfolio.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class Position:
    symbol: str
    base_asset: str
    quote_asset: str
    size: float
    entry_price: float
    opened_at: str
    imported: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        # Backward-compatible defaults
        return cls(
            symbol=data["symbol"],
            base_asset=data["base_asset"],
            quote_asset=data["quote_asset"],
            size=float(data["size"]),
            entry_price=float(data["entry_price"]),
            opened_at=data["opened_at"],
            imported=bool(data.get("imported", False)),
        )


@dataclass
class PaperPortfolio:
    base_currency: str = "USDT"
    balances: Dict[str, float] = field(default_factory=dict)
    positions: Dict[str, Position] = field(default_factory=dict)  # key = symbol
    realized_pnl: float = 0.0
    trade_history: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path, base_currency: str, initial_equity: float) -> "PaperPortfolio":
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)

            balances = {k: float(v) for k, v in raw.get("balances", {}).items()}
            positions_data = raw.get("positions", {})
            positions = {sym: Position.from_dict(p) for sym, p in positions_data.items()}

            return cls(
                base_currency=raw.get("base_currency", base_currency),
                balances=balances,
                positions=positions,
                realized_pnl=float(raw.get("realized_pnl", 0.0)),
                trade_history=raw.get("trade_history", []),
            )

        return cls(
            base_currency=base_currency,
            balances={base_currency: float(initial_equity)},
            positions={},
            realized_pnl=0.0,
            trade_history=[],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "base_currency": self.base_currency,
            "balances": self.balances,
            "positions": {sym: pos.to_dict() for sym, pos in self.positions.items()},
            "realized_pnl": self.realized_pnl,
            "trade_history": self.trade_history,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def open_positions_count(self) -> int:
        return len(self.positions)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def _split_symbol(self, symbol: str) -> tuple[str, str]:
        if "/" in symbol:
            base, quote = symbol.split("/", 1)
        else:
            base, quote = symbol, self.base_currency
        return base, quote

    # ------------------ Paper trade ops ------------------

    def buy(self, symbol: str, size: float, price: float) -> None:
        base, quote = self._split_symbol(symbol)
        notional = size * price
        ts = datetime.utcnow().isoformat()

        self.balances.setdefault(self.base_currency, 0.0)
        self.balances.setdefault(base, 0.0)

        self.balances[self.base_currency] -= notional
        self.balances[base] += size

        if symbol in self.positions:
            pos = self.positions[symbol]
            total_size = pos.size + size
            pos.entry_price = (pos.entry_price * pos.size + price * size) / total_size
            pos.size = total_size
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                base_asset=base,
                quote_asset=quote,
                size=size,
                entry_price=price,
                opened_at=ts,
                imported=False,
            )

        self.trade_history.append({
            "timestamp": ts,
            "side": "BUY",
            "symbol": symbol,
            "size": size,
            "price": price,
            "notional": notional,
        })

    def sell(self, symbol: str, size: float, price: float) -> float:
        base, quote = self._split_symbol(symbol)
        ts = datetime.utcnow().isoformat()

        pos = self.positions.get(symbol)
        if pos is None or size <= 0:
            return 0.0

        if size > pos.size:
            size = pos.size

        notional = size * price
        pnl = (price - pos.entry_price) * size
        self.realized_pnl += pnl

        self.balances.setdefault(self.base_currency, 0.0)
        self.balances.setdefault(base, 0.0)

        self.balances[self.base_currency] += notional
        self.balances[base] -= size

        pos.size -= size
        if pos.size <= 1e-12:
            self.positions.pop(symbol, None)

        self.trade_history.append({
            "timestamp": ts,
            "side": "SELL",
            "symbol": symbol,
            "size": size,
            "price": price,
            "notional": notional,
            "pnl": pnl,
        })
        return pnl

    # ------------------ Pricing / equity ------------------

    def value_in_base(self, exchange) -> float:
        equity = 0.0
        for asset, amount in self.balances.items():
            if amount <= 0:
                continue
            if asset == self.base_currency:
                equity += amount
                continue

            symbol = exchange.resolve_symbol(asset, quote_candidates=(self.base_currency, "USD", "USDT"))
            if not symbol:
                continue

            try:
                ticker = exchange.fetch_ticker(symbol)
                price = ticker.get("last")
                if price is None:
                    continue
                equity += float(amount) * float(price)
            except Exception:
                continue

        return equity

    # ------------------ Bootstrap from Kraken balances ------------------

    @staticmethod
    def _strip_suffixes(asset: str) -> List[str]:
        """
        Build candidates to resolve markets for Kraken assets like:
          ETH2.S -> ETH2 -> ETH
          SOL03.S -> SOL03 -> SOL
          BTC.B -> BTC
        """
        candidates = []

        # Original
        candidates.append(asset)

        # Remove ".S"/".B"/".M" suffix block
        if "." in asset:
            left = asset.split(".", 1)[0]
            candidates.append(left)

            # Remove trailing digits if present (SOL03 -> SOL, ETH2 -> ETH, ATOM21 -> ATOM)
            nodigits = re.sub(r"\d+$", "", left)
            if nodigits and nodigits != left:
                candidates.append(nodigits)

        # Also try removing trailing digits for assets without dot (ATOM21)
        nodigits2 = re.sub(r"\d+$", "", asset)
        if nodigits2 and nodigits2 != asset:
            candidates.append(nodigits2)

        # Deduplicate while preserving order
        out = []
        seen = set()
        for c in candidates:
            if c and c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def bootstrap_from_exchange(
        self,
        exchange,
        quote_candidates: Tuple[str, ...] = ("USDT", "USD"),
        min_balance: float = 0.00000001,
        create_positions: bool = True,
    ) -> None:
        """
        Import your real Kraken balances into the paper portfolio.
        For each asset > min_balance, we copy it to balances.
        Optionally create a paper 'Position' with entry_price = current last price
        (so PnL starts at ~0 from import time).
        """
        bal = exchange.get_full_balance()
        totals = bal.get("total", {})

        # Reset portfolio to reflect exchange snapshot (keeps realized pnl + history)
        # If you prefer merging, remove these two lines.
        self.balances = {}
        self.positions = {}

        ts = datetime.utcnow().isoformat()

        for asset, amount in totals.items():
            try:
                amt = float(amount)
            except (TypeError, ValueError):
                continue
            if amt <= min_balance:
                continue

            self.balances[asset] = amt

        # Ensure base currency key exists
        self.balances.setdefault(self.base_currency, float(totals.get(self.base_currency, 0.0) or 0.0))

        if not create_positions:
            self.trade_history.append({"timestamp": ts, "side": "IMPORT", "note": "Imported balances only"})
            return

        # Create positions for non-stable assets that have a resolvable market
        skip_assets = {"USD", "USDT", "EUR", "GBP", "USDG", "USDC"}

        for asset, amt in list(self.balances.items()):
            if asset in skip_assets or asset == self.base_currency:
                continue

            # Resolve a tradable symbol for this asset (trying suffix-stripped variants)
            symbol = None
            base_used = None

            for candidate in self._strip_suffixes(asset):
                symbol = exchange.resolve_symbol(candidate, quote_candidates=quote_candidates)
                if symbol:
                    base_used = candidate
                    break

            if not symbol or not base_used:
                continue

            try:
                ticker = exchange.fetch_ticker(symbol)
                last = ticker.get("last")
                if last is None:
                    continue
                last = float(last)
                if last <= 0:
                    continue
            except Exception:
                continue

            base, quote = self._split_symbol(symbol)
            size = float(amt) if base_used == base else float(amt)  # size means base units; good enough for import

            self.positions[symbol] = Position(
                symbol=symbol,
                base_asset=base,
                quote_asset=quote,
                size=size,
                entry_price=last,
                opened_at=ts,
                imported=True,
            )

        self.trade_history.append({
            "timestamp": ts,
            "side": "IMPORT",
            "note": "Imported balances and created positions with entry_price = current price",
        })
