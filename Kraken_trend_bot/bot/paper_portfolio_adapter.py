# bot/paper_portfolio_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Set

from .paper_portfolio import PaperPortfolio, Position


@dataclass
class PaperPositionView:
    symbol: str
    quote: str


class PaperPortfolioAdapter:
    """
    Adapter so TopNTradingEngine can run in paper mode using PaperPortfolio.
    """

    def __init__(self, exchange, portfolio: PaperPortfolio, base_currency: str = "USDT"):
        self.exchange = exchange
        self.p = portfolio
        self.base_currency = base_currency

    def refresh(self) -> None:
        # No-op for paper
        return

    def positions(self) -> Dict[str, PaperPositionView]:
        out: Dict[str, PaperPositionView] = {}
        for sym, pos in (self.p.positions or {}).items():
            quote = sym.split("/", 1)[1] if "/" in sym else self.base_currency
            out[sym] = PaperPositionView(symbol=sym, quote=quote)
        return out

    def open_positions_count(self, acceptable_quotes: Optional[Set[str]] = None) -> int:
        if acceptable_quotes is None:
            return len(self.p.positions or {})
        return sum(
            1 for sym in (self.p.positions or {}).keys()
            if (sym.split("/", 1)[1] if "/" in sym else self.base_currency) in acceptable_quotes
        )

    def get_cash_available(self) -> float:
        return float(self.p.balances.get(self.base_currency, 0.0) or 0.0)

    def get_position_qty_free(self, symbol: str) -> float:
        pos: Position | None = self.p.positions.get(symbol)
        if not pos:
            return 0.0
        return float(pos.size)

    def estimate_equity_in_base(self) -> float:
        return float(self.p.value_in_base(self.exchange))

    def buy(self, symbol: str, qty: float, price: float) -> None:
        self.p.buy(symbol, qty, price)

    def sell(self, symbol: str, qty: float, price: float) -> None:
        self.p.sell(symbol, qty, price)
