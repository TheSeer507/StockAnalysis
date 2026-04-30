# bot/executor.py
from typing import Optional

from .exchange import ExchangeClient
from .risk_manager import RiskManager
from .paper_portfolio import PaperPortfolio


class Executor:
    def __init__(
        self,
        exchange: ExchangeClient,
        risk_manager: RiskManager,
        portfolio: PaperPortfolio,
        mode: str,
        base_currency: str,
    ):
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.portfolio = portfolio
        self.mode = mode.lower()
        self.base_currency = base_currency

    # ---------- internal helpers ----------

    def _get_equity(self) -> float:
        """
        Equity used for risk sizing:
          - paper mode: paper portfolio equity in base_currency
          - live mode: free balance in base_currency (conservative)
        """
        if self.mode == "paper":
            return self.portfolio.value_in_base(self.exchange)
        return self.exchange.get_balance(self.base_currency)

    def _get_available_base(self) -> float:
        """
        Cash available to spend in base_currency.
        """
        if self.mode == "paper":
            return self.portfolio.balances.get(self.base_currency, 0.0)
        return self.exchange.get_balance(self.base_currency)

    def _clamp_size_to_cash(self, size: float, price: float) -> float:
        if size <= 0 or price <= 0:
            return 0.0
        available_notional = self._get_available_base()
        if available_notional <= 0:
            return 0.0
        max_size_by_cash = available_notional / price
        if max_size_by_cash <= 0:
            return 0.0
        return min(size, max_size_by_cash)

    def _can_open_new_position(self) -> bool:
        cfg = self.risk_manager.config
        max_open = getattr(cfg, "max_open_positions", 0)
        if max_open <= 0:
            return True
        current_open = self.portfolio.open_positions_count()
        return current_open < max_open

    # ---------- public API ----------

    def execute_for_symbol(
        self,
        symbol: str,
        signal: str,
        last_price: float,
        atr_value: Optional[float] = None,
    ) -> None:
        """
        Handle LONG / EXIT / HOLD for a given symbol using paper or live trading.

        LONG: open a new long position if none exists.
        EXIT: close existing long position if present.
        HOLD: do nothing.
        """
        in_position = self.portfolio.has_position(symbol)
        equity = self._get_equity()

        if equity <= 0:
            print(f"[EXEC] No equity available, skipping execution for {symbol}.")
            return

        if signal == "LONG" and not in_position:
            if not self._can_open_new_position():
                print(f"[EXEC] Max open positions reached, skipping LONG for {symbol}.")
                return

            raw_size = self.risk_manager.compute_position_size(
                equity_usd=equity,
                price=last_price,
                atr_value=atr_value,
            )
            size = self._clamp_size_to_cash(raw_size, last_price)

            if size <= 0:
                print(f"[EXEC] Computed size is zero for {symbol}, skipping LONG.")
                return

            if self.mode == "paper":
                self.portfolio.buy(symbol=symbol, size=size, price=last_price)
                print(
                    f"[PAPER-EXEC] OPEN LONG {symbol} size={size:.8f} "
                    f"price={last_price:.4f}, equity_before={equity:.2f}"
                )
            else:
                self.exchange.create_market_buy_order(symbol, size)
                print(
                    f"[LIVE-EXEC] OPEN LONG {symbol} size={size:.8f} "
                    f"price={last_price:.4f}, equity_est={equity:.2f}"
                )

        elif signal == "EXIT" and in_position:
            pos = self.portfolio.get_position(symbol)
            if pos is None or pos.size <= 0:
                print(f"[EXEC] No valid position found for EXIT on {symbol}.")
                return

            size_to_sell = pos.size
            if self.mode == "paper":
                pnl = self.portfolio.sell(symbol=symbol, size=size_to_sell, price=last_price)
                print(
                    f"[PAPER-EXEC] EXIT LONG {symbol} size={size_to_sell:.8f} "
                    f"price={last_price:.4f}, trade_pnl={pnl:.4f}, "
                    f"total_realized_pnl={self.portfolio.realized_pnl:.4f}"
                )
            else:
                self.exchange.create_market_sell_order(symbol, size_to_sell)
                print(
                    f"[LIVE-EXEC] EXIT LONG {symbol} size={size_to_sell:.8f} "
                    f"price={last_price:.4f}"
                )
        else:
            # HOLD or no state change
            pass
