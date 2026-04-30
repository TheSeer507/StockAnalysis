# bot/risk_manager.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskConfig:
    risk_per_trade_pct: float
    max_position_pct: float
    max_trade_notional_usdt: float = 0.0  # 0 = no extra cap
    min_equity_usdt: float = 0.0          # 0 = no minimum
    max_open_positions: int = 5           # 0 = unlimited


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config

    def compute_position_size(
        self,
        equity_usd: float,
        price: float,
        atr_value: Optional[float] = None,
    ) -> float:
        """
        Return position size in coin units.

        - Bound by risk_per_trade_pct of equity.
        - Bound by max_position_pct of equity.
        - Bound by max_trade_notional_usdt if > 0.
        - If equity < min_equity_usdt, returns 0.
        """
        cfg = self.config

        if equity_usd <= 0 or price <= 0:
            return 0.0

        if cfg.min_equity_usdt > 0 and equity_usd < cfg.min_equity_usdt:
            # Too small account, skip trading
            return 0.0

        risk_notional = equity_usd * (cfg.risk_per_trade_pct / 100.0)
        max_pos_notional = equity_usd * (cfg.max_position_pct / 100.0)

        notional = min(risk_notional, max_pos_notional)

        if cfg.max_trade_notional_usdt > 0:
            notional = min(notional, cfg.max_trade_notional_usdt)

        if notional <= 0:
            return 0.0

        size = notional / price
        return size
