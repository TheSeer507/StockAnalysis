# bot/state.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PositionState:
    is_long: bool = False
    entry_price: Optional[float] = None
    size: float = 0.0
    last_signal: str = "HOLD"

    def enter_long(self, price: float, size: float):
        self.is_long = True
        self.entry_price = price
        self.size = size
        self.last_signal = "LONG"

    def exit_position(self, price: float):
        self.is_long = False
        pnl = 0.0
        if self.entry_price is not None and self.size > 0:
            pnl = (price - self.entry_price) * self.size
        self.entry_price = None
        self.size = 0.0
        self.last_signal = "EXIT"
        return pnl
