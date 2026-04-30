# bot/tp_trailing.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class TPLevel:
    pct: float        # profit percent (e.g. 5.0 means +5%)
    sell_pct: float   # percent of current qty to sell (e.g. 25.0 means sell 25%)


@dataclass
class ManagedPositionState:
    entry_price: float
    peak_price: float
    trailing_stop: Optional[float]
    tp_hit: List[bool]
    created_at: str
    updated_at: str


@dataclass
class SellAction:
    symbol: str
    qty: float
    reason: str


class TPAndTrailingManager:
    """
    Tracks per-symbol state (entry, peak, trailing stop, TP levels hit).
    Saves/loads from JSON. Works for both paper and live.
    """

    def __init__(
        self,
        state_path: Path,
        tp_levels: Optional[List[TPLevel]] = None,
        enable_take_profit: bool = True,
        enable_trailing: bool = True,
        trailing_arm_after_pct: float = 2.0,     # start trailing after +2%
        trailing_pct: float = 3.0,               # stop = peak * (1 - 3%)
        enable_stop_loss: bool = True,
        stop_loss_pct: float = 6.0,              # cut if -6% from entry
    ):
        self.state_path = state_path
        self.enable_take_profit = enable_take_profit
        self.enable_trailing = enable_trailing
        self.trailing_arm_after_pct = float(trailing_arm_after_pct)
        self.trailing_pct = float(trailing_pct)
        self.enable_stop_loss = enable_stop_loss
        self.stop_loss_pct = float(stop_loss_pct)

        if tp_levels is None:
            tp_levels = [
                TPLevel(pct=5.0, sell_pct=25.0),
                TPLevel(pct=10.0, sell_pct=25.0),
                TPLevel(pct=20.0, sell_pct=50.0),
            ]
        self.tp_levels = tp_levels

        self.state: Dict[str, ManagedPositionState] = {}
        self.load()

    def load(self) -> None:
        if not self.state_path.exists():
            self.state = {}
            return
        with self.state_path.open("r", encoding="utf-8") as f:
            raw = json.load(f) or {}

        out: Dict[str, ManagedPositionState] = {}
        for sym, s in raw.items():
            out[sym] = ManagedPositionState(
                entry_price=float(s["entry_price"]),
                peak_price=float(s["peak_price"]),
                trailing_stop=s.get("trailing_stop", None),
                tp_hit=list(s.get("tp_hit", [])),
                created_at=s.get("created_at", ""),
                updated_at=s.get("updated_at", ""),
            )
        self.state = out

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        raw = {sym: asdict(st) for sym, st in self.state.items()}
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)

    def sync_symbols(self, currently_holding: List[str]) -> None:
        """
        Remove state for symbols no longer held.
        """
        holding_set = set(currently_holding)
        for sym in list(self.state.keys()):
            if sym not in holding_set:
                self.state.pop(sym, None)

    def ensure_state(self, symbol: str, current_price: float, entry_override: Optional[float] = None) -> None:
        """
        If no state exists, create one. For live holdings where entry is unknown,
        entry will start at first-seen price (or entry_override if supplied).
        """
        now = datetime.utcnow().isoformat()
        if symbol not in self.state:
            entry = float(entry_override) if entry_override is not None else float(current_price)
            self.state[symbol] = ManagedPositionState(
                entry_price=entry,
                peak_price=float(current_price),
                trailing_stop=None,
                tp_hit=[False] * len(self.tp_levels),
                created_at=now,
                updated_at=now,
            )

    def evaluate(self, symbol: str, current_price: float, current_qty: float) -> List[SellAction]:
        """
        Returns sell actions (partial or full) for TP/trailing/stop-loss rules.
        Also updates internal state (peak/trailing/tp flags).
        """
        actions: List[SellAction] = []
        if current_qty <= 0:
            return actions

        st = self.state.get(symbol)
        if not st:
            return actions

        now = datetime.utcnow().isoformat()
        st.updated_at = now

        entry = st.entry_price
        price = float(current_price)

        # Update peak
        if price > st.peak_price:
            st.peak_price = price

        # Stop loss (sell all)
        if self.enable_stop_loss and entry > 0:
            stop_price = entry * (1.0 - self.stop_loss_pct / 100.0)
            if price <= stop_price:
                actions.append(SellAction(symbol=symbol, qty=current_qty, reason=f"STOP_LOSS (-{self.stop_loss_pct:.1f}%)"))
                return actions  # prioritize hard stop

        # Take profit partials
        if self.enable_take_profit and entry > 0:
            for i, lvl in enumerate(self.tp_levels):
                if i >= len(st.tp_hit):
                    st.tp_hit.extend([False] * (i + 1 - len(st.tp_hit)))
                if st.tp_hit[i]:
                    continue
                target = entry * (1.0 + lvl.pct / 100.0)
                if price >= target:
                    qty_to_sell = current_qty * (lvl.sell_pct / 100.0)
                    if qty_to_sell > 0:
                        actions.append(SellAction(symbol=symbol, qty=qty_to_sell, reason=f"TAKE_PROFIT +{lvl.pct:.1f}% sell {lvl.sell_pct:.0f}%"))
                        st.tp_hit[i] = True

        # Trailing stop (sell all) — arms only after some profit
        if self.enable_trailing and entry > 0:
            arm_price = entry * (1.0 + self.trailing_arm_after_pct / 100.0)
            if st.peak_price >= arm_price:
                st.trailing_stop = st.peak_price * (1.0 - self.trailing_pct / 100.0)
                if price <= st.trailing_stop:
                    actions.append(SellAction(symbol=symbol, qty=current_qty, reason=f"TRAIL_STOP {self.trailing_pct:.1f}% below peak"))
                    return actions

        return actions
