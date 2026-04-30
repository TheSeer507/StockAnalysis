# bot/longterm_forecaster.py
"""
Long-term GRU forecaster for portfolio holding analysis.
Trained on DAILY candles, predicts MFE over multiple horizons (7d, 14d, 30d).
Outputs holding conviction scores and multi-horizon price targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn

from .longterm_features import build_longterm_features, LONGTERM_FEATURE_COUNT


class LongtermTemporalAttention(nn.Module):
    """Learned attention over GRU hidden states for long-term forecasting."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, gru_out: torch.Tensor) -> torch.Tensor:
        # gru_out: (B, T, H)
        weights = torch.softmax(self.attn(gru_out), dim=1)  # (B, T, 1)
        context = (gru_out * weights).sum(dim=1)             # (B, H)
        return context


class LongtermQuantileGRU(nn.Module):
    """
    Multi-horizon quantile GRU for long-term price prediction.
    Predicts MFE returns at quantiles [0.5, 0.8, 0.9] for each horizon.
    Uses unidirectional GRU + temporal attention (causal architecture).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        quantiles: Tuple[float, ...],
        num_horizons: int = 3,  # 7d, 14d, 30d
    ):
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)
        self.num_horizons = num_horizons
        self.num_outputs = len(self.quantiles) * num_horizons

        # Unidirectional GRU — causal: no future information leakage
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=False,
        )

        # Temporal attention: learn which days matter most
        self.attention = LongtermTemporalAttention(hidden_size)

        # Feature normalization layers
        self.norm_last = nn.LayerNorm(hidden_size)
        self.norm_attn = nn.LayerNorm(hidden_size)

        combined = hidden_size * 2  # last hidden + attention context
        self.head = nn.Sequential(
            nn.Linear(combined, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size, self.num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.gru(x)                           # out: (B, T, H)
        last = out[:, -1, :]                             # (B, H) most recent state
        attn_ctx = self.attention(out)                   # (B, H) attention-weighted context
        combined = torch.cat([self.norm_last(last), self.norm_attn(attn_ctx)], dim=-1)  # (B, 2H)
        raw = self.head(combined)  # (B, num_horizons * num_quantiles)
        return raw


def longterm_quantile_loss(
    pred: torch.Tensor,
    targets: torch.Tensor,
    quantiles: Tuple[float, ...],
    num_horizons: int,
) -> torch.Tensor:
    """
    Multi-horizon quantile loss.
    pred:    (B, num_horizons * Q)
    targets: (B, num_horizons) — actual MFE returns at each horizon
    """
    Q = len(quantiles)
    losses = []

    for h in range(num_horizons):
        t = targets[:, h]  # (B,)
        for qi, q in enumerate(quantiles):
            p = pred[:, h * Q + qi]  # (B,)
            e = t - p
            pinball = torch.where(e >= 0, q * e, (q - 1.0) * e)
            losses.append(pinball.mean())

    # Monotonicity penalty (per horizon, q50 <= q80 <= q90)
    crossing_penalty = 0.0
    for h in range(num_horizons):
        for qi in range(Q - 1):
            lower_q = pred[:, h * Q + qi]
            upper_q = pred[:, h * Q + qi + 1]
            crossing = torch.relu(lower_q - upper_q)
            crossing_penalty += crossing.mean()

    base_loss = torch.stack(losses).mean()
    return base_loss + 0.1 * crossing_penalty


# ---------------------------------------------------
# Prediction output
# ---------------------------------------------------

@dataclass
class HorizonPrediction:
    """Prediction at a single horizon (e.g. 7 days)."""
    horizon_days: int
    tp_prices: Dict[float, float]       # quantile -> target price
    mfe_returns: Dict[float, float]     # quantile -> expected max return %


@dataclass
class LongtermPrediction:
    """Full multi-horizon prediction for a position."""
    horizons: List[HorizonPrediction]
    conviction: str         # "STRONG_HOLD", "HOLD", "WEAK", "EXIT"
    conviction_score: float # 0.0 - 1.0
    trend_phase: str        # "UPTREND", "CONSOLIDATION", "DOWNTREND", "RECOVERY"
    summary: str            # human-readable summary


HORIZON_DAYS = [7, 14, 30]


class LongtermForecaster:
    """
    Loads and runs the long-term GRU model on daily OHLCV data.
    """

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LongtermQuantileGRU] = None
        self.meta: dict = {}
        self.feat_mean = None
        self.feat_std = None

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        blob = torch.load(self.model_path, map_location=self.device, weights_only=False)
        meta = blob.get("meta") or {}
        state = blob.get("state_dict")
        if state is None:
            return False

        self.meta = meta
        input_size = int(meta.get("input_size", LONGTERM_FEATURE_COUNT))
        hidden_size = int(meta.get("hidden_size", 192))
        num_layers = int(meta.get("num_layers", 3))
        dropout = float(meta.get("dropout", 0.25))
        quantiles = tuple(float(x) for x in (meta.get("quantiles") or [0.5, 0.8, 0.9]))
        num_horizons = int(meta.get("num_horizons", 3))

        self.model = LongtermQuantileGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            quantiles=quantiles,
            num_horizons=num_horizons,
        ).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.feat_mean = blob.get("feat_mean")
        self.feat_std = blob.get("feat_std")
        return True

    def _normalize_features(self, feats: List[List[float]]) -> List[List[float]]:
        if self.feat_mean is None or self.feat_std is None:
            return feats

        mean_vals = self.feat_mean
        std_vals = self.feat_std
        if isinstance(mean_vals, list) and len(mean_vals) > 0 and isinstance(mean_vals[0], list):
            mean_vals = mean_vals[0]
        if isinstance(std_vals, list) and len(std_vals) > 0 and isinstance(std_vals[0], list):
            std_vals = std_vals[0]

        normalized = []
        for row in feats:
            norm_row = []
            for j, val in enumerate(row):
                if j < len(mean_vals):
                    m, s = float(mean_vals[j]), float(std_vals[j])
                    norm_row.append((val - m) / (s + 1e-8))
                else:
                    norm_row.append(float(val))
            normalized.append(norm_row)
        return normalized

    def _classify_trend(self, feats: List[List[float]]) -> str:
        """Determine trend phase from recent features."""
        if len(feats) < 5:
            return "CONSOLIDATION"

        last = feats[-1]
        # EMA200 distance (idx 6), EMA50/200 cross (idx 8), drawdown (idx 15)
        ema200_dist = last[6]
        ema50_200_cross = last[8]
        drawdown = last[15]
        price_position = last[16]

        if ema200_dist > 0.03 and ema50_200_cross > 0:
            return "UPTREND"
        elif drawdown < -0.20:
            # Deep drawdown but check if recovering
            if ema200_dist > -0.05 and price_position > 0.5:
                return "RECOVERY"
            return "DOWNTREND"
        elif abs(ema200_dist) < 0.03:
            return "CONSOLIDATION"
        elif ema200_dist < -0.05:
            return "DOWNTREND"
        else:
            return "CONSOLIDATION"

    @torch.no_grad()
    def predict(self, ohlcv_daily: List[List[float]], lookback: int = 90) -> Optional[LongtermPrediction]:
        """
        Predict multi-horizon MFE from daily OHLCV data.

        Args:
            ohlcv_daily: Daily [ts, o, h, l, c, v] candles (need at least lookback+200 for EMA200)
            lookback: Number of days to feed to GRU (default 90)
        """
        if self.model is None:
            return None
        min_needed = max(lookback + 10, 210)
        if len(ohlcv_daily) < min_needed:
            return None

        feats = build_longterm_features(ohlcv_daily)
        if len(feats) < lookback:
            return None

        trend_phase = self._classify_trend(feats)

        feats_norm = self._normalize_features(feats)

        x = torch.tensor(feats_norm[-lookback:], dtype=torch.float32, device=self.device).unsqueeze(0)
        raw = self.model(x).squeeze(0).detach().cpu().tolist()

        quantiles = self.model.quantiles
        Q = len(quantiles)
        last_close = float(ohlcv_daily[-1][4])

        horizons: List[HorizonPrediction] = []
        horizon_days = self.meta.get("horizon_days", HORIZON_DAYS)

        for h_idx, h_days in enumerate(horizon_days):
            mfe_returns: Dict[float, float] = {}
            tp_prices: Dict[float, float] = {}

            for qi, q in enumerate(quantiles):
                r = float(raw[h_idx * Q + qi])
                r = max(-0.95, r)  # clamp
                mfe_returns[q] = r
                tp_prices[q] = max(0.0, last_close * (1.0 + r))

            horizons.append(HorizonPrediction(
                horizon_days=h_days,
                tp_prices=tp_prices,
                mfe_returns=mfe_returns,
            ))

        # Compute conviction score
        conviction_score = self._compute_conviction(horizons, feats, trend_phase)

        if conviction_score >= 0.7:
            conviction = "STRONG_HOLD"
        elif conviction_score >= 0.45:
            conviction = "HOLD"
        elif conviction_score >= 0.25:
            conviction = "WEAK"
        else:
            conviction = "EXIT"

        summary = self._build_summary(horizons, conviction, conviction_score, trend_phase, last_close)

        return LongtermPrediction(
            horizons=horizons,
            conviction=conviction,
            conviction_score=conviction_score,
            trend_phase=trend_phase,
            summary=summary,
        )

    def _compute_conviction(
        self,
        horizons: List[HorizonPrediction],
        feats: List[List[float]],
        trend_phase: str,
    ) -> float:
        """
        Compute 0-1 conviction score from ML predictions + trend context.
        Higher = stronger reason to hold.
        """
        score = 0.0

        # 1. Multi-horizon MFE alignment (all horizons positive = strong)
        for hp in horizons:
            med_ret = hp.mfe_returns.get(0.5, 0.0)
            if med_ret > 0.05:
                score += 0.12
            elif med_ret > 0.02:
                score += 0.06
            elif med_ret > 0:
                score += 0.02

        # 2. Longer horizons more upside than shorter (healthy trend)
        if len(horizons) >= 2:
            short_ret = horizons[0].mfe_returns.get(0.5, 0.0)
            long_ret = horizons[-1].mfe_returns.get(0.5, 0.0)
            if long_ret > short_ret > 0:
                score += 0.15  # accelerating uptrend
            elif long_ret > 0 and short_ret > 0:
                score += 0.08

        # 3. Trend phase bonus
        phase_bonus = {
            "UPTREND": 0.20,
            "RECOVERY": 0.10,
            "CONSOLIDATION": 0.05,
            "DOWNTREND": -0.10,
        }
        score += phase_bonus.get(trend_phase, 0.0)

        # 4. Technical features from last bar
        if feats:
            last = feats[-1]
            ema200_dist = last[6]
            higher_highs = last[17]
            higher_lows = last[18]
            rsi14 = last[9] * 100  # denormalize

            # Above 200 EMA
            if ema200_dist > 0:
                score += 0.08
            # Trend structure
            if higher_highs > 0.5 and higher_lows > 0.5:
                score += 0.10
            # RSI in healthy range (not overbought/oversold)
            if 40 <= rsi14 <= 70:
                score += 0.05

        return max(0.0, min(1.0, score))

    def _build_summary(
        self,
        horizons: List[HorizonPrediction],
        conviction: str,
        score: float,
        trend_phase: str,
        last_close: float,
    ) -> str:
        parts = [f"Trend: {trend_phase} | Conviction: {conviction} ({score:.0%})"]

        for hp in horizons:
            med = hp.mfe_returns.get(0.5, 0.0)
            tp80 = hp.tp_prices.get(0.8, last_close)
            parts.append(
                f"  {hp.horizon_days:2d}d: median MFE {med:+.1%}, TP80=${tp80:.2f}"
            )

        if conviction == "STRONG_HOLD":
            parts.append("  → Strong upside expected. Hold and let it run.")
        elif conviction == "HOLD":
            parts.append("  → Positive outlook. Continue holding.")
        elif conviction == "WEAK":
            parts.append("  → Weakening setup. Consider tightening stops.")
        else:
            parts.append("  → Negative outlook. Consider reducing position.")

        return "\n".join(parts)
