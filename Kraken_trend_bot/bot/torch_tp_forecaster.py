# bot/torch_tp_forecaster.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import torch  # noqa: E402
import torch.nn as nn


# -----------------------------
# Utilities (pure python)
# -----------------------------

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _ema_series(values: List[float], span: int) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [values[0]]
    e = values[0]
    for v in values[1:]:
        e = alpha * v + (1.0 - alpha) * e
        out.append(float(e))
    return out


def _rsi_series(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < 2:
        return [50.0] * len(closes)

    alpha = 1.0 / float(period)
    avg_gain = 0.0
    avg_loss = 0.0
    out = [50.0]  # first
    for i in range(1, len(closes)):
        chg = closes[i] - closes[i - 1]
        gain = max(chg, 0.0)
        loss = max(-chg, 0.0)
        avg_gain = alpha * gain + (1.0 - alpha) * avg_gain
        avg_loss = alpha * loss + (1.0 - alpha) * avg_loss
        if avg_loss <= 1e-12:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        out.append(float(rsi))
    return out


def _atr_pct_series(ohlcv: List[List[float]], period: int = 14) -> List[float]:
    # ATR / close
    if not ohlcv:
        return []
    alpha = 1.0 / float(period)
    prev_close = float(ohlcv[0][4])
    h0 = float(ohlcv[0][2])
    l0 = float(ohlcv[0][3])
    tr0 = max(h0 - l0, abs(h0 - prev_close), abs(l0 - prev_close))
    atr = tr0
    out = [0.0]
    for i in range(1, len(ohlcv)):
        h = float(ohlcv[i][2])
        l = float(ohlcv[i][3])
        c = float(ohlcv[i][4])
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        atr = alpha * tr + (1.0 - alpha) * atr
        prev_close = c
        out.append(float(atr / c) if c > 0 else 0.0)
    return out


def build_seq_features(
    ohlcv: List[List[float]],
    ema_fast: int = 20,
    ema_slow: int = 50,
    rsi_period: int = 14,
    atr_period: int = 14,
    btc_ohlcv: Optional[List[List[float]]] = None,
) -> List[List[float]]:
    """
    Returns per-bar feature vectors (len=24):
      0  log_return
      1  hl_range_pct
      2  oc_return
      3  vol_log_delta
      4  ema_spread_pct (ema_fast - ema_slow)/close
      5  rsi_norm (0..1)
      6  atr_pct
      7  mom_3 (close/close[-3]-1)
      8  mom_7 (close/close[-7]-1)
      9  mom_14 (close/close[-14]-1)
      10 upper_wick_pct (high-max(open,close))/close
      11 lower_wick_pct (min(open,close)-low)/close
      12 volume_ma_ratio vol/vol_ma10
      13 high_to_ema_fast (high - ema_fast) / close
      14 low_to_ema_slow (low - ema_slow) / close
      15 log_return_volatility (rolling std of log returns)
      16 price_acceleration (change in momentum)
      17 volume_surge (vol spike vs recent avg)
      18 hour_of_day_sin (cyclical time feature)
      19 hour_of_day_cos (cyclical time feature)
      20 order_flow_proxy  (close-low)/(high-low) — buy pressure
      21 bollinger_pctb    (close-bb_lower)/(bb_upper-bb_lower)
      22 vwap_distance     (close-vwap)/close — distance to VWAP
      23 btc_beta          BTC log return at same timestamp (cross-asset beta)

    Args:
        btc_ohlcv: Optional BTC OHLCV data (same timeframe). When provided,
                   BTC log returns are computed and matched by timestamp to fill
                   feature slot 23. This closes the train/inference gap where
                   training injected BTC returns but inference left slot 23 as 0.0.
    """
    if not ohlcv:
        return []

    # Pre-compute BTC log returns keyed by timestamp for slot 23
    btc_returns_map: Dict[int, float] = {}
    if btc_ohlcv and len(btc_ohlcv) > 1:
        prev_btc_c = float(btc_ohlcv[0][4])
        for bi in range(1, len(btc_ohlcv)):
            bts = int(btc_ohlcv[bi][0])
            btc_c = float(btc_ohlcv[bi][4])
            if prev_btc_c > 0 and btc_c > 0:
                btc_returns_map[bts] = math.log(btc_c / prev_btc_c)
            else:
                btc_returns_map[bts] = 0.0
            prev_btc_c = btc_c

    closes = [float(r[4]) for r in ohlcv]
    opens = [float(r[1]) for r in ohlcv]
    highs = [float(r[2]) for r in ohlcv]
    lows  = [float(r[3]) for r in ohlcv]
    vols  = [float(r[5]) for r in ohlcv]
    times = [int(r[0]) for r in ohlcv]

    ema_f = _ema_series(closes, ema_fast)
    ema_s = _ema_series(closes, ema_slow)
    rsi_s = _rsi_series(closes, rsi_period)
    atrp  = _atr_pct_series(ohlcv, atr_period)
    
    # Calculate log returns for volatility
    log_rets = [0.0]
    for i in range(1, len(closes)):
        if closes[i-1] > 0 and closes[i] > 0:
            log_rets.append(math.log(closes[i] / closes[i-1]))
        else:
            log_rets.append(0.0)

    # Volume moving average
    vol_ma = _ema_series(vols, 10)

    # Bollinger Bands (20-period SMA +/- 2 std)
    bb_period = 20
    bb_sma: List[float] = []
    bb_upper: List[float] = []
    bb_lower: List[float] = []
    for i in range(len(closes)):
        if i < bb_period - 1:
            bb_sma.append(closes[i])
            bb_upper.append(closes[i])
            bb_lower.append(closes[i])
        else:
            window = closes[i - bb_period + 1:i + 1]
            mean_val = sum(window) / bb_period
            var_val = sum((x - mean_val) ** 2 for x in window) / bb_period
            std_val = var_val ** 0.5
            bb_sma.append(mean_val)
            bb_upper.append(mean_val + 2.0 * std_val)
            bb_lower.append(mean_val - 2.0 * std_val)

    # Rolling VWAP (20-period)
    vwap_period = 20
    vwap_vals: List[float] = []
    for i in range(len(closes)):
        if i < vwap_period - 1:
            vwap_vals.append(closes[i])
        else:
            pv_sum = sum(closes[j] * vols[j] for j in range(i - vwap_period + 1, i + 1))
            v_sum = sum(vols[j] for j in range(i - vwap_period + 1, i + 1))
            vwap_vals.append(pv_sum / v_sum if v_sum > 0 else closes[i])

    feats: List[List[float]] = []
    prev_close = closes[0]
    prev_vol = vols[0]
    prev_mom = 0.0

    for i in range(len(ohlcv)):
        c = closes[i]
        o = opens[i]
        h = highs[i]
        l = lows[i]
        v = vols[i]

        # 0 log return
        if prev_close > 0 and c > 0:
            lr = math.log(c / prev_close)
        else:
            lr = 0.0

        # 1 high-low range pct
        hl = (h - l) / c if c > 0 else 0.0

        # 2 open->close return
        oc = (c - o) / o if o > 0 else 0.0

        # 3 volume log delta
        v1 = math.log1p(max(v, 0.0))
        v0 = math.log1p(max(prev_vol, 0.0))
        vd = v1 - v0

        # 4 EMA spread pct
        esf = (ema_f[i] - ema_s[i]) / c if c > 0 else 0.0

        # 5 RSI normalized
        rsi = float(rsi_s[i]) / 100.0

        # 6 ATR pct
        ap = float(atrp[i]) if i < len(atrp) else 0.0

        # 7 momentum 3
        if i >= 3 and closes[i - 3] > 0:
            mom3 = (c / closes[i - 3]) - 1.0
        else:
            mom3 = 0.0
        
        # 8 momentum 7
        if i >= 7 and closes[i - 7] > 0:
            mom7 = (c / closes[i - 7]) - 1.0
        else:
            mom7 = 0.0
        
        # 9 momentum 14
        if i >= 14 and closes[i - 14] > 0:
            mom14 = (c / closes[i - 14]) - 1.0
        else:
            mom14 = 0.0
        
        # 10 upper wick pct
        body_top = max(o, c)
        upper_wick = (h - body_top) / c if c > 0 else 0.0
        
        # 11 lower wick pct
        body_bottom = min(o, c)
        lower_wick = (body_bottom - l) / c if c > 0 else 0.0
        
        # 12 volume vs MA ratio
        vol_ratio = v / vol_ma[i] if vol_ma[i] > 0 else 1.0
        
        # 13 high distance to fast EMA
        high_ema_dist = (h - ema_f[i]) / c if c > 0 else 0.0
        
        # 14 low distance to slow EMA
        low_ema_dist = (l - ema_s[i]) / c if c > 0 else 0.0
        
        # 15 log return volatility (rolling std over 14 bars)
        if i >= 14:
            recent_rets = log_rets[max(0, i-13):i+1]
            ret_std = float(sum((r - sum(recent_rets)/len(recent_rets))**2 for r in recent_rets) / len(recent_rets))**0.5
        else:
            ret_std = 0.0
        
        # 16 price acceleration (change in momentum)
        accel = mom3 - prev_mom
        prev_mom = mom3
        
        # 17 volume surge (z-score like)
        if i >= 20:
            recent_vols = vols[max(0, i-19):i]
            vol_avg = sum(recent_vols) / len(recent_vols) if recent_vols else 1.0
            vol_std = float(sum((vv - vol_avg)**2 for vv in recent_vols) / len(recent_vols))**0.5 if recent_vols else 1.0
            vol_surge = (v - vol_avg) / vol_std if vol_std > 0 else 0.0
        else:
            vol_surge = 0.0
        
        # 18-19 cyclical time features
        #   For intraday candles: hour of day (sin/cos)
        #   For daily+ candles:   day of week (sin/cos) — more meaningful
        ts_sec = times[i] // 1000  # ms -> s
        if len(ohlcv) >= 2:
            bar_interval_ms = abs(int(ohlcv[min(i, len(ohlcv)-1)][0]) - int(ohlcv[max(i-1, 0)][0]))
        else:
            bar_interval_ms = 0
        if bar_interval_ms >= 86_400_000:  # >= 1 day bars
            # Day of week: Monday=0 .. Sunday=6
            import datetime as _dt
            dow = _dt.datetime.utcfromtimestamp(ts_sec).weekday()
            hour_sin = math.sin(2 * math.pi * dow / 7)
            hour_cos = math.cos(2 * math.pi * dow / 7)
        else:
            hour = (times[i] // 3600000) % 24
            hour_sin = math.sin(2 * math.pi * hour / 24)
            hour_cos = math.cos(2 * math.pi * hour / 24)

        # 20 order flow proxy: (close - low) / (high - low) — 1.0 = strong buying
        hl_range = h - l
        order_flow = (c - l) / hl_range if hl_range > 0 else 0.5

        # 21 Bollinger %B: (close - bb_lower) / (bb_upper - bb_lower)
        bb_width = bb_upper[i] - bb_lower[i]
        boll_pctb = (c - bb_lower[i]) / bb_width if bb_width > 0 else 0.5

        # 22 VWAP distance: (close - vwap) / close
        vwap_dist = (c - vwap_vals[i]) / c if c > 0 else 0.0

        # 23 BTC beta — cross-asset correlation feature
        #    Matches BTC log return at the same timestamp. During training this
        #    was injected from a pre-fetched BTC series; at inference time we now
        #    compute it from the optional btc_ohlcv parameter.
        btc_beta = btc_returns_map.get(times[i], 0.0)

        feats.append([
            float(lr), float(hl), float(oc), float(vd), float(esf), float(rsi),
            float(ap), float(mom3), float(mom7), float(mom14), float(upper_wick),
            float(lower_wick), float(vol_ratio), float(high_ema_dist), float(low_ema_dist),
            float(ret_std), float(accel), float(vol_surge), float(hour_sin), float(hour_cos),
            float(order_flow), float(boll_pctb), float(vwap_dist), float(btc_beta),
        ])

        prev_close = c
        prev_vol = v

    return feats


# -----------------------------
# Quantile model + loss
# -----------------------------

class TemporalAttention(nn.Module):
    """Learned attention over GRU hidden states to focus on the most informative bars."""

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


class QuantileGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        quantiles: Tuple[float, ...],
    ):
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)

        # Unidirectional GRU — causal: no future information leakage
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=False,
        )

        # Temporal attention: learn *which* bars matter most
        self.attention = TemporalAttention(hidden_size)

        # Feature normalization layers
        self.norm_last = nn.LayerNorm(hidden_size)
        self.norm_attn = nn.LayerNorm(hidden_size)

        # Prediction head (last hidden + attention context concatenated)
        combined = hidden_size * 2  # last hidden + attention context
        self.head = nn.Sequential(
            nn.Linear(combined, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size, len(self.quantiles)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.gru(x)                           # out: (B, T, H)
        last = out[:, -1, :]                             # (B, H) most recent state
        attn_ctx = self.attention(out)                   # (B, H) attention-weighted context
        combined = torch.cat([self.norm_last(last), self.norm_attn(attn_ctx)], dim=-1)  # (B, 2H)
        return self.head(combined)                       # (B, Q) predicted MFE returns


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: Tuple[float, ...]) -> torch.Tensor:
    """
    Improved quantile loss with:
    1. Proper quantile crossing penalty (monotonicity)
    2. Smoothed pinball loss for better gradients
    
    pred:   (B, Q)
    target: (B,) or (B,1)  scalar actual MFE return
    """
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    target = target.view(-1, 1)  # (B,1)
    
    losses = []
    for i, q in enumerate(quantiles):
        e = target[:, 0] - pred[:, i]
        # Smoothed pinball loss (Huber-like)
        pinball = torch.where(e >= 0, q * e, (q - 1.0) * e)
        losses.append(pinball.mean())
    
    # Add penalty for quantile crossing (q50 should be <= q80 <= q90)
    crossing_penalty = 0.0
    if len(quantiles) >= 2:
        for i in range(len(quantiles) - 1):
            # Penalize when lower quantile > higher quantile
            crossing = torch.relu(pred[:, i] - pred[:, i + 1])
            crossing_penalty += crossing.mean()
    
    base_loss = torch.stack(losses).mean()
    return base_loss + 0.1 * crossing_penalty  # Small penalty weight


# -----------------------------
# Prediction wrapper
# -----------------------------

@dataclass
class TPPrediction:
    tp_prices: Dict[float, float]
    mfe_returns: Dict[float, float]


class TorchTPForecaster:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[QuantileGRU] = None
        self.meta: dict = {}

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        blob = torch.load(self.model_path, map_location=self.device)
        meta = blob.get("meta") or {}
        state = blob.get("state_dict")
        if state is None:
            return False

        self.meta = meta
        input_size = int(meta.get("input_size", 24))
        hidden_size = int(meta.get("hidden_size", 128))
        num_layers = int(meta.get("num_layers", 2))
        dropout = float(meta.get("dropout", 0.15))
        quantiles = tuple(float(x) for x in (meta.get("quantiles") or [0.5, 0.8, 0.9]))

        self.model = QuantileGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            quantiles=quantiles,
        ).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        
        # Load feature normalization params if available
        self.feat_mean = blob.get("feat_mean")
        self.feat_std = blob.get("feat_std")
        
        # Load post-hoc calibration scales if available
        self.calibration_scales = blob.get("calibration_scales")
        if self.calibration_scales:
            print(f"[ML] Loaded calibration scales: {[f'{s:.3f}' for s in self.calibration_scales]}")
        
        return True

    def _normalize_features(self, feats: List[List[float]]) -> List[List[float]]:
        """Normalize features using saved mean/std"""
        if self.feat_mean is None or self.feat_std is None:
            return feats
        
        # Handle case where feat_mean/feat_std might be list of lists (bug in training)
        mean_vals = self.feat_mean
        std_vals = self.feat_std
        
        # If they're lists of lists, flatten them by taking the first element
        if isinstance(mean_vals, list) and len(mean_vals) > 0 and isinstance(mean_vals[0], list):
            mean_vals = mean_vals[0]  # Take first row
        if isinstance(std_vals, list) and len(std_vals) > 0 and isinstance(std_vals[0], list):
            std_vals = std_vals[0]  # Take first row
        
        normalized = []
        for row in feats:
            norm_row = []
            for i, val in enumerate(row):
                if i < len(mean_vals):
                    mean = float(mean_vals[i])
                    std = float(std_vals[i])
                    norm_val = (val - mean) / (std + 1e-8)
                    norm_row.append(float(norm_val))
                else:
                    norm_row.append(float(val))
            normalized.append(norm_row)
        return normalized

    @torch.no_grad()
    def predict(
        self,
        ohlcv: List[List[float]],
        lookback: int = 64,
        horizon_bars: int = 96,
        btc_ohlcv: Optional[List[List[float]]] = None,
    ) -> Optional[TPPrediction]:
        """Predict MFE quantiles for the most recent bar.

        Args:
            ohlcv:        Symbol OHLCV candles (same timeframe as model).
            lookback:     Number of bars to feed into the GRU.
            horizon_bars: Prediction horizon (unused at inference but kept for API compat).
            btc_ohlcv:    Optional BTC OHLCV data (same timeframe). When provided,
                          BTC log returns fill feature slot 23, closing the
                          train/inference mismatch for the cross-asset beta feature.
        """
        if self.model is None:
            return None
        if len(ohlcv) < max(lookback + 5, 80):
            return None

        ema_fast = int(self.meta.get("ema_fast", 20))
        ema_slow = int(self.meta.get("ema_slow", 50))
        rsi_period = int(self.meta.get("rsi_period", 14))
        atr_period = int(self.meta.get("atr_period", 14))

        feats = build_seq_features(
            ohlcv,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi_period=rsi_period,
            atr_period=atr_period,
            btc_ohlcv=btc_ohlcv,
        )
        if len(feats) < lookback:
            return None
        
        # Normalize features
        feats = self._normalize_features(feats)

        x = torch.tensor(feats[-lookback:], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,T,F)
        pred = self.model(x).squeeze(0).detach().cpu().tolist()  # (Q,)

        quantiles = self.model.quantiles
        last_close = float(ohlcv[-1][4])
        mfe_returns: Dict[float, float] = {}
        tp_prices: Dict[float, float] = {}

        for i, (q, r) in enumerate(zip(quantiles, pred)):
            rr = float(r)
            # Apply post-hoc calibration scaling if available
            if self.calibration_scales and i < len(self.calibration_scales):
                rr = rr * self.calibration_scales[i]
            # clamp to avoid pathological negatives
            rr = max(-0.95, rr)
            mfe_returns[q] = rr
            tp_prices[q] = max(0.0, last_close * (1.0 + rr))

        return TPPrediction(tp_prices=tp_prices, mfe_returns=mfe_returns)
