# bot/longterm_features.py
"""
Long-term features for portfolio holding analysis.
Uses DAILY candles and wider lookback windows suitable for
multi-week / multi-month holding decisions.
"""
from __future__ import annotations

import math
from typing import List


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


def _sma_series(values: List[float], period: int) -> List[float]:
    out = []
    for i in range(len(values)):
        if i < period - 1:
            out.append(sum(values[:i + 1]) / (i + 1))
        else:
            out.append(sum(values[i - period + 1:i + 1]) / period)
    return out


def _rsi_series(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < 2:
        return [50.0] * len(closes)
    alpha = 1.0 / float(period)
    avg_gain = 0.0
    avg_loss = 0.0
    out = [50.0]
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
    if not ohlcv:
        return []
    alpha = 1.0 / float(period)
    prev_close = float(ohlcv[0][4])
    h0, l0 = float(ohlcv[0][2]), float(ohlcv[0][3])
    tr0 = max(h0 - l0, abs(h0 - prev_close), abs(l0 - prev_close))
    atr = tr0
    out = [0.0]
    for i in range(1, len(ohlcv)):
        h, l, c = float(ohlcv[i][2]), float(ohlcv[i][3]), float(ohlcv[i][4])
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        atr = alpha * tr + (1.0 - alpha) * atr
        prev_close = c
        out.append(float(atr / c) if c > 0 else 0.0)
    return out


def build_longterm_features(
    ohlcv: List[List[float]],
) -> List[List[float]]:
    """
    Build per-bar feature vectors from DAILY OHLCV data for long-term analysis.

    Features (25 total):
      0   log_return_1d          (daily return)
      1   ret_5d                 (weekly return)
      2   ret_20d                (monthly return)
      3   ret_60d                (quarterly return)
      4   ema20_dist             (close - EMA20) / close
      5   ema50_dist             (close - EMA50) / close
      6   ema200_dist            (close - EMA200) / close — key trend indicator
      7   ema20_over_50          EMA20 / EMA50 - 1
      8   ema50_over_200         EMA50 / EMA200 - 1 (golden/death cross proxy)
      9   rsi14                  RSI 14-day normalized 0..1
      10  rsi28                  RSI 28-day normalized 0..1
      11  atr_pct_14             ATR(14) / close — daily volatility
      12  atr_ratio              ATR(14) / ATR(28) — vol acceleration
      13  vol_z_20               volume z-score vs 20d mean
      14  vol_trend              volume 10d EMA / 50d EMA — accumulation signal
      15  drawdown_from_60d_high  (close - high60) / high60 — how deep in drawdown
      16  price_position_60d     (close - low60) / (high60 - low60) — where in range
      17  higher_highs_20d       (high20 > high20_prev) as 0/1 — uptrend structure
      18  higher_lows_20d        (low20 > low20_prev) as 0/1 — uptrend structure
      19  weekly_trend_strength  5d return / atr — trend vs noise ratio
      20  monthly_momentum_accel change in 20d return momentum
      21  consolidation_squeeze  ATR(5) / ATR(30) — low = about to breakout
      22  body_avg_5d            avg |close-open| / close — conviction measure
      23  day_of_week_sin        cyclical day-of-week feature
      24  day_of_week_cos        cyclical day-of-week feature
      25  order_flow_proxy       (close-low)/(high-low) — daily buy pressure
      26  bollinger_pctb         (close-bb_lower)/(bb_upper-bb_lower)
      27  vwap_distance_20d      (close-vwap20)/close — distance to 20d VWAP
    """
    if not ohlcv:
        return []

    closes = [float(r[4]) for r in ohlcv]
    opens = [float(r[1]) for r in ohlcv]
    highs = [float(r[2]) for r in ohlcv]
    lows = [float(r[3]) for r in ohlcv]
    vols = [float(r[5]) for r in ohlcv]
    times = [int(r[0]) for r in ohlcv]

    # EMAs
    ema20 = _ema_series(closes, 20)
    ema50 = _ema_series(closes, 50)
    ema200 = _ema_series(closes, 200)

    # RSI
    rsi14 = _rsi_series(closes, 14)
    rsi28 = _rsi_series(closes, 28)

    # ATR
    atrp14 = _atr_pct_series(ohlcv, 14)
    atrp28 = _atr_pct_series(ohlcv, 28)
    atrp5 = _atr_pct_series(ohlcv, 5)
    atrp30 = _atr_pct_series(ohlcv, 30)

    # Volume MAs
    vol_ema10 = _ema_series(vols, 10)
    vol_ema50 = _ema_series(vols, 50)

    # Volume 20d stats
    vol_sma20 = _sma_series(vols, 20)

    # Log returns
    log_rets = [0.0]
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            log_rets.append(math.log(closes[i] / closes[i - 1]))
        else:
            log_rets.append(0.0)

    # Bollinger Bands (20-period SMA +/- 2 std)
    bb_period = 20
    bb_upper_vals: List[float] = []
    bb_lower_vals: List[float] = []
    for i in range(len(closes)):
        if i < bb_period - 1:
            bb_upper_vals.append(closes[i])
            bb_lower_vals.append(closes[i])
        else:
            window = closes[i - bb_period + 1:i + 1]
            mean_val = sum(window) / bb_period
            var_val = sum((x - mean_val) ** 2 for x in window) / bb_period
            std_val = var_val ** 0.5
            bb_upper_vals.append(mean_val + 2.0 * std_val)
            bb_lower_vals.append(mean_val - 2.0 * std_val)

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

    prev_mom20 = 0.0

    for i in range(len(ohlcv)):
        c = closes[i]
        o = opens[i]
        h = highs[i]
        l = lows[i]
        v = vols[i]

        # 0: daily log return
        lr = log_rets[i]

        # 1: 5-day return
        ret5 = (c / closes[i - 5] - 1.0) if i >= 5 and closes[i - 5] > 0 else 0.0

        # 2: 20-day return
        ret20 = (c / closes[i - 20] - 1.0) if i >= 20 and closes[i - 20] > 0 else 0.0

        # 3: 60-day return
        ret60 = (c / closes[i - 60] - 1.0) if i >= 60 and closes[i - 60] > 0 else 0.0

        # 4-6: EMA distances
        ema20_d = (c - ema20[i]) / c if c > 0 else 0.0
        ema50_d = (c - ema50[i]) / c if c > 0 else 0.0
        ema200_d = (c - ema200[i]) / c if c > 0 else 0.0

        # 7-8: EMA cross ratios
        ema20_o_50 = (ema20[i] / ema50[i] - 1.0) if ema50[i] > 0 else 0.0
        ema50_o_200 = (ema50[i] / ema200[i] - 1.0) if ema200[i] > 0 else 0.0

        # 9-10: RSI normalized
        rsi14_n = rsi14[i] / 100.0
        rsi28_n = rsi28[i] / 100.0

        # 11-12: ATR features
        atr14 = atrp14[i] if i < len(atrp14) else 0.0
        atr28 = atrp28[i] if i < len(atrp28) else 0.0
        atr_ratio = atr14 / (atr28 + 1e-8)

        # 13: volume z-score vs 20d
        vol20_mean = vol_sma20[i] if i < len(vol_sma20) else 1.0
        # manual std over last 20
        if i >= 20:
            recent_vols = vols[i - 19:i + 1]
            vol_avg = sum(recent_vols) / len(recent_vols)
            vol_std = (sum((vv - vol_avg) ** 2 for vv in recent_vols) / len(recent_vols)) ** 0.5
            vol_z = (v - vol_avg) / (vol_std + 1e-8)
        else:
            vol_z = 0.0

        # 14: volume trend (accumulation signal)
        vol_trend = (vol_ema10[i] / vol_ema50[i]) if vol_ema50[i] > 0 else 1.0

        # 15: drawdown from 60-day high
        if i >= 60:
            high60 = max(highs[i - 59:i + 1])
            drawdown = (c - high60) / high60 if high60 > 0 else 0.0
        else:
            high60 = max(highs[:i + 1])
            drawdown = (c - high60) / high60 if high60 > 0 else 0.0

        # 16: price position in 60d range
        if i >= 60:
            h60 = max(highs[i - 59:i + 1])
            l60 = min(lows[i - 59:i + 1])
            price_pos = (c - l60) / (h60 - l60 + 1e-8)
        else:
            h_all = max(highs[:i + 1])
            l_all = min(lows[:i + 1])
            price_pos = (c - l_all) / (h_all - l_all + 1e-8)

        # 17-18: trend structure (higher highs / higher lows over 20d windows)
        if i >= 40:
            hh_cur = max(highs[i - 19:i + 1])
            hh_prev = max(highs[i - 39:i - 19])
            higher_highs = 1.0 if hh_cur > hh_prev else 0.0

            ll_cur = min(lows[i - 19:i + 1])
            ll_prev = min(lows[i - 39:i - 19])
            higher_lows = 1.0 if ll_cur > ll_prev else 0.0
        else:
            higher_highs = 0.5
            higher_lows = 0.5

        # 19: weekly trend strength (return vs noise)
        weekly_str = ret5 / (atr14 + 1e-8) if atr14 > 0 else 0.0

        # 20: monthly momentum acceleration
        mom_accel = ret20 - prev_mom20
        prev_mom20 = ret20

        # 21: consolidation squeeze (low = tight range, potential breakout)
        atr5 = atrp5[i] if i < len(atrp5) else 0.0
        atr30 = atrp30[i] if i < len(atrp30) else 0.0
        squeeze = atr5 / (atr30 + 1e-8)

        # 22: average body size over 5 days (conviction)
        if i >= 5:
            body_avg = sum(abs(closes[j] - opens[j]) / closes[j] for j in range(i - 4, i + 1) if closes[j] > 0) / 5.0
        else:
            body_avg = abs(c - o) / c if c > 0 else 0.0

        # 23-24: cyclical day of week
        # Approximate day from timestamp (not perfect but useful)
        day_of_week = ((times[i] // 86400000) + 4) % 7  # Unix epoch was Thursday
        dow_sin = math.sin(2 * math.pi * day_of_week / 7)
        dow_cos = math.cos(2 * math.pi * day_of_week / 7)

        # 25: order flow proxy (close - low) / (high - low) — daily buy pressure
        hl_range_d = h - l
        order_flow_d = (c - l) / hl_range_d if hl_range_d > 0 else 0.5

        # 26: Bollinger %B (close - bb_lower) / (bb_upper - bb_lower)
        bb_width_d = bb_upper_vals[i] - bb_lower_vals[i]
        boll_pctb_d = (c - bb_lower_vals[i]) / bb_width_d if bb_width_d > 0 else 0.5

        # 27: VWAP distance (close - vwap20) / close
        vwap_dist_d = (c - vwap_vals[i]) / c if c > 0 else 0.0

        feats.append([
            float(lr), float(ret5), float(ret20), float(ret60),
            float(ema20_d), float(ema50_d), float(ema200_d),
            float(ema20_o_50), float(ema50_o_200),
            float(rsi14_n), float(rsi28_n),
            float(atr14), float(atr_ratio),
            float(vol_z), float(vol_trend),
            float(drawdown), float(price_pos),
            float(higher_highs), float(higher_lows),
            float(weekly_str), float(mom_accel),
            float(squeeze), float(body_avg),
            float(dow_sin), float(dow_cos),
            float(order_flow_d), float(boll_pctb_d), float(vwap_dist_d),
        ])

    return feats


LONGTERM_FEATURE_COUNT = 28
