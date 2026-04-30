# ml/train_price_forecaster.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml
import joblib

from sklearn.ensemble import GradientBoostingRegressor

from exchange import ExchangeClient
from data_handler import ohlcv_to_dataframe
from ml_features import build_features


CONFIG_PATH = Path("config/config.yaml")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fetch_ohlcv_history(exchange, symbol: str, timeframe: str, max_bars: int, batch_limit: int, sleep_sec: float) -> List[list]:
    """
    Pulls multiple batches using since to reach max_bars.
    """
    all_rows: List[list] = []
    since = None

    while len(all_rows) < max_bars:
        batch = exchange.fetch_ohlcv(symbol, timeframe, limit=batch_limit, since=since)
        if not batch:
            break

        # avoid duplicates if since lands on same candle
        if all_rows and batch[0][0] <= all_rows[-1][0]:
            batch = [r for r in batch if r[0] > all_rows[-1][0]]
        if not batch:
            break

        all_rows.extend(batch)
        since = all_rows[-1][0] + 1

        if len(batch) < batch_limit:
            break

        time.sleep(sleep_sec)

    return all_rows[-max_bars:]


def forward_max_high_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    y = (max_high_next_h / close_now) - 1
    Uses reverse rolling trick to avoid O(n*h).
    """
    high = df["high"].astype(float)
    close = df["close"].astype(float)

    # max over i..i+h-1 then shift to i+1..i+h
    fwd_max = high[::-1].rolling(horizon, min_periods=horizon).max()[::-1].shift(-1)
    y = (fwd_max / close) - 1.0
    return y


def main():
    cfg = load_config()
    exch_cfg = cfg.get("exchange", {}) or {}
    ml_cfg = cfg.get("ml", {}) or {}

    exchange_name = exch_cfg.get("name", "kraken")
    timeframe = exch_cfg.get("timeframe", "15m")
    base_currency = (exch_cfg.get("base_currency", "USD") or "USD").upper()

    model_path = Path(ml_cfg.get("model_path", "data/ml_price_forecaster.joblib"))
    horizon_bars = int(ml_cfg.get("horizon_bars", 96))
    quantiles = [float(x) for x in ml_cfg.get("quantiles", [0.5, 0.8, 0.9])]

    train_symbols = int(ml_cfg.get("train_symbols", 60))
    train_bars_per_symbol = int(ml_cfg.get("train_bars_per_symbol", 2500))
    batch_limit = int(ml_cfg.get("ohlcv_batch_limit", 720))
    sleep_sec = float(ml_cfg.get("rate_limit_sleep_sec", 1.0))

    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ML-TRAIN] Exchange={exchange_name} tf={timeframe} base={base_currency}")
    print(f"[ML-TRAIN] horizon_bars={horizon_bars} quantiles={quantiles}")
    print(f"[ML-TRAIN] symbols={train_symbols} bars_per_symbol={train_bars_per_symbol}")

    exchange = ExchangeClient(exchange_name=exchange_name, mode="paper", verbose=False)

    # Use tickers to select top quote-volume USD markets
    tickers = exchange.fetch_tickers()
    candidates: List[Tuple[str, float]] = []
    for sym, t in (tickers or {}).items():
        if "/" not in sym:
            continue
        base, quote = sym.split("/", 1)
        if quote.upper() != base_currency:
            continue
        if not exchange.is_spot_symbol(sym):
            continue
        qv = t.get("quoteVolume") or t.get("quote_volume") or 0
        try:
            qv = float(qv)
        except Exception:
            qv = 0.0
        if qv > 0:
            candidates.append((sym, qv))

    candidates.sort(key=lambda x: x[1], reverse=True)
    symbols = [s for s, _ in candidates[:train_symbols]]
    print(f"[ML-TRAIN] training on {len(symbols)} symbols (top quote volume)")

    X_all = []
    y_all = []

    feature_cols = None

    for i, sym in enumerate(symbols, start=1):
        try:
            print(f"[ML-TRAIN] ({i}/{len(symbols)}) fetch {sym} ...")
            raw = fetch_ohlcv_history(exchange, sym, timeframe, train_bars_per_symbol, batch_limit, sleep_sec)
            df = ohlcv_to_dataframe(raw)
            if df is None or df.empty or len(df) < (horizon_bars + 120):
                print(f"[ML-TRAIN] skip {sym}: not enough data ({0 if df is None else len(df)})")
                continue

            feats = build_features(df)
            y = forward_max_high_return(df, horizon_bars)

            # align and clean
            data = feats.copy()
            data["y"] = y
            data = data.dropna()

            if data.empty:
                continue

            yv = data["y"].astype(float).clip(-0.05, 2.0)  # clip extreme outliers
            X = data.drop(columns=["y"])

            if feature_cols is None:
                feature_cols = list(X.columns)
            else:
                # enforce same column order
                X = X[feature_cols]

            X_all.append(X)
            y_all.append(yv)

        except Exception as e:
            print(f"[ML-TRAIN] symbol failed {sym}: {e}")

    if not X_all:
        raise SystemExit("[ML-TRAIN] No training data collected.")

    X_train = pd.concat(X_all, axis=0).astype(float)
    y_train = pd.concat(y_all, axis=0).astype(float)

    print(f"[ML-TRAIN] rows={len(X_train)} features={len(X_train.columns)}")

    models: Dict[float, GradientBoostingRegressor] = {}

    for q in quantiles:
        print(f"[ML-TRAIN] training quantile model q={q} ...")
        mdl = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=600,              # increased from 400
            learning_rate=0.03,            # reduced from 0.05 for better convergence
            max_depth=4,                   # increased from 3 for more complexity
            min_samples_split=50,          # added for regularization
            min_samples_leaf=20,           # added for regularization
            subsample=0.8,                 # increased from 0.7
            max_features='sqrt',           # added for regularization
            random_state=42,
        )
        mdl.fit(X_train, y_train)
        models[q] = mdl

    bundle = {
        "feature_cols": feature_cols,
        "models": models,
        "horizon_bars": horizon_bars,
        "timeframe": timeframe,
        "base_currency": base_currency,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }

    joblib.dump(bundle, model_path)
    print(f"[ML-TRAIN] saved model -> {model_path}")


if __name__ == "__main__":
    main()
