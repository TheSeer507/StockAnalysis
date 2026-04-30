# bot/ml_forecaster.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from .ml_features import build_features


@dataclass
class MLPrediction:
    mfe_quantiles: Dict[float, float]   # quantile -> predicted max favorable return (e.g. 0.08 = +8%)
    tp_prices: Dict[float, float]       # quantile -> TP price
    horizon_bars: int


class MLPriceForecaster:
    """
    Predicts Max Favorable Excursion (MFE) over next horizon bars (forward max high vs current close).
    Uses quantile regression models to output a range (median/optimistic/aggressive).
    """

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.bundle = None

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        self.bundle = joblib.load(self.model_path)
        return True

    def is_ready(self) -> bool:
        return self.bundle is not None

    def predict(self, df: pd.DataFrame) -> Optional[MLPrediction]:
        if not self.is_ready():
            return None

        feat_cols: List[str] = self.bundle["feature_cols"]
        models: Dict[float, object] = self.bundle["models"]
        horizon_bars: int = int(self.bundle["horizon_bars"])

        feats = build_features(df)
        feats = feats[feat_cols].copy()

        # last row only
        x = feats.iloc[[-1]].copy()
        if x.isna().any(axis=1).iloc[0]:
            return None

        close = float(df["close"].iloc[-1])

        mfe_q: Dict[float, float] = {}
        tp_prices: Dict[float, float] = {}

        for q, mdl in models.items():
            pred = float(mdl.predict(x)[0])
            # keep sane bounds: [-5%, +200%] for meme spikes
            pred = max(-0.05, min(pred, 2.0))
            mfe_q[float(q)] = pred
            tp_prices[float(q)] = close * (1.0 + pred)

        return MLPrediction(mfe_quantiles=mfe_q, tp_prices=tp_prices, horizon_bars=horizon_bars)
