# bot/ml_runtime.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
import traceback
from typing import Optional

from torch_tp_forecaster import TorchTPForecaster


@dataclass
class MLRuntimeConfig:
    enabled: bool
    auto_train: bool
    retrain_days: int
    model_path: Path
    lookback: int
    horizon_bars: int
    quantiles: tuple[float, ...]


def _parse_ml_cfg(cfg: dict) -> MLRuntimeConfig:
    ml = cfg.get("ml", {}) or {}
    enabled = bool(ml.get("enabled", False))
    auto_train = bool(ml.get("auto_train", True))
    retrain_days = int(ml.get("retrain_days", 7))
    model_path = Path(ml.get("model_path", "data/torch_tp_forecaster.pt"))
    lookback = int(ml.get("lookback", 64))
    horizon_bars = int(ml.get("horizon_bars", 96))
    quantiles = tuple(float(x) for x in (ml.get("quantiles", [0.5, 0.8, 0.9]) or [0.5, 0.8, 0.9]))
    return MLRuntimeConfig(
        enabled=enabled,
        auto_train=auto_train,
        retrain_days=retrain_days,
        model_path=model_path,
        lookback=lookback,
        horizon_bars=horizon_bars,
        quantiles=quantiles,
    )


def _model_is_fresh(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(timezone.utc) - mtime
    return age <= timedelta(days=max_age_days)


def ensure_forecaster(cfg: dict) -> tuple[Optional[TorchTPForecaster], MLRuntimeConfig]:
    """
    Returns (forecaster_or_none, ml_runtime_cfg).
    If enabled and model missing/stale and auto_train, will train once.
    """
    mlcfg = _parse_ml_cfg(cfg)

    if not mlcfg.enabled:
        print("[ML] disabled in config (ml.enabled=false).")
        return None, mlcfg

    mlcfg.model_path.parent.mkdir(parents=True, exist_ok=True)

    fresh = _model_is_fresh(mlcfg.model_path, mlcfg.retrain_days)
    if fresh:
        print(f"[ML] model is fresh (<={mlcfg.retrain_days} days): {mlcfg.model_path}")
    else:
        print(f"[ML] model missing/stale -> {mlcfg.model_path}")
        if mlcfg.auto_train:
            print("[ML] auto_train=true: training model now...")
            try:
                # Ensure we run from repo root (folder that contains ml/ and bot/)
                repo_root = Path(__file__).resolve().parents[1]
                os.chdir(repo_root)

                from ml import train_torch_forecaster  # import your trainer module
                train_torch_forecaster.main()          # trains + saves model_path from config

            except Exception:
                print("[ML][WARN] training failed; continuing without ML.")
                traceback.print_exc()
        else:
            print("[ML] auto_train=false; continuing without ML.")

    forecaster = TorchTPForecaster(mlcfg.model_path)
    if forecaster.load():
        print(f"[ML] loaded model: {mlcfg.model_path}")
        return forecaster, mlcfg

    print("[ML][WARN] could not load model; continuing without ML.")
    return None, mlcfg
