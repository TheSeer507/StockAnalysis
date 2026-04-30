# main.py
from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import yaml
import ccxt

from bot.top5_trader import (
    Exchange, PaperPortfolio, PositionsBook, TrendStrategy, BotEngine,
    bars_in_24h
)
from bot.torch_tp_forecaster import TorchTPForecaster
from bot.longterm_forecaster import LongtermForecaster


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
DATA_DIR = REPO_ROOT / "data"
PAPER_PORTFOLIO_PATH = DATA_DIR / "paper_portfolio.json"
LIVE_STATE_PATH = DATA_DIR / "live_positions.json"


def load_dotenv_simple(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and (k not in os.environ):
            os.environ[k] = v


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_ccxt_exchange(cfg: dict) -> ccxt.Exchange:
    ex_cfg = cfg.get("exchange", {}) or {}
    name = (ex_cfg.get("name") or "kraken").strip()
    mode = (cfg.get("mode") or "paper").strip().lower()

    klass = getattr(ccxt, name)
    params = {"enableRateLimit": True}

    api_key = os.getenv("KRAKEN_API_KEY") or os.getenv("KRAKEN_KEY") or ""
    api_secret = os.getenv("KRAKEN_SECRET") or os.getenv("KRAKEN_API_SECRET") or ""

    if api_key and api_secret:
        params["apiKey"] = api_key
        params["secret"] = api_secret

    ex = klass(params)
    ex.load_markets()
    return ex


def model_is_fresh(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime) <= timedelta(days=max_age_days)


def ensure_ml_forecaster(cfg: dict) -> TorchTPForecaster | None:
    ml = cfg.get("ml", {}) or {}
    if not bool(ml.get("enabled", False)):
        print("[ML] disabled (ml.enabled=false).")
        return None

    model_path = Path(ml.get("model_path", str(DATA_DIR / "torch_tp_forecaster.pt")))
    retrain_days = int(ml.get("retrain_days", 7))
    auto_train = bool(ml.get("auto_train", True))

    model_path.parent.mkdir(parents=True, exist_ok=True)

    if (not model_is_fresh(model_path, retrain_days)) and auto_train:
        print(f"[ML] model missing/stale -> training (retrain_days={retrain_days})")
        os.chdir(REPO_ROOT)
        from ml import train_torch_forecaster
        train_torch_forecaster.main()
    else:
        if model_path.exists():
            print(f"[ML] using existing model: {model_path}")
        else:
            print("[ML] model missing, auto_train=false -> continuing without ML")
            return None

    f = TorchTPForecaster(model_path)
    try:
        if f.load():
            print(f"[ML] ✅ loaded successfully: {model_path}")
            # Check if model is compatible
            if hasattr(f, 'meta'):
                input_size = f.meta.get('input_size', 24)
                if input_size != 24:
                    print(f"[ML] ⚠️  WARNING: Model has input_size={input_size}, but code expects 24 features")
                    print(f"[ML] ⚠️  Model is from old architecture - predictions may fail")
                    print(f"[ML] 🔄 RECOMMENDATION: Retrain model with: python -m ml.train_torch_forecaster")
            return f
        else:
            print("[ML][WARN] failed to load model; continuing without ML")
            return None
    except Exception as e:
        print(f"[ML][ERROR] Exception loading model: {e}")
        print("[ML][WARN] Continuing without ML. Please retrain: python -m ml.train_torch_forecaster")
        return None


def ensure_longterm_forecaster(cfg: dict) -> LongtermForecaster | None:
    """Load or train the long-term (daily) ML forecaster for portfolio holding analysis."""
    lt = cfg.get("longterm_ml", {}) or {}
    if not bool(lt.get("enabled", False)):
        print("[LT-ML] Long-term forecaster disabled (longterm_ml.enabled=false).")
        return None

    model_path = Path(lt.get("model_path", str(DATA_DIR / "longterm_forecaster.pt")))
    retrain_days = int(lt.get("retrain_days", 14))
    auto_train = bool(lt.get("auto_train", True))

    model_path.parent.mkdir(parents=True, exist_ok=True)

    if (not model_is_fresh(model_path, retrain_days)) and auto_train:
        print(f"[LT-ML] Long-term model missing/stale -> training (retrain_days={retrain_days})")
        os.chdir(REPO_ROOT)
        from ml import train_longterm_forecaster
        train_longterm_forecaster.main()
    else:
        if model_path.exists():
            print(f"[LT-ML] Using existing long-term model: {model_path}")
        else:
            print("[LT-ML] Long-term model missing, auto_train=false -> continuing without it")
            return None

    f = LongtermForecaster(model_path)
    try:
        if f.load():
            print(f"[LT-ML] ✅ Long-term forecaster loaded: {model_path}")
            if hasattr(f, 'meta'):
                horizons = f.meta.get("horizon_days", [7, 14, 30])
                lookback = f.meta.get("lookback", 90)
                print(f"[LT-ML]    Horizons: {horizons} days | Lookback: {lookback} days | Timeframe: daily")
            return f
        else:
            print("[LT-ML][WARN] Failed to load long-term model")
            return None
    except Exception as e:
        print(f"[LT-ML][ERROR] Exception: {e}")
        return None


def main():
    os.chdir(REPO_ROOT)

    load_dotenv_simple(REPO_ROOT / ".env")

    cfg = load_config()

    mode = (cfg.get("mode") or "paper").strip().lower()
    ex_cfg = cfg.get("exchange", {}) or {}
    base_currency = (ex_cfg.get("base_currency") or "USD").upper()
    main_pair = (ex_cfg.get("pair") or f"BTC/{base_currency}").upper()
    timeframe = (ex_cfg.get("timeframe") or "15m").strip()

    safety = cfg.get("safety", {}) or {}
    allow_live = bool(safety.get("allow_live_trading", False))

    execute_trades = (mode == "live") and allow_live

    print(f"[INIT] Creating exchange client for {(ex_cfg.get('name') or 'kraken')} in {mode.upper()} mode...")
    ccxt_ex = create_ccxt_exchange(cfg)
    ex = Exchange(ccxt_ex)
    print(f"[OK] Connected. Loaded {len(ex.markets)} markets.")

    forecaster = ensure_ml_forecaster(cfg)
    longterm_forecaster = ensure_longterm_forecaster(cfg)

    bot_cfg = cfg.get("bot", {}) or {}
    min_candles = int(bot_cfg.get("min_candles", 120))

    strat = TrendStrategy(
        ema_fast=int(ex_cfg.get("ema_fast", cfg.get("strategy", {}).get("ema_fast", 21)) if isinstance(cfg.get("strategy", {}), dict) else 21),
        ema_slow=int(ex_cfg.get("ema_slow", cfg.get("strategy", {}).get("ema_slow", 55)) if isinstance(cfg.get("strategy", {}), dict) else 55),
        atr_period=int(cfg.get("risk", {}).get("atr_period", 14) if isinstance(cfg.get("risk", {}), dict) else 14),
        rsi_period=14,
        rsi_min=float((cfg.get("strategy", {}) or {}).get("rsi_min", 52.0)),
        max_extension_atr=float((cfg.get("strategy", {}) or {}).get("max_extension_atr", 1.8)),
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    paper_portfolio = None
    live_book = None

    if mode == "paper":
        paper_portfolio = PaperPortfolio(PAPER_PORTFOLIO_PATH, base_currency=base_currency)
        paper_portfolio.load()
        if base_currency not in paper_portfolio.balances:
            paper_portfolio.balances[base_currency] = 0.0
        paper_portfolio.save()
    else:
        live_book = PositionsBook(LIVE_STATE_PATH)
        live_book.load()
        live_book.save()

    engine = BotEngine(
        exchange=ex,
        mode=mode,
        base_currency=base_currency,
        strategy=strat,
        config=cfg,
        paper=paper_portfolio,
        live_book=live_book,
        forecaster=forecaster,
        longterm_forecaster=longterm_forecaster,
    )

    scr = cfg.get("screener", {}) or {}
    reco_count = int(scr.get("recommendations_count", 5))
    top_volume = int(scr.get("top_volume_count", 100))
    reco_every = int(scr.get("reco_every_loops", 60))

    bars24 = bars_in_24h(timeframe)

    explore_pct = float((cfg.get("portfolio", {}) or {}).get("explore_target_pct", 0.05)) * 100.0
    stable_pct = float((cfg.get("portfolio", {}) or {}).get("stable_target_pct", 0.10)) * 100.0

    print("\n=========== BOT STATUS ===========")
    print(f"Mode:                  {mode.upper()}")
    print(f"Base currency:          {base_currency}")
    print(f"Execute trades:         {execute_trades}")
    print(f"Main pair:              {main_pair}")
    print(f"Timeframe:              {timeframe}")
    print(f"Bars (24h):             {bars24}")
    print(f"Top volume markets:     {top_volume}")
    print(f"Recommendations count:  {reco_count}")
    print(f"Reco every loops:       {reco_every}")
    print(f"Explore cap:            {explore_pct:.1f}% of equity")
    print(f"Stable floor:           {stable_pct:.1f}% of equity")
    if forecaster is not None:
        mp = (cfg.get("ml") or {}).get("model_path", "data/torch_tp_forecaster.pt")
        print(f"ML model (short-term):  {mp}")
    else:
        print("ML model (short-term):  (disabled/unavailable)")
    if longterm_forecaster is not None:
        lt_mp = (cfg.get("longterm_ml") or {}).get("model_path", "data/longterm_forecaster.pt")
        lt_horizons = longterm_forecaster.meta.get("horizon_days", [7, 14, 30])
        print(f"ML model (long-term):   {lt_mp} (horizons: {lt_horizons}d)")
    else:
        print("ML model (long-term):   (disabled/unavailable)")
    print("=================================\n")

    poll = int((cfg.get("bot", {}) or {}).get("poll_interval_sec", 60))

    portfolio_scan_every = int((cfg.get("bot", {}) or {}).get("portfolio_scan_every_loops", 10))
    rebalance_cfg = ((cfg.get("portfolio", {}) or {}).get("rebalance") or {})
    rebalance_every = int(rebalance_cfg.get("every_loops", 10))
    
    # ML performance reporting
    ml_report_every = int((cfg.get("ml_tracking", {}) or {}).get("report_every_loops", 120))  # Every 2 hours by default

    loop = 0
    while True:
        loop += 1
        now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

        try:
            ohlcv = ex.fetch_ohlcv(main_pair, timeframe=timeframe, limit=max(min_candles, 120))
            last = float(ohlcv[-1][4]) if ohlcv else 0.0
            res = strat.evaluate(ohlcv) if ohlcv else None
            sig = res.signal if res else "HOLD"

            print(f"[{now}] {main_pair} close={last:.6f} signal={sig}")

            # -------- NEWS (NEW) --------
            # Safe every loop: module itself fetches only once per hour and prints only new items
            engine.maybe_print_news()

            # Manage positions
            engine.manage_positions(timeframe=timeframe, min_candles=min_candles, execute=execute_trades)

            # Rebalance core targets periodically
            if rebalance_cfg.get("enabled", False) and (loop % max(1, rebalance_every) == 0):
                engine.rebalance_core(execute=execute_trades)

            # Portfolio scan periodically
            if loop % max(1, portfolio_scan_every) == 0:
                print(f"\n[PORTFOLIO] Scanning held positions (loop {loop})...")
                engine.scan_portfolio_signals(timeframe=timeframe, min_candles=min_candles)

            # Top-N recommendations periodically
            if loop % max(1, reco_every) == 0:
                print(f"\n[SCREENER] Generating top {reco_count} recommendations (loop {loop})...")
                recos = engine.top_recommendations(
                    timeframe=timeframe,
                    top_volume_count=top_volume,
                    reco_count=reco_count,
                    min_candles=min_candles,
                )
                if recos:
                    print(f"[SCREENER] Found {len(recos)} recommendations")
                else:
                    print("[SCREENER] No recommendations met criteria this cycle")
                engine.act_on_top_recommendations(recos, execute=execute_trades)
            
            # ML Performance Report periodically
            if loop % max(1, ml_report_every) == 0:
                engine.print_ml_performance()

        except KeyboardInterrupt:
            print("\n[EXIT] Stopping bot.")
            break
        except Exception as e:
            print(f"[ERROR] Loop exception: {e}")

        time.sleep(poll)


if __name__ == "__main__":
    main()
