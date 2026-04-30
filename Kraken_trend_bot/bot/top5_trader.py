# bot/top5_trader.py
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import ccxt

from bot.torch_tp_forecaster import TorchTPForecaster, TPPrediction
from bot.longterm_forecaster import LongtermForecaster, LongtermPrediction
from bot.news_module import NewsModule, NewsItem
from bot.sentiment_analyzer import SentimentAnalyzer, SentimentScore
from bot.ml_performance_tracker import MLPerformanceTracker


# -----------------------------
# Helpers: timeframe math
# -----------------------------

def timeframe_to_minutes(tf: str) -> int:
    tf = (tf or "").strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 24 * 60
    # default fallback
    return 60


def bars_in_24h(tf: str) -> int:
    mins = timeframe_to_minutes(tf)
    if mins <= 0:
        return 24
    return max(1, int((24 * 60) // mins))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def canonical_asset(asset: str) -> str:
    """Normalise Kraken asset names to their base symbol.

    Examples:
        SOL03.S  -> SOL   (staked SOL variant 03)
        ETH2.S   -> ETH   (staked ETH)
        BTC.B    -> BTC   (bonded BTC)
        BTC.M    -> BTC
        ATOM21.S -> ATOM
        XRP      -> XRP   (unchanged)
    """
    import re
    a = (asset or "").strip().upper()
    if "." in a:
        a = a.split(".", 1)[0]          # SOL03.S  -> SOL03
    a = re.sub(r"\d+$", "", a) or a    # SOL03    -> SOL  (keep original if all digits)
    return a


FIAT_LIKE = {
    "USD", "EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "NZD", "SGD",
    "HKD", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "TRY", "MXN",
    "BRL", "ZAR", "RUB", "CNY",
}


# -----------------------------
# Portfolio persistence (paper)
# -----------------------------

@dataclass
class PositionMeta:
    symbol: str
    qty: float
    entry_price: float
    peak_price: float
    tp_hits: Dict[str, bool]


class PaperPortfolio:
    def __init__(self, path: Path, base_currency: str):
        self.path = Path(path)
        self.base_currency = base_currency.upper()
        self.balances: Dict[str, float] = {self.base_currency: 0.0}
        self.positions: Dict[str, PositionMeta] = {}
        self.realized_pnl: float = 0.0
        self.trade_history: List[dict] = []

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.base_currency = (data.get("base_currency") or self.base_currency).upper()
        self.balances = {k.upper(): safe_float(v) for k, v in (data.get("balances") or {}).items()}
        self.realized_pnl = safe_float(data.get("realized_pnl"), 0.0)
        self.trade_history = data.get("trade_history") or []

        self.positions = {}
        for sym, p in (data.get("positions") or {}).items():
            self.positions[sym] = PositionMeta(
                symbol=sym,
                qty=safe_float(p.get("qty")),
                entry_price=safe_float(p.get("entry_price")),
                peak_price=safe_float(p.get("peak_price")),
                tp_hits={str(k): bool(v) for k, v in (p.get("tp_hits") or {}).items()},
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "base_currency": self.base_currency,
            "balances": self.balances,
            "positions": {
                sym: {
                    "qty": p.qty,
                    "entry_price": p.entry_price,
                    "peak_price": p.peak_price,
                    "tp_hits": p.tp_hits,
                }
                for sym, p in self.positions.items()
            },
            "realized_pnl": self.realized_pnl,
            "trade_history": self.trade_history[-5000:],
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def cash(self) -> float:
        return safe_float(self.balances.get(self.base_currency), 0.0)

    def adjust_balance(self, asset: str, delta: float) -> None:
        a = canonical_asset(asset)
        self.balances[a] = safe_float(self.balances.get(a), 0.0) + float(delta)

    def set_balance(self, asset: str, value: float) -> None:
        a = canonical_asset(asset)
        self.balances[a] = float(value)

    def get_balance(self, asset: str) -> float:
        return safe_float(self.balances.get(canonical_asset(asset)), 0.0)

    def held_assets(self, stable_assets: List[str]) -> Dict[str, float]:
        out = {}
        st = {canonical_asset(x) for x in stable_assets}
        for a, v in self.balances.items():
            if safe_float(v) <= 0:
                continue
            aa = canonical_asset(a)
            if aa == self.base_currency:
                continue
            if aa in st:
                continue
            out[aa] = float(v)
        return out


# -----------------------------
# Live positions book (metadata)
# (spot balances are from exchange; cost-basis/trailing state stored locally)
# -----------------------------

class PositionsBook:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.positions: Dict[str, PositionMeta] = {}
        self.realized_pnl: float = 0.0
        self.trade_history: List[dict] = []

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.realized_pnl = safe_float(data.get("realized_pnl"), 0.0)
        self.trade_history = data.get("trade_history") or []
        self.positions = {}
        for sym, p in (data.get("positions") or {}).items():
            self.positions[sym] = PositionMeta(
                symbol=sym,
                qty=safe_float(p.get("qty")),
                entry_price=safe_float(p.get("entry_price")),
                peak_price=safe_float(p.get("peak_price")),
                tp_hits={str(k): bool(v) for k, v in (p.get("tp_hits") or {}).items()},
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "positions": {
                sym: {
                    "qty": p.qty,
                    "entry_price": p.entry_price,
                    "peak_price": p.peak_price,
                    "tp_hits": p.tp_hits,
                }
                for sym, p in self.positions.items()
            },
            "realized_pnl": self.realized_pnl,
            "trade_history": self.trade_history[-5000:],
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# -----------------------------
# Strategy (pure python, no pandas)
# -----------------------------

def ema_last(values: List[float], span: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    e = values[0]
    for v in values[1:]:
        e = alpha * v + (1.0 - alpha) * e
    return float(e)


def rsi_last(values: List[float], period: int = 14) -> float:
    if len(values) < 2:
        return 50.0
    alpha = 1.0 / float(period)
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, len(values)):
        chg = values[i] - values[i - 1]
        gain = max(chg, 0.0)
        loss = max(-chg, 0.0)
        avg_gain = alpha * gain + (1.0 - alpha) * avg_gain
        avg_loss = alpha * loss + (1.0 - alpha) * avg_loss
    if avg_loss <= 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def atr_pct_last(ohlcv: List[List[float]], period: int = 14) -> float:
    # ohlcv: [ts,o,h,l,c,v]
    if len(ohlcv) < 2:
        return 0.0
    alpha = 1.0 / float(period)
    prev_close = float(ohlcv[0][4])
    tr0 = max(float(ohlcv[0][2]) - float(ohlcv[0][3]),
              abs(float(ohlcv[0][2]) - prev_close),
              abs(float(ohlcv[0][3]) - prev_close))
    atr = tr0
    for i in range(1, len(ohlcv)):
        h = float(ohlcv[i][2])
        l = float(ohlcv[i][3])
        c = float(ohlcv[i][4])
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        atr = alpha * tr + (1.0 - alpha) * atr
        prev_close = c
    last_close = float(ohlcv[-1][4])
    if last_close <= 0:
        return 0.0
    return float(atr / last_close)


@dataclass
class StrategyResult:
    trend: str          # BULL / BEAR / FLAT
    signal: str         # LONG / EXIT / HOLD
    ema_fast: float
    ema_slow: float
    rsi: float
    atr_pct: float


class TrendStrategy:
    def __init__(self, ema_fast: int, ema_slow: int, atr_period: int = 14,
                 rsi_period: int = 14, rsi_min: float = 52.0,
                 max_extension_atr: float = 1.8):
        self.ema_fast = int(ema_fast)
        self.ema_slow = int(ema_slow)
        self.atr_period = int(atr_period)
        self.rsi_period = int(rsi_period)
        self.rsi_min = float(rsi_min)
        self.max_extension_atr = float(max_extension_atr)

    def evaluate(self, ohlcv: List[List[float]]) -> StrategyResult:
        closes = [float(r[4]) for r in ohlcv]
        if len(closes) < max(self.ema_fast, self.ema_slow, 20):
            return StrategyResult("FLAT", "HOLD", 0.0, 0.0, 50.0, 0.0)

        ef = ema_last(closes[-max(300, self.ema_fast * 4):], self.ema_fast)
        es = ema_last(closes[-max(300, self.ema_slow * 4):], self.ema_slow)
        rsi = rsi_last(closes[-max(200, self.rsi_period * 4):], self.rsi_period)
        atrp = atr_pct_last(ohlcv[-max(200, self.atr_period * 4):], self.atr_period)

        close = closes[-1]
        trend = "BULL" if ef > es else "BEAR" if ef < es else "FLAT"

        # extension guard: if price too far above EMA fast (measured by ATR), skip LONG
        atr_abs = atrp * close
        extended = False
        if atr_abs > 0:
            extended = (close - ef) > (self.max_extension_atr * atr_abs)

        if trend == "BULL" and close > ef and (rsi >= self.rsi_min) and (not extended):
            sig = "LONG"
        elif trend == "BEAR" and close < es:
            sig = "EXIT"
        else:
            sig = "HOLD"

        return StrategyResult(trend, sig, ef, es, rsi, atrp)


# -----------------------------
# Exchange wrapper (ccxt)
# -----------------------------

class Exchange:
    def __init__(self, ccxt_exchange: ccxt.Exchange):
        self.ex = ccxt_exchange
        self.ex.load_markets()

    @property
    def markets(self) -> dict:
        return getattr(self.ex, "markets", {}) or {}

    def fetch_tickers(self) -> dict:
        return self.ex.fetch_tickers()

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[list]:
        print(f"[DATA] Fetching OHLCV for {symbol} ({timeframe}, limit={limit})...")
        return self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_balance(self) -> dict:
        return self.ex.fetch_balance()

    def ticker_last(self, symbol: str) -> float:
        t = self.ex.fetch_ticker(symbol)
        return safe_float(t.get("last") or t.get("close") or 0.0)

    def symbol_exists(self, symbol: str) -> bool:
        return symbol in self.markets

    def is_spot(self, symbol: str) -> bool:
        m = self.markets.get(symbol) or {}
        if "spot" in m:
            return bool(m.get("spot"))
        return not (m.get("swap") or m.get("future") or m.get("option"))

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        try:
            s = self.ex.amount_to_precision(symbol, amount)
            return float(s)
        except Exception:
            return float(amount)

    def cost_min(self, symbol: str) -> float:
        m = self.markets.get(symbol) or {}
        lim = (m.get("limits") or {}).get("cost") or {}
        return safe_float(lim.get("min"), 0.0)

    def amount_min(self, symbol: str) -> float:
        m = self.markets.get(symbol) or {}
        lim = (m.get("limits") or {}).get("amount") or {}
        return safe_float(lim.get("min"), 0.0)

    def create_market_buy(self, symbol: str, qty: float) -> dict:
        return self.ex.create_market_buy_order(symbol, qty)

    def create_market_sell(self, symbol: str, qty: float) -> dict:
        return self.ex.create_market_sell_order(symbol, qty)


# -----------------------------
# Recommendations
# -----------------------------

@dataclass
class Recommendation:
    symbol: str
    base: str
    quote: str
    last: float
    chg24: float
    qv: float
    trend: str
    signal: str
    action: str  # BUY / COMPOUND / HOLD

    ml_tp50: Optional[float] = None
    ml_tp80: Optional[float] = None
    ml_tp90: Optional[float] = None
    ml_ret80: Optional[float] = None
    
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    sentiment_news_count: Optional[int] = None
    news_headlines: Optional[List[str]] = None


def split_symbol(sym: str) -> Tuple[str, str]:
    if "/" not in sym:
        return sym, ""
    a, b = sym.split("/", 1)
    return a.upper(), b.upper()


def compute_24h_change(ohlcv: List[List[float]], bars24: int) -> float:
    if not ohlcv:
        return 0.0
    if len(ohlcv) <= bars24:
        first = float(ohlcv[0][4])
    else:
        first = float(ohlcv[-bars24][4])
    last = float(ohlcv[-1][4])
    if first <= 0:
        return 0.0
    return float(((last / first) - 1.0) * 100.0)


def is_crypto_base(base: str, stable_assets: List[str]) -> bool:
    b = canonical_asset(base)
    if b in FIAT_LIKE:
        return False
    if b in {canonical_asset(x) for x in stable_assets}:
        return False
    return True


# -----------------------------
# Bot Engine (monitor + top-5 + position mgmt)
# -----------------------------

class BotEngine:
    def __init__(
        self,
        exchange: Exchange,
        mode: str,
        base_currency: str,
        strategy: TrendStrategy,
        config: dict,
        paper: Optional[PaperPortfolio] = None,
        live_book: Optional[PositionsBook] = None,
        forecaster: Optional[TorchTPForecaster] = None,
        longterm_forecaster: Optional[LongtermForecaster] = None,
    ):
        self.exchange = exchange
        self.mode = mode.lower()
        self.base_currency = base_currency.upper()
        self.strategy = strategy
        self.cfg = config
        self.paper = paper
        self.live_book = live_book
        self.forecaster = forecaster
        self.longterm_forecaster = longterm_forecaster

        # cache tickers for equity calc to avoid hammering
        self._ticker_cache: Dict[str, dict] = {}
        self._ticker_cache_ts: float = 0.0

        # BTC OHLCV cache for cross-asset beta feature (slot 23)
        self._btc_ohlcv_cache: Optional[List[list]] = None
        self._btc_ohlcv_cache_ts: float = 0.0
        self._btc_ohlcv_cache_tf: str = ""

        # -----------------
        # News module (NEW)
        # -----------------
        self.news: Optional[NewsModule] = None
        news_cfg = (self.cfg.get("news", {}) or {})
        if bool(news_cfg.get("enabled", False)):
            self.news = NewsModule(news_cfg)
        
        # Sentiment analyzer
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        sentiment_cfg = (self.cfg.get("sentiment", {}) or {})
        if bool(sentiment_cfg.get("enabled", False)):
            self.sentiment_analyzer = SentimentAnalyzer()
            print("[SENTIMENT] Sentiment analyzer enabled")
        
        # Cache for recent news items for sentiment analysis
        self._recent_news_cache: List[NewsItem] = []
        
        # ML Performance Tracker
        self.ml_tracker: Optional[MLPerformanceTracker] = None
        ml_tracking_cfg = (self.cfg.get("ml_tracking", {}) or {})
        if bool(ml_tracking_cfg.get("enabled", False)) and self.forecaster is not None:
            tracker_path = Path(ml_tracking_cfg.get("data_path", "data/ml_predictions.json"))
            self.ml_tracker = MLPerformanceTracker(tracker_path)
            print("[ML-TRACKER] ML performance tracking enabled")

    # -----------------
    # Balances / equity
    # -----------------

    def _refresh_tickers(self, max_age_sec: int = 15) -> None:
        now = time.time()
        if self._ticker_cache and (now - self._ticker_cache_ts) < max_age_sec:
            return
        self._ticker_cache = self.exchange.fetch_tickers() or {}
        self._ticker_cache_ts = now

    def _get_btc_ohlcv(self, timeframe: str, limit: int = 200) -> Optional[List[list]]:
        """Fetch (and cache) BTC OHLCV for the cross-asset beta feature.

        Cached for 5 minutes to avoid hammering the exchange.
        """
        now = time.time()
        if (
            self._btc_ohlcv_cache
            and self._btc_ohlcv_cache_tf == timeframe
            and (now - self._btc_ohlcv_cache_ts) < 300  # 5 min cache
        ):
            return self._btc_ohlcv_cache

        btc_sym = f"BTC/{self.base_currency}"
        if not self.exchange.symbol_exists(btc_sym):
            return None
        try:
            data = self.exchange.fetch_ohlcv(btc_sym, timeframe=timeframe, limit=limit)
            if data and len(data) > 1:
                self._btc_ohlcv_cache = data
                self._btc_ohlcv_cache_ts = now
                self._btc_ohlcv_cache_tf = timeframe
                return data
        except Exception:
            pass
        return self._btc_ohlcv_cache  # return stale cache on error

    def _find_market_for_asset(self, asset: str) -> Optional[str]:
        a = canonical_asset(asset)
        s1 = f"{a}/{self.base_currency}"
        if self.exchange.symbol_exists(s1) and self.exchange.is_spot(s1):
            return s1
        return None

    def _price_for_asset(self, asset: str) -> Optional[float]:
        sym = self._find_market_for_asset(asset)
        if not sym:
            return None
        self._refresh_tickers()
        t = self._ticker_cache.get(sym) or {}
        last = safe_float(t.get("last") or t.get("close") or 0.0, 0.0)
        if last > 0:
            return last
        try:
            return self.exchange.ticker_last(sym)
        except Exception:
            return None

    def current_balances(self) -> Dict[str, float]:
        """Return total balances (free + staked) grouped by canonical asset."""
        if self.mode == "paper":
            assert self.paper is not None
            return dict(self.paper.balances)

        bal = self.exchange.fetch_balance() or {}
        totals = bal.get("total") or {}
        out: Dict[str, float] = {}
        for k, v in totals.items():
            vv = safe_float(v, 0.0)
            if vv > 0:
                ca = canonical_asset(k)
                out[ca] = out.get(ca, 0.0) + vv   # sum staked + free
        return out

    def free_balances(self) -> Dict[str, float]:
        """Return only *tradable* (non-staked) balances grouped by canonical asset.

        Staked entries on Kraken appear with a dot-suffix (e.g. SOL.S)
        and are excluded here so the bot never tries to sell locked coins.
        """
        if self.mode == "paper":
            return self.current_balances()          # paper has no staking concept

        bal = self.exchange.fetch_balance() or {}
        totals = bal.get("total") or {}
        out: Dict[str, float] = {}
        for k, v in totals.items():
            if "." in k:                              # skip SOL.S, ETH2.S, etc.
                continue
            vv = safe_float(v, 0.0)
            if vv > 0:
                ca = canonical_asset(k)
                out[ca] = out.get(ca, 0.0) + vv
        return out

    def cash(self) -> float:
        b = self.current_balances()
        return safe_float(b.get(self.base_currency), 0.0)

    def held_assets(self) -> Dict[str, float]:
        stable_assets = (self.cfg.get("portfolio", {}) or {}).get("stable_assets", ["USD", "USDT", "USDC", "USDG"])
        b = self.current_balances()
        st = {canonical_asset(x) for x in stable_assets}
        out = {}
        for a, v in b.items():
            if safe_float(v) <= 0:
                continue
            aa = canonical_asset(a)
            if aa == self.base_currency:
                continue
            if aa in st:
                continue
            out[aa] = float(v)
        return out

    def equity(self) -> float:
        bals = self.current_balances()
        eq = safe_float(bals.get(self.base_currency), 0.0)

        stable_assets = (self.cfg.get("portfolio", {}) or {}).get("stable_assets", ["USD", "USDT", "USDC", "USDG"])
        st = {canonical_asset(x) for x in stable_assets}
        self._refresh_tickers()

        for asset, qty in bals.items():
            a = canonical_asset(asset)
            if a == self.base_currency:
                continue
            if a in st:
                eq += float(qty)
                continue
            px = self._price_for_asset(a)
            if px is None:
                continue
            eq += float(qty) * float(px)

        return float(eq)

    # -----------------
    # News printing (NEW)
    # -----------------
    
    def _get_top_volume_bases(self, count: int = 20) -> List[str]:
        """Get base assets from top volume pairs for news fetching."""
        cfg_port = self.cfg.get("portfolio", {}) or {}
        stable_assets = cfg_port.get("stable_assets", ["USD", "USDT", "USDC", "USDG"])
        
        candidates = []
        for sym, t in self._ticker_cache.items():
            if "/" not in sym:
                continue
            base, quote = split_symbol(sym)
            if quote != self.base_currency:
                continue
            if not is_crypto_base(base, stable_assets):
                continue
            
            qv = safe_float(t.get("quoteVolume") or t.get("quote_volume") or 0.0, 0.0)
            if qv > 0:
                candidates.append((base, qv))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [base for base, _ in candidates[:count]]

    def maybe_print_news(self) -> None:
        """
        Safe to call every loop:
          - fetches at most once per refresh_sec (e.g., hourly)
          - prints only NEW items (deduped via data/news_state.json)
          - updates sentiment cache
        """
        if self.news is None:
            return

        port = (self.cfg.get("portfolio", {}) or {})
        core_targets = (port.get("core_targets", {}) or {})
        allowed_explore = (port.get("allowed_explore_assets", []) or [])

        held = self.held_assets()

        assets = set()
        for a in held.keys():
            assets.add(str(a).upper())
        for a in core_targets.keys():
            assets.add(str(a).upper())
        for a in allowed_explore:
            assets.add(str(a).upper())
        
        # Also fetch news for top volume assets (for recommendations)
        self._refresh_tickers(max_age_sec=60)
        top_volume_assets = self._get_top_volume_bases(count=20)
        for asset in top_volume_assets:
            assets.add(str(asset).upper())

        items = self.news.poll_and_get_new(sorted(assets))
        
        # Update sentiment cache with new items
        if items:
            self._recent_news_cache.extend(items)
            # Keep only recent news (last 200 items)
            if len(self._recent_news_cache) > 200:
                self._recent_news_cache = self._recent_news_cache[-200:]
            print(f"[NEWS] Added {len(items)} new items to cache (total cached: {len(self._recent_news_cache)})")
            
            # Show what assets these news items are about
            asset_mentions = {}
            for item in items:
                for asset in item.matched_assets:
                    asset_mentions[asset] = asset_mentions.get(asset, 0) + 1
            if asset_mentions:
                top_mentions = sorted(asset_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"[NEWS] Top asset mentions: {', '.join([f'{a}({c})' for a, c in top_mentions])}")
        
        self.news.print_new_items(items)
    
    # -----------------
    # ML Performance Tracking
    # -----------------
    
    def print_ml_performance(self) -> None:
        """Print ML model performance metrics."""
        if self.ml_tracker is None:
            print("[ML-TRACKER] Performance tracking not enabled (ml_tracker is None)")
            return
        
        # Get metrics to show diagnostic info
        metrics = self.ml_tracker.get_metrics(days_lookback=None)
        
        # Show quick summary before full report
        print(f"\n[ML-TRACKER] Quick Summary:")
        print(f"  Total Predictions Logged: {metrics.total_predictions}")
        print(f"  Predictions with Outcomes: {metrics.predictions_with_outcomes}")
        
        if metrics.predictions_with_outcomes > 0:
            print(f"  TP80 Hit Rate: {metrics.tp80_hit_rate*100:.1f}%")
        else:
            print(f"  Note: No completed outcomes yet. Outcomes update when positions reach peaks.")
            print(f"        Keep running for 24-48 hours for first results.")
        
        # Full report
        report = self.ml_tracker.generate_report(days=30)  # Last 30 days
        print(report)
        
        # Check if retraining is recommended
        if metrics.predictions_with_outcomes >= 50:
            should_retrain, reason = self.ml_tracker.should_retrain(min_samples=50)
            if should_retrain:
                print(f"[ML-TRACKER][ALERT] Model retraining recommended: {reason}\n")

    # -----------------
    # Position metadata
    # -----------------

    def _book(self) -> Any:
        if self.mode == "paper":
            return self.paper
        return self.live_book

    def get_position_meta(self, symbol: str) -> Optional[PositionMeta]:
        book = self._book()
        if book is None:
            return None
        return (book.positions or {}).get(symbol)

    def set_position_meta(self, meta: PositionMeta) -> None:
        book = self._book()
        if book is None:
            return
        book.positions[meta.symbol] = meta

    def delete_position_meta(self, symbol: str) -> None:
        book = self._book()
        if book is None:
            return
        if symbol in book.positions:
            del book.positions[symbol]

    def persist(self) -> None:
        if self.mode == "paper":
            assert self.paper is not None
            self.paper.save()
        else:
            assert self.live_book is not None
            self.live_book.save()

    # -----------------
    # Trading primitives
    # -----------------

    def _normalize_qty(self, symbol: str, qty: float, price: float) -> float:
        qty = max(0.0, float(qty))
        if qty <= 0:
            return 0.0
        q = self.exchange.amount_to_precision(symbol, qty)
        if q <= 0:
            return 0.0

        min_amt = self.exchange.amount_min(symbol)
        if min_amt > 0 and q < min_amt:
            return 0.0

        min_cost = self.exchange.cost_min(symbol)
        if min_cost > 0 and (q * price) < min_cost:
            return 0.0

        return q

    def _paper_buy(self, symbol: str, notional: float, price: float) -> float:
        assert self.paper is not None
        base, quote = split_symbol(symbol)
        cash = self.paper.get_balance(quote)
        spend = min(float(notional), float(cash))
        if spend <= 0:
            return 0.0
        qty = spend / float(price)
        qty = self._normalize_qty(symbol, qty, price)
        if qty <= 0:
            return 0.0

        cost = qty * float(price)
        self.paper.adjust_balance(quote, -cost)
        self.paper.adjust_balance(base, +qty)

        pm = self.get_position_meta(symbol)
        if pm is None:
            pm = PositionMeta(symbol=symbol, qty=qty, entry_price=price, peak_price=price, tp_hits={})
        else:
            new_qty = pm.qty + qty
            if new_qty > 0:
                pm.entry_price = (pm.entry_price * pm.qty + price * qty) / new_qty
            pm.qty = new_qty
            pm.peak_price = max(pm.peak_price, price)
        self.set_position_meta(pm)

        self.paper.trade_history.append({
            "ts": time.time(),
            "mode": "paper",
            "side": "buy",
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "cost": cost,
        })
        self.persist()
        return qty

    def _paper_sell(self, symbol: str, qty: float, price: float) -> float:
        assert self.paper is not None
        base, quote = split_symbol(symbol)
        have = self.paper.get_balance(base)
        sell_qty = min(float(qty), float(have))
        if sell_qty <= 0:
            return 0.0
        sell_qty = self._normalize_qty(symbol, sell_qty, price)
        if sell_qty <= 0:
            return 0.0

        proceeds = sell_qty * float(price)
        self.paper.adjust_balance(base, -sell_qty)
        self.paper.adjust_balance(quote, +proceeds)

        pm = self.get_position_meta(symbol)
        if pm is not None:
            entry = pm.entry_price
            self.paper.realized_pnl += (price - entry) * sell_qty
            pm.qty = max(0.0, pm.qty - sell_qty)
            if pm.qty <= 1e-12:
                self.delete_position_meta(symbol)
            else:
                self.set_position_meta(pm)

        self.paper.trade_history.append({
            "ts": time.time(),
            "mode": "paper",
            "side": "sell",
            "symbol": symbol,
            "qty": sell_qty,
            "price": price,
            "proceeds": proceeds,
        })
        self.persist()
        return sell_qty

    def _live_buy(self, symbol: str, notional: float, price: float, execute: bool) -> float:
        base, quote = split_symbol(symbol)
        if not execute:
            print(f"[TRADE][DRY] BUY {symbol} notional={notional:.2f} {quote}")
            return 0.0

        qty = float(notional) / float(price)
        qty = self._normalize_qty(symbol, qty, price)
        if qty <= 0:
            return 0.0

        try:
            self.exchange.create_market_buy(symbol, qty)
            pm = self.get_position_meta(symbol)
            if pm is None:
                pm = PositionMeta(symbol=symbol, qty=qty, entry_price=price, peak_price=price, tp_hits={})
            else:
                new_qty = pm.qty + qty
                if new_qty > 0:
                    pm.entry_price = (pm.entry_price * pm.qty + price * qty) / new_qty
                pm.qty = new_qty
                pm.peak_price = max(pm.peak_price, price)
            self.set_position_meta(pm)

            book = self._book()
            book.trade_history.append({
                "ts": time.time(),
                "mode": "live",
                "side": "buy",
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "cost": qty * price,
            })
            self.persist()
            print(f"[TRADE] BUY {symbol} qty={qty:.8f} @~{price:.6f}")
            return qty

        except ccxt.InsufficientFunds:
            print(f"[TRADE][SKIP] BUY {symbol}: insufficient funds")
            return 0.0
        except Exception as e:
            msg = str(e)
            if "Insufficient funds" in msg or "EOrder:Insufficient funds" in msg:
                print(f"[TRADE][SKIP] BUY {symbol}: insufficient funds")
                return 0.0
            print(f"[TRADE][ERROR] BUY {symbol}: {e}")
            return 0.0

    def _live_sell(self, symbol: str, qty: float, price: float, execute: bool) -> float:
        if qty <= 0:
            return 0.0
        if not execute:
            print(f"[TRADE][DRY] SELL {symbol} qty={qty:.8f}")
            return 0.0

        # Cap at tradable (non-staked) balance so we never try to sell locked coins
        base, _quote = split_symbol(symbol)
        free = self.free_balances()
        tradable = safe_float(free.get(canonical_asset(base)), 0.0)
        if tradable <= 0:
            print(f"[TRADE][SKIP] SELL {symbol}: no tradable (non-staked) balance")
            return 0.0
        qty = min(qty, tradable)

        qty = self._normalize_qty(symbol, qty, price)
        if qty <= 0:
            return 0.0

        try:
            self.exchange.create_market_sell(symbol, qty)

            pm = self.get_position_meta(symbol)
            if pm is not None:
                entry = pm.entry_price
                book = self._book()
                book.realized_pnl += (price - entry) * qty
                pm.qty = max(0.0, pm.qty - qty)
                if pm.qty <= 1e-12:
                    self.delete_position_meta(symbol)
                else:
                    self.set_position_meta(pm)

            book = self._book()
            book.trade_history.append({
                "ts": time.time(),
                "mode": "live",
                "side": "sell",
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "proceeds": qty * price,
            })
            self.persist()
            print(f"[TRADE] SELL {symbol} qty={qty:.8f} @~{price:.6f}")
            return qty

        except ccxt.InsufficientFunds:
            print(f"[TRADE][SKIP] SELL {symbol}: insufficient funds")
            return 0.0
        except Exception as e:
            msg = str(e)
            if "Insufficient funds" in msg or "EOrder:Insufficient funds" in msg:
                print(f"[TRADE][SKIP] SELL {symbol}: insufficient funds")
                return 0.0
            print(f"[TRADE][ERROR] SELL {symbol}: {e}")
            return 0.0

    def buy(self, symbol: str, notional: float, price: float, execute: bool) -> float:
        if self.mode == "paper":
            return self._paper_buy(symbol, notional, price)
        return self._live_buy(symbol, notional, price, execute)

    def sell(self, symbol: str, qty: float, price: float, execute: bool) -> float:
        if self.mode == "paper":
            return self._paper_sell(symbol, qty, price)
        return self._live_sell(symbol, qty, price, execute)

    # -----------------
    # Position management: stops / TP / trailing
    # -----------------

    def manage_positions(self, timeframe: str, min_candles: int, execute: bool) -> None:
        pm_cfg = (self.cfg.get("position_management", {}) or {})
        if not bool(pm_cfg.get("manage_existing_positions", True)):
            return

        stable_assets = (self.cfg.get("portfolio", {}) or {}).get("stable_assets", ["USD", "USDT", "USDC", "USDG"])
        stops_abs = ((self.cfg.get("portfolio", {}) or {}).get("stops") or {})
        stop_loss_cfg = (pm_cfg.get("stop_loss", {}) or {})
        tp_cfg = (pm_cfg.get("take_profit", {}) or {})
        trail_cfg = (pm_cfg.get("trailing_stop", {}) or {})

        sl_enabled = bool(stop_loss_cfg.get("enabled", True))
        sl_pct = float(stop_loss_cfg.get("stop_pct", 6.0))

        tp_enabled = bool(tp_cfg.get("enabled", True))
        tp_levels = tp_cfg.get("levels") or []

        tr_enabled = bool(trail_cfg.get("enabled", True))
        arm_after = float(trail_cfg.get("arm_after_pct", 2.0))
        trail_pct = float(trail_cfg.get("trail_pct", 3.0))

        held = self.held_assets()
        for asset, qty in held.items():
            if qty <= 0:
                continue
            if canonical_asset(asset) in {canonical_asset(x) for x in stable_assets}:
                continue

            sym = self._find_market_for_asset(asset)
            if not sym:
                continue

            try:
                ohlcv = self.exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=max(min_candles, 120))
                if not ohlcv:
                    continue
                last = float(ohlcv[-1][4])
                base, quote = split_symbol(sym)

                pm = self.get_position_meta(sym)
                if pm is None:
                    pm = PositionMeta(symbol=sym, qty=float(qty), entry_price=last, peak_price=last, tp_hits={})
                    self.set_position_meta(pm)
                    self.persist()

                pm.qty = float(qty)
                pm.peak_price = max(pm.peak_price, last)
                
                # Update ML tracker with peak price (for outcome tracking)
                # Record ALL outcomes, not just profitable ones
                if self.ml_tracker:
                    self.ml_tracker.update_outcome(
                        symbol=sym,
                        entry_price=pm.entry_price,
                        actual_peak_price=pm.peak_price,
                        time_window_hours=168.0  # 7 days
                    )

                abs_stop = stops_abs.get(base)
                if abs_stop is not None:
                    if last <= float(abs_stop):
                        print(f"[RISK] ABS STOP hit for {sym}: last={last:.6f} <= stop={float(abs_stop):.6f} -> SELL ALL")
                        self.sell(sym, pm.qty, last, execute)
                        continue

                if sl_enabled and pm.entry_price > 0:
                    ret = (last / pm.entry_price - 1.0) * 100.0
                    if ret <= (-sl_pct):
                        print(f"[RISK] STOP-LOSS hit for {sym}: pnl={ret:.2f}% <= -{sl_pct:.2f}% -> SELL ALL")
                        self.sell(sym, pm.qty, last, execute)
                        continue

                if tp_enabled and pm.entry_price > 0 and pm.qty > 0:
                    ret = (last / pm.entry_price - 1.0) * 100.0
                    for lvl in tp_levels:
                        pct = float(lvl.get("pct"))
                        sell_pct = float(lvl.get("sell_pct"))
                        key = str(pct)
                        if ret >= pct and not pm.tp_hits.get(key, False):
                            part = pm.qty * (sell_pct / 100.0)
                            print(f"[TP] {sym} hit +{pct:.1f}% -> SELL {sell_pct:.0f}% (qty={part:.8f})")
                            sold = self.sell(sym, part, last, execute)
                            if sold > 0:
                                pm.tp_hits[key] = True
                                self.set_position_meta(pm)
                                self.persist()

                if tr_enabled and pm.entry_price > 0 and pm.qty > 0:
                    ret = (last / pm.entry_price - 1.0) * 100.0
                    if ret >= arm_after:
                        trail_price = pm.peak_price * (1.0 - trail_pct / 100.0)
                        if last <= trail_price:
                            print(f"[TRAIL] {sym} trailed out: last={last:.6f} <= trail={trail_price:.6f} -> SELL ALL")
                            self.sell(sym, pm.qty, last, execute)
                            continue

                self.set_position_meta(pm)
                self.persist()

            except Exception:
                continue

    # -----------------
    # Portfolio scan: signals for held assets
    # -----------------

    def scan_portfolio_signals(self, timeframe: str, min_candles: int) -> None:
        held = self.held_assets()
        if not held:
            print("[PORTFOLIO] No held crypto assets to scan.")
            return

        # Fetch free (tradable) balances to show staking breakdown
        free = self.free_balances()

        print("\n[PORTFOLIO] ══════════ Portfolio Long-Term Analysis ══════════")
        if self.longterm_forecaster is not None:
            print("[PORTFOLIO] Long-term ML forecaster: ✅ ACTIVE (daily data, 7/14/30d horizons)")
        else:
            print("[PORTFOLIO] Long-term ML forecaster: ❌ NOT LOADED (run: python -m ml.train_longterm_forecaster)")
        if self.sentiment_analyzer is not None:
            print(f"[PORTFOLIO] Sentiment analyzer enabled, news cache: {len(self._recent_news_cache)} items")
        
        for asset, qty in held.items():
            sym = self._find_market_for_asset(asset)
            if not sym:
                print(f"[PORTFOLIO] No market found for asset '{asset}', skipping.")
                continue
            try:
                # Short-term data (15m) for current signal
                ohlcv = self.exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=max(min_candles, 120))
                if not ohlcv:
                    continue
                last = float(ohlcv[-1][4])
                res = self.strategy.evaluate(ohlcv)

                pm = self.get_position_meta(sym)
                entry_price = pm.entry_price if pm else last
                pnl_pct = (last / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0
                value = qty * last

                # Staking breakdown
                free_qty = safe_float(free.get(asset), 0.0)
                staked_qty = qty - free_qty
                stake_tag = ""
                if staked_qty > 1e-8:
                    if free_qty < 1e-8:
                        stake_tag = "  🔒 100% STAKED"
                    else:
                        pct = staked_qty / qty * 100.0
                        stake_tag = f"  🔒 {pct:.0f}% staked (tradable: {free_qty:.8f})"

                print(f"\n  ┌─ {asset} ({sym}){stake_tag}")
                print(f"  │  Qty: {qty:.8f}  Price: {last:.6f}  Value: {value:.2f} {self.base_currency}")
                print(f"  │  Entry: {entry_price:.6f}  PnL: {pnl_pct:+.2f}%  Trend(15m): {res.trend}  Signal: {res.signal}")

                # ──── LONG-TERM ML ANALYSIS ────
                if self.longterm_forecaster is not None:
                    try:
                        # Fetch daily candles for long-term analysis
                        lt_cfg = self.cfg.get("longterm_ml", {}) or {}
                        lt_lookback = int(lt_cfg.get("lookback", 90))
                        daily_ohlcv = self.exchange.fetch_ohlcv(
                            sym, timeframe="1d", limit=max(lt_lookback + 210, 300)
                        )
                        if daily_ohlcv and len(daily_ohlcv) >= 210:
                            lt_pred = self.longterm_forecaster.predict(daily_ohlcv, lookback=lt_lookback)
                            if lt_pred is not None:
                                print(f"  │  ── Long-Term ML Outlook ──")
                                print(f"  │  Trend Phase: {lt_pred.trend_phase}  |  Conviction: {lt_pred.conviction} ({lt_pred.conviction_score:.0%})")
                                for hp in lt_pred.horizons:
                                    med = hp.mfe_returns.get(0.5, 0.0)
                                    tp80 = hp.tp_prices.get(0.8, last)
                                    tp90 = hp.tp_prices.get(0.9, last)
                                    print(f"  │    {hp.horizon_days:2d}d forecast: median MFE {med:+.1%}  TP80={tp80:.2f}  TP90={tp90:.2f}")
                                
                                # Holding recommendation
                                if lt_pred.conviction == "STRONG_HOLD":
                                    print(f"  │  📈 RECOMMENDATION: STRONG HOLD — Let this position run")
                                elif lt_pred.conviction == "HOLD":
                                    print(f"  │  ✅ RECOMMENDATION: HOLD — Positive multi-week outlook")
                                elif lt_pred.conviction == "WEAK":
                                    print(f"  │  ⚠️  RECOMMENDATION: WEAK — Consider tightening stops")
                                else:
                                    print(f"  │  🔴 RECOMMENDATION: EXIT — Negative outlook, reduce exposure")
                            else:
                                print(f"  │  Long-Term ML: insufficient daily data for prediction")
                        else:
                            print(f"  │  Long-Term ML: need ≥210 daily candles (got {len(daily_ohlcv) if daily_ohlcv else 0})")
                    except Exception as e:
                        print(f"  │  Long-Term ML: error: {e}")

                # ──── SHORT-TERM ML (swing trade targets) ────
                if self.forecaster is not None:
                    btc_ohlcv = self._get_btc_ohlcv(timeframe)
                    pred = self.forecaster.predict(
                        ohlcv,
                        lookback=int((self.cfg.get("ml") or {}).get("lookback", 64)),
                        horizon_bars=bars_in_24h(timeframe),
                        btc_ohlcv=btc_ohlcv,
                    )
                    if pred is not None:
                        tp50 = pred.tp_prices.get(0.5)
                        tp80 = pred.tp_prices.get(0.8)
                        tp90 = pred.tp_prices.get(0.9)
                        ret80 = pred.mfe_returns.get(0.8, 0.0)
                        if tp50 is not None and tp80 is not None and tp90 is not None:
                            print(f"  │  ── Short-Term ML (next 24h) ──")
                            print(f"  │    TP50={tp50:.6f}  TP80={tp80:.6f}  TP90={tp90:.6f}  (ret80={ret80:+.1%})")
                            if ret80 > 0.06:
                                print(f"  │    💡 Short-term swing opportunity: {ret80:.1%} upside in 24h")
                
                # ──── SENTIMENT ────
                if self.sentiment_analyzer is not None and self._recent_news_cache:
                    sentiment = self.sentiment_analyzer.analyze_asset_sentiment(
                        asset, self._recent_news_cache, lookback_hours=24.0
                    )
                    if sentiment.news_count > 0:
                        label = self.sentiment_analyzer.get_sentiment_label(sentiment.score)
                        print(f"  │  Sentiment: {label} ({sentiment.score:+.2f}, {sentiment.news_count} news)")
                        if sentiment.recent_headlines:
                            for idx, headline in enumerate(sentiment.recent_headlines[:2], 1):
                                print(f"  │    📰 {idx}. {headline}")
                
                print(f"  └─")

            except Exception as e:
                print(f"[PORTFOLIO][ERROR] Failed to scan {asset}: {e}")
                continue
        print("[PORTFOLIO] ══════════════════════════════════════════════════\n")

    # -----------------
    # Top-5 recommendations with ML
    # -----------------

    def top_recommendations(
        self,
        timeframe: str,
        top_volume_count: int,
        reco_count: int,
        min_candles: int,
    ) -> List[Recommendation]:
        cfg_port = self.cfg.get("portfolio", {}) or {}
        stable_assets = cfg_port.get("stable_assets", ["USD", "USDT", "USDC", "USDG"])
        quotes = (self.base_currency,)

        self._refresh_tickers(max_age_sec=5)
        tickers = self._ticker_cache

        candidates = []
        for sym, t in tickers.items():
            if "/" not in sym:
                continue
            base, quote = split_symbol(sym)
            if quote not in quotes:
                continue
            if not self.exchange.symbol_exists(sym) or not self.exchange.is_spot(sym):
                continue
            if not is_crypto_base(base, stable_assets):
                continue

            qv = safe_float(t.get("quoteVolume") or t.get("quote_volume") or 0.0, 0.0)
            if qv <= 0:
                continue
            candidates.append((sym, qv))

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[: max(10, int(top_volume_count))]
        
        print(f"[SCREENER] Evaluating {len(candidates)} high-volume symbols...")
        if self.sentiment_analyzer is not None:
            print(f"[SCREENER] Sentiment analyzer enabled, news cache: {len(self._recent_news_cache)} items")
        else:
            print("[SCREENER] Sentiment analyzer disabled")

        held_assets = self.held_assets()
        held_set = set(held_assets.keys())

        bars24 = bars_in_24h(timeframe)

        ml_cfg = self.cfg.get("ml", {}) or {}
        ml_lookback = int(ml_cfg.get("lookback", 64))
        min_tp80_pct = float(ml_cfg.get("min_tp80_pct", 0.06))

        # Pre-fetch BTC OHLCV once for cross-asset beta (slot 23)
        btc_ohlcv = self._get_btc_ohlcv(timeframe) if self.forecaster is not None else None

        pm_cfg = self.cfg.get("position_management", {}) or {}
        require_long_for_entry = bool(pm_cfg.get("require_long_for_entry", True))
        allow_compound = bool(pm_cfg.get("allow_compound", True))

        recos: List[Recommendation] = []
        for sym, qv in candidates:
            try:
                ohlcv = self.exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=max(min_candles, bars24))
                if not ohlcv or len(ohlcv) < max(30, bars24):
                    continue

                last = float(ohlcv[-1][4])
                chg = compute_24h_change(ohlcv, bars24)
                res = self.strategy.evaluate(ohlcv)
                base, quote = split_symbol(sym)

                action = "HOLD"

                tp_pred: Optional[TPPrediction] = None
                ml_tp50 = ml_tp80 = ml_tp90 = None
                ml_ret80 = None

                if self.forecaster is not None:
                    try:
                        tp_pred = self.forecaster.predict(ohlcv, lookback=ml_lookback, horizon_bars=bars24, btc_ohlcv=btc_ohlcv)
                        if tp_pred is not None:
                            ml_tp50 = tp_pred.tp_prices.get(0.5)
                            ml_tp80 = tp_pred.tp_prices.get(0.8)
                            ml_tp90 = tp_pred.tp_prices.get(0.9)
                            ml_ret80 = tp_pred.mfe_returns.get(0.8)
                            
                            # Log prediction for performance tracking
                            if self.ml_tracker and ml_tp50 and ml_tp80 and ml_tp90 and ml_ret80 is not None:
                                self.ml_tracker.log_prediction(
                                    symbol=sym,
                                    entry_price=last,
                                    pred_tp50=ml_tp50,
                                    pred_tp80=ml_tp80,
                                    pred_tp90=ml_tp90,
                                    pred_ret80=ml_ret80,
                                    trade_taken=False  # Will be updated if trade is executed
                                )
                    except Exception as e:
                        # ML prediction failed - continue without ML for this symbol
                        pass

                # Sentiment analysis
                sentiment_score = None
                sentiment_label = None
                sentiment_news_count = None
                news_headlines = None
                
                if self.sentiment_analyzer is not None:
                    if self._recent_news_cache:
                        sentiment = self.sentiment_analyzer.analyze_asset_sentiment(
                            base, self._recent_news_cache, lookback_hours=24.0
                        )
                        if sentiment.news_count > 0:
                            sentiment_score = sentiment.score
                            sentiment_label = self.sentiment_analyzer.get_sentiment_label(sentiment.score)
                            sentiment_news_count = sentiment.news_count
                            news_headlines = sentiment.recent_headlines
                        else:
                            # No news for this specific asset, but show that we checked
                            sentiment_label = "No news"
                            sentiment_news_count = 0
                    else:
                        # News cache is empty
                        sentiment_label = "News pending"
                        sentiment_news_count = 0

                bullish_enough = (res.trend == "BULL")
                long_ok = (res.signal == "LONG") if require_long_for_entry else (res.signal != "EXIT")

                ml_ok = True
                if ml_ret80 is not None:
                    ml_ok = (ml_ret80 >= min_tp80_pct)
                
                # Consider sentiment in decision (optional: adjust based on config)
                sentiment_cfg = self.cfg.get("sentiment", {}) or {}
                use_sentiment_filter = bool(sentiment_cfg.get("use_for_recommendations", True))
                min_sentiment_score = float(sentiment_cfg.get("min_score_for_buy", -0.3))
                
                sentiment_ok = True
                if use_sentiment_filter and sentiment_score is not None:
                    sentiment_ok = (sentiment_score >= min_sentiment_score)

                if base in held_set:
                    if allow_compound and bullish_enough and long_ok and ml_ok and sentiment_ok:
                        action = "COMPOUND"
                else:
                    if bullish_enough and long_ok and ml_ok and sentiment_ok:
                        action = "BUY"

                r = Recommendation(
                    symbol=sym, base=base, quote=quote, last=last,
                    chg24=chg, qv=qv, trend=res.trend, signal=res.signal, action=action,
                    ml_tp50=ml_tp50, ml_tp80=ml_tp80, ml_tp90=ml_tp90, ml_ret80=ml_ret80,
                    sentiment_score=sentiment_score, sentiment_label=sentiment_label,
                    sentiment_news_count=sentiment_news_count, news_headlines=news_headlines
                )
                recos.append(r)

            except Exception:
                continue

        def score(x: Recommendation) -> float:
            base_score = 0.0
            if x.ml_ret80 is not None:
                base_score = float(x.ml_ret80) * 100.0
            else:
                base_score = float(x.chg24)
            
            # Add sentiment boost (optional)
            sentiment_cfg = self.cfg.get("sentiment", {}) or {}
            sentiment_weight = float(sentiment_cfg.get("score_weight", 0.1))
            if x.sentiment_score is not None and sentiment_weight > 0:
                # Sentiment ranges -1 to +1, scale it to percentage points
                sentiment_boost = x.sentiment_score * sentiment_weight * 100.0
                base_score += sentiment_boost
            
            return base_score + 1e-12 * float(x.qv)

        recos.sort(key=score, reverse=True)
        
        buy_recos = [r for r in recos if r.action in ("BUY", "COMPOUND")]
        print(f"[SCREENER] Found {len(buy_recos)} BUY/COMPOUND opportunities out of {len(recos)} evaluated")
        
        return recos[: max(1, int(reco_count))]

    # -----------------
    # Execute Top-5 decisions under strict caps
    # -----------------

    def act_on_top_recommendations(
        self,
        recos: List[Recommendation],
        execute: bool,
    ) -> None:
        risk = self.cfg.get("risk", {}) or {}
        pm = self.cfg.get("position_management", {}) or {}
        port = self.cfg.get("portfolio", {}) or {}

        min_equity = float(risk.get("min_equity", 10.0))
        max_open = int(risk.get("max_open_positions", 12))
        max_pos_pct = float(risk.get("max_position_pct", 12.0))
        max_trade_notional = float(risk.get("max_trade_notional", 10.0))
        min_trade_notional = float(pm.get("min_trade_notional", 5.0))

        stable_floor_pct = float(port.get("stable_target_pct", 0.10))
        explore_pct = float(port.get("explore_target_pct", 0.05))
        stable_assets = port.get("stable_assets", ["USD", "USDT", "USDC", "USDG"])

        eq = self.equity()
        cash = self.cash()
        stable_floor = eq * stable_floor_pct

        held = self.held_assets()
        explore_now = 0.0
        core_targets = port.get("core_targets", {}) or {}
        core_set = {canonical_asset(k) for k in core_targets.keys()}
        for asset, qty in held.items():
            if canonical_asset(asset) in core_set:
                continue
            px = self._price_for_asset(asset)
            if px is None:
                continue
            explore_now += float(qty) * float(px)

        explore_cap = eq * explore_pct
        budget_remaining = max(0.0, explore_cap - explore_now)

        open_positions = len(held)

        print(f"[ENGINE] Equity≈{eq:.2f} {self.base_currency} | Cash≈{cash:.2f} | Held positions={open_positions}")
        print(f"[BUDGET] explore_cap={explore_cap:.2f} explore_now={explore_now:.2f} budget_remaining={budget_remaining:.2f} {self.base_currency}")
        print(f"[DEBUG] caps: max_open={max_open}, max_pos_pct={max_pos_pct:.1f}%, max_trade_notional={max_trade_notional:.2f}, min_trade_notional={min_trade_notional:.2f}, execute={execute}")

        if eq < min_equity:
            print(f"[ENGINE][SKIP] equity {eq:.2f} < min_equity {min_equity:.2f}")
            return

        if not recos:
            print("[ENGINE] No recommendations this cycle.")
            return

        print(f"[ENGINE] Top {len(recos)} recommendations (quote={self.base_currency}):")
        for i, r in enumerate(recos, start=1):
            ml_part = ""
            if r.ml_tp50 is not None and r.ml_tp80 is not None and r.ml_tp90 is not None:
                ml_part = f" | ML TP50={r.ml_tp50:.6f} TP80={r.ml_tp80:.6f} TP90={r.ml_tp90:.6f}"
            
            sentiment_part = ""
            if r.sentiment_label:
                if r.sentiment_score is not None:
                    sentiment_part = f" | SENTIMENT: {r.sentiment_label} ({r.sentiment_score:+.2f}, {r.sentiment_news_count} news)"
                else:
                    # Show status even when no sentiment score (e.g., "No news", "News pending")
                    sentiment_part = f" | SENTIMENT: {r.sentiment_label}"
            
            print(f"  {i:2d}. {r.symbol:12s} price={r.last:.6f} 24h={r.chg24:+.2f}% trend={r.trend} sig={r.signal} -> {r.action}{ml_part}{sentiment_part}")
            
            # Display news headlines if available
            if r.news_headlines and len(r.news_headlines) > 0:
                print(f"      📰 Recent News for {r.base}:")
                for idx, headline in enumerate(r.news_headlines[:3], 1):  # Show top 3
                    print(f"         {idx}. {headline}")
                if len(r.news_headlines) > 3:
                    print(f"         ... and {len(r.news_headlines) - 3} more")
                print()  # Empty line for readability

        for r in recos:
            if r.action not in ("BUY", "COMPOUND"):
                continue

            if open_positions >= max_open and (r.base not in held):
                print(f"[DEBUG] Skip {r.symbol}: max_open_positions reached ({open_positions}/{max_open})")
                continue

            pos_cap_notional = eq * (max_pos_pct / 100.0)
            allowed = min(max_trade_notional, pos_cap_notional)

            core_targets = port.get("core_targets", {}) or {}
            core_set = {canonical_asset(k) for k in core_targets.keys()}
            is_core = canonical_asset(r.base) in core_set

            if not is_core:
                allowed = min(allowed, budget_remaining)

            spendable_cash = max(0.0, cash - stable_floor)
            allowed = min(allowed, spendable_cash)

            if allowed < min_trade_notional:
                print(f"[DEBUG] Skip {r.symbol}: allowed_notional {allowed:.4f} < min_trade_notional {min_trade_notional:.4f} (cash={cash:.4f}, stable_floor={stable_floor:.4f})")
                continue

            bought_qty = self.buy(r.symbol, allowed, r.last, execute)
            if bought_qty > 0:
                cash = self.cash()
                eq = self.equity()
                if not is_core:
                    budget_remaining = max(0.0, budget_remaining - allowed)
                if r.base not in held:
                    open_positions += 1
                    held[r.base] = bought_qty
                print(f"[ENGINE] executed {r.action} {r.symbol}: notional≈{allowed:.2f} -> qty≈{bought_qty:.8f}")

    # -----------------
    # Core rebalance (lightweight)
    # -----------------

    def rebalance_core(self, execute: bool) -> None:
        rb = ((self.cfg.get("portfolio", {}) or {}).get("rebalance") or {})
        if not bool(rb.get("enabled", False)):
            return

        band = float(rb.get("band_pct", 0.02))
        max_trade = float(rb.get("max_trade_notional", 10.0))
        max_total = float(rb.get("max_total_notional_per_cycle", 20.0))

        port = self.cfg.get("portfolio", {}) or {}
        core_targets = port.get("core_targets", {}) or {}
        stable_floor_pct = float(port.get("stable_target_pct", 0.10))

        eq = self.equity()
        cash = self.cash()
        stable_floor = eq * stable_floor_pct
        spendable = max(0.0, cash - stable_floor)

        if eq <= 0:
            return

        held = self.held_assets()
        values: Dict[str, float] = {}
        for asset, qty in held.items():
            px = self._price_for_asset(asset)
            if px is None:
                continue
            values[canonical_asset(asset)] = float(qty) * float(px)

        values[self.base_currency] = cash

        total_spent = 0.0

        print(f"[REBAL] equity≈{eq:.2f} {self.base_currency} cash≈{cash:.2f} stable_floor≈{stable_floor:.2f}")

        for asset, tgt in core_targets.items():
            a = canonical_asset(asset)
            target_value = eq * float(tgt)
            cur_value = float(values.get(a, 0.0))
            drift = (cur_value - target_value) / eq

            if abs(drift) <= band:
                continue

            sym = self._find_market_for_asset(a)
            if not sym:
                continue

            if drift < -band:
                need = min(max_trade, (target_value - cur_value), spendable, (max_total - total_spent))
                if need <= 0:
                    continue
                price = self._price_for_asset(a)
                if price is None or price <= 0:
                    continue
                qty = need / price
                qty = self._normalize_qty(sym, qty, price)
                if qty <= 0:
                    print(f"[REBAL][SKIP] BUY {sym}: qty normalized to 0 (min/precision/cost)")
                    continue
                bought = self.buy(sym, need, price, execute)
                if bought > 0:
                    total_spent += need
                    spendable = max(0.0, spendable - need)
            else:
                have_qty = held.get(a, 0.0)
                price = self._price_for_asset(a)
                if price is None or price <= 0:
                    continue
                excess = min(max_trade, (cur_value - target_value), (max_total - total_spent))
                if excess <= 0:
                    continue
                sell_qty = excess / price
                sell_qty = min(float(sell_qty), float(have_qty))
                sell_qty = self._normalize_qty(sym, sell_qty, price)
                if sell_qty <= 0:
                    print(f"[REBAL][SKIP] SELL {sym}: qty normalized to 0")
                    continue
                sold = self.sell(sym, sell_qty, price, execute)
                if sold > 0:
                    total_spent += excess

        if total_spent > 0:
            print(f"[REBAL] cycle notional moved≈{total_spent:.2f} {self.base_currency}")
