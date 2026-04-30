import os
from typing import Dict, Any, Optional, Tuple

from dotenv import load_dotenv
import ccxt


load_dotenv()


class ExchangeClient:
    def __init__(self, exchange_name: str, mode: str = "paper", verbose: bool = True):
        self.mode = mode
        self.verbose = verbose

        if self.verbose:
            print(f"[INIT] Creating exchange client for {exchange_name} in {mode.upper()} mode...")

        api_key = os.getenv("KRAKEN_API_KEY")
        secret = os.getenv("KRAKEN_SECRET")

        self.exchange = getattr(ccxt, exchange_name)({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
        })

        self.markets: Dict[str, Any] = {}
        try:
            self.markets = self.exchange.load_markets()
            if self.verbose:
                print(f"[OK] Connected to {self.exchange.id}. Loaded {len(self.markets)} markets.")
        except Exception as e:
            print(f"[WARN] Could not load markets on init: {e}")

    # ------------ Market data methods ------------

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        if self.verbose:
            print(f"[DATA] Fetching OHLCV for {symbol} ({timeframe}, limit={limit})...")
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        return self.exchange.fetch_ticker(symbol)

    def fetch_all_tickers(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch tickers for all markets from the exchange.
        """
        return self.exchange.fetch_tickers()

    # ------------ Account methods ------------

    def get_balance(self, currency: str) -> float:
        balance = self.exchange.fetch_balance()
        return balance.get(currency, {}).get("free", 0.0)

    def get_full_balance(self) -> Dict[str, Any]:
        return self.exchange.fetch_balance()

    # ------------ Trading methods ------------

    def create_market_buy_order(self, symbol: str, amount: float):
        print(f"[TRADE] Requested BUY {symbol} amount={amount} (mode={self.mode.upper()})")
        if self.mode == "paper":
            print("[PAPER] Not sending real BUY order.")
            return None
        return self.exchange.create_market_buy_order(symbol, amount)

    def create_market_sell_order(self, symbol: str, amount: float):
        print(f"[TRADE] Requested SELL {symbol} amount={amount} (mode={self.mode.upper()})")
        if self.mode == "paper":
            print("[PAPER] Not sending real SELL order.")
            return None
        return self.exchange.create_market_sell_order(symbol, amount)

    # ------------ Helper methods ------------

    def resolve_symbol(
        self,
        base_asset: str,
        quote_candidates: Tuple[str, ...] = ("USDT", "USD", "EUR"),
    ) -> Optional[str]:
        """
        Try to find a tradable market symbol for a given asset.
        Example: base_asset='BTC' -> 'BTC/USDT' or 'BTC/USD'.
        Returns None if no suitable market is found.
        """
        # Ensure markets are loaded
        if not self.markets:
            try:
                self.markets = self.exchange.load_markets()
            except Exception:
                return None

        # 1) Direct matches: BASE/QUOTE (e.g. BTC/USDT, BTC/USD)
        for quote in quote_candidates:
            candidate = f"{base_asset}/{quote}"
            if candidate in self.markets:
                return candidate

        # 2) Fallback: find first market where base matches
        for sym, m in self.markets.items():
            if m.get("base") == base_asset:
                return sym

        # 3) Nothing found
        return None
