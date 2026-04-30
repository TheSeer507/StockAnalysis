# bot/account_info.py
from .exchange import ExchangeClient

def print_account_overview(exchange: ExchangeClient, min_balance: float = 0.0001):
    """
    Prints non-zero balances (total, free, used) for each asset.
    """
    print("\n========== ACCOUNT OVERVIEW ==========")
    balance = exchange.get_full_balance()
    total = balance.get("total", {})
    free = balance.get("free", {})
    used = balance.get("used", {})

    for asset, amount in total.items():
        if not amount or amount < min_balance:
            continue
        f = free.get(asset, 0.0)
        u = used.get(asset, 0.0)
        print(f"{asset:>6}  total={amount:.8f}  free={f:.8f}  used={u:.8f}")

    print("======================================\n")
# bot/account_info.py (add below print_account_overview)
def estimate_total_equity_usdt(exchange: ExchangeClient, quote: str = "USDT") -> float:
    """
    Very rough estimate of total equity in USDT.
    - Uses asset/USDT or asset/USD pairs when possible.
    """
    balance = exchange.get_full_balance()
    total = balance.get("total", {})
    equity = 0.0

    for asset, amount in total.items():
        if not amount or amount <= 0:
            continue

        if asset == quote:
            equity += amount
            continue

        symbol_candidates = [f"{asset}/{quote}", f"{asset}/USD"]
        price = None

        for sym in symbol_candidates:
            try:
                ticker = exchange.fetch_ticker(sym)
                price = ticker.get("last")
                if price:
                    break
            except Exception:
                continue

        if price:
            equity += amount * price

    return equity
