# What it is:
- 20MA = Average price over 20 days
- 50MA = Average price over 50 days

# How to use:
- Price > Both MAs = Strong uptrend
- Price < Both MAs = Strong downtrend
- 20MA > 50MA = Positive trend

# Scale: 0-100
- <30 = Oversold (Potential buy)
- >70 = Overbought (Potential sell)
- 40-60 = Neutral

# Example:
if rsi < 30 and price > 50MA:
    print("Good buy opportunity!")

# Support = Price floor where buyers enter
# Resistance = Price ceiling where sellers appear

# Trading rule:
if price > resistance:
    print("Breakout! Potential uptrend continuation")
elif price < support:
    print("Breakdown! Potential downtrend")

# Always set stop-loss (max loss per trade)
def calculate_stop_loss(price, risk_percent=2):
    return price * (1 - risk_percent/100)

# Example:
buy_price = 100
stop_loss = 98  # 2% risk

def calculate_position_size(account_size, risk_per_trade=1):
    """Never risk more than 1% per trade"""
    return (account_size * risk_per_trade/100) / (entry_price - stop_loss)


def update_trailing_stop(current_price, highest_price, trail_percent=3):
    return highest_price * (1 - trail_percent/100)

# Never allocate more than 5% to one trade
max_position = total_capital * 0.05

# Add these to your config.py
ALERT_EMAIL = "your@email.com"
EMAIL_PASSWORD = "your-email-password"
SMTP_SERVER = "smtp.gmail.com"  # For Gmail
SMTP_PORT = 587