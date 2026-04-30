########################################
# HARDCODED WATCHLIST (For Prod)
########################################
def get_trending_instruments(cst, x_sec_token):
    demo_trending = [
        "INTC", "NVDA", "AMD", "MRNA", "WMT", "XOM", "NVO", "CVX",
        "AAPL", "PYPL", "ABT", "JNJ", "ORCL", "QCOM", "SWKS", "META",
        "AMZN", "PANW", "UBER", "ZS", "MSFT", "AVGO", "GOOGL", "GE", "US100",
        'US500', "RBC", "CQTUSD", "SRMUSD"
    ]
    prod_trending = [
        "INTC","NVDA","AMD","MRNA","WMT","XOM","NVO","CVX",
        "AAPL","PYPL","ABT","JNJ","ORCL","QCOM","SWKS","META",
        "AMZN","PANW","UBER","ZS","MSFT","AVGO","GOOGL","GE", "US100",
        'US500', "RBC","CQTUSD","SRMUSD"
    ]

    if IS_DEMO:
        logging.info("IS_DEMO=True => ignoring this function if we do dynamic approach.")
        print("[DEBUG] get_trending_instruments => not used in demo.")
        return demo_trending
    else:
        logging.info("Using PROD trending instruments.")
        print("[DEBUG] Using PROD trending instruments.")
        return prod_trending