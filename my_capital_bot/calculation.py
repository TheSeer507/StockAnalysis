import logging
import requests
import pandas as pd
from typing import Dict, List
from config import API_KEY, API_URL, LOGIN, PASSWORD

########################################
# SETUP LOGGING
########################################
logging.basicConfig(
    filename="long_term_calc.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


########################################
# AUTH
########################################
def authenticate():
    session_url = f"{API_URL}/api/v1/session"
    headers = {"X-CAP-API-KEY": API_KEY, "Content-Type": "application/json"}
    payload = {"identifier": LOGIN, "password": PASSWORD}

    try:
        resp = requests.post(session_url, headers=headers, json=payload)
        resp.raise_for_status()
        cst = resp.headers.get("CST")
        x_sec_token = resp.headers.get("X-SECURITY-TOKEN")
        if not cst or not x_sec_token:
            raise ValueError("Missing authentication tokens in headers")
        return cst, x_sec_token
    except Exception as e:
        logging.error(f"Authentication failed: {str(e)}")
        raise


########################################
# DATA FETCHING
########################################
def fetch_long_term_data(cst, x_sec_token, epic, resolution="MINUTE_60", max_candles=200):
    url = f"{API_URL}/api/v1/prices/{epic}"
    headers = {
        "X-CAP-API-KEY": API_KEY,
        "CST": cst,
        "X-SECURITY-TOKEN": x_sec_token
    }
    params = {"resolution": resolution, "max": max_candles}

    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        prices = resp.json().get("prices", [])

        if not prices:
            return pd.DataFrame()

        df = pd.DataFrame([{
            "timestamp": pd.to_datetime(p["snapshotTimeUTC"]),
            "open": p["openPrice"]["bid"],
            "high": p["highPrice"]["bid"],
            "low": p["lowPrice"]["bid"],
            "close": p["closePrice"]["bid"]
        } for p in prices]).set_index("timestamp")

        return df

    except Exception as e:
        logging.warning(f"Failed to fetch data for {epic}: {str(e)}")
        return pd.DataFrame()


########################################
# ENHANCED METRICS CALCULATION
########################################
def compute_long_term_metrics(df, ma_short=20, ma_long=50, rsi_window=14):
    if df.empty or len(df) < ma_long:
        return {}

    try:
        # Moving Averages
        df["ma_short"] = df["close"].rolling(ma_short).mean()
        df["ma_long"] = df["close"].rolling(ma_long).mean()

        # RSI Calculation
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1 / rsi_window, min_periods=rsi_window).mean()
        avg_loss = loss.ewm(alpha=1 / rsi_window, min_periods=rsi_window).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        last = df.iloc[-1]
        metrics = {
            "epic": "",  # Will be filled later
            "price": last["close"],
            "ma_short": last["ma_short"],
            "ma_long": last["ma_long"],
            "trend": "Uptrend" if last["ma_short"] > last["ma_long"] else "Downtrend",
            "rsi": last["rsi"],
            "support": df["low"].tail(14).min(),
            "resistance": df["high"].tail(14).max(),
            "score": 0
        }

        # Scoring System
        score = 0
        score += 40 if metrics["trend"] == "Uptrend" else 0
        if metrics["price"] > metrics["ma_short"] and metrics["price"] > metrics["ma_long"]:
            score += 30
        elif metrics["price"] > metrics["ma_long"]:
            score += 15
        if metrics["rsi"] < 40:
            score += 20
        elif metrics["rsi"] < 60:
            score += 10
        score += min(10, ((metrics["price"] - metrics["support"]) / metrics["price"]) * 100)
        metrics["score"] = max(0, min(100, score))

        return metrics

    except Exception as e:
        logging.error(f"Metrics calculation failed: {str(e)}")
        return {}


########################################
# ANALYSIS & RANKING
########################################
def analyze_long_term_stocks(watchlist, resolution="DAY", ma_short=20, ma_long=50, rsi_window=14):
    try:
        cst, x_sec_token = authenticate()
        results = []

        for epic in watchlist:
            df = fetch_long_term_data(
                cst, x_sec_token,
                epic,
                resolution=resolution,
                max_candles=ma_long * 2  # Ensure enough historical data
            )

            if not df.empty:
                metrics = compute_long_term_metrics(df, ma_short, ma_long, rsi_window)
                if metrics:
                    metrics["epic"] = epic
                    results.append(metrics)

        # Ranking System
        ranked = sorted(results, key=lambda x: x["score"], reverse=True)
        for idx, item in enumerate(ranked):
            item["rank"] = idx + 1
            if item["score"] >= 80:
                item["priority"] = "Strong Buy"
            elif item["score"] >= 60:
                item["priority"] = "Buy"
            elif item["score"] >= 40:
                item["priority"] = "Hold"
            else:
                item["priority"] = "Sell"

        # Print formatted results
        print("\n" + "=" * 80)
        print(f"{'Stock Analysis Results':^80}")
        print("=" * 80)
        print(f"{'Rank':<5}{'Symbol':<8}{'Price':<10}{'Trend':<12}{'RSI':<8}{'Score':<8}{'Priority':<12}")
        print("-" * 80)
        for stock in ranked:
            print(f"{stock['rank']:<5}{stock['epic']:<8}${stock['price']:<9.2f}"
                  f"{stock['trend'][:10]:<12}{stock['rsi']:<8.1f}"
                  f"{stock['score']:<8.1f}{stock['priority']:<12}")

        return ranked

    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        return []