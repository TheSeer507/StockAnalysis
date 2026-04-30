import logging
import schedule
import time
import pandas as pd
import requests
import json
import pytz
import datetime
import numpy as np
from config import LOGIN, PASSWORD, API_KEY, API_URL, IS_DEMO
# ============== NEW IMPORTS ==================
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import hashlib  # DEBUG

# ============== CONFIGURATION ==================
BLUE_CHIPS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "NFLX", "AMD", "META", "TSM", "V", "MA", "ORCL", "UNH",
              "XOM", "MRK", "AXP", "ABT", "TXN", "ARM", "C", "SCHW", "BA", "VRTX", "AVGO", "JPM", "LLY"
            , "PG", "JNJ", "CRM"]
INDICES = ["US500", "US100", "US30", "VOO"]
SYMBOL_LIST = BLUE_CHIPS + INDICES
RISK_PER_TRADE = 0.02  # 2% risk per trade
TIMEFRAME = "HOUR_4"  # 4-hour candles
TOP_PICKS = 3  # Number of top picks to trade


# ============== TECHNICAL INDICATORS ==================
def calculate_pivot_points(df):
    """Calculate daily pivot points for support/resistance"""
    if len(df) < 2:
        return 0, 0, 0
    prev_day = df[-2:]
    pivot = (prev_day['high'].mean() + prev_day['low'].mean() + prev_day['close'].mean()) / 3
    r1 = 2 * pivot - prev_day['low'].mean()
    s1 = 2 * pivot - prev_day['high'].mean()
    return pivot, r1, s1


def detect_support_resistance(df, window=20):
    """Identify key support/resistance levels"""
    df['support'] = df['low'].rolling(window).min()
    df['resistance'] = df['high'].rolling(window).max()
    return df


# ============== TRADE MANAGER ===================
class TradeManager:
    def __init__(self):
        self.max_daily_trades = 3
        self.trade_count = 0
        self.profit_target = 5.00
        self.daily_loss_limit = -3.00
        self.min_trade_size = 0.01
        self.active_trades = set()

    def should_enter_trade(self, score, epic):
        return (self.trade_count < self.max_daily_trades and
                score >= 70 and
                self.profit_target > self.daily_loss_limit and
                epic not in self.active_trades)

    def update_performance(self, profit, epic):
        self.trade_count += 1
        self.profit_target -= profit
        self.active_trades.add(epic)
        if self.profit_target <= 0:
            print_log("✅ Daily target achieved! Stopping bot.")
            exit()
        elif self.profit_target <= self.daily_loss_limit:
            print_log("❌ Daily loss limit hit! Stopping bot.")
            exit()


trade_mgr = TradeManager()


# ============== LOGGING ===================
def print_log(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    logging.info(formatted)


# ============== AUTH & ACCOUNT MGMT ==============
def authenticate():
    print_log("🛂 Authenticating with API...")
    try:
        resp = requests.post(
            f"{API_URL}/api/v1/session",
            headers={"X-CAP-API-KEY": API_KEY, "Content-Type": "application/json"},
            json={"identifier": LOGIN, "password": PASSWORD}
        )
        resp.raise_for_status()
        return resp.headers.get("CST"), resp.headers.get("X-SECURITY-TOKEN")
    except Exception as e:
        print_log(f"❌ Authentication failed: {str(e)}")
        raise


def get_current_equity(cst, x_sec_token):
    try:
        resp = requests.get(
            f"{API_URL}/api/v1/accounts",
            headers={"X-CAP-API-KEY": API_KEY, "CST": cst, "X-SECURITY-TOKEN": x_sec_token}
        )
        if resp.status_code == 200:
            balance = resp.json()["accounts"][0]["balance"]["balance"]
            return float(balance)
        return 10000.00
    except Exception as e:
        print_log(f"❌ Equity check failed: {str(e)}")
        return 10000.00


# ============== DATA PROCESSING ==================
def historical_data(cst, x_sec_token, epic):
    try:
        resp = requests.get(
            f"{API_URL}/api/v1/prices/{epic}",
            headers={"X-CAP-API-KEY": API_KEY, "CST": cst, "X-SECURITY-TOKEN": x_sec_token},
            params={"resolution": TIMEFRAME, "max": 500}
        )
        resp.raise_for_status()
        prices = resp.json().get("prices", [])

        if not prices:
            return pd.DataFrame()

        df = pd.DataFrame([{
            "timestamp": p["snapshotTimeUTC"],
            "open": p["openPrice"]["bid"],
            "high": p["highPrice"]["bid"],
            "low": p["lowPrice"]["bid"],
            "close": p["closePrice"]["bid"],
            "volume": p.get("lastTradedVolume", 0)
        } for p in prices])

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        if len(df) >= 200:
            df['sma50'] = df['close'].rolling(50, min_periods=50).mean()
            df['sma200'] = df['close'].rolling(200, min_periods=200).mean()
        else:
            return pd.DataFrame()

        df['tr'] = df['high'] - df['low']
        df['atr'] = df['tr'].rolling(14, min_periods=14).mean()
        df = detect_support_resistance(df)

        if len(df) >= 2:
            df['pivot'], df['r1'], df['s1'] = calculate_pivot_points(df)

        return df.iloc[-200:]
    except Exception as e:
        print_log(f"❌ Data error for {epic}: {str(e)}")
        return pd.DataFrame()


# ============== FIBONACCI IMPLEMENTATION ==================
def calculate_fib_levels(df, lookback=50):
    """Identify swing highs/lows and calculate fib levels"""
    df = df.copy()
    df['swing_high'] = df['high'].rolling(lookback, center=True).max()
    df['swing_low'] = df['low'].rolling(lookback, center=True).min()
    recent_high = df[df['high'] == df['swing_high']]['high'].dropna().iloc[-1] if not df.empty else 0
    recent_low = df[df['low'] == df['swing_low']]['low'].dropna().iloc[-1] if not df.empty else 0

    if recent_high <= recent_low:
        return None

    diff = recent_high - recent_low
    return {
        '23.6%': recent_high - diff * 0.236,
        '38.2%': recent_high - diff * 0.382,
        '50%': recent_high - diff * 0.5,
        '61.8%': recent_high - diff * 0.618,
        '78.6%': recent_high - diff * 0.786
    }


# ============== CORRECTED LSTM IMPLEMENTATION ==================
class PriceDataset(Dataset):
    def __init__(self, data, seq_length=20):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = np.array(self.data[idx:idx + self.seq_length])
        target = np.array([self.data[idx + self.seq_length, 3]])
        return torch.FloatTensor(seq), torch.FloatTensor(target)


class PricePredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_price_model(df, epochs=200):
    """Train LSTM model on historical price data"""
    # DEBUG START
    data_hash = hashlib.md5(df[['open', 'high', 'low', 'close', 'volume']].values.tobytes()).hexdigest()[:6]
    print_log(f"📦 Training Data Hash: {data_hash} (Ensures data consistency)")
    # DEBUG END

    features = df[['open', 'high', 'low', 'close', 'volume']].values
    close_prices = df['close'].values.reshape(-1, 1)

    feature_scaler = StandardScaler()
    price_scaler = StandardScaler()

    scaled_features = feature_scaler.fit_transform(features)
    scaled_prices = price_scaler.fit_transform(close_prices)
    combined_data = np.concatenate([scaled_features, scaled_prices], axis=1)

    # DEBUG START
    print_log(f"📊 Training Data Shape: {combined_data.shape}")
    print_log(f"📈 Sample Training Data (First 3 rows):\n{combined_data[:3]}")
    print_log(f"🔢 Feature Scaler Mean: {feature_scaler.mean_} | Scale: {feature_scaler.scale_}")
    print_log(f"💰 Price Scaler Mean: {price_scaler.mean_[0]:.2f} | Scale: {price_scaler.scale_[0]:.2f}")
    # DEBUG END

    dataset = PriceDataset(combined_data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PricePredictor(input_size=6)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # DEBUG TRAINING LOOP
    for epoch in range(epochs):
        total_loss = 0
        for seq, labels in loader:
            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print_log(f"🏋️ Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

        # DEBUG SAMPLE PREDICTIONS
        if epoch == 0 or (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_seq = seq[0].unsqueeze(0)
                sample_pred = model(sample_seq)
                sample_pred_price = price_scaler.inverse_transform(sample_pred.numpy())[0][0]
                sample_true_price = price_scaler.inverse_transform(labels[0].unsqueeze(0).numpy())[0][0]
                print_log(f"🔮 Sample Prediction: {sample_pred_price:.2f} vs Actual: {sample_true_price:.2f}")

    print_log("✅ Model training completed")
    return model, feature_scaler, price_scaler


def predict_price(model, feature_scaler, price_scaler, df):
    """Predict next period price using trained model"""
    seq_length = 20
    features = df[['open', 'high', 'low', 'close', 'volume']].values[-seq_length:]
    close_prices = df['close'].values[-seq_length:].reshape(-1, 1)

    if len(features) < seq_length:
        return None

    scaled_features = feature_scaler.transform(features)
    scaled_prices = price_scaler.transform(close_prices)
    combined_data = np.concatenate([scaled_features, scaled_prices], axis=1)

    seq = torch.FloatTensor(combined_data).unsqueeze(0)

    # DEBUG START
    print_log(f"🧠 Prediction Input Shape: {seq.shape}")
    print_log(f"📌 Last 5 Actual Closes: {df['close'].values[-5:]}")
    # DEBUG END

    with torch.no_grad():
        scaled_pred = model(seq).item()

    prediction = price_scaler.inverse_transform([[scaled_pred]])[0][0]

    # DEBUG
    print_log(f"🔮 Price Prediction: {prediction:.2f} (Current: {df['close'].iloc[-1]:.2f})")

    return prediction


# ============== SCORING ==================
def calculate_score(df):
    if df.empty or len(df) < 50:
        return 0
    try:
        score = 0
        current_price = df['close'].iloc[-1]

        # Trend Analysis
        if 'sma50' in df and 'sma200' in df:
            sma50 = df['sma50'].iloc[-1]
            sma200 = df['sma200'].iloc[-1]
            if sma50 > sma200:
                score += 40
            if current_price > sma200:
                score += 20

        # Support/Resistance
        if 'support' in df and 'resistance' in df:
            support_level = df['support'].iloc[-1]
            resistance_level = df['resistance'].iloc[-1]
            if current_price <= support_level * 1.02:
                score += 30
            elif current_price >= resistance_level * 0.98:
                score -= 10

        # Volatility
        if 'atr' in df and current_price > 0:
            atr_ratio = df['atr'].iloc[-1] / current_price
            if 0.005 < atr_ratio < 0.05:
                score += 20

        # Fibonacci Retracement Scoring
        fib_levels = calculate_fib_levels(df)
        if fib_levels:
            for level in ['38.2%', '61.8%']:
                if abs(current_price - fib_levels[level]) < current_price * 0.005:
                    score += 15
                    print_log(f"💰 Fib level hit: {level} at {fib_levels[level]:.2f}")

        # AI Prediction Scoring
        if 'predictor' in globals() and 'feature_scaler' in globals() and 'price_scaler' in globals():  # DEBUG
            predicted_price = predict_price(predictor, feature_scaler, price_scaler, df)  # DEBUG
            if predicted_price:
                price_diff = (predicted_price - current_price) / current_price
                # DEBUG START
                print_log(
                    f"🤖 Prediction Impact: {price_diff:.2%} | Score Adjustment: {20 if price_diff > 0.02 else -10 if price_diff < -0.02 else 0}")
                # DEBUG END
                if predicted_price > current_price * 1.02:
                    score += 20
                elif predicted_price < current_price * 0.98:
                    score -= 10

        # Price Position
        if 'pivot' in df:
            if current_price > df['pivot'].iloc[-1]:
                score += 10

        return max(min(score, 100), 0)

    except Exception as e:
        print_log(f"⚠️ Scoring error: {str(e)}")
        return 0


# ============== TRADE EXECUTION ==================
def get_instrument_details(cst, x_sec_token, epic):
    try:
        resp = requests.get(
            f"{API_URL}/api/v1/markets/{epic}",
            headers={"X-CAP-API-KEY": API_KEY, "CST": cst, "X-SECURITY-TOKEN": x_sec_token}
        )
        data = resp.json()
        instrument = data.get("instrument", {})
        return {
            "min_step": float(instrument.get("stepDistance", 0.01)),
            "min_stop": float(instrument.get("minStopDistance", 0.05)),
            "lot_size": float(instrument.get("lotSize", 1.0)),
            "precision": instrument.get("precision", 2)
        }
    except Exception as e:
        print_log(f"⚠️ Instrument error: {str(e)}")
        return None


def calculate_position_size(price, stop_loss, risk_amount, instrument):
    try:
        price_diff = abs(price - stop_loss)
        if price_diff == 0:
            return 0

        raw_size = risk_amount / price_diff
        base_size = max(raw_size, instrument["lot_size"])
        if IS_DEMO:
            base_size = min(base_size, 1.0)
        else:
            base_size = min(base_size, 0.5)

        step_size = instrument["min_step"]
        normalized = round(base_size / step_size) * step_size
        final_size = round(normalized, int(abs(np.log10(step_size))))

        print_log(f"🧮 Size Calculation | Raw: {raw_size:.2f} → Base: {base_size:.2f} → Final: {final_size:.2f}")
        return final_size
    except Exception as e:
        print_log(f"⚠️ Size error: {str(e)}")
        return 0


def execute_trade(cst, x_sec_token, epic, df, score):
    try:
        instrument = get_instrument_details(cst, x_sec_token, epic)
        if not instrument or df.empty:
            return

        cp = round(df['close'].iloc[-1], instrument["precision"])
        atr = df['atr'].iloc[-1]
        direction = "BUY"

        print_log(f"🔍 Analyzing {epic} | Price: {cp} | Direction: {direction}")

        if df['sma50'].iloc[-1] <= df['sma200'].iloc[-1]:
            print_log(f"❌ Bearish trend - skipping {epic}")
            return

        equity = get_current_equity(cst, x_sec_token)
        risk_amount = equity * RISK_PER_TRADE
        stop_loss = round(cp - (atr * 1.5), instrument["precision"])
        position_size = calculate_position_size(cp, stop_loss, risk_amount, instrument)

        position_value = position_size * cp
        if position_value > equity * 0.1:
            print_log(f"❌ Position exceeds 10% equity: ${position_value:.2f}/${equity:.2f}")
            return

        payload = {
            "epic": epic,
            "direction": direction,
            "size": position_size,
            "orderType": "MARKET",
            "stopLevel": stop_loss,
            "profitLevel": round(cp + (atr * 3), instrument["precision"]),
            "currencyCode": "USD",
            "timeInForce": "GOOD_TILL_CANCELLED"
        }

        resp = requests.post(
            f"{API_URL}/api/v1/positions",
            headers={"X-CAP-API-KEY": API_KEY, "CST": cst, "X-SECURITY-TOKEN": x_sec_token},
            json=payload
        )

        if resp.status_code in [200, 201]:
            print_log(f"✅ Order success: {resp.json().get('dealReference')}")
            trade_mgr.update_performance(0, epic)
        else:
            print_log(f"❌ Order failed: {resp.text}")
    except Exception as e:
        print_log(f"🔥 Trade error: {str(e)}")


# ============== MARKET CONDITIONS ==================
def is_market_hours(dt):
    ny_tz = pytz.timezone('America/New_York')
    ny_time = dt.astimezone(ny_tz)
    return (9 <= ny_time.hour < 16) and ny_time.weekday() < 5


# ============== MODIFIED TRADING SESSION ==================
def trading_session():
    global predictor, feature_scaler, price_scaler  # DEBUG

    print_log("⚡ Starting trading session")
    try:
        if not is_market_hours(datetime.datetime.now(pytz.utc)):
            print_log("⏸️ Outside market hours")
            return

        cst, x_sec_token = authenticate()

        # DEBUG MODEL STATUS
        if 'predictor' not in globals():
            print_log("🆕 No existing model found - will train new model")
        else:
            print_log("♻️ Using existing trained model")

        # DEBUG RETRAINING LOGIC
        if datetime.datetime.now().weekday() == 0:
            print_log("📅 Monday detected - forcing model retraining")

        scored_assets = []
        for epic in SYMBOL_LIST:
            df = historical_data(cst, x_sec_token, epic)
            if df.empty:
                continue

            # Retrain model weekly or if missing
            if 'predictor' not in globals() or datetime.datetime.now().weekday() == 0:
                print_log("🔄 Retraining prediction model...")
                predictor, feature_scaler, price_scaler = train_price_model(df)

            score = calculate_score(df)
            print_log(f"📊 {epic} => Score: {score:.1f}")

            if score >= 70:
                scored_assets.append((epic, score, df))

        scored_assets.sort(key=lambda x: x[1], reverse=True)
        top_assets = scored_assets[:TOP_PICKS]

        for epic, score, df in top_assets:
            if trade_mgr.should_enter_trade(score, epic):
                print_log(f"🚀 Executing trade => {epic} (Rank Score: {score:.1f})")
                execute_trade(cst, x_sec_token, epic, df, score)
                if trade_mgr.trade_count >= trade_mgr.max_daily_trades:
                    break

    except Exception as e:
        print_log(f"🔥 Session error: {str(e)}")
    finally:
        print_log("🏁 Session complete")


# ============== MAIN LOOP ==================
if __name__ == "__main__":
    logging.basicConfig(
        filename="longterm_trader.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    print_log("🤖 Blue-Chip Long-Term Trader Activated")
    schedule.every().hour.do(trading_session)
    trading_session()

    while True:
        schedule.run_pending()
        time.sleep(60)