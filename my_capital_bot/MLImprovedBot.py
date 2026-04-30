import logging
import schedule
import time
import pandas as pd
import requests
import json
import pytz
import datetime
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ================== CONFIG (EDIT AS NEEDED) ==================
from config import (
    LOGIN,
    PASSWORD,
    API_KEY,
    API_URL,
    DEMO_API_URL,
    IS_DEMO
)

# 1) Retain your existing BLUE_CHIPS & INDICES
BLUE_CHIPS = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","NFLX","AMD",
    "META","TSM","V","MA","ORCL","UNH","XOM","MRK","AXP","ABT",
    "TXN","C","SCHW","BA","VRTX","AVGO","JPM","LLY","PG",
    "JNJ","CRM","WMT","COST","HD","NVO","BAC","TMUS","KO","CVX",
    "WFC","PLTR","CSCO","ACN","IMB","PM"
]
INDICES = ["US500","US100","US30","VOO"]

# We start with these as BASE_SYMBOLS
BASE_SYMBOLS = BLUE_CHIPS + INDICES

RISK_PER_TRADE = 0.02
TOP_PICKS = 10
MODEL_PATH = "lstm_model_cls.pth"
SCALER_PATH = "lstm_scaler_cls.npy"
SEQ_LENGTH = 20
UP_THRESHOLD = 0.01  # 1% threshold for labeling "up"

# We'll fill FINAL_SYMBOL_LIST after we authenticate and fetch "most traded"
FINAL_SYMBOL_LIST = []

# --------------------------------------------------------------------
#   Logging / Utility
# --------------------------------------------------------------------
def print_log(message):
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    txt = f"[{stamp}] {message}"
    print(txt)
    logging.info(txt)

def get_active_api_url():
    """Return the DEMO URL if IS_DEMO = True, else the real/prod API_URL."""
    return DEMO_API_URL if IS_DEMO else API_URL

def is_market_hours(dt):
    """
    Returns True if we are within US/Eastern hours: 9:30–16:00, Mon–Fri.
    """
    ny_tz = pytz.timezone("America/New_York")
    dt_local = dt.astimezone(ny_tz)
    if dt_local.weekday() >= 5:  # Weekend
        return False
    hour = dt_local.hour
    minute = dt_local.minute
    open_time = 9 * 60 + 30  # 9:30
    close_time = 16 * 60      # 16:00
    now_in_minutes = hour * 60 + minute
    return (open_time <= now_in_minutes < close_time)


# ================== BROKER AUTH + EQUITY ==================
def authenticate():
    current_api_url = get_active_api_url()
    print_log(f"🛂 Authenticating with API => {current_api_url}")
    try:
        resp = requests.post(
            f"{current_api_url}/api/v1/session",
            headers={"X-CAP-API-KEY": API_KEY, "Content-Type": "application/json"},
            json={"identifier": LOGIN, "password": PASSWORD}
        )
        resp.raise_for_status()
        print_log(f"🔑 Auth success with API: {current_api_url}")
        return resp.headers.get("CST"), resp.headers.get("X-SECURITY-TOKEN")
    except Exception as e:
        print_log(f"❌ Authentication failed: {str(e)}")
        raise

def get_current_equity(cst, x_sec_token):
    current_api_url = get_active_api_url()
    try:
        resp = requests.get(
            f"{current_api_url}/api/v1/accounts",
            headers={
                "X-CAP-API-KEY": API_KEY,
                "CST": cst,
                "X-SECURITY-TOKEN": x_sec_token
            }
        )
        resp.raise_for_status()
        data = resp.json()
        balance = data["accounts"][0]["balance"]["balance"]
        eq = float(balance)
        print_log(f"💰 Account Equity: {eq:.2f}")
        return eq
    except Exception as e:
        print_log(f"❌ Equity check failed: {str(e)}")
        return 10000.0


# ================== FETCH MOST TRADED ==================
def get_most_traded_list(cst, x_sec_token, limit=50):
    """
    Calls the "Most Traded" navigation endpoint to get up to 'limit' instruments
    and returns a list of epics (strings) we can trade.

    GET /api/v1/marketnavigation/hierarchy_v1.commons.most_traded?limit=500
    """
    current_api_url = get_active_api_url()
    endpoint = f"{current_api_url}/api/v1/marketnavigation/hierarchy_v1.commons.most_traded?limit=500"
    try:
        headers = {
            "X-CAP-API-KEY": API_KEY,
            "CST": cst,
            "X-SECURITY-TOKEN": x_sec_token
        }
        resp = requests.get(endpoint, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        # Suppose data["markets"] is an array of markets, each with "epic"
        markets = data.get("markets", [])
        epics = []
        for m in markets:
            epic = m.get("epic")
            if epic:
                epics.append(epic)
                if len(epics) >= limit:
                    break

        print_log(f"🔎 Fetched {len(epics)} 'most traded' epics from the API.")
        return epics
    except Exception as e:
        print_log(f"❌ get_most_traded_list error => {e}")
        return []


# ================== DATA FETCH ==================
def fetch_history_range(cst, x_sec_token, epic, resolution="DAY", from_date=None, to_date=None, max_bars=500):
    current_api_url = get_active_api_url()
    try:
        params = {"resolution": resolution, "max": max_bars}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        resp = requests.get(
            f"{current_api_url}/api/v1/prices/{epic}",
            headers={
                "X-CAP-API-KEY": API_KEY,
                "CST": cst,
                "X-SECURITY-TOKEN": x_sec_token
            },
            params=params
        )
        resp.raise_for_status()
        raw = resp.json().get("prices", [])
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "timestamp": p["snapshotTimeUTC"],
                "open": p["openPrice"]["bid"],
                "high": p["highPrice"]["bid"],
                "low": p["lowPrice"]["bid"],
                "close": p["closePrice"]["bid"],
                "volume": p.get("lastTradedVolume", 0)
            }
            for p in raw
        ])
        return df
    except Exception as e:
        print_log(f"❌ fetch_history_range error => {e}")
        return pd.DataFrame()

def get_big_history(cst, x_sec_token, epic, resolution="DAY", years=2):
    end_dt = datetime.datetime.now(datetime.timezone.utc)
    start_dt = end_dt - datetime.timedelta(days=365 * years)
    def fmt(dt):
        return dt.strftime("%Y-%m-%dT00:00:00")

    all_data = []
    current_start = start_dt
    while current_start < end_dt:
        chunk_end = current_start + datetime.timedelta(days=600)
        if chunk_end > end_dt:
            chunk_end = end_dt

        df_chunk = fetch_history_range(
            cst, x_sec_token, epic,
            resolution=resolution,
            from_date=fmt(current_start),
            to_date=fmt(chunk_end),
            max_bars=500
        )
        if df_chunk.empty:
            break
        all_data.append(df_chunk)
        current_start = chunk_end
        if chunk_end >= end_dt:
            break

    if not all_data:
        return pd.DataFrame()

    df_full = pd.concat(all_data, ignore_index=True).drop_duplicates(subset=["timestamp"])
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
    df_full.set_index("timestamp", inplace=True)
    df_full.sort_index(inplace=True)
    return df_full


# ================== PREPARE INDICATORS ==================
def prepare_indicators(df):
    if df.empty or len(df) < 200:
        return pd.DataFrame()

    df["sma50"] = df["close"].rolling(50, min_periods=50).mean()
    df["sma200"] = df["close"].rolling(200, min_periods=200).mean()

    df["tr"] = df[["high","close"]].max(axis=1) - df[["low","close"]].min(axis=1)
    df["atr"] = df["tr"].rolling(14, min_periods=14).mean()

    df["support"] = df["low"].rolling(20, min_periods=1).min()
    df["resistance"] = df["high"].rolling(20, min_periods=1).max()

    # pivot/r1/s1
    if len(df) > 2:
        prev_slice = df.iloc[-3:-1]
        pivot = (prev_slice["high"].mean() + prev_slice["low"].mean() + prev_slice["close"].mean()) / 3
        r1 = 2*pivot - prev_slice["low"].mean()
        s1 = 2*pivot - prev_slice["high"].mean()
        df["pivot"] = np.nan
        df["r1"] = np.nan
        df["s1"] = np.nan
        df.iloc[-1, df.columns.get_loc("pivot")] = pivot
        df.iloc[-1, df.columns.get_loc("r1")] = r1
        df.iloc[-1, df.columns.get_loc("s1")] = s1

    return df.iloc[-200:]


# ================== CLASSIFICATION LOGIC ==================
def create_class_labels(df, future_horizon=5, up_threshold=UP_THRESHOLD):
    close_vals = df["close"].values
    labels = []
    for i in range(len(close_vals)):
        if i + future_horizon >= len(close_vals):
            labels.append(np.nan)
        else:
            base_price = close_vals[i] if close_vals[i] != 0 else 1e-8
            ret = (close_vals[i + future_horizon] - close_vals[i]) / base_price
            labels.append(1 if ret >= up_threshold else 0)
    return np.array(labels)

class ClassifierDataset(Dataset):
    def __init__(self, data, labels, seq_length=SEQ_LENGTH):
        self.data = data
        self.labels = labels
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.data[idx: idx + self.seq_length]
        lab = self.labels[idx + self.seq_length]
        return torch.FloatTensor(seq), torch.LongTensor([int(lab)])

class PriceClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sm(out)

def save_model(model, scaler):
    torch.save(model.state_dict(), MODEL_PATH)
    np.save(SCALER_PATH, {"mean": scaler.mean_, "scale": scaler.scale_})

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    try:
        model = PriceClassifier()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        data = np.load(SCALER_PATH, allow_pickle=True).item()
        sc = StandardScaler()
        sc.mean_ = data["mean"]
        sc.scale_ = data["scale"]
        sc.var_ = data["scale"]**2
        return model, sc
    except Exception as e:
        print_log(f"⚠️ load_model => {e}")
        return None, None

def train_classifier_model(df, seq_length=SEQ_LENGTH, future_horizon=5, up_threshold=UP_THRESHOLD, epochs=100):
    print_log("🔧 Starting Model Training Process...")

    data_arr = df[["open","high","low","close","volume"]].values
    sc = StandardScaler()
    data_scaled = sc.fit_transform(data_arr)

    labels = create_class_labels(df, future_horizon, up_threshold)
    valid_len = np.sum(~np.isnan(labels))
    data_scaled = data_scaled[:valid_len]
    labels = labels[:valid_len]
    if len(data_scaled) < seq_length + 1:
        print_log("❌ Not enough data after label creation.")
        return None, None

    idx_split = int(len(data_scaled)*0.8)
    trainX, testX = data_scaled[:idx_split], data_scaled[idx_split:]
    trainY, testY = labels[:idx_split], labels[idx_split:]

    train_ds = ClassifierDataset(trainX, trainY, seq_length=seq_length)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

    test_ds, test_dl = None, None
    if len(testX) > seq_length:
        test_ds = ClassifierDataset(testX, testY, seq_length=seq_length)
        test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = PriceClassifier()
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.NLLLoss()

    best_loss = float('inf')
    for ep in range(epochs):
        model.train()
        total_loss = 0
        for X_seq, Y_seq in train_dl:
            opt.zero_grad()
            out = model(X_seq)
            loss = crit(out, Y_seq.squeeze())
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)

        # Evaluate on test each epoch
        test_acc = 0.0
        if test_dl is not None:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_t, Y_t in test_dl:
                    logp = model(X_t)
                    preds = torch.argmax(logp, dim=1)
                    correct += (preds == Y_t.squeeze()).sum().item()
                    total += len(Y_t)
            test_acc = correct/total if total>0 else 0

        print_log(f"📉 Epoch {ep+1}/{epochs} | Loss={avg_loss:.4f} | TestAcc={test_acc*100:.2f}%")
        if avg_loss < best_loss:
            best_loss = avg_loss

    print_log(f"✅ Training done. Best loss = {best_loss:.4f}")
    return model, sc

def predict_classification(model, scaler, df, seq_length=SEQ_LENGTH):
    """
    Return a probability (float) that the market will go Up.
    We feed the last `seq_length` bars to the LSTM.
    """
    if model is None or scaler is None:
        return None
    data_arr = df[["open","high","low","close","volume"]].values
    if len(data_arr) < seq_length:
        return None
    chunk = data_arr[-seq_length:]
    chunk_scaled = scaler.transform(chunk)
    X_ = torch.FloatTensor(chunk_scaled).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logp = model(X_)
        probs = torch.exp(logp)
        up_prob = probs[0,1].item()
        return up_prob


# ================== SCORE LOGIC ==================
def calculate_fib_levels(df, lookback=50):
    if df.empty or len(df)< lookback:
        return None
    sub = df.iloc[-lookback:]
    rh = sub["high"].max()
    rl = sub["low"].min()
    if rh<= rl:
        return None
    diff = rh - rl
    return {
        "23.6%": rh - diff*0.236,
        "38.2%": rh - diff*0.382,
        "50%":   rh - diff*0.50,
        "61.8%": rh - diff*0.618,
        "78.6%": rh - diff*0.786
    }

def calculate_score(df, classifier, scaler):
    """
    Return an integer 'score' for the last bar, combining
    classical TA + ML classification.
    """
    if df.empty or len(df) < 200:
        return 0

    row = df.iloc[-1]
    cp      = row["close"]
    sma50   = row["sma50"]
    sma200  = row["sma200"]
    sup     = row["support"]
    res     = row["resistance"]
    atrv    = row["atr"]

    final_score = 0
    partial_logs = []

    # Trend-based
    if sma50 > sma200:
        final_score += 40
        partial_logs.append("+40 (sma50> sma200)")
    else:
        final_score -= 20
        partial_logs.append("-20 (sma50< sma200)")

    # Price > sma200 => bullish
    if cp > sma200:
        final_score += 20
        partial_logs.append("+20 (cp> sma200)")
    else:
        final_score -= 10
        partial_logs.append("-10 (cp< sma200)")

    # Support/Resistance
    if cp <= sup*1.02:
        final_score += 30
        partial_logs.append("+30 (near support)")
    if cp >= res*0.98:
        final_score -= 10
        partial_logs.append("-10 (near resistance)")

    # ATR ratio
    if atrv>0:
        atr_ratio = atrv / cp
        if 0.005< atr_ratio<0.05:
            final_score += 20
            partial_logs.append("+20 (moderate volatility)")
        elif atr_ratio>=0.05:
            final_score -= 10
            partial_logs.append("-10 (high volatility)")

    fibs = calculate_fib_levels(df)
    if fibs:
        for lvl in ["23.6%", "38.2%", "50%", "61.8%", "78.6%"]:
            val = fibs[lvl]
            if abs(cp - val) < cp*0.005:
                final_score += 10
                partial_logs.append(f"+10 (close near fib {lvl})")

    # ML Classification => up_prob
    if classifier is not None and scaler is not None:
        up_prob = predict_classification(classifier, scaler, df)
        if up_prob is not None:
            if up_prob>0.7:
                final_score += 20
                partial_logs.append(f"+20 (up_prob={up_prob:.2f} >0.7)")
            elif up_prob<0.3:
                final_score -= 20
                partial_logs.append(f"-20 (up_prob={up_prob:.2f} <0.3)")
            else:
                partial_logs.append(f"(0 for up_prob={up_prob:.2f} => neutral)")

    # Clip and log
    final_score = max(min(final_score, 100), -100)
    partial_logs.append(f"Final Score={final_score}")
    print_log("📝 Score breakdown => " + ",".join(partial_logs))

    return final_score


# ================== TRADE MGMT ==================
class TradeManager:
    def __init__(self):
        self.max_daily_trades = 5
        self.trade_count = 0
        self.profit_target = 5.0
        self.daily_loss_limit = -3.0
        self.active_trades = set()

    def can_take_new_trade(self, epic):
        return (
            self.trade_count< self.max_daily_trades and
            self.profit_target> self.daily_loss_limit and
            epic not in self.active_trades
        )

    def update_performance(self, profit, epic):
        self.trade_count += 1
        self.profit_target -= profit
        self.active_trades.add(epic)
        if self.profit_target <= 0:
            print_log("✅ daily target done => stopping.")
            exit()
        elif self.profit_target <= self.daily_loss_limit:
            print_log("❌ daily loss => stopping.")
            exit()

trade_mgr = TradeManager()

def get_instrument_details(cst, x_sec_token, epic):
    current_api_url = get_active_api_url()
    try:
        resp = requests.get(
            f"{current_api_url}/api/v1/markets/{epic}",
            headers={
                "X-CAP-API-KEY": API_KEY,
                "CST": cst,
                "X-SECURITY-TOKEN": x_sec_token
            }
        )
        resp.raise_for_status()
        data = resp.json()
        inst = data.get("instrument",{})
        return {
            "min_step": float(inst.get("stepDistance",0.01)),
            "min_stop": float(inst.get("minStopDistance",0.05)),
            "lot_size": float(inst.get("lotSize",1.0)),
            "precision": inst.get("precision",2)
        }
    except Exception as e:
        print_log(f"⚠️ Instrument error => {e}")
        return None

def calculate_position_size(entry_price, stop_loss, risk_amount, instrument):
    try:
        diff = abs(entry_price - stop_loss)
        if diff==0:
            return 0
        raw = risk_amount / diff
        base_size = max(raw, instrument["lot_size"])

        if IS_DEMO:
            base_size = min(base_size, 5.0)
        else:
            base_size = min(base_size, 0.1)

        step_size = instrument["min_step"]
        normalized = round(base_size / step_size)* step_size
        final_size = round(normalized, int(abs(np.log10(step_size))))
        print_log(f"🧮 Size Calculation => raw: {raw:.3f} => final {final_size:.3f}")
        return final_size
    except Exception as e:
        print_log(f"Size error => {e}")
        return 0

def execute_trade(cst, x_sec_token, epic, df, score):
    try:
        instr = get_instrument_details(cst, x_sec_token, epic)
        if not instr or df.empty:
            return

        cp  = round(df["close"].iloc[-1], instr["precision"])
        atr = df["atr"].iloc[-1]
        eq  = get_current_equity(cst, x_sec_token)
        risk_amt = eq * RISK_PER_TRADE

        if score >= 70:
            direction = "BUY"
            stop_loss   = round(cp - (atr*1.5), instr["precision"])
            take_profit = round(cp + (atr*3.0), instr["precision"])
        elif score <= 30:
            direction = "SELL"
            stop_loss   = round(cp + (atr*1.5), instr["precision"])
            take_profit = round(cp - (atr*3.0), instr["precision"])
        else:
            print_log(f"⚪ {epic} => Score not strong enough => skip.")
            return

        pos_size = calculate_position_size(cp, stop_loss, risk_amt, instr)
        if pos_size<=0:
            print_log(f"❌ {epic} => Invalid position size => skip.")
            return

        val = pos_size * cp
        if val> eq*0.1:
            print_log(f"❌ {epic} => trade value {val:.2f} >10% eq {eq:.2f}, skip.")
            return

        payload = {
            "epic": epic,
            "direction": direction,
            "size": pos_size,
            "orderType": "MARKET",
            "stopLevel": stop_loss,
            "profitLevel": take_profit,
            "currencyCode":"USD"
        }

        current_api_url = get_active_api_url()
        print_log(f"📩 {epic} Payload:\n{json.dumps(payload, indent=2)}")
        resp = requests.post(
            f"{current_api_url}/api/v1/positions",
            headers={
                "X-CAP-API-KEY": API_KEY,
                "CST": cst,
                "X-SECURITY-TOKEN": x_sec_token,
                "Content-Type":"application/json"
            },
            json=payload
        )
        if resp.status_code in [200,201]:
            print_log(f"✅ {direction} Order => {resp.json().get('dealReference')}")
            trade_mgr.update_performance(0, epic)
        else:
            print_log(f"❌ Order fail => {resp.text}")
    except Exception as e:
        print_log(f"🔥 Execute trade error => {e}")

def manage_open_positions(cst, x_sec_token, classifier, scaler):
    pass


# --------------------------------------------------------------------
#   Main Trading Session
# --------------------------------------------------------------------
def trading_session():
    global classifier, scaler, FINAL_SYMBOL_LIST
    print_log("⚡ Start Trading Session")

    try:
        if not IS_DEMO:
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            if not is_market_hours(now_utc):
                print_log("⏸ Outside Market Hours => skip.")
                return
        else:
            print_log("🏖 DEMO => ignoring market hours check.")

        cst, x_sec_token = authenticate()

        # 1) Build our final symbol list if empty
        if not FINAL_SYMBOL_LIST:
            # fetch top 50 from "Most Traded"
            most_traded = get_most_traded_list(cst, x_sec_token, limit=50)

            # combine them with your base BLUE_CHIPS + INDICES
            from collections import OrderedDict

            combined = list(set(BLUE_CHIPS + INDICES + most_traded))
            # Sort them just to have a stable ordering
            combined.sort()

            FINAL_SYMBOL_LIST = combined
            print_log(f"✅ Using {len(FINAL_SYMBOL_LIST)} symbols => BlueChips+Indices+MostTraded")

        # 2) load existing model
        if ("classifier" not in globals()) or ("scaler" not in globals()) or classifier is None or scaler is None:
            classifier, scaler = load_model()

        # 3) if no model => train new (on the first valid symbol)
        if classifier is None or scaler is None:
            print_log("🟡 No saved model => training (first-run) on first valid symbol.")
            trained = False
            for epic in FINAL_SYMBOL_LIST:
                df_full = get_big_history(cst, x_sec_token, epic, resolution="DAY", years=2)
                if len(df_full)>200:
                    df_full = prepare_indicators(df_full)
                    if df_full.empty:
                        continue
                    print_log(f"🔄 Training classifier on {epic} data.")
                    classifier, scaler = train_classifier_model(
                        df_full, seq_length=SEQ_LENGTH, future_horizon=5, epochs=100
                    )
                    if classifier is not None and scaler is not None:
                        save_model(classifier, scaler)
                        trained = True
                        break
            if not trained:
                print_log("❌ No symbol had enough data => Skipping ML.")
                classifier, scaler = None, None

        # 4) manage existing positions if needed
        manage_open_positions(cst, x_sec_token, classifier, scaler)

        # 5) gather scores
        scored = []
        for epic in FINAL_SYMBOL_LIST:
            df = get_big_history(cst, x_sec_token, epic, resolution="DAY", years=3)
            print_log(f"🔎 {epic} => fetched {len(df)} rows from API")
            if len(df)<200:
                print_log(f"⚠️ {epic} => Not enough data => skip.")
                continue
            df = prepare_indicators(df)
            if df.empty:
                continue
            sc = calculate_score(df, classifier, scaler)
            print_log(f"📊 {epic} => Score {sc:.1f}")

            if sc>=70 or sc<=30:
                scored.append((epic, sc, df))

        # 6) sort by absolute score so strong signals come first
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        top = scored[:TOP_PICKS]

        # 7) Place trades for top picks
        for (epic,score_,df_) in top:
            if trade_mgr.can_take_new_trade(epic):
                print_log(f"🚀 Trading => {epic} => sc={score_}")
                execute_trade(cst, x_sec_token, epic, df_, score_)
                if trade_mgr.trade_count>= trade_mgr.max_daily_trades:
                    break

    except Exception as e:
        print_log(f"🔥 Session error => {e}")
    finally:
        print_log("🏁 session done")


# --------------------------------------------------------------------
#   MAIN
# --------------------------------------------------------------------
if __name__=="__main__":
    logging.basicConfig(
        filename="enhanced_bot.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    print_log("🤖 Enhanced ML Trader (Classification) with Indicators starts up.")
    classifier, scaler = None, None  # global references

    trading_session()  # run once

    schedule.every(30).minutes.do(trading_session)

    while True:
        schedule.run_pending()
        time.sleep(60)
