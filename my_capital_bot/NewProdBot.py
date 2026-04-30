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
from numpy import nan as npNaN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# For advanced TA
import pandas_ta as ta

# === USER CONFIG ===
from config import (
    LOGIN, PASSWORD, API_KEY, API_URL, IS_DEMO, CUSTOMPASSWORD
)
from my_capital_bot.MLImprovedBot import get_active_api_url

# ============== GLOBAL CONFIG ==============
BLUE_CHIPS = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","NFLX","AMD",
    "META","TSM","V","MA","ORCL","UNH","XOM","MRK","AXP","ABT",
    "TXN","C","SCHW","BA","VRTX","JPM","LLY","PG",
    "JNJ","CRM","WMT","COST","HD","NVO","BAC","TMUS","KO","CVX",
    "WFC","PLTR","CSCO","ACN","IMB","PM","AVGO","INTC","SMCI",
    "MU"
]
INDICES = ["US500", "US100", "US30", "VOO"]
SYMBOL_LIST = BLUE_CHIPS + INDICES

RISK_PER_TRADE = 0.02
TIMEFRAME = "DAY"        # VALID: DAY, HOUR_4, WEEK
TOP_PICKS = 3
SEQ_LENGTH = 20

predictor = None
feature_scaler = None
price_scaler = None

# This dictionary will store a separate ML model for each symbol:
SYMBOL_MODELS = {}  # { epic: { 'model':..., 'data_scaler':..., 'close_scaler':..., 'last_train_date': datetime } }

FINAL_SYMBOL_LIST = []

# --------------------------------------------------------------------
#   Logging / Utility
# --------------------------------------------------------------------
def print_log(message):
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    txt = f"[{stamp}] {message}"
    print(txt)
    logging.info(txt)

def is_market_hours(dt):
    ny_tz = pytz.timezone("America/New_York")
    dt_local = dt.astimezone(ny_tz)
    if dt_local.weekday() >= 5:
        return False
    hour = dt_local.hour
    minute = dt_local.minute
    open_time = 9 * 60 + 30
    close_time = 16 * 60
    now_in_minutes = hour*60 + minute
    return (open_time <= now_in_minutes < close_time)

# ================== FETCH MOST TRADED ==================
def get_most_traded_list(cst, x_sec_token, limit=50):
    """
    Calls the "Most Traded" navigation endpoint to get up to 'limit' instruments
    and returns a list of epics (strings) we can trade.
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

# --------------------------------------------------------------------
#   Dynamic Risk Manager
# --------------------------------------------------------------------
class DynamicRiskManager:
    def __init__(self):
        self.volatility_factor = 1.0
        self.profit_lock = 0.5
    def adjust_risk(self, recent_performance):
        if recent_performance > 0:
            self.volatility_factor = max(0.8, self.volatility_factor * 0.95)
        else:
            self.volatility_factor = min(2.0, self.volatility_factor * 1.1)

risk_mgr = DynamicRiskManager()

# --------------------------------------------------------------------
#   Trade Manager
# --------------------------------------------------------------------
class TradeManager:
    def __init__(self):
        self.max_daily_trades = 25
        self.trade_count = 0
        self.profit_target = 5.0
        self.daily_loss_limit = -3.0
        self.active_trades = set()

    def can_take_new_trade(self, epic):
        return (
            self.trade_count < self.max_daily_trades and
            self.profit_target > self.daily_loss_limit and
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

# --------------------------------------------------------------------
#   Broker Auth / Equity
# --------------------------------------------------------------------
def authenticate():
    print_log("🛂 Authenticating with API...")
    try:
        resp = requests.post(
            f"{API_URL}/api/v1/session",
            headers={
                "X-CAP-API-KEY": API_KEY,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "identifier": LOGIN,
                "password":  CUSTOMPASSWORD,
                "encryptedPassword": False      # ⭐ NEW – required
            },
            timeout=10
        )
        resp.raise_for_status()
        cst = resp.headers["CST"]
        xst = resp.headers["X-SECURITY-TOKEN"]
        print_log("🔑 Auth success")
        return cst, xst
    except requests.HTTPError as e:
        print_log(f"❌ Authentication failed: {e} – check API-key status & creds")
        raise

def get_current_equity(cst, x_sec_token):
    try:
        resp = requests.get(
            f"{API_URL}/api/v1/accounts",
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

# --------------------------------------------------------------------
#   Data / Indicators
# --------------------------------------------------------------------
def fetch_data(cst, x_sec_token, epic, resolution="DAY", max_bars=500):
    try:
        params = {"resolution": resolution, "max": max_bars}
        resp = requests.get(
            f"{API_URL}/api/v1/prices/{epic}",
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
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print_log(f"❌ Fetch error for {epic}: {e}")
        return pd.DataFrame()

def prepare_main_indicators(df):
    if df.empty or len(df) < 200:
        return pd.DataFrame()

    df['sma50'] = df['close'].rolling(50, min_periods=50).mean()
    df['sma200'] = df['close'].rolling(200, min_periods=200).mean()
    df['tr'] = df['high'] - df['low']
    df['atr'] = df['tr'].rolling(14, min_periods=14).mean()
    df['support'] = df['low'].rolling(20).min()
    df['resistance'] = df['high'].rolling(20).max()
    return df.iloc[-200:]

def calculate_extra_indicators(df):
    if df.empty:
        return df

    st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
    if st is not None:
        df['supertrend']     = st['SUPERT_10_3.0']
        df['supertrend_dir'] = st['SUPERTd_10_3.0']

    sqz = ta.squeeze(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        bb_length=20,
        bb_std=2.0,
        kc_length=20,
        kc_factor=1.5
    )
    if sqz is not None:
        df['sqz_on'] = sqz['SQZ_ON']
        df['sqz_off'] = sqz['SQZ_OFF']
        df['sqz_no'] = sqz['SQZ_NO']

    stoch_rsi = ta.stochrsi(df['close'], length=14, rsi_length=14, k=3, d=3)
    if stoch_rsi is not None:
        df['stoch_rsi_k'] = stoch_rsi['STOCHRSIk_14_14_3_3']
        df['stoch_rsi_d'] = stoch_rsi['STOCHRSId_14_14_3_3']

    return df

def calculate_fib_levels(df, lookback=50):
    if df.empty or len(df) < lookback:
        return None
    sub = df.iloc[-lookback:]
    rh = sub['high'].max()
    rl = sub['low'].min()
    if rh <= rl:
        return None
    diff = rh - rl
    return {
        "23.6%": rh - diff*0.236,
        "38.2%": rh - diff*0.382,
        "50%":   rh - diff*0.50,
        "61.8%": rh - diff*0.618,
        "78.6%": rh - diff*0.786
    }

# ============== LSTM Dataset/Model for each symbol ==============
class PriceDataset(Dataset):
    def __init__(self, data, seq_length=SEQ_LENGTH):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length - 1

    def __getitem__(self, idx):
        seq = self.data[idx: idx + self.seq_length]
        target = self.data[idx + self.seq_length, 3]  # 'close' is column idx=3
        return torch.FloatTensor(seq), torch.FloatTensor([target])

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_lstm_for_symbol(epic, df):
    """
    Train an LSTM specifically for the given epic.
    Return (model, data_scaler, close_scaler, last_train_date).
    """
    try:
        print_log(f"🔄 Training LSTM model for {epic} ...")

        # Prepare the features
        features = df[['open','high','low','close','volume']].values
        data_scaler = StandardScaler()
        scaled_data = data_scaler.fit_transform(features)

        # Create sequences
        dataset = PriceDataset(scaled_data, SEQ_LENGTH)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Model
        model = LSTMPredictor(input_size=5)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train loop
        epochs = 50
        for epoch in range(epochs):
            total_loss=0
            for X_seq, Y_seq in loader:
                optimizer.zero_grad()
                out = model(X_seq)
                loss = criterion(out, Y_seq)
                loss.backward()
                optimizer.step()
                total_loss+= loss.item()
            print_log(f"{epic} | Epoch {epoch+1}/{epochs} Loss={total_loss/len(loader):.4f}")

        # Fit a close_scaler
        close_scaler = StandardScaler()
        close_scaler.fit(df['close'].values.reshape(-1,1))

        last_train_date = datetime.datetime.now()
        return model, data_scaler, close_scaler, last_train_date

    except Exception as e:
        print_log(f"{epic} => Model training failed: {str(e)}")
        return None, None, None, None

def get_ml_prediction_for_symbol(epic, df):
    """
    1) Check if we have a model in SYMBOL_MODELS for this epic, if not or stale => train
    2) Predict direction & confidence
    """
    # Ensure model is up to date
    if epic not in SYMBOL_MODELS:
        # train fresh
        model_, ds_, cs_, date_ = train_lstm_for_symbol(epic, df)
        if model_ is not None:
            SYMBOL_MODELS[epic] = {
                'model': model_,
                'data_scaler': ds_,
                'close_scaler': cs_,
                'last_train_date': date_
            }
        else:
            return None, 0.0
    else:
        # check last train date
        delta_days = (datetime.datetime.now() - SYMBOL_MODELS[epic]['last_train_date']).days
        if delta_days>=7:
            # retrain
            model_, ds_, cs_, date_ = train_lstm_for_symbol(epic, df)
            if model_ is not None:
                SYMBOL_MODELS[epic] = {
                    'model': model_,
                    'data_scaler': ds_,
                    'close_scaler': cs_,
                    'last_train_date': date_
                }
            else:
                return None,0.0

    # Now do predict
    try:
        md = SYMBOL_MODELS[epic]['model']
        ds = SYMBOL_MODELS[epic]['data_scaler']
        cs = SYMBOL_MODELS[epic]['close_scaler']

        # We need last SEQ_LENGTH rows
        if len(df)< SEQ_LENGTH:
            return None,0.0

        feats = df[['open','high','low','close','volume']].values[-SEQ_LENGTH:]
        feats_scaled = ds.transform(feats)

        x_ = torch.FloatTensor(feats_scaled).unsqueeze(0)
        with torch.no_grad():
            pred = md(x_).item()
        # invert
        cp_now = df['close'].iloc[-1]
        pred_real = cs.inverse_transform([[pred]])[0][0]
        diff = (pred_real - cp_now)/ cp_now
        confidence = min(abs(diff*100),100)

        direction = 'bull' if diff>0 else 'bear'
        return direction, confidence

    except Exception as e:
        print_log(f"{epic} => ML predict error => {e}")
        return None, 0.0

# --------------------------------------------------------------------
#   Weighted/Conditional Scoring with ML
# --------------------------------------------------------------------
def calculate_score(df, epic):
    if df.empty or len(df) < 200:
        return {'bull_score': 0, 'bear_score': 0, 'trend': 'none'}

    row = df.iloc[-1]
    cp = row['close']

    # 1) Basic Trend
    trend_score_bull = 0
    trend_score_bear = 0
    sma50 = row['sma50']
    sma200= row['sma200']
    if sma50 > sma200:
        trend_score_bull+=40
    else:
        trend_score_bear+=40

    if cp> sma200:
        trend_score_bull+=20
    else:
        trend_score_bear+=20

    # Supertrend
    if 'supertrend_dir' in row:
        if row['supertrend_dir']==1:
            trend_score_bull+=20
        elif row['supertrend_dir']==-1:
            trend_score_bear+=20

    # Momentum
    momentum_bull=0
    momentum_bear=0

    # RSI
    if 'rsi' in row and row['rsi']>0:
        if row['rsi']<30:
            momentum_bull+=15
        elif row['rsi']>70:
            momentum_bear+=15

    # Stoch RSI
    if 'stoch_rsi_k' in row:
        k_val = row['stoch_rsi_k']
        if k_val<20:
            momentum_bull+=10
        elif k_val>80:
            momentum_bear+=10

    # TTM Squeeze
    if 'sqz_on' in row and row['sqz_on']:
        momentum_bull+=5

    # Fib
    fib_bull=0
    fib_bear=0
    fibs= calculate_fib_levels(df, lookback=50)
    if fibs:
        if abs(cp- fibs["38.2%"])< cp*0.005:
            fib_bull+=10
        if abs(cp- fibs["61.8%"])< cp*0.005:
            fib_bull+=10

    # Vol
    vol_bull=0
    vol_bear=0
    atrv= row['atr'] if 'atr' in row else 0
    if atrv>0 and cp>0:
        atr_ratio= atrv/cp
        if 0.005< atr_ratio<0.05:
            vol_bull+=15
        elif atr_ratio>=0.05:
            vol_bear+=10

    # S/R
    sup= row['support'] if 'support' in row else 0
    res= row['resistance'] if 'resistance' in row else 999999
    if cp<= sup*1.02:
        trend_score_bull+=20
    if cp>= res*0.98:
        trend_score_bear+=10

    # Summation
    bull_score = trend_score_bull+ momentum_bull+ fib_bull+ vol_bull
    bear_score = trend_score_bear+ momentum_bear+ fib_bear+ vol_bear

    # Use ML
    ml_dir, ml_conf= get_ml_prediction_for_symbol(epic, df)
    if ml_dir == 'bull':
        bull_score += ml_conf*0.5
        print_log(f"📈 ML {epic} => bullish conf={ml_conf:.2f}")
    elif ml_dir == 'bear':
        bear_score += ml_conf*0.5
        print_log(f"📉 ML {epic} => bearish conf={ml_conf:.2f}")

    final = 'bull' if bull_score>= bear_score else 'bear'

    bull_score= max(min(bull_score,100),0)
    bear_score= max(min(bear_score,100),0)
    return {
        'bull_score': bull_score,
        'bear_score': bear_score,
        'trend': final
    }

# --------------------------------------------------------------------
#   Position Sizing & Execution
# --------------------------------------------------------------------
def get_instrument_details(cst, x_sec_token, epic):
    try:
        resp = requests.get(
            f"{API_URL}/api/v1/markets/{epic}",
            headers={
                "X-CAP-API-KEY":API_KEY,
                "CST": cst,
                "X-SECURITY-TOKEN": x_sec_token
            }
        )
        resp.raise_for_status()
        data= resp.json()
        inst= data.get("instrument",{})
        return {
            "min_step": float(inst.get("stepDistance", 0.01)),
            "min_stop": float(inst.get("minStopDistance", 0.05)),
            "lot_size": float(inst.get("lotSize", 1.0)),
            "precision": inst.get("precision", 2)
        }
    except Exception as e:
        print_log(f"⚠️ Instrument error => {e}")
        return None

def calculate_position_size(cp, stop_loss, risk_amount, instrument, is_long=True):
    try:
        diff= abs(cp- stop_loss)
        if diff==0:
            return 0
        raw= risk_amount/diff
        base= max(raw, instrument["lot_size"])
        if IS_DEMO:
            base= min(base, 5.0)
        else:
            base= min(base, 1.0)
        step= instrument["min_step"]
        normalized= round(base/ step)* step
        final_size= round(normalized, int(abs(np.log10(step))))
        print_log(f"🧮 Size => raw:{raw:.3f}, final:{final_size:.3f}")
        return final_size
    except Exception as e:
        print_log(f"Size error => {e}")
        return 0

def execute_trade(cst, x_sec_token, epic, df, bull_score, bear_score):
    try:
        instr= get_instrument_details(cst, x_sec_token, epic)
        if not instr or df.empty:
            return

        cp= round(df['close'].iloc[-1], instr["precision"])
        atr= df['atr'].iloc[-1]
        eq= get_current_equity(cst, x_sec_token)
        risk_amt= eq* RISK_PER_TRADE

        # decide direction
        if bull_score>=70 and bull_score>= bear_score:
            direction= "BUY"
            atr_stop= atr* risk_mgr.volatility_factor
            stop_loss= round(cp- (atr_stop*1.5), instr["precision"])
            take_profit= round(cp+ (atr_stop*3.0), instr["precision"])
        elif bear_score>=70 and bear_score> bull_score:
            direction= "SELL"
            atr_stop= atr* risk_mgr.volatility_factor
            stop_loss= round(cp+ (atr_stop*1.5), instr["precision"])
            take_profit= round(cp- (atr_stop*3.0), instr["precision"])
        else:
            print_log(f"⚪ No strong direction => skip {epic}")
            return

        size= calculate_position_size(cp, stop_loss, risk_amt, instr, direction=="BUY")
        if size<=0:
            print_log(f"❌ No valid size => skip {epic}")
            return

        payload= {
            "epic": epic,
            "direction": direction,
            "size": size,
            "orderType":"MARKET",
            "stopLevel": stop_loss,
            "profitLevel": take_profit,
            "currencyCode":"USD"
        }
        print_log(f"📩 {epic} Payload:\n{json.dumps(payload, indent=2)}")
        resp= requests.post(
            f"{API_URL}/api/v1/positions",
            headers={
                "X-CAP-API-KEY":API_KEY,
                "CST": cst,
                "X-SECURITY-TOKEN": x_sec_token,
                "Content-Type":"application/json"
            },
            json=payload
        )
        if resp.status_code in [200,201]:
            print_log(f"✅ {direction} => {resp.json().get('dealReference')}")
            trade_mgr.update_performance(0, epic)
        else:
            print_log(f"❌ Order failed => {resp.text}")

    except Exception as e:
        print_log(f"🔥 Execution error => {e}")

# --------------------------------------------------------------------
#   Trading Session
# --------------------------------------------------------------------
def trading_session():
    global FINAL_SYMBOL_LIST
    print_log("⚡ Start Trading Session")
    try:
        now_utc= datetime.datetime.now(pytz.utc)
        if not IS_DEMO and not is_market_hours(now_utc):
            print_log("⏸ Outside Market Hours => skip.")
            return

        cst, x_sec_token= authenticate()

        if not FINAL_SYMBOL_LIST:
            from my_capital_bot.MLImprovedBot import get_most_traded_list as _get_mt
            most_traded= _get_mt(cst, x_sec_token, limit=50)
            combined= list(set(SYMBOL_LIST + most_traded))
            combined.sort()
            FINAL_SYMBOL_LIST= combined
            print_log(f"✅ Using {len(FINAL_SYMBOL_LIST)} total symbols => base + most_traded")

        scored= []
        for epic in FINAL_SYMBOL_LIST:
            df= fetch_data(cst, x_sec_token, epic, resolution=TIMEFRAME, max_bars=500)
            if len(df)<200:
                continue

            df= prepare_main_indicators(df)
            if df.empty:
                continue

            df= calculate_extra_indicators(df)

            sc= calculate_score(df, epic)
            bull= sc['bull_score']
            bear= sc['bear_score']
            trend= sc['trend']
            print_log(f"📊 {epic} => bull:{bull:.1f} | bear:{bear:.1f} => trend:{trend}")

            if bull>=70 or bear>=70:
                scored.append((epic, bull, bear, df))

        scored.sort(key=lambda x: max(x[1], x[2]), reverse=True)
        top= scored[:TOP_PICKS]

        for (epic, bscore, bescore, df_) in top:
            if trade_mgr.can_take_new_trade(epic):
                print_log(f"🚀 Trading => {epic} => bull:{bscore:.1f} bear:{bescore:.1f}")
                execute_trade(cst, x_sec_token, epic, df_, bscore, bescore)
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
        filename="longterm_trader.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    print_log("🤖 Enhanced Dual-Direction ML Trader Activated (Daily, Each Symbol ML)")

    trading_session()  # run once
    schedule.every(30).minutes.do(trading_session)

    while True:
        schedule.run_pending()
        time.sleep(60)

