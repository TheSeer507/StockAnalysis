#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  Enhanced Dual‑Direction ML Trader  –  v2.2  (2025‑07‑18)
# ---------------------------------------------------------------------------
import logging, time, json, math, os, datetime as dt
from collections import deque
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
import pytz, requests, schedule
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

try:
    import pandas_ta as ta         # optional – not required
    _HAVE_PANDAS_TA = True
except ImportError:
    _HAVE_PANDAS_TA = False

# ---------------- USER / ENV CONFIG ----------------------------------------
from config import LOGIN, PASSWORD, API_KEY, API_URL, IS_DEMO, CUSTOMPASSWORD
from my_capital_bot.MLImprovedBot import get_active_api_url

# ======= STRATEGY & MODEL PARAMETERS =======================================
CFG: Dict[str, Any] = {
    "BLUE_CHIPS": [
        "AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","NFLX","AMD","META","TSM",
        "V","MA","ORCL","UNH","XOM","MRK","AXP","ABT","TXN","C","SCHW","BA",
        "VRTX","JPM","LLY","PG","JNJ","CRM","WMT","COST","HD","NVO","BAC",
        "TMUS","KO","CVX","WFC","PLTR","CSCO","ACN","IBM","PM","AVGO","INTC",
        "SMCI","MU"
    ],
    "INDICES": ["US500","US100","US30","VOO"],

    # Data
    "TIMEFRAME": "HOUR_4",
    "MAX_BARS": 800,

    # Indicator params
    "INDICATORS": {
        "sma_fast": 50, "sma_slow": 200, "atr_len": 14,
        "bb_len": 20, "bb_std": 2.0,
        "stoch_k": 14, "stoch_d": 3,
        "ichi_conv": 9, "ichi_base": 26, "ichi_span": 52,
    },

    # Scoring weights
    "WEIGHTS": {
        "trend": 40, "momentum": 30, "volatility": 15, "sr_zones": 15,
        "ml_conf_scale": 0.5, "score_threshold": 70,
    },

    # ML
    "SEQ_LEN": 30, "EPOCHS": 60, "BATCH": 32, "HIDDEN": 96,
    "MODEL_TYPE": "GRU", "RETRAIN_DAYS": 7,

    # Risk
    "ACCOUNT_RISK_PCT": 0.02,
    "TRAIL_ATR_MULT": 1.0, "TP_ATR_MULT": 3.0, "SL_ATR_MULT": 1.5,
    "MAX_DAILY_TRADES": 25, "DAILY_PROFIT_TARGET": 5.0,
    "DAILY_LOSS_LIMIT": -3.0,
}

SYMBOL_LIST: List[str] = CFG["BLUE_CHIPS"] + CFG["INDICES"]
SYMBOL_MODELS: Dict[str, Dict[str, Any]] = {}
TRADE_HISTORY: deque = deque(maxlen=500)

# ---------------- LOGGING --------------------------------------------------
LOG_DIR = Path("./logs"); LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "longterm_trader_v2_2.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
def log(msg: str) -> None:
    print(msg); logging.info(msg)

# ---------------- TIME HELPERS --------------------------------------------
def in_market_hours(ts: dt.datetime) -> bool:
    ny = pytz.timezone("America/New_York")
    loc = ts.astimezone(ny)
    if loc.weekday() >= 5: return False
    return 9*60+30 <= loc.hour*60+loc.minute < 16*60

# ---------------- AUTH -----------------------------------------------------
def authenticate() -> Tuple[str, str]:
    log("🛂 Authenticating …")
    r = requests.post(
        f"{API_URL}/api/v1/session",
        headers={
            "X-CAP-API-KEY": API_KEY, "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={"identifier": LOGIN, "password": CUSTOMPASSWORD,
              "encryptedPassword": False},
        timeout=10,
    )
    r.raise_for_status(); log("🔑 Auth OK")
    return r.headers["CST"], r.headers["X-SECURITY-TOKEN"]

def account_equity(cst: str, xst: str) -> float:
    r = requests.get(f"{API_URL}/api/v1/accounts",
        headers={"X-CAP-API-KEY": API_KEY,"CST": cst,"X-SECURITY-TOKEN": xst})
    r.raise_for_status()
    bal = float(r.json()["accounts"][0]["balance"]["balance"])
    log(f"💰 Equity: {bal:,.2f}"); return bal

# ---------------- MOST TRADED ---------------------------------------------
def get_most_traded_list(cst: str, xst: str, limit: int=50)->List[str]:
    base = get_active_api_url() or API_URL
    url = f"{base}/api/v1/marketnavigation/hierarchy_v1.commons.most_traded?limit=500"
    try:
        r = requests.get(url, headers={
            "X-CAP-API-KEY": API_KEY,"CST": cst,"X-SECURITY-TOKEN": xst})
        r.raise_for_status()
        epics=[m["epic"] for m in r.json().get("markets",[]) if m.get("epic")]
        log(f"🔎 most‑traded fetched: {len(epics[:limit])}")
        return epics[:limit]
    except Exception as ex:
        log(f"❌ most_traded error: {ex}"); return []

# ---------------- DATA -----------------------------------------------------
def fetch_prices(cst: str, xst: str, epic: str) -> pd.DataFrame:
    try:
        r = requests.get(f"{API_URL}/api/v1/prices/{epic}",
            headers={"X-CAP-API-KEY": API_KEY,"CST": cst,
                     "X-SECURITY-TOKEN": xst},
            params={"resolution": CFG["TIMEFRAME"],"max": CFG["MAX_BARS"]},
            timeout=10)
        r.raise_for_status()
        raw = r.json().get("prices", [])
        if not raw: return pd.DataFrame()
        df = pd.DataFrame([{
            "timestamp": p["snapshotTimeUTC"],
            "open": p["openPrice"]["bid"], "high": p["highPrice"]["bid"],
            "low": p["lowPrice"]["bid"], "close": p["closePrice"]["bid"],
            "volume": p.get("lastTradedVolume",0)
        } for p in raw])
        df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True,errors="coerce")
        return (df.dropna(subset=["timestamp"])
                  .set_index("timestamp").sort_index())
    except Exception as ex:
        log(f"❌ fetch {epic}: {ex}"); return pd.DataFrame()

# ---------------- INDICATORS ----------------------------------------------
def _rsi(series: pd.Series, n: int=14)->pd.Series:
    delta = series.diff(); up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up/roll_dn; return 100 - 100/(1+rs)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    p=CFG["INDICATORS"]; df=df.copy()
    if len(df)<max(p["sma_slow"],p["bb_len"],p["ichi_span"]): return pd.DataFrame()

    # SMA, ATR
    df["sma_fast"]=df["close"].rolling(p["sma_fast"]).mean()
    df["sma_slow"]=df["close"].rolling(p["sma_slow"]).mean()
    prev=df["close"].shift(1)
    tr=pd.concat([(df["high"]-df["low"]),
                  (df["high"]-prev).abs(),
                  (df["low"]-prev).abs()],axis=1).max(axis=1)
    df["atr"]=tr.rolling(p["atr_len"]).mean()

    # Bollinger
    mid=df["close"].rolling(p["bb_len"]).mean()
    std=df["close"].rolling(p["bb_len"]).std(ddof=0)
    df["bb_low"]=mid-p["bb_std"]*std; df["bb_high"]=mid+p["bb_std"]*std
    df["bb_perc"]=(df["close"]-df["bb_low"])/(df["bb_high"]-df["bb_low"])

    # Stoch
    hh=df["high"].rolling(p["stoch_k"]).max()
    ll=df["low"].rolling(p["stoch_k"]).min()
    k=(df["close"]-ll)/(hh-ll)*100
    df["stoch_k"]=k; df["stoch_d"]=k.rolling(p["stoch_d"]).mean()

    # RSI
    df["rsi"]=_rsi(df["close"],14)

    # Ichimoku
    conv=(df["high"].rolling(p["ichi_conv"]).max()+
          df["low"].rolling(p["ichi_conv"]).min())/2
    base=(df["high"].rolling(p["ichi_base"]).max()+
          df["low"].rolling(p["ichi_base"]).min())/2
    spanA=(conv+base)/2
    spanB=(df["high"].rolling(p["ichi_span"]).max()+
           df["low"].rolling(p["ichi_span"]).min())/2
    df["ichi_conv"]=conv; df["ichi_base"]=base
    df["ichi_spanA"]=spanA; df["ichi_spanB"]=spanB
    top=pd.concat([spanA,spanB],axis=1).max(axis=1)
    bot=pd.concat([spanA,spanB],axis=1).min(axis=1)
    df["ichi_dist"]=np.where(df["close"]>top,
        (df["close"]-top)/df["close"],
        np.where(df["close"]<bot,(bot-df["close"])/df["close"],0))

    # Support/Resistance
    df["support"]=df["low"].rolling(20).min()
    df["resistance"]=df["high"].rolling(20).max()

    return df.dropna()

# ---------------- SCORING --------------------------------------------------
def score_row(r: pd.Series)->Dict[str,float]:
    w=CFG["WEIGHTS"]; bull=bear=0.0
    if r["sma_fast"]>r["sma_slow"]: bull+=w["trend"]*0.6
    else: bear+=w["trend"]*0.6
    if r["ichi_dist"]>0: bull+=w["trend"]*0.4
    elif r["ichi_dist"]<0: bear+=w["trend"]*0.4
    if r["rsi"]<30 or r["bb_perc"]<0.1 or r["stoch_k"]<20: bull+=w["momentum"]
    if r["rsi"]>70 or r["bb_perc"]>0.9 or r["stoch_k"]>80: bear+=w["momentum"]
    atr_pct=r["atr"]/r["close"]
    if 0.005<atr_pct<0.05: bull+=w["volatility"]*0.7
    elif atr_pct>=0.05: bear+=w["volatility"]*0.7
    if r["close"]<=r["support"]*1.02: bull+=w["sr_zones"]
    if r["close"]>=r["resistance"]*0.98: bear+=w["sr_zones"]
    return {"bull":min(bull,100),"bear":min(bear,100)}

# ---------------- ML -------------------------------------------------------
class PriceDataset(Dataset):
    def __init__(self,arr:np.ndarray,seq:int): self.arr, self.seq=arr, seq
    def __len__(self): return len(self.arr)-self.seq-1
    def __getitem__(self,idx):
        X=self.arr[idx:idx+self.seq]; y=self.arr[idx+self.seq,3]
        return torch.tensor(X,dtype=torch.float32),torch.tensor([y],dtype=torch.float32)
class RNNPredictor(nn.Module):
    def __init__(self,input_size:int,hidden:int,mdl:str="GRU"):
        super().__init__()
        rnn=nn.GRU if mdl.upper()=="GRU" else nn.LSTM
        self.rnn=rnn(input_size,hidden,num_layers=2,batch_first=True)
        self.fc=nn.Linear(hidden,1)
    def forward(self,x): out,_=self.rnn(x); return self.fc(out[:,-1,:])

def _train_model(epic:str,df:pd.DataFrame)->None:
    log(f"🔄 train {epic}")
    feats=["open","high","low","close","volume",
           "atr","bb_perc","stoch_d","ichi_dist"]
    data=df[feats].values
    sc=StandardScaler(); scaled=sc.fit_transform(data)
    ds=PriceDataset(scaled,CFG["SEQ_LEN"])
    if len(ds)<5: log(f"⚠️ few samples {epic}"); return
    dl=DataLoader(ds,batch_size=CFG["BATCH"],shuffle=True)
    mdl=RNNPredictor(scaled.shape[1],CFG["HIDDEN"],CFG["MODEL_TYPE"])
    opt=optim.Adam(mdl.parameters(),lr=1e-3); crit=nn.MSELoss()
    for e in range(CFG["EPOCHS"]):
        tot=0.0
        for Xb,yb in dl:
            opt.zero_grad(); loss=crit(mdl(Xb),yb); loss.backward(); opt.step()
            tot+=loss.item()
        log(f"{epic} ep{e+1}/{CFG['EPOCHS']} loss {tot/len(dl):.4f}")
    SYMBOL_MODELS[epic]={
        "model":mdl.eval(),"scaler":sc,
        "last_train":dt.datetime.utcnow(),
        "close_scaler":StandardScaler().fit(df["close"].values.reshape(-1,1))
    }

def ml_predict(epic:str,df:pd.DataFrame)->Tuple[Optional[str],float]:
    if len(df)<CFG["SEQ_LEN"]: return None,0.0
    mdl=SYMBOL_MODELS.get(epic)
    if mdl is None or (dt.datetime.utcnow()-mdl["last_train"]).days>=CFG["RETRAIN_DAYS"]:
        _train_model(epic,df); mdl=SYMBOL_MODELS.get(epic)
        if mdl is None: return None,0.0
    feats=["open","high","low","close","volume",
           "atr","bb_perc","stoch_d","ichi_dist"]
    X=mdl["scaler"].transform(df[feats].values[-CFG["SEQ_LEN"]:])
    with torch.no_grad():
        pred=mdl["model"](torch.tensor(X,dtype=torch.float32).unsqueeze(0)).item()
    pred_real=mdl["close_scaler"].inverse_transform([[pred]])[0][0]
    diff=(pred_real-df["close"].iloc[-1])/df["close"].iloc[-1]
    return ("bull" if diff>0 else "bear"),min(abs(diff)*100,100)

# ---------------- RISK & TELEMETRY ----------------------------------------
class TradeManager:
    def __init__(self): self.max=CFG["MAX_DAILY_TRADES"]; self.reset_day()
    def reset_day(self): self.count=0; self.day_pnl=0.0
    def allowed(self): return self.count<self.max
    def register(self,pnl:float): self.count+=1; self.day_pnl+=pnl; TRADE_HISTORY.append(pnl)
tm=TradeManager()
def telemetry():
    if len(TRADE_HISTORY)<30: return
    pnl=np.array(TRADE_HISTORY)
    sharpe=(pnl.mean()/pnl.std(ddof=1))*math.sqrt(252) if pnl.std(ddof=1) else 0
    cum=np.cumsum(pnl); peak=np.maximum.accumulate(cum)
    dd=(peak-cum).max(); dd_pct=dd/peak.max() if peak.max() else 0
    log(f"📈 Sharpe {sharpe:.2f}  MaxDD {dd_pct:.2%}")

# ---------------- EXECUTION ------------------------------------------------
CFG.setdefault("SEND_STOPS", True)          # flip to False to omit SL/TP at source
CFG.setdefault("RETRY_NO_STOPS", True)      # try again without stops on rejection


def instrument_meta(cst: str, xst: str, epic: str) -> Dict[str, Any]:
    r = requests.get(f"{API_URL}/api/v1/markets/{epic}",
                     headers={"X-CAP-API-KEY": API_KEY,
                              "CST": cst,
                              "X-SECURITY-TOKEN": xst},
                     timeout=10)
    r.raise_for_status()
    js   = r.json()
    inst = js.get("instrument", {})
    snap = js.get("snapshot", {})
    return {
        "step":       int(float(inst.get("stepDistance", 1))),          # **int points**
        "min_stop":   int(float(inst.get("minStopDistance", 1))),
        "lot":        float(inst.get("lotSize", 1)),
        "prec":       int(inst.get("precision", 2)),
        "status":     snap.get("marketStatus", "UNKNOWN"),
    }


def _ceiling_multiple(x: float, step: int) -> int:
    """Return the smallest int >= x that is a multiple of *step*."""
    return int(math.ceil(x / step) * step)


def _position_size(eq: float, risk_amt: float, stop_pts: int,
                   lot: float) -> float:
    if stop_pts <= 0:
        return 0.0
    raw = risk_amt / stop_pts
    size = max(raw, lot)
    size = min(size, 5.0 if IS_DEMO else 1.0)
    return math.floor(size / lot) * lot


def _submit(payload: dict, cst: str, xst: str) -> Tuple[bool, str]:
    r = requests.post(f"{API_URL}/api/v1/positions",
                      headers={"X-CAP-API-KEY": API_KEY,
                               "CST": cst,
                               "X-SECURITY-TOKEN": xst,
                               "Content-Type": "application/json"},
                      json=payload,
                      timeout=10)
    ok = r.status_code in (200, 201)
    return ok, r.text


def execute_trade(cst: str, xst: str, epic: str,
                  row: pd.Series, bull: float, bear: float) -> None:

    inst = instrument_meta(cst, xst, epic)
    if inst["status"] != "TRADEABLE":
        log(f"⏸ {epic} market closed")
        return

    cp   = round(float(row["close"]), inst["prec"])
    atr  = float(row["atr"])
    step = inst["step"]
    eq   = account_equity(cst, xst)
    risk = eq * CFG["ACCOUNT_RISK_PCT"]

    # ---- pick direction ---------------------------------------------------
    if bull >= CFG["WEIGHTS"]["score_threshold"] and bull >= bear:
        side = "BUY"
    elif bear >= CFG["WEIGHTS"]["score_threshold"] and bear > bull:
        side = "SELL"
    else:
        return

    # ---- distance in **points** -------------------------------------------
    target_pts = max(inst["min_stop"],
                     _ceiling_multiple(atr * CFG["SL_ATR_MULT"], step))

    # profit distance symmetrical (integer points)
    profit_pts  = _ceiling_multiple(target_pts *
                                    CFG["TP_ATR_MULT"] / CFG["SL_ATR_MULT"],
                                    step)

    size = _position_size(eq, risk, target_pts, inst["lot"])
    if size <= 0:
        log(f"❌ {epic} size calc ≤ 0")
        return

    base_payload = {
        "epic":      epic,
        "direction": side,
        "size":      size,
        "orderType": "MARKET",
        "currencyCode": "USD",
    }

    if CFG["SEND_STOPS"]:
        base_payload.update({
            "stopDistance":   target_pts,
            "profitDistance": profit_pts,
            "trailingStop":   True,
            "trailingStopIncrement": target_pts,  # safest
        })

    log(f"📩 {epic} {side} size={size} stopPts={target_pts} "
        f"profitPts={profit_pts if CFG['SEND_STOPS'] else 'N/A'}")

    ok, txt = _submit(base_payload, cst, xst)

    # ------------- retry without stops on stopDistance error ---------------
    if (not ok and CFG["SEND_STOPS"] and CFG["RETRY_NO_STOPS"]
            and "stopDistance" in txt):
        log(f"⚠️  {epic} rejected for stopDistance – retrying plain market …")
        plain = {k: base_payload[k] for k in
                 ("epic", "direction", "size", "orderType", "currencyCode")}
        ok, txt = _submit(plain, cst, xst)

    if ok:
        try:
            deal = json.loads(txt).get("dealReference", "??")
        except Exception:
            deal = "??"
        log(f"✅ {epic} dealRef {deal}")
        tm.register(0.0)
    else:
        log(f"❌ {epic} order failed → {txt}")

# ---------------- SESSION --------------------------------------------------
_FINAL_SYMBOLS: List[str]=[]
def trading_session():
    global _FINAL_SYMBOLS
    now=dt.datetime.utcnow().replace(tzinfo=pytz.utc)
    if not IS_DEMO and not in_market_hours(now):
        log("⏸ Out of market hours"); return
    try:
        cst,xst=authenticate()
        if not _FINAL_SYMBOLS:
            _FINAL_SYMBOLS=sorted(set(SYMBOL_LIST+get_most_traded_list(cst,xst)))
            log(f"Universe {_FINAL_SYMBOLS[:10]} … total {len(_FINAL_SYMBOLS)}")
        picks=[]
        for epic in _FINAL_SYMBOLS:
            df=fetch_prices(cst,xst,epic); df=add_indicators(df)
            if df.empty: continue
            row=df.iloc[-1]; sc=score_row(row)
            ml_dir,conf=ml_predict(epic,df)
            if ml_dir=="bull": sc["bull"]+=conf*CFG["WEIGHTS"]["ml_conf_scale"]
            elif ml_dir=="bear": sc["bear"]+=conf*CFG["WEIGHTS"]["ml_conf_scale"]
            log(f"📊 {epic}: bull{sc['bull']:.1f} bear{sc['bear']:.1f}")
            if max(sc["bull"],sc["bear"])>=CFG["WEIGHTS"]["score_threshold"]:
                picks.append((epic,sc,row))
        picks.sort(key=lambda x:max(x[1]["bull"],x[1]["bear"]),reverse=True)
        for epic,sc,row in picks[:10]:
            if tm.allowed(): execute_trade(cst,xst,epic,row,sc["bull"],sc["bear"])
    except Exception as ex:
        log(f"🔥 session error {ex}")
    finally: telemetry(); log("🏁 session done")

# ---------------- MAIN -----------------------------------------------------
if __name__=="__main__":
    log("🤖 Trader v2.2 starting")
    trading_session()
    schedule.every(30).minutes.do(trading_session)
    while True:
        schedule.run_pending(); time.sleep(60)
