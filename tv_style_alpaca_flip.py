#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GME TRADING BOT - HYBRID VERSION (YAHOO + ALPACA)
Strategia: ADX 20 + EMA Cross
Feature Nou: Verificare "Lag" (Întârziere date)
"""

import os
import time
import json
import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================================================
#                    1. CONFIGURARE
# ============================================================

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET") or os.environ.get("APCA_API_SECRET_KEY")
ALPACA_BASE_URL = (os.environ.get("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

DEFAULT_SYMBOL = "GME"

# ============================================================
#                    2. UTILITARE
# ============================================================

def tg_send(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print(f"TG Error: {e}")

# ============================================================
#                    3. INDICATORI
# ============================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0); down = (-delta).clip(lower=0.0)
    rs = up.ewm(alpha=1/length, adjust=False).mean() / down.ewm(alpha=1/length, adjust=False).mean()
    return 100.0 - (100.0 / (1.0 + rs)).fillna(50.0)

def macd(close: pd.Series) -> Tuple[pd.Series, pd.Series]:
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd_line = fast - slow
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal

def adx(high, low, close, length=14):
    tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    up, down = high-high.shift(1), low.shift(1)-low
    
    # Conversie explicită pentru siguranță
    p_dm_val = np.where((up>down)&(up>0), up, 0.0)
    m_dm_val = np.where((down>up)&(down>0), down, 0.0)
    
    p_dm = pd.Series(p_dm_val, index=close.index)
    m_dm = pd.Series(m_dm_val, index=close.index)
    
    tr_s = tr.ewm(alpha=1/length, adjust=False).mean()
    p_di = 100 * (p_dm.ewm(alpha=1/length, adjust=False).mean() / tr_s)
    m_di = 100 * (m_dm.ewm(alpha=1/length, adjust=False).mean() / tr_s)
    dx = 100 * abs(p_di - m_di) / (p_di + m_di)
    return dx.ewm(alpha=1/length, adjust=False).mean().fillna(0.0)

# ============================================================
#                    4. ALPACA CLIENT
# ============================================================

class Alpaca:
    def __init__(self):
        if not ALPACA_API_KEY: raise RuntimeError("Lipsă API Key")
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET})

    def get_position(self, symbol):
        r = self.s.get(f"{ALPACA_BASE_URL}/v2/positions/{symbol}")
        return r.json() if r.status_code == 200 else None

    def close_position(self, symbol):
        self.s.delete(f"{ALPACA_BASE_URL}/v2/positions/{symbol}")

    def submit_order(self, **kwargs):
        self.s.post(f"{ALPACA_BASE_URL}/v2/orders", json=kwargs).raise_for_status()

# ============================================================
#                    5. LOGICA PRINCIPALĂ
# ============================================================

@dataclass
class Params:
    symbol: str = DEFAULT_SYMBOL
    qty: int = 250        
    state_path: str = "./state.json"
    adx_thresh: float = 20.0 

def get_yahoo_data(symbol, period="5d", interval="15m"):
    """
    Descarcă date Yahoo și repară structura MultiIndex.
    """
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        
        # 1. Aplatizare MultiIndex (Price, Ticker) -> (Price)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 2. Redenumire coloane alternative
        if 'Adj Close' in df.columns and 'Close' not in df.columns:
            df['Close'] = df['Adj Close']
            
        # 3. Păstrare coloane esențiale
        req = ['Open', 'High', 'Low', 'Close']
        df = df[[c for c in req if c in df.columns]].copy()
        
        return df.dropna()
    except Exception as e:
        print(f"Yahoo Error: {e}")
        return pd.DataFrame()

def run_once(p: Params):
    alp = Alpaca()
    now_utc = dt.datetime.now(dt.timezone.utc)
    
    # 1. DESCĂRCARE DATE
    df = get_yahoo_data(p.symbol, period="5d", interval="15m")
    if df.empty or len(df) < 50: 
        print(f"{p.symbol}: Date insuficiente Yahoo.")
        return

    # 2. CALCUL INDICATORI
    df["EMA"] = ema(df["Close"], 50)
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["SIG"] = macd(df["Close"])
    df["ADX"] = adx(df["High"], df["Low"], df["Close"])

    # 3. VERIFICARE "PROSPEȚIME" DATE (LAG CHECK)
    r = df.iloc[-1] # Ultima lumânare disponibilă
    last_ts = df.index[-1]
    
    # Convertim timestamp-ul din Yahoo la UTC (dacă nu e deja)
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=dt.timezone.utc)
    else:
        last_ts = last_ts.astimezone(dt.timezone.utc)
        
    # Calculăm diferența în minute
    lag_minutes = (now_utc - last_ts).total_seconds() / 60.0
    
    print(f"📅 Ultima lumânare: {last_ts.strftime('%H:%M:%S UTC')}")
    print(f"⏰ Ora curentă:     {now_utc.strftime('%H:%M:%S UTC')}")
    print(f"⚠️ Lag (Întârziere): {lag_minutes:.1f} minute")

    # REGULĂ DE SIGURANȚĂ: Dacă datele sunt mai vechi de 20 minute, NU tranzacționăm
    # (Yahoo are delay 15 min oficial, deci 20 e o marjă bună. Dacă e sub 20, e acceptabil)
    if lag_minutes > 25:
        print(f"⛔ STOP: Datele sunt prea vechi (>25 min). Piața e închisă sau Yahoo are delay.")
        # Putem lăsa botul să ruleze doar pt debug, dar nu executăm ordine.
        # Uncomment linia de mai jos dacă vrei să blochezi execuția:
        # return 

    price = float(r["Close"])
    
    # 4. STRATEGIA
    trend_ok = r["ADX"] > p.adx_thresh
    buy_signal = (price > r["EMA"]) and (r["MACD"] > r["SIG"]) and (r["RSI"] > 45) and trend_ok
    sell_signal = (price < r["EMA"]) and (r["MACD"] < r["SIG"]) and (r["RSI"] < 55) and trend_ok

    # 5. POZIȚIE CURENTĂ
    pos = alp.get_position(p.symbol)
    pos_qty = int(float(pos["qty"])) if pos else 0
    in_long = pos_qty > 0
    in_short = pos_qty < 0

    action = "HOLD"
    if buy_signal and not in_long: action = "LONG"
    elif sell_signal and not in_short: action = "SHORT"
    
    # 6. MEMORIA BOTULUI
    state_file = Path(p.state_path)
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    bar_key = f"{p.symbol}_{last_ts}" # Folosim timpul lumânării ca ID
    
    print(f"📊 {p.symbol} ${price:.2f} | ADX: {r['ADX']:.2f} | RSI: {r['RSI']:.1f} | Signal: {action}")

    if state.get("last_bar") == bar_key: 
        print(" -> Bară deja procesată. Aștept următoarea.")
        return

    # 7. EXECUȚIA
    if action != "HOLD":
        if pos: 
            alp.close_position(p.symbol)
            tg_send(f"🔄 FLIP {p.symbol} la ${price:.2f}")
            time.sleep(2)

        side = "buy" if action == "LONG" else "sell"
        try:
            alp.submit_order(symbol=p.symbol, qty=p.qty, side=side, type="market", time_in_force="day")
            tg_send(f"🚀 OPEN {action} {p.symbol}\nQty: {p.qty} @ ${price:.2f}\nADX: {r['ADX']:.2f}")
            state["last_bar"] = bar_key
            state_file.write_text(json.dumps(state))
        except Exception as e:
            print(f"Order Error: {e}")
            tg_send(f"❌ EROARE: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL)
    ap.add_argument("--qty", type=int, default=250)
    ap.add_argument("--state", default="./state.json")
    args = ap.parse_args()

    run_once(Params(symbol=args.symbol, qty=args.qty, state_path=args.state))

if __name__ == "__main__":
    main()
