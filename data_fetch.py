# data_fetch.py

import pandas as pd
from binance.client import Client
from datetime import datetime
import time
import os

# Binance API anahtarları
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Binance istemcisi
client = Client(API_KEY, API_SECRET)

def fetch_ohlcv(symbol="BTCUSDT", interval="1h", limit=500):
    print(f"🔄 {symbol} için veri çekiliyor ({interval})...")
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    return df

if __name__ == "__main__":
    try:
        df = fetch_ohlcv(symbol="BTCUSDT", interval="1h", limit=500)
        df.to_csv("historical_data.csv", index=False)
        print("✅ historical_data.csv dosyası oluşturuldu.")
    except Exception as e:
        print(f"❌ Veri çekme hatası: {e}")
