import pandas as pd
import numpy as np
import requests
import time
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

symbol = "BTCUSDT"
interval = "1h"
limit = 1000
lookback_hours = 24 * 365

def fetch_binance_ohlcv(symbol, interval, lookback_hours):
    url = f"https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = int(time.time() * 1000)

    while len(all_data) * int(interval[:-1]) < lookback_hours:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "endTime": end_time
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Binance API error: {response.text}")
        data = response.json()
        if not data:
            break
        all_data = data + all_data
        end_time = data[0][0] - 1
        time.sleep(0.2)

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def add_indicators(df):
    df["RSI"] = RSIIndicator(df["close"]).rsi()
    macd = MACD(df["close"])
    df["MACD"] = macd.macd()
    df["Signal"] = macd.macd_signal()
    df["MA_5"] = SMAIndicator(df["close"], window=5).sma_indicator()
    df["MA_20"] = SMAIndicator(df["close"], window=20).sma_indicator()
    df["Volatility"] = df["close"].rolling(window=10).std()
    df["Momentum"] = df["close"] - df["close"].shift(10)
    df["Price_Change"] = df["close"].pct_change()
    df["Volume_Change"] = df["volume"].pct_change()
    return df

def label_rows(df, tp_pct=0.015, sl_pct=0.01, max_horizon=6):
    labels = []
    tp_values = []
    sl_values = []
    closes = df["close"].values
    for i in range(len(closes) - max_horizon):
        entry = closes[i]
        future = closes[i+1:i+1+max_horizon]
        tp_level = entry * (1 + tp_pct)
        sl_level = entry * (1 - sl_pct)
        label = None
        for price in future:
            if price >= tp_level:
                label = 1
                break
            elif price <= sl_level:
                label = 0
                break
        labels.append(label)
        tp_values.append(tp_level)
        sl_values.append(sl_level)

    pad = [None] * max_horizon
    df["Label"] = labels + pad
    df["TP"] = tp_values + pad
    df["SL"] = sl_values + pad
    return df

print("📥 Binance verisi alınıyor...")
df = fetch_binance_ohlcv(symbol, interval, lookback_hours)
print("📊 Teknik göstergeler hesaplanıyor...")
df = add_indicators(df)
print("🏷️ TP/SL etiketleri oluşturuluyor...")
df = label_rows(df)

# Temizleme
df = df.dropna(subset=["Label", "TP", "SL"])
features = ["RSI", "MACD", "Signal", "MA_5", "MA_20", "Volatility", "Momentum", "Price_Change", "Volume_Change"]
final_df = df[features + ["Label", "TP", "SL", "close"]].dropna()

final_df.to_csv("training_data.csv", index=False)
print(f"✅ Eğitim verisi kaydedildi: training_data.csv ({len(final_df)} satır)")
