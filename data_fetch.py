import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange

def fetch_binance_ohlcv(symbol="BTCUSDT", interval="1h", limit=3000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    return df[["open", "high", "low", "close", "volume"]]

def calculate_features(df):
    df["RSI"] = RSIIndicator(close=df["close"]).rsi()
    macd = MACD(close=df["close"])
    df["MACD"] = macd.macd()
    df["Signal"] = macd.macd_signal()
    df["MA_5"] = SMAIndicator(close=df["close"], window=5).sma_indicator()
    df["MA_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
    df["Volatility"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"]).average_true_range()
    df["Momentum"] = df["close"].diff()
    df["Price_Change"] = df["close"].pct_change() * 100
    df["Volume_Change"] = df["volume"].pct_change() * 100
    return df

def calculate_targets(df, tp_pct=2, sl_pct=1, future_steps=10):
    tp = []
    sl = []
    label = []

    for i in range(len(df)):
        future_prices = df["close"].iloc[i+1:i+1+future_steps]
        if len(future_prices) < future_steps:
            tp.append(np.nan)
            sl.append(np.nan)
            label.append(np.nan)
            continue

        entry = df["close"].iloc[i]
        tp_target = entry * (1 + tp_pct / 100)
        sl_target = entry * (1 - sl_pct / 100)

        hit_tp = (future_prices >= tp_target).any()
        hit_sl = (future_prices <= sl_target).any()

        if hit_tp and not hit_sl:
            label.append(1)
        else:
            label.append(0)

        tp.append(tp_target)
        sl.append(sl_target)

    df["TP"] = tp
    df["SL"] = sl
    df["Label"] = label
    return df

def fetch_training_data(symbol="BTCUSDT", interval="1h"):
    print(f"📡 {symbol} verisi çekiliyor...")
    df = fetch_binance_ohlcv(symbol, interval)
    df = calculate_features(df)
    df = calculate_targets(df)

    # Yalnızca gerekli sütunlara göre filtreleme
    required = ["RSI", "MACD", "Signal", "MA_5", "MA_20",
                "Volatility", "Momentum", "Price_Change", "Volume_Change",
                "TP", "SL", "Label"]
    df.dropna(subset=required, inplace=True)

    print(f"✅ Temizlenmiş veri boyutu: {df.shape}")
    if df.shape[0] < 50:
        print("❗ Uyarı: Eğitim için çok az veri kaldı. Daha uzun zaman aralığı veya farklı parite denenebilir.")

    # Opsiyonel kolonlar sıfırla
    df["X_Mentions"] = 0
    df["Google_Trends"] = 0
    df["Sentiment_Score"] = 0

    df.to_csv("training_data.csv", index=False)
    print("✅ Veri kaydedildi: training_data.csv")

if __name__ == "__main__":
    fetch_training_data()
