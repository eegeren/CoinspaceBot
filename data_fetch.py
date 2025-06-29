import pandas as pd
from binance.client import Client
from datetime import datetime
import time
import os

# Binance API anahtarlarını kontrol et
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("❌ BINANCE_API_KEY veya BINANCE_API_SECRET çevre değişkenleri eksik!")

# Binance istemcisi
client = Client(API_KEY, API_SECRET)

def fetch_ohlcv(symbol="BTCUSDT", interval="1h", limit=500):
    """
    Binance'tan OHLCV verisi çeker ve pandas DataFrame'e dönüştürür.
    
    Args:
        symbol (str): İşlem çifti (ör. BTCUSDT)
        interval (str): Zaman aralığı (ör. 1h)
        limit (int): Veri limiti
    
    Returns:
        pd.DataFrame: OHLCV verisi
    """
    try:
        print(f"🔄 {symbol} için veri çekiliyor ({interval})...")
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

        if not klines:
            raise ValueError("Boş veri alındı!")

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

    except Exception as e:
        print(f"❌ Veri çekme hatası: {e}")
        return None

if __name__ == "__main__":
    df = fetch_ohlcv(symbol="BTCUSDT", interval="1h", limit=500)
    if df is not None:
        df.to_csv("historical_data.csv", index=False)
        print("✅ historical_data.csv dosyası oluşturuldu.")
    else:
        print("❌ Veri çekme başarısız, distribütör oluşturulmadı.")