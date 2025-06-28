import os
from dotenv import load_dotenv
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

# .env dosyasını yükle
load_dotenv()

# API key ve secret'ı al
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Binance bağlantısı
client = Client(API_KEY, API_SECRET)

def fetch_ohlcv(symbol="BTCUSDT", interval="4h", days=90, save_path="raw_data.csv"):
    print(f"📊 Veri çekiliyor: {symbol}, {interval}, {days} gün")

    interval_map = {
        "1m": Client.KLINE_INTERVAL_1MINUTE,
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "30m": Client.KLINE_INTERVAL_30MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
    }

    if interval not in interval_map:
        raise ValueError("❌ Geçersiz zaman aralığı")

    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    klines = client.get_historical_klines(
        symbol,
        interval_map[interval],
        start_time.strftime("%d %b %Y %H:%M:%S"),
        requests_params={"timeout": 30}
    )

    if not klines:
        raise Exception("❌ Kline verisi alınamadı.")

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df.to_csv(save_path)
    
    print(f"✅ Veri kaydedildi: {save_path}")

if __name__ == "__main__":
    fetch_ohlcv(symbol="BTCUSDT", interval="4h", days=90)
