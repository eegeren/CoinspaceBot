import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import numpy as np

# Veriyi yükle
df = pd.read_csv("historical_data.csv")

# Teknik göstergeleri hesapla
df["RSI"] = RSIIndicator(close=df["close"]).rsi()
macd = MACD(close=df["close"])
df["MACD"] = macd.macd()
df["Signal"] = macd.macd_signal()
df["MA_5"] = df["close"].rolling(window=5).mean()
df["MA_20"] = df["close"].rolling(window=20).mean()
df["Volatility"] = df["close"].rolling(window=10).std()
df["Momentum"] = df["close"].diff()
df["Price_Change"] = df["close"].pct_change()
df["Volume_Change"] = df["volume"].pct_change()
df["CCI"] = CCIIndicator(high=df["high"], low=df["low"], close=df["close"]).cci()
df["ADX"] = ADXIndicator(high=df["high"], low=df["low"], close=df["close"]).adx()
df["OBV"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
df["CMF"] = ChaikinMoneyFlowIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]).chaikin_money_flow()
df["ATR"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"]).average_true_range()
df["Force_Index"] = df["close"].diff() * df["volume"]
stoch = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"])
df["Stoch_K"] = stoch.stoch()
df["Stoch_D"] = stoch.stoch_signal()
bb = BollingerBands(close=df["close"])
df["BB_Band_Width"] = bb.bollinger_hband() - bb.bollinger_lband()
df["Tenkan"] = (df["high"].rolling(window=9).max() + df["low"].rolling(window=9).min()) / 2
df["Kijun"] = (df["high"].rolling(window=26).max() + df["low"].rolling(window=26).min()) / 2
df["Senkou_A"] = (df["Tenkan"] + df["Kijun"]) / 2
df["Senkou_B"] = (df["high"].rolling(window=52).max() + df["low"].rolling(window=52).min()) / 2

# Hedef değişkenleri oluştur
future_window = 12  # örneğin 12 saatlik hareketi tahmin et
df["future_return"] = df["close"].shift(-future_window) / df["close"] - 1
df["target"] = (df["future_return"] > 0).astype(int)

# TP ve SL yüzdesi (pozisyon sonrası maksimum kar ve zarar)
df["tp_pct"] = (df["high"].rolling(window=future_window).max().shift(-future_window) / df["close"]) - 1
df["sl_pct"] = (df["low"].rolling(window=future_window).min().shift(-future_window) / df["close"]) - 1

# Temizlik
df.dropna(inplace=True)

# Kaydet
df.to_csv("training_data.csv", index=False)
print("✅ Feature engineering complete. Saved to training_data.csv")
