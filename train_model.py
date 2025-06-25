import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Veri çekme
symbol = "BTC-USD"
print("Veri çekiliyor...")
data = yf.download(symbol, period="180d", interval="1h", auto_adjust=True)

# MultiIndex kontrolü
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

print(f"Raw data shape: {data.shape}")
if data.empty:
    raise ValueError("Veri seti boş!")

print("Close sütunu ilk 5 satır:", data["Close"].head())

# Teknik göstergeler
def calculate_rsi(prices, period=14):
    prices = pd.Series(prices)
    delta = prices.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, 0.001)
    rsi = 100 - (100 / (1 + rs))

    # BU SATIRI KALDIR (hatalı): print("RSI ilk 10 değer:\n", data["RSI"].head(10))
    return rsi

data["RSI"] = calculate_rsi(data["Close"])
print("RSI ilk 10 değer:\n", data["RSI"].head(10))  # ← buraya taşı


def calculate_macd(prices):
    prices = pd.Series(prices).dropna()
    if len(prices) < 26:
        return pd.Series([np.nan] * len(prices), index=prices.index), pd.Series([np.nan] * len(prices), index=prices.index)
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# Göstergeleri hesapla
data["RSI"] = calculate_rsi(data["Close"])
macd, signal = calculate_macd(data["Close"])
data["MACD"] = macd
data["Signal"] = signal
data["MA_5"] = data["Close"].rolling(window=5).mean()
data["MA_20"] = data["Close"].rolling(window=20).mean()

# Hedef sütunları
data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
data["TP_Target"] = data["Close"].shift(-5)
data["SL_Target"] = data["Close"].shift(-5) - 2 * data["Close"].rolling(window=5).std()

# Özellikleri ayıkla
features = data[["RSI", "MACD", "Signal", "MA_5", "MA_20"]].copy()
target = data["Target"].copy()
tp_target = data["TP_Target"].copy()
sl_target = data["SL_Target"].copy()

print("Raw features shape:", features.shape)
print("NaN values before imputation:", features.isna().sum())

# NaN temizliği
valid_index = features.notna().all(axis=1) & target.notna() & tp_target.notna() & sl_target.notna()
features = features[valid_index]
target = target[valid_index]
tp_target = tp_target[valid_index]
sl_target = sl_target[valid_index]

print("NaN values after cleaning:", features.isna().sum())

# Eğitim/test böl
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
tp_train, tp_test = train_test_split(tp_target, test_size=0.2, random_state=42)
sl_train, sl_test = train_test_split(sl_target, test_size=0.2, random_state=42)

# AI sinyal modeli
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"📈 Sinyal Model Doğruluk: {accuracy:.2f}")
joblib.dump(model, "model.pkl")

# TP tahmin modeli
tp_model = RandomForestRegressor(n_estimators=100, random_state=42)
tp_model.fit(X_train, tp_train)
joblib.dump(tp_model, "tp_model.pkl")
print("🎯 TP Model eğitildi ve kaydedildi.")

# SL tahmin modeli
sl_model = RandomForestRegressor(n_estimators=100, random_state=42)
sl_model.fit(X_train, sl_train)
joblib.dump(sl_model, "sl_model.pkl")
print("🛡️ SL Model eğitildi ve kaydedildi.")

print(f"✅ Tüm modeller başarıyla kaydedildi. Final veri şekli: {features.shape}")
