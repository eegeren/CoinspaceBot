import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Veri çekme
symbol = "BTC-USD"
print("Veri çekiliyor...")
data = yf.download(symbol, period="180d", interval="1h", auto_adjust=True)

# MultiIndex sorununu çöz
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

# Veri boyutunu kontrol et
print(f"Raw data shape: {data.shape}")
if data.empty:
    raise ValueError("Veri seti boş! Lütfen internet bağlantısını ve sembolü kontrol edin.")

# Close sütununu kontrol et
print("Close sütunu ilk 5 satır:", data["Close"].head())

# Teknik göstergeler ekleme
def calculate_rsi(prices, period=14):
    # Fiyatların bir dizi olduğundan emin ol ve NaN'leri kaldır
    prices = pd.Series(prices).dropna()
    print(f"RSI için prices uzunluğu: {len(prices)}, ilk 5 değer: {prices.head()}")
    if len(prices) < period + 1:
        return pd.Series([np.nan] * len(prices), index=prices.index)
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 0.001)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    prices = pd.Series(prices).dropna()
    if len(prices) < 26:
        return pd.Series([np.nan] * len(prices), index=prices.index), pd.Series([np.nan] * len(prices), index=prices.index)
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# Teknik göstergeleri hesapla
data["RSI"] = calculate_rsi(data["Close"])
macd, signal = calculate_macd(data["Close"])
data["MACD"] = macd
data["Signal"] = signal
data["MA_5"] = data["Close"].rolling(window=5).mean()
data["MA_20"] = data["Close"].rolling(window=20).mean()

# Hedef değişkeni oluşturma
data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)

# Veri temizliği: NaN'leri doldurarak veri kaybını önle
features = data[["RSI", "MACD", "Signal", "MA_5", "MA_20"]].copy()
target = data["Target"].copy()

print("Raw features shape:", features.shape)
print("NaN values before imputation:", features.isna().sum())
# NaN'leri doldur
features = features.fillna(method='ffill').fillna(method='bfill')
target = target.fillna(method='ffill').fillna(method='bfill')
print("NaN values after imputation:", features.isna().sum())

# Eğitim ve test seti
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model eğitimi
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model doğruluğunu kontrol etme
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Modeli kaydetme
try:
    joblib.dump(model, "model.pkl")
    print("✅ Model 'model.pkl' olarak kaydedildi.")
except Exception as e:
    print(f"❌ Model kaydedilemedi: {e}")

print(f"Cleaned data shape: {data.shape}")