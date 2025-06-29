import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from ta.trend import CCIIndicator, ADXIndicator, IchimokuIndicator
from ta.momentum import KAMAIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import sys

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def run_training(csv_path="training_data.csv"):
    logger = DualLogger("training_log.txt")
    sys.stdout = sys.stderr = logger

    print("📂 Eğitim verisi yükleniyor...")
    df = pd.read_csv(csv_path)

    required_cols = ["RSI", "MACD", "Signal", "MA_5", "MA_20", "Volatility",
                     "Momentum", "Price_Change", "Volume_Change", "TP", "SL", "Label"]
    optional_cols = ["Volume", "High", "Low", "Close"]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Eksik zorunlu sütun(lar): {missing}")

    for col in optional_cols:
        if col not in df.columns:
            print(f"⚠️ Opsiyonel sütun eksik: {col} — varsayılan olarak 0 atanıyor.")
            df[col] = 0

    print("🧪 Sütunlar:", df.columns.tolist())

    price_col = None
    for col in ["Price", "Close", "close"]:
        if col in df.columns:
            price_col = col
            break
    if not price_col:
        raise ValueError("❌ 'Price' veya 'Close' sütunu bulunamadı.")

    print("➕ Yeni göstergeler ekleniyor...")
    close = df[price_col]
    high = df.get("High", close)
    low = df.get("Low", close)
    volume = df["Volume"]

    df["CCI"] = CCIIndicator(high=high, low=low, close=close).cci()
    df["ADX"] = ADXIndicator(high=high, low=low, close=close).adx()
    df["KAMA"] = KAMAIndicator(close=close).kama()
    df["OBV"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df["CMF"] = ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume).chaikin_money_flow()
    df["ATR"] = AverageTrueRange(high=high, low=low, close=close).average_true_range()
    df["Force_Index"] = (close.diff() * volume).ewm(span=13, adjust=False).mean()
    stoch = StochasticOscillator(high=high, low=low, close=close)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    ichimoku = IchimokuIndicator(high=high, low=low)
    df["Tenkan"] = ichimoku.ichimoku_conversion_line()
    df["Kijun"] = ichimoku.ichimoku_base_line()
    df["Senkou_A"] = ichimoku.ichimoku_a()
    df["Senkou_B"] = ichimoku.ichimoku_b()
    bb = BollingerBands(close=close)
    df["BB_Band_Width"] = bb.bollinger_wband()

    print("🔁 TP_PCT ve SL_PCT hesaplanıyor...")
    df["TP_PCT"] = ((df["TP"] - close) / close) * 100
    df["SL_PCT"] = ((close - df["SL"]) / close) * 100

    features = [
        "RSI", "MACD", "Signal", "MA_5", "MA_20", "Volatility", "Momentum",
        "Price_Change", "Volume_Change", "CCI", "ADX", "KAMA", "OBV",
        "Force_Index", "CMF", "ATR", "Stoch_K", "Stoch_D",
        "Tenkan", "Kijun", "Senkou_A", "Senkou_B", "BB_Band_Width"
    ]

    print("📉 NaN içeren sütunlar:")
    print(df[features + ["TP", "SL", "Label"]].isna().sum().sort_values(ascending=False))

    df.dropna(subset=features + ["TP", "SL", "Label"], inplace=True)
    print(f"✅ Temizlenmiş veri boyutu: {df.shape}")

    X = df[features]
    y = df["Label"]
    tp_pct = df["TP_PCT"]
    sl_pct = df["SL_PCT"]

    print("✂️ Eğitim/test verisi ayrılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tp_train, tp_test = train_test_split(tp_pct, test_size=0.2, random_state=42)
    sl_train, sl_test = train_test_split(sl_pct, test_size=0.2, random_state=42)

    print("🤖 Sinyal modeli eğitiliyor...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    }
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    clf.fit(X_train, y_train)
    best_clf = clf.best_estimator_
    acc = accuracy_score(y_test, best_clf.predict(X_test))
    print(f"📈 Sinyal Model Doğruluk: {acc:.4f}")
    joblib.dump(best_clf, "model.pkl")
    joblib.dump(features, "features_list.pkl")

    print("📊 Özellik önemi kaydediliyor...")
    importances = best_clf.feature_importances_
    plt.figure(figsize=(12, 7))
    plt.barh(features, importances)
    plt.title("Özellik Önem Grafiği")
    plt.tight_layout()
    plt.savefig("feature_importance.png")

    print("🎯 TP modeli eğitiliyor...")
    tp_model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring="neg_mean_squared_error")
    tp_model.fit(X_train, tp_train)
    best_tp = tp_model.best_estimator_
    tp_mse = mean_squared_error(tp_test, best_tp.predict(X_test))
    print(f"🎯 TP MSE: {tp_mse:.4f}")
    joblib.dump(best_tp, "tp_model.pkl")

    print("🛑 SL modeli eğitiliyor...")
    sl_model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring="neg_mean_squared_error")
    sl_model.fit(X_train, sl_train)
    best_sl = sl_model.best_estimator_
    sl_mse = mean_squared_error(sl_test, best_sl.predict(X_test))
    print(f"🛑 SL MSE: {sl_mse:.4f}")
    joblib.dump(best_sl, "sl_model.pkl")

    print(f"✅ Eğitim tamamlandı. Veri şekli: {X.shape}")
    logger.close()

    return {
        "accuracy": round(acc, 4),
        "tp_mse": round(tp_mse, 4),
        "sl_mse": round(sl_mse, 4),
        "features_used": features
    }

if __name__ == "__main__":
    run_training()