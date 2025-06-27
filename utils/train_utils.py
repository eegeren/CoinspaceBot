# utils/train_utils.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

def run_training():
    print("📂 Eğitim verisi yükleniyor...")
    df = pd.read_csv("training_data.csv")

    if "Price" in df.columns:
        price_column = "Price"
    elif "Close" in df.columns:
        price_column = "Close"
    elif "close" in df.columns:
        price_column = "close"
    else:
        raise ValueError("❌ 'Price' veya 'Close' sütunu bulunamadı.")

    df["TP_PCT"] = ((df["TP"] - df[price_column]) / df[price_column]) * 100
    df["SL_PCT"] = ((df[price_column] - df["SL"]) / df[price_column]) * 100

    features = [
        "RSI", "MACD", "Signal", "MA_5", "MA_20",
        "Volatility", "Momentum", "Price_Change", "Volume_Change"
    ]
    X = df[features]
    y = df["Label"]
    tp_pct = df["TP_PCT"]
    sl_pct = df["SL_PCT"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tp_train, tp_test = train_test_split(tp_pct, test_size=0.2, random_state=42)
    sl_train, sl_test = train_test_split(sl_pct, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [100],
        "max_depth": [10],
        "min_samples_split": [2]
    }

    signal_model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    signal_model.fit(X_train, y_train)
    joblib.dump(signal_model.best_estimator_, "model.pkl")

    tp_model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
    tp_model.fit(X_train, tp_train)
    joblib.dump(tp_model.best_estimator_, "tp_model.pkl")

    sl_model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
    sl_model.fit(X_train, sl_train)
    joblib.dump(sl_model.best_estimator_, "sl_model.pkl")

    print("✅ Eğitim tamamlandı.")
    return {
        "accuracy": signal_model.best_score_,
        "tp_mse": mean_squared_error(tp_test, tp_model.predict(X_test)),
        "sl_mse": mean_squared_error(sl_test, sl_model.predict(X_test))
    }
