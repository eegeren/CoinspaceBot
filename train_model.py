
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import pickle

def train():
    df = pd.read_csv("training_data.csv", index_col="timestamp", parse_dates=True)

    feature_cols = ["open", "high", "low", "close", "volume", "rsi", "macd", "sma_20", "atr"]
    X = df[feature_cols]
    y_class = df["target"]
    y_tp = df["TP_PCT"]
    y_sl = df["SL_PCT"]

    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Classification Accuracy: {acc:.2f}")

    # TP Model
    tp_model = RandomForestRegressor(n_estimators=100, random_state=42)
    tp_model.fit(X, y_tp)

    # SL Model
    sl_model = RandomForestRegressor(n_estimators=100, random_state=42)
    sl_model.fit(X, y_sl)

    # Save models
    joblib.dump(clf, "model.pkl")
    joblib.dump(tp_model, "tp_model.pkl")
    joblib.dump(sl_model, "sl_model.pkl")

    with open("features_list.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print("✅ Modeller ve özellik listesi kaydedildi.")

if __name__ == "__main__":
    train()
