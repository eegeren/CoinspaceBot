import pandas as pd
import joblib
import os

def predict():
    if not os.path.exists("model.pkl"):
        raise FileNotFoundError("❌ model.pkl dosyası bulunamadı!")
    model = joblib.load("model.pkl")
    if not os.path.exists("features.csv"):
        raise FileNotFoundError("❌ features.csv dosyası bulunamadı!")
    df = pd.read_csv("features.csv")
    if 'target' in df.columns:
        df = df.drop(columns=['target'])
    predictions = model.predict(df)
    for i, pred in enumerate(predictions):
        signal = "BUY" if pred == 1 else "SELL"
        print(f"Data Point {i + 1}: {signal}")

if __name__ == "__main__":
    predict()