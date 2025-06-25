import pandas as pd
import joblib

# Modeli yükle
model = joblib.load("model.pkl")

# Özellik verilerini yükle (örnek: teknik göstergelerle zenginleştirilmiş veri)
df = pd.read_csv("features.csv")  # Bu dosya eğitimdeki formatla aynı olmalı

# Gerekirse hedef sütununu çıkar
if 'target' in df.columns:
    df = df.drop(columns=['target'])

# Tahmin yap
predictions = model.predict(df)

# Sonuçları ekrana yaz
for i, pred in enumerate(predictions):
    signal = "BUY" if pred == 1 else "SELL"
    print(f"Data Point {i + 1}: {signal}")
