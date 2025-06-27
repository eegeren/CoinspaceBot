import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error

# 📂 1. Eğitim verisini yükle
print("📂 Eğitim verisi yükleniyor...")
df = pd.read_csv("training_data.csv")

# 🧪 2. Sütunları kontrol et
print("🧪 Sütunlar:", df.columns.tolist())
if "Price" in df.columns:
    price_column = "Price"
elif "Close" in df.columns:
    price_column = "Close"
elif "close" in df.columns:
    price_column = "close"
else:
    raise ValueError("❌ 'Price' veya 'Close' sütunu bulunamadı. Lütfen eğitim verinizi kontrol edin.")

# 🔁 3. TP_PCT ve SL_PCT hesapla
print("🔁 TP_PCT ve SL_PCT hesaplanıyor...")
df["TP_PCT"] = ((df["TP"] - df[price_column]) / df[price_column]) * 100
df["SL_PCT"] = ((df[price_column] - df["SL"]) / df[price_column]) * 100

# 🎯 4. Özellik ve hedefleri ayır
features = [
    "RSI", "MACD", "Signal", "MA_5", "MA_20",
    "Volatility", "Momentum", "Price_Change", "Volume_Change"
]
X = df[features]
y = df["Label"]
tp_pct = df["TP_PCT"]
sl_pct = df["SL_PCT"]

# ✂️ 5. Eğitim/test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tp_train, tp_test = train_test_split(tp_pct, test_size=0.2, random_state=42)
sl_train, sl_test = train_test_split(sl_pct, test_size=0.2, random_state=42)

# 🤖 6. AI sinyal modeli (BUY/SELL) - GridSearch
print("🤖 AI sinyal modeli eğitiliyor...")
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}
signal_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(signal_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"📈 Sinyal Model Doğruluk: {accuracy:.2f}")
joblib.dump(best_model, "model.pkl")

# 📊 7. Özellik önem grafiği
importances = best_model.feature_importances_
plt.figure(figsize=(10, 5))
plt.barh(features, importances)
plt.title("Feature Importances (BUY/SELL Model)")
plt.xlabel("Önem")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("📊 Özellik önem grafiği kaydedildi: feature_importance.png")

# 🎯 8. TP_PCT regresyon modeli
print("🎯 TP Model eğitiliyor...")
tp_model = RandomForestRegressor(random_state=42)
tp_grid = GridSearchCV(tp_model, param_grid, cv=5, scoring='neg_mean_squared_error')
tp_grid.fit(X_train, tp_train)
best_tp_model = tp_grid.best_estimator_
tp_pred = best_tp_model.predict(X_test)
tp_mse = mean_squared_error(tp_test, tp_pred)
print(f"🎯 TP Model MSE: {tp_mse:.2f}")
joblib.dump(best_tp_model, "tp_model.pkl")

# 🛑 9. SL_PCT regresyon modeli
print("🛑 SL Model eğitiliyor...")
sl_model = RandomForestRegressor(random_state=42)
sl_grid = GridSearchCV(sl_model, param_grid, cv=5, scoring='neg_mean_squared_error')
sl_grid.fit(X_train, sl_train)
best_sl_model = sl_grid.best_estimator_
sl_pred = best_sl_model.predict(X_test)
sl_mse = mean_squared_error(sl_test, sl_pred)
print(f"🛑 SL Model MSE: {sl_mse:.2f}")
joblib.dump(best_sl_model, "sl_model.pkl")

# ✅ 10. Tamamlandı
print(f"\n✅ Tüm modeller başarıyla eğitildi ve kaydedildi. Final veri şekli: {X.shape}")
