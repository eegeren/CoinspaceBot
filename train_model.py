import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error

# Eğitim verisini yükle
print("📂 Eğitim verisi yükleniyor...")
df = pd.read_csv("training_data.csv")

# Özellik ve hedef değişkenleri
features = [
    "RSI", "MACD", "Signal", "MA_5", "MA_20",
    "Volatility", "Momentum", "Price_Change", "Volume_Change"
]
X = df[features]
y = df["Label"]
tp_pct = df["TP_PCT"]
sl_pct = df["SL_PCT"]

# Eğitim/test bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tp_train, tp_test = train_test_split(tp_pct, test_size=0.2, random_state=42)
sl_train, sl_test = train_test_split(sl_pct, test_size=0.2, random_state=42)

# AI sinyal modeli (BUY/SELL) - Hiperparametre optimizasyonu
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
print(f"\n📈 Sinyal Model Doğruluk (Optimize): {accuracy:.2f}")
joblib.dump(best_model, "model.pkl")

# Özellik önemi
importances = best_model.feature_importances_
plt.figure(figsize=(10, 5))
plt.barh(features, importances)
plt.title("Feature Importances (BUY/SELL Model)")
plt.xlabel("Önem")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("📊 Özellik önem grafiği kaydedildi: feature_importance.png")

# TP_PCT tahmin modeli - Hiperparametre optimizasyonu
tp_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}
tp_model = RandomForestRegressor(random_state=42)
tp_grid_search = GridSearchCV(tp_model, tp_param_grid, cv=5, scoring='neg_mean_squared_error')
tp_grid_search.fit(X_train, tp_train)
best_tp_model = tp_grid_search.best_estimator_
tp_pred = best_tp_model.predict(X_test)
tp_mse = mean_squared_error(tp_test, tp_pred)
print(f"🎯 TP Model MSE: {tp_mse:.2f}")
joblib.dump(best_tp_model, "tp_model.pkl")
print("🎯 TP Model (yüzde) eğitildi ve kaydedildi.")

# SL_PCT tahmin modeli - Hiperparametre optimizasyonu
sl_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}
sl_model = RandomForestRegressor(random_state=42)
sl_grid_search = GridSearchCV(sl_model, sl_param_grid, cv=5, scoring='neg_mean_squared_error')
sl_grid_search.fit(X_train, sl_train)
best_sl_model = sl_grid_search.best_estimator_
sl_pred = best_sl_model.predict(X_test)
sl_mse = mean_squared_error(sl_test, sl_pred)
print(f"🛑 SL Model MSE: {sl_mse:.2f}")
joblib.dump(best_sl_model, "sl_model.pkl")
print("🛑 SL Model (yüzde) eğitildi ve kaydedildi.")

print(f"\n✅ Tüm modeller başarıyla kaydedildi. Final veri şekli: {X.shape}")