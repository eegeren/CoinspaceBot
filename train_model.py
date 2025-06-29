import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

def train():
    print("📥 Veriler yükleniyor...")
    df = pd.read_csv("training_data.csv")  # open_time datetime olduğundan çıkarılmalı

    print("📊 Özellik sütunları:", df.columns.tolist())

    # open_time dışlanıyor çünkü sayısal değil
    features = [col for col in df.columns if col not in ["open_time", "target", "tp_pct", "sl_pct"]]
    X = df[features]
    y = df["target"]
    tp_y = df["tp_pct"]
    sl_y = df["sl_pct"]

    # Eğitim ve test bölünmesi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Sınıflandırma modeli
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"🎯 Classification Accuracy: {acc:.2f}")

    # TP regresyon modeli
    tp_model = RandomForestRegressor(n_estimators=100, random_state=42)
    tp_model.fit(X_train, tp_y)

    # SL regresyon modeli
    sl_model = RandomForestRegressor(n_estimators=100, random_state=42)
    sl_model.fit(X_train, sl_y)

    # Modelleri kaydet
    joblib.dump(clf, "model.pkl")
    joblib.dump(tp_model, "tp_model.pkl")
    joblib.dump(sl_model, "sl_model.pkl")

    # Özellik listesini kaydet
    joblib.dump(features, "features_list.pkl")

    print("✅ Modeller ve özellik listesi kaydedildi.")

if __name__ == "__main__":
    train()
