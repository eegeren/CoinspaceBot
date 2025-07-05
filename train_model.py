import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import os

def train():
    print("📥 Veriler yükleniyor...")
    if not os.path.exists("training_data.csv"):
        raise FileNotFoundError("❌ training_data.csv dosyası bulunamadı!")
    df = pd.read_csv("training_data.csv")

    # Özellik sütunlarını belirle (open_time hariç)
    features = [col for col in df.columns if col not in ["open_time", "target", "tp_pct", "sl_pct"]]
    if not features:
        raise ValueError("❌ Eğitim için özellik sütunu bulunamadı!")

    X = df[features]
    y = df["target"]
    tp_y = df["tp_pct"]
    sl_y = df["sl_pct"]

    # Eğitim ve test verisini bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tp_train, tp_test = train_test_split(tp_y, test_size=0.2, random_state=42)
    sl_train, sl_test = train_test_split(sl_y, test_size=0.2, random_state=42)

    # Boyutların eşleşmesini kontrol et
    assert X_train.shape[0] == tp_train.shape[0], f"Mismatch: {X_train.shape[0]} != {tp_train.shape[0]}"
    assert X_train.shape[0] == sl_train.shape[0], f"Mismatch: {X_train.shape[0]} != {sl_train.shape[0]}"

    # Sınıflandırma modeli
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"🎯 Sınıflandırma Doğruluğu: {acc:.2f}")

    # TP regresyon modeli
    tp_model = RandomForestRegressor(n_estimators=100, random_state=42)
    tp_model.fit(X_train, tp_train)

    # SL regresyon modeli
    sl_model = RandomForestRegressor(n_estimators=100, random_state=42)
    sl_model.fit(X_train, sl_train)

    # Modelleri ve özellik listesini kaydet
    try:
        joblib.dump(clf, "model.pkl")
        joblib.dump(tp_model, "tp_model.pkl")
        joblib.dump(sl_model, "sl_model.pkl")
        joblib.dump(features, "features_list.pkl")
        print("✅ Modeller ve özellik listesi kaydedildi.")
    except Exception as e:
        print(f"❌ Modelleri kaydetme hatası: {e}")

if __name__ == "__main__":
    train()
