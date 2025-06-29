import json
import os

ALERT_FILE = "data/alerts.json"

def load_alerts():
    """Uyarıları JSON dosyasından yükler."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(ALERT_FILE):
        return []
    try:
        with open(ALERT_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("❌ Geçersiz JSON formatı, varsayılan liste döndürülüyor.")
        return []

def save_alerts(alerts):
    """Uyarıları JSON dosyasına kaydeder."""
    try:
        os.makedirs("data", exist_ok=True)
        with open(ALERT_FILE, "w") as f:
            json.dump(alerts, f, indent=4)
    except Exception as e:
        print(f"❌ Uyarıları kaydetme hatası: {e}")

def add_alert(user_id, coin_id, target):
    """Yeni bir uyarı ekler."""
    alerts = load_alerts()
    alerts.append({
        "user_id": user_id,
        "coin_id": coin_id,
        "target": target
    })
    save_alerts(alerts)

def get_all_alerts():
    """Tüm uyarıları döndürür."""
    return load_alerts()

def delete_alert(user_id, coin_id):
    """Belirli bir uyarıyı siler."""
    alerts = load_alerts()
    alerts = [a for a in alerts if not (a["user_id"] == user_id and a["coin_id"] == coin_id)]
    save_alerts(alerts)