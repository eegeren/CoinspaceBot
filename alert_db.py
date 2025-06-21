import json
import os

ALERT_FILE = "alerts.json"

def load_alerts():
    if os.path.exists(ALERT_FILE):
        with open(ALERT_FILE, "r") as f:
            return json.load(f)
    return []

def save_alerts(alerts):
    with open(ALERT_FILE, "w") as f:
        json.dump(alerts, f, indent=2)

def add_alert(user_id, coin_id, target):
    alerts = load_alerts()
    alerts.append({
        "user_id": user_id,
        "coin_id": coin_id,
        "target": target
    })
    save_alerts(alerts)

def get_all_alerts():
    return load_alerts()

def delete_alert(user_id, coin_id):
    alerts = load_alerts()
    alerts = [a for a in alerts if not (a["user_id"] == user_id and a["coin_id"] == coin_id)]
    save_alerts(alerts)
