import json
import datetime

FILE_PATH = "data/premium_users.json"

def fix_premium_users_json():
    try:
        with open(FILE_PATH, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            print("🔧 Fixing list-based premium_users.json...")

            fixed_data = {}
            today = datetime.date.today()
            for user_id in data:
                fixed_data[str(user_id)] = {
                    "start": str(today),
                    "end": str(today + datetime.timedelta(days=30))
                }

            with open(FILE_PATH, "w") as f:
                json.dump(fixed_data, f, indent=4)

            print("✅ premium_users.json successfully fixed.")
        else:
            print("✅ premium_users.json is already in correct format.")

    except Exception as e:
        print("❌ Error while fixing premium_users.json:", e)


if __name__ == "__main__":
    fix_premium_users_json()
