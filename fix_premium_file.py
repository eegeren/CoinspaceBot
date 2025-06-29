import json
import datetime
import os

FILE_PATH = "data/premium_users.json"

def fix_premium_users_json():
    """
    Premium kullanıcılar JSON dosyasını düzeltir veya oluşturur.
    """
    try:
        # Dosya yoksa boş bir dict ile başla
        if not os.path.exists(FILE_PATH):
            os.makedirs("data", exist_ok=True)
            with open(FILE_PATH, "w") as f:
                json.dump({}, f)
            print("✅ Yeni premium_users.json dosyası oluşturuldu.")

        with open(FILE_PATH, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            print("🔧 Listeye dayalı premium_users.json düzeltiliyor...")
            fixed_data = {}
            today = datetime.date.today()
            for user_id in data:
                fixed_data[str(user_id)] = {
                    "start": str(today),
                    "end": str(today + datetime.timedelta(days=30))
                }

            with open(FILE_PATH, "w") as f:
                json.dump(fixed_data, f, indent=4)

            print("✅ premium_users.json başarıyla düzeltildi.")
        else:
            print("✅ premium_users.json zaten doğru formatta.")

    except json.JSONDecodeError:
        print("❌ Geçersiz JSON formatı, dosya sıfırlanıyor...")
        with open(FILE_PATH, "w") as f:
            json.dump({}, f)
    except Exception as e:
        print(f"❌ premium_users.json düzeltme hatası: {e}")

if __name__ == "__main__":
    fix_premium_users_json()