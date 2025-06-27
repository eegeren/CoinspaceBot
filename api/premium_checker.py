import asyncio
import datetime
from bot import load_json, save_json, notify_user_if_expired  # notify_user_if_expired fonksiyonu lazim
from telegram import Bot

TELEGRAM_TOKEN = "7021119338:AAF7i5HCFOce6-9kESFzpkk51Nem7HZjckA"
bot = Bot(token=TELEGRAM_TOKEN)

async def check_and_notify_expired_premium():
    while True:
        print("🔁 Checking for expired premium users...")
        premium_users = load_json("data/premium_users.json")
        today = datetime.date.today()
        changed = False

        for user_id, info in premium_users.copy().items():
            try:
                end_date = datetime.datetime.strptime(info["end"], "%Y-%m-%d").date()
                if today > end_date:
                    # Süresi geçmiş kullanıcıya mesaj gönder
                    try:
                        await bot.send_message(
                            chat_id=user_id,
                            text="🔔 Your premium subscription has expired. To continue accessing full features, please renew via /premium"
                        )
                        print(f"🔔 Notified expired premium: {user_id}")
                    except Exception as e:
                        print(f"❌ Failed to message user {user_id}: {e}")

                    del premium_users[user_id]
                    changed = True
            except Exception as e:
                print(f"⚠️ Error processing user {user_id}: {e}")

        if changed:
            save_json("data/premium_users.json", premium_users)
        await asyncio.sleep(86400)  # 24 saat
