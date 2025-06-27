# api/payment_api.py

from fastapi import APIRouter, Request
import datetime
from bot import load_json, save_json

router = APIRouter()

@router.post("/payment_callback")
async def payment_callback(request: Request):
    try:
        data = await request.json()
        print("🚀 Payment callback received:", data)

        if data.get("payment_status") == "finished":
            user_id = str(data.get("order_id"))  # Telegram user ID

            # JSON'u güvenli şekilde yükle, bozuksa boş dict olarak başla
            try:
                premium_users = load_json("data/premium_users.json")
                if not isinstance(premium_users, dict):
                    premium_users = {}
            except Exception:
                premium_users = {}

            today = datetime.datetime.now()
            end_date = today + datetime.timedelta(days=30)

            # Yeni kullanıcıyı kaydet
            premium_users[user_id] = {
                "start": today.strftime("%Y-%m-%d %H:%M:%S"),
                "end": end_date.strftime("%Y-%m-%d %H:%M:%S")
            }

            save_json("data/premium_users.json", premium_users)

            return {"status": "success", "message": f"✅ Premium granted to user {user_id}"}
        else:
            return {"status": "ignored", "message": "Payment not completed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}



@router.get("/check_premium/{user_id}")
def check_premium(user_id: str):
    premium_users = load_json("data/premium_users.json")
    
    if not isinstance(premium_users, dict):
        return {"status": "error", "message": "Invalid data format"}

    today = datetime.date.today()

    if user_id in premium_users:
        try:
            end = datetime.datetime.strptime(premium_users[user_id]["end"], "%Y-%m-%d").date()
            if today <= end:
                return {"status": "active", "expires_at": premium_users[user_id]["end"]}
            else:
                return {"status": "expired", "expired_at": premium_users[user_id]["end"]}
        except Exception as e:
            return {"status": "error", "message": f"Invalid date format: {e}"}

    return {"status": "not_found"}
