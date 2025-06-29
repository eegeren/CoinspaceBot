from fastapi import FastAPI
import asyncio
import os
from dotenv import load_dotenv
from bot import run_bot, generate_ai_comment, fetch_price
from api.payment_api import router as payment_router
from api.premium_checker import check_and_notify_expired_premium
from api.train_api import router as train_router

# Çevre değişkenlerini yükle ve kontrol et
load_dotenv()
required_env_vars = ["BOT_TOKEN", "BINANCE_API_KEY", "OPENAI_API_KEY", "NEWS_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"❌ {var} çevre değişkeni eksik!")

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_bot())
    asyncio.create_task(check_and_notify_expired_premium())

@app.get("/api")
def root():
    return {"status": "Coinspace API running"}

@app.get("/api/analysis/{symbol}")
async def get_analysis(symbol: str):
    try:
        price_data = await fetch_price(symbol)
        if price_data is None:
            return {"error": f"{symbol} için fiyat verisi alınamadı"}
        coin_data = {"symbol": f"{symbol.upper()}USDT", "price": price_data}
        analysis = await generate_ai_comment(coin_data)
        return {"analysis": analysis}
    except Exception as e:
        return {"error": str(e)}

# Eğitim ve ödeme rotaları
app.include_router(train_router, prefix="/api")
app.include_router(payment_router, prefix="/api")