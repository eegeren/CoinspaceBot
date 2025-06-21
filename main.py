from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
from bot import run_bot

# Lifespan context for startup and cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 FastAPI lifespan başlatılıyor...")
    asyncio.create_task(run_bot())  # Telegram botu başlat
    yield  # Uygulama çalıştığı sürece burada bekler

# FastAPI uygulaması
app = FastAPI(lifespan=lifespan)

# Basit durum kontrol endpoint’i
@app.get("/")
def read_root():
    return {"status": "✅ Coinspace bot is running"}
