from fastapi import FastAPI
import asyncio
from bot import run_bot  # bot.py içindeki async run_bot()

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_bot())

@app.get("/api")
def root():
    return {"status": "Coinspace API running"}
