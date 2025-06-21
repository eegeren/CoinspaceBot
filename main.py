from fastapi import FastAPI
import asyncio
from bot import run_bot

app = FastAPI() 

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_bot())

@app.get("/")
def read_root():
    return {"status": "Coinspace bot is running"}
