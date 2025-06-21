# main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
from bot import run_bot

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(run_bot())
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "Coinspace bot is running"}
