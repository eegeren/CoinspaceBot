from fastapi import FastAPI
import asyncio
from bot import run_bot, generate_ai_comment, fetch_price

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_bot())

@app.get("/api")
def root():
    return {"status": "Coinspace API running"}

@app.get("/api/analysis/{symbol}")
async def get_analysis(symbol: str):
    try:
        price_data = await fetch_price(symbol)
        if price_data is None:
            return {"error": f"Failed to fetch price for {symbol}"}
        coin_data = {"symbol": f"{symbol.upper()}USDT", "price": price_data}
        analysis = await generate_ai_comment(coin_data)
        return {"analysis": analysis}
    except Exception as e:
        return {"error": str(e)}