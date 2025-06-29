from fastapi import APIRouter
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
import joblib
import os

router = APIRouter()

@router.get("/leverage-signal")
def ai_leverage_signal():
    try:
        if not os.path.exists("model.joblib"):
            raise FileNotFoundError("❌ model.joblib dosyası bulunamadı!")
        model = joblib.load("model.joblib")
        df = yf.download("BTC-USD", interval="15m", period="1d")
        if df.empty:
            raise ValueError("❌ Veri alınamadı!")
        last_close = df["Close"].iloc[-1]
        rsi = RSIIndicator(close=df["Close"]).rsi().iloc[-1]
        macd = MACD(close=df["Close"]).macd_diff().iloc[-1]
        X = [[rsi, macd]]
        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        direction = "LONG" if prediction == 1 else "SHORT"
        entry = round(last_close, 2)
        target = round(entry * (1.015 if direction == "LONG" else 0.985), 2)
        stop = round(entry * (0.985 if direction == "LONG" else 1.015), 2)
        return {
            "pair": "BTC/USDT",
            "direction": direction,
            "entry": entry,
            "target": target,
            "stop": stop,
            "confidence": int(prob * 100)
        }
    except Exception as e:
        return {"error": str(e)}