
import os
import asyncio
import aiohttp
from telegram import Update, InputFile, InlineKeyboardMarkup, InlineKeyboardButton, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
import matplotlib.pyplot as plt
import io
import json
import joblib
import pandas as pd
from html import escape
from ta.trend import CCIIndicator, ADXIndicator, IchimokuIndicator
from ta.momentum import KAMAIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from openai import AsyncOpenAI
import requests
import numpy as np
from urllib.parse import urlparse, urlunparse
import hashlib
import logging
from dotenv import load_dotenv
from functools import wraps
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from binance.client import Client as BinanceClient

client = BinanceClient()

# Logging configuration
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN", "dummy_token")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "dummy_api_key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy_openai_key")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "506d562e0d4a434c97df2e3a51e4cd1c")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0"))

# OpenAI client
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Load models globally
model = None
tp = None
sl = None
try:
    model_path = os.path.abspath("model.pkl")
    model = joblib.load(model_path)
    logger.info(f"✅ Model yüklendi: {model_path}")
    try:
        logger.info(f"Modelin beklediği özellikler: {model.feature_names_in_}")
    except AttributeError:
        logger.warning("Model feature_names_in_ özniteliğine sahip değil")
    tp_model = joblib.load(os.path.abspath("tp_model.pkl"))
    sl_model = joblib.load(os.path.abspath("sl_model.pkl"))
    logger.info(f"✅ TP/SL modelleri yüklendi")
except Exception as e:
    logger.error(f"❌ Model yükleme başarısız: {e}")

# Load feature list
try:
    expected_features = joblib.load(os.path.abspath("features_list.pkl"))
    logger.info(f"✅ Özellik listesi yüklendi: {expected_features}")
except Exception as e:
    logger.error(f"❌ Özellik listesi yükleme başarısız: {e}")
    expected_features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'sma_20', 'atr']

# Accepted users
def load_accepted_users():
    if not os.path.exists("data/accepted_users.json"):
        return set()
    with open("data/accepted_users.json", "r") as f:
        try:
            return set(json.load(f))
        except json.JSONDecodeError:
            return set()

def save_accepted_users(users):
    os.makedirs("data", exist_ok=True)
    with open("data/accepted_users.json", "w") as f:
        json.dump(list(users), f)

async def check_user_accepted(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    user_info = load_json("data/user_info.json")
    return str(user_id) in user_info and user_info[str(user_id)].get("accepted") == True

# Admin users
def load_admin_users():
    if not os.path.exists("data/admins.json"):
        return set()
    with open("data/admins.json", "r") as f:
        try:
            return set(json.load(f))
        except json.JSONDecodeError:
            return set()

def save_admin_users(admin_set):
    with open("data/admins.json", "w") as f:
        json.dump(list(admin_set), f, indent=2)

def check_admin(user_id):
    return str(user_id) in load_admin_users() or user_id == OWNER_CHAT_ID

# Premium users
def load_premium_users():
    if not os.path.exists("data/premium_users.json"):
        return {}
    with open("data/premium_users.json", "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_premium_users(users):
    os.makedirs("data", exist_ok=True)
    with open("data/premium_users.json", "w") as f:
        json.dump(users, f)

def check_premium_status(user_id: int) -> bool:
    try:
        premium_users = load_premium_users()
        user_str = str(user_id)
        if user_str not in premium_users:
            return False
        end_date_str = premium_users[user_str].get("end")
        if not end_date_str:
            return False
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        return datetime.today().date() <= end_date
    except Exception as e:
        logger.error(f"Premium kontrol hatası: {e}")
        return False

# News and signal files
SENT_NEWS_FILE = "data/sent_news.json"
if os.path.exists(SENT_NEWS_FILE):
    with open(SENT_NEWS_FILE, "r") as f:
        try:
            sent_news_urls = set(json.load(f))
        except:
            sent_news_urls = set()
else:
    sent_news_urls = set()
SIGNAL_FILE = "data/signals.json"

def load_signals():
    if os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_signals(data):
    os.makedirs("data", exist_ok=True)
    with open(SIGNAL_FILE, "w") as f:
        json.dump(data, f, indent=2)

# Coin symbol map
symbol_to_id_map = {}

async def load_symbol_map():
    global symbol_to_id_map
    url = "https://api.binance.com/api/v3/exchangeInfo"
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY} if BINANCE_API_KEY != "dummy_api_key" else {}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                symbol_to_id_map.update({symbol["symbol"].replace("USDT", "").upper(): symbol["symbol"] for symbol in data["symbols"] if symbol["status"] == "TRADING" and symbol["symbol"].endswith("USDT")})
                additional_coins = {"BNB", "ADA", "XRP", "DOT", "LINK"}
                for coin in additional_coins:
                    if any(symbol["symbol"] == f"{coin}USDT" for symbol in data["symbols"] if symbol["status"] == "TRADING"):
                        symbol_to_id_map[coin.upper()] = f"{coin}USDT"
                logger.info(f"✅ Coin sembolleri yüklendi, ek coinler: {list(symbol_to_id_map.keys())}")
            else:
                logger.error(f"❌ Coin listesi alınamadı: durum={response.status}")
                symbol_to_id_map = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

# Helper functions
def normalize_url(raw_url):
    if not raw_url:
        return ""
    parsed = urlparse(raw_url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

def save_sent_urls():
    os.makedirs("data", exist_ok=True)
    with open(SENT_NEWS_FILE, "w") as f:
        json.dump(list(sent_news_urls), f)

def get_news_key(url, title):
    norm_url = normalize_url(url)
    key_base = f"{norm_url}|{title.strip().lower()}"
    return hashlib.md5(key_base.encode()).hexdigest()

async def fetch_price(symbol: str):
    full_symbol = symbol_to_id_map.get(symbol.upper(), f"{symbol.upper()}USDT")
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": full_symbol}
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY} if BINANCE_API_KEY != "dummy_api_key" else {}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return float(data["price"])
            logger.error(f"❌ {full_symbol} için fiyat alınamadı: durum={response.status}")
            return None

async def pr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Lütfen bir coin girin: /pr BTC")
        return
    symbol = context.args[0].upper()
    if symbol not in symbol_to_id_map:
        await update.message.reply_text(f"❌ {symbol} için eşleşme bulunamadı.")
        return
    price = await fetch_price(symbol)
    if price is not None:
        await update.message.reply_text(f"{symbol} fiyatı: ${price:.2f}")
    else:
        await update.message.reply_text(f"❌ {symbol} için fiyat alınamadı.")

async def fetch_ohlc_data(symbol: str, days=7):
    full_symbol = symbol_to_id_map.get(symbol.upper(), f"{symbol.upper()}USDT")
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": full_symbol, "interval": "1h", "limit": days * 24}
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY} if BINANCE_API_KEY != "dummy_api_key" else {}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if not data or not isinstance(data, list) or len(data) == 0:
                    logger.error(f"❌ {full_symbol} için boş veya geçersiz OHLC verisi")
                    return None, 0.0, 0.0
                try:
                    df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QuoteVolume", "Trades", "TakerBuyBase", "TakerBuyQuote", "Ignore"])
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
                    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                    change_24h = ((df["Close"].iloc[-1] - df["Close"].iloc[-24]) / df["Close"].iloc[-24]) * 100 if len(df) >= 24 else 0.0
                    change_7d = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
                    return df, change_24h, change_7d
                except Exception as e:
                    logger.error(f"❌ {full_symbol} için OHLC verisi işlenirken hata: {e}")
                    return None, 0.0, 0.0
            logger.error(f"❌ {full_symbol} için OHLC verisi alınamadı: durum={response.status}")
            return None, 0.0, 0.0

def prepare_features(df):
    """Modelin beklediği özellikleri hesaplar"""
    try:
        required_columns = ["Close", "High", "Low", "Volume"]
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Eksik gerekli sütunlar: {missing_cols}")

        logger.info(f"Giriş DataFrame sütunları: {list(df.columns)}")

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        features_df = pd.DataFrame(index=df.index)

        features_df["RSI"] = RSIIndicator(close=close).rsi()
        macd = MACD(close=close)
        features_df["MACD"] = macd.macd()
        features_df["Signal"] = macd.macd_signal()
        features_df["MA_5"] = SMAIndicator(close=close, window=5).sma_indicator()
        features_df["MA_20"] = SMAIndicator(close=close, window=20).sma_indicator()
        features_df["Volatility"] = close.pct_change().rolling(window=14).std()
        features_df["Momentum"] = close.diff()
        features_df["Price_Change"] = close.pct_change()
        features_df["Volume_Change"] = volume.pct_change()
        features_df["CCI"] = CCIIndicator(high=high, low=low, close=close).cci()
        features_df["ADX"] = ADXIndicator(high=high, low=low, close=close).adx()
        features_df["KAMA"] = KAMAIndicator(close=close).kama()
        features_df["OBV"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        features_df["CMF"] = ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume).chaikin_money_flow()
        features_df["ATR"] = AverageTrueRange(high=high, low=low, close=close).average_true_range()
        features_df["Force_Index"] = (close.diff() * volume).ewm(span=13, adjust=False).mean()
        stoch = StochasticOscillator(high=high, low=low, close=close)
        features_df["Stoch_K"] = stoch.stoch()
        features_df["Stoch_D"] = stoch.stoch_signal()
        ichimoku = IchimokuIndicator(high=high, low=low)
        features_df["Tenkan"] = ichimoku.ichimoku_conversion_line()
        features_df["Kijun"] = ichimoku.ichimoku_base_line()
        features_df["Senkou_A"] = ichimoku.ichimoku_a()
        features_df["Senkou_B"] = ichimoku.ichimoku_b()
        bb = BollingerBands(close=close)
        features_df["BB_Band_Width"] = bb.bollinger_wband()

        features_df.fillna(0, inplace=True)

        logger.info(f"Hesaplanan özellikler: {list(features_df.columns)}")

        return features_df
    except Exception as e:
        logger.error(f"❌ Özellik hesaplama hatası: {str(e)}")
        return None



async def generate_ai_comment(symbol: str) -> str:
    try:
        klines = client.get_klines(symbol=f"{symbol.upper()}USDT", interval=BinanceClient.KLINE_INTERVAL_15MINUTE, limit=50)
        if not klines or len(klines) < 50:
            raise ValueError(f"{symbol} için yetersiz veri: {len(klines)} satır alındı")

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])

        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        df['macd'] = MACD(close=df['close']).macd_diff()
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

        latest = df.dropna().iloc[-1]
        current_price = latest['close']

        with open(os.path.abspath("features_list.pkl"), "rb") as f:
            features_list = joblib.load(f)
        logger.info(f"Model için kullanılan özellikler: {features_list}")
        model_input = pd.DataFrame([latest[features_list]], columns=features_list)

        model = joblib.load(os.path.abspath("model.pkl"))
        prediction = model.predict(model_input)[0]

        # TP/SL tahminini mevcut fiyatla ölçeklendir (faktör: 10000)
        tp = tp_model.predict(model_input)[0] * current_price * 10000 if tp_model else None
        sl = sl_model.predict(model_input)[0] * current_price * 10000 if sl_model else None

        rsi = latest['rsi']
        macd = latest['macd']
        sma_5 = SMAIndicator(close=df['close'], window=5).sma_indicator().iloc[-1]
        sma_20 = latest['sma_20']
        rsi_comment = "RSI aşırı satım bölgesinde." if rsi < 30 else ("RSI aşırı alım bölgesinde." if rsi > 70 else "RSI nötr.")
        macd_comment = "MACD yükseliş eğiliminde." if macd > 0 else "MACD düşüş eğiliminde."
        trend_comment = "MA5, MA20’nin üzerinde (yükseliş)." if sma_5 > sma_20 else "MA5, MA20’nin altında (düşüş)."
        short_comment = f"{rsi_comment} {macd_comment} {trend_comment}"

        # Risk analizi
        risk = "⚠️ ✅ Düşük Risk" if (30 < rsi < 70 and abs(macd) > 0.05 and tp and sl and 
                                      0.02 < abs((tp - sl) / current_price) < 0.1 and 
                                      latest['atr'] / current_price < 0.05) else "⚠️ 🚨 Yüksek Risk"

        ai_signal = "📈 AL" if prediction == 1 else "📉 SAT"
        tp_text = f"🎯 TP: ${tp:.2f}" if tp and isinstance(tp, (int, float)) else "❌ TP tahmini başarısız."
        sl_text = f"🛑 SL: ${sl:.2f}" if sl and isinstance(sl, (int, float)) else "❌ SL tahmini başarısız."
        leverage = "💪 📈 Kaldıraç: 5x Uzun" if prediction == 1 else "💪 📉 Kaldıraç: 5x Kısa"

        _, change_24h, change_7d = await fetch_ohlc_data(symbol, days=7)
        change_24h_str = f"{change_24h:+.2f}%" if change_24h else "Yok"
        change_7d_str = f"{change_7d:+.2f}%" if change_7d else "Yok"

        return (
            f"📊 {symbol.upper()} (${current_price:.2f})\n"
            f"📈 24s: {change_24h_str} | 🗕️ 7g: {change_7d_str}\n\n"
            f"💡 {ai_signal}\n"
            f"📉 RSI: {rsi:.2f} | 🧺 MACD: {macd:.2f}\n"
            f"📈 MA(5): {sma_5:.2f} | MA(20): {sma_20:.2f}\n\n"
            f"{tp_text} | {sl_text}\n\n"
            f"{leverage} | {risk}\n\n"
            f"🧠 AI Yorumu: {short_comment}"
        )

    except Exception as e:
        logger.error(f"❌ Özellik hesaplama hatası: {e}")
        return f"⚠️ AI yorum hatası oluştu: {e}"
    
    


async def ai_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower()
    if not text.startswith("/ai"):
        return

    symbol = text.replace("/ai", "").strip().upper()
    user_id = update.effective_user.id

    logger.info(f"ai_comment: {user_id} kullanıcısından {symbol} için komut alındı")

    if not check_premium_status(user_id):
        await update.message.reply_text("❌ Bu komut (/ai) yalnızca Premium kullanıcılar için geçerlidir. /prem ile yükseltin.")
        return

    if not symbol or symbol not in symbol_to_id_map:
        logger.warning(f"ai_comment: {user_id} kullanıcısı için geçersiz sembol {symbol}")
        await update.message.reply_text(
            f"❌ Geçersiz coin sembolü. Lütfen Binance’ta işlem gören geçerli bir coin kullanın (ör. /ai BTC, /ai ETH, /ai SOL, /ai BNB, /ai ADA, /ai XRP, /ai DOT, /ai LINK)."
        )
        return

    await update.message.reply_text("💬 AI yorumu hazırlanıyor...")

    try:
        logger.info(f"ai_comment: {symbol} için fiyat alınıyor")
        price_data = await fetch_price(symbol)
        if price_data is None:
            logger.error(f"ai_comment: {symbol} için fiyat alınamadı")
            await update.message.reply_text(
                f"❌ {symbol} için fiyat verisi alınamadı. Lütfen internet bağlantınızı veya API anahtarınızı kontrol edin."
            )
            return

        logger.info(f"ai_comment: {symbol} için yorum oluşturuluyor")
        comment = await generate_ai_comment(symbol)

        logger.info(f"ai_comment: {symbol} için yorum oluşturuldu, geri bildirim düğmeleriyle sinyal gönderiliyor")
        await send_ai_signal(update, context, comment)

    except Exception as e:
        logger.error(f"ai_comment hatası: {str(e)}, sembol={symbol}, kullanıcı_id={user_id}")
        await update.message.reply_text(f"❌ İşlem sırasında hata oluştu: {str(e)}. Lütfen daha sonra tekrar deneyin.")

async def send_ai_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, signal_text: str):
    try:
        logger.info(f"send_ai_signal: {update.effective_chat.id} sohbeti için geri bildirim düğmeleriyle mesaj hazırlanıyor")
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("👍 Beğen", callback_data="feedback:like"), InlineKeyboardButton("👎 Beğenme", callback_data="feedback:dislike")]])
        message = await context.bot.send_message(chat_id=update.effective_chat.id, text=signal_text, reply_markup=keyboard)
        logger.info(f"send_ai_signal: {update.effective_chat.id} sohbeti için mesaj başarıyla gönderildi, mesaj_id {message.message_id}")
        signals = load_signals()
        signals.append({"message_id": message.message_id, "text": signal_text, "likes": [], "dislikes": []})
        save_signals(signals)
    except Exception as e:
        logger.error(f"send_ai_signal hatası: {e}")
        if update.message:
            await update.message.reply_text("❌ AI sinyali geri bildirim düğmeleriyle gönderilemedi. Lütfen tekrar deneyin.")

def get_portfolio(user_id):
    users = load_json("data/portfolios.json")
    return users.get(str(user_id), {})


def add_coin(user_id, symbol, amount, buy_price=None):
    users = load_json("data/portfolios.json")
    user_id_str = str(user_id)
    if user_id_str not in users:
        users[user_id_str] = {}
    users[user_id_str][symbol.lower()] = {"amount": amount, "buy_price": buy_price}
    save_json("data/portfolios.json", users)
    return True


def remove_coin(user_id, symbol):
    users = load_json("data/portfolios.json")
    user_id_str = str(user_id)
    symbol = symbol.lower()
    if user_id_str in users and symbol in users[user_id_str]:
        del users[user_id_str][symbol]
        if not users[user_id_str]:
            del users[user_id_str]
        save_json("data/portfolios.json", users)
        return True
    return False

def update_coin(user_id, symbol, amount):
    users = load_json("data/portfolios.json")
    user_id_str = str(user_id)
    symbol = symbol.lower()
    if user_id_str in users and symbol in users[user_id_str]:
        users[user_id_str][symbol]["amount"] = amount
        save_json("data/portfolios.json", users)
        return True
    return False

def clear_portfolio(user_id):
    users = load_json("data/portfolios.json")
    user_id_str = str(user_id)
    if user_id_str in users:
        del users[user_id_str]
        save_json("data/portfolios.json", users)
        return True
    return False

def get_all_alerts():
    alerts = load_json("data/alerts.json")
    return [alert for user_alerts in alerts.values() for alert in user_alerts]

def delete_alert(user_id, symbol):
    alerts = load_json("data/alerts.json")
    user_id_str = str(user_id)
    symbol = symbol.upper()
    if user_id_str in alerts:
        alerts[user_id_str] = [alert for alert in alerts[user_id_str] if alert["symbol"] != symbol]
        if not alerts[user_id_str]:
            del alerts[user_id_str]
        save_json("data/alerts.json", alerts)
        return True
    return False

def add_alert(user_id, symbol, target_price):
    alerts = load_json("data/alerts.json")
    user_id_str = str(user_id)
    if user_id_str not in alerts:
        alerts[user_id_str] = []
    alerts[user_id_str].append({"symbol": symbol.upper(), "target_price": target_price, "user_id": user_id})
    save_json("data/alerts.json", alerts)
    return True

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    save_user(user_id)
    update_user_metadata(user_id)

    message = update.message or (update.callback_query and update.callback_query.message)

    if not await check_user_accepted(update, context):
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Anladım", callback_data="accept_disclaimer")]])
        disclaimer_text = (
            "📢 Uyarı\n\n"
            "Coinspace Bot, kripto piyasasında bilinçli kararlar almanıza yardımcı olmak için piyasa içgörüleri ve yapay zeka destekli sinyaller sağlar.\n"
            "Bu sinyaller yalnızca bilgilendirme amaçlıdır ve finansal tavsiye niteliği taşımaz.\n\n"
            "Lütfen devam etmek için onaylayın."
        )
        if message:
            await message.reply_text(disclaimer_text, reply_markup=keyboard, parse_mode="MarkdownV2")
        return

    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("📖 Komutları Görüntüle (/help)", callback_data="help")]])
    msg = (
        "<b>👋 Coinspace Bot’a tekrar hoş geldiniz!</b>\n\n"
        f"<b>🚀 Kullanıcı ID’niz:</b> <code>{user_id}</code>\n\n"
        "<b>🚀 Günlük yapay zeka destekli işlem sinyalleri, fiyat uyarıları, portföy takibi ve canlı piyasa güncellemeleri alın.</b>\n\n"
        "<b>🔐 Premium’a Yükseltin:</b>\n"
        "• Sınırsız AI Kaldıraç Sinyalleri (Ücretsiz kullanıcılar günde sadece 2 sinyal alır)\n"
        "• Tam piyasa genel görünüm erişimi\n"
        "• Öncelikli destek ve erken özellik erişimi\n\n"
        "<b>💳 Abonelik Planları:</b>\n"
        "• 1 Ay: $29.99\n"
        "• 3 Ay: $69.99\n"
        "• 1 Yıl: $399.99\n\n"
        "<b>👉 Yükseltmek için bir plan seçin ve ödemeyi tamamlayın:</b>\n"
        "• <a href='https://nowpayments.io/payment/?iid=5260731771'>1 Aylık Ödeme</a>\n"
        "• <a href='https://nowpayments.io/payment/?iid=4400895826'>3 Aylık Ödeme</a>\n"
        "• <a href='https://nowpayments.io/payment/?iid=4501340550'>1 Yıllık Ödeme</a>\n\n"
        "✅ Ödeme sonrası, aboneliğinizi <b>/activate_premium</b> komutuyla etkinleştirin.\n\n"
        "👇 Mevcut komutları görmek için aşağıdaki düğmeye tıklayın:"
    )

    if message:
        await message.reply_text(msg, reply_markup=keyboard, parse_mode="HTML", disable_web_page_preview=True)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Güncelleme işlenirken hata: {context.error}")
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text("❌ Bir hata oluştu. Lütfen tekrar deneyin veya destekle iletişime geçin.")
        except Exception:
            pass

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"help_command: Başlıyor, kullanıcı: {update.effective_user.id}")
    if not await check_user_accepted(update, context):
        logger.warning(f"help_command: Kullanıcı {update.effective_user.id} şartları kabul etmedi.")
        if update.message:
            await update.message.reply_text("⚠️ Lütfen /start komutunu çalıştırın ve şartları kabul edin.")
        return

    if not update.message and not update.callback_query:
        logger.error(f"help_command hatası: Kullanıcı {update.effective_user.id} için geçerli güncelleme mesajı veya geri arama sorgusu yok")
        return

    msg = (
        "*🤖 Coinspace Komutları*\n\n"
        "📊 *Portföy*\n"
        "• `/add BTC 0\\.5 30000` \\- Coin ekle\n"
        "• `/upd BTC 1\\.0` \\- Coin güncelle\n"
        "• `/rm BTC` \\- Coin kaldır\n"
        "• `/clr` \\- Portföyü temizle\n"
        "• `/port` \\- Portföyü görüntüle\n"
        "• `/perf` \\- Performansı görüntüle\n"
        "• `/gr` \\- Grafik görüntüle\n\n"
        "💹 *Piyasa Araçları*\n"
        "• `/pr BTC` \\- Fiyat kontrol et\n"
        "• `/alert BTC 70000` \\- Uyarı ayarla\n"
        "• `/ai BTC` \\- AI sinyali \\(Premium\\)\n"
        "• `/bt BTC` \\- Geri test\n\n"
        "📰 *Haberler & Premium*\n"
        "• `/nw` \\- Son haberler\n"
        "• `/nmore` \\- Daha fazla haber\n"
        "• `/prem` \\- Premium’a yükselt\n"
    )

    try:
        if update.message:
            await update.message.reply_text(msg, parse_mode="MarkdownV2")
        elif update.callback_query:
            await update.callback_query.message.edit_text(msg, parse_mode="MarkdownV2")
        logger.info(f"help_command: Yanıt gönderildi, kullanıcı: {update.effective_user.id}")
    except Exception as e:
        logger.error(f"help_command hatası: {e}")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "help":
        await help_command(update, context)
    elif query.data == "accept_disclaimer":
        await accept_disclaimer(update, context)

async def port(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    holdings = get_portfolio(user_id)
    if not holdings:
        await update.message.reply_text("📭 Henüz coin eklenmedi. /add kullanın.")
        return
    symbols = [sym.upper() for sym in holdings.keys() if sym.upper() in symbol_to_id_map]
    total_value = 0
    msg = "📊 Portföy:\n"
    for symbol in symbols:
        price = await fetch_price(symbol)
        amount = holdings.get(symbol.lower(), {}).get("amount", 0)
        if price:
            value = price * amount
            total_value += value
            msg += f"• {symbol}: {amount} × ${price:.2f} = ${value:.2f}\n"
        else:
            msg += f"• {symbol}: Fiyat mevcut değil\n"
    msg += f"\n💰 Toplam Değer: ${total_value:.2f}"
    await update.message.reply_text(msg)

async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) not in [2, 3]:
        await update.message.reply_text("❌ Kullanım: /add <coin> <miktar> [alım_fiyatı] (ör. /add BTC 0.5 30000)")
        return
    symbol = context.args[0].upper()
    try:
        amount = float(context.args[1])
        buy_price = float(context.args[2]) if len(context.args) == 3 else None
    except ValueError:
        await update.message.reply_text("❌ Geçersiz miktar veya fiyat.")
        return
    user_id = update.effective_user.id
    add_coin(user_id, symbol, amount, buy_price)
    msg = f"✅ {amount} {symbol} portföye eklendi."
    if buy_price:
        msg += f" Alım fiyatı: ${buy_price}"
    await update.message.reply_text(msg)

async def rm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("❌ Kullanım: /rm <coin> (ör. /rm BTC)")
        return
    symbol = context.args[0].upper()
    user_id = update.effective_user.id
    success = remove_coin(user_id, symbol)
    if success:
        await update.message.reply_text(f"🗑️ {symbol} portföyden kaldırıldı.")
    else:
        await update.message.reply_text(f"⚠️ {symbol} portföyde bulunamadı.")

async def upd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.message.reply_text("❌ Kullanım: /upd <coin> <miktar> (ör. /upd BTC 1.0)")
        return
    symbol = context.args[0].upper()
    try:
        amount = float(context.args[1])
    except ValueError:
        await update.message.reply_text("❌ Geçersiz miktar.")
        return
    user_id = update.effective_user.id
    success = update_coin(user_id, symbol, amount)
    if success:
        await update.message.reply_text(f"🔗 {symbol} miktarı {amount} olarak güncellendi.")
    else:
        await update.message.reply_text(f"⚠️ {symbol} portföyde bulunamadı, önce /add ile ekleyin.")

async def clr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    success = clear_portfolio(user_id)
    if success:
        await update.message.reply_text("🧼 Portföy başarıyla temizlendi.")
    else:
        await update.message.reply_text("❗ Temizlenecek veri yok.")

async def gr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_premium_status(user_id):
        await update.message.reply_text("❌ Bu komut (/gr) yalnızca Premium kullanıcılar için geçerlidir. /prem ile yükseltin.")
        return
    holdings = get_portfolio(user_id)
    if not holdings:
        await update.message.reply_text("📭 Portföy boş. Önce /add ile coin ekleyin.")
        return
    symbols = [sym.upper() for sym in holdings.keys() if sym.upper() in symbol_to_id_map]
    labels, values = [], []
    for symbol in symbols:
        price = await fetch_price(symbol)
        amount = holdings.get(symbol.lower(), {}).get("amount", 0)
        if price:
            value = price * amount
            labels.append(symbol)
            values.append(value)
    if not values:
        await update.message.reply_text("⚠️ Fiyat verisi mevcut değil.")
        return
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    ax1.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    ax1.set_title("📈 Portföy Dağılımı")
    df = await fetch_ohlc_data("ETH")[0]
    if df is not None and not df.empty and isinstance(df, pd.DataFrame):
        closes = df["Close"].values
        rsi = [RSIIndicator(close=pd.Series(closes[:i+1])).rsi()[-1] for i in range(len(closes)) if i >= 14]
        macd, signal = zip(*[(MACD(close=pd.Series(closes[:i+1])).macd()[-1], MACD(close=pd.Series(closes[:i+1])).macd_signal()[-1]) for i in range(len(closes)) if i >= 26])
        timestamps = range(len(rsi))
        ax2.plot(timestamps, rsi[-len(closes)+14:], label="RSI", color="purple")
        ax2.axhline(y=70, color="orange", linestyle="--", label="Aşırı Alım (70)")
        ax2.axhline(y=30, color="orange", linestyle="--", label="Aşırı Satım (30)")
        ax2.set_title("ETH RSI")
        ax2.set_xlabel("Zaman (Saat)")
        ax2.set_ylabel("RSI Değeri")
        ax2.legend()
        ax3.plot(timestamps, macd[-len(closes)+26:], label="MACD", color="green")
        ax3.plot(timestamps, signal[-len(closes)+26:], label="Sinyal", color="red", linestyle="--")
        ax3.set_title("ETH MACD")
        ax3.set_xlabel("Zaman (Saat)")
        ax3.set_ylabel("MACD Değeri")
        ax3.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    await update.message.reply_photo(photo=InputFile(buf, filename="portfolio_graph.png"))

async def perf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    portfolio = get_portfolio(user_id)
    if not portfolio:
        await update.message.reply_text("📭 Portföy boş.")
        return
    symbols = [sym.upper() for sym in portfolio.keys() if sym.upper() in symbol_to_id_map]
    msg, total_pl = "📈 Portföy Performansı:\n", 0
    for symbol in symbols:
        current_price = await fetch_price(symbol)
        amount = portfolio.get(symbol.lower(), {}).get("amount", 0)
        buy_price = portfolio.get(symbol.lower(), {}).get("buy_price")
        if not current_price:
            msg += f"• {symbol}: Fiyat mevcut değil\n"
            continue
        if buy_price:
            current_value, cost_basis = current_price * amount, buy_price * amount
            pl = current_value - cost_basis
            total_pl += pl
            msg += f"• {symbol}: Alış ${buy_price:.2f} → Şu Anki ${current_price:.2f} | K/Z: ${pl:.2f}\n"
        else:
            msg += f"• {symbol}: Alış fiyatı bilinmiyor\n"
    msg += f"\n💼 Toplam K/Z: ${total_pl:.2f}"
    await update.message.reply_text(msg)

async def alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.message.reply_text("Kullanım: /alert <coin> <fiyat> (ör. /alert BTC 70000)")
        return
    symbol = context.args[0].upper()
    try:
        target_price = float(context.args[1])
    except ValueError:
        await update.message.reply_text("❌ Geçersiz fiyat.")
        return
    if symbol not in symbol_to_id_map:
        await update.message.reply_text("❌ Geçersiz coin sembolü.")
        return
    user_id = update.effective_user.id
    if not check_premium_status(user_id):
        await update.message.reply_text("❌ Bu komut (/alert) yalnızca Premium kullanıcılar için geçerlidir. /prem ile yükseltin.")
        return
    add_alert(user_id, symbol, target_price)
    await update.message.reply_text(f"🔔 {symbol} için ${target_price}’da uyarı ayarlandı.")

async def check_alerts(app):
    while True:
        alerts = get_all_alerts() or []
        if not alerts:
            await asyncio.sleep(300)
            continue
        valid_alerts = [alert for alert in alerts if all(k in alert for k in ("symbol", "target_price", "user_id"))]
        symbol_map = {}
        for alert in valid_alerts:
            symbol = alert["symbol"].lower()
            if symbol not in symbol_map:
                symbol_map[symbol] = []
            symbol_map[symbol].append(alert)
        for symbol, alerts_list in symbol_map.items():
            price = await fetch_price(symbol.upper())
            if price is None:
                continue
            for alert in alerts_list:
                if price >= alert["target_price"]:
                    try:
                        await app.bot.send_message(
                            chat_id=alert["user_id"],
                            text=f"📢 *{symbol.upper()}* ${alert['target_price']}’a ulaştı!\nŞu anki fiyat: ${price:.2f}",
                            parse_mode="Markdown"
                        )
                        delete_alert(alert["user_id"], symbol.upper())
                    except Exception as e:
                        logger.error(f"❌ Bildirim başarısız: {e}")
        await asyncio.sleep(300)



async def fetch_newsapi_news():
    url = f"https://newsapi.org/v2/top-headlines?category=business&q=crypto&apiKey={NEWS_API_KEY}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            logger.info(f"🌐 NewsAPI durumu: {response.status}")
            if response.status == 200:
                return await response.json()
            return None

async def summarize_news(title, description):
    prompt = f"Aşağıdaki haber için kısa bir özet yaz:\n\nBaşlık: {title}\nAçıklama: {description}\n\nYatırımcılar için kısa bir özet hazırla."
    try:
        response = await openai_client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}], max_tokens=100, temperature=0.7)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ Özet oluşturma hatası: {e}")
        return "⚠️ Özet oluşturma başarısız."

async def nw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("🚀 /news komutu tetiklendi")
    news_data = await fetch_newsapi_news()
    if not news_data or "articles" not in news_data:
        await update.message.reply_text("❌ Haber verisi mevcut değil.")
        return
    sent_count = 0
    for article in news_data["articles"][:5]:
        url = article.get("url")
        title = article.get("title", "Başlık Yok")
        description = article.get("description", "Açıklama Yok")
        norm_url = normalize_url(url)
        if norm_url and norm_url not in sent_news_urls:
            summary = await summarize_news(title, description)
            text = f"📰 <b>{escape(title)}</b>\n{escape(summary)}\n<a href=\"{url}\">🔗 Daha Fazla Oku</a>"
            try:
                await update.message.reply_text(text, parse_mode="HTML")
                sent_news_urls.add(norm_url)
                save_sent_urls()
                sent_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Gönderme hatası: {e}")
    if sent_count == 0:
        await update.message.reply_text("⚠️ Yeni haber yok.")

async def rmore(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🧭 Haber linklerini /nw komutuyla görüntüleyin.")

async def bt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Kullanım: /bt <coin> (ör. /bt BTC)")
        return
    symbol = context.args[0].upper()
    if symbol not in symbol_to_id_map:
        await update.message.reply_text("❌ Coin bulunamadı.")
        return
    user_id = update.effective_user.id
    if not check_premium_status(user_id):
        await update.message.reply_text("❌ Bu komut (/bt) yalnızca Premium kullanıcılar için geçerlidir. /prem ile yükseltin.")
        return
    df, _, _ = await fetch_ohlc_data(symbol, days=30)
    if df is None or df.empty:
        await update.message.reply_text("❌ Veri mevcut değil.")
        return
    df["RSI"] = RSIIndicator(df["Close"]).rsi()
    df["MA"] = SMAIndicator(df["Close"], window=14).sma_indicator()
    buy_points, sell_points, position, entry_price, pnl = [], [], None, 0, 0
    for i in range(1, len(df)):
        rsi, price, ma = df["RSI"].iloc[i], df["Close"].iloc[i], df["MA"].iloc[i]
        if rsi < 30 and price > ma and not position:
            entry_price, position = price, "LONG"
            buy_points.append((df.index[i], price))
        elif rsi > 70 and position == "LONG":
            pnl += price - entry_price
            sell_points.append((df.index[i], price))
            position = None
    msg = f"📈 {symbol} RSI + MA Geri Test Sonucu (30 gün):\n✅ Alım sayısı: {len(buy_points)}\n💰 Toplam Kâr: ${pnl:.2f}"
    await update.message.reply_text(msg)

async def prem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🟢 1 Ay – $29.99", url="https://nowpayments.io/payment/?iid=5260731771")],
        [InlineKeyboardButton("🔵 3 Ay – $69.99", url="https://nowpayments.io/payment/?iid=4400895826")],
        [InlineKeyboardButton("🟣 1 Yıl – $399.99", url="https://nowpayments.io/payment/?iid=4501340550")],
    ])
    msg = (
        "👑 <b>Coinspace Premium Planları!</b>\n\n"
        "⚡️ <b>Faydalar:</b>\n"
        "• Sınırsız AI Kaldıraç Sinyalleri (Ücretsiz kullanıcılar günde sadece 2 sinyal alır)\n"
        "• Tam piyasa genel görünüm erişimi\n"
        "• Öncelikli destek & erken özellik erişimi\n\n"
        "💳 <b>Planlar:</b>\n"
        "1 Ay: $29.99\n"
        "3 Ay: $69.99\n"
        "1 Yıl: $399.99\n\n"
        "👉 <b>Yükseltmek için bir plan seçin ve ödemeyi tamamlayın:</b>\n"
        "• <a href='https://nowpayments.io/payment/?iid=5260731771'>1 Aylık Ödeme</a>\n"
        "• <a href='https://nowpayments.io/payment/?iid=4400895826'>3 Aylık Ödeme</a>\n"
        "• <a href='https://nowpayments.io/payment/?iid=4501340550'>1 Yıllık Ödeme</a>\n\n"
        "✅ Ödeme sonrası, aboneliğinizi /activate_premium komutuyla etkinleştirin."
    )
    await update.message.reply_text(msg, parse_mode="HTML", reply_markup=keyboard, disable_web_page_preview=True)

async def activate_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("❌ Kullanım: /activate_premium <ödeme_id> (ör. /activate_premium 5260731771)")
        return
    payment_id = context.args[0]
    user_id = str(update.effective_user.id)
    valid_payments = {
        "5260731771": 30,  # 1 ay
        "4400895826": 90,  # 3 ay
        "4501340550": 365  # 1 yıl
    }
    if payment_id not in valid_payments:
        await update.message.reply_text("❌ Geçersiz ödeme ID’si. Lütfen bot sahibiyle iletişime geçin.")
        return
    premium_users = load_premium_users()
    today = datetime.today().date()
    end_date = today + timedelta(days=valid_payments[payment_id])
    premium_users[user_id] = {"start": str(today), "end": str(end_date)}
    save_premium_users(premium_users)
    await update.message.reply_text(f"✅ Premium aboneliğiniz {end_date} tarihine kadar etkinleştirildi!")

async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.message.reply_text("❌ Yönetici yetkiniz yok.")
        return
    msg = (
        "<b>🔧 Yönetici Paneli</b>\n\n"
        "• <code>/broadcast</code> – Tüm kullanıcılara mesaj gönder\n"
        "• <code>/users</code> – Toplam kayıtlı kullanıcıları göster\n"
        "• <code>/make_admin [user_id]</code> – Yönetici yetkisi ver\n"
        "• <code>/remove_admin [user_id]</code> – Yönetici yetkisini kaldır\n"
        "• <code>/make_premium [user_id]</code> – Premium statüsü ver\n"
        "• <code>/remove_premium [user_id]</code> – Premium statüsünü kaldır\n"
        "• <code>/admin_list</code> – Yönetici kullanıcıları listele\n"
        "• <code>/premium_list</code> – Premium kullanıcıları listele"
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def make_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.message.reply_text("❌ Yönetici yetkiniz yok.")
        return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("❌ Kullanım: /make_admin <user_id>")
        return
    target_user_id = str(context.args[0])
    admin_users = load_admin_users()
    if target_user_id not in admin_users:
        admin_users.add(target_user_id)
        save_admin_users(admin_users)
        await update.message.reply_text(f"✅ Kullanıcı {target_user_id} yönetici yapıldı.")
    else:
        await update.message.reply_text(f"⚠️ Kullanıcı {target_user_id} zaten yönetici.")

async def remove_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.message.reply_text("❌ Yönetici yetkiniz yok.")
        return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("❌ Kullanım: /remove_admin <user_id>")
        return
    target_user_id = str(context.args[0])
    admin_users = load_admin_users()
    if target_user_id in admin_users:
        admin_users.remove(target_user_id)
        save_admin_users(admin_users)
        await update.message.reply_text(f"✅ Kullanıcı {target_user_id} için yönetici statüsü kaldırıldı.")
    else:
        await update.message.reply_text(f"⚠️ Kullanıcı {target_user_id} yönetici değil.")

async def make_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.message.reply_text("❌ Yönetici yetkiniz yok.")
        return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("❌ Kullanım: /make_premium <user_id>")
        return
    target_user_id = str(context.args[0])
    premium_users = load_premium_users()
    if target_user_id in premium_users:
        await update.message.reply_text(f"⚠️ Kullanıcı {target_user_id} zaten Premium kullanıcı.")
        return
    today = datetime.today().date()
    end_date = today + timedelta(days=30)
    premium_users[target_user_id] = {"start": str(today), "end": str(end_date)}
    save_premium_users(premium_users)
    await update.message.reply_text(f"✅ Kullanıcı {target_user_id}’a {end_date} tarihine kadar Premium erişimi verildi.")

async def remove_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.message.reply_text("❌ Yönetici yetkiniz yok.")
        return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("❌ Kullanım: /remove_premium <user_id>")
        return
    target_user_id = str(context.args[0])
    premium_users = load_premium_users()
    if target_user_id in premium_users:
        del premium_users[target_user_id]
        save_premium_users(premium_users)
        await update.message.reply_text(f"✅ Kullanıcı {target_user_id} için Premium statüsü kaldırıldı.")
    else:
        await update.message.reply_text(f"⚠️ Kullanıcı {target_user_id} Premium kullanıcı değil.")

def admin_required(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if not check_admin(user_id):
            await update.message.reply_text("❌ Yönetici yetkiniz yok.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapper

async def users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        users_data = load_json("data/users.json")
        total = len(users_data)
        premium = sum(1 for u in users_data.values() if check_premium_status(int(u)))
        admins = len(load_admin_users())
        msg = (
            f"👥 <b>Toplam Kullanıcı:</b> {total}\n"
            f"💎 <b>Premium Kullanıcılar:</b> {premium}\n"
            f"🛡️ <b>Yöneticiler:</b> {admins}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")
    except Exception as e:
        logger.error(f"/users hatası: {e}")
        await update.message.reply_text("❌ Kullanıcı verileri yüklenemedi.")

def load_json(file_path):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(file_path, data):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

async def user_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Lütfen bir kullanıcı ID’si sağlayın. Kullanım: /user_info <user_id>")
        return
    user_id = context.args[0]
    users = load_json("data/users.json")
    if user_id not in users:
        await update.message.reply_text("❌ Kullanıcı bulunamadı.")
        return
    user_data = users[user_id]
    premium = "✅ Evet" if check_premium_status(int(user_id)) else "❌ Hayır"
    admin = "✅ Evet" if check_admin(int(user_id)) else "❌ Hayır"
    created_at = user_data.get("created_at", "Yok")
    last_active = user_data.get("last_active", "Yok")
    msg = (
        f"👤 <b>Kullanıcı ID:</b> <code>{user_id}</code>\n"
        f"💎 <b>Premium:</b> {premium}\n"
        f"🛠 <b>Yönetici:</b> {admin}\n"
        f"📆 <b>Katılım:</b> {created_at}\n"
        f"🕒 <b>Son Aktif:</b> {last_active}"
    )
    await update.message.reply_text(msg, parse_mode="HTML")

def save_user(user_id: int):
    users = load_json("data/users.json")
    if str(user_id) not in users:
        users[str(user_id)] = {}
        save_json("data/users.json", users)

def update_user_metadata(user_id):
    users = load_json("data/users.json")
    now = datetime.utcnow().isoformat()
    if str(user_id) not in users:
        users[str(user_id)] = {
            "premium": False,
            "admin": False,
            "created_at": now
        }
    users[str(user_id)]["last_active"] = now
    save_json("data/users.json", users)

bot_instance = Bot(token=TOKEN)
async def notify_user_if_expired(user_id: int):
    try:
        await bot_instance.send_message(
            chat_id=user_id,
            text="⚠️ Premium aboneliğiniz sona erdi. Yenilemek için /premium kullanın."
        )
    except Exception as e:
        logger.error(f"Kullanıcı {user_id} bildirimi sırasında hata: {e}")

async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("📢 Lütfen yayınlanacak bir mesaj sağlayın:\n\nKullanım:\n`/broadcast Mesajınız burada`", parse_mode="Markdown")
        return
    message_text = "📢 Yayın:\n" + " ".join(context.args)
    try:
        user_data = load_json("data/users.json")
    except FileNotFoundError:
        await update.message.reply_text("⚠️ Kullanıcı bulunamadı.")
        return
    count = 0
    for user_id in user_data.keys():
        try:
            await context.bot.send_message(chat_id=int(user_id), text=message_text)
            count += 1
        except Exception as e:
            logger.warning(f"{user_id}’e mesaj gönderilemedi: {e}")
    await update.message.reply_text(f"✅ Yayın {count} kullanıcıya gönderildi.")

async def accept_disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_info = load_json("data/user_info.json")
    user_info[str(user_id)] = {"accepted": True}
    save_json("data/user_info.json", user_info)
    accepted_users = load_json("data/accepted_users.json")
    if not isinstance(accepted_users, list):
        accepted_users = []
    if user_id not in accepted_users:
        accepted_users.append(user_id)
        save_json("data/accepted_users.json", accepted_users)
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("✅ Şartlar kabul edildi. Hoş geldiniz!")
    await start(update, context)

async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    callback_data = query.data
    logger.debug(f"[DEBUG] Geri bildirim alındı: {callback_data}, kullanıcı: {query.from_user.id}")
    if not callback_data.startswith("feedback:"):
        return
    feedback = callback_data.split(":")[1]
    message_id = query.message.message_id
    user_id = query.from_user.id
    signals = load_signals()
    found = False
    for s in signals:
        if s.get("message_id") == message_id:
            found = True
            if user_id in s.get("likes", []) or user_id in s.get("dislikes", []):
                logger.warning(f"feedback_handler: Kullanıcı {user_id} zaten mesaj_id {message_id} için geri bildirim verdi")
                await query.answer("❌ Bu mesaj için zaten geri bildirim verdiniz.", show_alert=True)
                return
            if feedback == "like":
                s.setdefault("likes", []).append(user_id)
                logger.info(f"feedback_handler: Kullanıcı {user_id} mesaj_id {message_id}’i beğendi")
            elif feedback == "dislike":
                s.setdefault("dislikes", []).append(user_id)
                logger.info(f"feedback_handler: Kullanıcı {user_id} mesaj_id {message_id}’i beğenmedi")
            break
    if not found:
        signals.append({"message_id": message_id, "text": query.message.text, "likes": [user_id] if feedback == "like" else [], "dislikes": [user_id] if feedback == "dislike" else []})
        logger.info(f"feedback_handler: Mesaj_id {message_id} için yeni sinyal girişi oluşturuldu, geri bildirim: {feedback}")
    save_signals(signals)
    await query.edit_message_reply_markup(reply_markup=None)
    await update.message.reply_text("✅ Geri bildiriminiz için teşekkürler!")

async def check_and_send_news(app):
    while True:
        news_data = await fetch_newsapi_news()
        if news_data and "articles" in news_data:
            for article in news_data["articles"]:
                url, title, description = article.get("url"), article.get("title", "Başlık Yok"), article.get("description", "Açıklama Yok")
                news_key = get_news_key(url, title)
                if news_key not in sent_news_urls:
                    summary = await summarize_news(title, description)
                    text = f"📰 <b>{escape(title)}</b>\n{escape(summary)}\n<a href=\"{url}\">🔗 Daha Fazla Oku</a>"
                    try:
                        await app.bot.send_message(chat_id=OWNER_CHAT_ID, text=text, parse_mode="HTML", disable_web_page_preview=True)
                        sent_news_urls.add(news_key)
                        save_sent_urls()
                    except Exception as e:
                        logger.error(f"⚠️ Haber gönderimi başarısız: {e}")
        await asyncio.sleep(1800)

async def premium_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.message.reply_text("❌ Yönetici yetkiniz yok.")
        return
    premium_users = load_premium_users()
    if not premium_users:
        await update.message.reply_text("❌ Premium kullanıcı bulunamadı.")
        return
    msg = "<b>💎 Premium Kullanıcı Listesi</b>\n\n"
    for uid, info in premium_users.items():
        msg += f"• <code>{uid}</code> – Geçerlilik: <b>{info.get('end', 'Yok')}</b>\n"
    await update.message.reply_text(msg, parse_mode="HTML")

async def admin_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    admins = load_admin_users()
    if str(user_id) not in admins:
        await update.message.reply_text("❌ Yönetici yetkiniz yok.")
        return
    if not admins:
        await update.message.reply_text("⚠️ Yönetici bulunamadı.")
        return
    msg = "<b>👮 Yönetici Kullanıcılar:</b>\n"
    for admin_id in admins:
        try:
            user = await context.bot.get_chat(int(admin_id))
            name = user.full_name or "Bilinmiyor"
            msg += f"• {name} (<code>{admin_id}</code>)\n"
        except Exception:
            msg += f"• Bilinmiyor (<code>{admin_id}</code>)\n"
    await update.message.reply_text(msg, parse_mode="HTML")

async def check_premium_expiry(bot: Bot):
    premium_users = load_premium_users()
    today = datetime.today().date()
    for user_id, info in premium_users.copy().items():
        end_date_str = info.get("end")
        if not end_date_str:
            continue
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        if today > end_date:
            await notify_user_if_expired(int(user_id))
            del premium_users[user_id]
    save_premium_users(premium_users)

async def background_tasks(bot: Bot):
    logger.info("🔄 Arka plan görevleri başladı.")
    while True:
        await check_premium_expiry(bot)
        await asyncio.sleep(3600)

async def run_bot():
    logger.info("🚀 Bot başlıyor...")
    app = ApplicationBuilder().token(TOKEN).build()
    logger.info("✅ Telegram bot uygulaması oluşturuldu.")
    await load_symbol_map()
    logger.info("✅ Coin sembolleri yüklendi.")
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("pr", pr))
    app.add_handler(CommandHandler("port", port))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("rm", rm))
    app.add_handler(CommandHandler("upd", upd))
    app.add_handler(CommandHandler("clr", clr))
    app.add_handler(CommandHandler("alert", alert))
    app.add_handler(CommandHandler("gr", gr))
    app.add_handler(CommandHandler("perf", perf))
    app.add_handler(CommandHandler("nw", nw))
    app.add_handler(CommandHandler("rmore", rmore))
    app.add_handler(CommandHandler("bt", bt))
    app.add_handler(CommandHandler("prem", prem))
    app.add_handler(CommandHandler("activate_premium", activate_premium))
    app.add_handler(CommandHandler("admin_panel", admin_panel))
    app.add_handler(CommandHandler("broadcast", broadcast))
    app.add_handler(CommandHandler("user_info", user_info))
    app.add_handler(CommandHandler("make_admin", make_admin))
    app.add_handler(CommandHandler("remove_admin", remove_admin))
    app.add_handler(CommandHandler("admin_list", admin_list))
    app.add_handler(CommandHandler("make_premium", make_premium))
    app.add_handler(CommandHandler("remove_premium", remove_premium))
    app.add_handler(CommandHandler("users", users))
    app.add_handler(CommandHandler("premium_list", premium_list))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(CallbackQueryHandler(feedback_handler))
    app.add_handler(CommandHandler("ai", ai_comment))
    app.add_error_handler(error_handler)
    asyncio.create_task(check_alerts(app))
    asyncio.create_task(check_and_send_news(app))
    asyncio.create_task(background_tasks(app.bot))
    logger.info("🔄 Arka plan görevleri başladı.")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    logger.info("✅ Bot başladı.")

if __name__ == "__main__":
    asyncio.run(run_bot())
