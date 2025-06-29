
import os
import asyncio
import aiohttp
from telegram import Update, InputFile, InlineKeyboardMarkup, InlineKeyboardButton, Bot, ChatMemberUpdated
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler, ChatMemberHandler
import matplotlib.pyplot as plt
import io
import json
import joblib
import logging
from dotenv import load_dotenv
import pandas as pd
from html import escape
from ta.trend import CCIIndicator, ADXIndicator, IchimokuIndicator
from ta.momentum import KAMAIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from openai import AsyncOpenAI
import numpy as np
from urllib.parse import urlparse, urlunparse
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from binance.client import Client as BinanceClient

# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID", "0")

# Validate environment variables
required_env_vars = {
    "BOT_TOKEN": TOKEN,
    "BINANCE_API_KEY": BINANCE_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "NEWS_API_KEY": NEWS_API_KEY,
}
for var_name, var_value in required_env_vars.items():
    if not var_value:
        raise ValueError(f"❌ {var_name} environment variable is missing!")

# Initialize clients
client = BinanceClient(BINANCE_API_KEY, os.getenv("BINANCE_API_SECRET"))
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Load machine learning models and features
model = None
tp_model = None
sl_model = None
expected_features = None

# Load machine learning models and expected features from files
def load_models_and_features():
    global model, tp_model, sl_model, expected_features
    try:
        model = joblib.load("model.pkl")
        tp_model = joblib.load("tp_model.pkl")
        sl_model = joblib.load("sl_model.pkl")
        expected_features = joblib.load("features_list.pkl")
        logger.info("✅ Models and features loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models or features: {e}")
        model = tp_model = sl_model = None
        expected_features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'sma_20', 'atr']

load_models_and_features()

# User management functions

# Load JSON data from a file, return default if file doesn't exist
def load_json(filename, default=None):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            # alerts.json düzeltmesi
            if filename == "data/alerts.json" and isinstance(data, list):
                logger.warning("⚠️ alerts.json list formatında, dict'e dönüştürülüyor...")
                fixed_data = {}
                for alert in data:
                    uid = str(alert["user_id"])
                    fixed_data.setdefault(uid, []).append({
                        "symbol": alert["symbol"],
                        "target_price": alert["target_price"]
                    })
                save_json("data/alerts.json", fixed_data)
                return fixed_data
            return data
    except Exception as e:
        logger.error(f"❌ JSON yükleme hatası ({filename}): {e}")
        return default if default is not None else {}

# Save data to a JSON file
def save_json(file_path, data):
    try:
        os.makedirs("data", exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"❌ Error writing to {file_path}: {e}")

# Load list of users who accepted the disclaimer
def load_accepted_users():
    return set(load_json("data/accepted_users.json", []))

# Save list of accepted users to file
def save_accepted_users(users):
    save_json("data/accepted_users.json", list(users))

# Check if a user has accepted the disclaimer (currently disabled)
async def check_user_accepted(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    return True  # Feragatname kontrolü şimdilik devre dışı

# Load list of admin users
def load_admin_users():
    return set(load_json("data/admins.json", []))

# Save list of admin users to file
def save_admin_users(admin_set):
    save_json("data/admins.json", list(admin_set))

# Check if a user is an admin or the owner
def check_admin(user_id):
    return str(user_id) in load_admin_users() or str(user_id) == OWNER_CHAT_ID

# Save a new user to the user database
def save_user(user_id: int):
    users = load_json("data/users.json")
    if str(user_id) not in users:
        users[str(user_id)] = {}
        save_json("data/users.json", users)

# Update a user's metadata (e.g., last active time)
def update_user_metadata(user_id):
    users = load_json("data/users.json")
    now = datetime.utcnow().isoformat()
    if str(user_id) not in users:
        users[str(user_id)] = {"created_at": now}
    users[str(user_id)]["last_active"] = now
    save_json("data/users.json", users)

# News and signal files
SENT_NEWS_FILE = "data/sent_news.json"
sent_news_urls = set(load_json(SENT_NEWS_FILE, []))
SIGNAL_FILE = "data/signals.json"

# Load trading signals from file
def load_signals():
    return load_json(SIGNAL_FILE, [])

# Save trading signals to file
def save_signals(data):
    save_json(SIGNAL_FILE, data)

# Coin symbol mapping
symbol_to_id_map = {}

# Fetch and map trading symbols from Binance API
async def load_symbol_map():
    global symbol_to_id_map
    url = "https://api.binance.com/api/v3/exchangeInfo"
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    symbol_to_id_map.update({
                        symbol["symbol"].replace("USDT", "").upper(): symbol["symbol"]
                        for symbol in data["symbols"]
                        if symbol["status"] == "TRADING" and symbol["symbol"].endswith("USDT")
                    })
                    logger.info(f"✅ Coin symbols loaded: {list(symbol_to_id_map.keys())[:5]}...")
                else:
                    logger.error(f"❌ Failed to fetch coin list: status={response.status}")
                    symbol_to_id_map = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
    except Exception as e:
        logger.error(f"❌ Symbol map load error: {e}")
        symbol_to_id_map = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}

# Utility functions

# Normalize a URL by removing query parameters and fragments
def normalize_url(raw_url):
    if not raw_url:
        return ""
    parsed = urlparse(raw_url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

# Save sent news URLs to file
def save_sent_urls():
    save_json(SENT_NEWS_FILE, list(sent_news_urls))

# Fetch the current price of a cryptocurrency from Binance
async def fetch_price(symbol):
    if symbol is None:
        logger.warning("⚠️ fetch_price: symbol is None, fiyat alınamıyor.")
        return None
    full_symbol = symbol_to_id_map.get(symbol.upper(), f"{symbol.upper()}USDT")
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": full_symbol}
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data["price"])
                logger.error(f"❌ Failed to fetch price for {full_symbol}: status={response.status}")
                return None
    except Exception as e:
        logger.error(f"❌ Price fetch error for {full_symbol}: {e}")
        return None

# Fetch OHLC (Open, High, Low, Close) data for a cryptocurrency
async def fetch_ohlc_data(symbol: str, days=7):
    if not symbol:
        logger.warning("⚠️ fetch_ohlc_data: symbol is None.")
        return None, 0.0, 0.0
    full_symbol = symbol_to_id_map.get(symbol.upper(), f"{symbol.upper()}USDT")
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": full_symbol, "interval": "1h", "limit": days * 24}
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if not data or len(data) == 0:
                        logger.error(f"❌ Empty OHLC data for {full_symbol}")
                        return None, 0.0, 0.0
                    df = pd.DataFrame(data, columns=[
                        "Timestamp", "Open", "High", "Low", "Close", "Volume",
                        "CloseTime", "QuoteVolume", "Trades", "TakerBuyBase",
                        "TakerBuyQuote", "Ignore"
                    ])
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
                    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                    change_24h = ((df["Close"].iloc[-1] - df["Close"].iloc[-24]) / df["Close"].iloc[-24]) * 100 if len(df) >= 24 else 0.0
                    change_7d = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
                    return df, change_24h, change_7d
                logger.error(f"❌ Failed to fetch OHLC for {full_symbol}: status={response.status}")
                return None, 0.0, 0.0
    except Exception as e:
        logger.error(f"❌ OHLC fetch error for {full_symbol}: {e}")
        return None, 0.0, 0.0

# Prepare technical indicators for machine learning model input
def prepare_features(df):
    try:
        required_columns = ["Close", "High", "Low", "Volume"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {[col for col in required_columns if col not in df.columns]}")
        
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        features_df = pd.DataFrame(index=df.index)
        features_df["RSI"] = RSIIndicator(close=close).rsi()
        macd = MACD(close=close)
        features_df["MACD"] = macd.macd()
        features_df["MA_20"] = SMAIndicator(close=close, window=20).sma_indicator()
        features_df["ATR"] = AverageTrueRange(high=high, low=low, close=close).average_true_range()
        features_df.fillna(0, inplace=True)
        return features_df
    except Exception as e:
        logger.error(f"❌ Feature preparation error: {e}")
        return None

# Generate an AI-driven trading signal for a given cryptocurrency
async def generate_ai_comment(symbol: str) -> str:
    if not symbol:
        logger.error("❌ generate_ai_comment: symbol is None")
        return "⚠️ Invalid coin symbol: None provided."
    try:
        klines = client.get_klines(symbol=f"{symbol.upper()}USDT", interval=BinanceClient.KLINE_INTERVAL_15MINUTE, limit=50)
        if len(klines) < 50:
            raise ValueError(f"Insufficient data for {symbol}: {len(klines)} rows")

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        df['macd'] = MACD(close=df['close']).macd_diff()
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()

        latest = df.dropna().iloc[-1]
        current_price = latest['close']

        if not model or not expected_features:
            raise ValueError("Model or feature list not loaded!")
        
        model_input = pd.DataFrame([latest[expected_features]], columns=expected_features)
        prediction = model.predict(model_input)[0]
        tp = tp_model.predict(model_input)[0] * current_price * 0.01 if tp_model else None
        sl = sl_model.predict(model_input)[0] * current_price * 0.01 if sl_model else None

        rsi = latest['rsi']
        macd = latest['macd']
        sma_20 = latest['sma_20']
        rsi_comment = "RSI in oversold territory." if rsi < 30 else ("RSI in overbought territory." if rsi > 70 else "RSI neutral.")
        macd_comment = "MACD bullish." if macd > 0 else "MACD bearish."
        ai_signal = "📈 BUY" if prediction == 1 else "📉 SELL"
        tp_text = f"🎯 TP: ${tp:.2f}" if tp else "❌ TP prediction failed."
        sl_text = f"🛑 SL: ${sl:.2f}" if sl else "❌ SL prediction failed."

        _, change_24h, change_7d = await fetch_ohlc_data(symbol, days=7)
        return (
            f"📊 {symbol.upper()} (${current_price:.2f})\n"
            f"📈 24h: {change_24h:+.2f}% | 🗕 7d: {change_7d:+.2f}%\n\n"
            f"💡 {ai_signal}\n"
            f"📉 RSI: {rsi:.2f} | 🧺 MACD: {macd:.2f}\n"
            f"🧠 AI Comment: {rsi_comment} {macd_comment}"
        )
    except Exception as e:
        logger.error(f"❌ AI comment error for {symbol}: {e}")
        return f"⚠️ Failed to generate AI comment for {symbol}: {e}"

# Send an AI trading signal with feedback buttons
async def send_ai_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, signal_text: str):
    try:
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("👍 Like", callback_data="feedback:like"),
             InlineKeyboardButton("👎 Dislike", callback_data="feedback:dislike")]
        ])
        message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=signal_text,
            reply_markup=keyboard
        )
        signals = load_signals()
        signals.append({"message_id": message.message_id, "text": signal_text, "likes": [], "dislikes": []})
        save_signals(signals)
    except Exception as e:
        logger.error(f"❌ Signal send error: {e}")
        await update.effective_message.reply_text("❌ Failed to send signal\\.", parse_mode="MarkdownV2")

# Portfolio management functions

# Retrieve a user's portfolio from file
def get_portfolio(user_id):
    return load_json("data/portfolios.json").get(str(user_id), {})

# Add a coin to a user's portfolio
def add_coin(user_id, symbol, amount, buy_price=None):
    users = load_json("data/portfolios.json")
    user_id_str = str(user_id)
    if user_id_str not in users:
        users[user_id_str] = {}
    users[user_id_str][symbol.lower()] = {"amount": amount, "buy_price": buy_price}
    save_json("data/portfolios.json", users)
    return True

# Remove a coin from a user's portfolio
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

# Update the amount of a coin in a user's portfolio
def update_coin(user_id, symbol, amount):
    users = load_json("data/portfolios.json")
    user_id_str = str(user_id)
    symbol = symbol.lower()
    if user_id_str in users and symbol in users[user_id_str]:
        users[user_id_str][symbol]["amount"] = amount
        save_json("data/portfolios.json", users)
        return True
    return False

# Clear a user's entire portfolio
def clear_portfolio(user_id):
    users = load_json("data/portfolios.json")
    user_id_str = str(user_id)
    if user_id_str in users:
        del users[user_id_str]
        save_json("data/portfolios.json", users)
        return True
    return False

# News-related functions

# Fetch and display the latest cryptocurrency news
async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("🗞️ Fetching crypto news from all sources...")
    articles = await fetch_all_crypto_news()
    if not articles:
        await update.effective_message.reply_text("❌ No news data found\\.", parse_mode="MarkdownV2")
        return

    sent_count = 0
    for article in articles[:8]:
        url = article.get("url")
        title = article.get("title", "No Title")
        description = article.get("description", "No Description")
        norm_url = normalize_url(url)
        if norm_url and norm_url not in sent_news_urls:
            summary = await summarize_news(title, description)
            text = f"📰 *{escape(title)}*\n_{escape(article['source'])}_\n{escape(summary)}\n<a href=\"{url}\">🔗 Read More</a>"
            try:
                await update.effective_message.reply_text(text, parse_mode="HTML")
                sent_news_urls.add(norm_url)
                save_sent_urls()
                sent_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Send error: {e}")

    if sent_count == 0:
        await update.effective_message.reply_text("⚠️ No new news found\\.", parse_mode="MarkdownV2")

# Fetch crypto news from multiple sources
async def fetch_all_crypto_news():
    async with aiohttp.ClientSession() as session:
        newsapi = await fetch_newsapi_news()
        cointelegraph = await fetch_rss_feed(session, "https://cointelegraph.com/rss", "Cointelegraph")
        return newsapi + cointelegraph

# Fetch news from NewsAPI
async def fetch_newsapi_news():
    url = f"https://newsapi.org/v2/top-headlines?category=business&q=crypto&apiKey={NEWS_API_KEY}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        {
                            "title": a["title"],
                            "description": a.get("description", ""),
                            "url": a.get("url"),
                            "source": "NewsAPI"
                        } for a in data.get("articles", [])
                    ]
                logger.error(f"❌ NewsAPI status: {response.status}")
    except Exception as e:
        logger.error(f"❌ NewsAPI fetch error: {e}")
    return []

# Fetch news from an RSS feed
async def fetch_rss_feed(session, rss_url, source_name):
    try:
        api_url = f"https://api.rss2json.com/v1/api.json?rss_url={rss_url}"
        async with session.get(api_url) as response:
            if response.status == 200:
                data = await response.json()
                return [
                    {
                        "title": item["title"],
                        "description": item.get("description", ""),
                        "url": item["link"],
                        "source": source_name
                    } for item in data.get("items", [])
                ]
            logger.warning(f"⚠️ {source_name} RSS status: {response.status}")
    except Exception as e:
        logger.error(f"❌ {source_name} RSS error: {e}")
    return []

# Generate a summary of a news article using OpenAI
async def summarize_news(title, description):
    prompt = f"Summarize the following news article for investors:\n\nTitle: {title}\nDescription: {description}\n\nProvide a brief summary."
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ Summary error: {e}")
        return "⚠️ Failed to generate summary."

# Periodically fetch and send new news to all users
async def send_periodic_news(app):
    while True:
        users = load_json("data/users.json")
        articles = await fetch_all_crypto_news()
        if not articles:
            logger.warning("⚠️ No news data found for periodic update.")
            await asyncio.sleep(300)  # Wait 5 minutes
            continue

        sent_count = 0
        for article in articles:
            url = article.get("url")
            title = article.get("title", "No Title")
            description = article.get("description", "No Description")
            norm_url = normalize_url(url)
            if norm_url and norm_url not in sent_news_urls:
                summary = await summarize_news(title, description)
                text = f"📰 *{escape(title)}*\n_{escape(article['source'])}_\n{escape(summary)}\n<a href=\"{url}\">🔗 Read More</a>"
                for user_id in users.keys():
                    try:
                        await app.bot.send_message(
                            chat_id=int(user_id),
                            text=text,
                            parse_mode="HTML",
                            disable_web_page_preview=True
                        )
                        sent_count += 1
                    except Exception as e:
                        logger.warning(f"❌ Failed to send news to {user_id}: {e}")
                sent_news_urls.add(norm_url)
                save_sent_urls()

        if sent_count == 0:
            logger.info("⚠️ No new news sent in periodic update.")
        else:
            logger.info(f"✅ Sent {sent_count} news articles to users.")

        await asyncio.sleep(300)  # Wait 5 minutes

# Notification-related functions

# Notify a user if their premium subscription has expired
async def notify_user_if_expired(user_id: int, chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    try:
        await context.bot.send_message(chat_id=chat_id, text="⚠️ Your Premium subscription has expired\\. Renew with /premium\\.", parse_mode="MarkdownV2")
        logger.info(f"🔔 Premium expiration notification sent to user: {user_id}")
    except Exception as e:
        logger.error(f"❌ Failed to send notification to user {user_id}: {e}")

# Periodically check price alerts and notify users when triggered
async def check_alerts(app):
    while True:
        alerts = load_json("data/alerts.json", {})
        if not alerts:
            await asyncio.sleep(300)
            continue
        for user_id, user_alerts in alerts.items():
            for alert in user_alerts:
                symbol = alert.get("symbol")
                target_price = alert.get("target_price")
                if not symbol:
                    logger.warning(f"⚠️ check_alerts: symbol is None for user {user_id}")
                    continue
                price = await fetch_price(symbol)
                if price and target_price and price >= target_price:
                    try:
                        await app.bot.send_message(
                            chat_id=int(user_id),
                            text=f"📢 *{symbol}* reached ${target_price:.2f}! Current price: ${price:.2f}",
                            parse_mode="Markdown"
                        )
                        delete_alert(user_id, symbol)
                    except Exception as e:
                        logger.error(f"❌ Notification failed for user {user_id}, symbol {symbol}: {e}")
        await asyncio.sleep(300)

# Alert management functions

# Add a price alert for a user
def add_alert(user_id, symbol, target_price):
    alerts = load_json("data/alerts.json", {})
    user_id_str = str(user_id)
    if user_id_str not in alerts:
        alerts[user_id_str] = []
    alerts[user_id_str].append({"symbol": symbol.upper(), "target_price": target_price, "user_id": user_id})
    save_json("data/alerts.json", alerts)
    return True

# Delete a price alert for a user
def delete_alert(user_id, symbol):
    alerts = load_json("data/alerts.json", {})
    user_id_str = str(user_id)
    symbol = symbol.upper()
    if user_id_str in alerts:
        alerts[user_id_str] = [alert for alert in alerts[user_id_str] if alert["symbol"] != symbol]
        if not alerts[user_id_str]:
            del alerts[user_id_str]
        save_json("data/alerts.json", alerts)
        return True
    return False

# Premium-related functions

# Load list of premium users with their subscription details
def load_premium_users():
    return load_json("data/premium_users.json", {})

# Save premium users to file
def save_premium_users(users):
    save_json("data/premium_users.json", users)

# Check if a user's premium subscription is active
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
        logger.error(f"❌ Premium check error for user {user_id}: {e}")
        return False

# Check for expired premium subscriptions and notify users
async def check_premium_expiry(bot: Bot):
    premium_users = load_premium_users()
    today = datetime.today().date()
    for user_id, info in premium_users.copy().items():
        end_date_str = info.get("end")
        if not end_date_str:
            continue
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        if today > end_date:
            try:
                await bot.send_message(
                    chat_id=int(user_id),
                    text="⚠️ Your Premium subscription has expired\\. Renew with /premium\\.",
                    parse_mode="MarkdownV2"
                )
            except Exception as e:
                logger.error(f"❌ Notification error for user {user_id}: {e}")
            del premium_users[user_id]
    save_premium_users(premium_users)

# Display premium subscription plans and payment links
async def premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🟢 1 Month \\- $29\\.99", url="https://nowpayments.io/payment/?iid=5260731771")],
        [InlineKeyboardButton("🔵 3 Months \\- $69\\.99", url="https://nowpayments.io/payment/?iid=4400895826")],
        [InlineKeyboardButton("🟣 1 Year \\- $399\\.99", url="https://nowpayments.io/payment/?iid=4501340550")],
    ])
    msg = (
        "*👑 Coinspace Premium Plans!*\n\n"
        "*⚡️ Benefits:*\n"
        "• Unlimited AI Leverage Signals\n"
        "• Full market overview access\n"
        "• Priority support & early feature access\n\n"
        "*💳 Plans:*\n"
        "• 1 Month: $29\\.99\n"
        "• 3 Months: $69\\.99\n"
        "• 1 Year: $399\\.99\n\n"
        "*✅ After payment: /activate_premium*"
    )
    await update.effective_message.reply_text(msg, parse_mode="MarkdownV2", reply_markup=keyboard, disable_web_page_preview=True)

# Activate a premium subscription using a payment ID
async def activate_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) != 1:
        await update.effective_message.reply_text *("❌ Usage: /activate_premium <payment_id>", parse_mode="MarkdownV2")
        return
    payment_id = context.args[0]
    user_id = str(update.effective_user.id)
    valid_payments = {"5260731771": 30, "4400895826": 90, "4501340550": 365}
    if payment_id not in valid_payments:
        await update.effective_message.reply_text("❌ Invalid payment ID\\.", parse_mode="MarkdownV2")
        return
    premium_users = load_premium_users()
    today = datetime.today().date()
    end_date = today + timedelta(days=valid_payments[payment_id])
    premium_users[user_id] = {"start": str(today), "end": str(end_date)}
    save_premium_users(premium_users)
    await update.effective_message.reply_text(f"✅ Premium activated until {end_date}\\!", parse_mode="MarkdownV2")

# Grant premium status to a user
async def make_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.effective_message.reply_text("❌ No admin privileges\\.", parse_mode="MarkdownV2")
        return
    if not context.args or len(context.args) != 1:
        await update.effective_message.reply_text("❌ Usage: /make_premium <user_id>", parse_mode="MarkdownV2")
        return
    target_user_id = str(context.args[0])
    premium_users = load_premium_users()
    if target_user_id in premium_users:
        await update.effective_message.reply_text(f"⚠️ {target_user_id} is already Premium\\.", parse_mode="MarkdownV2")
        return
    today = datetime.today().date()
    end_date = today + timedelta(days=30)
    premium_users[target_user_id] = {"start": str(today), "end": str(end_date)}
    save_premium_users(premium_users)
    await update.effective_message.reply_text(f"✅ Granted Premium to {target_user_id} until {end_date}\\.", parse_mode="MarkdownV2")

# Revoke premium status from a user
async def remove_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.effective_message.reply_text("❌ No admin privileges\\.", parse_mode="MarkdownV2")
        return
    if not context.args or len(context.args) != 1:
        await update.effective_message.reply_text("❌ Usage: /remove_premium <user_id>", parse_mode="MarkdownV2")
        return
    target_user_id = str(context.args[0])
    premium_users = load_premium_users()
    if target_user_id in premium_users:
        del premium_users[target_user_id]
        save_premium_users(premium_users)
        await update.effective_message.reply_text(f"✅ Premium status removed for {target_user_id}\\.", parse_mode="MarkdownV2")
    else:
        await update.effective_message.reply_text(f"⚠️ {target_user_id} is not Premium\\.", parse_mode="MarkdownV2")

# List all premium users and their subscription details
async def premium_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_admin(update.effective_user.id):
        await update.effective_message.reply_text("❌ No admin privileges\\.", parse_mode="MarkdownV2")
        return
    premium_users = load_premium_users()
    if not premium_users:
        await update.effective_message.reply_text("❌ No premium users found\\.", parse_mode="MarkdownV2")
        return
    msg = "*💎 Premium User List*\n\n"
    for uid, info in premium_users.items():
        msg += f"• `{uid}` \\- Valid until: *{info.get('end', 'None')}*\n"
    await update.effective_message.reply_text(msg, parse_mode="MarkdownV2")

# Admin-related functions

# Display the admin panel with available admin commands
async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.effective_message.reply_text("❌ No admin privileges\\.", parse_mode="MarkdownV2")
        return
    msg = (
        "*🔧 Admin Panel*\n\n"
        "• `/broadcast <message>` \\- Send broadcast\n"
        "• `/make_admin <user_id>` \\- Add admin\n"
        "• `/remove_admin <user_id>` \\- Remove admin\n"
        "• `/make_premium <user_id>` \\- Grant premium\n"
        "• `/remove_premium <user_id>` \\- Revoke premium\n"
        "• `/admin_list` \\- List admins\n"
        "• `/premium_list` \\- List premium users"
    )
    await update.effective_message.reply_text(msg, parse_mode="MarkdownV2")

# Grant admin privileges to a user
async def make_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.effective_message.reply_text("❌ No admin privileges\\.", parse_mode="MarkdownV2")
        return
    if not context.args or len(context.args) != 1:
        await update.effective_message.reply_text("❌ Usage: /make_admin <user_id>", parse_mode="MarkdownV2")
        return
    target_user_id = str(context.args[0])
    admin_users = load_admin_users()
    if target_user_id not in admin_users:
        admin_users.add(target_user_id)
        save_admin_users(admin_users)
        await update.effective_message.reply_text(f"✅ {target_user_id} is now an admin\\.", parse_mode="MarkdownV2")
    else:
        await update.effective_message.reply_text(f"⚠️ {target_user_id} is already an admin\\.", parse_mode="MarkdownV2")

# Revoke admin privileges from a user
async def remove_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.effective_message.reply_text("❌ No admin privileges\\.", parse_mode="MarkdownV2")
        return
    if not context.args or len(context.args) != 1:
        await update.effective_message.reply_text("❌ Usage: /remove_admin <user_id>", parse_mode="MarkdownV2")
        return
    target_user_id = str(context.args[0])
    admin_users = load_admin_users()
    if target_user_id in admin_users:
        admin_users.remove(target_user_id)
        save_admin_users(admin_users)
        await update.effective_message.reply_text(f"✅ Admin status removed for {target_user_id}\\.", parse_mode="MarkdownV2")
    else:
        await update.effective_message.reply_text(f"⚠️ {target_user_id} is not an admin\\.", parse_mode="MarkdownV2")

# Send a broadcast message to all users
async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_admin(update.effective_user.id):
        await update.effective_message.reply_text("❌ No admin privileges\\.", parse_mode="MarkdownV2")
        return
    if not context.args:
        await update.effective_message.reply_text("📢 Usage: /broadcast <message>", parse_mode="MarkdownV2")
        return
    message_text = "📢 Broadcast:\n" + " ".join(context.args)
    users = load_json("data/users.json")
    count = 0
    for user_id in users.keys():
        try:
            await context.bot.send_message(chat_id=int(user_id), text=message_text)
            count += 1
        except Exception as e:
            logger.warning(f"❌ Failed to send to {user_id}: {e}")
    await update.effective_message.reply_text(f"✅ Broadcast sent to {count} users\\.", parse_mode="MarkdownV2")

# List all admin users
async def admin_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_admin(update.effective_user.id):
        await update.effective_message.reply_text("❌ No admin privileges\\.", parse_mode="MarkdownV2")
        return
    admins = load_admin_users()
    if not admins:
        await update.effective_message.reply_text("⚠️ No admins found\\.", parse_mode="MarkdownV2")
        return
    msg = "*👮 Admin Users:*\n"
    for admin_id in admins:
        msg += f"• `{admin_id}` \\- {admin_id}\n"
    await update.effective_message.reply_text(msg, parse_mode="MarkdownV2")

# Bot command handlers

# Show disclaimer for acceptance (currently disabled)
async def accept(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    save_user(user_id)
    update_user_metadata(user_id)
    await start(update, context)

# Handle new chat member events (currently disabled for disclaimer)
async def handle_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_member_update = update.chat_member
    if chat_member_update.new_chat_member.status in ["member", "administrator", "creator"]:
        user_id = chat_member_update.new_chat_member.user.id
        save_user(user_id)
        update_user_metadata(user_id)

# Initialize the bot and show welcome message
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    save_user(user_id)
    update_user_metadata(user_id)

    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("📖 View Commands (/help)", callback_data="help")]])
    msg = (
        "*👋 Welcome to Coinspace Bot!*\n\n"
        f"*🚀 Your User ID:* `{user_id}`\n\n"
        "*🚀 Get daily AI-powered trading signals, price alerts, portfolio tracking, and live market updates.*\n\n"
        "*🔐 Upgrade to Premium:*\n"
        "• Unlimited AI Leverage Signals\n"
        "• Full market overview access\n"
        "• Priority support & early feature access\n\n"
        "*✅ Upgrade: /premium*"
    )
    await update.effective_message.reply_text(msg, parse_mode="MarkdownV2", reply_markup=keyboard, disable_web_page_preview=True)

# Display the list of available commands
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "*🤖 Coinspace Commands*\n\n"
        "*📊 Portfolio*\n"
        "• `/add BTC 0\\.5 30000` \\- Add coin\n"
        "• `/update BTC 1\\.0` \\- Update coin\n"
        "• `/remove BTC` \\- Remove coin\n"
        "• `/clear` \\- Clear portfolio\n"
        "• `/portfolio` \\- View portfolio\n"
        "• `/performance` \\- View performance\n"
        "• `/graph` \\- View portfolio graph\n\n"
        "*💹 Market Tools*\n"
        "• `/price BTC` \\- Check price\n"
        "• `/alert BTC 70000` \\- Set alert\n"
        "• `/ai BTC` \\- AI signal \\(Premium\\)\n"
        "• `/backtest BTC` \\- Backtest strategy\n\n"
        "*📰 News & Premium*\n"
        "• `/news` \\- Latest news\n"
        "• `/premium` \\- Upgrade to Premium"
    )
    try:
        await update.effective_message.reply_text(msg, parse_mode="MarkdownV2")
    except Exception as e:
        logger.error(f"❌ Help command error: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="❌ Failed to display commands\\.",
            parse_mode="MarkdownV2"
        )

# Fetch and display the current price of a cryptocurrency
async def price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.effective_message.reply_text("Please provide a coin: /price BTC", parse_mode="MarkdownV2")
        return
    symbol = context.args[0].upper()
    if not symbol or symbol not in symbol_to_id_map:
        await update.effective_message.reply_text(f"❌ Invalid coin symbol: {symbol or 'None'}\\\.", parse_mode="MarkdownV2")
        return
    price = await fetch_price(symbol)
    if price is not None:
        await update.effective_message.reply_text(f"{symbol} price: ${price:.2f}")
    else:
        await update.effective_message.reply_text(f"❌ Failed to fetch price for {symbol}\\.", parse_mode="MarkdownV2")

# Display the user's portfolio with current values
async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    holdings = get_portfolio(user_id)
    if not holdings:
        await update.effective_message.reply_text("📭 Portfolio is empty\\. Add coins with /add\\.", parse_mode="MarkdownV2")
        return
    symbols = [sym.upper() for sym in holdings.keys() if sym.upper() in symbol_to_id_map]
    total_value = 0
    msg = "*📊 Portfolio:*\n"
    for symbol in symbols:
        price = await fetch_price(symbol)
        amount = holdings.get(symbol.lower(), {}).get("amount", 0)
        if price:
            value = price * amount
            total_value += value
            msg += f"• `{symbol}`: {amount} × ${price:.2f} \\= ${value:.2f}\n"
        else:
            msg += f"• `{symbol}`: Price unavailable\n"
    msg += f"\n*💰 Total Value:* ${total_value:.2f}"
    await update.effective_message.reply_text(msg, parse_mode="MarkdownV2")

# Add a coin to the user's portfolio
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) not in [2, 3]:
        await update.effective_message.reply_text("❌ Usage: /add <coin> <amount> [buy_price] \\(e\\.g\\., /add BTC 0\\.5 30000\\)", parse_mode="MarkdownV2")
        return
    symbol = context.args[0].upper()
    if not symbol or symbol not in symbol_to_id_map:
        await update.effective_message.reply_text(f"❌ Invalid coin symbol: {symbol or 'None'}\\\.", parse_mode="MarkdownV2")
        return
    try:
        amount = float(context.args[1])
        buy_price = float(context.args[2]) if len(context.args) == 3 else None
    except ValueError:
        await update.effective_message.reply_text("❌ Invalid amount or price\\.", parse_mode="MarkdownV2")
        return
    user_id = update.effective_user.id
    add_coin(user_id, symbol, amount, buy_price)
    msg = f"✅ Added {amount} {symbol} to portfolio\\."
    if buy_price:
        msg += f" Buy price: ${buy_price:.2f}"
    await update.effective_message.reply_text(msg, parse_mode="MarkdownV2")

# Remove a coin from the user's portfolio
async def remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.effective_message.reply_text("❌ Usage: /remove <coin> \\(e\\.g\\., /remove BTC\\)", parse_mode="MarkdownV2")
        return
    symbol = context.args[0].upper()
    if not symbol or symbol not in symbol_to_id_map:
        await update.effective_message.reply_text(f"❌ Invalid coin symbol: {symbol or 'None'}\\\.", parse_mode="MarkdownV2")
        return
    user_id = update.effective_user.id
    success = remove_coin(user_id, symbol)
    if success:
        await update.effective_message.reply_text(f"🗑️ {symbol} removed from portfolio\\.", parse_mode="MarkdownV2")
    else:
        await update.effective_message.reply_text(f"⚠️ {symbol} not found in portfolio\\.", parse_mode="MarkdownV2")

# Update the amount of a coin in the user's portfolio
async def update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.effective_message.reply_text("❌ Usage: /update <coin> <amount> \\(e\\.g\\., /update BTC 1\\.0\\)", parse_mode="MarkdownV2")
        return
    symbol = context.args[0].upper()
    if not symbol or symbol not in symbol_to_id_map:
        await update.effective_message.reply_text(f"❌ Invalid coin symbol: {symbol or 'None'}\\\.", parse_mode="MarkdownV2")
        return
    try:
        amount = float(context.args[1])
    except ValueError:
        await update.effective_message.reply_text("❌ Invalid amount\\.", parse_mode="MarkdownV2")
        return
    user_id = update.effective_user.id
    success = update_coin(user_id, symbol, amount)
    if success:
        await update.effective_message.reply_text(f"🔗 Updated {symbol} amount to {amount}\\.", parse_mode="MarkdownV2")
    else:
        await update.effective_message.reply_text(f"⚠️ {symbol} not found in portfolio\\.", parse_mode="MarkdownV2")

# Clear the user's entire portfolio
async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    success = clear_portfolio(user_id)
    if success:
        await update.effective_message.reply_text("🧼 Portfolio cleared\\.", parse_mode="MarkdownV2")
    else:
        await update.effective_message.reply_text("❗ No data to clear\\.", parse_mode="MarkdownV2")

# Generate and display a pie chart of the user's portfolio distribution
async def graph(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_premium_status(user_id):
        await update.effective_message.reply_text("❌ The /graph command is for Premium users only\\. Upgrade with /premium\\.", parse_mode="MarkdownV2")
        return
    holdings = get_portfolio(user_id)
    if not holdings:
        await update.effective_message.reply_text("📭 Portfolio is empty\\.", parse_mode="MarkdownV2")
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
        await update.effective_message.reply_text("⚠️ No price data available\\.", parse_mode="MarkdownV2")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    ax.set_title("📈 Portfolio Distribution")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    await update.effective_message.reply_photo(photo=InputFile(buf, filename="portfolio_graph.png"))

# Display the performance (profit/loss) of the user's portfolio
async def performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    portfolio = get_portfolio(user_id)
    if not portfolio:
        await update.effective_message.reply_text("📭 Portfolio is empty\\.", parse_mode="MarkdownV2")
        return
    symbols = [sym.upper() for sym in portfolio.keys() if sym.upper() in symbol_to_id_map]
    msg, total_pl = "*📈 Portfolio Performance:*\n", 0
    for symbol in symbols:
        current_price = await fetch_price(symbol)
        amount = portfolio.get(symbol.lower(), {}).get("amount", 0)
        buy_price = portfolio.get(symbol.lower(), {}).get("buy_price")
        if not current_price:
            msg += f"• `{symbol}`: Price unavailable\n"
            continue
        if buy_price:
            current_value, cost_basis = current_price * amount, buy_price * amount
            pl = current_value - cost_basis
            total_pl += pl
            msg += f"• `{symbol}`: Buy ${buy_price:.2f} \\→ Current ${current_price:.2f} \\| P/L: ${pl:.2f}\n"
        else:
            msg += f"• `{symbol}`: Buy price unknown\n"
    msg += f"\n*💼 Total P/L:* ${total_pl:.2f}"
    await update.effective_message.reply_text(msg, parse_mode="MarkdownV2")

# Set a price alert for a cryptocurrency
async def alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.effective_message.reply_text("Usage: /alert <coin> <price> \\(e\\.g\\., /alert BTC 70000\\)", parse_mode="MarkdownV2")
        return
    symbol = context.args[0].upper()
    if not symbol or symbol not in symbol_to_id_map:
        await update.effective_message.reply_text(f"❌ Invalid coin symbol: {symbol or 'None'}\\\.", parse_mode="MarkdownV2")
        return
    try:
        target_price = float(context.args[1])
    except ValueError:
        await update.effective_message.reply_text("❌ Invalid price\\.", parse_mode="MarkdownV2")
        return
    user_id = update.effective_user.id
    if not check_premium_status(user_id):
        await update.effective_message.reply_text("❌ The /alert command is for Premium users only\\.", parse_mode="MarkdownV2")
        return
    add_alert(user_id, symbol, target_price)
    await update.effective_message.reply_text(f"🔔 Alert set for {symbol} at ${target_price:.2f}\\.", parse_mode="MarkdownV2")

# Generate and send an AI-driven trading signal for a cryptocurrency
async def ai_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.effective_message.reply_text("❌ Usage: /ai <coin> \\(e\\.g\\., /ai BTC\\)", parse_mode="MarkdownV2")
        return
    symbol = context.args[0].upper()
    user_id = update.effective_user.id
    if not symbol or symbol not in symbol_to_id_map:
        await update.effective_message.reply_text(f"❌ Invalid coin symbol: {symbol or 'None'}\\\.", parse_mode="MarkdownV2")
        return
    if not check_premium_status(user_id):
        await update.effective_message.reply_text("❌ The /ai command is for Premium users only\\. Upgrade with /premium\\.", parse_mode="MarkdownV2")
        return
    await update.effective_message.reply_text("💬 Generating AI comment...")
    comment = await generate_ai_comment(symbol)
    await send_ai_signal(update, context, comment)

# Perform a backtest of a simple trading strategy for a cryptocurrency
async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.effective_message.reply_text("Usage: /backtest <coin> \\(e\\.g\\., /backtest BTC\\)", parse_mode="MarkdownV2")
        return
    symbol = context.args[0].upper()
    if not symbol or symbol not in symbol_to_id_map:
        await update.effective_message.reply_text(f"❌ Invalid coin symbol: {symbol or 'None'}\\\.", parse_mode="MarkdownV2")
        return
    user_id = update.effective_user.id
    if not check_premium_status(user_id):
        await update.effective_message.reply_text("❌ The /backtest command is for Premium users only\\.", parse_mode="MarkdownV2")
        return
    df, _, _ = await fetch_ohlc_data(symbol, days=30)
    if df is None or df.empty:
        await update.effective_message.reply_text("❌ No data available\\.", parse_mode="MarkdownV2")
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
    msg = f"*📈 {symbol} RSI + MA Backtest \\(30 days\\):*\n✅ Trades: {len(buy_points)}\n*💰 Total Profit:* ${pnl:.2f}"
    await update.effective_message.reply_text(msg, parse_mode="MarkdownV2")

# Handle user acceptance of the disclaimer (currently disabled)
async def accept_disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    save_user(user_id)
    update_user_metadata(user_id)
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("✅ Terms accepted\\. Welcome!", parse_mode="MarkdownV2")
    await start(update, context)

# Handle user feedback (like/dislike) for AI signals
async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    feedback = query.data.split(":")[1]
    message_id = query.message.message_id
    user_id = query.from_user.id
    signals = load_signals()
    for s in signals:
        if s.get("message_id") == message_id:
            if user_id in s.get("likes", []) or user_id in s.get("dislikes", []):
                await query.answer("❌ You already provided feedback for this message\\.", show_alert=True)
                return
            if feedback == "like":
                s.setdefault("likes", []).append(user_id)
            elif feedback == "dislike":
                s.setdefault("dislikes", []).append(user_id)
            break
    save_signals(signals)
    await query.edit_message_reply_markup(reply_markup=None)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="✅ Thanks for your feedback!"
    )

# Handle errors during bot operation
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"❌ Error: {context.error}", exc_info=True)
    if update and update.effective_message:
        await update.effective_message.reply_text("❌ An error occurred\\. Please try again\\.", parse_mode="MarkdownV2")

# Handle button clicks (e.g., help or accept disclaimer)
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    try:
        if query.data == "help":
            await help_command(update, context)
        elif query.data == "accept_disclaimer":
            await accept_disclaimer(update, context)
        else:
            logger.warning(f"⚠️ Unknown callback data: {query.data}")
    except Exception as e:
        logger.error(f"❌ Button handler error: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="❌ Failed to process button click\\.",
            parse_mode="MarkdownV2"
        )

# Run background tasks like checking premium expirations
async def background_tasks(bot: Bot):
    logger.info("🔄 Background tasks started.")
    while True:
        await check_premium_expiry(bot)
        await asyncio.sleep(3600)

# Initialize and run the Telegram bot
async def run_bot():
    logger.info("🚀 Bot starting...")
    app = ApplicationBuilder().token(TOKEN).build()
    await load_symbol_map()
    app.add_handler(ChatMemberHandler(handle_new_member, ChatMemberHandler.CHAT_MEMBER))
    app.add_handler(CommandHandler("accept", accept))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("portfolio", portfolio))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("remove", remove))
    app.add_handler(CommandHandler("update", update))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("alert", alert))
    app.add_handler(CommandHandler("graph", graph))
    app.add_handler(CommandHandler("performance", performance))
    app.add_handler(CommandHandler("news", news))
    app.add_handler(CommandHandler("backtest", backtest))
    app.add_handler(CommandHandler("premium", premium))
    app.add_handler(CommandHandler("activate_premium", activate_premium))
    app.add_handler(CommandHandler("admin_panel", admin_panel))
    app.add_handler(CommandHandler("broadcast", broadcast))
    app.add_handler(CommandHandler("make_admin", make_admin))
    app.add_handler(CommandHandler("remove_admin", remove_admin))
    app.add_handler(CommandHandler("admin_list", admin_list))
    app.add_handler(CommandHandler("make_premium", make_premium))
    app.add_handler(CommandHandler("remove_premium", remove_premium))
    app.add_handler(CommandHandler("premium_list", premium_list))
    app.add_handler(CommandHandler("ai", ai_comment))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(CallbackQueryHandler(feedback_handler))
    app.add_error_handler(error_handler)
    asyncio.create_task(check_alerts(app))
    asyncio.create_task(background_tasks(app.bot))
    asyncio.create_task(send_periodic_news(app))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    logger.info("✅ Bot started.")

if __name__ == "__main__":
    asyncio.run(run_bot())
