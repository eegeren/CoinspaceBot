import os
import asyncio
import aiohttp
from telegram import Update, InputFile, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
import matplotlib.pyplot as plt
import io
import json
import joblib
import pandas as pd
from html import escape
from openai import AsyncOpenAI
import requests
import numpy as np
from urllib.parse import urlparse, urlunparse
import hashlib
import logging
from dotenv import load_dotenv

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

# OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Load models globally
model = None
tp_model = None
sl_model = None
try:
    model_path = os.path.abspath("model.pkl")
    model = joblib.load(model_path)
    logger.info(f"✅ Model loaded: {model_path}")
    tp_model = joblib.load("tp_model.pkl")
    sl_model = joblib.load("sl_model.pkl")
    logger.info(f"✅ TP/SL models loaded")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")

# Accepted users
def load_accepted_users():
    if not os.path.exists("accepted_users.json"):
        return set()
    with open("accepted_users.json", "r") as f:
        try:
            return set(json.load(f))
        except json.JSONDecodeError:
            return set()

def save_accepted_users(users):
    with open("accepted_users.json", "w") as f:
        json.dump(list(users), f)

async def check_user_accepted(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    return update.effective_user.id in load_accepted_users()

accepted_users = load_accepted_users()

# News and signal files
SENT_NEWS_FILE = "sent_news.json"
if os.path.exists(SENT_NEWS_FILE):
    with open(SENT_NEWS_FILE, "r") as f:
        try:
            sent_news_urls = set(json.load(f))
        except:
            sent_news_urls = set()
else:
    sent_news_urls = set()

SIGNAL_FILE = "signals.json"

def load_signals():
    if os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_signals(data):
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
                logger.info(f"✅ Coin symbols loaded.")
            else:
                logger.error(f"❌ Failed to fetch coin list: status={response.status}")

# Helper functions
def normalize_url(raw_url):
    if not raw_url:
        return ""
    parsed = urlparse(raw_url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

def save_sent_urls():
    with open(SENT_NEWS_FILE, "w") as f:
        json.dump(list(sent_news_urls), f)

def get_news_key(url, title):
    norm_url = normalize_url(url)
    key_base = f"{norm_url}|{title.strip().lower()}"
    return hashlib.md5(key_base.encode()).hexdigest()

# Fetch price
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
            logger.error(f"❌ Failed to fetch price for {full_symbol}: status={response.status}")
            return None

# Price command
async def pr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Please enter a coin: /pr BTC")
        return
    symbol = context.args[0].upper()
    if symbol not in symbol_to_id_map:
        await update.message.reply_text(f"❌ No match found for {symbol}.")
        return
    price = await fetch_price(symbol)
    if price is not None:
        await update.message.reply_text(f"{symbol} price: ${price:.2f}")
    else:
        await update.message.reply_text(f"❌ Failed to retrieve price for {symbol}.")

# Fetch OHLC data
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
                    logger.error(f"❌ Empty or invalid OHLC data for {full_symbol}")
                    return None, 0.0, 0.0
                try:
                    prices = [float(item[4]) for item in data]
                    timestamps = [int(item[0]) for item in data]
                    df = pd.DataFrame({"price": prices}, index=pd.to_datetime(timestamps, unit="ms"))
                    if df.empty or not isinstance(df, pd.DataFrame) or len(df) < 26:
                        logger.error(f"❌ Invalid or insufficient data for {full_symbol}")
                        return None, 0.0, 0.0
                    change_24h = ((df["price"].iloc[-1] - df["price"].iloc[-24]) / df["price"].iloc[-24]) * 100 if len(df) >= 24 else 0.0
                    change_7d = ((df["price"].iloc[-1] - df["price"].iloc[0]) / df["price"].iloc[0]) * 100
                    return df, change_24h, change_7d
                except (ValueError, IndexError) as e:
                    logger.error(f"❌ Error processing OHLC data for {full_symbol}: {e}")
                    return None, 0.0, 0.0
            logger.error(f"❌ Failed to fetch OHLC data for {full_symbol}: status={response.status}")
            return None, 0.0, 0.0

# Technical indicators
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        delta = prices[-i] - prices[-i - 1]
        gains.append(max(0, delta))
        losses.append(max(0, -delta))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period if losses else 0.001
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def exponential_moving_average(prices, window):
    if len(prices) < window:
        return None
    ema = []
    k = 2 / (window + 1)
    for i, price in enumerate(prices):
        if i == 0:
            ema.append(price)
        else:
            ema.append(price * k + ema[-1] * (1 - k))
    return ema

def calculate_macd(prices):
    if len(prices) < 26:
        return None, None
    ema12 = exponential_moving_average(prices, 12)
    ema26 = exponential_moving_average(prices, 26)
    if ema12 is None or ema26 is None:
        return None, None
    min_len = min(len(ema12), len(ema26))
    ema12 = ema12[-min_len:]
    ema26 = ema26[-min_len:]
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal_line = exponential_moving_average(macd_line, 9)
    return round(macd_line[-1], 2), round(signal_line[-1], 2) if signal_line else 0.0

# Model predictions
def predict_signal(features_df):
    global model
    if not model:
        logger.error("❌ Model not loaded!")
        return None
    try:
        prediction = model.predict(features_df)
        return int(prediction[0])
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return None

def predict_tp(features):
    global tp_model
    if not tp_model:
        logger.error("❌ TP model not loaded!")
        return None
    try:
        return tp_model.predict(features)[0]
    except Exception as e:
        logger.error(f"❌ TP prediction error: {e}")
        return None

def predict_sl(features):
    global sl_model
    if not sl_model:
        logger.error("❌ SL model not loaded!")
        return None
    try:
        return sl_model.predict(features)[0]
    except Exception as e:
        logger.error(f"❌ SL prediction error: {e}")
        return None

# Generate AI comment
async def generate_ai_comment(coin_data):
    name = coin_data["symbol"].replace("USDT", "")
    price = coin_data["price"]
    df, change_24h, change_7d = await fetch_ohlc_data(name)
    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        logger.error(f"❌ Invalid data for {name}")
        return f"❌ Data unavailable for {name}. Please try again later."

    closes = df["price"].values
    if not isinstance(closes, np.ndarray) or len(closes) < 26:
        logger.error(f"❌ Insufficient data for {name}")
        return f"❌ Insufficient data for {name}. More historical data required."

    rsi = calculate_rsi(closes)
    macd, signal = calculate_macd(closes)
    ma_5 = sum(closes[-5:]) / 5
    ma_20 = sum(closes[-20:]) / 20
    volatility = df["price"].rolling(window=10).std().iloc[-1]
    momentum = df["price"].iloc[-1] - df["price"].shift(10).iloc[-1]
    price_change = df["price"].pct_change().iloc[-1]
    volume_change = df["price"].rolling(window=1).mean().pct_change().iloc[-1]

    features = pd.DataFrame([{
        "RSI": rsi if rsi is not None else 50.0,
        "MACD": macd if macd is not None else 0.0,
        "Signal": signal if signal is not None else 0.0,
        "MA_5": ma_5,
        "MA_20": ma_20,
        "Volatility": volatility if not np.isnan(volatility) else 0.0,
        "Momentum": momentum if not np.isnan(momentum) else 0.0,
        "Price_Change": price_change if not np.isnan(price_change) else 0.0,
        "Volume_Change": volume_change if not np.isnan(volume_change) else 0.0
    }])

    prediction = predict_signal(features)
    tp_raw = predict_tp(features) if predict_tp(features) is not None else 1.0
    sl_raw = predict_sl(features) if predict_sl(features) is not None else 2.0

    tp = None
    sl = None
    try:
        if prediction == 1:
            tp = price * (1 + max(0.01, tp_raw / 100))
            sl = price * (1 - max(0.02, sl_raw / 100))
        elif prediction == 0:
            tp = price * (1 - max(0.01, tp_raw / 100))
            sl = price * (1 + max(0.02, sl_raw / 100))

        min_tp_sl_diff = price * 0.02
        if tp and sl:
            if prediction == 0 and tp >= sl:
                tp = max(price - min_tp_sl_diff, price * 0.9)
            elif prediction == 1 and tp <= sl:
                tp = min(price + min_tp_sl_diff, price * 1.1)
            if abs(tp - sl) < min_tp_sl_diff:
                sl = tp + min_tp_sl_diff if prediction == 0 else tp - min_tp_sl_diff
    except Exception as e:
        logger.error(f"❌ Error calculating TP/SL: {e}")
        return f"❌ Error calculating TP/SL: {str(e)}"

    def generate_natural_comment():
        if rsi < 30:
            rsi_c = "RSI is in oversold territory, indicating potential upside."
        elif rsi > 70:
            rsi_c = "RSI is in overbought territory, indicating a correction risk."
        else:
            rsi_c = "RSI is balanced, showing a neutral trend."

        if macd > signal:
            macd_c = "MACD is above the signal line, indicating positive momentum."
        elif macd < signal:
            macd_c = "MACD is below the signal line, indicating a weak trend."
        else:
            macd_c = "MACD is close to the signal line, direction unclear."

        if ma_5 > ma_20:
            trend_c = "Short-term MA is above, suggesting a potential bullish trend."
        else:
            trend_c = "Short-term MA is below, indicating bearish pressure."

        return f"{rsi_c} {macd_c} {trend_c}"

    short_comment = generate_natural_comment()
    ai_signal = "⚠️ AI prediction failed." if prediction is None else ("📈 BUY" if prediction == 1 else "📉 SELL")
    tp_text = f"🎯 TP: ${tp:.2f}" if tp else "❌ TP prediction failed."
    sl_text = f"🛑 SL: ${sl:.2f}" if sl else "❌ SL prediction failed."

    leverage = "⚠️ Leverage not recommended"
    risk = "✅ Low Risk" if 30 < rsi < 70 and abs(macd - signal) > 0.05 and tp and sl and abs((tp - sl) / price) < 0.1 and volatility / price < 0.05 else "⚠️ High Risk"

    comment = (
        f"📊 {name} (${price:.2f})\n"
        f"24h: %{change_24h:.2f} | 7d: %{change_7d:.2f}\n\n"
        f"{ai_signal}\n"
        f"📉 RSI: {rsi:.2f} | 🧮 MACD: {macd:.2f}\n"
        f"📈 MA(5): {ma_5:.2f} | MA(20): {ma_20:.2f}\n\n"
        f"{tp_text}\n{sl_text}\n\n"
        f"{leverage}\n{risk}\n\n"
        f"🧠 AI Comment: {short_comment}"
    )
    return comment

# Portfolio and Alert functions
def get_portfolio(user_id):
    return {}  # Simulated

def add_coin(user_id, symbol, amount, buy_price=None):
    return True  # Simulated

def remove_coin(user_id, symbol):
    return True  # Simulated

def update_coin(user_id, symbol, amount):
    return True  # Simulated

def clear_portfolio(user_id):
    return True  # Simulated

def get_all_alerts():
    return []  # Simulated

def delete_alert(user_id, symbol):
    return True  # Simulated

# Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not await check_user_accepted(update, context):
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("✅ I Understand", callback_data="accept_disclaimer")]])
        disclaimer_text = (
            "📢 Warning\n\n"
            "Coinspace Bot provides market insights and AI\\-supported signals to help you make informed decisions in the crypto market\\.\n"
            "These signals are for informational purposes only and do not constitute financial advice\\.\n\n"
            "Please confirm to proceed\\."
        )
        await update.message.reply_text(disclaimer_text, reply_markup=keyboard, parse_mode="MarkdownV2")
    else:
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("📖 View Commands (/help)", callback_data="help")]])
        msg = (
            "👋 Welcome back to Coinspace Bot\\!\n\n"
            "🚀 Get daily AI\\-supported trading signals, price alerts, portfolio tracking, and live market updates\\.\n\n"
            "🔐 Upgrade to Premium:\n"
            "• Unlimited AI Leverage Signals \\(Free users get only 2 signals per day\\)\n"
            "• Full market overview access\n"
            "• Priority support and early feature access\n\n"
            "💳 Subscription Plans:\n"
            "• 1 Month: \\$29\\.99\n"
            "• 3 Months: \\$69\\.99\n"
            "• 1 Year: \\$399\\.99\n\n"
            "👉 To upgrade, select a plan and complete the payment:\n"
            "[1 Month Payment](https://nowpayments.io/payment/?iid\\=5260731771)\n"
            "[3 Months Payment](https://nowpayments.io/payment/?iid\\=4400895826)\n"
            "[1 Year Payment](https://nowpayments.io/payment/?iid\\=4501340550)\n\n"
            "✅ Activate your subscription with the ⁠/prem⁠ command\\.\n\n"
            "Click the button below to view available commands: /help"
        )
        await update.message.reply_text(msg, reply_markup=keyboard, parse_mode="MarkdownV2", disable_web_page_preview=True)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception while handling an update: {context.error}")
    if update and update.message:
        try:
            await update.message.reply_text("❌ An error occurred. Please try again or contact support.")
        except Exception:
            pass


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"help_command: Starting, user: {update.effective_user.id}")
    if not await check_user_accepted(update, context):
        logger.warning(f"help_command: User {update.effective_user.id} has not accepted terms.")
        if update.message:
            await update.message.reply_text("⚠️ Please run /start and accept the terms.")
        return

    if not update.message and not update.callback_query:
        logger.error(f"help_command error: No valid update message or callback query for user {update.effective_user.id}")
        return

    msg = (
    "*📚 Coinspace Commands*\n\n"
    "*💼 Portfolio*\n"
    "➕ `/add BTC 1 30000` — Add coin\n"
    "🔁 `/upd BTC 1.5` — Update amount\n"
    "🗑 `/rm BTC` — Remove coin\n"
    "🧹 `/clr` — Clear portfolio\n"
    "📊 `/port` — View portfolio\n"
    "📈 `/perf` — View performance\n"
    "📉 `/gr` — Portfolio graph\n\n"

    "*📌 Market Tools*\n"
    "💰 `/pr BTC` — Price info\n"
    "⏰ `/alert BTC 70K` — Price alert\n"
    "🧠 `/ai BTC` — AI comment\n"
    "🧪 `/bt BTC` — Backtest\n"
    "⚙️ `/lev` — Leverage signal\n\n"

    "*📰 News & Premium*\n"
    "🗞 `/nw` — News\n"
    "🔗 `/rmore` — Links\n"
    "💎 `/prem` — Premium"
)

    try:
        if update.message:
            await update.message.reply_text(msg, parse_mode="MarkdownV2")
        elif update.callback_query:
            await update.callback_query.message.edit_text(msg, parse_mode="MarkdownV2")
        logger.info(f"help_command: Response sent, user: {update.effective_user.id}")
    except Exception as e:
        logger.error(f"help_command error: {e}")

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
        await update.message.reply_text("📭 No coins added yet. Use /add.")
        return
    symbols = [sym.upper() for sym in holdings.keys() if sym.upper() in symbol_to_id_map]
    total_value = 0
    msg = "📊 Portfolio:\n"
    for symbol in symbols:
        price = await fetch_price(symbol)
        amount = holdings.get(symbol.lower(), {}).get("amount", 0)
        if price:
            value = price * amount
            total_value += value
            msg += f"• {symbol}: {amount} × ${price:.2f} = ${value:.2f}\n"
        else:
            msg += f"• {symbol}: Price not available\n"
    msg += f"\n💰 Total Value: ${total_value:.2f}"
    await update.message.reply_text(msg)

async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) not in [2, 3]:
        await update.message.reply_text("❌ Usage: /add <coin> <amount> [buy_price] (e.g., /add BTC 0.5 30000)")
        return
    symbol = context.args[0].upper()
    try:
        amount = float(context.args[1])
        buy_price = float(context.args[2]) if len(context.args) == 3 else None
    except ValueError:
        await update.message.reply_text("❌ Invalid amount or price.")
        return
    user_id = update.effective_user.id
    add_coin(user_id, symbol, amount, buy_price)
    msg = f"✅ {amount} {symbol} added to portfolio."
    if buy_price:
        msg += f" Buy price: ${buy_price}"
    await update.message.reply_text(msg)

async def rm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("❌ Usage: /rm <coin> (e.g., /rm BTC)")
        return
    symbol = context.args[0].upper()
    user_id = update.effective_user.id
    success = remove_coin(user_id, symbol)
    if success:
        await update.message.reply_text(f"🗑️ {symbol} removed from portfolio.")
    else:
        await update.message.reply_text(f"⚠️ {symbol} not found in portfolio.")

async def upd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.message.reply_text("❌ Usage: /upd <coin> <amount> (e.g., /upd BTC 1.0)")
        return
    symbol = context.args[0].upper()
    try:
        amount = float(context.args[1])
    except ValueError:
        await update.message.reply_text("❌ Invalid amount.")
        return
    user_id = update.effective_user.id
    success = update_coin(user_id, symbol, amount)
    if success:
        await update.message.reply_text(f"🔗 {symbol} quantity updated to {amount}.")
    else:
        await update.message.reply_text(f"⚠️ {symbol} not found in portfolio, add it with /add first.")

async def clr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    success = clear_portfolio(user_id)
    if success:
        await update.message.reply_text("🧼 Portfolio successfully cleared.")
    else:
        await update.message.reply_text("❗ No data to clear.")

async def gr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    holdings = get_portfolio(user_id)
    if not holdings:
        await update.message.reply_text("📭 Portfolio is empty. Add a coin with /add first.")
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
        await update.message.reply_text("⚠️ Price data not available.")
        return
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    ax1.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    ax1.set_title("📈 Portfolio Distribution")
    df = await fetch_ohlc_data("ETH")[0]
    if df is not None and not df.empty and isinstance(df, pd.DataFrame):
        closes = df["price"].values
        rsi = [calculate_rsi(closes[:i+1]) for i in range(len(closes)) if i >= 14]
        macd, signal = zip(*[calculate_macd(closes[:i+1]) for i in range(len(closes)) if calculate_macd(closes[:i+1])[0] is not None])
        timestamps = range(len(rsi))
        ax2.plot(timestamps, rsi[-len(closes)+14:], label="RSI", color="purple")
        ax2.axhline(y=70, color="orange", linestyle="--", label="Overbought (70)")
        ax2.axhline(y=30, color="orange", linestyle="--", label="Oversold (30)")
        ax2.set_title("ETH RSI")
        ax2.set_xlabel("Time (Hours)")
        ax2.set_ylabel("RSI Value")
        ax2.legend()
        ax3.plot(timestamps, macd[-len(closes)+26:], label="MACD", color="green")
        ax3.plot(timestamps, signal[-len(closes)+26:], label="Signal", color="red", linestyle="--")
        ax3.set_title("ETH MACD")
        ax3.set_xlabel("Time (Hours)")
        ax3.set_ylabel("MACD Value")
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
        await update.message.reply_text("📭 Portfolio is empty.")
        return
    symbols = [sym.upper() for sym in portfolio.keys() if sym.upper() in symbol_to_id_map]
    msg, total_pl = "📈 Portfolio Performance:\n", 0
    for symbol in symbols:
        current_price = await fetch_price(symbol)
        amount = portfolio.get(symbol.lower(), {}).get("amount", 0)
        buy_price = portfolio.get(symbol.lower(), {}).get("buy_price")
        if not current_price:
            msg += f"• {symbol}: Price not available\n"
            continue
        if buy_price:
            current_value, cost_basis = current_price * amount, buy_price * amount
            pl = current_value - cost_basis
            total_pl += pl
            msg += f"• {symbol}: Buy ${buy_price:.2f} → Current ${current_price:.2f} | P/L: ${pl:.2f}\n"
        else:
            msg += f"• {symbol}: Buy price unknown\n"
    msg += f"\n💼 Total P/L: ${total_pl:.2f}"
    await update.message.reply_text(msg)

async def alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.message.reply_text("Usage: /alert <coin> <price> (e.g., /alert BTC 70000)")
        return
    symbol = context.args[0].upper()
    try:
        target_price = float(context.args[1])
    except ValueError:
        await update.message.reply_text("❌ Invalid price.")
        return
    if symbol not in symbol_to_id_map:
        await update.message.reply_text("❌ Invalid coin symbol.")
        return
    user_id = update.effective_user.id
    add_alert(user_id, symbol, target_price)
    await update.message.reply_text(f"🔔 Alert set for {symbol} at ${target_price}.")

async def check_alerts(app):
    while True:
        alerts = get_all_alerts() or []
        if not alerts:
            await asyncio.sleep(300)
            continue
        valid_alerts = [alert for alert in alerts if all(k in alert for k in ("symbol", "target_price", "user_id"))]
        symbol_map = {alert["symbol"].lower(): alerts_list for alert in valid_alerts for alerts_list in [symbol_map.get(alert["symbol"].lower(), []) + [alert]]}
        for symbol, alerts_list in symbol_map.items():
            price = await fetch_price(symbol.upper())
            if price is None:
                continue
            for alert in alerts_list:
                if price >= alert["target_price"]:
                    try:
                        await app.bot.send_message(chat_id=alert["user_id"], text=f"📢 *{symbol.upper()}* reached ${alert['target_price']}!\nCurrent price: ${price:.2f}", parse_mode="Markdown")
                        delete_alert(alert["user_id"], symbol.upper())
                    except Exception as e:
                        logger.error(f"❌ Notification failed: {e}")
        await asyncio.sleep(300)

async def fetch_newsapi_news():
    url = f"https://newsapi.org/v2/top-headlines?category=business&q=crypto&apiKey={NEWS_API_KEY}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            logger.info(f"🌐 NewsAPI status: {response.status}")
            if response.status == 200:
                return await response.json()
            return None

async def summarize_news(title, description):
    prompt = f"Write a short summary of the following news:\n\nTitle: {title}\nDescription: {description}\n\nPrepare a concise summary for investors."
    try:
        response = await client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}], max_tokens=100, temperature=0.7)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ Summary generation error: {e}")
        return "⚠️ Summary generation failed."

async def nw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("🚀 /news command triggered")
    news_data = await fetch_newsapi_news()
    if not news_data or "articles" not in news_data:
        await update.message.reply_text("❌ News data not available.")
        return
    sent_count = 0
    for article in news_data["articles"][:5]:
        url = article.get("url")
        title = article.get("title", "No Title")
        description = article.get("description", "No Description")
        norm_url = normalize_url(url)
        if norm_url and norm_url not in sent_news_urls:
            summary = await summarize_news(title, description)
            text = f"📰 <b>{escape(title)}</b>\n{escape(summary)}\n<a href=\"{url}\">🔗 Read More</a>"
            try:
                await update.message.reply_text(text, parse_mode="HTML")
                sent_news_urls.add(norm_url)
                save_sent_urls()
                sent_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Sending error: {e}")
    if sent_count == 0:
        await update.message.reply_text("⚠️ No new news available.")

async def rmore(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🧭 View news links with the /nw command.")

async def bt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /bt <coin> (e.g., /bt BTC)")
        return
    symbol = context.args[0].upper()
    if symbol not in symbol_to_id_map:
        await update.message.reply_text("❌ Coin not found.")
        return
    df, _, _ = await fetch_ohlc_data(symbol, days=30)
    if df is None or df.empty:
        await update.message.reply_text("❌ Data not available.")
        return
    from ta.momentum import RSIIndicator
    from ta.trend import SMAIndicator
    df["RSI"] = RSIIndicator(df["price"]).rsi()
    df["MA"] = SMAIndicator(df["price"], window=14).sma_indicator()
    buy_points, sell_points, position, entry_price, pnl = [], [], None, 0, 0
    for i in range(1, len(df)):
        rsi, price, ma = df["RSI"].iloc[i], df["price"].iloc[i], df["MA"].iloc[i]
        if rsi < 30 and price > ma and not position:
            entry_price, position = price, "LONG"
            buy_points.append((df.index[i], price))
        elif rsi > 70 and position == "LONG":
            pnl += price - entry_price
            sell_points.append((df.index[i], price))
            position = None
    msg = f"📈 {symbol} RSI + MA Backtest Result (30 days):\n✅ Buy count: {len(buy_points)}\n💰 Total Profit: ${pnl:.2f}"
    await update.message.reply_text(msg)

async def prem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🟢 1 Month – $29.99", url="https://nowpayments.io/payment/?iid=5260731771")],
        [InlineKeyboardButton("🔵 3 Months – $69.99", url="https://nowpayments.io/payment/?iid=4400895826")],
        [InlineKeyboardButton("🟣 1 Year – $399.99", url="https://nowpayments.io/payment/?iid=4501340550")],
    ])
    msg = (
        "✨ *Upgrade to Coinspace Premium!*\n\n"
        "🚀 Benefits:\n"
        "• Up to 10 AI-based trading signals per day\n"
        "• Leverage trading suggestions\n"
        "• Priority market alerts and news\n"
        "• Portfolio analysis tools\n\n"
        "💰 *Pricing:*\n"
        "1 Month: $29.99\n"
        "3 Months: $69.99\n"
        "1 Year: $399.99\n\n"
        "_Secure payment via USDT (TRC20) through NOWPayments._"
    )
    await update.message.reply_text(msg, reply_markup=keyboard, parse_mode="MarkdownV2")

async def accept_disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    accepted_users.add(user_id)
    save_accepted_users(accepted_users)
    await query.answer()
    await query.edit_message_text("✅ Terms accepted. You can start using commands.")

async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    callback_data = query.data
    logger.debug(f"[DEBUG] Feedback received: {callback_data} from user {query.from_user.id}")
    if not callback_data.startswith("feedback:"):
        return
    feedback = callback_data.split(":")[1]
    message_id = query.message.message_id
    user_id = query.from_user.id
    signals = load_signals()
    found = False
    for s in signals:
        if s.get("message_id") == message_id:
            if feedback == "like":
                if user_id not in s.setdefault("likes", []):
                    s["likes"].append(user_id)
                if user_id in s.get("dislikes", []):
                    s["dislikes"].remove(user_id)
            elif feedback == "dislike":
                if user_id not in s.setdefault("dislikes", []):
                    s["dislikes"].append(user_id)
                if user_id in s.get("likes", []):
                    s["likes"].remove(user_id)
            found = True
            break
    if not found:
        signals.append({"message_id": message_id, "likes": [user_id] if feedback == "like" else [], "dislikes": [user_id] if feedback == "dislike" else []})
    save_signals(signals)
    await query.edit_message_reply_markup(reply_markup=None)
    await query.message.reply_text("✅ Thank you for your feedback.")

async def send_ai_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, signal_text: str):
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("👍", callback_data="feedback:like"), InlineKeyboardButton("👎", callback_data="feedback:dislike")]])
    message = await context.bot.send_message(chat_id=update.effective_chat.id, text=signal_text, reply_markup=keyboard)
    signals = load_signals()
    signals.append({"message_id": message.message_id, "text": signal_text, "likes": [], "dislikes": []})
    save_signals(signals)

async def check_and_send_news(app):
    while True:
        news_data = await fetch_newsapi_news()
        if news_data and "articles" in news_data:
            for article in news_data["articles"]:
                url, title, description = article.get("url"), article.get("title", "No Title"), article.get("description", "No Description")
                news_key = get_news_key(url, title)
                if news_key not in sent_news_urls:
                    summary = await summarize_news(title, description)
                    text = f"📰 <b>{escape(title)}</b>\n{escape(summary)}\n<a href=\"{url}\">🔗 Read More</a>"
                    try:
                        await app.bot.send_message(chat_id=os.getenv("OWNER_CHAT_ID", "dummy_owner_id"), text=text, parse_mode="HTML", disable_web_page_preview=True)
                        sent_news_urls.add(news_key)
                        save_sent_urls()
                    except Exception as e:
                        logger.error(f"⚠️ News sending failed: {e}")
        await asyncio.sleep(7200)

async def fetch_coin_data(symbol):
    price = await fetch_price(symbol)
    return {"symbol": f"{symbol.upper()}USDT", "price": price} if price is not None else None

async def ai_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower()
    if not text.startswith("/ai"):
        return
    symbol = text.replace("/ai", "").strip().upper()
    logger.info(f"ai_comment: Processing request for symbol {symbol}")
    if not symbol or symbol not in symbol_to_id_map:
        await update.message.reply_text("❌ Invalid coin symbol. Please use a valid coin traded on Binance (e.g., /ai BTC, /ai ETH).")
        return
    await update.message.reply_text("💬 Preparing AI comment...")
    try:
        price_data = await fetch_price(symbol)
        if price_data is None:
            await update.message.reply_text(f"❌ Failed to retrieve price data for {symbol}. Please try again.")
            return
        coin_data = {"symbol": f"{symbol}USDT", "price": price_data}
        comment = await generate_ai_comment(coin_data)
        await send_ai_signal(update, context, comment)
    except Exception as e:
        await update.message.reply_text(f"❌ Error occurred during processing: {str(e)}")
        logger.error(f"ai_comment error: {e}, symbol={symbol}, coin_data={coin_data}")

async def run_bot():
    logger.info("🚀 Bot starting...")
    app = ApplicationBuilder().token(TOKEN).build()
    logger.info("✅ Telegram bot application created.")
    await load_symbol_map()
    logger.info("✅ Coin symbols loaded.")
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
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(CallbackQueryHandler(feedback_handler))
    app.add_error_handler(error_handler)
    asyncio.create_task(check_alerts(app))
    asyncio.create_task(check_and_send_news(app))
    logger.info("🔄 Background tasks started.")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    logger.info("✅ Bot started.")

if __name__ == "__main__":
    asyncio.run(run_bot())