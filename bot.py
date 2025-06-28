import os
import asyncio
import aiohttp
from telegram import Update, InputFile, InlineKeyboardMarkup, InlineKeyboardButton, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
import matplotlib.pyplot as plt
import joblib
import pickle
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import io
import json
import logging
from dotenv import load_dotenv
from functools import wraps
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse
import hashlib
import pandas as pd
from ta import add_all_ta_features
from binance.client import Client

# Load models
model = joblib.load("model.pkl")
tp_model = joblib.load("tp_model.pkl")
sl_model = joblib.load("sl_model.pkl")
with open("features_list.pkl", "rb") as f:
    feature_list = pickle.load(f)

# Load environment variables
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN", "dummy_token")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "506d562e0d4a434c97df2e3a51e4cd1c")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0"))

# Logging configuration
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY} if BINANCE_API_KEY else {}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                symbol_to_id_map.update({symbol["symbol"].replace("USDT", "").upper(): symbol["symbol"] for symbol in data["symbols"] if symbol["status"] == "TRADING" and symbol["symbol"].endswith("USDT")})
                additional_coins = {"BNB", "ADA", "XRP", "DOT", "LINK"}
                for coin in additional_coins:
                    if any(symbol["symbol"] == f"{coin}USDT" for symbol in data["symbols"] if symbol["status"] == "TRADING"):
                        symbol_to_id_map[coin.upper()] = f"{coin}USDT"
                logger.info(f"✅ Coin symbols loaded with additional coins: {list(symbol_to_id_map.keys())}")
            else:
                logger.error(f"❌ Failed to fetch coin list: status={response.status}")
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
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY} if BINANCE_API_KEY else {}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return float(data["price"])
            logger.error(f"❌ Failed to fetch price for {full_symbol}: status={response.status}")
            return None

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

async def fetch_ohlc_data(symbol: str, days=7):
    full_symbol = symbol_to_id_map.get(symbol.upper(), f"{symbol.upper()}USDT")
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": full_symbol, "interval": "1h", "limit": days * 24}
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY} if BINANCE_API_KEY else {}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if not data or not isinstance(data, list) or len(data) == 0:
                    logger.error(f"❌ Empty or invalid OHLC data for {full_symbol}")
                    return None, 0.0, 0.0
                try:
                    df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QuoteVolume", "Trades", "TakerBuyBase", "TakerBuyQuote", "Ignore"])
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
                    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                    change_24h = ((df["Close"].iloc[-1] - df["Close"].iloc[-24]) / df["Close"].iloc[-24]) * 100 if len(df) >= 24 else 0.0
                    change_7d = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
                    return df, change_24h, change_7d
                except Exception as e:
                    logger.error(f"❌ Error processing OHLC data for {full_symbol}: {e}")
                    return None, 0.0, 0.0
            logger.error(f"❌ Failed to fetch OHLC data for {full_symbol}: status={response.status}")
            return None, 0.0, 0.0

def get_portfolio(user_id):
    return {}

def add_coin(user_id, symbol, amount, buy_price=None):
    return True

def remove_coin(user_id, symbol):
    return True

def update_coin(user_id, symbol, amount):
    return True

def clear_portfolio(user_id):
    return True

def get_all_alerts():
    return []

def delete_alert(user_id, symbol):
    return True

def add_alert(user_id, symbol, target_price):
    return True

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    save_user(user_id)
    update_user_metadata(user_id)

    message = update.message or (update.callback_query and update.callback_query.message)

    if not await check_user_accepted(update, context):
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("✅ I Understand", callback_data="accept_disclaimer")]])
        disclaimer_text = (
            "📢 Warning\n\n"
            "Coinspace Bot provides market insights and AI\\-supported signals to help you make informed decisions in the crypto market\\.\n"
            "These signals are for informational purposes only and do not constitute financial advice\\.\n\n"
            "Please confirm to proceed\\."
        )
        if message:
            await message.reply_text(disclaimer_text, reply_markup=keyboard, parse_mode="MarkdownV2")
        return

    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("📖 View Commands (/help)", callback_data="help")]])
    msg = (
        "<b>👋 Welcome back to Coinspace Bot!</b>\n\n"
        f"<b>🚀 Your User ID:</b> <code>{user_id}</code>\n\n"
        "<b>🚀 Get daily AI-supported trading signals, price alerts, portfolio tracking, and live market updates.</b>\n\n"
        "<b>🔐 Upgrade to Premium:</b>\n"
        "• Unlimited AI Leverage Signals (Free users get only 2 signals per day)\n"
        "• Full market overview access\n"
        "• Priority support and early feature access\n\n"
        "<b>💳 Subscription Plans:</b>\n"
        "• 1 Month: $29.99\n"
        "• 3 Months: $69.99\n"
        "• 1 Year: $399.99\n\n"
        "<b>👉 To upgrade, select a plan and complete the payment:</b>\n"
        "🔗 <a href='https://nowpayments.io/payment/?iid=5260731771'>1 Month Payment</a>\n"
        "🔗 <a href='https://nowpayments.io/payment/?iid=4400895826'>3 Months Payment</a>\n"
        "🔗 <a href='https://nowpayments.io/payment/?iid=4501340550'>1 Year Payment</a>\n\n"
        "✅ After payment, activate your subscription with the <b>/activate_premium</b> command.\n\n"
        "👇 Click the button below to view available commands:"
    )

    if message:
        await message.reply_text(msg, reply_markup=keyboard, parse_mode="HTML", disable_web_page_preview=True)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception while handling an update: {context.error}")
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text("❌ An error occurred. Please try again or contact support.")
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
        "*🤖 Coinspace Commands*\n\n"
        "📊 *Portfolio*\n"
        "• `/add BTC 0\\.5 30000` \\- Add coin\n"
        "• `/upd BTC 1\\.0` \\- Update coin\n"
        "• `/rm BTC` \\- Remove coin\n"
        "• `/clr` \\- Clear portfolio\n"
        "• `/port` \\- View portfolio\n"
        "• `/perf` \\- View performance\n"
        "• `/gr` \\- View graph\n\n"
        "💹 *Market Tools*\n"
        "• `/pr BTC` \\- Check price\n"
        "• `/alert BTC 70000` \\- Set alert\n"
        "• `/ai BTC` \\- AI signal \\(Premium\\)\n"
        "• `/bt BTC` \\- Backtest\n\n"
        "📰 *News & Premium*\n"
        "• `/nw` \\- Latest news\n"
        "• `/nmore` \\- More news\n"
        "• `/prem` \\- Upgrade to Premium\n"
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

def prepare_features(df):
    df["rsi"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["macd"] = MACD(close=df["Close"]).macd_diff()
    df["sma_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["atr"] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"]).average_true_range()
    df.dropna(inplace=True)
    return df[feature_list].copy()

def generate_features_from_live(symbol="BTCUSDT", interval="1h", lookback=100):
    # Binance API bağlantısı
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume", 
        "close_time", "quote_asset_volume", "number_of_trades", 
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)

    # Teknik göstergeler
    df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)

    # Son satırın özellikleri çıkarılır
    with open("features_list.pkl", "rb") as f:
        features = pickle.load(f)
    input_data = df.iloc[-1:][features]
    return input_data

async def ai_signal_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        symbol = context.args[0].upper() + "USDT"
    except IndexError:
        await update.message.reply_text("❌ Lütfen bir coin sembolü girin. Örnek: /ai BTC")
        return

    try:
        features = generate_features_from_live(symbol=symbol)
        model = joblib.load("model.pkl")
        tp_model = joblib.load("tp_model.pkl")
        sl_model = joblib.load("sl_model.pkl")

        prediction = model.predict(features)[0]
        tp_pct = tp_model.predict(features)[0]
        sl_pct = sl_model.predict(features)[0]

        latest_price = features["close"].values[0]
        tp_price = round(latest_price * (1 + tp_pct), 4)
        sl_price = round(latest_price * (1 - sl_pct), 4)

        comment = "This is a basic AI signal based on live data."
        emoji = "📈" if prediction == 1 else "📉"

        text = f"""
📊 {symbol.replace("USDT", "")} (${latest_price})
🤖 AI Signal: {emoji} {'BUY' if prediction == 1 else 'SELL' if prediction == -1 else 'HOLD'}
🎯 TP: ${tp_price}
🛑 SL: ${sl_price}
💬 {comment}
"""
        await update.message.reply_text(text.strip())
    except Exception as e:
        await update.message.reply_text(f"⚠️ Hata oluştu: {str(e)}")

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
    if not check_premium_status(user_id):
        await update.message.reply_text("❌ This command (/gr) is only available for Premium users. Upgrade via /prem.")
        return
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
        closes = df["Close"].values
        rsi = [RSIIndicator(close=pd.Series(closes[:i+1])).rsi()[-1] for i in range(len(closes)) if i >= 14]
        macd, signal = zip(*[(MACD(close=pd.Series(closes[:i+1])).macd()[-1], MACD(close=pd.Series(closes[:i+1])).macd_signal()[-1]) for i in range(len(closes)) if i >= 26])
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
    if not check_premium_status(user_id):
        await update.message.reply_text("❌ This command (/alert) is only available for Premium users. Upgrade via /prem.")
        return
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
    user_id = update.effective_user.id
    if not check_premium_status(user_id):
        await update.message.reply_text("❌ This command (/bt) is only available for Premium users. Upgrade via /prem.")
        return
    df, _, _ = await fetch_ohlc_data(symbol, days=30)
    if df is None or df.empty:
        await update.message.reply_text("❌ Data not available.")
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
    msg = f"📈 {symbol} RSI + MA Backtest Result (30 days):\n✅ Buy count: {len(buy_points)}\n💰 Total Profit: ${pnl:.2f}"
    await update.message.reply_text(msg)

async def prem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🟢 1 Month – $29.99", url="https://nowpayments.io/payment/?iid=5260731771")],
        [InlineKeyboardButton("🔵 3 Months – $69.99", url="https://nowpayments.io/payment/?iid=4400895826")],
        [InlineKeyboardButton("🟣 1 Year – $399.99", url="https://nowpayments.io/payment/?iid=4501340550")],
    ])
    msg = (
        "👑 <b>Coinspace Premium Plans!</b>\n\n"
        "⚡️ <b>Benefits:</b>\n"
        "• Unlimited AI Leverage Signals (Free users get only 2 signals per day)\n"
        "• Full market overview access\n"
        "• Priority support & early feature access\n\n"
        "💳 <b>Plans:</b>\n"
        "1 Month: $29.99\n"
        "3 Months: $69.99\n"
        "1 Year: $399.99\n\n"
        "👉 <b>To upgrade, select a plan and complete the payment:</b>\n"
        "• <a href='https://nowpayments.io/payment/?iid=5260731771'>1 Month Payment</a>\n"
        "• <a href='https://nowpayments.io/payment/?iid=4400895826'>3 Months Payment</a>\n"
        "• <a href='https://nowpayments.io/payment/?iid=4501340550'>1 Year Payment</a>\n\n"
        "✅ After payment, activate your subscription with the /activate_premium command."
    )
    await update.message.reply_text(msg, parse_mode="HTML", reply_markup=keyboard, disable_web_page_preview=True)

async def activate_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("❌ Usage: /activate_premium <payment_id> (e.g., /activate_premium 5260731771)")
        return
    payment_id = context.args[0]
    user_id = str(update.effective_user.id)
    valid_payments = {
        "5260731771": 30,  # 1 ay
        "4400895826": 90,  # 3 ay
        "4501340550": 365  # 1 yıl
    }
    if payment_id not in valid_payments:
        await update.message.reply_text("❌ Invalid payment ID. Please contact the bot owner for assistance.")
        return
    premium_users = load_premium_users()
    today = datetime.today().date()
    end_date = today + timedelta(days=valid_payments[payment_id])
    premium_users[user_id] = {"start": str(today), "end": str(end_date)}
    save_premium_users(premium_users)
    await update.message.reply_text(f"✅ Your Premium subscription has been activated until {end_date}!")

async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.message.reply_text("❌ You do not have admin privileges.")
        return
    msg = (
        "<b>🔧 Admin Panel</b>\n\n"
        "• <code>/broadcast</code> – Send a message to all users\n"
        "• <code>/users</code> – Show total registered users\n"
        "• <code>/make_admin [user_id]</code> – Grant admin rights\n"
        "• <code>/remove_admin [user_id]</code> – Revoke admin rights\n"
        "• <code>/make_premium [user_id]</code> – Grant premium status\n"
        "• <code>/remove_premium [user_id]</code> – Revoke premium status\n"
        "• <code>/admin_list</code> – List admin users\n"
        "• <code>/premium_list</code> – List premium users"
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def make_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.message.reply_text("❌ You do not have admin privileges.")
        return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("❌ Usage: /make_admin <user_id>")
        return
    target_user_id = str(context.args[0])
    admin_users = load_admin_users()
    if target_user_id not in admin_users:
        admin_users.add(target_user_id)
        save_admin_users(admin_users)
        await update.message.reply_text(f"✅ User {target_user_id} has been made an admin.")
    else:
        await update.message.reply_text(f"⚠️ User {target_user_id} is already an admin.")

async def remove_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.message.reply_text("❌ You do not have admin privileges.")
        return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("❌ Usage: /remove_admin <user_id>")
        return
    target_user_id = str(context.args[0])
    admin_users = load_admin_users()
    if target_user_id in admin_users:
        admin_users.remove(target_user_id)
        save_admin_users(admin_users)
        await update.message.reply_text(f"✅ Admin status removed from user {target_user_id}.")
    else:
        await update.message.reply_text(f"⚠️ User {target_user_id} is not an admin.")

async def make_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.message.reply_text("❌ You do not have admin privileges.")
        return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("❌ Usage: /make_premium <user_id>")
        return
    target_user_id = str(context.args[0])
    premium_users = load_premium_users()
    if target_user_id in premium_users:
        await update.message.reply_text(f"⚠️ User {target_user_id} is already a Premium user.")
        return
    today = datetime.today().date()
    end_date = today + timedelta(days=30)
    premium_users[target_user_id] = {"start": str(today), "end": str(end_date)}
    save_premium_users(premium_users)
    await update.message.reply_text(f"✅ User {target_user_id} has been granted Premium access until {end_date}.")

async def remove_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.message.reply_text("❌ You do not have admin privileges.")
        return
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("❌ Usage: /remove_premium <user_id>")
        return
    target_user_id = str(context.args[0])
    premium_users = load_premium_users()
    if target_user_id in premium_users:
        del premium_users[target_user_id]
        save_premium_users(premium_users)
        await update.message.reply_text(f"✅ Premium status removed from user {target_user_id}.")
    else:
        await update.message.reply_text(f"⚠️ User {target_user_id} is not a Premium user.")

def admin_required(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if not check_admin(user_id):
            await update.message.reply_text("❌ You do not have admin privileges.")
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
            f"👥 <b>Total Users:</b> {total}\n"
            f"💎 <b>Premium Users:</b> {premium}\n"
            f"🛡️ <b>Admins:</b> {admins}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")
    except Exception as e:
        logger.error(f"/users error: {e}")
        await update.message.reply_text("❌ Failed to load user data.")

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
        await update.message.reply_text("⚠️ Please provide a user ID. Usage: /user_info <user_id>")
        return
    user_id = context.args[0]
    users = load_json("data/users.json")
    if user_id not in users:
        await update.message.reply_text("❌ User not found.")
        return
    user_data = users[user_id]
    premium = "✅ Yes" if check_premium_status(int(user_id)) else "❌ No"
    admin = "✅ Yes" if check_admin(int(user_id)) else "❌ No"
    created_at = user_data.get("created_at", "N/A")
    last_active = user_data.get("last_active", "N/A")
    msg = (
        f"👤 <b>User ID:</b> <code>{user_id}</code>\n"
        f"💎 <b>Premium:</b> {premium}\n"
        f"🛠 <b>Admin:</b> {admin}\n"
        f"📆 <b>Joined:</b> {created_at}\n"
        f"🕒 <b>Last Active:</b> {last_active}"
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
            text="⚠️ Your premium subscription has expired. Use /premium to renew it."
        )
    except Exception as e:
        logger.error(f"Error while notifying user {user_id}: {e}")

async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("📢 Please provide a message to broadcast:\n\nUsage:\n`/broadcast Your message here`", parse_mode="Markdown")
        return
    message_text = "📢 Broadcast:\n" + " ".join(context.args)
    try:
        user_data = load_json("data/users.json")
    except FileNotFoundError:
        await update.message.reply_text("⚠️ No users found.")
        return
    count = 0
    for user_id in user_data.keys():
        try:
            await context.bot.send_message(chat_id=int(user_id), text=message_text)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to send message to {user_id}: {e}")
    await update.message.reply_text(f"✅ Broadcast sent to {count} users.")

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
    await query.edit_message_text("✅ Terms accepted. Welcome!")
    await start(update, context)

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
            found = True
            if user_id in s.get("likes", []) or user_id in s.get("dislikes", []):
                logger.warning(f"feedback_handler: User {user_id} already provided feedback for message_id {message_id}")
                await query.answer("❌ You have already provided feedback for this message.", show_alert=True)
                return
            if feedback == "like":
                s.setdefault("likes", []).append(user_id)
                logger.info(f"feedback_handler: User {user_id} liked message_id {message_id}")
            elif feedback == "dislike":
                s.setdefault("dislikes", []).append(user_id)
                logger.info(f"feedback_handler: User {user_id} disliked message_id {message_id}")
            break
    if not found:
        signals.append({"message_id": message_id, "text": query.message.text, "likes": [user_id] if feedback == "like" else [], "dislikes": [user_id] if feedback == "dislike" else []})
        logger.info(f"feedback_handler: New signal entry created for message_id {message_id} with feedback {feedback}")
    save_signals(signals)
    await query.edit_message_reply_markup(reply_markup=None)
    await query.message.reply_text("✅ Thank you for your feedback!")

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
                        await app.bot.send_message(chat_id=OWNER_CHAT_ID, text=text, parse_mode="HTML", disable_web_page_preview=True)
                        sent_news_urls.add(news_key)
                        save_sent_urls()
                    except Exception as e:
                        logger.error(f"⚠️ News sending failed: {e}")
        await asyncio.sleep(1800)

async def premium_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not check_admin(user_id):
        await update.message.reply_text("❌ You do not have admin privileges.")
        return
    premium_users = load_premium_users()
    if not premium_users:
        await update.message.reply_text("❌ No premium users found.")
        return
    msg = "<b>💎 Premium Users List</b>\n\n"
    for uid, info in premium_users.items():
        msg += f"• <code>{uid}</code> – Valid until: <b>{info.get('end', 'N/A')}</b>\n"
    await update.message.reply_text(msg, parse_mode="HTML")

async def admin_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    admins = load_admin_users()
    if str(user_id) not in admins:
        await update.message.reply_text("❌ You do not have admin privileges.")
        return
    if not admins:
        await update.message.reply_text("⚠️ No admins found.")
        return
    msg = "<b>👮 Admin Users:</b>\n"
    for admin_id in admins:
        try:
            user = await context.bot.get_chat(int(admin_id))
            name = user.full_name or "Unknown"
            msg += f"• {name} (<code>{admin_id}</code>)\n"
        except Exception:
            msg += f"• Unknown (<code>{admin_id}</code>)\n"
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
    logger.info("🔄 Background tasks started.")
    while True:
        await check_premium_expiry(bot)
        await asyncio.sleep(3600)

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
    app.add_handler(CommandHandler("ai", ai_signal_handler)) 
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(CallbackQueryHandler(feedback_handler))
    app.add_error_handler(error_handler)
    asyncio.create_task(check_alerts(app))
    asyncio.create_task(check_and_send_news(app))
    asyncio.create_task(background_tasks(app.bot))
    logger.info("🔄 Background tasks started.")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    logger.info("✅ Bot started.")

if __name__ == "__main__":
    asyncio.run(run_bot())