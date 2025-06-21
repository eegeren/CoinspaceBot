import os
import asyncio
import aiohttp
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from portfolio_db import add_coin, get_portfolio, remove_coin, update_coin, clear_portfolio
from alert_db import add_alert, get_all_alerts, delete_alert
from telegram.constants import ParseMode 
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import time 
import io
import hashlib
from html import escape
from openai import AsyncOpenAI
import pandas as pd
import ta

#BACKTEST **********************************

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import pandas as pd
from datetime import datetime, timedelta

async def fetch_ohlc_data(coin_id, days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            prices = data.get("prices", [])
            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df

async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Kullanım: /backtest BTC")
        return

    symbol = context.args[0].upper()
    coin_id = symbol_to_id_map.get(symbol)
    if not coin_id:
        await update.message.reply_text("❌ Coin bulunamadı.")
        return

    df = await fetch_ohlc_data(coin_id)
    if df is None or df.empty:
        await update.message.reply_text("❌ Veri alınamadı.")
        return

    df["RSI"] = RSIIndicator(df["price"]).rsi()
    df["MA"] = SMAIndicator(df["price"], window=14).sma_indicator()

    buy_points = []
    sell_points = []
    position = None
    entry_price = 0
    pnl = 0

    for i in range(1, len(df)):
        rsi = df["RSI"].iloc[i]
        price = df["price"].iloc[i]
        ma = df["MA"].iloc[i]

        if rsi < 30 and price > ma and not position:
            entry_price = price
            position = "LONG"
            buy_points.append((df.index[i], price))

        elif rsi > 70 and position == "LONG":
            pnl += price - entry_price
            sell_points.append((df.index[i], price))
            position = None

    msg = f"📈 {symbol} için RSI + MA Backtest Sonucu (30 gün):\n"
    msg += f"✅ Alım sayısı: {len(buy_points)}\n"
    msg += f"💰 Toplam Kar: ${pnl:.2f}"

    await update.message.reply_text(msg)



# *******************************************************

import json
from urllib.parse import urlparse, urlunparse

SENT_NEWS_FILE = "sent_news.json"



def normalize_url(raw_url):
    if not raw_url:
        return ""
    parsed = urlparse(raw_url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

def save_sent_urls():
    with open(SENT_NEWS_FILE, "w") as f:
        json.dump(list(sent_news_urls), f)

if os.path.exists(SENT_NEWS_FILE):
    with open(SENT_NEWS_FILE, "r") as f:
        try:
            sent_news_urls = set(json.load(f))
        except:
            sent_news_urls = set()
else:
    sent_news_urls = set()

# *******************************************************
def get_news_key(url, title):
    norm_url = normalize_url(url)
    key_base = f"{norm_url}|{title.strip().lower()}"
    return hashlib.md5(key_base.encode()).hexdigest()


sent_news_keys = set()
symbol_to_id_map = {}
load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")
#news_api_key = os.getenv("NEWS_API_KEY")
client = AsyncOpenAI(api_key=openai_api_key)
news_api_key ="506d562e0d4a434c97df2e3a51e4cd1c"

async def load_symbol_map():
    global symbol_to_id_map
    url = "https://api.coingecko.com/api/v3/coins/list"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f"CoinGecko status: {response.status}")  # <--- Bunu ekle
            if response.status == 200:
                ...
            else:
                print("❌ Coin listesi alınamadı.")



async def readmore(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🧭 Please use the /news command to see clickable news links.")

# --- HELP COMMAND ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "*📚 Coinspace Commands*\n\n"
        "💰 `/add BTC 0.5 30000`\n"
        "📊 `/portfolio`\n"
        "🔁 `/update BTC 1.0`\n"
        "🗑 `/remove BTC`\n"
        "🧹 `/clear`\n"
        "📈 `/performance`\n"
        "💵 `/price BTC`\n"
        "📉 `/graph`\n"
        "🔔 `/setalert BTC 70000`\n"
        "🤖 `/ai_btc`"
    )
    await update.message.reply_text(msg, parse_mode="MarkdownV2")

# --- BASIC ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    print(f"Kullanıcı ID: {user_id}")
    await update.message.reply_text(f"👋 Coinspace bot'a hoş geldin!\nSenin Telegram ID’n: `{user_id}`", parse_mode="Markdown")


async def fetch_price(coin_id: str):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data.get(coin_id, {}).get("usd")
            return None
        
        
async def price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Lütfen bir coin gir: /price BTC")
        return

    symbol = context.args[0].upper()
    coin_id = symbol_to_id_map.get(symbol)
    if not coin_id:
        await update.message.reply_text(f"❌ {symbol} için eşleşme bulunamadı.")
        return

    price = await fetch_price(coin_id)
    if price is not None:
        await update.message.reply_text(f"{symbol} fiyatı: ${price}")
    else:
        await update.message.reply_text(f"❌ {symbol} için fiyat alınamadı.")

# --- CRYPTOPANIC NEWS FETCH ---
async def fetch_newsapi_news():
    url = f"https://newsapi.org/v2/top-headlines?category=business&q=crypto&apiKey={news_api_key}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f"🌐 NewsAPI status: {response.status}")
            if response.status == 200:
                return await response.json()
                print(f"✅ NewsAPI'den {len(data.get('articles', []))} haber çekildi.")
                return data
            else:
                print(f"❌ NewsAPI isteği başarısız. Status: {response.status}")
                return None
            

    # --- AI destekli haber özetleme fonksiyonu ---
async def summarize_news(title, description):
    prompt = (
        f"Aşağıdaki haberin kısa bir özetini yaz:\n\n"
        f"Başlık: {title}\n"
        f"Açıklama: {description}\n\n"
        f"Kısa ve net bir şekilde yatırımcıya yönelik özet hazırla."
    )
    try:
        response = await client.chat.completions.create(
            model="gpt-4",  # veya gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Özetleme hatası:", e)
        return "⚠️ Özetlenemedi."

# --- Otomatik haber çekme ve paylaşma ---
async def check_and_send_news(app):
    while True:
        news_data = await fetch_newsapi_news()
        if news_data and "articles" in news_data:
            for article in news_data["articles"]:
                url = article.get("url")
                title = article.get("title", "No Title")
                description = article.get("description", "No Comment")

                # Normalize edilmiş benzersiz anahtar üret
                news_key = get_news_key(url, title)
                if news_key in sent_news_urls:
                    continue

                # GPT ile haber özetle
                summary = await summarize_news(title, description)

                # Mesaj metni
                text = (
                    f"📰 <b>{escape(title)}</b>\n"
                    f"{escape(summary)}\n"
                    f"<a href=\"{url}\">🔗 Habere Git</a>"
                )

                try:
                    await app.bot.send_message(
                        chat_id=os.getenv("OWNER_CHAT_ID"),
                        text=text,
                        parse_mode="HTML",
                        disable_web_page_preview=True
                    )
                    # Haber gönderildi olarak işaretle
                    sent_news_urls.add(news_key)
                    save_sent_urls()
                except Exception as e:
                    print("⚠️ Haber gönderilemedi:", e)

        await asyncio.sleep(7200)  # 2 saatte bir tekrar


async def run_bot():
    await load_symbol_map()
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("portfolio", portfolio))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("remove", remove))
    app.add_handler(CommandHandler("update", update_command))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("setalert", setalert))
    app.add_handler(CommandHandler("graph", graph))
    app.add_handler(CommandHandler("performance", performance))
    app.add_handler(CommandHandler("news", news_command))
    asyncio.create_task(check_and_send_news(app))
    app.add_handler(CommandHandler("readmore", readmore))
    app.add_handler(CommandHandler("backtest", backtest))
    app.add_handler(CallbackQueryHandler(feedback_handler))


    for cmd in ["ai_btc", "ai_eth", "ai_sol"]:
        app.add_handler(CommandHandler(cmd, ai_comment))

    await app.initialize()
    await app.start()
    asyncio.create_task(check_alerts(app))
    print("Telegram bot started")
    await app.updater.start_polling()

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

async def fetch_coin_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
        
        # AI COMMENTS **************

from openai import AsyncOpenAI
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import statistics

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def fetch_ohlc_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=7"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            return None

def calculate_technical_indicators(ohlc_data):
    closes = [item[4] for item in ohlc_data]  # Kapanış fiyatları
    if len(closes) < 26:
        return None

    rsi = calculate_rsi(closes)
    macd, signal = calculate_macd(closes)
    ma20 = statistics.mean(closes[-20:])
    ma5 = statistics.mean(closes[-5:])

    return {
        "RSI": rsi,
        "MACD": macd,
        "Signal": signal,
        "MA_20": ma20,
        "MA_5": ma5
    }

def calculate_rsi(prices, period=14):
    gains, losses = [], []
    for i in range(1, period + 1):
        delta = prices[-i] - prices[-i - 1]
        if delta > 0:
            gains.append(delta)
        else:
            losses.append(abs(delta))
    average_gain = sum(gains) / period
    average_loss = sum(losses) / period if losses else 0.001  # sıfıra bölünmesin
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def calculate_macd(prices):
    ema12 = exponential_moving_average(prices, 12)
    ema26 = exponential_moving_average(prices, 26)
    macd_line = [a - b for a, b in zip(ema12[-len(ema26):], ema26)]
    signal_line = exponential_moving_average(macd_line, 9)
    return round(macd_line[-1], 2), round(signal_line[-1], 2)

def exponential_moving_average(prices, window):
    ema = []
    k = 2 / (window + 1)
    for i, price in enumerate(prices):
        if i == 0:
            ema.append(price)
        else:
            ema.append(price * k + ema[-1] * (1 - k))
    return ema

async def generate_ai_comment(coin_data):
    name = coin_data["name"]
    price = coin_data["market_data"]["current_price"]["usd"]
    change_24h = coin_data["market_data"]["price_change_percentage_24h"]
    change_7d = coin_data["market_data"]["price_change_percentage_7d"]
    coin_id = coin_data["id"]

    ohlc_data = await fetch_ohlc_data(coin_id)
    if not ohlc_data:
        technicals = {}
    else:
        technicals = calculate_technical_indicators(ohlc_data) or {}

    rsi = technicals.get("RSI", "Bilinmiyor")
    macd = technicals.get("MACD", "Bilinmiyor")
    signal = technicals.get("Signal", "Bilinmiyor")
    ma_5 = technicals.get("MA_5", "Bilinmiyor")
    ma_20 = technicals.get("MA_20", "Bilinmiyor")

    prompt = (
        f"{name} için kısa ve net bir piyasa yorumu yaz.\n"
        f"Anlık fiyat: ${price:.2f}\n"
        f"24 saatlik değişim: {change_24h:.2f}%\n"
        f"7 günlük değişim: {change_7d:.2f}%\n"
        f"RSI: {rsi}, MACD: {macd}, Signal: {signal}, MA5: {ma_5}, MA20: {ma_20}\n\n"
        f"Ayrıca kaldıraçlı işlem önerisi yap:\n"
        f"- Pozisyon: LONG veya SHORT\n"
        f"- Giriş fiyatı\n"
        f"- Hedef fiyat\n"
        f"- Stop-loss\n"
        f"- Kaldıraç seviyesi (örn: 5x)\n\n"
        f"Kısa, yatırımcı odaklı ve teknik terimlerle sadeleştirilmiş bir şekilde maksimum 100 kelime kullan."
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ AI yorum hatası:", e)
        return "⚠️ AI yorum alınamadı."


async def ai_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower()
    symbol = text.replace("/ai_", "")
    symbol_map = {
        "btc": "bitcoin",
        "eth": "ethereum",
        "sol": "solana"
    }
    coin_id = symbol_map.get(symbol)
    if not coin_id:
        await update.message.reply_text("❌ Bu coin için yorum mevcut değil.")
        return

    await update.message.reply_text("💬 AI yorum hazırlanıyor...")

    coin_data = await fetch_coin_data(coin_id)
    comment = await generate_ai_comment(coin_data)
    await send_ai_signal(update, context, comment)


    # ALERT SYSTEM ---- ---- ---- 


from alert_db import add_alert  # alert_db.py içinde tanımlı olmalı

async def setalert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.message.reply_text("Kullanım: /setalert BTC 70000")
        return

    symbol = context.args[0].upper()
    target_price = float(context.args[1])

    coin_id = symbol_to_id_map.get(symbol)
    if not coin_id:
        await update.message.reply_text("❌ Geçersiz coin sembolü.")
        return

    user_id = update.effective_user.id
    add_alert(user_id, coin_id, target_price)

    await update.message.reply_text(f"🔔 {symbol} ({coin_id}) için {target_price}$ hedefli uyarı ayarlandı.")



from alert_db import get_all_alerts, delete_alert

async def check_alerts(app):
    while True:
        alerts = get_all_alerts()
        if not alerts:
            await asyncio.sleep(300)
            continue

        # Eksik verileri ele
        valid_alerts = []
        for alert in alerts:
            if all(k in alert for k in ("symbol", "target_price", "user_id")):
                valid_alerts.append(alert)

        # Sembolleri grupla
        symbol_map = {}
        for alert in valid_alerts:
            symbol = alert["symbol"].lower()
            symbol_map.setdefault(symbol, []).append(alert)

        # Fiyatları al ve kontrol et
        for symbol, alerts_list in symbol_map.items():
            coin_id = symbol_to_id_map.get(symbol.upper())
            if not coin_id:
                continue

            price = await fetch_price(coin_id)
            if price is None:
                continue  # API hatası varsa atla

            for alert in alerts_list:
                target_price = alert["target_price"]
                user_id = alert["user_id"]

                # Fiyat eşleşmesi
                if price >= target_price:
                    try:
                        await app.bot.send_message(
                            chat_id=user_id,
                            text=f"📢 *{symbol.upper()}* {target_price} $ seviyesini geçti!\nAnlık fiyat: {price:.2f} $",
                            parse_mode="Markdown"
                        )
                        delete_alert(user_id, symbol.upper())
                    except Exception as e:
                        print("❌ Bildirim gönderilemedi:", e)

        await asyncio.sleep(300)  # 5 dakikada bir kontrol



          # AI COMMENTS **************

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("🚀 /news komutu tetiklendi")

    news_data = await fetch_newsapi_news()
    if not news_data or "articles" not in news_data:
        await update.message.reply_text("❌ Haber verisi alınamadı.")
        return

    sent_count = 0
    for article in news_data["articles"][:5]:
        url = article.get("url")
        title = article.get("title", "No Title")
        description = article.get("description", "No Comment")

        norm_url = normalize_url(url)

        if norm_url and norm_url not in sent_news_urls:
            summary = await summarize_news(title, description)

            text = (
                f"📰 <b>{title}</b>\n"
                f"{summary}\n"
                f"<a href=\"{url}\">🔗 Habere Git</a>"
            )
            try:
                await update.message.reply_text(text, parse_mode="HTML")
                sent_news_urls.add(norm_url)
                save_sent_urls()
                sent_count += 1
            except Exception as e:
                print("⚠️ Gönderim hatası:", e)

    if sent_count == 0:
        await update.message.reply_text("⚠️ Gösterilecek yeni haber bulunamadı.")


# PORTFOLIO FONKS ---- ----- -----
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) not in [2, 3]:
        await update.message.reply_text("❌ Kullanım: /add BTC 0.5 [alış fiyatı]")
        return

    symbol = context.args[0].upper()
    try:
        amount = float(context.args[1])
        buy_price = float(context.args[2]) if len(context.args) == 3 else None
    except ValueError:
        await update.message.reply_text("❌ Miktar veya fiyat geçersiz.")
        return

    user_id = update.effective_user.id
    add_coin(user_id, symbol, amount, buy_price)
    msg = f"✅ {amount} {symbol} portföye eklendi."
    if buy_price:
        msg += f" Alış fiyatı: ${buy_price}"
    await update.message.reply_text(msg)


async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    holdings = get_portfolio(user_id)

    if not holdings:
        await update.message.reply_text("📭 Henüz hiç coin eklenmemiş. `/add` komutunu kullan.")
        return

    ids = [symbol_to_id_map.get(sym.upper()) for sym in holdings.keys() if symbol_to_id_map.get(sym.upper())]
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(ids)}&vs_currencies=usd"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            prices = await response.json()

    total_value = 0
    msg = "📊 Portföy:\n"
    for symbol, data in holdings.items():
        coin_id = symbol_to_id_map.get(symbol.upper())
        price = prices.get(coin_id, {}).get("usd")
        amount = data["amount"] if isinstance(data, dict) else data

        if price:
            value = price * amount
            total_value += value
            msg += f"• {symbol.upper()}: {amount} × ${price:.2f} = ${value:.2f}\n"
        else:
            msg += f"• {symbol.upper()}: Fiyat alınamadı\n"

    msg += f"\n💰 Toplam Değer: ${total_value:.2f}"
    await update.message.reply_text(msg)


async def remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("❌ Kullanım: /remove BTC")
        return

    symbol = context.args[0].upper()
    user_id = update.effective_user.id
    success = remove_coin(user_id, symbol)

    if success:
        await update.message.reply_text(f"🗑️ {symbol} portföyünden silindi.")
    else:
        await update.message.reply_text(f"⚠️ {symbol} portföyünde bulunamadı.")

        from portfolio_db import update_coin, clear_portfolio

async def update_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.message.reply_text("❌ Kullanım: /update BTC 1.2")
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
        await update.message.reply_text(f"🔄 {symbol} miktarı {amount} olarak güncellendi.")
    else:
        await update.message.reply_text(f"⚠️ {symbol} portföyde bulunamadı, önce `/add` ile ekleyin.")

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    success = clear_portfolio(user_id)
    if success:
        await update.message.reply_text("🧹 Portföyünüz başarıyla temizlendi.")
    else:
        await update.message.reply_text("❗ Temizlenecek veri bulunamadı.")

# GRAPHIC ---- ---- ----

async def graph(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    holdings = get_portfolio(user_id)

    if not holdings:
        await update.message.reply_text("📭 Portföy boş. Önce /add ile coin ekleyin.")
        return

    ids = [symbol_to_id_map.get(sym.upper()) for sym in holdings.keys() if symbol_to_id_map.get(sym.upper())]
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(ids)}&vs_currencies=usd"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            prices = await response.json()

    labels = []
    values = []

    for symbol, data in holdings.items():
        coin_id = symbol_to_id_map.get(symbol.upper())
        price = prices.get(coin_id, {}).get("usd")
        amount = data["amount"] if isinstance(data, dict) else data

        if price:
            value = price * amount
            labels.append(symbol.upper())
            values.append(value)

    if not values:
        await update.message.reply_text("⚠️ Fiyat verisi alınamadı.")
        return

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    plt.title("📊 Portföy Dağılımı")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    await update.message.reply_photo(photo=InputFile(buf, filename="portfolio.png"))

    # PERFORMANCE ---- ---- ----

async def performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    portfolio = get_portfolio(user_id)

    if not portfolio:
        await update.message.reply_text("📭 Portföy boş.")
        return

    ids = [symbol_to_id_map.get(sym.upper()) for sym in portfolio.keys() if symbol_to_id_map.get(sym.upper())]
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(ids)}&vs_currencies=usd"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            prices = await response.json()

    msg = "📈 Portföy Performansı:\n"
    total_pl = 0
    for symbol, data in portfolio.items():
        coin_id = symbol_to_id_map.get(symbol.upper())
        current_price = prices.get(coin_id, {}).get("usd")
        amount = data["amount"] if isinstance(data, dict) else data
        buy_price = data.get("buy_price") if isinstance(data, dict) else None

        if not current_price:
            msg += f"• {symbol}: Fiyat alınamadı\n"
            continue

        if buy_price:
            current_value = current_price * amount
            cost_basis = buy_price * amount
            pl = current_value - cost_basis
            total_pl += pl
            msg += f"• {symbol}: Alış ${buy_price:.2f} → Şu an ${current_price:.2f} | P&L: ${pl:.2f}\n"
        else:
            msg += f"• {symbol}: Alış fiyatı bilinmiyor\n"

    msg += f"\n💼 Toplam Kar/Zarar: ${total_pl:.2f}"
    await update.message.reply_text(msg)

    import json

SIGNAL_FILE = "signals.json"

def load_signals():
    if os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE, "r") as f:
            return json.load(f)
    return []

def save_signals(data):
    with open(SIGNAL_FILE, "w") as f:
        json.dump(data, f, indent=2)

async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    message_id = query.message.message_id
    user_id = query.from_user.id
    feedback = query.data  # "like" or "dislike"

    signals = load_signals()
    for s in signals:
        if s.get("message_id") == message_id:
            s.setdefault("likes", []).append(user_id) if feedback == "like" else s.setdefault("dislikes", []).append(user_id)
            break
    else:
        # ilk kez tepki geliyor
        signals.append({
            "message_id": message_id,
            "likes": [user_id] if feedback == "like" else [],
            "dislikes": [user_id] if feedback == "dislike" else []
        })

    save_signals(signals)
    await query.edit_message_reply_markup(reply_markup=None)
    await query.message.reply_text("✅ Geri bildirimin için teşekkürler.")

async def send_ai_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, signal_text: str):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("👍", callback_data="feedback:like"),
            InlineKeyboardButton("👎", callback_data="feedback:dislike")
        ]
    ])
    message = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=signal_text,
        reply_markup=keyboard
    )

    # ⬇️ Eksik olan satır buydu
    signals = load_signals()
    signals.append({
        "message_id": message.message_id,
        "text": signal_text,
        "likes": [],
        "dislikes": []
    })
    save_signals(signals)



async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    callback_data = query.data
    print(f"[DEBUG] Callback received: {callback_data} from user {query.from_user.id}")

    if not callback_data.startswith("feedback:"):
        return

    feedback = callback_data.split(":")[1]  # "like" veya "dislike"
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
        signals.append({
            "message_id": message_id,
            "likes": [user_id] if feedback == "like" else [],
            "dislikes": [user_id] if feedback == "dislike" else []
        })

    save_signals(signals)
    await query.edit_message_reply_markup(reply_markup=None)
    await query.message.reply_text("✅ Geri bildirimin için teşekkürler.")
