import os
import asyncio
import aiohttp
from telegram import Update, InputFile, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from portfolio_db import add_coin, get_portfolio, remove_coin, update_coin, clear_portfolio
from alert_db import add_alert, get_all_alerts, delete_alert
from telegram.constants import ParseMode
from dotenv import load_dotenv
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

# Ortam değişkenlerini yükle
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "506d562e0d4a434c97df2e3a51e4cd1c")

# OpenAI istemcisi
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Modeli global olarak yükle
model = None
try:
    model_path = os.path.abspath("model.pkl")
    model = joblib.load(model_path)
    print(f"✅ Model yüklendi: {model_path}")
except Exception as e:
    print(f"❌ Model yüklenemedi: {e}")

# Kabul edilen kullanıcılar
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

accepted_users = load_accepted_users()

# Haber dosyası
SENT_NEWS_FILE = "sent_news.json"
if os.path.exists(SENT_NEWS_FILE):
    with open(SENT_NEWS_FILE, "r") as f:
        try:
            sent_news_urls = set(json.load(f))
        except:
            sent_news_urls = set()
else:
    sent_news_urls = set()

# Sinyal dosyası
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

# Coin sembol haritası
symbol_to_id_map = {}

async def load_symbol_map():
    global symbol_to_id_map
    url = "https://api.coingecko.com/api/v3/coins/list"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f"CoinGecko status: {response.status}")
            if response.status == 200:
                coins = await response.json()
                symbol_to_id_map.update({coin["symbol"].upper(): coin["id"] for coin in coins})
            else:
                print("❌ Coin listesi alınamadı.")

# Yardımcı fonksiyonlar
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

# Fiyat alma
async def fetch_price(coin_id: str):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data.get(coin_id, {}).get("usd")
            return None

# Price komutu
async def price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Lütfen bir coin gir: /price BTC")
        return

    symbol = context.args[0].upper()
    coin_id = symbol_to_id_map.get(symbol)
    if not coin_id:
        await update.message.reply_text(f"❌ {symbol} için eşleşme bulunamadı.")
        return

    price_value = await fetch_price(coin_id)
    if price_value is not None:
        await update.message.reply_text(f"{symbol} fiyatı: ${price_value:.2f}")
    else:
        await update.message.reply_text(f"❌ {symbol} için fiyat alınamadı.")

# OHLC veri alma
async def fetch_ohlc_data(coin_id, days=7):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                prices = data.get("prices", [])
                df = pd.DataFrame(prices, columns=["timestamp", "price"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                return df
            return None

# Teknik göstergeler
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        delta = prices[-i] - prices[-i - 1]
        if delta > 0:
            gains.append(delta)
        else:
            losses.append(abs(delta))
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
    macd_line = [a - b for a, b in zip(ema12[-len(ema26):], ema26)]
    signal_line = exponential_moving_average(macd_line, 9)
    return round(macd_line[-1], 2), round(signal_line[-1], 2)

def calculate_technical_indicators(ohlc_data):
    closes = [item[4] for item in ohlc_data]
    if len(closes) < 26:
        return None
    rsi = calculate_rsi(closes)
    macd, signal = calculate_macd(closes)
    ma5 = sum(closes[-5:]) / 5
    ma20 = sum(closes[-20:]) / 20
    return {
        "RSI": rsi,
        "MACD": macd,
        "Signal": signal,
        "MA_5": ma5,
        "MA_20": ma20
    }

# Model tahmini
def predict_signal(features_df):
    if not model:
        print("❌ Model yüklü değil!")
        return None
    print(f"Özellik şekli: {features_df.shape}, türü: {type(features_df)}")
    print(f"Özellik sütunları: {features_df.columns}")
    print(f"Özellik verisi:\n{features_df}")
    try:
        prediction = model.predict(features_df)
        return prediction[0]
    except Exception as e:
        print(f"❌ Tahmin hatası: {e}")
        return None

# AI yorumu
async def generate_ai_comment(coin_data):
    name = coin_data["name"]
    price = coin_data["market_data"]["current_price"]["usd"]
    change_24h = coin_data["market_data"]["price_change_percentage_24h"]
    change_7d = coin_data["market_data"]["price_change_percentage_7d"]
    coin_id = coin_data["id"]

    df = await fetch_ohlc_data(coin_id)
    if df is None or df.empty:
        return f"{name} için veri alınamadı."

    closes = df["price"].values
    if len(closes) < 26:
        return f"{name} için yeterli veri yok."

    rsi = calculate_rsi(closes)
    macd, signal = calculate_macd(closes)
    ma_5 = sum(closes[-5:]) / 5
    ma_20 = sum(closes[-20:]) / 20

    print(f"RSI: {rsi}, MACD: {macd}, Signal: {signal}, MA_5: {ma_5}, MA_20: {ma_20}")

    features = pd.DataFrame([{
        "RSI": rsi,
        "MACD": macd,
        "Signal": signal,
        "MA_5": ma_5,
        "MA_20": ma_20
    }])

    prediction = predict_signal(features)
    if prediction is None:
        ai_signal = "⚠️ AI modeli yüklü değil veya hata oluştu."
        leverage_suggestion = ""
        risk_level = ""
    else:
        ai_signal = "📈 AI Tahmin: BUY" if prediction == 1 else "📉 AI Tahmin: SELL"

        # Kaldıraç önerisi
        if prediction == 1 and rsi < 70:
            leverage_suggestion = "📊 Önerilen kaldıraç: 5x Long"
        elif prediction == 0 and rsi > 30:
            leverage_suggestion = "📊 Önerilen kaldıraç: 5x Short"
        else:
            leverage_suggestion = "⚠️ Kaldıraçlı işlem önerilmez"

        # Risk değerlendirmesi
        if rsi > 75 or abs(macd) < 0.05:
            risk_level = "⚠️ Yüksek risk (kaldıraç önerilmez)"
        else:
            risk_level = "✅ Düşük risk (kaldıraç kullanılabilir)"

    comment = (
        f"{name} fiyatı: ${price:.2f}\n"
        f"24 saatlik değişim: %{change_24h:.2f}, 7 gün: %{change_7d:.2f}\n\n"
        f"{ai_signal}\n\n"
        f"RSI: {rsi}, MACD: {macd}, Signal: {signal}, MA_5: {ma_5:.2f}, MA_20: {ma_20:.2f}\n\n"
        f"{leverage_suggestion}\n"
        f"{risk_level}"
    )
    return comment


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
    if not coin_data:
        await update.message.reply_text("❌ Coin verisi alınamadı.")
        return
    comment = await generate_ai_comment(coin_data)
    await send_ai_signal(update, context, comment)

# Kaldıraç sinyali
async def leverage_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        response = requests.get("http://localhost:8000/api/leverage-signal")
        response.raise_for_status()
        data = response.json()
        msg = (
            f"💹 *AI Kaldıraçlı Sinyal* ({data['pair']} – 5x {data['direction']})\n\n"
            f"📈 *Giriş:* {data['entry']}\n"
            f"🎯 *Hedef:* {data['target']}\n"
            f"🛑 *Stop:* {data['stop']}\n"
            f"🤖 *Güven:* %{data['confidence']}"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        print(f"❌ leverage_signal hatası: {e}")
        await update.message.reply_text("❌ Sinyal alınamadı. Daha sonra tekrar deneyin.")

# Diğer komutlar
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in accepted_users:
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ I Agree", callback_data="accept_disclaimer")]
        ])
        disclaimer_text = (
            "📢 *Sorumluluk Reddi*\n\n"
            "Coinspace Bot, kripto piyasasında bilinçli kararlar almanıza yardımcı olmak için piyasa içgörüleri ve AI destekli sinyaller sunar. "
            "Bu sinyaller yalnızca bilgi amaçlıdır ve finansal tavsiye niteliği taşımaz.\n\n"
            "Botu kullanmaya devam etmek için lütfen bunu anladığınızı onaylayın."
        )
        await update.message.reply_text(disclaimer_text, reply_markup=keyboard, parse_mode="Markdown")
    else:
        msg = (
            "👋 *Coinspace Bot’a tekrar hoş geldiniz!*\n\n"
            "🚀 Günlük AI destekli ticaret sinyalleri, fiyat uyarıları, portföy takibi ve canlı piyasa güncellemeleri alın.\n\n"
            "🔐 *Premium’a Yükselt*:\n"
            "• Sınırsız AI Kaldıraç Sinyalleri (Ücretsiz kullanıcılar günde sadece 2 sinyal alır)\n"
            "• Tam piyasa özetlerine erişim\n"
            "• Öncelikli destek ve erken özellik erişimi\n\n"
            "*💳 Abonelik Planları:*\n"
            "• 1 Ay: $29.99\n"
            "• 3 Ay: $69.99\n"
            "• 1 Yıl: $399.99\n\n"
            "👉 *Yükseltmek için*, bir plan seçin ve ödemeyi tamamlayın:\n"
            "[1 Ay Ödeme](https://nowpayments.io/payment/?iid=5260731771)\n"
            "[3 Ay Ödeme](https://nowpayments.io/payment/?iid=4400895826)\n"
            "[1 Yıl Ödeme](https://nowpayments.io/payment/?iid=4501340550)\n\n"
            "Ödeme sonrası /premium komutunu kullanarak aboneliğinizi aktifleştirin."
        )
        await update.message.reply_text(msg, parse_mode="Markdown", disable_web_page_preview=True)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "*📚 Coinspace Komutları*\n\n"
        "💰 `/add BTC 0.5 30000` - Coin ekle\n"
        "📊 `/portfolio` - Portföyü göster\n"
        "🔁 `/update BTC 1.0` - Coin miktarını güncelle\n"
        "🗑 `/remove BTC` - Coin sil\n"
        "🧹 `/clear` - Portföyü temizle\n"
        "📈 `/performance` - Portföy performansı\n"
        "💵 `/price BTC` - Coin fiyatı\n"
        "📉 `/graph` - Portföy grafiği\n"
        "🔔 `/setalert BTC 70000` - Fiyat uyarısı\n"
        "🤖 `/ai_btc` - AI yorumu\n"
        "📰 `/news` - Kripto haberleri\n"
        "🔗 `/readmore` - Haber linkleri\n"
        "📈 `/backtest BTC` - Strateji testi\n"
        "💎 `/premium` - Premium abonelik\n"
        "💹 `/leverage_signal` - Kaldıraçlı sinyal"
    )
    await update.message.reply_text(msg, parse_mode="MarkdownV2")

async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    holdings = get_portfolio(user_id)
    if not holdings:
        await update.message.reply_text("📭 Henüz coin eklenmemiş. `/add` komutunu kullan.")
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
        await update.message.reply_text(f"🔗 {symbol} miktarı {amount} olarak güncellendi.")
    else:
        await update.message.reply_text(f"⚠️ {symbol} portföyde bulunamadı, önce `/add` ile ekleyin.")

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    success = clear_portfolio(user_id)
    if success:
        await update.message.reply_text("🧼 Portföyünüz başarıyla temizlendi.")
    else:
        await update.message.reply_text("❗ Temizlenecek veri bulunamadı.")

async def graph(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    holdings = get_portfolio(user_id)
    if not holdings:
        await update.message.reply_text("📭 Portföy boş. Önce /add ile ekleyin.")
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
    plt.title("📈 Portföy Dağılımı")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    await update.message.reply_photo(photo=InputFile(buf, filename="portfolio.png"))

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
            msg += f"• {symbol}: Alış ${buy_price:.2f} → Şimdi ${current_price:.2f} | Kâr/Zarar: ${pl:.2f}\n"
        else:
            msg += f"• {symbol}: Alış fiyatı bilinmiyor\n"
    msg += f"\n💼 Toplam Kâr/Zarar: ${total_pl:.2f}"
    await update.message.reply_text(msg)

async def setalert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.message.reply_text("Kullanım: /setalert BTC 70000")
        return
    symbol = context.args[0].upper()
    try:
        target_price = float(context.args[1])
    except ValueError:
        await update.message.reply_text("❌ Geçersiz fiyat.")
        return
    coin_id = symbol_to_id_map.get(symbol)
    if not coin_id:
        await update.message.reply_text("❌ Geçersiz coin sembolü.")
        return
    user_id = update.effective_user.id
    add_alert(user_id, coin_id, target_price)
    await update.message.reply_text(f"🔔 {symbol} ({coin_id}) için ${target_price} hedefli uyarı ayarlandı.")

async def check_alerts(app):
    while True:
        alerts = get_all_alerts()
        if not alerts:
            await asyncio.sleep(300)
            continue
        valid_alerts = [alert for alert in alerts if all(k in alert for k in ("symbol", "target_price", "user_id"))]
        symbol_map = {}
        for alert in valid_alerts:
            symbol = alert["symbol"].lower()
            symbol_map.setdefault(symbol, []).append(alert)
        for symbol, alerts_list in symbol_map.items():
            coin_id = symbol_to_id_map.get(symbol.upper())
            if not coin_id:
                continue
            price = await fetch_price(coin_id)
            if price is None:
                continue
            for alert in alerts_list:
                target_price = alert["target_price"]
                user_id = alert["user_id"]
                if price >= target_price:
                    try:
                        await app.bot.send_message(
                            chat_id=user_id,
                            text=f"📢 *{symbol.upper()}* ${target_price} seviyesini geçti!\nŞu anki fiyat: ${price:.2f}",
                            parse_mode="Markdown"
                        )
                        delete_alert(user_id, symbol.upper())
                    except Exception as e:
                        print(f"❌ Bildirim gönderilemedi: {e}")
        await asyncio.sleep(300)

async def fetch_newsapi_news():
    url = f"https://newsapi.org/v2/top-headlines?category=business&q=crypto&apiKey={NEWS_API_KEY}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f"🌐 NewsAPI status: {response.status}")
            if response.status == 200:
                return await response.json()
            return None

async def summarize_news(title, description):
    prompt = (
        f"Aşağıdaki haberin kısa bir özetini yaz:\n\n"
        f"Başlık: {title}\n"
        f"Açıklama: {description}\n\n"
        f"Kısa ve net bir şekilde yatırımcıya yönelik özet hazırla."
    )
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Özetleme hatası: {e}")
        return "⚠️ Özetlenemedi."

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
                f"📰 <b>{escape(title)}</b>\n"
                f"{escape(summary)}\n"
                f"<a href=\"{url}\">🔗 Habere Git</a>"
            )
            try:
                await update.message.reply_text(text, parse_mode="HTML")
                sent_news_urls.add(norm_url)
                save_sent_urls()
                sent_count += 1
            except Exception as e:
                print(f"⚠️ Gönderim hatası: {e}")
    if sent_count == 0:
        await update.message.reply_text("⚠️ Gösterilecek yeni haber bulunamadı.")

async def readmore(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🧭 Haber linklerini görmek için /news komutunu kullanın.")

async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Kullanım: /backtest BTC")
        return
    symbol = context.args[0].upper()
    coin_id = symbol_to_id_map.get(symbol)
    if not coin_id:
        await update.message.reply_text("❌ Coin bulunamadı.")
        return
    df = await fetch_ohlc_data(coin_id, days=30)
    if df is None or df.empty:
        await update.message.reply_text("❌ Veri alınamadı.")
        return
    from ta.momentum import RSIIndicator
    from ta.trend import SMAIndicator
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
    msg = f"📈 {symbol} RSI + MA Geri Test Sonucu (30 gün):\n"
    msg += f"✅ Alım sayısı: {len(buy_points)}\n"
    msg += f"💰 Toplam Kar: ${pnl:.2f}"
    await update.message.reply_text(msg)

async def premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🟢 1 Ay – $29.99", url="https://nowpayments.io/payment/?iid=5260731771")],
        [InlineKeyboardButton("🔵 3 Ay – $70", url="https://nowpayments.io/payment/?iid=4400895826")],
        [InlineKeyboardButton("🟣 1 Yıl – $399", url="https://nowpayments.io/payment/?iid=4501340550")],
    ])
    msg = (
        "✨ *Coinspace Premium’a Yükselt!*\n\n"
        "🚀 Avantajlar:\n"
        "• Günde 10’a kadar AI tabanlı ticaret sinyali\n"
        "• Kaldıraçlı ticaret önerileri\n"
        "• Öncelikli piyasa uyarıları ve haberler\n"
        "• Portföy analiz araçları\n\n"
        "💰 *Fiyatlar:*\n"
        "1 Ay: $29.99\n"
        "3 Ay: $70\n"
        "1 Yıl: $399\n\n"
        "_USDT (TRC20) ile NOWPayments üzerinden güvenli ödeme yapın._"
    )
    await update.message.reply_text(msg, reply_markup=keyboard, parse_mode="Markdown")

async def accept_disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    accepted_users.add(user_id)
    save_accepted_users(accepted_users)
    await query.answer()
    await query.edit_message_text("✅ Şartları kabul ettiniz. Komutları kullanmaya başlayabilirsiniz.")

async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    callback_data = query.data
    print(f"[DEBUG] Geri bildirim alındı: {callback_data} from user {query.from_user.id}")
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
        signals.append({
            "message_id": message_id,
            "likes": [user_id] if feedback == "like" else [],
            "dislikes": [user_id] if feedback == "dislike" else []
        })
    save_signals(signals)
    await query.edit_message_reply_markup(reply_markup=None)
    await query.message.reply_text("✅ Geri bildirimin için teşekkürler.")

async def send_ai_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, signal_text: str):
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
    signals = load_signals()
    signals.append({
        "message_id": message.message_id,
        "text": signal_text,
        "likes": [],
        "dislikes": []
    })
    save_signals(signals)

async def check_and_send_news(app):
    while True:
        news_data = await fetch_newsapi_news()
        if news_data and "articles" in news_data:
            for article in news_data["articles"]:
                url = article.get("url")
                title = article.get("title", "No Title")
                description = article.get("description", "No Comment")
                news_key = get_news_key(url, title)
                if news_key in sent_news_urls:
                    continue
                summary = await summarize_news(title, description)
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
                    sent_news_urls.add(news_key)
                    save_sent_urls()
                except Exception as e:
                    print(f"⚠️ Haber gönderilemedi: {e}")
        await asyncio.sleep(7200)

async def fetch_coin_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            return None

async def run_bot():
    print("🚀 Bot başlatılıyor...")
    app = ApplicationBuilder().token(TOKEN).build()
    print("✅ Telegram bot uygulaması oluşturuldu.")
    await load_symbol_map()
    print("✅ Coin sembolleri yüklendi.")
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
    app.add_handler(CommandHandler("readmore", readmore))
    app.add_handler(CommandHandler("backtest", backtest))
    app.add_handler(CommandHandler("premium", premium))
    app.add_handler(CommandHandler("leverage_signal", leverage_signal))
    app.add_handler(CallbackQueryHandler(accept_disclaimer, pattern="^accept_disclaimer$"))
    app.add_handler(CallbackQueryHandler(feedback_handler))
    for cmd in ["ai_btc", "ai_eth", "ai_sol"]:
        app.add_handler(CommandHandler(cmd, ai_comment))
    asyncio.create_task(check_alerts(app))
    asyncio.create_task(check_and_send_news(app))
    print("🔄 Arka plan görevleri başlatıldı.")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    print("✅ Bot başlatıldı.")

if __name__ == "__main__":
    asyncio.run(run_bot())