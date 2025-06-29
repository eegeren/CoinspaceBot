import requests
import datetime

def fetch_cryptopanic(api_key):
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&public=true"
        response = requests.get(url)
        data = response.json()
        return [
            {
                "title": item.get("title"),
                "source": "CryptoPanic",
                "url": item.get("url"),
                "published_at": item.get("published_at"),
                "summary": item.get("slug"),
            }
            for item in data.get("results", [])
        ]
    except Exception as e:
        print(f"❌ CryptoPanic error: {e}")
        return []

def fetch_coinstats():
    try:
        url = "https://api.coinstats.app/public/v1/news"
        response = requests.get(url)
        data = response.json()
        return [
            {
                "title": item.get("title"),
                "source": "CoinStats",
                "url": item.get("link"),
                "published_at": datetime.datetime.utcfromtimestamp(item.get("feedDate") / 1000).isoformat(),
                "summary": item.get("description"),
            }
            for item in data.get("news", [])
        ]
    except Exception as e:
        print(f"❌ CoinStats error: {e}")
        return []

def fetch_coingecko():
    try:
        url = "https://api.coingecko.com/api/v3/status_updates"
        response = requests.get(url)
        data = response.json()
        return [
            {
                "title": item.get("project", {}).get("name"),
                "source": "CoinGecko",
                "url": item.get("project", {}).get("homepage", ""),
                "published_at": item.get("created_at"),
                "summary": item.get("description"),
            }
            for item in data.get("status_updates", [])
        ]
    except Exception as e:
        print(f"❌ CoinGecko error: {e}")
        return []

def fetch_coindesk():
    try:
        url = "https://api.coindesk.com/v1/news"
        response = requests.get(url)
        data = response.json()
        return [
            {
                "title": item.get("headline"),
                "source": "CoinDesk",
                "url": item.get("url"),
                "published_at": item.get("datetime"),
                "summary": item.get("summary"),
            }
            for item in data.get("articles", [])
        ]
    except Exception as e:
        print(f"❌ CoinDesk error: {e}")
        return []

def fetch_all_news(cryptopanic_key=None):
    news = []
    if cryptopanic_key:
        news += fetch_cryptopanic(cryptopanic_key)
    news += fetch_coinstats()
    news += fetch_coingecko()
    news += fetch_coindesk()
    return sorted(news, key=lambda x: x["published_at"], reverse=True)
