from tinydb import TinyDB, Query

db = TinyDB("portfolio.json")
Portfolio = Query()

def add_coin(user_id: int, symbol: str, amount: float, buy_price: float = None):
    symbol = symbol.upper()
    user_data = db.get(Portfolio.user_id == user_id)
    if user_data:
        portfolio = user_data["portfolio"]
        if symbol in portfolio:
            portfolio[symbol]["amount"] += amount
        else:
            portfolio[symbol] = {"amount": amount, "buy_price": buy_price}
        db.update({"portfolio": portfolio}, Portfolio.user_id == user_id)
    else:
        db.insert({
            "user_id": user_id,
            "portfolio": {
                symbol: {"amount": amount, "buy_price": buy_price}
            }
        })


def get_portfolio(user_id: int):
    user_data = db.get(Portfolio.user_id == user_id)
    return user_data["portfolio"] if user_data else {}


def remove_coin(user_id: int, symbol: str):
    user_data = db.get(Portfolio.user_id == user_id)
    if user_data:
        portfolio = user_data["portfolio"]
        if symbol in portfolio:
            del portfolio[symbol]
            db.update({"portfolio": portfolio}, Portfolio.user_id == user_id)
            return True
    return False

def update_coin(user_id: int, symbol: str, new_amount: float):
    user_data = db.get(Portfolio.user_id == user_id)
    if user_data:
        portfolio = user_data["portfolio"]
        portfolio[symbol] = new_amount
        db.update({"portfolio": portfolio}, Portfolio.user_id == user_id)
        return True
    return False

def clear_portfolio(user_id: int):
    user_data = db.get(Portfolio.user_id == user_id)
    if user_data:
        db.update({"portfolio": {}}, Portfolio.user_id == user_id)
        return True
    return False
