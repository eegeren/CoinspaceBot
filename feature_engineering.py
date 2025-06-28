
import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

def add_indicators(df):
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['macd'] = MACD(close=df['close']).macd_diff()
    df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    return df

def generate_targets(df, tp_pct=0.02, sl_pct=0.01, horizon=6):
    df['target'] = 0  # 1 = BUY, -1 = SELL, 0 = HOLD
    df['TP_PCT'] = 0.0
    df['SL_PCT'] = 0.0

    for i in range(len(df) - horizon):
        future_prices = df['close'].iloc[i+1:i+1+horizon].values
        entry_price = df['close'].iloc[i]
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)

        hit_tp = np.any(future_prices >= tp_price)
        hit_sl = np.any(future_prices <= sl_price)

        if hit_tp and not hit_sl:
            df.at[df.index[i], 'target'] = 1
        elif hit_sl and not hit_tp:
            df.at[df.index[i], 'target'] = -1
        elif hit_tp and hit_sl:
            tp_index = np.argmax(future_prices >= tp_price)
            sl_index = np.argmax(future_prices <= sl_price)
            df.at[df.index[i], 'target'] = 1 if tp_index < sl_index else -1

        df.at[df.index[i], 'TP_PCT'] = (tp_price - entry_price) / entry_price
        df.at[df.index[i], 'SL_PCT'] = (entry_price - sl_price) / entry_price

    return df

def main():
    df = pd.read_csv("raw_data.csv", index_col="timestamp", parse_dates=True)
    df = add_indicators(df)
    df = generate_targets(df)
    df.dropna(inplace=True)
    df.to_csv("training_data.csv")
    print("✅ Göstergeler ve hedefler oluşturuldu: training_data.csv")

if __name__ == "__main__":
    main()
