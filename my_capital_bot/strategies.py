# strategies.py

import pandas as pd
import numpy as np

def moving_average_crossover_strategy(df, short_window=10, long_window=30):
    """
    Short-term moving average crossover for day trading.
    - Best used with 15-minute bars.
    - short_ma=10, long_ma=30 => ~2.5hr vs. ~7.5hr moving averages.
    """
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()

    df['signal'] = 0
    df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1  # Buy
    df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1 # Sell

    return df

def bollinger_bands_strategy(df, window=14, num_std=2):
    """
    Bollinger Bands strategy for day trading with window=14.
    - Works well with 15-minute bars for short-term signals.
    """
    df['rolling_mean'] = df['close'].rolling(window=window).mean()
    df['rolling_std'] = df['close'].rolling(window=window).std()

    df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * num_std)
    df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * num_std)

    df['signal'] = 0
    df.loc[df['close'] < df['lower_band'], 'signal'] = 1   # Buy (oversold)
    df.loc[df['close'] > df['upper_band'], 'signal'] = -1  # Sell (overbought)

    return df

def rsi_strategy(df, window=12, upper=75, lower=25):
    """
    RSI strategy for intraday with window=12, upper=75, lower=25.
    - Tighter bounds for short-term overbought/oversold signals.
    - Good match for 15-minute bars.
    """
    delta = df['close'].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gains = gains.ewm(alpha=1/window, min_periods=window).mean()
    avg_losses = losses.ewm(alpha=1/window, min_periods=window).mean()

    rs = avg_gains / avg_losses
    df['rsi'] = 100 - (100 / (1 + rs))

    df['signal'] = 0
    # Overbought => Sell
    df.loc[df['rsi'] > upper, 'signal'] = -1
    # Oversold => Buy
    df.loc[df['rsi'] < lower, 'signal'] = 1

    return df