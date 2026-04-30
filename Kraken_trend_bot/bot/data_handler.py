# bot/data_handler.py
import pandas as pd
from typing import List

def ohlcv_to_dataframe(ohlcv: List[list]) -> pd.DataFrame:
    """
    ohlcv: list of [timestamp, open, high, low, close, volume]
    """
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df
