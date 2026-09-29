"""
Step 1 - Daily realized-variance dataset for BTC/USDT
from 5-minute Binance spot klines (free public data, no API key).

Output: data/btc_daily.csv
  ret      daily log-return (sum of 5-min log-returns)
  rv       daily realized variance (sum of squared 5-min log-returns)
  logrv    log(rv), the forecasting target
  vol_ann  annualized volatility in % (BTC trades 365 days a year)
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOL, FREQ = "BTCUSDT", "5m"
START, END = "2019-01", "2026-08"            # monthly files, inclusive
BARS_PER_DAY = 288                           # 24 h x 12 bars of 5 min
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
RAW = Path("data/raw")


def load_month(ym: str) -> pd.Series:
    """Download (once, then cache) one month of klines; return 5-min closes."""
    fname = RAW / f"{SYMBOL}-{FREQ}-{ym}.zip"
    if not fname.exists():
        url = (f"https://data.binance.vision/data/spot/monthly/klines/"
               f"{SYMBOL}/{FREQ}/{fname.name}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        fname.write_bytes(r.content)
    with zipfile.ZipFile(fname) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLS)
    df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
    ts = df["open_time"].astype("int64")
    unit = "us" if ts.iloc[0] > 1e14 else "ms"   # Binance moved to microseconds in 2025
    return pd.Series(df["close"].astype(float).values,
                     index=pd.to_datetime(ts.values, unit=unit, utc=True))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(START, END, freq="M").strftime("%Y-%m")
    close = pd.concat([load_month(m) for m in months]).sort_index()
    close = close[~close.index.duplicated()]

    r = np.log(close).diff().dropna()
    daily = r.groupby(r.index.floor("D")).agg(
        ret="sum", rv=lambda x: (x ** 2).sum(), n="size")
    daily = daily[daily["n"] >= 0.9 * BARS_PER_DAY]      # drop incomplete days
    daily["logrv"] = np.log(daily["rv"])
    daily["vol_ann"] = 100 * np.sqrt(365 * daily["rv"])
    daily.index = daily.index.tz_localize(None)
    daily.drop(columns="n").to_csv("data/btc_daily.csv")
    print(f"{len(daily)} days, from {daily.index[0].date()} to {daily.index[-1].date()}")


if __name__ == "__main__":
    main()
