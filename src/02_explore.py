"""
Step 2 - Stylized facts of BTC realized variance.
  * RV is extremely skewed, log RV is close to Gaussian  -> model log RV
  * returns are almost unpredictable (ACF ~ 0)
  * log RV is highly persistent (long memory) with a 7-day seasonality
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

Path("results").mkdir(exist_ok=True)
df = pd.read_csv("data/btc_daily.csv", index_col=0, parse_dates=True)

for col in ["rv", "logrv"]:
    print(f"{col:6s} skew = {stats.skew(df[col]):8.2f}   excess kurtosis = {stats.kurtosis(df[col]):8.2f}")

LAGS = 100
acf_ret = acf(df["ret"], nlags=LAGS)
acf_lrv = acf(df["logrv"], nlags=LAGS)
print(f"ACF returns lag 1 = {acf_ret[1]:+.2f} | ACF log RV lag 1 = {acf_lrv[1]:.2f}, lag 30 = {acf_lrv[30]:.2f}")
print("\n5 most volatile days:\n", df.nlargest(5, "vol_ann")[["ret", "vol_ann"]].round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(df["logrv"], bins=60, density=True, alpha=0.6)
x = np.linspace(df["logrv"].min(), df["logrv"].max(), 200)
ax[0].plot(x, stats.norm.pdf(x, df["logrv"].mean(), df["logrv"].std()), "k")
ax[0].set_title("Distribution of log RV (black: fitted normal)")
band = 1.96 / np.sqrt(len(df))
ax[1].plot(acf_ret[1:], label="returns")
ax[1].plot(acf_lrv[1:], label="log RV")
ax[1].axhspan(-band, band, color="grey", alpha=0.3)
ax[1].set_title("Autocorrelation"); ax[1].set_xlabel("lag (days)"); ax[1].legend()
plt.tight_layout(); plt.savefig("results/step2_stylized_facts.png", dpi=150); plt.show()
