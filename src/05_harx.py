"""
Step 5 - Robustness: does the ML gain come from the features or from non-linearity?
  HAR+dow  HAR + one dummy per day of the week of the target day
  HAR-X    linear regression on XGBoost's full feature set
Output: data/forecasts.csv (updated), results/step5_results.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

WINDOW = 1000
Path("results").mkdir(exist_ok=True)
df = pd.read_csv("data/btc_daily.csv", index_col=0, parse_dates=True)
fc = pd.read_csv("data/forecasts.csv", index_col=0, parse_dates=True)
test_idx = fc.index

F = pd.DataFrame(index=df.index)
for k in range(7):
    F[f"lag{k}"] = df["logrv"].shift(k)
F["weekly"] = df["logrv"].rolling(7).mean()
F["monthly"] = df["logrv"].rolling(30).mean()
F["volvol"] = df["logrv"].rolling(7).std()
F["ret"] = df["ret"]
F["absret"] = df["ret"].abs()
F["negret"] = df["ret"].clip(upper=0)
nxt = (df.index + pd.Timedelta(days=1)).dayofweek
DOW = [f"dow{d}" for d in range(1, 7)]                   # Monday = reference day
for d in range(1, 7):
    F[f"dow{d}"] = (nxt == d).astype(float)
F["target"] = df["logrv"].shift(-1)


def lin_forecast(cols):
    preds = []
    for t in test_idx:
        pos = F.index.get_loc(t)
        tr = F.iloc[pos - WINDOW:pos].dropna()
        X = np.column_stack([np.ones(len(tr)), tr[cols].values])
        beta, *_ = np.linalg.lstsq(X, tr["target"].values, rcond=None)
        s2 = np.var(tr["target"].values - X @ beta)
        preds.append(np.exp(np.r_[1.0, F.loc[t, cols].values] @ beta + 0.5 * s2))
    return preds


fc["HAR+dow"] = lin_forecast(["lag0", "weekly", "monthly"] + DOW)
fc["HAR-X"] = lin_forecast([f"lag{k}" for k in range(7)] +
                           ["weekly", "monthly", "volvol", "ret", "absret", "negret"] + DOW)
fc.to_csv("data/forecasts.csv")


def qlike_loss(a, f):
    return a / f - np.log(a / f) - 1


def dm_p(l1, l2, lag=5):
    d = (l1 - l2).values; n = len(d); dc = d - d.mean()
    v = dc @ dc / n + 2 * sum((1 - h / (lag + 1)) * (dc[h:] @ dc[:-h]) / n for h in range(1, lag + 1))
    t = d.mean() / np.sqrt(v / n)
    return f"{t:+.2f} (p={2 * (1 - norm.cdf(abs(t))):.3f})"


a = fc["actual"]
names = ["Naive", "GARCH", "HAR", "HAR+weekend", "HAR+dow", "HAR-X", "XGBoost", "LSTM", "Ensemble"]
L = {m: qlike_loss(a, fc[m]) for m in names}
res = pd.DataFrame({
    "QLIKE": {m: l.mean() for m, l in L.items()},
    "MSE log": {m: np.mean((np.log(a) - np.log(fc[m])) ** 2) for m in L},
    "DM vs HAR+weekend": {m: dm_p(l, L["HAR+weekend"]) if m != "HAR+weekend" else "-" for m, l in L.items()},
    "DM vs XGBoost": {m: dm_p(l, L["XGBoost"]) if m != "XGBoost" else "-" for m, l in L.items()},
})
print(res.round(4).to_string())
print("\n(DM t < 0 = row model better than the reference)")
res.to_csv("results/step5_results.csv")
