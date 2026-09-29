"""
Step 3 - Econometric baselines for next-day realized variance.
  Naive        RV(t+1) = RV(t)
  GARCH(1,1)   on daily returns, Student-t innovations
  HAR          log RV(t+1) = b0 + bd*daily + bw*weekly + bm*monthly   (Corsi, 2009)
  HAR+weekend  HAR + dummy "tomorrow is Saturday or Sunday"
Rolling out-of-sample forecasts from 2023 with a 1000-day window (no look-ahead).
Output: data/forecasts.csv, results/step3_*.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model

TEST_START = "2023-01-01"
WINDOW = 1000          # training days (~2.7 years), rolling
REFIT_GARCH = 20       # GARCH re-estimated every 20 days (HAR is refit daily)

Path("results").mkdir(exist_ok=True)
df = pd.read_csv("data/btc_daily.csv", index_col=0, parse_dates=True)

# Features known at the end of day t; target = log RV of day t+1
F = pd.DataFrame(index=df.index)
F["daily"] = df["logrv"]
F["weekly"] = df["logrv"].rolling(7).mean()      # crypto trades 7 days a week
F["monthly"] = df["logrv"].rolling(30).mean()
F["weekend"] = (pd.Series(df.index.dayofweek, index=df.index).shift(-1) >= 5).astype(float)
F["target"] = df["logrv"].shift(-1)

# In-sample HAR+weekend on the full sample, for interpretation only
full = F.dropna()
ols = sm.OLS(full["target"], sm.add_constant(full[["daily", "weekly", "monthly", "weekend"]])).fit()
print("In-sample HAR+weekend (full sample, interpretation only)")
print(ols.summary().tables[1], f"\nR2 = {ols.rsquared:.3f}\n")

# Rolling out-of-sample forecasts
test_idx = F.index[(F.index >= TEST_START) & F["target"].notna()]
out = pd.DataFrame(index=test_idx)
out["actual"] = np.exp(F.loc[test_idx, "target"])
out["Naive"] = np.exp(F.loc[test_idx, "daily"])


def har_forecast(cols):
    preds = []
    for t in test_idx:
        pos = F.index.get_loc(t)
        tr = F.iloc[pos - WINDOW:pos].dropna()           # every target here is known at t
        X = np.column_stack([np.ones(len(tr)), tr[cols].values])
        beta, *_ = np.linalg.lstsq(X, tr["target"].values, rcond=None)
        s2 = np.var(tr["target"].values - X @ beta)
        mu = np.r_[1.0, F.loc[t, cols].values] @ beta
        preds.append(np.exp(mu + 0.5 * s2))               # E[RV] under a log-normal forecast
    return preds


out["HAR"] = har_forecast(["daily", "weekly", "monthly"])
out["HAR+weekend"] = har_forecast(["daily", "weekly", "monthly", "weekend"])

am = arch_model(100 * df["ret"], mean="Constant", vol="GARCH", p=1, q=1, dist="t")
garch = []
for k in range(0, len(test_idx), REFIT_GARCH):
    block = test_idx[k:k + REFIT_GARCH]
    pos = df.index.get_loc(block[0])
    res = am.fit(first_obs=pos - WINDOW + 1, last_obs=pos + 1, disp="off")
    fc = res.forecast(horizon=1, start=block[0], reindex=False)
    garch.append(fc.variance.loc[block, "h.1"] / 1e4)
out["GARCH"] = pd.concat(garch)
out.to_csv("data/forecasts.csv")


def qlike(a, f):
    return np.mean(a / f - np.log(a / f) - 1)


models = ["Naive", "GARCH", "HAR", "HAR+weekend"]
res = pd.DataFrame({
    "MSE (x1e6)": [1e6 * np.mean((out["actual"] - out[m]) ** 2) for m in models],
    "QLIKE": [qlike(out["actual"], out[m]) for m in models],
    "MSE log": [np.mean((np.log(out["actual"]) - np.log(out[m])) ** 2) for m in models],
}, index=models)
res["QLIKE vs Naive"] = (res["QLIKE"] / res.loc["Naive", "QLIKE"] - 1).map("{:+.0%}".format)
print(f"Out-of-sample {test_idx[0].date()} -> {test_idx[-1].date()} ({len(test_idx)} days)")
print(res.round(4))
res.to_csv("results/step3_results.csv")

z = out.loc["2024-01-01":"2024-06-30"]
ann = lambda s: 100 * np.sqrt(365 * s)
plt.figure(figsize=(12, 4))
plt.plot(ann(z["actual"]), color="grey", lw=1, label="Realized")
for m in ["GARCH", "HAR+weekend"]:
    plt.plot(ann(z[m]), lw=1.5, label=m)
plt.title("Next-day volatility forecasts, H1 2024 (annualized %)"); plt.legend()
plt.tight_layout(); plt.savefig("results/step3_forecasts.png", dpi=150); plt.show()
