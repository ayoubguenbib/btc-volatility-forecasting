"""
Step 4 - Machine-learning forecasts (XGBoost, LSTM) vs the econometric baselines.
Same rolling protocol as step 3: 1000-day window, refit every 60 days,
early stopping on the last 200 days of each window, no look-ahead.
Evaluation: MSE, QLIKE, MSE on logs, Diebold-Mariano test vs HAR+weekend.
Output: data/forecasts.csv (updated), results/step4_*.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.stats import norm

WINDOW, VAL, REFIT, SEQ = 1000, 200, 60, 30
torch.manual_seed(0); np.random.seed(0)

Path("results").mkdir(exist_ok=True)
df = pd.read_csv("data/btc_daily.csv", index_col=0, parse_dates=True)
fc = pd.read_csv("data/forecasts.csv", index_col=0, parse_dates=True)
test_idx = fc.index

# Features known at the end of day t; target = log RV(t+1)
F = pd.DataFrame(index=df.index)
for k in range(7):
    F[f"lag{k}"] = df["logrv"].shift(k)
F["weekly"] = df["logrv"].rolling(7).mean()
F["monthly"] = df["logrv"].rolling(30).mean()
F["volvol"] = df["logrv"].rolling(7).std()
F["ret"] = df["ret"]
F["absret"] = df["ret"].abs()
F["negret"] = df["ret"].clip(upper=0)                 # leverage effect
nxt = df.index + pd.Timedelta(days=1)
F["dow_next"] = nxt.dayofweek
F["weekend"] = (nxt.dayofweek >= 5).astype(float)
F["target"] = df["logrv"].shift(-1)
FEATS = [c for c in F.columns if c != "target"]


# ---------------------------------------------------------------- XGBoost
def xgb_forecast():
    preds = pd.Series(index=test_idx, dtype=float)
    for k in range(0, len(test_idx), REFIT):
        block = test_idx[k:k + REFIT]
        pos = F.index.get_loc(block[0])
        tr = F.iloc[pos - WINDOW:pos].dropna()
        fit, val = tr.iloc[:-VAL], tr.iloc[-VAL:]
        m = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                             early_stopping_rounds=50)
        m.fit(fit[FEATS], fit["target"], eval_set=[(val[FEATS], val["target"])], verbose=False)
        s2 = np.var(val["target"] - m.predict(val[FEATS]))
        preds[block] = np.exp(m.predict(F.loc[block, FEATS]) + 0.5 * s2)
    return preds, m


# ---------------------------------------------------------------- LSTM
X_all = F[FEATS].values.astype(np.float32)
y_all = F["target"].values.astype(np.float32)


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.lstm = nn.LSTM(n, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


def seqs(idx, mu, sd):
    return torch.tensor(np.stack([(X_all[i - SEQ + 1:i + 1] - mu) / sd for i in idx]))


def lstm_forecast():
    preds = pd.Series(index=test_idx, dtype=float)
    n_refit = -(-len(test_idx) // REFIT)
    for k in range(0, len(test_idx), REFIT):
        block = test_idx[k:k + REFIT]
        pos = F.index.get_loc(block[0])
        idx = [i for i in range(pos - WINDOW, pos) if i >= SEQ - 1
               and not np.isnan(X_all[i - SEQ + 1:i + 1]).any() and not np.isnan(y_all[i])]
        fit_i, val_i = idx[:-VAL], idx[-VAL:]
        mu, sd = X_all[fit_i].mean(0), X_all[fit_i].std(0) + 1e-8
        ym, ys = y_all[fit_i].mean(), y_all[fit_i].std()
        Xf, Xv = seqs(fit_i, mu, sd), seqs(val_i, mu, sd)
        yf = torch.tensor((y_all[fit_i] - ym) / ys)
        yv = torch.tensor((y_all[val_i] - ym) / ys)
        net = Net(len(FEATS))
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        best, state, wait = np.inf, None, 0
        for _ in range(300):
            net.train()
            for b in torch.randperm(len(Xf)).split(64):
                opt.zero_grad()
                loss = ((net(Xf[b]) - yf[b]) ** 2).mean()
                loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = ((net(Xv) - yv) ** 2).mean().item()
            if v < best:
                best, wait, state = v, 0, {n: t.clone() for n, t in net.state_dict().items()}
            else:
                wait += 1
                if wait >= 20:
                    break
        net.load_state_dict(state); net.eval()
        with torch.no_grad():
            s2 = np.var(net(Xv).numpy() * ys + ym - y_all[val_i])
            pb = [F.index.get_loc(t) for t in block]
            preds[block] = np.exp(net(seqs(pb, mu, sd)).numpy() * ys + ym + 0.5 * s2)
        print(f"LSTM refit {k // REFIT + 1}/{n_refit}", end="\r")
    print()
    return preds


fc["XGBoost"], last_xgb = xgb_forecast()
fc["LSTM"] = lstm_forecast()
fc["Ensemble"] = fc[["HAR+weekend", "XGBoost", "LSTM"]].mean(axis=1)
fc.to_csv("data/forecasts.csv")


# ---------------------------------------------------------------- evaluation
def qlike_loss(a, f):
    return a / f - np.log(a / f) - 1


def dm(l1, l2, lag=5):
    """Diebold-Mariano test with a Newey-West (Bartlett) long-run variance."""
    d = (l1 - l2).values; n = len(d); dc = d - d.mean()
    v = dc @ dc / n + 2 * sum((1 - h / (lag + 1)) * (dc[h:] @ dc[:-h]) / n for h in range(1, lag + 1))
    t = d.mean() / np.sqrt(v / n)
    return t, 2 * (1 - norm.cdf(abs(t)))


a = fc["actual"]
ref = qlike_loss(a, fc["HAR+weekend"])
models = ["Naive", "GARCH", "HAR", "HAR+weekend", "XGBoost", "LSTM", "Ensemble"]
rows = []
for m in models:
    l = qlike_loss(a, fc[m])
    t, p = (np.nan, np.nan) if m == "HAR+weekend" else dm(l, ref)
    rows.append([1e6 * np.mean((a - fc[m]) ** 2), l.mean(),
                 np.mean((np.log(a) - np.log(fc[m])) ** 2), t, p])
res = pd.DataFrame(rows, index=models,
                   columns=["MSE (x1e6)", "QLIKE", "MSE log", "DM t vs HAR+w", "p-value"])
print(f"Out-of-sample {test_idx[0].date()} -> {test_idx[-1].date()} ({len(test_idx)} days)")
print(res.round(4))
print("\n(DM t < 0 = better than HAR+weekend; significant if p < 0.05)\n")
res.to_csv("results/step4_results.csv")

imp = pd.Series(last_xgb.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
print("XGBoost feature importance (gain, last refit):")
print(imp.round(2).head(8))

z = fc.loc["2024-01-01":"2024-06-30"]
ann = lambda s: 100 * np.sqrt(365 * s)
plt.figure(figsize=(12, 4))
plt.plot(ann(z["actual"]), color="grey", lw=1, label="Realized")
for m in ["HAR+weekend", "XGBoost", "LSTM"]:
    plt.plot(ann(z[m]), lw=1.3, label=m)
plt.title("Next-day volatility forecasts, H1 2024 (annualized %)"); plt.legend()
plt.tight_layout(); plt.savefig("results/step4_forecasts.png", dpi=150); plt.show()
