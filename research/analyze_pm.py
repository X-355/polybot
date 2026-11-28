"""Are Polymarket's 5-minute Up/Down prices fair? Calibration + strategy tests, net of fees."""
import os, glob, json, math
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

COIN = os.environ.get("COIN", "BTC")
OUT = os.environ.get("OUT", "results/polymarket")
LAG_MAX = float(os.environ.get("LAG_MAX", "inf"))     # keep only snapshots whose recorded book was at most this far behind
os.makedirs(OUT, exist_ok=True)
FEES = {"no fee (maker)": 0.0, "fee 0.05": 0.05, "fee 0.072": 0.072}
MAIN_FEE = 0.05

files = sorted(glob.glob("pm_out/snap_*.csv.gz"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
infos = [json.load(open(f)) for f in sorted(glob.glob("pm_out/info_*.json"))]
failed = [json.load(open(f)) for f in sorted(glob.glob("pm_out/failed_*.json"))]
days = sorted(df.day.unique())
print("rows", len(df), "days", len(days))

# ---------- outcome: Polymarket's official resolution (Gamma); Chainlink only as a cross-check
df["y"] = df["res"]
agree = {}
for c in ("res_meta",):
    m = df.res.notna() & df[c].notna()
    agree[c] = round(float((df.loc[m, "res"] == df.loc[m, c]).mean()), 4) if m.any() else None
df = df[df.y.notna()].copy()
df["y"] = df.y.astype(int)

# ---------- features known at the snapshot moment
df["mid_up"] = (df.up_bid + df.up_ask) / 2
# Resolution rule (verified against Polymarket's own priceToBeat/finalPrice): Up wins if the
# Chainlink 60-second TWAP at the end of the window >= the 60-second TWAP at its start (ptb).
# The end TWAP is ~ the average Chainlink price over the window's last 60 seconds.
df["tl"] = (300 - df.cl_now_sec).clip(lower=1)                 # seconds of price path still unknown
S, sig = df.cl_now, df.vol_s
k = (60 - df.tl).clip(lower=0)                                  # seconds of the final-TWAP window already seen
A = df.win_avg.fillna(S)
mean_final = np.where(df.tl >= 60, S, (k * A + df.tl * S) / 60)
sd_final = np.where(df.tl >= 60, sig * S * np.sqrt((df.tl - 40).clip(lower=1)),
                    sig * S * np.sqrt(df.tl / 3) * df.tl / 60)
df["x"] = (S - df.ptb) / df.ptb
df["xf"] = (mean_final - df.ptb) / df.ptb
df["sdf"] = sd_final / df.ptb
df["z"] = (mean_final - df.ptb) / np.maximum(sd_final, 1e-9)
from math import erf
df["p_digital"] = 0.5 * (1 + np.vectorize(erf)(df.z.fillna(0).clip(-30, 30) / math.sqrt(2)))
df["spr_up"] = df.up_ask - df.up_bid
df["spr_dn"] = df.dn_ask - df.dn_bid
df["overround"] = df.up_ask + df.dn_ask - 1
df["imb"] = np.log((df.up_ask_sz + 1) / (df.dn_ask_sz + 1))
ok = (df.up_age < 30) & (df.dn_age < 30) & df.vol_s.notna() & (df.up_ask <= 1) & (df.dn_ask <= 1)
lag_info = dict(lag_max=LAG_MAX, share_over_2s=round(float((df.book_lag5 > 2).mean()), 4),
                share_over_2s_last30=round(float((df[df.left <= 30].book_lag5 > 2).mean()), 4),
                median_lag=round(float(df.book_lag.median()), 3), p95_lag5=round(float(df.book_lag5.quantile(0.95)), 3))
print("LAG", lag_info)
if LAG_MAX != float("inf"):
    keep = df.groupby(["start", "left"]).book_lag5.transform("max") <= LAG_MAX   # all three fill moments must be fresh
    ok &= keep.fillna(False)
df = df[ok].copy()

base = df[df.lag == 0].copy()
key = ["start", "left"]
fills = {L: df[df.lag == L].set_index(key)[["up_ask", "dn_ask"]] for L in (0, 1, 2)}

# ---------- train / test split by day (first 60% of days train, last 40% test)
SPLIT = os.environ.get("SPLIT", "forward")
if SPLIT == "reverse":      # train on the last 60% of days, test on the first 40%
    cut = days[int(len(days) * 0.4)]
    base["test"] = base.day < cut
else:
    cut = days[int(len(days) * 0.6)]
    base["test"] = base.day >= cut
FEATS = ["x", "xf", "sdf", "vol_s", "tl", "z", "mid_up", "spr_up", "spr_dn", "imb", "overround"]
tr = base[~base.test]
print("split", SPLIT, "train days", tr.day.nunique(), "test days", base[base.test].day.nunique())
gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                     min_samples_leaf=200, l2_regularization=1.0, random_state=1)
gbm.fit(tr[FEATS], tr.y)
base["p_gbm"] = gbm.predict_proba(base[FEATS])[:, 1]


def brier(p, y):
    return float(np.mean((np.clip(p, 0, 1) - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------- 1) forecast quality: does anything beat the market's own price?
forecast = []
for left in sorted(base.left.unique(), reverse=True):
    t = base[(base.left == left) & base.test]
    row = dict(left=int(left), n=len(t))
    for name, col in (("market", "mid_up"), ("digital", "p_digital"), ("gbm", "p_gbm")):
        row[f"brier_{name}"] = round(brier(t[col], t.y), 5)
        row[f"logloss_{name}"] = round(logloss(t[col], t.y), 5)
    forecast.append(row)

# ---------- 2) calibration of the market price
calib = []
bins = np.linspace(0, 1, 11)
for left in sorted(base.left.unique(), reverse=True):
    t = base[base.left == left]
    idx = np.digitize(t.mid_up, bins[1:-1])
    for b in range(10):
        s = t[idx == b]
        if len(s) < 30:
            continue
        n, k = len(s), int(s.y.sum())
        ph = k / n; zc = 1.96
        den = 1 + zc ** 2 / n
        c = (ph + zc ** 2 / (2 * n)) / den
        h = zc * math.sqrt(ph * (1 - ph) / n + zc ** 2 / (4 * n ** 2)) / den
        calib.append(dict(left=int(left), bin=f"{bins[b]:.1f}-{bins[b+1]:.1f}", n=n,
                          price=round(float(s.mid_up.mean()), 4), won=round(ph, 4),
                          lo=round(c - h, 4), hi=round(c + h, 4)))

# ---------- 3) strategies: buy one side at the ask (taker), hold to settlement
def run(sig, name, lag=0, only_test=False):
    """sig: Series indexed like base with +1 (buy Up), -1 (buy Down), 0 (no trade)."""
    t = base[sig != 0].copy()
    if only_test:
        t = t[t.test]
    t["side"] = sig[t.index]
    f = fills[lag]
    j = t.join(f, on=key, rsuffix="_fill", how="inner")
    price = np.where(j.side > 0, j.up_ask_fill, j.dn_ask_fill)
    win = np.where(j.side > 0, j.y == 1, j.y == 0).astype(float)
    res = dict(strategy=name, lag=lag, test_only=only_test, trades=int(len(j)))
    if len(j) < 20:
        return res
    res["win_rate"] = round(float(win.mean()), 4)
    res["avg_price"] = round(float(price.mean()), 4)
    for fname, coef in FEES.items():
        pnl = win - price - coef * price * (1 - price)
        sd = pnl.std(ddof=1)
        res[fname] = dict(cents_per_share=round(float(pnl.mean() * 100), 3),
                          return_pct=round(float(pnl.sum() / price.sum() * 100), 3),
                          t=round(float(pnl.mean() / (sd / math.sqrt(len(pnl)))), 2) if sd > 0 else None)
    return res


strategies = []
fee_mid = lambda p: MAIN_FEE * p * (1 - p)
for left in sorted(base.left.unique(), reverse=True):
    at = base.left == left
    up_m = base.mid_up > 0.5
    sigs = {
        "Random side (control)": None,
        "Always buy the favourite": np.where(at, np.where(up_m, 1, -1), 0),
        "Always buy the underdog": np.where(at, np.where(up_m, -1, 1), 0),
    }
    for margin in (0.02, 0.05):
        for model in ("digital", "gbm"):
            p = base[f"p_{model}"]
            eu = p - base.up_ask - fee_mid(base.up_ask)
            ed = (1 - p) - base.dn_ask - fee_mid(base.dn_ask)
            s = np.where(at & (eu > margin) & (eu >= ed), 1, np.where(at & (ed > margin), -1, 0))
            label = {"digital": "Price-vs-start model", "gbm": "Gradient tree model"}[model]
            sigs[f"{label}, edge > {int(margin*100)}c"] = s
    for name, s in sigs.items():
        if s is None:
            # exact expectation of a coin flip: average of buying Up and buying Down
            r_up = run(pd.Series(np.where(at, 1, 0), index=base.index), name, 0, True)
            r_dn = run(pd.Series(np.where(at, -1, 0), index=base.index), name, 0, True)
            if r_up.get("trades", 0) >= 20:
                r = dict(strategy=name, lag=0, test_only=True, trades=r_up["trades"],
                         win_rate=round((r_up["win_rate"] + r_dn["win_rate"]) / 2, 4),
                         avg_price=round((r_up["avg_price"] + r_dn["avg_price"]) / 2, 4))
                for fname in FEES:
                    r[fname] = dict(cents_per_share=round((r_up[fname]["cents_per_share"] + r_dn[fname]["cents_per_share"]) / 2, 3),
                                    return_pct=round((r_up[fname]["return_pct"] + r_dn[fname]["return_pct"]) / 2, 3), t=None)
                strategies.append(dict(left=int(left), **r))
            continue
        s = pd.Series(s, index=base.index)
        for lag in (0, 1, 2):
            strategies.append(dict(left=int(left), **run(s, name, lag, True)))

# ---------- market structure facts
struct = dict(
    median_overround_c=round(float(base.overround.median() * 100), 2),
    median_spread_up_c=round(float(base.spr_up.median() * 100), 2),
    median_ask_size_shares=round(float(base.up_ask_sz.median()), 1),
    median_depth_2c_shares=round(float(base.up_depth2.median()), 1),
)
fee_seen = {}
for i in infos:
    for k, v in i.get("fee_rate_bps", {}).items():
        fee_seen[k] = fee_seen.get(k, 0) + v

out = dict(coin=COIN, days=len(days), first_day=days[0], last_day=days[-1], test_from=cut,
           markets=int(base.start.nunique()), snapshots=int(len(base)),
           outcome_agreement_with_chainlink=agree, gbm_features=FEATS,
           forecast=forecast, calibration=calib, strategies=strategies, structure=struct,
           trade_fee_rate_bps_seen=fee_seen, failed_days=failed,
           extract=[dict(day=i["day"], markets=i["markets"], secs=i["secs"],
                         snaps=i["stats"].get("snap_ok"), missing=i["stats"].get("snap_missing"),
                         chk=i["stats"].get("chk"), chk_mismatch=i["stats"].get("chk_mismatch"))
                    for i in infos])
out["split"] = SPLIT
out["lag"] = lag_info
tag = os.environ.get("TAG", "")
json.dump(out, open(f"{OUT}/pm_study_{COIN}{tag}" + ("" if SPLIT == "forward" else "_" + SPLIT) + ".json", "w"), indent=1)

# readable summary
print(json.dumps({k: out[k] for k in ("days", "first_day", "last_day", "test_from", "markets", "snapshots",
                                      "outcome_agreement_with_chainlink", "structure", "trade_fee_rate_bps_seen")}, indent=1))
print("FORECAST (test days, lower is better)")
for r in forecast:
    print(r)
print("STRATEGIES (test days)")
for r in strategies:
    if r.get("trades", 0) >= 20:
        f = r["fee 0.05"]; z = r["no fee (maker)"]
        print(f"{r['left']:>4}s left lag{r['lag']} {r['strategy'][:38]:<38} n={r['trades']:>6} win={r['win_rate']:.3f} "
              f"price={r['avg_price']:.3f} nofee={z['cents_per_share']:+.2f}c fee={f['cents_per_share']:+.2f}c ret={f['return_pct']:+.2f}% t={f['t']}")
