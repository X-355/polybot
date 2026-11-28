"""Freeze the 15-minute ALT/BTC model for the live paper bot.

1. Out-of-sample check in the way the live bot will trade: model trained on 2021-2023 only, run on every candle
   of 2024+ with at most one open position per pair (a new trade only after the last one ended). This gives the
   expected trade rate that the live activity check is measured against.
2. Final model: same settings, trained on all history. Saved with reference rows so the bot can prove at start-up
   that its own feature code and model give identical numbers.
Outputs: altbot/model/alt15.joblib, altbot/model/alt15_meta.json, research/results/altbtc/train_live.json
"""
import json, os, sys
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(__file__))
import altbtc_fast as A

H = 3                        # 3 candles = 15 minutes
BASE_COST = 0.003            # fees 0.2% round trip + 0.1% slippage allowance
THR_MULTS = (1.0, 1.5)
PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=1.0, random_state=1)
MODEL_DIR, RES_DIR = "altbot/model", "research/results/altbtc"


def live_like(te, pred, mult):
    """Every candle, one position per pair at a time, enter one candle late (conservative)."""
    out = []
    for p, g in te.assign(pred=pred).groupby("pair"):
        g = g.sort_index(); busy_until = pd.Timestamp(0)
        for t, r in zip(g.index, g.itertuples()):
            if t < busy_until:
                continue
            c = BASE_COST + r.tick_pct
            if abs(r.pred) > mult * c:
                side = 1 if r.pred > 0 else -1
                out.append((t, p, side * (np.exp(r.yd3) - 1) - c))
                busy_until = t + pd.Timedelta(minutes=5 * (H + 1))
    return pd.DataFrame(out, columns=["t", "pair", "net"])


def main():
    os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(RES_DIR, exist_ok=True)
    ticks = A.tick_sizes()
    btc = A.load("BTCUSDT")
    frames, raw = [], {}
    for p in A.PAIRS:
        d = A.load(p)
        if d is None or len(d) < 50000:
            print("skip", p); continue
        f = A.features(d, btc); f["pair"] = p
        f["tick_pct"] = (ticks.get(p) or float(d.close.diff().abs().replace(0, np.nan).min())) / d.close
        frames.append(f); raw[p] = d
        print(p, len(d), flush=True)
    F = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
    X = [c for c in F.columns if not c.startswith("y") and c not in ("pair", "tick_pct")]
    D = F.dropna(subset=X + ["y3", "yd3"])
    tr, te = D[D.index < A.SPLIT], D[D.index >= A.SPLIT]
    m0 = HistGradientBoostingRegressor(**PARAMS).fit(tr[X], tr["y3"])
    pred = m0.predict(te[X])
    days = max(1, (te.index.max() - te.index.min()).days)
    res = dict(test_from=str(te.index.min()), test_to=str(te.index.max()), days=days, rules={})
    for mult in THR_MULTS:
        L = live_like(te, pred, mult)
        n = len(L)
        cl = L.groupby("t").net.mean()
        se = float(cl.std(ddof=1) / np.sqrt(len(cl))) if len(cl) > 1 else None
        res["rules"][f"{mult}x"] = dict(trades=n, per_day=round(n / days, 3), net_pct=round(float(L.net.mean() * 100), 4),
                                        t_cluster=round(float(cl.mean() / se), 2) if se else None,
                                        by_year={int(y): dict(trades=int(len(g)), net_pct=round(float(g.net.mean() * 100), 3))
                                                 for y, g in L.groupby(L.t.dt.year)},
                                        by_pair={p: round(float(g.net.mean() * 100), 3) for p, g in L.groupby("pair")})
        print(mult, res["rules"][f"{mult}x"], flush=True)
    # final model on everything
    m = HistGradientBoostingRegressor(**PARAMS).fit(D[X], D["y3"])
    joblib.dump(m, f"{MODEL_DIR}/alt15.joblib")
    # reference rows: raw candle windows + expected features + prediction (last usable candle for 3 pairs)
    refs = []
    for p in list(raw)[:3]:
        d = raw[p]; row = F[(F.pair == p)].dropna(subset=X).iloc[-1]
        t = row.name; i = d.index.get_loc(t)
        w = d.iloc[i - 60:i + 1]
        b = btc.close.reindex(w.index)
        refs.append(dict(pair=p, t=str(t), close=w.close.tolist(), qvol=w.qvol.tolist(), tbq=w.tbq.tolist(),
                         btc=b.tolist(), features=[float(row[c]) for c in X],
                         pred=float(m.predict(pd.DataFrame([row[X]]))[0])))
    meta = dict(features=X, horizon_candles=H, base_cost=BASE_COST, trained_rows=len(D), trained_to=str(D.index.max()),
                pairs=sorted(raw), tick_sizes=ticks, params=PARAMS, refs=refs, oos=res)
    json.dump(meta, open(f"{MODEL_DIR}/alt15_meta.json", "w"), indent=1)
    json.dump(res, open(f"{RES_DIR}/train_live.json", "w"), indent=1)


if __name__ == "__main__":
    main()
