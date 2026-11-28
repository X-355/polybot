"""ALT/BTC fast test: can a gradient-tree model predict the next 5 minutes / 15 minutes / 1 hour of liquid
Binance ALT/BTC pairs well enough to beat trading costs?

Data: Binance public archive, 5-minute candles, Jan 2021 - last full month, for the most traded BTC pairs
plus BTCUSDT (so the model can see what Bitcoin itself is doing).
Features (known at the candle's close): the pair's returns over 1/3/6/12/48 candles, volatility over 12 and 48
candles, volume vs its 48-candle average, share of volume from buyers, BTCUSDT returns over 1/3/12 candles,
hour of day. Target: the pair's return over the next H candles (H = 1, 3, 12).
Train: 2021-2023. Test: 2024 onward, never seen in training.
Trading rule: buy the ALT (with BTC) when the predicted return is above the round-trip cost, or sell it when it is
below minus the cost (an existing holder switching to BTC, i.e. the mirror trade); hold H candles.
Costs per round trip: 0.3% (0.1% fee + 0.05% slippage each way) and an optimistic 0.2%.
Control: the same number of trades at random times, same pairs, same holding period.
"""
import io, json, os, re, sys, time, zipfile, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

OUT = sys.argv[1] if len(sys.argv) > 1 else "research/results/altbtc"
DL = "https://data.binance.vision/data/spot/monthly/klines"
PAIRS = ["ETHBTC", "BNBBTC", "SOLBTC", "XRPBTC", "ADABTC", "DOGEBTC", "LTCBTC", "LINKBTC", "DOTBTC", "AVAXBTC"]
SPLIT = pd.Timestamp("2024-01-01")
HS = (1, 3, 12)
COSTS = (0.003, 0.002)


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "altbtc-fast"}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** i)
        except Exception:
            time.sleep(2 ** i)
    return None


def months():
    out, d = [], pd.Timestamp("2021-01-01")
    last = (pd.Timestamp.utcnow().tz_localize(None).replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")
    while d.strftime("%Y-%m") <= last:
        out.append(d.strftime("%Y-%m")); d += pd.DateOffset(months=1)
    return out


def load_month(sym, m):
    b = get(f"{DL}/{sym}/5m/{sym}-5m-{m}.zip")
    if not b:
        return None
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        raw = z.read(z.namelist()[0])
    first = raw.split(b",", 1)[0].strip()
    d = pd.read_csv(io.BytesIO(raw), header=None if first.isdigit() else 0).iloc[:, [0, 4, 7, 10]]
    d.columns = ["t", "close", "qvol", "tbq"]
    t = pd.to_numeric(d.t)
    d["t"] = pd.to_datetime(t, unit="us" if t.max() > 1e14 else "ms")
    return d


def load(sym):
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(lambda m: load_month(sym, m), months()) if p is not None]
    if not parts:
        return None
    d = pd.concat(parts).drop_duplicates("t").sort_values("t").set_index("t")
    return d


def features(d, btc):
    f = pd.DataFrame(index=d.index)
    lr = np.log(d.close).diff()
    for k in (1, 3, 6, 12, 48):
        f[f"r{k}"] = np.log(d.close / d.close.shift(k))
    f["vol12"] = lr.rolling(12).std(); f["vol48"] = lr.rolling(48).std()
    f["vratio"] = d.qvol / d.qvol.rolling(48).mean()
    f["buyshare"] = (d.tbq / d.qvol).clip(0, 1)
    b = btc.close.reindex(d.index)
    for k in (1, 3, 12):
        f[f"btc{k}"] = np.log(b / b.shift(k))
    f["hour"] = d.index.hour
    ts = pd.Series(d.index, index=d.index)
    for h in HS:
        y = np.log(d.close.shift(-h) / d.close)
        yd = np.log(d.close.shift(-h - 1) / d.close.shift(-1))       # enter one candle LATER (no use of the signal candle's close)
        ok = (ts.shift(-h - 1) - ts) == pd.Timedelta(minutes=5 * (h + 1))   # no missing candles in the window
        f[f"y{h}"] = y.where(ok); f[f"yd{h}"] = yd.where(ok)
    return f


def tick_sizes():
    """Smallest price step per pair, from Binance's public market info (mirror that works from cloud servers)."""
    b = get("https://data-api.binance.vision/api/v3/exchangeInfo?symbols=" + json.dumps(PAIRS).replace(" ", ""))
    out = {}
    if b:
        for s in json.loads(b).get("symbols", []):
            for f in s.get("filters", []):
                if f.get("filterType") == "PRICE_FILTER":
                    out[s["symbol"]] = float(f["tickSize"])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    ticks = tick_sizes()
    print("tick sizes", ticks, flush=True)
    btc = load("BTCUSDT")
    frames = []
    for p in PAIRS:
        d = load(p)
        if d is None or len(d) < 50000:
            print("skip", p); continue
        f = features(d, btc); f["pair"] = p
        # one price step as a share of the price: the least you lose crossing the spread (buy at ask, sell at bid)
        tk = ticks.get(p) or float(d.close.diff().abs().replace(0, np.nan).min())
        f["tick_pct"] = tk / d.close
        frames.append(f)
        print(p, len(d), "candles", d.index[0], "->", d.index[-1], flush=True)
    F = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
    X = [c for c in F.columns if not c.startswith("y") and c not in ("pair", "tick_pct")]
    rng = np.random.default_rng(7)
    res = dict(pairs=sorted(F.pair.unique().tolist()), split=str(SPLIT.date()), first=str(F.index.min()), last=str(F.index.max()),
               features=X, tick_sizes=ticks, horizons={},
               median_tick_pct={p: round(float(F[F.pair == p].tick_pct.median() * 100), 4) for p in F.pair.unique()})
    for h in HS:
        D = F.dropna(subset=X + [f"y{h}", f"yd{h}"])
        # every h-th candle so trades do not overlap within a pair
        D = D[(D.index.minute // 5 + D.index.hour * 12) % h == 0]
        tr, te = D[D.index < SPLIT], D[D.index >= SPLIT]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
                                          l2_regularization=1.0, random_state=1).fit(tr[X], tr[f"y{h}"])
        pred = m.predict(te[X]); y = te[f"y{h}"].to_numpy(); yd = te[f"yd{h}"].to_numpy()
        yrs = te.index.year.to_numpy()
        corr = float(np.corrcoef(pred, y)[0, 1])
        hit = float(np.mean(np.sign(pred) == np.sign(y)))
        out = dict(train_rows=len(tr), test_rows=len(te), corr=round(corr, 4), direction_hit=round(hit, 4),
                   typical_move_pct=round(float(np.median(np.abs(y)) * 100), 3), rules={})
        tick = te.tick_pct.to_numpy()
        out["share_of_moves_within_one_tick"] = round(float(np.mean(np.abs(np.exp(y) - 1) <= tick * 1.01)), 4)
        for cost in COSTS:
            for k in (1.0, 1.5, 2.0):
                cvec = cost + tick                              # fees + slippage + one full price step of spread
                thr = cvec * k
                side = np.where(pred > thr, 1, np.where(pred < -thr, -1, 0))
                sel = side != 0
                n = int(sel.sum())
                if n < 30:
                    out["rules"][f"cost {cost*100:.1f}%+tick thr {k}x"] = dict(trades=n); continue
                gross = side[sel] * (np.exp(y[sel]) - 1)
                net = gross - cvec[sel]
                # control: same number of random trades with random sides
                idx = rng.choice(len(y), n, replace=False); rs = rng.choice([-1, 1], n)
                ctl = rs * (np.exp(y[idx]) - 1) - cvec[idx]
                t = float(net.mean() / (net.std(ddof=1) / np.sqrt(n))) if net.std() > 0 else None
                by_pair = {p: round(float(net[(te.pair.to_numpy()[sel] == p)].mean() * 100), 3) for p in res["pairs"]
                           if (te.pair.to_numpy()[sel] == p).sum() >= 20}
                g_sorted = np.sort(net)[::-1]
                net_d = side[sel] * (np.exp(yd[sel]) - 1) - cvec[sel]
                td = float(net_d.mean() / (net_d.std(ddof=1) / np.sqrt(n))) if net_d.std() > 0 else None
                by_year = {int(yr): dict(trades=int((yrs[sel] == yr).sum()), net_pct=round(float(net[yrs[sel] == yr].mean() * 100), 3),
                                         delayed_net_pct=round(float(net_d[yrs[sel] == yr].mean() * 100), 3))
                           for yr in np.unique(yrs[sel])}
                out["rules"][f"cost {cost*100:.1f}%+tick thr {k}x"] = dict(
                    delayed_entry_net_pct=round(float(net_d.mean() * 100), 4), delayed_t=round(td, 2) if td else None,
                    by_year=by_year,
                    median_net_pct=round(float(np.median(net) * 100), 4),
                    net_without_best_5pct=round(float(g_sorted[int(len(g_sorted) * 0.05):].mean() * 100), 4),
                    avg_tick_pct=round(float(tick[sel].mean() * 100), 4),
                    trades=n, per_day=round(n / max(1, (te.index.max() - te.index.min()).days), 1),
                    win_rate=round(float((net > 0).mean()), 4), gross_pct=round(float(gross.mean() * 100), 4),
                    net_pct=round(float(net.mean() * 100), 4), t=round(t, 2) if t else None,
                    random_control_net_pct=round(float(ctl.mean() * 100), 4), by_pair_net_pct=by_pair)
        res["horizons"][f"{h * 5} min"] = out
        print(f"--- {h*5} min: corr {corr:.4f} hit {hit:.3f} typical move {out['typical_move_pct']}%")
        for k2, v in out["rules"].items():
            if v.get("trades", 0) >= 30:
                print(f"   {k2:<22} trades {v['trades']:>7} ({v['per_day']}/day) win {v['win_rate']:.3f} gross {v['gross_pct']:+.3f}% "
                      f"net {v['net_pct']:+.3f}% t={v['t']} | delayed {v['delayed_entry_net_pct']:+.3f}% t={v['delayed_t']} | random {v['random_control_net_pct']:+.3f}%", flush=True)
    json.dump(res, open(f"{OUT}/altbtc_fast.json", "w"), indent=1)


if __name__ == "__main__":
    main()
