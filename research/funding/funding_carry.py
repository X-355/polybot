"""Funding-rate carry test (Binance USDT perpetuals) - research only, no trading.

The trade: hold $1 of a coin on spot AND short $1 of the same coin's perpetual future. Price moves cancel out
(except small gaps between the spot and futures price, which we count). The short side receives the funding
rate when it is positive and pays it when it is negative.

Data (Binance public archive): funding-rate history, daily spot and futures candles.
Costs: entering or leaving = spot fee 0.1% + futures taker fee 0.05% + 0.05% slippage on each leg's side
       -> 0.25% of notional each way, 0.5% round trip.
Returns are per $1 of notional. Capital needed is more than $1 (the spot coin plus margin for the short);
we report returns on 2x notional too (spot fully paid + 100% margin - very safe, no liquidation unless price doubles).

Rules tested (decided before running):
  always     : hold every coin all the time
  filter X   : hold a coin only while its average funding over the previous 7 days is above X a year (X = 0%, 5%, 10%, 20%)
  top3       : each Monday hold the 3 coins with the highest previous-7-day funding (if above 0)
Reported by year, plus worst drawdown, share of negative-funding days, and the biggest 1-day and 3-day price spikes
(how much margin the short side would have needed).
"""
import io, json, os, sys, time, zipfile, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "research/results/funding"
DL = "https://data.binance.vision/data"
COINS = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "LINK", "LTC", "AVAX", "DOT"]
COST_SIDE = 0.0025
FILTERS = (0.0, 0.05, 0.10, 0.20)


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "funding-test"}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** i)
        except Exception:
            time.sleep(2 ** i)
    return None


def months(start="2019-09"):
    out, d = [], pd.Timestamp(start + "-01")
    last = (pd.Timestamp.utcnow().tz_localize(None).replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")
    while d.strftime("%Y-%m") <= last:
        out.append(d.strftime("%Y-%m")); d += pd.DateOffset(months=1)
    return out


def read_zip(b):
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        raw = z.read(z.namelist()[0])
    first = raw.split(b",", 1)[0].strip()
    return pd.read_csv(io.BytesIO(raw), header=None if first[:1].isdigit() else 0)


def ts(x):
    x = pd.to_numeric(x)
    return pd.to_datetime(x, unit="us" if x.max() > 1e14 else "ms")


def load_funding(sym):
    def one(m):
        b = get(f"{DL}/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip")
        if not b:
            return None
        d = read_zip(b)
        d = d.iloc[:, [0, -1]]; d.columns = ["t", "rate"]
        d["t"] = ts(d.t); return d
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(one, months()) if p is not None]
    if not parts:
        return None
    d = pd.concat(parts).drop_duplicates("t").sort_values("t")
    return d.set_index("t").rate.astype(float)


def load_klines(kind, sym):
    base = f"{DL}/spot/monthly/klines/{sym}/1d/{sym}-1d-" if kind == "spot" else f"{DL}/futures/um/monthly/klines/{sym}/1d/{sym}-1d-"
    def one(m):
        b = get(f"{base}{m}.zip")
        if not b:
            return None
        d = read_zip(b).iloc[:, [0, 2, 4]]; d.columns = ["t", "high", "close"]
        d["t"] = ts(d.t).dt.normalize(); return d
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(one, months()) if p is not None]
    if not parts:
        return None
    return pd.concat(parts).drop_duplicates("t").sort_values("t").set_index("t").astype(float)


def stats(r, held=None):
    """r: daily returns per $1 notional (portfolio). held: number of coins held per day (for reporting)."""
    r = r.dropna()
    if len(r) < 30:
        return None
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    yrs = len(r) / 365
    ann = float(eq.iloc[-1] ** (1 / yrs) - 1)
    out = dict(days=len(r), annual_pct=round(ann * 100, 2), annual_on_2x_capital_pct=round(float((1 + r / 2).prod() ** (1 / yrs) - 1) * 100, 2),
               worst_drawdown_pct=round(dd * 100, 2), worst_day_pct=round(float(r.min() * 100), 3),
               share_losing_days=round(float((r < 0).mean()), 3),
               by_year={int(y): round(float(((1 + g).prod() - 1) * 100), 2) for y, g in r.groupby(r.index.year)})
    if held is not None:
        out["avg_coins_held"] = round(float(held.reindex(r.index).mean()), 2)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    per = {}
    for c in COINS:
        sym = c + "USDT"
        f = load_funding(sym); s = load_klines("spot", sym); p = load_klines("fut", sym)
        if f is None or s is None or p is None:
            print("skip", c, f is None, s is None, p is None, flush=True); continue
        fd = f.groupby(f.index.normalize()).sum()                 # funding received by the short, per $1, per day
        nfund = f.groupby(f.index.normalize()).size()
        idx = s.index.intersection(p.index).intersection(fd.index)
        d = pd.DataFrame(dict(spot=s.close.reindex(idx), fut=p.close.reindex(idx), fhigh=p.high.reindex(idx), fund=fd.reindex(idx)))
        d["hedge"] = d.spot.pct_change() - d.fut.pct_change()     # long spot + short future, rebalanced daily
        d["day"] = d.hedge + d.fund
        d["spike1"] = d.fhigh / d.fut.shift(1) - 1
        d["spike3"] = d.fhigh.rolling(3).max() / d.fut.shift(3) - 1
        per[c] = d.iloc[1:]
        print(c, len(d), "days", d.index[0].date(), "->", d.index[-1].date(), "fund/day", round(d.fund.mean() * 100, 4), "% intervals/day", round(float(nfund.median()), 1), flush=True)
    res = dict(coins=list(per), cost_each_way_pct=COST_SIDE * 100, coins_detail={}, rules={})
    for c, d in per.items():
        res["coins_detail"][c] = dict(first=str(d.index[0].date()), last=str(d.index[-1].date()),
                                      funding_annual_pct=round(float(d.fund.mean() * 365 * 100), 2),
                                      share_negative_funding_days=round(float((d.fund < 0).mean()), 3),
                                      hedge_noise_daily_std_pct=round(float(d.hedge.std() * 100), 3),
                                      biggest_1day_spike_pct=round(float(d.spike1.max() * 100), 1),
                                      biggest_3day_spike_pct=round(float(d.spike3.max() * 100), 1),
                                      always_on=stats(d.day - 0.0))
    D = pd.DataFrame({c: d.day for c, d in per.items()})
    F = pd.DataFrame({c: d.fund for c, d in per.items()})
    trail = F.rolling(7, min_periods=7).mean().shift(1) * 365      # previous 7 days, annualised, known before the day starts
    # always on: equal weight over coins that exist that day, one entry cost when a coin first appears
    held = D.notna().astype(float)
    def run(hold):
        hold = hold.reindex(D.index).fillna(0) * D.notna()
        w = hold.div(hold.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)   # equal weight across held coins
        gross = (w * D.fillna(0)).sum(axis=1)
        turnover = w.diff().abs().sum(axis=1); turnover.iloc[0] = w.iloc[0].abs().sum()
        net = gross - turnover * COST_SIDE
        return net, hold.sum(axis=1)
    r, h = run(held); res["rules"]["always (all coins)"] = stats(r, h)
    for x in FILTERS:
        r, h = run((trail > x).astype(float)); res["rules"][f"filter > {int(x*100)}%/yr"] = stats(r, h)
    # top 3, weekly
    mon = trail[trail.index.dayofweek == 0]
    rows = {}
    for t, row in mon.iterrows():
        top = row[row > 0].nlargest(3).index
        rows[t] = pd.Series(1.0, index=top).reindex(trail.columns).fillna(0.0)
    pick = pd.DataFrame(rows).T.reindex(trail.index).ffill().fillna(0)
    r, h = run(pick); res["rules"]["top 3 weekly"] = stats(r, h)
    # BTC alone, the simplest version
    r, h = run(pd.DataFrame({"BTC": 1.0}, index=D.index).reindex(columns=D.columns).fillna(0)); res["rules"]["BTC only, always"] = stats(r, h)
    for x in (0.05, 0.10):
        hb = (trail[["BTC"]] > x).astype(float).reindex(columns=D.columns).fillna(0)
        r, h = run(hb); res["rules"][f"BTC only, filter > {int(x*100)}%/yr"] = stats(r, h)
    json.dump(res, open(f"{OUT}/funding_carry.json", "w"), indent=1)
    for k, v in res["rules"].items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
