"""
ALT/BTC trend test: does holding altcoins that are beating Bitcoin (measured in BTC) beat simply holding BTC?
Same machinery as the USDT Trend Guard stress test, applied to every Binance spot pair quoted in BTC.

Universe (point in time, no survivorship bias): every Binance spot BTC-quoted pair in Binance's public archive,
including pairs that were later delisted. Each week the universe is the top N pairs by the previous 4 weeks'
BTC trading volume, among pairs with enough history. Stablecoins, fiat pairs, wrapped/staked tokens and
leveraged tokens (UP/DOWN/BULL/BEAR) are excluded.

Rules (all decided on the previous week's close, applied to the next week's return: no look-ahead):
  MOM L    hold a coin if its price is up over the last L weeks
  MA20     hold a coin if its price is above its 20-week average
  Prices and returns are ALT/BTC, so "0%" means "same as holding BTC".
  Each of the N slots gets 1/N of the money; a slot whose coin fails the filter sits in BTC ("guard").
  FULL     variant: the money is split only across coins that pass (fully invested whenever any pass).
Benchmarks: holding BTC (0% in BTC terms) and equal-weight buy & hold of the same top N (rebalanced weekly).
Results are given in BTC terms and, converted with BTCUSDT, in US dollar terms.
Costs: fee per side x weekly turnover (0.1%, 0.2%, 0.4% per side tested).
Delisting: a coin that stops trading is exited at its last weekly close (slightly generous; counted).

Usage: python trend_stress.py <out_dir>
"""
import io, json, os, re, sys, time, zipfile, urllib.request, urllib.error
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "research/results/altbtc"
CACHE = "data/weekly_btc"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DL = "https://data.binance.vision"
START = pd.Timestamp("2018-01-01")
SPLIT = pd.Timestamp("2024-01-01")
EXCL = set("USDC BUSD TUSD USDP DAI FDUSD PAX PAXG XAUT USDS USDSB SUSD UST USTC EUR GBP AUD TRY BRL RUB NGN UAH ZAR BIDR IDRT BVND "
           "AEUR EURI USDE USD1 RLUSD BFUSD USDF WBTC BTCB BTCST RENBTC TBTC WETH WBETH BETH STETH BNSOL CBETH RETH WEETH LDO_".split())
LEV = re.compile(r"(UP|DOWN|BULL|BEAR)$")


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "altbtc-trend"}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** i)
        except Exception:
            time.sleep(2 ** i)
    return None


def s3_list(prefix, delimiter=True):
    """List keys or sub-prefixes under a prefix in Binance's public bucket (handles paging)."""
    out, marker = [], ""
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        q = f"{S3}?prefix={prefix}" + ("&delimiter=/" if delimiter else "") + (f"&marker={marker}" if marker else "")
        b = get(q)
        if b is None:
            break
        root = ET.fromstring(b)
        items = [p.find(ns + "Prefix").text for p in root.findall(ns + "CommonPrefixes")] if delimiter \
            else [c.find(ns + "Key").text for c in root.findall(ns + "Contents")]
        out += items
        if root.find(ns + "IsTruncated").text != "true" or not items:
            break
        marker = items[-1]
    return out


def symbols():
    pref = s3_list("data/spot/monthly/klines/")
    syms = [p.rstrip("/").split("/")[-1] for p in pref]
    keep = []
    for s in syms:
        if not s.endswith("BTC") or s.endswith("USDTBTC"):
            continue
        base = s[:-3]
        if base in EXCL or LEV.search(base) or not base:
            continue
        keep.append(s)
    return sorted(keep)


def weekly(sym):
    path = f"{CACHE}/{sym}.csv"
    have = pd.read_csv(path, parse_dates=["t"]) if os.path.exists(path) else None
    keys = [k for k in s3_list(f"data/spot/monthly/klines/{sym}/1w/", delimiter=False) if k.endswith(".zip")]
    done = set(have["month"].unique()) if have is not None and "month" in have else set()
    frames = [have] if have is not None else []
    this_month = pd.Timestamp.utcnow().strftime("%Y-%m")
    for k in keys:
        m = re.search(r"-(\d{4}-\d{2})\.zip$", k)
        if not m or m.group(1) in done or m.group(1) >= this_month:
            continue
        b = get(f"{DL}/{k}")
        if not b:
            continue
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            raw = z.read(z.namelist()[0])
        first = raw.split(b",", 1)[0].strip()
        d = pd.read_csv(io.BytesIO(raw), header=None if first.isdigit() else 0).iloc[:, [0, 4, 7]]
        d.columns = ["t", "close", "qvol"]
        t = pd.to_numeric(d.t)
        d["t"] = pd.to_datetime(t, unit="us" if t.max() > 1e14 else "ms")
        d["month"] = m.group(1)
        frames.append(d)
    if not frames:
        return sym, None
    d = pd.concat(frames).drop_duplicates("t").sort_values("t")
    os.makedirs(CACHE, exist_ok=True)
    d.to_csv(path, index=False)
    return sym, d


def stats(r, w=None):
    r = pd.Series(r).dropna()
    if len(r) < 10:
        return None
    eq = (1 + r).cumprod()
    yrs = len(r) / 52.18
    out = {"per_year_pct": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1), "total_pct": round((eq.iloc[-1] - 1) * 100, 1),
           "max_drawdown_pct": round((eq / eq.cummax() - 1).min() * 100, 1),
           "sharpe": round(float(r.mean() / r.std() * np.sqrt(52.18)) if r.std() > 0 else 0.0, 2),
           "weeks": int(len(r))}
    if w is not None:
        out["avg_invested_pct"] = round(float(pd.Series(w).dropna().mean()) * 100, 0)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    syms = symbols()
    print(f"{len(syms)} BTC-quoted spot pairs in the archive (incl. delisted)", flush=True)
    _, btc = weekly("BTCUSDT")
    BTCUSD = btc.set_index("t").close.pct_change()
    with ThreadPoolExecutor(16) as ex:
        data = {s: d for s, d in ex.map(weekly, syms) if d is not None and len(d) >= 8}
    print(f"{len(data)} pairs with weekly data", flush=True)
    C = pd.DataFrame({s: d.set_index("t").close for s, d in data.items()}).sort_index()
    V = pd.DataFrame({s: d.set_index("t").qvol for s, d in data.items()}).sort_index()
    C = C[C.index >= START - pd.Timedelta(weeks=30)]
    V = V.reindex(C.index)
    R = C.pct_change(fill_method=None)                      # week t return (close t / close t-1)
    alive_next = C.notna()                                    # has a candle this week
    vol4 = V.rolling(4, min_periods=4).sum()
    hist = C.notna().cumsum()

    def run(N, rule, fee, mode="guard", btc_only=False):
        rets, inv, turn = [], [], []
        prev_w = pd.Series(dtype=float)
        delist = 0
        idx = C.index
        for i in range(27, len(idx)):
            t, tp = idx[i], idx[i - 1]
            if t < START:
                continue
            if btc_only:
                univ = ["BTCUSDT"]
            else:
                elig = vol4.loc[tp].dropna()
                elig = elig[(hist.loc[tp].reindex(elig.index) >= 27) & C.loc[tp].reindex(elig.index).notna()]
                univ = elig.sort_values(ascending=False).index[:N].tolist()
            c = C[univ]
            if rule == "hold":
                ok = pd.Series(True, index=univ)
            elif rule.startswith("MOM"):
                L = int(rule[3:])
                ok = (c.iloc[i - 1] / c.iloc[i - 1 - L] - 1) > 0
            elif rule == "MA20":
                ok = c.iloc[i - 1] > c.iloc[i - 20:i].mean()
            ok = ok.fillna(False)
            n = len(univ)
            if mode == "guard":
                w = ok.astype(float) / n
            else:
                w = ok.astype(float) / max(1, ok.sum())
            r_t = R.loc[t, univ].copy()
            gone = ~alive_next.loc[t, univ]
            delist += int((gone & (w > 0)).sum())
            r_t[gone] = 0.0                                  # exited at last close
            gross = float((w * r_t.fillna(0)).sum())
            tw = (w.reindex(w.index.union(prev_w.index), fill_value=0) - prev_w.reindex(w.index.union(prev_w.index), fill_value=0)).abs().sum()
            cost = tw * fee
            rets.append((t, gross - cost)); inv.append((t, float(w.sum()))); turn.append(tw)
            # weights drift with returns until next rebalance
            grown = w * (1 + r_t.fillna(0))
            tot = grown.sum() + (1 - w.sum())
            prev_w = grown / tot if tot > 0 else grown
        s = pd.Series(dict(rets)); iv = pd.Series(dict(inv))
        return s, iv, float(np.mean(turn)) if turn else 0.0, delist

    res = {"settings": {"start": str(START.date()), "split": str(SPLIT.date()), "pairs_in_archive": len(syms), "pairs_with_data": len(data),
                        "last_week": str(C.index[-1].date())}, "results": []}
    plans = []
    for N in (5, 10, 20):
        for rule in ("hold", "MOM4", "MOM8", "MOM12", "MOM26", "MA20"):
            for mode in (("guard", "full") if rule != "hold" else ("full",)):
                for fee in (0.001, 0.002, 0.004):
                    plans.append(dict(N=N, rule=rule, mode=mode, fee=fee))
    for p in plans:
        s, iv, to, dl = run(p["N"], p["rule"], p["fee"], p["mode"], p.get("btc_only", False))
        name = ("BTC only" if p.get("btc_only") else f"top {p['N']}") + f" · {p['rule']}" + ("" if p["rule"] == "hold" else f" · {p['mode']}")
        usd = ((1 + s) * (1 + BTCUSD.reindex(s.index).fillna(0)) - 1)
        row = {"name": name, **{k: v for k, v in p.items()}, "turnover_per_week": round(to, 3), "delisted_exits": dl,
               "all": stats(s, iv), "train": stats(s[s.index < SPLIT], iv[iv.index < SPLIT]), "test": stats(s[s.index >= SPLIT], iv[iv.index >= SPLIT]),
               "usd_all": stats(usd), "usd_test": stats(usd[usd.index >= SPLIT]),
               "by_year": {str(y): round(((1 + g).prod() - 1) * 100, 1) for y, g in s.groupby(s.index.year)}}
        # different start years: return per year from each start to the end
        row["from_start_year"] = {str(y): (stats(s[s.index >= pd.Timestamp(f"{y}-01-01")]) or {}).get("per_year_pct") for y in range(2018, 2025)}
        res["results"].append(row)
        a, te = row["all"] or {}, row["test"] or {}
        print(f"{name:32s} fee {p['fee']*100:.1f}% | all {a.get('per_year_pct')}%/y DD {a.get('max_drawdown_pct')} Sh {a.get('sharpe')} inv {a.get('avg_invested_pct')} "
              f"| test {te.get('per_year_pct')}%/y DD {te.get('max_drawdown_pct')} Sh {te.get('sharpe')} | delist {dl}", flush=True)
    res["btc_usd"] = {"all": stats(BTCUSD[(BTCUSD.index >= START)]), "test": stats(BTCUSD[BTCUSD.index >= SPLIT])}
    print("BTC itself in USD:", res["btc_usd"])
    json.dump(res, open(f"{OUT}/altbtc_trend.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
