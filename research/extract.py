"""Replay one coin-day of Polymarket 5-minute Up/Down order books and take snapshots.

For every 5-minute market we rebuild both order books (Up token and Down token) from the
recorded WebSocket frames and record the state at fixed 'seconds left' checkpoints, plus the
same checkpoint 1s and 2s later (to measure the cost of being slower). Alongside we record the
Chainlink price the markets settle on, as it was available at that moment.
Outcomes come from Polymarket's Gamma API (official resolution); a Chainlink-derived outcome
is stored too as a cross-check.
"""
import os, sys, json, gzip, time, math, bisect, subprocess, shutil, csv, collections, calendar
import orjson, requests
from huggingface_hub import hf_hub_download

R = "polydata5/polymarket-updown-5m-ws"
GAMMA = "https://gamma-api.polymarket.com/markets"
LEFTS = [240, 180, 120, 60, 30, 10]      # seconds left in the 5-minute window
LAGS = [0, 1, 2]                         # snapshot again 1s and 2s later
TOK = os.environ["HF_TOKEN"]


def gamma_markets(coin, day_start):
    slugs = [f"{coin.lower()}-updown-5m-{s}" for s in range(day_start, day_start + 86400, 300)]
    got = {}
    for closed in ("true", None):
        todo = [s for s in slugs if s not in got]
        for i in range(0, len(todo), 40):
            params = [("slug", s) for s in todo[i:i + 40]] + [("limit", "100")]
            if closed:
                params.append(("closed", closed))
            for attempt in range(4):
                try:
                    r = requests.get(GAMMA, params=params, timeout=30)
                    r.raise_for_status()
                    for m in r.json():
                        got[m["slug"]] = m
                    break
                except Exception as e:
                    print("gamma retry", e, file=sys.stderr); time.sleep(3 * (attempt + 1))
    out = {}
    for slug, m in got.items():
        try:
            outcomes = json.loads(m["outcomes"]); toks = json.loads(m["clobTokenIds"])
            prices = json.loads(m.get("outcomePrices") or "[]")
        except Exception:
            continue
        if [o.lower() for o in outcomes] != ["up", "down"]:
            continue
        start = int(slug.rsplit("-", 1)[1])
        res = None
        if m.get("closed") and len(prices) == 2:
            p0 = float(prices[0])
            res = 1 if p0 > 0.99 else 0 if p0 < 0.01 else None
        meta = ((m.get("events") or [{}])[0].get("eventMetadata") or {})
        out[start] = dict(up=toks[0], dn=toks[1], res=res,
                          ptb=meta.get("priceToBeat"), final=meta.get("finalPrice"))
    return out


def load_rtds(path):
    cl, tw = {}, {}                     # obs second -> value
    avail = []                          # (recv_ts, obs_sec) for chainlink, in receive order
    with gzip.open(path, "rb") as fh:
        for line in fh:
            try:
                d = orjson.loads(line)
                raw = d.get("raw")
                if not raw:
                    continue
                m = orjson.loads(raw)
            except Exception:
                continue
            topic, pl = m.get("topic"), m.get("payload") or {}
            tgt = cl if topic == "crypto_prices_chainlink" else tw if topic == "crypto_prices_twap_sixty" else None
            if tgt is None:
                continue
            items = pl.get("data") if isinstance(pl.get("data"), list) else [pl]
            for it in items:
                if "timestamp" in it and "value" in it:
                    s = int(it["timestamp"]) // 1000
                    tgt[s] = float(it["value"])
                    if tgt is cl:
                        avail.append((d["recv_ts"], s))
    avail.sort()
    # running max of obs second known by each receive time
    rts, best, mx = [], [], -1
    for r, s in avail:
        mx = max(mx, s); rts.append(r); best.append(mx)
    return cl, tw, rts, best


def cl_known_at(T, rts, best):
    i = bisect.bisect_right(rts, T) - 1
    return best[i] if i >= 0 else None


def vol300(cl, end_sec):
    rets, prev = [], None
    for s in range(end_sec - 300, end_sec + 1):
        v = cl.get(s)
        if v is not None and prev is not None and v > 0 and prev > 0:
            rets.append(math.log(v / prev))
        if v is not None:
            prev = v
    if len(rets) < 100:
        return None
    mu = sum(rets) / len(rets)
    return math.sqrt(sum((x - mu) ** 2 for x in rets) / (len(rets) - 1))   # per-second stdev


class Book:
    __slots__ = ("bids", "asks", "t", "fresh")
    def __init__(self):
        self.bids, self.asks, self.t, self.fresh = {}, {}, 0.0, False


def top(book):
    if not book.bids and not book.asks:
        return None
    bb = max(book.bids) if book.bids else 0.0
    ba = min(book.asks) if book.asks else 1.0
    asz = book.asks.get(ba, 0.0)
    depth2 = sum(sz for p, sz in book.asks.items() if p <= ba + 0.02 + 1e-9)
    return bb, ba, asz, depth2


def process_day(coin, day, outdir):
    day_start = calendar.timegm(time.strptime(day, "%Y-%m-%d"))
    t0 = time.time()
    mk = gamma_markets(coin, day_start)
    print(day, "gamma markets", len(mk), "resolved", sum(v["res"] is not None for v in mk.values()), file=sys.stderr)
    tmp = f"/tmp/hf_{coin}_{day}"
    rt = hf_hub_download(R, f"data/{day}/{coin}_rtds.jsonl.gz", repo_type="dataset", token=TOK, local_dir=tmp)
    cl, tw, rts, best = load_rtds(rt)
    cb = hf_hub_download(R, f"data/{day}/{coin}_clob.jsonl.gz", repo_type="dataset", token=TOK, local_dir=tmp)
    print(day, "downloaded", round(os.path.getsize(cb) / 1e9, 2), "GB in", round(time.time() - t0), "s", file=sys.stderr)

    asset_side = {}
    for st, v in mk.items():
        asset_side[v["up"]] = (st, "up"); asset_side[v["dn"]] = (st, "dn")
    sched = sorted((st + 300 - left + lag, st, left, lag) for st in mk for left in LEFTS for lag in LAGS)
    si = 0
    books = collections.defaultdict(Book)
    stats = collections.Counter()
    fee_bps = collections.Counter()
    rows = []
    recent_lag = collections.deque()      # (recv_ts, recv_ts - Polymarket timestamp) for the last 5 s
    last_lag = [None]

    def emit(T, st, left, lag):
        v = mk[st]
        bu, bd = books.get(v["up"]), books.get(v["dn"])
        if not bu or not bd or not bu.fresh or not bd.fresh:
            stats["snap_missing"] += 1; return
        tu, td = top(bu), top(bd)
        if not tu or not td:
            stats["snap_empty"] += 1; return
        known = cl_known_at(T, rts, best)
        ref = v.get("ptb")
        if ref is None:
            ref = tw.get(st)
        if known is None or ref is None or known not in cl:
            stats["snap_no_cl"] += 1; return
        end = st + 300
        win = [cl[s2] for s2 in range(max(end - 60, st), known + 1) if s2 in cl] if known >= end - 60 else []
        twk = next((tw[s2] for s2 in range(known, known - 5, -1) if s2 in tw), None)
        rows.append([day, st, left, lag, *[round(x, 4) for x in tu], *[round(x, 4) for x in td],
                     round(T - bu.t, 2), round(T - bd.t, 2), cl[known], known - st, ref, twk,
                     (sum(win) / len(win)) if win else None, len(win),
                     vol300(cl, known), v["res"],
                     None if v.get("final") is None or v.get("ptb") is None else int(v["final"] >= v["ptb"]),
                     None if last_lag[0] is None else round(last_lag[0], 3),
                     round(max((l for r, l in recent_lag if r >= T - 5), default=float("nan")), 3) if recent_lag else None])
        stats["snap_ok"] += 1

    p = subprocess.Popen(["pigz", "-dc", cb], stdout=subprocess.PIPE, bufsize=1 << 20)
    n = 0
    for line in p.stdout:
        n += 1
        try:
            d = orjson.loads(line)
            ts = d["recv_ts"]
        except Exception:
            stats["bad_line"] += 1
            continue
        while si < len(sched) and sched[si][0] <= ts:
            emit(*sched[si]); si += 1
        raw = d.get("raw")
        if raw is None:
            if d.get("event") in ("connect", "reconnect", "disconnect"):
                for b in books.values():
                    b.fresh = False
            continue
        try:
            m = orjson.loads(raw)
        except Exception:
            stats["bad_raw"] += 1
            continue
        msgs = m if isinstance(m, list) else [m]
        for x in msgs:
            et = x.get("event_type")
            pts = x.get("timestamp")
            if pts:
                try:
                    lg = ts - int(pts) / 1000
                    last_lag[0] = lg
                    recent_lag.append((ts, lg))
                    while recent_lag and recent_lag[0][0] < ts - 6:
                        recent_lag.popleft()
                    stats["lag_n"] += 1
                    if lg > 2:
                        stats["lag_over_2s"] += 1
                except (TypeError, ValueError):
                    pass
            if et == "book":
                a = x["asset_id"]
                b = books[a]
                b.bids = {round(float(l["price"]), 4): float(l["size"]) for l in x.get("bids", []) if float(l["size"]) > 0}
                b.asks = {round(float(l["price"]), 4): float(l["size"]) for l in x.get("asks", []) if float(l["size"]) > 0}
                b.t = ts; b.fresh = True
                stats["book"] += 1
            elif et == "price_change":
                for c in x.get("price_changes", []):
                    a = c["asset_id"]
                    b = books.get(a)
                    if b is None or not b.fresh:
                        continue
                    pr, sz = round(float(c["price"]), 4), float(c["size"])
                    side = b.bids if c["side"] == "BUY" else b.asks
                    if sz > 0:
                        side[pr] = sz
                    else:
                        side.pop(pr, None)
                    b.t = ts
                    # integrity check against the exchange's own best bid/ask
                    if "best_ask" in c and n % 50 == 0:
                        tb = top(b)
                        stats["chk"] += 1
                        if tb and (abs(tb[0] - float(c["best_bid"])) > 1e-6 or abs(tb[1] - float(c["best_ask"])) > 1e-6):
                            stats["chk_mismatch"] += 1
            elif et == "last_trade_price":
                fee_bps[x.get("fee_rate_bps")] += 1
                if x.get("asset_id") in asset_side:
                    stats["trades"] += 1
                    stats["trade_usd_x100"] += int(float(x["price"]) * float(x["size"]) * 100)
        # forget books of markets that ended more than 60s ago
        if n % 200000 == 0:
            for a in list(books):
                st = asset_side.get(a, (0,))[0]
                if st and st + 360 < ts:
                    del books[a]
    p.wait()
    while si < len(sched):
        si += 1; stats["snap_after_end"] += 1
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    with gzip.open(f"{outdir}/snap_{coin}_{day}.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["day", "start", "left", "lag", "up_bid", "up_ask", "up_ask_sz", "up_depth2",
                    "dn_bid", "dn_ask", "dn_ask_sz", "dn_depth2", "up_age", "dn_age",
                    "cl_now", "cl_now_sec", "ptb", "tw_now", "win_avg", "win_n", "vol_s", "res", "res_meta", "book_lag", "book_lag5"])
        w.writerows(rows)
    info = dict(day=day, lines=n, markets=len(mk), secs=round(time.time() - t0),
                stats=dict(stats), fee_rate_bps=dict((str(k), v) for k, v in fee_bps.items()))
    json.dump(info, open(f"{outdir}/info_{coin}_{day}.json", "w"))
    print(json.dumps(info), file=sys.stderr)


if __name__ == "__main__":
    coin = os.environ.get("COIN", "BTC")
    for day in sys.argv[1].split(","):
        try:
            process_day(coin, day, "pm_out")
        except Exception as e:
            import traceback; traceback.print_exc()
            os.makedirs("pm_out", exist_ok=True)
            json.dump(dict(day=day, error=repr(e)), open(f"pm_out/failed_{coin}_{day}.json", "w"))
