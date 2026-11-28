"""Polybot: a PAPER-trading research bot for Polymarket's 5-minute Bitcoin Up/Down markets.

No wallet, no keys, no orders, no real money. It watches the public order book and the Chainlink
price feed, and 30 s and 10 s before each window ends it asks the gradient-tree model (trained on
73 days of recorded order books) whether either side is cheap enough after Polymarket's taker fee.
If so it logs a pretend $10 buy at the best ask it saw, then scores it against Polymarket's official
result. The point is to find out whether the backtest's "last 30 seconds" lead survives live.
"""
import asyncio, collections, csv, json, logging, math, os, subprocess, time
from datetime import datetime, timezone

import joblib, numpy as np, orjson, requests, websockets
import features as F

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("POLYBOT_DATA", "/opt/polybot/data")
TEST_MINUTES = float(os.environ.get("POLYBOT_TEST_MINUTES", "0"))   # >0: run briefly, don't push
GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RTDS_WS = "wss://ws-live-data.polymarket.com"
STAKE = 10.0              # pretend dollars per trade
PUBLISH_EVERY = 1800      # seconds
os.makedirs(DATA, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("polybot")
HTTP = requests.Session()
HTTP.headers["User-Agent"] = "polybot-paper-research/1.0"

TRADE_COLS = ["decided_utc", "window_start", "left", "side", "p_model", "p_formula", "edge", "price", "ask_size",
              "price_1s", "up_bid", "up_ask", "dn_bid", "dn_ask", "price_to_beat", "btc_price", "result", "win",
              "shares", "fee_usd", "pnl_usd", "pnl_1s_usd", "lag_s"]
DEC_COLS = ["decided_utc", "window_start", "left", "status", "p_model", "p_formula", "edge", "side",
            "up_bid", "up_ask", "up_ask_sz", "dn_bid", "dn_ask", "dn_ask_sz", "price_to_beat", "btc_price", "lag_s"]
MAX_LAG = 2.0             # refuse to trade if our view of the order book is more than 2 s behind Polymarket


def utc(ts=None):
    return datetime.fromtimestamp(ts or time.time(), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class State:
    def __init__(self):
        self.cl, self.tw = {}, {}            # observation second -> price (Chainlink spot, 60 s TWAP)
        self.cl_known, self.cl_recv = None, 0.0
        self.books, self.fresh = {}, set()
        self.book_recv = 0.0
        self.lag = None                      # seconds between Polymarket's timestamp and our processing, latest message
        self.lags = collections.deque(maxlen=3000)
        self.cl_gaps = collections.deque(maxlen=600)       # seconds between Chainlink updates arriving
        self.cl_obs_lag = collections.deque(maxlen=600)    # arrival time minus the price's own timestamp
        self.rtds_msgs = collections.Counter()
        self.cur = None                      # current market dict
        self.done = set()                    # (start, left) already decided
        self.pending = self._load("pending.json", [])
        self.results = self._load("results.json", {})    # window start (str) -> "Up"/"Down"
        self.need = set()                                 # decided windows still waiting for a result
        self.status = dict(started_utc=utc(), errors=[], decisions=0, trades=0, settled=0, skipped={})
        self.model_ok = False

    def _load(self, name, default):
        try:
            return json.load(open(os.path.join(DATA, name)))
        except Exception:
            return default

    def save_pending(self):
        json.dump(self.pending, open(os.path.join(DATA, "pending.json"), "w"))
        json.dump(self.results, open(os.path.join(DATA, "results.json"), "w"))

    def err(self, msg):
        log.warning(msg)
        self.status["errors"] = (self.status["errors"] + [utc() + " " + msg])[-20:]


S = State()


# ---------------------------------------------------------------- model
def load_model():
    m = joblib.load(os.path.join(REPO, "model", "gbm.joblib"))
    meta = json.load(open(os.path.join(REPO, "model", "gbm_meta.json")))
    assert meta["features"] == F.FEATS, "feature list mismatch"
    worst = max(abs(m.predict_proba(np.array([r["features"]]))[0, 1] - r["p_up"]) for r in meta["reference"])
    S.status["model"] = dict(trained_days=len(meta["days"]), first_day=meta["first_day"], last_day=meta["last_day"],
                             reference_max_diff=worst)
    if worst > 1e-6:
        raise RuntimeError(f"model does not reproduce its reference predictions (max diff {worst})")
    S.model_ok = True
    return m


MODEL = None


# ---------------------------------------------------------------- Polymarket market data
def gamma_market(start):
    js = []
    for extra in ({}, {"closed": "true"}):        # finished markets are only listed with closed=true
        r = HTTP.get(GAMMA, params={"slug": f"btc-updown-5m-{start}", **extra}, timeout=15)
        r.raise_for_status()
        js = r.json()
        if js:
            break
    if not js:
        return None
    m = js[0]
    outcomes, toks = json.loads(m["outcomes"]), json.loads(m["clobTokenIds"])
    if [o.lower() for o in outcomes] != ["up", "down"]:
        return None
    meta = ((m.get("events") or [{}])[0].get("eventMetadata") or {})
    prices = json.loads(m.get("outcomePrices") or "[]")
    res = None
    if m.get("closed") and len(prices) == 2:
        p0 = float(prices[0])
        res = "Up" if p0 > 0.99 else "Down" if p0 < 0.01 else None
    return dict(start=start, up=toks[0], dn=toks[1], ptb=meta.get("priceToBeat"), result=res)


def top(asset):
    b = S.books.get(asset)
    if not b or asset not in S.fresh or (not b["bids"] and not b["asks"]):
        return None
    bid = max(b["bids"]) if b["bids"] else 0.0
    ask = min(b["asks"]) if b["asks"] else 1.0
    return dict(bid=bid, ask=ask, ask_sz=b["asks"].get(ask, 0.0), age=time.time() - b["t"])


def on_book_msg(x):
    et, now = x.get("event_type"), time.time()
    ts = x.get("timestamp")
    if ts:
        try:
            S.lag = now - int(ts) / 1000
            S.lags.append(S.lag)
        except (TypeError, ValueError):
            pass
    if et == "book":
        a = x["asset_id"]
        S.books[a] = dict(bids={round(float(l["price"]), 4): float(l["size"]) for l in x.get("bids", []) if float(l["size"]) > 0},
                          asks={round(float(l["price"]), 4): float(l["size"]) for l in x.get("asks", []) if float(l["size"]) > 0},
                          t=now)
        S.fresh.add(a)
    elif et == "price_change":
        for c in x.get("price_changes", []):
            b = S.books.get(c["asset_id"])
            if b is None or c["asset_id"] not in S.fresh:
                continue
            side = b["bids"] if c["side"] == "BUY" else b["asks"]
            pr, sz = round(float(c["price"]), 4), float(c["size"])
            if sz > 0:
                side[pr] = sz
            else:
                side.pop(pr, None)
            b["t"] = now
    S.book_recv = now


async def text_ping(ws, every):
    while True:
        await asyncio.sleep(every)
        await ws.send("PING")


async def clob_session(m, until):
    """Follow one market's order book until `until` (unix seconds)."""
    while time.time() < until:
        try:
            async with websockets.connect(CLOB_WS, ping_interval=None, max_size=2 ** 24, open_timeout=10) as ws:
                for a in (m["up"], m["dn"]):
                    S.fresh.discard(a)
                await ws.send(json.dumps({"assets_ids": [m["up"], m["dn"]], "type": "market"}))
                pinger = asyncio.create_task(text_ping(ws, 10))
                try:
                    while time.time() < until:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=max(0.1, until - time.time()))
                        except asyncio.TimeoutError:
                            break
                        if not msg or msg[0] not in "[{":
                            continue
                        d = orjson.loads(msg)
                        for x in (d if isinstance(d, list) else [d]):
                            on_book_msg(x)
                finally:
                    pinger.cancel()
        except Exception as e:
            S.err(f"order book connection: {e!r}")
            await asyncio.sleep(2)


async def market_loop():
    while True:
        start = int(time.time() // 300) * 300
        try:
            m = await asyncio.to_thread(gamma_market, start)
        except Exception as e:
            S.err(f"market lookup: {e!r}"); m = None
        if not m:
            await asyncio.sleep(5)
            continue
        S.cur = m
        for a in list(S.books):                       # forget older markets' books
            if a not in (m["up"], m["dn"]):
                S.books.pop(a, None); S.fresh.discard(a)
        await clob_session(m, start + 300 + 2)


# ---------------------------------------------------------------- Chainlink price feed (RTDS)
async def rtds_loop():
    flt = json.dumps({"symbol": "btc/usd"}, separators=(",", ":"))
    sub = {"action": "subscribe", "subscriptions": [{"topic": t, "type": "*", "filters": flt}
                                                    for t in ("crypto_prices_chainlink", "crypto_prices_twap_sixty")]}
    backoff = 3
    while True:
        try:
            async with websockets.connect(RTDS_WS, ping_interval=None, max_size=2 ** 23, open_timeout=10) as ws:
                await ws.send(json.dumps(sub))
                conn_start = time.time()              # give each new connection its own 15 s to deliver a price
                got_price = False
                pinger = asyncio.create_task(text_ping(ws, 5))
                try:
                    while True:
                        try:                                  # a silent feed is reconnected after 15 s
                            msg = await asyncio.wait_for(ws.recv(), timeout=15)
                        except asyncio.TimeoutError:
                            S.err("price feed silent for 15 s; reconnecting")
                            break
                        if time.time() - max(S.cl_recv or 0, conn_start) > 15:
                            S.err("no Chainlink price for 15 s on this connection; reconnecting")
                            break
                        if not msg or msg[0] != "{":
                            continue
                        m = orjson.loads(msg)
                        topic, pl = m.get("topic"), m.get("payload") or {}
                        S.rtds_msgs[(topic or "none") + ("/list" if isinstance(pl.get("data"), list) else "")] += 1
                        tgt = S.cl if topic == "crypto_prices_chainlink" else S.tw if topic == "crypto_prices_twap_sixty" else None
                        if tgt is None:
                            continue
                        items = pl["data"] if isinstance(pl.get("data"), list) else [pl]
                        for it in items:
                            if "timestamp" not in it or "value" not in it:
                                continue
                            if it.get("symbol") not in (None, "btc/usd"):
                                continue
                            s = int(it["timestamp"]) // 1000
                            tgt[s] = float(it["value"])
                            if tgt is S.cl:
                                now_ = time.time()
                                if S.cl_recv:
                                    S.cl_gaps.append(now_ - S.cl_recv)
                                S.cl_obs_lag.append(now_ - int(it["timestamp"]) / 1000)
                                S.cl_known = s if S.cl_known is None else max(S.cl_known, s)
                                S.cl_recv = now_
                                if not got_price:
                                    got_price = True; backoff = 3
                        if len(S.cl) > 1200:
                            cut = max(S.cl) - 900
                            for d in (S.cl, S.tw):
                                for k in [k for k in d if k < cut]:
                                    del d[k]
                finally:
                    pinger.cancel()
            if not got_price:                         # connection gave no price at all: wait longer before trying again
                await asyncio.sleep(backoff); backoff = min(backoff * 2, 120)
        except Exception as e:
            S.err(f"price feed connection: {e!r}")
            await asyncio.sleep(backoff); backoff = min(backoff * 2, 120)


# ---------------------------------------------------------------- decisions
def append_csv(name, cols, row):
    path = os.path.join(DATA, name)
    if os.path.exists(path):
        with open(path) as fh:
            head = fh.readline().strip().split(",")
        if head != cols:                     # columns changed: keep the old file, start a new one
            os.rename(path, path.replace(".csv", "_v1.csv"))
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(cols)
        w.writerow([row.get(c) for c in cols])


def skip(start, left, why):
    S.status["skipped"][why] = S.status["skipped"].get(why, 0) + 1
    append_csv("decisions.csv", DEC_COLS, dict(decided_utc=utc(), window_start=start, left=left, status="skipped: " + why))


async def decide(m, left):
    start, now = m["start"], time.time()
    up, dn = top(m["up"]), top(m["dn"])
    if not S.model_ok:
        return skip(start, left, "model not loaded")
    if not up or not dn or up["age"] > 30 or dn["age"] > 30 or up["ask"] > 1 or dn["ask"] > 1:
        return skip(start, left, "no fresh order book")
    if now - S.cl_recv > 10 or S.cl_known is None:
        return skip(start, left, "price feed stale")
    if S.lag is None or S.lag > MAX_LAG or now - S.book_recv > 5:
        return skip(start, left, "order book lagging")
    ptb = m.get("ptb")
    if ptb is None:
        ptb = next((S.tw[s] for s in (start, start - 1, start + 1) if s in S.tw), None)
    if ptb is None:
        try:
            ptb = (await asyncio.to_thread(gamma_market, start) or {}).get("ptb")
        except Exception:
            ptb = None
    if ptb is None:
        return skip(start, left, "no price to beat")
    f, pf = F.build(start, S.cl_known, S.cl, float(ptb), up, dn)
    if f is None:
        return skip(start, left, pf)
    p = float(MODEL.predict_proba(np.array([[f[k] for k in F.FEATS]]))[0, 1])
    side, edge = F.decide(p, up["ask"], dn["ask"])
    S.status["decisions"] += 1
    row = dict(decided_utc=utc(now), window_start=start, left=left, status="trade" if side else "no trade",
               p_model=round(p, 4), p_formula=round(pf, 4), edge=round(edge, 4), side=side,
               up_bid=up["bid"], up_ask=up["ask"], up_ask_sz=up["ask_sz"], dn_bid=dn["bid"], dn_ask=dn["ask"],
               dn_ask_sz=dn["ask_sz"], price_to_beat=ptb, btc_price=S.cl[S.cl_known], lag_s=round(S.lag, 3))
    append_csv("decisions.csv", DEC_COLS, row)
    if not side:
        return
    q = up if side == "Up" else dn
    t = dict(row, side=side, price=q["ask"], ask_size=q["ask_sz"], price_1s=None)
    await asyncio.sleep(1.0)                      # what the price looked like one second later
    q1 = top(m["up"] if side == "Up" else m["dn"])
    t["price_1s"] = q1["ask"] if q1 else None
    S.pending.append(t)
    S.save_pending()
    S.status["trades"] += 1
    log.info("paper trade %s at %.3f, %ds left, model %.3f, edge %.3f", side, t["price"], left, p, edge)


async def scheduler():
    while True:
        m = S.cur
        if m:
            for left in F.LEFTS:
                key = (m["start"], left)
                if key not in S.done and time.time() >= m["start"] + 300 - left:
                    S.done.add(key)
                    asyncio.create_task(decide(m, left))
        await asyncio.sleep(0.1)


# ---------------------------------------------------------------- settlement and publishing
async def settler():
    while True:
        for t in S.pending:                               # results are only needed for windows we traded
            if str(t["window_start"]) not in S.results:
                S.need.add(int(t["window_start"]))
        await asyncio.sleep(30)
        now = time.time()
        for start in sorted(S.need)[:20]:                 # a few lookups per round, gently
            if now < start + 360:
                continue
            try:
                res = (await asyncio.to_thread(gamma_market, start) or {}).get("result")
            except Exception as e:
                S.err(f"result lookup: {e!r}"); res = None
            if res:
                S.results[str(start)] = res
                S.need.discard(start)
            elif now > start + 6 * 3600:
                S.need.discard(start)
                S.err(f"no result for window {start} after 6 h")
        still = []
        for t in S.pending:
            start = int(t["window_start"])
            res = S.results.get(str(start))
            if res is None:
                if now > start + 6 * 3600:
                    S.err(f"trade in window {start} dropped: no result after 6 h")
                else:
                    still.append(t)
                    S.need.add(start)
                continue
            win = 1 if res == t["side"] else 0
            price = float(t["price"]); shares = STAKE / price
            t.update(result=res, win=win, shares=round(shares, 4), fee_usd=round(shares * F.fee(price), 4),
                     pnl_usd=round(shares * (win - price - F.fee(price)), 4))
            if t.get("price_1s") is not None:
                p1 = float(t["price_1s"]); s1 = STAKE / p1
                t["pnl_1s_usd"] = round(s1 * (win - p1 - F.fee(p1)), 4)
            append_csv("trades.csv", TRADE_COLS, t)
            S.status["settled"] += 1
        S.pending = still
        S.save_pending()


# ---------------------------------------------------------------- the pre-registered test
MIN_TRADES = 300      # no verdict before this many settled trades
MAX_TRADES = 1000     # by this many, an unproven edge is called "not proven" and the test ends
# Early-stop rule, added 28 Nov 2026 (before results): the backtest (order-book delay <= 2 s, edge > 2c) traded in
# 4.9% of 5-minute windows (30 s and 10 s checks together). If two weeks after the freshness check began the bot has
# traded in fewer than a third of that, the backtest's opportunities are not showing up live and the test stops.
EXPECTED_RATE = 0.049
EARLY_STOP_AT = "2026-12-11 15:40:00"     # UTC, 14 days after the freshness check went live


def ret_per_dollar(price, win):
    return (win - price - F.fee(price)) / price


def conservative(r):
    """Score at the WORSE of the price at the signal and the price one second later."""
    p = float(r["price"])
    if r.get("price_1s") not in (None, ""):
        p = max(p, float(r["price_1s"]))
    return ret_per_dollar(min(p, 0.999), int(r["win"]))


def cluster_ci(rows, f):
    """Mean return per $1 and a 95% interval that treats the two checks in one window as related."""
    by = {}
    for r in rows:
        by.setdefault(r["window_start"], []).append(f(r))
    n = sum(len(v) for v in by.values()); g = len(by)
    if n == 0:
        return None
    m = sum(sum(v) for v in by.values()) / n
    if g < 2:
        return dict(mean=round(m, 4), lo=None, hi=None, n=n, windows=g)
    var = sum((sum(v) - m * len(v)) ** 2 for v in by.values()) / n ** 2 * g / (g - 1)
    se = math.sqrt(var)
    return dict(mean=round(m, 4), lo=round(m - 1.96 * se, 4), hi=round(m + 1.96 * se, 4), n=n, windows=g)


def controls(rows):
    """Same windows, same moments: what a coin flip and 'always buy the favourite' would have made."""
    def side_ret(r, side):
        a = float(r["up_ask"] if side == "Up" else r["dn_ask"])
        return ret_per_dollar(min(max(a, 0.001), 0.999), 1 if r["result"] == side else 0)
    def fav(r):
        mid_up = (float(r["up_bid"]) + float(r["up_ask"])) / 2
        return side_ret(r, "Up" if mid_up >= 0.5 else "Down")
    def coin(r):
        return (side_ret(r, "Up") + side_ret(r, "Down")) / 2
    return dict(favourite=cluster_ci(rows, fav), coin_flip=cluster_ci(rows, coin))


def windows_checked():
    """Five-minute windows where the bot made a real decision (not skipped) since the freshness check."""
    w = set()
    try:
        for r in csv.DictReader(open(os.path.join(DATA, "decisions.csv"))):
            if r["status"] in ("trade", "no trade"):
                w.add(r["window_start"])
    except FileNotFoundError:
        pass
    return len(w)


def activity(n):
    w = windows_checked()
    exp = EXPECTED_RATE * w
    return dict(windows=w, trades=n, expected=round(exp, 1), ratio=round(n / exp, 2) if exp else None)


def verdict(rows):
    c = cluster_ci(rows, conservative)
    n = len(rows)
    act = activity(n)
    if utc() >= EARLY_STOP_AT and n < MIN_TRADES and act["expected"] and n < act["expected"] / 3:
        return dict(state="stopped_low_activity", stats=c, activity=act,
                    text=f"Stopped early (rule set 28 Nov): after two weeks the bot made {n} trades where the backtest "
                         f"predicted about {act['expected']:.0f}. The backtest's opportunities are not showing up live.")
    if not c or n < MIN_TRADES:
        return dict(state="collecting", stats=c, activity=act,
                    text=f"Collecting: {n} of {MIN_TRADES} trades needed before any verdict. Activity check on {EARLY_STOP_AT[:10]}: "
                         f"{n} trades so far vs about {act['expected']:.0f} expected from the backtest.")
    ctl = controls(rows)["favourite"]
    if c["lo"] is not None and c["lo"] > 0 and (ctl is None or c["mean"] > ctl["mean"]):
        return dict(state="edge", text="Edge confirmed: profitable after fees at the worse fill, 95% interval above zero, and better than always buying the favourite.", stats=c)
    if c["hi"] is not None and c["hi"] < 0:
        return dict(state="no_edge", text="No edge: the bot loses money after fees (95% interval below zero).", stats=c)
    if n >= MAX_TRADES:
        return dict(state="not_proven", text=f"Not proven after {n} trades: results are within noise. The test ends here.", stats=c)
    return dict(state="undecided", text=f"Undecided after {n} trades: still within noise, keep collecting (up to {MAX_TRADES}).", stats=c)



def lag_stats(src=None):
    xs = sorted(S.lags if src is None else src)
    if not xs:
        return None
    q = lambda f: round(xs[min(len(xs) - 1, int(f * len(xs)))], 3)
    return dict(p50=q(0.5), p95=q(0.95), max=round(xs[-1], 3), n=len(xs))

def summarize():
    rows = []
    for name in ("trades_v1.csv", "trades.csv"):           # older-format log first, then the current one
        path = os.path.join(DATA, name)
        if os.path.exists(path):
            rows += list(csv.DictReader(open(path)))
    def block(rs):
        if not rs:
            return dict(trades=0)
        stake = STAKE * len(rs); pnl = sum(float(r["pnl_usd"]) for r in rs)
        r1 = [r for r in rs if r.get("pnl_1s_usd")]
        return dict(trades=len(rs), wins=sum(int(r["win"]) for r in rs), win_rate=round(sum(int(r["win"]) for r in rs) / len(rs), 4),
                    avg_price=round(sum(float(r["price"]) for r in rs) / len(rs), 4), staked_usd=stake,
                    pnl_usd=round(pnl, 2), return_pct=round(pnl / stake * 100, 2),
                    return_1s_pct=round(sum(float(r["pnl_1s_usd"]) for r in r1) / (STAKE * len(r1)) * 100, 2) if r1 else None)
    all_rows = rows
    rows = [r for r in rows if r.get("lag_s") not in (None, "") and float(r["lag_s"]) <= MAX_LAG]
    fits = [r for r in rows if float(r["ask_size"] or 0) * float(r["price"]) >= STAKE]
    test = dict(rule=(f"Pre-registered (27 Nov 2026): no verdict before {MIN_TRADES} settled trades. Trades are scored at the worse of "
                      "the price at the signal and one second later, after Polymarket's taker fee. Edge confirmed only if the 95% interval "
                      "of return per $1 (windows treated as clusters) is above zero AND beats always buying the favourite on the same "
                      f"windows. Interval below zero = no edge. Still unclear at {MAX_TRADES} trades = not proven, test ends. "
                      f"Added 28 Nov 2026 before further results: if on {EARLY_STOP_AT[:10]} the bot has made fewer than a third of "
                      f"the trades the backtest predicts ({EXPECTED_RATE:.1%} of checked windows), the test stops early."),
                verdict=verdict(rows), conservative=cluster_ci(rows, conservative), controls=controls(rows),
                stake_fits_at_best_ask_pct=round(100 * len(fits) / len(rows), 1) if rows else None,
                excluded_trades=len(all_rows) - len(rows),
                excluded_reason="trades without a measured order-book delay (before 27 Nov 2026 ~16:00 UTC) or with delay over 2 s")
    checks = collections.Counter()
    try:
        for r in csv.DictReader(open(os.path.join(DATA, "decisions.csv"))):     # since the freshness check began
            checks["skipped" if r["status"].startswith("skipped") else r["status"]] += 1
            if r["status"].startswith("skipped"):
                checks[r["status"]] += 1
    except FileNotFoundError:
        pass
    test["checks"] = dict(total=checks["trade"] + checks["no trade"] + checks["skipped"], trade=checks["trade"],
                          no_trade=checks["no trade"], skipped=checks["skipped"],
                          skipped_reasons={k[9:]: v for k, v in checks.items() if k.startswith("skipped: ")})
    out = dict(updated_utc=utc(), stake_per_trade_usd=STAKE, fee_coef=F.FEE_COEF, edge=F.EDGE, test=test,
               overall=block(rows), overall_including_excluded=block(all_rows), by_left={str(l): block([r for r in rows if int(r["left"]) == l]) for l in F.LEFTS},
               first_trade_utc=all_rows[0]["decided_utc"] if all_rows else None, recent=all_rows[-50:][::-1],
               status=dict(S.status, pending=len(S.pending), awaiting_results=len(S.need),
                           book_delay_s=lag_stats(), price_feed_gap_s=lag_stats(S.cl_gaps),
                           price_feed_obs_lag_s=lag_stats(S.cl_obs_lag), price_feed_msgs=dict(S.rtds_msgs),
                           price_feed_last_obs_age_s=round(time.time() - S.cl_known, 1) if S.cl_known else None, load_avg=[round(v, 2) for v in os.getloadavg()],
                           price_feed_age_s=round(time.time() - S.cl_recv, 1) if S.cl_recv else None,
                           order_book_age_s=round(time.time() - S.book_recv, 1) if S.book_recv else None))
    return out


def publish(push=True):
    out = summarize()
    os.makedirs(os.path.join(REPO, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(REPO, "data", "summary.json"), "w"), indent=1)
    for name in sorted(f for f in os.listdir(DATA) if f.endswith(".csv")):
        src = os.path.join(DATA, name)
        if os.path.exists(src):
            with open(src) as a, open(os.path.join(REPO, "data", name), "w") as b:
                b.write(a.read())
    if not push:
        print(json.dumps(out["status"], indent=1))
        return
    g = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, timeout=120)
    import fcntl
    os.makedirs("/opt/polybot/state", exist_ok=True)
    with open("/opt/polybot/state/git.lock", "w") as lk:      # shared with the ALT/BTC bot so pushes never collide
        fcntl.flock(lk, fcntl.LOCK_EX)
        g("add", "data")
        if g("diff", "--cached", "--quiet").returncode == 0:
            return
        g("commit", "-q", "-m", f"Paper log {utc()} UTC")
        r = g("pull", "-q", "--rebase", "--autostash")
        r = g("push", "-q")
        if r.returncode:
            S.err("push failed: " + r.stderr.strip()[:200])


async def publisher():
    await asyncio.sleep(150)          # first report soon after start, so a (re)start is visible quickly
    while True:
        try:
            await asyncio.to_thread(publish, True)
        except Exception as e:
            S.err(f"publish: {e!r}")
        await asyncio.sleep(PUBLISH_EVERY)


async def main():
    global MODEL
    try:
        MODEL = load_model()
        log.info("model loaded: %s", S.status["model"])
    except Exception as e:
        S.err(f"model: {e!r}")
    tasks = [asyncio.create_task(c()) for c in (rtds_loop, market_loop, scheduler, settler)]
    if TEST_MINUTES > 0:
        await asyncio.sleep(TEST_MINUTES * 60)
        publish(push=False)
        print(open(os.path.join(DATA, "decisions.csv")).read() if os.path.exists(os.path.join(DATA, "decisions.csv")) else "no decisions")
        for t in tasks:
            t.cancel()
        return
    tasks.append(asyncio.create_task(publisher()))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
