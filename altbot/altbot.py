"""ALT/BTC paper bot - NO real money, no API keys, no orders.

Tests one idea, with rules fixed before it starts: a frozen gradient-tree model predicts the next 15 minutes of
10 Binance ALT/BTC pairs from 5-minute candles. When the prediction is bigger than the cost, it pretends to trade:
buy the alt with BTC (or sell it for BTC) at the REAL best ask (or bid) shown at that moment, and close 15 minutes
later at the real bid (or ask). Binance fee 0.1% each way. Results in BTC terms.

Market data: Binance public market-data endpoints (data-stream / data-api .binance.vision) - read only.
"""
import asyncio, csv, fcntl, json, logging, math, os, random, subprocess, time, warnings
from collections import deque
from datetime import datetime, timezone
import numpy as np, joblib, requests, websockets

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.environ.get("ALTBOT_DATA", os.path.join(HERE, "data_local"))
LOCK = os.environ.get("GIT_LOCK", "/opt/polybot/state/git.lock")
TEST_MINUTES = float(os.environ.get("ALTBOT_TEST_MINUTES", "0"))
WS = "wss://data-stream.binance.vision/stream?streams="
REST = "https://data-api.binance.vision/api/v3"

# ---- rules decided in advance (see README) ----
FEE = 0.001                 # per side
THR_MULT = 1.0              # trade when |predicted 15-min return| > 1.0 x (0.3% + one price step)
STAKE_BTC = 0.01            # paper size; we record whether the best quote covers it
MAX_BOOK_AGE = 60.0         # skip if no best bid/ask update for this pair in the last 60 s
MAX_CANDLE_DELAY = 10.0     # skip if the closed candle arrived more than 10 s after its close
MIN_TRADES, MAX_TRADES = 200, 600
ACTIVITY_CHECK_DAYS = 30    # after 30 days: stop if trades < one third of the expected count
HOLD = 15 * 60
PUBLISH_EVERY = 1800

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("altbot")
utc = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class St:
    def __init__(self):
        self.candles = {}          # sym -> deque of (open_ms, close, qvol, tbq)
        self.book = {}             # pair -> (bid, bidqty, ask, askqty, recv_time)
        self.open = {}             # pair -> trade dict
        self.errors = deque(maxlen=20)
        self.counts = dict(checks=0, skipped=0, signals=0)
        self.skip_reasons = {}
        self.started = utc()
        self.ws_msgs = 0; self.ws_reconnects = 0; self.last_msg = 0.0
        self.candle_delay = deque(maxlen=200)

    def err(self, m):
        log.warning(m); self.errors.append(f"{utc()} {m}")


S = St()
META = json.load(open(os.path.join(HERE, "model", "alt15_meta.json")))
FEATS = META["features"]
PAIRS = META["pairs"]
EXPECTED_PER_DAY = META["oos"]["rules"][f"{THR_MULT}x"]["per_day"]
MODEL = None
TICK = {}


def features(c, b, open_ms):
    """c: arrays close/qvol/tbq for the pair (last 61 candles, contiguous); b: BTCUSDT closes aligned. Same maths as training."""
    close, qvol, tbq = c
    lr = np.diff(np.log(close))
    f = {}
    for k in (1, 3, 6, 12, 48):
        f[f"r{k}"] = math.log(close[-1] / close[-1 - k])
    f["vol12"] = float(np.std(lr[-12:], ddof=1)); f["vol48"] = float(np.std(lr[-48:], ddof=1))
    m48 = float(np.mean(qvol[-48:]))
    f["vratio"] = qvol[-1] / m48 if m48 > 0 else float("nan")
    f["buyshare"] = min(1.0, max(0.0, tbq[-1] / qvol[-1])) if qvol[-1] > 0 else float("nan")
    for k in (1, 3, 12):
        f[f"btc{k}"] = math.log(b[-1] / b[-1 - k]) if (b[-1] == b[-1] and b[-1 - k] == b[-1 - k]) else float("nan")
    f["hour"] = datetime.fromtimestamp(open_ms / 1000, timezone.utc).hour
    return [f[k] for k in FEATS]


def predict(x):
    return float(MODEL.predict(np.array([x], dtype=float))[0])


def self_check():
    """Refuse to trade unless our feature code + model reproduce the training numbers exactly."""
    worst = 0.0
    for r in META["refs"]:
        t = datetime.strptime(r["t"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        x = features((np.array(r["close"]), np.array(r["qvol"]), np.array(r["tbq"])),
                     np.array([v if v is not None else float("nan") for v in r["btc"]]), t.timestamp() * 1000)
        for a, e in zip(x, r["features"]):
            if not (a != a and e != e):
                worst = max(worst, abs(a - e))
        worst = max(worst, abs(predict(x) - r["pred"]))
    if worst > 1e-6:
        raise RuntimeError(f"self-check failed: max difference {worst}")
    return worst


def rest(path, **params):
    r = requests.get(f"{REST}/{path}", params=params, timeout=20); r.raise_for_status(); return r.json()


def backfill(sym):
    k = rest("klines", symbol=sym, interval="5m", limit=200)
    now = time.time() * 1000
    d = deque(maxlen=300)
    for row in k:
        if row[6] < now:  # closed candles only
            d.append((int(row[0]), float(row[4]), float(row[7]), float(row[10])))
    S.candles[sym] = d


def window(sym, open_ms, n=61):
    d = S.candles.get(sym)
    if not d:
        return None
    rows = [r for r in d if r[0] <= open_ms][-n:]
    if len(rows) < n or rows[-1][0] != open_ms or rows[-1][0] - rows[0][0] != (n - 1) * 300000:
        return None
    return rows


def append_csv(name, row):
    p = os.path.join(DATA, name); new = not os.path.exists(p)
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)


def skip(pair, why):
    S.counts["skipped"] += 1; S.skip_reasons[why] = S.skip_reasons.get(why, 0) + 1


async def decide(open_ms, arrived):
    """Called once per 5-minute candle after all candles have (probably) arrived."""
    close_s = (open_ms + 300000) / 1000
    bw = window("BTCUSDT", open_ms)
    for p in PAIRS:
        S.counts["checks"] += 1
        if bw is None:
            skip(p, "missing BTC candles"); continue
        if MODEL is None:
            skip(p, "model not loaded"); continue
        if p in S.open:
            skip(p, "position already open"); continue
        w = window(p, open_ms)
        if w is None:
            skip(p, "missing candles"); continue
        if arrived.get(p, 1e18) - close_s > MAX_CANDLE_DELAY:
            skip(p, "candle arrived late"); continue
        bk = S.book.get(p)
        if not bk or time.time() - bk[4] > MAX_BOOK_AGE:
            skip(p, "bid/ask stale"); continue
        bmap = {r[0]: r[1] for r in (bw or [])}
        b = np.array([bmap.get(r[0], float("nan")) for r in w])
        arr = np.array(w)
        x = features((arr[:, 1], arr[:, 2], arr[:, 3]), b, open_ms)
        pred = predict(x)
        tick_pct = TICK.get(p, 0) / arr[-1, 1]
        thr = THR_MULT * (META["base_cost"] + tick_pct)
        if abs(pred) <= thr:
            continue
        S.counts["signals"] += 1
        side = 1 if pred > 0 else -1
        asyncio.create_task(trade(p, side, pred, thr, open_ms, bw is not None and len(bw) == 61))


def side_net(side, ent, ext):
    """ent/ext: two (bid, ask) snapshots 1 s apart. Buy at the worse ask, sell at the worse bid. Net after fees."""
    if side > 0:
        entry = max(q[1] for q in ent); exitp = min(q[0] for q in ext); gross = exitp / entry - 1
    else:
        entry = min(q[0] for q in ent); exitp = max(q[1] for q in ext); gross = entry / exitp - 1
    return entry, exitp, (1 + gross) * (1 - FEE) ** 2 - 1


async def trade(p, side, pred, thr, open_ms, btc_ok):
    bid, bq, ask, aq, _ = S.book[p]
    fits = (aq * ask if side > 0 else bq * bid) >= STAKE_BTC
    T = dict(opened_utc=utc(), pair=p, side="buy alt" if side > 0 else "sell alt", pred_pct=round(pred * 100, 4),
             threshold_pct=round(thr * 100, 4), candle_open_utc=datetime.fromtimestamp(open_ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M"),
             btc_feature_ok=btc_ok, spread_pct=round((ask - bid) / bid * 100, 4), stake_fits=fits)
    S.open[p] = T
    try:
        ent = [(bid, ask)]
        await asyncio.sleep(1); b2 = S.book[p]; ent.append((b2[0], b2[2]))
        await asyncio.sleep(HOLD - 1); b3 = S.book[p]; ext = [(b3[0], b3[2])]
        await asyncio.sleep(1); b4 = S.book[p]; ext.append((b4[0], b4[2]))
        entry, exitp, net = side_net(side, ent, ext)
        coin = random.Random(f"{p}{open_ms}").choice([1, -1])   # control: same pair, same moment, coin-flip side
        cnet = net if coin == side else side_net(coin, ent, ext)[2]
        T.update(closed_utc=utc(), entry=entry, exit=exitp, exit_book_age_s=round(time.time() - b4[4], 1),
                 net_pct=round(net * 100, 4), coin_side=coin, coin_net_pct=round(cnet * 100, 4))
        append_csv("trades.csv", T)
        log.info("trade %s %s net %.3f%%", p, T["side"], net * 100)
    except Exception as e:
        S.err(f"trade {p}: {e!r}")
    finally:
        S.open.pop(p, None)


async def ws_loop():
    syms = PAIRS + ["BTCUSDT"]
    streams = [f"{s.lower()}@kline_5m" for s in syms] + [f"{p.lower()}@bookTicker" for p in PAIRS]
    pending = {}     # open_ms -> {sym: arrival time}
    decided = deque(maxlen=50)
    while True:
        try:
            for s in syms:
                await asyncio.to_thread(backfill, s)
            async with websockets.connect(WS + "/".join(streams), ping_interval=20, max_size=2 ** 22) as ws:
                log.info("connected"); S.last_msg = time.time()
                while True:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                    S.ws_msgs += 1; S.last_msg = time.time()
                    d = m.get("data", {})
                    if d.get("e") == "kline":
                        k = d["k"]
                        if not k["x"]:
                            continue
                        sym = k["s"]; o = int(k["t"])
                        dq = S.candles.setdefault(sym, deque(maxlen=300))
                        if not dq or dq[-1][0] < o:
                            dq.append((o, float(k["c"]), float(k["q"]), float(k["Q"])))
                        now = time.time()
                        S.candle_delay.append(now - (o + 300000) / 1000)
                        if o in decided:
                            continue
                        pending.setdefault(o, {})[sym] = now
                        if len(pending[o]) == len(syms):
                            decided.append(o); await decide(o, pending.pop(o))
                    elif "b" in d and "a" in d and "s" in d:
                        S.book[d["s"]] = (float(d["b"]), float(d["B"]), float(d["a"]), float(d["A"]), time.time())
                    # candles whose stragglers never came: decide 8 s after the close anyway
                    for o in [o for o, v in pending.items() if time.time() - (o + 300000) / 1000 > 8]:
                        decided.append(o); await decide(o, pending.pop(o))
        except Exception as e:
            S.ws_reconnects += 1; S.err(f"ws: {e!r}"); await asyncio.sleep(5)


# ---------------- scoring ----------------
def read_trades():
    p = os.path.join(DATA, "trades.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def stats(xs, clusters):
    if not xs:
        return None
    by = {}
    for x, c in zip(xs, clusters):
        by.setdefault(c, []).append(x)
    cm = np.array([np.mean(v) for v in by.values()])
    mean = float(np.mean(xs))
    se = float(np.std(cm, ddof=1) / math.sqrt(len(cm))) if len(cm) > 1 else None
    return dict(trades=len(xs), mean_net_pct=round(mean, 4), win_rate=round(float(np.mean(np.array(xs) > 0)), 4),
                ci95_pct=[round(mean - 1.96 * se, 4), round(mean + 1.96 * se, 4)] if se else None,
                total_btc=round(sum(xs) / 100 * STAKE_BTC, 6))


def summarize():
    rows = read_trades()
    xs = [float(r["net_pct"]) for r in rows]
    cl = [r["candle_open_utc"] for r in rows]
    st = stats(xs, cl)
    ctl = stats([float(r["coin_net_pct"]) for r in rows], cl)
    days = (datetime.now(timezone.utc) - datetime.strptime(META_START, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)).total_seconds() / 86400
    expected = EXPECTED_PER_DAY * days
    n = len(xs)
    if days >= ACTIVITY_CHECK_DAYS and n < expected / 3 and n < MIN_TRADES:
        verdict = "stopped_low_activity"
    elif n < MIN_TRADES:
        verdict = "collecting"
    else:
        lo, hi = st["ci95_pct"] or [None, None]
        if lo is not None and lo > 0:
            verdict = "edge"
        elif hi is not None and hi < 0:
            verdict = "no_edge"
        else:
            verdict = "not_proven" if n >= MAX_TRADES else "undecided"
    return dict(
        updated_utc=utc(), started_utc=META_START, paper_only=True,
        rules=dict(pairs=PAIRS, horizon_min=15, threshold=f"|predicted 15-min move| > {THR_MULT} x (0.3% + one price step)",
                   prices="real best ask to buy / best bid to sell, worse of two quotes 1 s apart, both at entry and exit",
                   fee_per_side_pct=FEE * 100, stake_btc=STAKE_BTC, min_trades=MIN_TRADES, max_trades=MAX_TRADES,
                   activity_check=f"after {ACTIVITY_CHECK_DAYS} days stop if trades < 1/3 of expected ({EXPECTED_PER_DAY}/day)",
                   verdict_rule="edge if 95% CI of mean net per trade (clustered by candle) is above 0; no edge if below 0"),
        verdict=verdict, days_running=round(days, 2), expected_trades_so_far=round(expected, 1),
        result=st, coin_flip_control=ctl, open_positions=list(S.open), stake_fits_share=round(float(np.mean([r["stake_fits"] == "True" for r in rows])), 3) if rows else None,
        backtest_expectation=META["oos"]["rules"][f"{THR_MULT}x"],
        recent=rows[-50:],
        status=dict(checks=S.counts["checks"], skipped=S.counts["skipped"], signals=S.counts["signals"], skip_reasons=S.skip_reasons,
                    ws_msgs=S.ws_msgs, ws_reconnects=S.ws_reconnects, last_msg_age_s=round(time.time() - S.last_msg, 1),
                    median_candle_delay_s=round(float(np.median(S.candle_delay)), 2) if S.candle_delay else None,
                    books=len(S.book), process_started_utc=S.started, errors=list(S.errors)[-5:],
                    model_trained_to=META["trained_to"]))


def publish(push=True):
    out = summarize()
    dst = os.path.join(REPO, "data", "altbtc"); os.makedirs(dst, exist_ok=True)
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        json.dump(out, open(os.path.join(dst, "summary.json"), "w"), indent=1)
        src = os.path.join(DATA, "trades.csv")
        if os.path.exists(src):
            open(os.path.join(dst, "trades.csv"), "w").write(open(src).read())
        if not push:
            print(json.dumps(out, indent=1)[:3000]); return
        g = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, timeout=120)
        g("add", "data/altbtc")
        if g("diff", "--cached", "--quiet").returncode == 0:
            return
        g("commit", "-q", "-m", f"ALT/BTC paper log {utc()} UTC")
        g("pull", "-q", "--rebase", "--autostash")
        r = g("push", "-q")
        if r.returncode:
            S.err("push failed: " + r.stderr.strip()[:200])


async def publisher():
    await asyncio.sleep(120)
    while True:
        try:
            await asyncio.to_thread(publish, True)
        except Exception as e:
            S.err(f"publish: {e!r}")
        await asyncio.sleep(PUBLISH_EVERY)


def start_time():
    p = os.path.join(DATA, "started.txt")
    if not os.path.exists(p):
        open(p, "w").write(utc())
    return open(p).read().strip()


async def main():
    global MODEL, TICK, META_START
    os.makedirs(DATA, exist_ok=True)
    META_START = start_time()
    try:
        MODEL = joblib.load(os.path.join(HERE, "model", "alt15.joblib"))
        log.info("self-check ok, max diff %.2e", self_check())
    except Exception as e:
        MODEL = None; S.err(f"model: {e!r}")
    try:
        info = rest("exchangeInfo", symbols=json.dumps(PAIRS).replace(" ", ""))
        for s in info["symbols"]:
            for f in s["filters"]:
                if f["filterType"] == "PRICE_FILTER":
                    TICK[s["symbol"]] = float(f["tickSize"])
    except Exception as e:
        TICK.update(META.get("tick_sizes", {})); S.err(f"exchangeInfo: {e!r}")
    tasks = [asyncio.create_task(ws_loop())]
    if TEST_MINUTES > 0:
        await asyncio.sleep(TEST_MINUTES * 60); publish(push=False); return
    tasks.append(asyncio.create_task(publisher()))
    await asyncio.gather(*tasks)


META_START = None
if __name__ == "__main__":
    asyncio.run(main())
