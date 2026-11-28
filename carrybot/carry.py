"""BTC funding-rate carry - paper tracker. NO real money, no keys, no orders.

Rules fixed at start (28 November 2026):
- Once, at start: pretend-buy $10,000 of BTC on Binance spot at the real best ask (fee 0.1%) and pretend-short the
  same amount of BTC on the BTCUSDT perpetual at the real best bid (taker fee 0.05%). Never rebalanced, never closed
  by the bot - this is a tracker, the test is "what does holding it earn".
- Every funding payment (every 8 h) is credited (or charged, if negative) on the short: rate x quantity x mark price
  at that funding time, exactly as Binance does.
- Hourly mark-to-market: spot leg at the best bid, short leg at the best ask (the prices you would really close at),
  minus the fees to close. Capital counted = $20,000 (the BTC bought + 100% margin for the short: no realistic
  liquidation risk unless BTC doubles).
Public data only: data-api.binance.vision (spot) and fapi.binance.com (futures).
"""
import fcntl, json, os, subprocess, time
from datetime import datetime, timezone
import requests

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
DATA = os.environ.get("CARRY_DATA", os.path.join(HERE, "data_local"))
LOCK = os.environ.get("GIT_LOCK", "/opt/polybot/state/git.lock")
ONCE = os.environ.get("CARRY_ONCE") == "1"
SPOT = "https://data-api.binance.vision/api/v3"
FUT = "https://fapi.binance.com/fapi/v1"
NOTIONAL, CAPITAL = 10000.0, 20000.0
SPOT_FEE, FUT_FEE = 0.001, 0.0005
STATE = os.path.join(DATA, "state.json")
utc = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get(url, **p):
    r = requests.get(url, params=p, timeout=20); r.raise_for_status(); return r.json()


def quotes():
    s = get(f"{SPOT}/ticker/bookTicker", symbol="BTCUSDT")
    f = get(f"{FUT}/ticker/bookTicker", symbol="BTCUSDT")
    return float(s["bidPrice"]), float(s["askPrice"]), float(f["bidPrice"]), float(f["askPrice"])


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else None


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1); os.replace(STATE + ".tmp", STATE)


def open_trade():
    sb, sa, fb, fa = quotes()
    qty = NOTIONAL / sa
    st = dict(opened_utc=utc(), opened_ms=int(time.time() * 1000), qty_btc=qty, spot_entry=sa, perp_entry=fb,
              entry_fees=qty * sa * SPOT_FEE + qty * fb * FUT_FEE, funding_usd=0.0, payments=[], last_funding_ms=int(time.time() * 1000),
              errors=[])
    save(st); return st


def collect_funding(st):
    rows = get(f"{FUT}/fundingRate", symbol="BTCUSDT", startTime=st["last_funding_ms"] + 1, limit=1000)
    for r in rows:
        t = int(r["fundingTime"])
        if t <= st["last_funding_ms"]:
            continue
        rate = float(r["fundingRate"])
        mark = float(r.get("markPrice") or 0) or quotes()[2]
        usd = rate * st["qty_btc"] * mark          # positive rate: longs pay shorts, we are short -> we receive
        st["funding_usd"] += usd
        st["payments"].append(dict(time_utc=datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M"),
                                   rate_pct=round(rate * 100, 5), mark=round(mark, 2), usd=round(usd, 4)))
        st["last_funding_ms"] = t


def summarize(st):
    sb, sa, fb, fa = quotes()
    q = st["qty_btc"]
    spot_pnl = q * (sb - st["spot_entry"])
    perp_pnl = q * (st["perp_entry"] - fa)
    exit_fees = q * sb * SPOT_FEE + q * fa * FUT_FEE
    net = spot_pnl + perp_pnl + st["funding_usd"] - st["entry_fees"] - exit_fees
    days = (time.time() * 1000 - st["opened_ms"]) / 86400000
    pays = st["payments"]
    fund_ann = (st["funding_usd"] / NOTIONAL) / days * 365 * 100 if days > 0.5 else None
    return dict(
        updated_utc=utc(), paper_only=True, opened_utc=st["opened_utc"], days=round(days, 2),
        rules=dict(notional_usd=NOTIONAL, capital_usd=CAPITAL, spot_fee_pct=SPOT_FEE * 100, futures_fee_pct=FUT_FEE * 100,
                   note="long BTC spot + short the same BTC on the perpetual; never rebalanced; marked at the prices you could close at"),
        btc_price=round(sb, 2), qty_btc=round(q, 6), spot_entry=st["spot_entry"], perp_entry=st["perp_entry"],
        funding_received_usd=round(st["funding_usd"], 2), funding_payments=len(pays),
        negative_payments=sum(p["usd"] < 0 for p in pays),
        hedge_pnl_usd=round(spot_pnl + perp_pnl, 2), fees_usd=round(st["entry_fees"] + exit_fees, 2),
        net_if_closed_now_usd=round(net, 2), net_on_capital_pct=round(net / CAPITAL * 100, 3),
        funding_annualised_on_notional_pct=round(fund_ann, 2) if fund_ann is not None else None,
        funding_annualised_on_capital_pct=round(fund_ann / 2, 2) if fund_ann is not None else None,
        short_margin_used_pct=round(max(0.0, -perp_pnl) / NOTIONAL * 100, 1),
        backtest=dict(btc_always_on_net_on_capital_pct_by_year={"2021": 18, "2024": 6.3, "2025": 2.6, "2026 to Oct": 1.3},
                      source="research/results/funding/funding_carry.json"),
        recent_payments=pays[-30:], errors=st.get("errors", [])[-5:])


def publish(out):
    dst = os.path.join(REPO, "data", "carry"); os.makedirs(dst, exist_ok=True)
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        json.dump(out, open(os.path.join(dst, "summary.json"), "w"), indent=1)
        if ONCE:
            print(json.dumps(out, indent=1)); return
        g = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, timeout=120)
        g("add", "data/carry")
        if g("diff", "--cached", "--quiet").returncode == 0:
            return
        g("commit", "-q", "-m", f"Carry paper log {utc()} UTC")
        g("pull", "-q", "--rebase", "--autostash")
        g("push", "-q")


def main():
    os.makedirs(DATA, exist_ok=True)
    while True:
        try:
            st = load() or open_trade()
            try:
                collect_funding(st)
            except Exception as e:
                st.setdefault("errors", []).append(f"{utc()} funding: {e!r}"[:300])
            st["errors"] = st.get("errors", [])[-20:]
            save(st)
            publish(summarize(st))
        except Exception as e:
            print(utc(), "error", repr(e), flush=True)
            try:
                publish(dict(updated_utc=utc(), paper_only=True, opened_utc=None, error=repr(e)[:300]))
            except Exception:
                pass
        if ONCE:
            return
        time.sleep(3600)


if __name__ == "__main__":
    main()
