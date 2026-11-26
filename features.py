"""Feature maths shared by the backtest training and the live paper bot. Keep the two in sync."""
import math

FEATS = ["x", "xf", "sdf", "vol_s", "tl", "z", "mid_up", "spr_up", "spr_dn", "imb", "overround"]
FEE_COEF = 0.05          # Polymarket taker fee for BTC 5-minute markets: coef * p * (1 - p) per share
EDGE = 0.02              # trade only if model probability - ask - fee > 2 cents
LEFTS = (30, 10)         # seconds before the end of the window when the bot decides


def fee(p):
    return FEE_COEF * p * (1 - p)


def vol_per_second(cl, end_sec):
    """Per-second stdev of Chainlink log returns over the 300 s up to end_sec (needs 100+ returns)."""
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
    return math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1))


def build(start, known, cl, ptb, up, dn):
    """start: window start (unix s); known: latest Chainlink observation second we have;
    cl: {sec: price}; ptb: price to beat; up/dn: dicts with bid, ask, ask_sz.
    Returns (feature dict, p_formula) or (None, reason)."""
    if known is None or known not in cl:
        return None, "no price"
    sig = vol_per_second(cl, known)
    if sig is None:
        return None, "no volatility"
    S, end = cl[known], start + 300
    tl = max(1, 300 - (known - start))
    win = [cl[s] for s in range(max(end - 60, start), known + 1) if s in cl] if known >= end - 60 else []
    A = sum(win) / len(win) if win else S
    k = max(0, 60 - tl)
    if tl >= 60:
        mean_f, sd_f = S, sig * S * math.sqrt(max(1, tl - 40))
    else:
        mean_f, sd_f = (k * A + tl * S) / 60, sig * S * math.sqrt(tl / 3) * tl / 60
    z = (mean_f - ptb) / max(sd_f, 1e-9)          # left unclipped, exactly as in training
    f = dict(x=(S - ptb) / ptb, xf=(mean_f - ptb) / ptb, sdf=sd_f / ptb, vol_s=sig, tl=tl, z=z,
             mid_up=(up["bid"] + up["ask"]) / 2, spr_up=up["ask"] - up["bid"], spr_dn=dn["ask"] - dn["bid"],
             imb=math.log((up["ask_sz"] + 1) / (dn["ask_sz"] + 1)), overround=up["ask"] + dn["ask"] - 1)
    p_formula = 0.5 * (1 + math.erf(max(-30, min(30, z)) / math.sqrt(2)))
    return f, p_formula


def decide(p_up, up_ask, dn_ask):
    """Same rule as the backtest: buy the side whose edge after the fee beats EDGE."""
    eu = p_up - up_ask - fee(up_ask)
    ed = (1 - p_up) - dn_ask - fee(dn_ask)
    if eu > EDGE and eu >= ed:
        return "Up", eu
    if ed > EDGE:
        return "Down", ed
    return None, max(eu, ed)
