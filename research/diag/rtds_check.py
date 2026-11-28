"""Listen to Polymarket's live price feed for 90 s and count what arrives (diagnostic)."""
import asyncio, json, time, collections, websockets
async def main():
    flt = json.dumps({"symbol": "btc/usd"}, separators=(",", ":"))
    sub = {"action": "subscribe", "subscriptions": [{"topic": t, "type": "*", "filters": flt} for t in ("crypto_prices_chainlink", "crypto_prices_twap_sixty")]}
    c = collections.Counter(); last = {}; t0 = time.time()
    async with websockets.connect("wss://ws-live-data.polymarket.com", ping_interval=None, open_timeout=10) as ws:
        await ws.send(json.dumps(sub))
        async def pinger():
            while True:
                await asyncio.sleep(5); await ws.send("PING")
        p = asyncio.create_task(pinger())
        while time.time() - t0 < 90:
            try:
                m = await asyncio.wait_for(ws.recv(), timeout=20)
            except asyncio.TimeoutError:
                print("silent 20 s"); continue
            if not m or m[0] != "{":
                c["non-json:" + m[:20]] += 1; continue
            j = json.loads(m); k = str(j.get("topic")) + "/" + str(j.get("type"))
            c[k] += 1; last[k] = m[:300]
        p.cancel()
    print(dict(c))
    for k, v in last.items(): print(k, v)
asyncio.run(main())
