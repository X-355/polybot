"""Side-by-side: does the last-30-seconds edge survive when stale order-book snapshots are removed?"""
import json, glob
out = {}
for f in sorted(glob.glob("research/results/pm_study_BTC*.json")):
    d = json.load(open(f)); key = f.split("pm_study_BTC")[1].replace(".json", "") or "_all"
    rows = {}
    for r in d["strategies"]:
        if r["left"] in (60, 30, 10) and r["lag"] in (0, 1) and r.get("trades", 0) >= 20 and (
                r["strategy"].startswith("Gradient tree") or r["strategy"].startswith("Always buy the fav") or r["strategy"].startswith("Random")):
            f05 = r["fee 0.05"]
            rows[f'{r["left"]}s lag{r["lag"]} {r["strategy"]}'] = dict(n=r["trades"], ret=f05["return_pct"], t=f05["t"])
    out[key] = dict(lag=d.get("lag"), days=d["days"], snapshots=d["snapshots"],
                    brier_last30=[(q["left"], q["brier_market"], q["brier_gbm"]) for q in d["forecast"] if q["left"] <= 30],
                    strategies=rows)
json.dump(out, open("research/results/compare.json", "w"), indent=1)
for k, v in out.items():
    print("=====", k, v["lag"], "snapshots", v["snapshots"])
    print("  brier (left, market, gbm):", v["brier_last30"])
    for name, r in v["strategies"].items():
        if "Gradient" in name or "favourite" in name:
            print(f"  {name:<60} n={r['n']:>5} ret={r['ret']:+7.2f}% t={r['t']}")
