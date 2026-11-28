"""Pick well-covered days from the dataset manifest and split them into job chunks."""
import os, json, sys
from huggingface_hub import hf_hub_download
R = "polydata5/polymarket-updown-5m-ws"
coin = os.environ.get("COIN", "BTC")
min_minutes = int(os.environ.get("MIN_MINUTES", "1380"))   # >= 95.8% of the day captured
chunk = int(os.environ.get("CHUNK", "3"))
p = hf_hub_download(R, "manifest.json", repo_type="dataset", token=os.environ["HF_TOKEN"])
man = json.load(open(p))["days"]
days = []
for d, info in sorted(man.items()):
    f = info.get("files", {})
    c, r = f.get(f"{coin}_clob"), f.get(f"{coin}_rtds")
    if c and r and c["minutes"] >= min_minutes and r["minutes"] >= min_minutes:
        days.append(d)
only = os.environ.get("ONLY_DAYS", "").strip()
if only:
    days = [d for d in days if d in only.split(",")]
chunks = [",".join(days[i:i + chunk]) for i in range(0, len(days), chunk)]
print(f"{len(days)} usable {coin} days of {len(man)}", file=sys.stderr)
with open(os.environ.get("GITHUB_OUTPUT", "/dev/stdout"), "a") as fh:
    fh.write("chunks=" + json.dumps(chunks) + "\n")
    fh.write("ndays=" + str(len(days)) + "\n")
