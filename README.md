# polybot

A **paper-trading research bot** for Polymarket's 5-minute Bitcoin "Up or Down" markets.
It has no wallet, no keys and places no orders: it only reads public market data and logs pretend $10 trades.

**Why:** a backtest on 73 days of recorded order books (20,995 markets, Sep–Nov 2026) found the market's own price
is a fair forecast for most of each window, and every simple rule loses money after Polymarket's taker fee. The only
lead was a gradient-tree model in the **last 30 seconds**, profitable on both halves of the data. This bot checks
whether that lead survives live, where faster bots compete for the same prices.
Study: [Research Lab, section 5](https://x-355.github.io/atl-radar/lab/#polymarket).

**How it decides:** at 30 s and 10 s before each window closes it computes the same features as the backtest
(Chainlink price vs the price to beat, volatility, time left, both order books) and buys one side at the best ask
only if the model's probability minus the ask and the fee (0.05 × p × (1 − p) per share) is more than 2¢.
It records the ask one second later too, to see the cost of being slower, and scores every trade against
Polymarket's official result.

**Files:** `data/summary.json` (totals), `data/trades.csv` (every paper trade), `data/decisions.csv` (every check,
including skipped and no-trade ones), updated every 30 minutes by the server.

## The test, decided in advance (27 November 2026)

So the result can be trusted whichever way it goes, the rules were fixed before the results came in:

- **No verdict before 300 settled paper trades.** Early wins or losses are noise.
- **Pessimistic scoring:** every trade is scored at the *worse* of the price when the bot decided and the price one
  second later, after Polymarket's taker fee (0.05 × p × (1 − p) per share). The share of trades where a $10 stake
  would not fit at the best price is reported too.
- **Edge confirmed** only if the 95% interval of return per $1 (the two checks in one window treated as related)
  is above zero **and** the bot beats simply buying the favourite in the same windows at the same moments.
- **No edge** if the interval is below zero. **Not proven** if still unclear at 1,000 trades; the test then ends.
- **Fresh prices only:** the bot measures how far its view of the order book is behind Polymarket's own timestamps and
  refuses to trade if it is more than 2 seconds behind. The first 6 trades (27 Nov, before this check existed) are
  excluded from the test, because a slow server can show prices that no longer exist and produce fake wins.
- **Early stop (added 28 Nov 2026, before further results):** the backtest found a trade in 4.9% of 5-minute
  windows. If on 11 December 2026 (two weeks of fresh-price checks) the bot has traded in fewer than a third of
  that, the opportunities are not showing up live and the test stops early.
- A watchdog opens a GitHub issue (and emails the owner) if the bot stops reporting for 2 hours.

The running verdict is in `data/summary.json` under `test`.

Research only, not financial advice.

## ALT/BTC paper bot (`altbot/`) — no real money

A second paper test on the same server. It checks one idea with rules fixed in advance: a frozen
gradient-tree model (trained on Binance 5-minute candles 2021 – Oct 2026) predicts the next 15 minutes of 10
ALT/BTC pairs (ETH, BNB, SOL, XRP, ADA, DOGE, LTC, LINK, DOT, AVAX against BTC).

**Rules (decided before the first trade, not changed afterwards)**
- At every 5-minute candle close: trade only if |predicted 15-min move| > 1.0 × (0.3% + one price step).
- Prices are the real best ask (to buy) or best bid (to sell) from Binance, the worse of two quotes 1 s apart,
  both when opening and when closing 15 minutes later. Binance fee 0.1% each way. Results in BTC terms.
- One open position per pair. Skips: missing candles, candle arrived >10 s late, bid/ask older than 60 s.
- Start-up self-check: the bot refuses to trade unless its own feature code reproduces the training numbers.
- Control: on every trade, a coin-flip side on the same pair at the same moment, priced the same way.
- Verdict after 200 trades (final at 600): edge only if the 95% confidence interval of the mean net per
  trade (clustered by candle, since pairs often move together) is above zero.
- Activity check: after 30 days, stop if trades are fewer than one third of the expected pace.

**What the backtest says to expect (honest):** run the way the bot trades (every candle, one position per
pair, entering one candle late), 2024 – Oct 2026 gave 290 trades (0.28 a day), +0.46% average per trade —
but 2024 was negative (−0.38%) and most of the profit came from a few sharp moves in 2025 when many pairs
moved together. So a verdict will take a long time, and the most likely result is "no edge".

## BTC funding-rate carry tracker (`carrybot/`) — no real money

Holds a pretend $10,000 of BTC on Binance spot and shorts the same amount of BTC on the BTCUSDT perpetual,
opened once at the real best ask/bid with fees (spot 0.1%, futures 0.05%), never rebalanced. Every 8-hour funding
payment is credited or charged as Binance does. Marked hourly at the prices you could close at, after closing fees.
Return is shown on $20,000 of capital (the BTC + 100% margin). Backtest (2020 – Oct 2026): it earned about
+6% a year on capital, but only about 2.6% in 2025 and 1.3% in 2026 to October. The question it answers: is it
currently paying more than cash?
