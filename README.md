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
- A watchdog opens a GitHub issue (and emails the owner) if the bot stops reporting for 2 hours.

The running verdict is in `data/summary.json` under `test`.

Research only, not financial advice.
