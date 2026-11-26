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

Research only, not financial advice.
