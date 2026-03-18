# cobweb-py

Python SDK for realistic backtesting with [CobwebSim](https://cobweb.market).

**Your backtest is lying to you.** Most backtesters only model transaction fees. cobweb-py adds the friction costs they miss: bid-ask spread, slippage, and market impact.

## Features

- **71 technical features** computed server-side (RSI, SMA, EMA, ATR, Bollinger, MACD, and more)
- **Realistic execution modeling** with fees, spread, slippage, and market impact
- **27 interactive Plotly charts** (equity curves, drawdowns, regime analysis, trade logs)
- Works with **any OHLCV data** — CSV files, pandas DataFrames, or yfinance

## Installation

```bash
pip install cobweb-py
```

With visualization support (Plotly charts):

```bash
pip install cobweb-py[viz]
```

Requires Python 3.9+. No API key required.

## Quick Start

```python
import cobweb_py as cw

# 1. Connect & download data
sim = cw.CobwebSim("https://web-production-83f3e.up.railway.app")
asset_df = cw.from_yfinance("AAPL", "2020-01-01", "2024-12-31")
bench_df = cw.from_yfinance("SPY", "2020-01-01", "2024-12-31")

# 2. Enrich with technical features
rows = sim.enrich_rows(asset_df, feature_ids=[1, 2, 3, 11, 36, 14, 70, 71])

# 3. Score and generate signals
scores = cw.score_by_id(rows, {11: 0.3, 36: 0.3, 14: 0.4})
signals = cw.to_signals(scores, entry_th=0.20, exit_th=0.05, use_shorts=False)

# 4. Backtest with realistic friction
bt = sim.backtest(
    rows, signals=signals,
    compute_features=True, feature_ids=[70, 71],
    benchmark=bench_df,
    config=cw.BacktestConfig(initial_cash=10_000, exec_horizon="swing", fee_bps=1.0),
)

# 5. View results
cw.print_signal(bt)
metrics = cw.format_metrics(bt)
for label, value in metrics.items():
    print(f"{label}: {value}")

# 6. Save interactive charts
cw.plots.save_equity_plot(bt, out_html="out/equity.html")
cw.plots.save_metrics_table(bt, out_html="out/metrics.html")
cw.plots.save_trades_table(bt, out_html="out/trades.html")
cw.plots.save_score_plot(rows, scores, out_html="out/score.html")
cw.plots.save_price_and_score_plot(rows, scores, out_html="out/price_and_score.html")
```

## Examples

See [backtest_under_20](https://github.com/cobwebSim/backtest_under_20) for a complete walkthrough notebook with explanations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cobwebSim/backtest_under_20/blob/main/backtest.ipynb)

## API Reference

### CobwebSim Client

```python
sim = CobwebSim("https://web-production-83f3e.up.railway.app")
```

| Method | Description |
|--------|-------------|
| `sim.health()` | Check API status |
| `sim.enrich(data, feature_ids=[...])` | Compute technical features |
| `sim.backtest(data, signals, config=...)` | Run backtest with realistic execution |
| `sim.plots(data, bt, plot_ids=[...])` | Generate interactive charts |

### BacktestConfig

```python
BacktestConfig(
    initial_cash=10_000,       # Starting capital
    exec_horizon="swing",      # intraday | swing | longterm
    fee_bps=1.0,               # Broker fee (basis points)
    half_spread_bps=2.0,       # Bid-ask half-spread
    base_slippage_bps=1.0,     # Slippage
    impact_coeff=1.0,          # Market impact multiplier
    allow_margin=False,        # Allow short positions
    max_leverage=1.0,          # Maximum leverage
)
```

### Scoring & Signals

```python
# Weight features by ID
scores = score_by_id(rows, {11: 0.3, 36: 0.3, 14: 0.4})

# Convert to trading signals
signals = to_signals(scores, entry_th=0.20, exit_th=0.05, use_shorts=False)
```

### Pipeline (High-Level)

```python
from cobweb_py import Pipeline

pipe = Pipeline("https://web-production-83f3e.up.railway.app", "stock.csv")
result = pipe.run(weights={36: 0.3, 11: 0.3, 1: 0.4})
print(result["metrics"])
```

### Helper Functions

| Function | Description |
|----------|-------------|
| `fix_timestamps(rows)` | Normalize dates to ISO format |
| `load_csv(path)` | Load CSV into API-ready format |
| `align(base, benchmark)` | Align two datasets by timestamp |
| `to_signals(scores, entry_th, exit_th)` | Convert scores to buy/hold/sell |
| `score_by_id(rows, weights)` | Weighted feature scoring |
| `print_signal(bt)` | Print current signal |
| `save_all_plots(sim, data, bt, plot_ids)` | Save interactive HTML charts |
| `show_features()` | Print feature reference table |
| `show_plots()` | Print plot reference table |

## Documentation

Full documentation at [cobweb.market/docs.html](https://cobweb.market/docs.html).

## License

MIT
