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
from cobweb_py import CobwebSim, BacktestConfig, fix_timestamps, to_signals, score_by_id
import yfinance as yf

# 1. Connect
sim = CobwebSim("https://web-production-83f3e.up.railway.app")

# 2. Download data
df = yf.download("AAPL", start="2020-01-01", end="2024-12-31")
df.columns = df.columns.get_level_values(0)
df = df.reset_index().rename(columns={"Date": "timestamp"})
rows = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].to_dict("records")
data = fix_timestamps(rows)

# 3. Enrich with technical features
feats = sim.enrich(data, feature_ids=[1, 11, 36])
enriched_rows = feats["rows"]

# 4. Score and generate signals
scores = score_by_id(enriched_rows, {11: 0.3, 36: 0.3, 1: 0.4})
signals = to_signals(scores, entry_th=0.20, exit_th=0.05, use_shorts=False)

# 5. Backtest with realistic friction
bt = sim.backtest(data, signals=signals, config=BacktestConfig(
    initial_cash=10_000,
    exec_horizon="swing",
    fee_bps=1.0,
))

print(f"Return: {bt['metrics']['total_return']:.2%}")
print(f"Sharpe: {bt['metrics']['sharpe_ann']:.2f}")
print(f"Max DD: {bt['metrics']['max_drawdown']:.2%}")
```

## Examples

See [cobweb-py-examples](https://github.com/cobwebSim/cobweb-py-examples) for 8 runnable Jupyter notebook examples.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cobwebSim/cobweb-py-examples/blob/main/cobweb_py_examples.ipynb)

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
