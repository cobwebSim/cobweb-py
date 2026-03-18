# cobweb_py/sweep.py
"""
Strategy-agnostic market sweep and parameter sweep.

market_sweep: apply one strategy across many tickers → SweepResult
param_sweep:  grid-search strategy parameters on one ticker → ParamSweepResult

Strategies can be:
  - WeightedStrategy (built-in, wraps score_by_id + to_signals)
  - RuleStrategy     (user-defined rule function on a DataFrame)
  - ModelStrategy    (any ML model with .predict() or __call__)
  - Any callable     (rows → List[float] of signals)
"""
from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from typing import (
    Any, Callable, Dict, Iterator, List, Mapping, Optional,
    Protocol, Sequence, Tuple, Union, runtime_checkable,
)

import pandas as pd

from .client import CobwebSim, CobwebError, BacktestConfig, from_yfinance
from .scoring import score_by_id, FEATURES
from .utils import to_signals, signal_label


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------

Rows = List[Dict[str, Any]]


@runtime_checkable
class Strategy(Protocol):
    """Any callable that converts enriched OHLCV rows into trading signals."""

    def __call__(self, rows: Rows) -> List[float]:
        """
        Takes enriched OHLCV rows (list of dicts with all 71 features).
        Returns a list of position signals, one per row:
          1.0 = long, 0.0 = flat, -1.0 = short
        """
        ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

class WeightedStrategy:
    """
    Score features by weighted sum, convert to signals via thresholds.

    This is the standard cobweb-py strategy: assign weights to features,
    compute a composite score, enter/exit based on threshold crossings.

    Example::

        strategy = WeightedStrategy(
            weights={36: 0.3, 11: 0.3, 1: 0.4},
            entry_th=0.20,
            exit_th=0.05,
        )
        signals = strategy(enriched_rows)
    """

    def __init__(
        self,
        weights: Mapping[Union[int, str], float],
        *,
        entry_th: float = 0.20,
        exit_th: float = 0.05,
        use_shorts: bool = False,
        normalize: bool = True,
        rsi_to_unit: bool = True,
    ):
        self.weights = dict(weights)
        self.entry_th = entry_th
        self.exit_th = exit_th
        self.use_shorts = use_shorts
        self.normalize = normalize
        self.rsi_to_unit = rsi_to_unit

    def __call__(self, rows: Rows) -> List[float]:
        scores = score_by_id(
            rows, self.weights,
            normalize=self.normalize,
            rsi_to_unit=self.rsi_to_unit,
        )
        return to_signals(scores, self.entry_th, self.exit_th, self.use_shorts)

    @property
    def params(self) -> Dict[str, Any]:
        """Return parameters dict for result tracking."""
        return {
            "weights": self.weights,
            "entry_th": self.entry_th,
            "exit_th": self.exit_th,
            "use_shorts": self.use_shorts,
        }

    def __repr__(self) -> str:
        w = ", ".join(f"{k}:{v}" for k, v in self.weights.items())
        return f"WeightedStrategy({{{w}}}, entry={self.entry_th}, exit={self.exit_th})"


class RuleStrategy:
    """
    Apply a user-defined rule function that returns signals directly.

    The rule function receives a DataFrame with all enriched columns and
    must return a list of signals (1.0 / 0.0 / -1.0).

    Example::

        def mean_reversion(df):
            signals, pos = [], 0
            for _, row in df.iterrows():
                if pos == 0 and row["rsi_14"] < 30:
                    pos = 1
                elif pos == 1 and row["rsi_14"] > 70:
                    pos = 0
                signals.append(float(pos))
            return signals

        strategy = RuleStrategy(mean_reversion)
    """

    def __init__(self, rule_fn: Callable[[pd.DataFrame], List[float]], **params: Any):
        self._fn = rule_fn
        self._params = params

    def __call__(self, rows: Rows) -> List[float]:
        df = pd.DataFrame(rows)
        result = self._fn(df, **self._params)
        return [float(x) for x in result]

    @property
    def params(self) -> Dict[str, Any]:
        return dict(self._params)

    def __repr__(self) -> str:
        name = getattr(self._fn, "__name__", "rule")
        return f"RuleStrategy({name}, {self._params})"


class ModelStrategy:
    """
    Wrap any ML model that has a ``.predict()`` method or is callable.

    The model receives a DataFrame of selected feature columns and must
    return continuous scores.  Scores are converted to signals via thresholds.

    Example — sklearn::

        model = joblib.load("my_random_forest.pkl")
        strategy = ModelStrategy(
            model=model,
            feature_cols=["rsi_14", "vol_20", "ret_1d", "macd_hist"],
            entry_th=0.5,
            exit_th=0.3,
        )

    Example — any callable::

        def my_neural_net(df):
            tensor = torch.tensor(df.values, dtype=torch.float32)
            return model(tensor).detach().numpy().flatten()

        strategy = ModelStrategy(model=my_neural_net, feature_cols=[...])
    """

    def __init__(
        self,
        model: Any,
        feature_cols: List[str],
        *,
        entry_th: float = 0.5,
        exit_th: float = 0.3,
        use_shorts: bool = False,
    ):
        self.model = model
        self.feature_cols = list(feature_cols)
        self.entry_th = entry_th
        self.exit_th = exit_th
        self.use_shorts = use_shorts

    def __call__(self, rows: Rows) -> List[float]:
        df = pd.DataFrame(rows)
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise CobwebError(
                f"ModelStrategy: missing columns in data: {missing}. "
                f"Available: {list(df.columns)[:15]}..."
            )
        X = df[self.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        if hasattr(self.model, "predict"):
            scores = self.model.predict(X)
        elif callable(self.model):
            scores = self.model(X)
        else:
            raise ValueError("Model must have a .predict() method or be callable")

        scores = [float(s) for s in scores]
        return to_signals(scores, self.entry_th, self.exit_th, self.use_shorts)

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "feature_cols": self.feature_cols,
            "entry_th": self.entry_th,
            "exit_th": self.exit_th,
        }

    def __repr__(self) -> str:
        n = len(self.feature_cols)
        return f"ModelStrategy({n} features, entry={self.entry_th}, exit={self.exit_th})"


# ---------------------------------------------------------------------------
# Result classes
# ---------------------------------------------------------------------------

@dataclass
class SweepRow:
    """Result for a single ticker in a market sweep."""
    ticker: str
    signal: str                              # "BUY" | "SELL" | "HOLD"
    close: float                             # latest close price
    signal_age: int                          # bars since signal last changed
    signals: List[float]                     # full signal series
    enriched_rows: List[Dict[str, Any]]      # full enriched data (for drill-down)
    error: Optional[str] = None              # None if success


class SweepResult:
    """
    Result of :func:`market_sweep`.

    Behaves like a list of :class:`SweepRow` with convenience methods for
    filtering, sorting, and display.
    """

    def __init__(
        self,
        rows: List[SweepRow],
        elapsed_ms: float,
        errors: List[Tuple[str, str]],
    ):
        self.rows = rows
        self.elapsed_ms = elapsed_ms
        self.errors = errors

    # --- Filtering ---

    def buys(self) -> "SweepResult":
        """Filter to BUY signals only."""
        return SweepResult(
            [r for r in self.rows if r.signal == "BUY"],
            self.elapsed_ms, self.errors,
        )

    def sells(self) -> "SweepResult":
        """Filter to SELL signals only."""
        return SweepResult(
            [r for r in self.rows if r.signal == "SELL"],
            self.elapsed_ms, self.errors,
        )

    def holds(self) -> "SweepResult":
        """Filter to HOLD signals only."""
        return SweepResult(
            [r for r in self.rows if r.signal == "HOLD"],
            self.elapsed_ms, self.errors,
        )

    # --- Sorting ---

    def sort_by(self, key: str = "signal_age", ascending: bool = True) -> "SweepResult":
        """Sort by ``close``, ``signal_age``, or ``ticker``."""
        reverse = not ascending
        self.rows.sort(key=lambda r: getattr(r, key, 0), reverse=reverse)
        return self

    def top(self, n: int = 10) -> "SweepResult":
        """Return top *n* rows."""
        return SweepResult(self.rows[:n], self.elapsed_ms, self.errors)

    # --- Display ---

    def to_df(self) -> pd.DataFrame:
        """Convert to DataFrame (ticker, signal, close, signal_age)."""
        return pd.DataFrame([
            {
                "ticker": r.ticker,
                "signal": r.signal,
                "close": r.close,
                "signal_age": r.signal_age,
            }
            for r in self.rows
            if r.error is None
        ])

    def print_summary(self) -> None:
        """Pretty-print results table to stdout."""
        ok = [r for r in self.rows if r.error is None]
        if not ok:
            print("No results.")
            return

        w_t = max(len(r.ticker) for r in ok)
        print()
        print(f"{'Ticker':<{w_t}}  {'Signal':<6}  {'Close':>10}  {'Age':>4}")
        print(f"{'-'*w_t}  {'-'*6}  {'-'*10}  {'-'*4}")
        for r in ok:
            print(f"{r.ticker:<{w_t}}  {r.signal:<6}  {r.close:>10.2f}  {r.signal_age:>4}")

        n_buy = sum(1 for r in ok if r.signal == "BUY")
        n_sell = sum(1 for r in ok if r.signal == "SELL")
        n_hold = sum(1 for r in ok if r.signal == "HOLD")
        n_err = len(self.errors)
        print()
        parts = [f"{n_buy} BUY", f"{n_sell} SELL", f"{n_hold} HOLD"]
        if n_err:
            parts.append(f"{n_err} errors")
        print(" | ".join(parts))
        print(f"Completed in {self.elapsed_ms / 1000:.1f}s")
        print()

    # --- Dunder ---

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[SweepRow]:
        return iter(self.rows)

    def __getitem__(self, idx: int) -> SweepRow:
        return self.rows[idx]

    def __repr__(self) -> str:
        ok = sum(1 for r in self.rows if r.error is None)
        return f"SweepResult({ok} tickers, {len(self.errors)} errors)"


@dataclass
class ParamRow:
    """Result for a single parameter combo in a parameter sweep."""
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    sharpe: float
    total_return: float
    max_dd: float
    sortino: float
    trades: int
    signals: List[float]
    error: Optional[str] = None


class ParamSweepResult:
    """
    Result of :func:`param_sweep`.

    Behaves like a list of :class:`ParamRow` with convenience methods for
    analysis, filtering, and display.
    """

    def __init__(
        self,
        rows: List[ParamRow],
        elapsed_ms: float,
        total_combos: int,
        errors: List[Tuple[Dict, str]],
        enriched_rows: List[Dict[str, Any]],
    ):
        self.rows = rows
        self.elapsed_ms = elapsed_ms
        self.total_combos = total_combos
        self.errors = errors
        self.enriched_rows = enriched_rows

    # --- Selection ---

    def best(self, n: int = 1) -> "ParamSweepResult":
        """Return top *n* combos (already sorted by rank metric)."""
        return ParamSweepResult(
            self.rows[:n], self.elapsed_ms, self.total_combos,
            self.errors, self.enriched_rows,
        )

    def best_params(self) -> Dict[str, Any]:
        """Return the parameter dict of the best combo."""
        if not self.rows:
            return {}
        return dict(self.rows[0].params)

    # --- Sorting ---

    def sort_by(self, metric: str, ascending: bool = False) -> "ParamSweepResult":
        """Re-sort by a different metric (sharpe, total_return, max_dd, sortino, trades)."""
        reverse = not ascending
        self.rows.sort(
            key=lambda r: getattr(r, metric, r.metrics.get(metric, 0)),
            reverse=reverse,
        )
        return self

    # --- Filtering ---

    def filter(
        self,
        min_sharpe: Optional[float] = None,
        min_trades: Optional[int] = None,
        max_dd: Optional[float] = None,
    ) -> "ParamSweepResult":
        """Filter results by metric thresholds."""
        filtered = self.rows
        if min_sharpe is not None:
            filtered = [r for r in filtered if r.sharpe >= min_sharpe]
        if min_trades is not None:
            filtered = [r for r in filtered if r.trades >= min_trades]
        if max_dd is not None:
            filtered = [r for r in filtered if r.max_dd >= max_dd]  # max_dd is negative
        return ParamSweepResult(
            filtered, self.elapsed_ms, self.total_combos,
            self.errors, self.enriched_rows,
        )

    # --- Display ---

    def to_df(self) -> pd.DataFrame:
        """DataFrame with one row per combo. Params become columns."""
        records = []
        for r in self.rows:
            if r.error is not None:
                continue
            row = dict(r.params)
            row.update({
                "sharpe": r.sharpe,
                "total_return": r.total_return,
                "max_dd": r.max_dd,
                "sortino": r.sortino,
                "trades": r.trades,
            })
            records.append(row)
        return pd.DataFrame(records)

    def print_summary(self, top_n: int = 10) -> None:
        """Pretty-print top *n* results."""
        ok = [r for r in self.rows if r.error is None]
        if not ok:
            print("No results.")
            return

        # Collect all param keys for column headers
        param_keys = list(ok[0].params.keys())

        print()
        # Header
        parts = [f"{'Rank':>4}"]
        for k in param_keys:
            parts.append(f"{k:>8}")
        parts.extend([f"{'Sharpe':>8}", f"{'Return':>8}", f"{'MaxDD':>8}", f"{'Trades':>7}"])
        print("  ".join(parts))
        print("  ".join(["-" * len(p) for p in parts]))

        # Rows
        for i, r in enumerate(ok[:top_n]):
            parts = [f"{i+1:>4}"]
            for k in param_keys:
                v = r.params.get(k, "")
                if isinstance(v, float):
                    parts.append(f"{v:>8.3f}")
                elif isinstance(v, dict):
                    parts.append(f"{'(dict)':>8}")
                else:
                    parts.append(f"{str(v):>8}")
            parts.extend([
                f"{r.sharpe:>8.2f}",
                f"{r.total_return:>+7.1%}",
                f"{r.max_dd:>+7.1%}",
                f"{r.trades:>7}",
            ])
            print("  ".join(parts))

        print()
        best_sharpe = max(r.sharpe for r in ok) if ok else 0
        print(f"{len(ok)} combos tested | best Sharpe: {best_sharpe:.2f} | {len(self.errors)} errors")
        print(f"Completed in {self.elapsed_ms / 1000:.1f}s")
        print()

    def heatmap(self, x: str, y: str, metric: str = "sharpe") -> Any:
        """
        2D heatmap of *metric* across two swept parameters.

        Requires plotly (optional dependency).
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("heatmap() requires plotly: pip install plotly")

        df = self.to_df()
        if x not in df.columns or y not in df.columns:
            raise ValueError(f"Columns '{x}' and '{y}' must be in params. Available: {list(df.columns)}")

        pivot = df.pivot_table(index=y, columns=x, values=metric, aggfunc="mean")
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale="RdYlGn",
            colorbar_title=metric,
        ))
        fig.update_layout(
            title=f"{metric} by {x} vs {y}",
            xaxis_title=x,
            yaxis_title=y,
        )
        return fig

    # --- Dunder ---

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[ParamRow]:
        return iter(self.rows)

    def __getitem__(self, idx: int) -> ParamRow:
        return self.rows[idx]

    def __repr__(self) -> str:
        ok = sum(1 for r in self.rows if r.error is None)
        return f"ParamSweepResult({ok} combos, {len(self.errors)} errors)"


# ---------------------------------------------------------------------------
# market_sweep
# ---------------------------------------------------------------------------

def _compute_signal_age(signals: List[float]) -> int:
    """Count bars since the last signal change."""
    if not signals:
        return 0
    current = signals[-1]
    age = 0
    for s in reversed(signals):
        if s == current:
            age += 1
        else:
            break
    return age


def _sweep_one_ticker(
    sim: CobwebSim,
    ticker: str,
    strategy: Callable[[Rows], List[float]],
    start: str,
    end: Optional[str],
    feature_ids: Optional[List[int]],
) -> SweepRow:
    """Process a single ticker for market_sweep. Runs in a worker thread."""
    # 1. Fetch data
    data = from_yfinance(ticker, start, end)

    # 2. Enrich
    resp = sim.enrich(data, feature_ids=feature_ids)
    enriched = resp.get("rows", [])
    if not enriched:
        raise CobwebError(f"No enriched rows for {ticker}")

    # 3. Run strategy
    signals = strategy(enriched)
    if len(signals) != len(enriched):
        raise CobwebError(
            f"Strategy returned {len(signals)} signals for {len(enriched)} rows"
        )

    # 4. Extract latest info
    latest = enriched[-1]
    close_col = "Close" if "Close" in latest else "close"
    latest_close = float(latest.get(close_col, 0.0))
    latest_signal = signal_label(signals[-1])
    age = _compute_signal_age(signals)

    return SweepRow(
        ticker=ticker,
        signal=latest_signal,
        close=latest_close,
        signal_age=age,
        signals=signals,
        enriched_rows=enriched,
    )


def market_sweep(
    sim: CobwebSim,
    tickers: List[str],
    strategy: Optional[Callable[[Rows], List[float]]] = None,
    *,
    # Shorthand for WeightedStrategy (ignored if strategy is provided)
    weights: Optional[Mapping[Union[int, str], float]] = None,
    entry_th: float = 0.20,
    exit_th: float = 0.05,
    use_shorts: bool = False,
    # Data options
    start: Optional[str] = None,
    end: Optional[str] = None,
    feature_ids: Optional[List[int]] = None,
    # Execution options
    max_workers: int = 4,
    on_error: str = "warn",
    progress: bool = True,
) -> SweepResult:
    """
    Apply one strategy across many tickers.

    Accepts either a ``strategy`` callable or shorthand ``weights`` +
    threshold kwargs (which auto-build a :class:`WeightedStrategy`).

    Args:
        sim:          Connected :class:`CobwebSim` client.
        tickers:      List of ticker symbols (e.g. ``["AAPL", "MSFT"]``).
        strategy:     Any callable that takes enriched rows and returns
                      signals. If provided, takes precedence over *weights*.
        weights:      Shorthand -- auto-builds a :class:`WeightedStrategy`.
        entry_th:     Entry threshold (used with *weights* shorthand).
        exit_th:      Exit threshold (used with *weights* shorthand).
        use_shorts:   Enable shorts (used with *weights* shorthand).
        start:        yfinance start date (default ``"2023-01-01"``).
        end:          yfinance end date (default: today).
        feature_ids:  Feature IDs to enrich (default: all 71).
        max_workers:  Thread pool size for parallel processing.
        on_error:     ``"warn"`` (default), ``"raise"``, or ``"skip"``.
        progress:     Print progress to stdout.

    Returns:
        :class:`SweepResult` with one :class:`SweepRow` per ticker.

    Example::

        # Shorthand (weights-based):
        result = market_sweep(sim, ["AAPL", "MSFT"],
                              weights={36: 0.3, 11: 0.3, 1: 0.4})

        # Custom strategy:
        result = market_sweep(sim, ["AAPL", "MSFT"],
                              strategy=my_ml_strategy)
    """
    # Resolve strategy
    if strategy is not None:
        _strategy = strategy
    elif weights is not None:
        _strategy = WeightedStrategy(
            weights, entry_th=entry_th, exit_th=exit_th, use_shorts=use_shorts,
        )
    else:
        raise ValueError("Provide either `strategy` or `weights`.")

    if start is None:
        start = "2023-01-01"

    t0 = time.time()
    results: List[SweepRow] = []
    errors: List[Tuple[str, str]] = []
    total = len(tickers)

    def _process(ticker: str, idx: int) -> SweepRow:
        row = _sweep_one_ticker(sim, ticker, _strategy, start, end, feature_ids)
        if progress:
            print(f"[{idx}/{total}] {ticker}: {row.signal}")
        return row

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_sweep_one_ticker, sim, t, _strategy, start, end, feature_ids): (t, i + 1)
            for i, t in enumerate(tickers)
        }
        for future in as_completed(futures):
            ticker, idx = futures[future]
            try:
                row = future.result()
                if progress:
                    print(f"[{idx}/{total}] {ticker}: {row.signal}")
                results.append(row)
            except Exception as e:
                err_msg = str(e)
                errors.append((ticker, err_msg))
                if on_error == "raise":
                    raise
                elif on_error == "warn":
                    warnings.warn(f"market_sweep: {ticker} failed: {err_msg}", stacklevel=2)
                    if progress:
                        print(f"[{idx}/{total}] {ticker}: ERROR - {err_msg}")
                # on_error == "skip": silently skip

    elapsed = (time.time() - t0) * 1000
    return SweepResult(results, elapsed, errors)


# ---------------------------------------------------------------------------
# param_sweep
# ---------------------------------------------------------------------------

def _extract_metrics(bt: Dict[str, Any]) -> Dict[str, float]:
    """Pull standard metrics from a backtest result."""
    m = bt.get("metrics", {})
    return {
        "sharpe": float(m.get("sharpe", 0.0)),
        "total_return": float(m.get("total_return", 0.0)),
        "max_dd": float(m.get("max_drawdown", 0.0)),
        "sortino": float(m.get("sortino", 0.0)),
        "trades": int(m.get("total_trades", 0)),
    }


def _build_weighted_combos(
    weight_grid: Mapping[Union[int, str], List[float]],
    entry_thresholds: List[float],
    exit_thresholds: List[float],
    use_shorts: bool,
) -> Tuple[Callable, Dict[str, List[Any]]]:
    """
    Convert weight_grid shorthand into a strategy_fn + param_grid pair.
    """
    # Build param_grid from weight_grid + thresholds
    param_grid: Dict[str, List[Any]] = {}
    feature_keys = []
    for fid, values in weight_grid.items():
        key = f"w_{fid}"
        param_grid[key] = values if isinstance(values, list) else [values]
        feature_keys.append((key, fid))

    param_grid["entry_th"] = entry_thresholds
    param_grid["exit_th"] = exit_thresholds

    # Capture use_shorts in closure
    _use_shorts = use_shorts
    _feature_keys = feature_keys

    def strategy_fn(**params: Any) -> WeightedStrategy:
        weights = {fid: params[key] for key, fid in _feature_keys}
        return WeightedStrategy(
            weights,
            entry_th=params["entry_th"],
            exit_th=params["exit_th"],
            use_shorts=_use_shorts,
        )

    return strategy_fn, param_grid


def param_sweep(
    sim: CobwebSim,
    data: Any,
    strategy_fn: Optional[Callable[..., Callable[[Rows], List[float]]]] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    *,
    # Shorthand for WeightedStrategy (ignored if strategy_fn is provided)
    weight_grid: Optional[Mapping[Union[int, str], List[float]]] = None,
    entry_thresholds: Optional[List[float]] = None,
    exit_thresholds: Optional[List[float]] = None,
    use_shorts: bool = False,
    # Common options
    config: Optional[Union[Dict, BacktestConfig]] = None,
    benchmark: Any = None,
    feature_ids: Optional[List[int]] = None,
    rank_by: str = "sharpe",
    max_workers: int = 2,
    on_error: str = "warn",
    progress: bool = True,
) -> ParamSweepResult:
    """
    Grid-search strategy parameters on a single ticker.

    Accepts either ``strategy_fn`` + ``param_grid`` for custom strategies,
    or ``weight_grid`` + threshold lists for the common weights-based case.

    Args:
        sim:                Connected :class:`CobwebSim` client.
        data:               CSV path, DataFrame, list of dicts, or
                            ``{"rows": [...]}``.
        strategy_fn:        Factory function: ``(**params) -> strategy callable``.
                            Takes precedence over *weight_grid*.
        param_grid:         ``{"param_name": [val1, val2, ...]}``.
        weight_grid:        Shorthand -- ``{feature_id: [w1, w2, ...]}``.
        entry_thresholds:   Grid of entry thresholds (used with *weight_grid*).
        exit_thresholds:    Grid of exit thresholds (used with *weight_grid*).
        use_shorts:         Enable shorts (used with *weight_grid*).
        config:             Shared :class:`BacktestConfig` for all combos.
        benchmark:          Benchmark data for alpha/beta metrics.
        feature_ids:        Feature IDs to enrich (default: all 71).
        rank_by:            Metric to sort results by (default ``"sharpe"``).
        max_workers:        Thread pool size for parallel backtest calls.
        on_error:           ``"warn"`` (default), ``"raise"``, or ``"skip"``.
        progress:           Print progress to stdout.

    Returns:
        :class:`ParamSweepResult` sorted by *rank_by*.

    Example::

        # Shorthand (weights-based):
        result = param_sweep(sim, "data.csv",
                             weight_grid={36: [0.2, 0.3], 11: [0.3, 0.5]})

        # Custom strategy factory:
        result = param_sweep(sim, "data.csv",
                             strategy_fn=make_my_strategy,
                             param_grid={"threshold": [0.1, 0.2, 0.3]})
    """
    # Resolve strategy_fn + param_grid
    if strategy_fn is not None and param_grid is not None:
        _strategy_fn = strategy_fn
        _param_grid = param_grid
    elif weight_grid is not None:
        _entry_ths = entry_thresholds if entry_thresholds is not None else [0.15, 0.20, 0.25]
        _exit_ths = exit_thresholds if exit_thresholds is not None else [0.03, 0.05, 0.10]
        _strategy_fn, _param_grid = _build_weighted_combos(
            weight_grid, _entry_ths, _exit_ths, use_shorts,
        )
    else:
        raise ValueError("Provide either `strategy_fn` + `param_grid`, or `weight_grid`.")

    # Generate all parameter combos
    keys = list(_param_grid.keys())
    value_lists = [_param_grid[k] for k in keys]
    combos = [dict(zip(keys, vals)) for vals in product(*value_lists)]
    total = len(combos)

    if progress:
        print(f"[param_sweep] {total} combos = {total} backtest API calls")

    # Enrich data ONCE
    resp = sim.enrich(data, feature_ids=feature_ids)
    enriched = resp.get("rows", [])
    if not enriched:
        raise CobwebError("No rows returned from enrich().")

    # Prepare base data for backtest calls
    base_data = {"rows": enriched}

    t0 = time.time()
    results: List[ParamRow] = []
    errors: List[Tuple[Dict, str]] = []

    def _run_combo(combo: Dict[str, Any], idx: int) -> ParamRow:
        # Build strategy from params
        strat = _strategy_fn(**combo)

        # Run strategy (local math)
        signals = strat(enriched)

        # Backtest (API call)
        bt = sim.backtest(
            base_data,
            signals=signals,
            config=config,
            benchmark=benchmark,
            compute_features=False,
        )

        metrics = _extract_metrics(bt)
        return ParamRow(
            params=combo,
            metrics=bt.get("metrics", {}),
            sharpe=metrics["sharpe"],
            total_return=metrics["total_return"],
            max_dd=metrics["max_dd"],
            sortino=metrics["sortino"],
            trades=metrics["trades"],
            signals=signals,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_combo, combo, i + 1): (combo, i + 1)
            for i, combo in enumerate(combos)
        }
        for future in as_completed(futures):
            combo, idx = futures[future]
            try:
                row = future.result()
                if progress:
                    print(
                        f"[{idx}/{total}] "
                        + " ".join(f"{k}={v}" for k, v in combo.items())
                        + f" -> Sharpe={row.sharpe:.2f}"
                    )
                results.append(row)
            except Exception as e:
                err_msg = str(e)
                errors.append((combo, err_msg))
                if on_error == "raise":
                    raise
                elif on_error == "warn":
                    warnings.warn(f"param_sweep: combo {combo} failed: {err_msg}", stacklevel=2)
                    if progress:
                        print(f"[{idx}/{total}] ERROR - {err_msg}")

    # Sort by rank metric
    reverse = rank_by != "max_dd"  # max_dd is negative, lower is worse
    results.sort(
        key=lambda r: getattr(r, rank_by, r.metrics.get(rank_by, 0)),
        reverse=reverse,
    )

    elapsed = (time.time() - t0) * 1000
    return ParamSweepResult(results, elapsed, total, errors, enriched)
