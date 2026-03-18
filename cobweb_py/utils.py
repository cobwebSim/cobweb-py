# cobweb_py/utils.py
from __future__ import annotations

from html import escape as _esc
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Iterable
import warnings

import pandas as pd

from .client import CobwebSim, CobwebError
from .scoring import PLOTS, _resolve_plot_id


RowsDict = Dict[str, Any]  # expects {"rows": List[Dict[str, Any]]}


def save_table(
    rows: List[Dict[str, Any]],
    out_path: Union[str, Path],
    *,
    title: str = "Table",
    max_rows: int = 500,
) -> str:
    """
    Save a lightweight HTML table (standalone HTML document).

    This is intentionally simple (no pandas Styler dependency/version issues).
    For a fancier report-style table, prefer cobweb_py.plots.save_features_table.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    safe_title = _esc(title)
    df = pd.DataFrame(rows[:max_rows])
    html = df.to_html(index=False, border=0)
    out_path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>{safe_title}</title>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'></head>"
        f"<body style='font-family:sans-serif;padding:16px'>"
        f"<h2>{safe_title}</h2>{html}</body></html>",
        encoding="utf-8",
    )
    return str(out_path)


def fix_timestamps(
    rows: List[Dict[str, Any]],
    *,
    name: str = "ROWS",
    timestamp_col: str = "timestamp",
    dayfirst: bool = True,
    drop_bad: bool = True,
    sort: bool = True,
    out_format: str = "%Y-%m-%d %H:%M:%S",
) -> RowsDict:
    """
    Normalize timestamps client-side to an ISO-ish format so the API won't reject them.

    Handles day-first formats like: '14/09/2020 00:00'.
    Returns {"rows": [...]} (API-ready).
    """
    df = pd.DataFrame(rows).copy()
    if timestamp_col not in df.columns:
        raise CobwebError(f"{name}: missing '{timestamp_col}' column")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", dayfirst=dayfirst)

    bad = int(df[timestamp_col].isna().sum())
    if bad and drop_bad:
        df = df.dropna(subset=[timestamp_col]).copy()

    if sort:
        df = df.sort_values(timestamp_col).reset_index(drop=True)

    # Convert to string format the API parses reliably
    df[timestamp_col] = df[timestamp_col].dt.strftime(out_format)

    if bad:
        # keep as print to avoid forcing logging dependency
        print(f"{name}: dropped {bad} rows with unparseable timestamps" if drop_bad else f"{name}: {bad} unparseable timestamps")

    return {"rows": df.to_dict("records")}


def load_csv(
    csv_path: Union[str, Path],
    *,
    name: str = "ROWS",
    dayfirst: bool = True,
) -> RowsDict:
    """
    Read a CSV file and normalize timestamps into an API-ready {"rows": [...]} dict.

    Combines pd.read_csv + fix_timestamps in one call.

    Example:
        base_rows = load_csv("apple_data.csv", name="ASSET")
        benchmark_rows = load_csv("spy.csv", name="BENCHMARK")
    """
    df = pd.read_csv(csv_path)

    # Normalise OHLCV column names to lowercase to match the API schema
    # (OHLCRow expects open/high/low/close/volume; CSVs often use Open/High/etc.)
    _OHLCV = {"open", "high", "low", "close", "volume"}
    df = df.rename(columns={c: c.lower() for c in df.columns if c.lower() in _OHLCV})

    # Auto-detect the timestamp column regardless of what the CSV calls it.
    _TS_CANDIDATES = ("timestamp", "Timestamp", "time", "Time", "date", "Date", "datetime", "Datetime")
    ts_col = next((c for c in _TS_CANDIDATES if c in df.columns), None)
    if ts_col is None:
        # Case-insensitive fallback
        lower_map = {c.lower(): c for c in df.columns}
        ts_col = next((lower_map[k] for k in ("timestamp", "time", "date", "datetime") if k in lower_map), None)

    if ts_col is None:
        # No recognisable timestamp column -- return rows as-is
        return {"rows": df.to_dict("records")}

    # Normalise to "timestamp" so align() and the API always see the same key.
    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})

    return fix_timestamps(df.to_dict("records"), name=name, timestamp_col="timestamp", dayfirst=dayfirst)


def align(
    a_rows: RowsDict,
    b_rows: RowsDict,
    *,
    a_name: str = "A",
    b_name: str = "B",
    timestamp_col: str = "timestamp",
    out_format: str = "%Y-%m-%d %H:%M:%S",
) -> Tuple[RowsDict, RowsDict]:
    """
    Ensure two datasets share timestamps (useful for alpha/beta/benchmark plots).

    Input must be {"rows":[...]} and timestamps should already be parseable.
    Returns trimmed (A, B) as {"rows": [...]} each, aligned on common timestamps.

    Note:
        Timestamps are normalised to UTC internally so that resolution mismatches
        (e.g. yfinance us vs s) do not cause merge errors.  Non-UTC timezone-aware
        timestamps are converted to UTC; timezone-naive timestamps are assumed UTC.
    """
    a_df = pd.DataFrame(a_rows.get("rows", [])).copy()
    b_df = pd.DataFrame(b_rows.get("rows", [])).copy()

    if timestamp_col not in a_df.columns or timestamp_col not in b_df.columns:
        raise CobwebError(f"Missing '{timestamp_col}' in {a_name} or {b_name}")

    a_df[timestamp_col] = pd.to_datetime(a_df[timestamp_col], errors="coerce", utc=True)
    b_df[timestamp_col] = pd.to_datetime(b_df[timestamp_col], errors="coerce", utc=True)

    a_df = a_df.dropna(subset=[timestamp_col]).copy()
    b_df = b_df.dropna(subset=[timestamp_col]).copy()

    # Normalize resolution so set intersection works even when
    # sources have different precisions (e.g. yfinance us vs s)
    _NORM_DTYPE = "datetime64[ns, UTC]"
    a_df[timestamp_col] = a_df[timestamp_col].astype(_NORM_DTYPE)
    b_df[timestamp_col] = b_df[timestamp_col].astype(_NORM_DTYPE)

    common = set(a_df[timestamp_col]).intersection(set(b_df[timestamp_col]))
    if not common:
        raise CobwebError(
            f"{a_name} and {b_name} have no overlapping timestamps. "
            f"Use matching date ranges and bar size."
        )

    a_df = a_df[a_df[timestamp_col].isin(common)].sort_values(timestamp_col).reset_index(drop=True)
    b_df = b_df[b_df[timestamp_col].isin(common)].sort_values(timestamp_col).reset_index(drop=True)

    a_df[timestamp_col] = a_df[timestamp_col].dt.strftime(out_format)
    b_df[timestamp_col] = b_df[timestamp_col].dt.strftime(out_format)

    return {"rows": a_df.to_dict("records")}, {"rows": b_df.to_dict("records")}


def to_signals(
    scores: Sequence[float],
    entry_th: float,
    exit_th: float,
    use_shorts: bool = False,
) -> List[float]:
    """
    Convert continuous scores into position signals: 0/1 (or -1/0/1 if use_shorts=True).

    - Enter long if score >= entry_th
    - Exit long if score <= exit_th
    - If shorts enabled:
        enter short if score <= -entry_th
        exit short if score >= -exit_th

    NaN scores are treated as 0.0 (no signal change).
    """
    signals: List[float] = []
    pos = 0

    for s in scores:
        s = float(s)
        # NaN/inf guard: treat non-finite as neutral
        if not (s == s) or s == float("inf") or s == float("-inf"):
            signals.append(float(pos))
            continue

        if not use_shorts:
            if pos == 0 and s >= entry_th:
                pos = 1
            elif pos == 1 and s <= exit_th:
                pos = 0
        else:
            if pos == 0:
                if s >= entry_th:
                    pos = 1
                elif s <= -entry_th:
                    pos = -1
            elif pos == 1:
                if s <= exit_th:
                    pos = 0
            elif pos == -1:
                if s >= -exit_th:
                    pos = 0

        signals.append(float(pos))

    return signals


def get_plot(
    sim: CobwebSim,
    data_rows: RowsDict,
    bt: Dict[str, Any],
    plot_id: Union[int, str],
    benchmark_rows: Optional[RowsDict] = None,
    *,
    # behavior knobs:
    benchmark_required_for: Iterable[int] = (3, 8, 9),
    regime_plot_ids: Iterable[int] = (20, 21, 22),
    regime_feature_ids: Sequence[int] = (70, 71),
    corr_plot_id: int = 23,
    all_feature_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Wrapper around sim.plots() that:
      - accepts a plot ID (int) or a fuzzy plot name (str)
      - auto-skips benchmark-required plots if benchmark is missing
      - auto-computes required feature deps for regime plots and correlation heatmap

    plot_id examples:
        get_plot(sim, rows, bt, 5)              # by ID
        get_plot(sim, rows, bt, "sharpe")       # -> rolling_sharpe (ID 5)
        get_plot(sim, rows, bt, "drawdown")     # -> ambiguous, be more specific
        get_plot(sim, rows, bt, "drawdown_u")   # -> drawdown_underwater (ID 10)
        get_plot(sim, rows, bt, "equity_lin")   # -> equity_curve_linear (ID 1)

    Returns the same shape as sim.plots(), typically {"payloads": {...}}.
    """
    pid = _resolve_plot_id(plot_id) if isinstance(plot_id, str) else int(plot_id)

    # Benchmark-required plots
    if pid in set(int(x) for x in benchmark_required_for):
        if benchmark_rows is None:
            print(f"plot_id={pid}: skipped (benchmark not available)")
            return {"payloads": {}}
        return sim.plots(data_rows, backtest_result=bt, plot_ids=[pid], benchmark=benchmark_rows)

    # Regime plots need regime features computed in THIS /plots call
    if pid in set(int(x) for x in regime_plot_ids):
        return sim.plots(
            data_rows,
            backtest_result=bt,
            plot_ids=[pid],
            compute_features=True,
            feature_ids=list(regime_feature_ids),
        )

    # Correlation heatmap needs many features computed in THIS /plots call
    if pid == int(corr_plot_id):
        if not all_feature_ids:
            raise ValueError("get_plot: all_feature_ids must be provided for correlation heatmap plot 23")
        return sim.plots(
            data_rows,
            backtest_result=bt,
            plot_ids=[pid],
            compute_features=True,
            feature_ids=list(all_feature_ids),
        )

    return sim.plots(data_rows, backtest_result=bt, plot_ids=[pid])


def get_plot_df(
    sim: CobwebSim,
    data_rows: RowsDict,
    bt: Dict[str, Any],
    plot_id: Union[int, str],
    benchmark_rows: Optional[RowsDict] = None,
    *,
    all_feature_ids: Optional[Sequence[int]] = None,
) -> "pd.DataFrame":
    """
    Fetch a single plot by ID or fuzzy name and return its data as a DataFrame.

    Combines :func:`get_plot` and :func:`payload_to_df` into one call so the
    user never needs to know the internal payload key name.

    Parameters
    ----------
    sim : CobwebSim
        Connected client instance.
    data_rows : RowsDict
        Enriched OHLCV rows (list or ``{"rows": [...]}``).
    bt : dict
        Backtest result from ``sim.backtest()``.
    plot_id : int or str
        Plot ID (1-27) or fuzzy name (e.g. ``"sharpe"``, ``"drawdown"``).
    benchmark_rows : optional
        Benchmark data (needed for plots 3, 8, 9).
    all_feature_ids : optional
        Required for correlation heatmap (plot 23).

    Returns
    -------
    pd.DataFrame
        The plot data as a pandas DataFrame.
    """
    from .plots import payload_to_df

    pl = get_plot(
        sim, data_rows, bt, plot_id,
        benchmark_rows=benchmark_rows,
        all_feature_ids=all_feature_ids,
    )
    payloads = pl.get("payloads", {}) or {}
    if not payloads:
        import pandas as pd
        return pd.DataFrame()
    first_payload = next(iter(payloads.values()))
    return payload_to_df(first_payload)


def save_all_plots(
    sim: CobwebSim,
    data_rows: RowsDict,
    bt: Dict[str, Any],
    plot_ids: Iterable[int],
    out_dir: Union[str, Path],
    *,
    benchmark_rows: Optional[RowsDict] = None,
    all_feature_ids: Optional[Sequence[int]] = None,
) -> Dict[int, List[str]]:
    """
    Iterate over plot_ids, fetch each via get_plot(), and write HTML files.

    Skips plots that return no payloads or raise errors (prints a warning).
    Returns {plot_id: [html_file_paths]} for every plot that succeeded.

    Example:
        results = save_all_plots(
            sim, base_rows, bt, range(1, 28), out_dir="out/api_plots",
            benchmark_rows=benchmark_rows, all_feature_ids=list(range(1, 72)),
        )
    """
    from .plots import save_api_payloads_to_html

    out_dir = Path(out_dir)
    results: Dict[int, List[str]] = {}

    for pid in plot_ids:
        try:
            pl = get_plot(
                sim,
                data_rows,
                bt,
                pid,
                benchmark_rows=benchmark_rows,
                all_feature_ids=all_feature_ids,
            )
        except CobwebError as e:
            print(f"plot_id={pid}: API error -> {e}")
            continue
        except Exception as e:
            print(f"plot_id={pid}: unexpected error -> {e}")
            continue

        payloads = pl.get("payloads", {}) or {}
        if not payloads:
            print(f"plot_id={pid}: no payloads returned")
            continue

        html_files = save_api_payloads_to_html(
            payloads,
            out_dir=out_dir / f"plot_id_{pid}",
            title_prefix=f"plot_id {pid} - ",
        )
        print(f"plot_id={pid}: wrote {len(html_files)} html files")
        results[pid] = html_files

    return results


def signal_label(value: float) -> str:
    """
    Convert a numeric signal value to a human-readable label.

    Returns ``"BUY"`` for 1.0, ``"SELL"`` for -1.0, ``"HOLD"`` otherwise.

    Example::

        label = signal_label(signals[-1])  # "BUY"
    """
    if value == 1.0:
        return "BUY"
    elif value == -1.0:
        return "SELL"
    return "HOLD"


def format_metrics(bt: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract and format backtest metrics as a display-ready dict.

    Returns a dict with human-readable keys and pre-formatted values
    (e.g. percentages, dollar amounts). Suitable for passing to
    ``save_table()`` or ``pd.DataFrame()``.

    Example::

        cw.save_table([cw.format_metrics(bt)], "metrics.html", title="Metrics")
    """
    from .plots import _METRIC_LABELS, _fmt_metric

    m = bt.get("metrics", {})
    result: Dict[str, str] = {}
    for key, label in _METRIC_LABELS.items():
        if key in m:
            result[label] = _fmt_metric(key, m[key])
    return result


def get_signal(bt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the current trading signal from a backtest result.

    Returns a dict with three keys:
        signal   - "buy" | "sell" | "hold"
        exposure - float in [-1, 1], current portfolio exposure
        strength - float in [0, 1], conviction (0 = threshold, 1 = max)

    Example:
        s = get_signal(bt)
        if s["signal"] == "buy":
            print(f"Long {s['exposure']:.0%} with {s['strength']:.0%} conviction")
    """
    return {
        "signal":   bt.get("current_signal", "hold"),
        "exposure": float(bt.get("current_exposure", 0.0)),
        "strength": float(bt.get("signal_strength", 0.0)),
    }


def print_signal(bt: Dict[str, Any]) -> None:
    """
    Pretty-print the current trading signal from a backtest result.

    Example:
        print_signal(bt)
        # Signal: BUY  |  exposure=0.87  |  strength=0.72
    """
    s = get_signal(bt)
    print(
        f"Signal: {s['signal'].upper():<5}"
        f"  |  exposure={s['exposure']:+.2f}"
        f"  |  strength={s['strength']:.2f}"
    )
