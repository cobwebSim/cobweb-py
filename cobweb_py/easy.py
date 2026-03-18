from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union, Sequence
import warnings

from .client import CobwebSim, CobwebError, BacktestConfig
from .scoring import score, auto_score, score_by_id
from .plots import (
    save_equity_plot, save_score_plot, save_price_and_score_plot,
    save_features_table, payloads_to_figures,
)
from .utils import to_signals, get_signal, get_plot, load_csv, align, format_metrics, print_signal

import pandas as pd


def quickstart(
    base_url: str,
    csv_path: Union[str, Path],
    *,
    api_key: Optional[str] = None,
    feature_ids: Optional[List[int]] = None,
    weights: Optional[Mapping[Union[str, int], float]] = None,
    out_dir: Union[str, Path] = ".",
    signals: Optional[Union[List[float], str]] = "long",
    config: Optional[Union[Dict, BacktestConfig]] = None,
    plot_ids: Optional[List[int]] = None,
    normalize_dayfirst_timestamps: bool = False,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    One-call experience for beginners.

    Outputs:
      - features_table.html + features_table.csv (OHLCV pinned first)
      - score.html
      - price_and_score.html
      - equity.html
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure feature dependencies for plot requests (e.g., regime plots)
    feature_ids = _ensure_feature_deps(feature_ids, plot_ids)

    sim = CobwebSim(base_url, api_key=api_key)

    # Optionally normalize day-first timestamps (client-side) before hitting API.
    # Uses load_csv which auto-detects timestamp columns (Date, Datetime, time, etc.)
    # instead of requiring a literal "timestamp" column.
    data_for_api: Union[str, Path, Dict[str, Any]] = csv_path
    if normalize_dayfirst_timestamps:
        data_for_api = load_csv(csv_path, name="ASSET", dayfirst=True)
        if max_rows is not None:
            rows_list = data_for_api.get("rows", [])
            data_for_api = {"rows": rows_list[:int(max_rows)]}

    feats = sim.enrich(data_for_api, feature_ids=feature_ids, max_rows=max_rows)

    rows = feats.get("rows", [])
    if not rows:
        raise CobwebError("No rows returned from enrich().")

    # --- tables: html + csv ---
    features_html = out_dir / "features_table.html"
    features_csv = out_dir / "features_table.csv"
    save_features_table(
        rows,
        out_html=features_html,
        out_csv=features_csv,
        title="Features Table (OHLCV first)",
        max_rows=None,
    )

    # --- scoring ---
    if weights is None:
        scores = auto_score(rows)
    else:
        # If any key looks like an int, assume weights are feature IDs and use score_by_id
        use_ids = any(isinstance(k, int) or (isinstance(k, str) and k.strip().isdigit()) for k in weights.keys())
        scores = score_by_id(rows, weights) if use_ids else score(rows, weights)  # type: ignore[arg-type]

    # --- plots: score + price/score ---
    score_path = out_dir / "score.html"
    price_score_path = out_dir / "price_and_score.html"
    equity_path = out_dir / "equity.html"

    save_score_plot(rows, scores, out_html=score_path)
    save_price_and_score_plot(rows, scores, out_html=price_score_path)

    # --- backtest ---
    # Use same input used for features to avoid timestamp/row mismatch
    bt = sim.backtest(data_for_api, compute_features=False, signals=signals, config=config, max_rows=max_rows, plot_ids=plot_ids)
    save_equity_plot(bt, out_html=equity_path)

    return {
        "features_response": feats,
        "backtest_response": bt,
        "scores_preview": scores[:10],
        "plots": {
            "equity": str(equity_path),
            "score": str(score_path),
            "price_and_score": str(price_score_path),
        },
        "tables": {
            "features_html": str(features_html),
            "features_csv": str(features_csv),
        },
    }


def _ensure_feature_deps(
    feature_ids: Optional[Sequence[int]],
    plot_ids: Optional[Sequence[int]],
) -> Optional[List[int]]:
    """
    Auto-add required feature IDs for certain plot IDs.

    - plot 21 (performance_by_trend_regime) needs trend_regime (70)
    - plot 20 (performance_by_volatility_regime) needs vol_regime (71)

    Returns None when both inputs are None (preserves "compute all" default).
    """
    plot_ids_set = set(int(x) for x in (plot_ids or []))

    # If no feature_ids and no plot deps needed, return None (= compute all)
    if feature_ids is None and not (21 in plot_ids_set or 20 in plot_ids_set):
        return None

    feature_ids_set = set(int(x) for x in (feature_ids or []))

    if 21 in plot_ids_set:  # performance_by_trend_regime
        feature_ids_set.add(70)  # trend_regime
    if 20 in plot_ids_set:  # performance_by_volatility_regime
        feature_ids_set.add(71)  # vol_regime

    return sorted(feature_ids_set) if feature_ids_set else None


# ---------------------------------------------------------------------------
# Pipeline -- reusable enrich-once, run-many workflow
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Reusable pipeline: enrich once, re-run scoring/backtest/plots many times.

    The expensive API call (enrich) is cached after the first ``.enrich()`` or
    ``.run()`` call.  Scoring, signal generation, backtest, and plot generation
    rerun on every ``.run()`` call, which is cheap by comparison.

    Example::

        pipe = Pipeline("http://localhost:8000", "stock.csv")
        result = pipe.run(weights={36: 0.3, 11: 0.3, 1: 0.4})

        # Change parameters and re-run (no re-enrichment):
        result2 = pipe.run(weights={36: 0.5, 1: 0.5}, entry_th=0.30)

        # Access in-memory Plotly figures:
        for name, fig in result["figures"].items():
            fig.show()

    Args:
        base_url:    URL of the CobwebSim API (e.g. ``"http://localhost:8000"``).
        data:        CSV path, pandas DataFrame, ``list[dict]``, or
                     ``{"rows": [...]}``.
        benchmark:   Optional benchmark data (same formats as *data*).
        feature_ids: Feature IDs to compute.  Defaults to all 71.
        api_key:     Optional API key.
        timeout:     HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        data: Any,
        *,
        benchmark: Any = None,
        feature_ids: Optional[List[int]] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        self._sim = CobwebSim(base_url, api_key=api_key, timeout=timeout)
        self._raw_data = data
        self._raw_benchmark = benchmark
        self._feature_ids = feature_ids or list(range(1, 72))

        # Enrichment cache (populated by .enrich())
        self._enriched_rows: Optional[List[Dict[str, Any]]] = None
        self._base_rows: Optional[Dict[str, Any]] = None
        self._benchmark_rows: Optional[Dict[str, Any]] = None
        self._feature_columns: Optional[List[str]] = None
        self._enrich_response: Optional[Dict[str, Any]] = None

    @property
    def is_enriched(self) -> bool:
        """True if ``enrich()`` has been called and cached."""
        return self._enriched_rows is not None

    @property
    def enriched_rows(self) -> Optional[List[Dict[str, Any]]]:
        """The cached enriched rows, or ``None`` if not yet enriched."""
        return self._enriched_rows

    @property
    def feature_columns(self) -> Optional[List[str]]:
        """List of feature column names from the last enrichment."""
        return self._feature_columns

    def enrich(self) -> "Pipeline":
        """
        Call the API to enrich data with features.  Caches the result.

        Subsequent calls are no-ops unless ``.reset()`` is called first.
        Returns *self* for chaining: ``pipe.enrich().run(...)``.
        """
        if self.is_enriched:
            return self

        # --- load data ---
        data = self._raw_data
        if isinstance(data, (str, Path)):
            data = load_csv(str(data), name="ASSET")
        if isinstance(data, list):
            data = {"rows": data}
        self._base_rows = data

        # --- load and align benchmark ---
        if self._raw_benchmark is not None:
            bench = self._raw_benchmark
            if isinstance(bench, (str, Path)):
                bench = load_csv(str(bench), name="BENCHMARK")
            if isinstance(bench, list):
                bench = {"rows": bench}
            self._benchmark_rows = bench

            if self._base_rows and self._benchmark_rows:
                self._base_rows, self._benchmark_rows = align(
                    self._base_rows, self._benchmark_rows,
                    a_name="ASSET", b_name="BENCHMARK",
                )

        # --- enrich via API ---
        resp = self._sim.enrich(self._base_rows, feature_ids=self._feature_ids)
        rows = resp.get("rows", [])
        if not rows:
            raise CobwebError("No rows returned from enrich().")

        self._enriched_rows = rows
        self._base_rows = {"rows": rows}
        self._feature_columns = resp.get("feature_columns", [])
        self._enrich_response = resp

        return self

    def reset(self) -> "Pipeline":
        """Clear the enrichment cache.  The next ``.run()`` will re-enrich."""
        self._enriched_rows = None
        self._base_rows = None
        self._benchmark_rows = None
        self._feature_columns = None
        self._enrich_response = None
        return self

    def run(
        self,
        *,
        weights: Optional[Mapping[Union[str, int], float]] = None,
        entry_th: float = 0.20,
        exit_th: float = 0.05,
        use_shorts: bool = False,
        config: Optional[Union[Dict, BacktestConfig]] = None,
        plot_ids: Optional[List[int]] = None,
        layout_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline: score -> signals -> backtest -> plots.

        Auto-enriches if not yet done.  The enrichment is cached and reused;
        everything else reruns with each call.

        Args:
            weights:          Feature weights for scoring.  Keys can be feature
                              IDs (``int``) or column names (``str``).  If
                              ``None``, uses ``auto_score()``.
            entry_th:         Score threshold to enter a position.
            exit_th:          Score threshold to exit a position.
            use_shorts:       Allow short positions in signal generation.
            config:           ``BacktestConfig`` or dict of backtest parameters.
            plot_ids:         List of plot IDs to fetch (default: all 27).
                              These are fetched individually via ``get_plot()``,
                              not passed to the backtest call.
            layout_overrides: Dict passed to each Plotly figure's
                              ``update_layout()``.

        Returns:
            Dict with keys: ``metrics``, ``signal``, ``trades``, ``scores``,
            ``signals``, ``equity_curve``, ``backtest``, ``figures``.
        """
        # Auto-enrich if needed
        if not self.is_enriched:
            self.enrich()

        rows = self._enriched_rows
        base_rows = self._base_rows

        # --- Scoring ---
        if weights is None:
            scores = auto_score(rows)
        else:
            use_ids = any(
                isinstance(k, int) or (isinstance(k, str) and k.strip().isdigit())
                for k in weights.keys()
            )
            scores = score_by_id(rows, weights) if use_ids else score(rows, weights)  # type: ignore[arg-type]

        # --- Signal generation ---
        signals = to_signals(scores, entry_th, exit_th, use_shorts)

        # --- Backtest ---
        effective_plot_ids = plot_ids if plot_ids is not None else list(range(1, 28))

        bt = self._sim.backtest(
            base_rows,
            signals=signals,
            compute_features=True,
            feature_ids=[70, 71],
            benchmark=self._benchmark_rows,
            config=config,
        )

        # --- Signal info ---
        signal_info = get_signal(bt)

        # --- Plots (fetched individually via API) ---
        figures: Dict[str, Any] = {}
        for pid in effective_plot_ids:
            try:
                pl = get_plot(
                    self._sim, base_rows, bt, pid,
                    benchmark_rows=self._benchmark_rows,
                    all_feature_ids=list(range(1, 72)),
                )
            except Exception as e:
                warnings.warn(
                    f"Pipeline.run(): plot_id={pid} failed: {e}",
                    stacklevel=2,
                )
                continue
            payloads = pl.get("payloads", {}) or {}
            figs = payloads_to_figures(payloads, layout_overrides=layout_overrides)
            figures.update(figs)

        return DeployableResult(
            metrics=bt.get("metrics", {}),
            signal=signal_info,
            trades=bt.get("trades", []),
            scores=scores,
            signals=signals,
            equity_curve=bt.get("equity_curve", []),
            backtest=bt,
            figures=figures,
        )


# ---------------------------------------------------------------------------
# DeployableResult -- dict-like result with deploy(), print_metrics(), etc.
# ---------------------------------------------------------------------------

class DeployableResult(dict):
    """
    Result from ``Pipeline.run()`` with convenience methods for deployment.

    Behaves like a regular dict (backward-compatible) but adds:

    - ``.print_metrics()`` -- pretty-print backtest metrics
    - ``.print_signal()``  -- pretty-print the current trading signal
    - ``.deploy(broker)``  -- deploy the signal to a live/paper broker

    Example::

        result = pipe.run(weights={11: 0.3, 36: 0.3, 14: 0.4})
        result.print_metrics()
        result.print_signal()
        result.deploy(broker, symbol="AAPL", dry_run=True)
    """

    def __init__(
        self,
        metrics: Dict,
        signal: Dict,
        trades: list,
        scores: list,
        signals: list,
        equity_curve: list,
        backtest: Dict,
        figures: Dict,
    ):
        super().__init__(
            metrics=metrics,
            signal=signal,
            trades=trades,
            scores=scores,
            signals=signals,
            equity_curve=equity_curve,
            backtest=backtest,
            figures=figures,
        )

    def print_metrics(self) -> None:
        """Pretty-print backtest performance metrics."""
        bt = self["backtest"]
        formatted = format_metrics(bt)
        print()
        print("=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        for key, val in formatted.items():
            print(f"  {key}: {val}")
        print()

    def print_signal(self) -> None:
        """Pretty-print the current trading signal."""
        print()
        print("=" * 50)
        print("CURRENT SIGNAL")
        print("=" * 50)
        bt = self["backtest"]
        print_signal(bt)
        print()

    def deploy(
        self,
        broker: Any,
        symbol: str,
        *,
        price: Optional[float] = None,
        sizer: Any = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Deploy the current signal to a broker.

        Args:
            broker:   A connected broker instance (e.g. ``cw.AlpacaBroker(paper=True)``).
            symbol:   Ticker symbol to trade (e.g. "AAPL").
            price:    Override price for sizing. If None, fetched from broker.
            sizer:    Position sizing strategy (e.g. ``cw.PercentOfEquity(0.95)``).
                      Defaults to ``FullCash()``.
            dry_run:  If True, log what *would* happen without placing orders.
            force:    If True, skip duplicate signal check.

        Returns:
            Dict with keys: action, signal, symbol, qty, order.
        """
        from .execution import deploy as _deploy

        signal_info = self["signal"]

        account = broker.get_account_info()
        print()
        print("=" * 50)
        print("DEPLOYING")
        print("=" * 50)
        print(f"  Account status: {account['status']}")
        print(f"  Buying power:   ${account['buying_power']:,.2f}")
        print(f"  Portfolio value: ${account['portfolio_value']:,.2f}")
        current_qty = broker.get_position(symbol)
        if current_qty > 0:
            print(f"  Current {symbol}:   {current_qty} shares")
        else:
            print(f"  Current {symbol}:   No position")
        print()

        return _deploy(
            signal_info,
            broker,
            symbol,
            price=price,
            sizer=sizer,
            dry_run=dry_run,
            force=force,
        )
