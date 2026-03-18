from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
import pandas as pd


class CobwebError(RuntimeError):
    """Raised when the API request fails or input data is invalid."""


def from_alpaca(
    ticker: str,
    days: int = 252 * 5,
    *,
    broker: Any = None,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    paper: bool = True,
) -> pd.DataFrame:
    """
    Download historical daily bars from Alpaca's data API.

    Uses an existing ``AlpacaBroker`` instance or creates a connection from
    credentials.  Falls back to environment variables if no keys are given.

    Args:
        ticker:     Symbol to download (e.g. "AAPL").
        days:       Number of calendar days of history (default ~5 years).
        broker:     Optional ``AlpacaBroker`` instance (reuses its connection).
        api_key:    Alpaca API key ID (or set ``ALPACA_API_KEY``).
        secret_key: Alpaca secret key (or set ``ALPACA_SECRET_KEY``).
        paper:      Use paper trading endpoint (default True).

    Returns:
        pandas DataFrame with Date, Open, High, Low, Close, Volume columns.

    Example::

        df = cw.from_alpaca("AAPL", days=252*5)
        rows = sim.enrich_rows(df, feature_ids=[12, 36])
    """
    import os
    from datetime import datetime, timedelta

    try:
        import alpaca_trade_api as tradeapi  # type: ignore
    except ImportError:
        raise CobwebError(
            "alpaca-trade-api is required for from_alpaca(). "
            "Install with: pip install alpaca-trade-api"
        )

    # Reuse broker's API connection if provided
    if broker is not None and hasattr(broker, "_api"):
        api = broker._api
    else:
        key = api_key or os.environ.get("ALPACA_API_KEY")
        secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise CobwebError(
                "Alpaca API keys not found. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables, or pass api_key/secret_key."
            )
        base_url = (
            "https://paper-api.alpaca.markets"
            if paper
            else "https://api.alpaca.markets"
        )
        api = tradeapi.REST(key, secret, base_url, api_version="v2")

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    bars = api.get_bars(
        ticker, "1Day", start=start_str, end=end_str, limit=10000
    )

    if not bars:
        raise CobwebError(f"Alpaca returned no data for {ticker}")

    records = []
    for bar in bars:
        records.append({
            "Date": bar.t.strftime("%Y-%m-%d"),
            "Open": float(bar.o),
            "High": float(bar.h),
            "Low": float(bar.l),
            "Close": float(bar.c),
            "Volume": int(bar.v),
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise CobwebError(f"Alpaca returned no data for {ticker} ({start_str} to {end_str})")

    return df


def from_yfinance(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance and return a clean DataFrame.

    Handles the yfinance MultiIndex quirk and resets the index so the
    resulting DataFrame is ready to pass straight into ``sim.enrich()``
    or ``sim.backtest()``.

    Requires the ``yfinance`` package (``pip install yfinance``).

    Example::

        df = cw.from_yfinance("AAPL", "2020-01-01", "2024-12-31")
        rows = sim.enrich_rows(df, feature_ids=[12, 36])
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        raise CobwebError(
            "yfinance is required for from_yfinance(). "
            "Install with: pip install yfinance"
        )

    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise CobwebError(f"yfinance returned no data for {ticker} ({start} to {end})")
    # Flatten MultiIndex columns (yfinance quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.reset_index()


def _is_pathlike(x: Any) -> bool:
    return isinstance(x, (str, Path))


def _pick_col(lower_map: Dict[str, str], *names: str) -> Optional[str]:
    """Case-insensitive column lookup: returns the original column name or None."""
    for n in names:
        if n.lower() in lower_map:
            return lower_map[n.lower()]
    return None


def _df_to_ohlc_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame with OHLCV (+timestamp) columns into API-ready row dicts.

    Shared by ``_read_csv_simple()`` and the DataFrame branch of ``_to_rows()``.
    Uses ``df.to_dict("records")`` instead of ``iterrows()`` for performance.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    c_open = _pick_col(lower_map, "open")
    c_high = _pick_col(lower_map, "high")
    c_low = _pick_col(lower_map, "low")
    c_close = _pick_col(lower_map, "close")
    c_vol = _pick_col(lower_map, "volume", "vol")
    c_ts = _pick_col(lower_map, "timestamp", "time", "date", "datetime")

    missing = [
        n for n, c in [("open", c_open), ("high", c_high), ("low", c_low), ("close", c_close)]
        if c is None
    ]
    if missing:
        raise CobwebError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    for c in [c_open, c_high, c_low, c_close]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if c_vol is not None:
        df[c_vol] = pd.to_numeric(df[c_vol], errors="coerce")

    df = df.dropna(subset=[c_open, c_high, c_low, c_close]).reset_index(drop=True)

    # Vectorised conversion (avoids slow iterrows)
    records = df.to_dict("records")
    rows: List[Dict[str, Any]] = []
    for r in records:
        row: Dict[str, Any] = {
            "open": float(r[c_open]),
            "high": float(r[c_high]),
            "low": float(r[c_low]),
            "close": float(r[c_close]),
        }
        if c_vol is not None and pd.notna(r.get(c_vol)):
            row["volume"] = float(r[c_vol])
        if c_ts is not None and pd.notna(r.get(c_ts)):
            row["timestamp"] = str(r[c_ts])
        rows.append(row)

    if len(rows) < 20:
        raise CobwebError("Need at least 20 rows for meaningful indicators like SMA/RSI.")

    return rows


def _read_csv_simple(csv_path: Union[str, Path], max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read a CSV file and convert to the API's OHLCRow list."""
    p = Path(csv_path)
    if not p.exists():
        raise CobwebError(f"CSV not found: {p}")

    df = pd.read_csv(p)
    if max_rows:
        df = df.head(max_rows)
    return _df_to_ohlc_rows(df)


_OHLCV_FIELDS = {"open", "high", "low", "close", "volume"}


def _normalize_ohlcv_keys(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lowercase any OHLCV keys that came back capitalised from the API (e.g. Open -> open)."""
    if not rows:
        return rows
    if not any(k for k in rows[0] if k.lower() in _OHLCV_FIELDS and k != k.lower()):
        return rows  # nothing to fix -- fast path
    return [
        {(k.lower() if k.lower() in _OHLCV_FIELDS else k): v for k, v in row.items()}
        for row in rows
    ]


def _to_rows(data: Any, *, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Accepts: CSV path, pandas DataFrame, list[dict] rows, or dict {'rows':[...]}"""
    if _is_pathlike(data):
        return _read_csv_simple(data, max_rows=max_rows)

    if isinstance(data, dict) and "rows" in data and isinstance(data["rows"], list):
        return _normalize_ohlcv_keys(list(data["rows"]))

    if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], dict)):
        return _normalize_ohlcv_keys(data)

    if isinstance(data, pd.DataFrame):
        df = data
        if max_rows:
            df = df.head(max_rows)
        return _df_to_ohlc_rows(df)

    raise CobwebError(
        "Unsupported data input. Use a CSV path, a pandas DataFrame, "
        "a list of OHLC row dicts, or a dict with {'rows': [...]}."
    )


@dataclass
class BacktestConfig:
    """
    Configuration for a backtest run.

    Pass to sim.backtest(config=...) -- all fields are optional, defaults match the API.

    Example:
        cfg = BacktestConfig(
            initial_cash=50_000,
            exec_horizon="longterm",
            fee_bps=0.5,
            rebalance_mode="calendar",
            rebalance_every_n_bars=5,
        )
        bt = sim.backtest(data, signals=signals, config=cfg)
    """
    # --- Capital ---
    initial_cash: float = 10_000.0
    max_leverage: float = 1.0
    allow_margin: bool = False

    # --- Execution ---
    exec_horizon: str = "swing"         # "intraday" | "swing" | "longterm"
    asset_type: str = "equities"        # "equities" | "crypto"
    max_participation: Optional[float] = None  # max fraction of bar volume per trade

    # --- Costs ---
    fee_bps: float = 1.0
    half_spread_bps: float = 2.0
    base_slippage_bps: float = 1.0
    impact_coeff: float = 1.0

    # --- Rebalancing ---
    rebalance_mode: str = "on_signal_change"  # "on_signal_change" | "every_bar" | "calendar"
    rebalance_every_n_bars: int = 0           # only used when rebalance_mode="calendar"
    min_trade_units: float = 0.0
    max_order_age_bars: int = 0

    # --- Signal thresholds ---
    buy_threshold: float = 0.10
    sell_threshold: float = -0.10

    # --- Analytics ---
    risk_free_annual: float = 0.0
    periods_per_year: int = 252

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict for passing as config= to sim.backtest()."""
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


@dataclass
class APIResponse:
    status_code: int
    json: Any
    elapsed_ms: int


class CobwebSim:
    """Friendly wrapper around the CobwebSim API."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _call(self, path: str, payload: Optional[dict] = None, method: str = "POST") -> APIResponse:
        url = f"{self.base_url}{path}"
        t0 = time.time()
        try:
            r = self._session.request(
                method.upper(),
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise CobwebError(f"Request failed: {e}") from e
        elapsed_ms = int((time.time() - t0) * 1000)

        try:
            data = r.json()
        except Exception:
            data = {"text": (r.text or "")[:4000]}

        if r.status_code >= 400:
            raise CobwebError(f"{method} {path} -> HTTP {r.status_code}: {data}")

        return APIResponse(status_code=r.status_code, json=data, elapsed_ms=elapsed_ms)

    def health(self) -> dict:
        return self._call("/health", payload=None, method="GET").json

    def _prepare(self, data: Any, *, max_rows: Optional[int] = None) -> dict:
        return {"rows": _to_rows(data, max_rows=max_rows)}

    def enrich(
        self, data: Any, feature_ids: Optional[List[int]] = None, *, max_rows: Optional[int] = None
    ) -> dict:
        """Return OHLCV rows enriched with computed feature columns."""
        payload = {"data": self._prepare(data, max_rows=max_rows), "feature_ids": feature_ids}
        return self._call("/features", payload).json

    def enrich_rows(
        self, data: Any, feature_ids: Optional[List[int]] = None, *, max_rows: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return just the enriched rows list (shorthand for ``enrich(...)["rows"]``)."""
        return self.enrich(data, feature_ids=feature_ids, max_rows=max_rows).get("rows", [])

    def features(self, data: Any, feature_ids: Optional[List[int]] = None, *, max_rows: Optional[int] = None) -> dict:
        """Alias for enrich()."""
        return self.enrich(data, feature_ids=feature_ids, max_rows=max_rows)

    def backtest(
        self,
        data: Any,
        *,
        signals: Union[List[float], str],
        compute_features: bool = True,
        feature_ids: Optional[List[int]] = None,
        plot_ids: Optional[List[int]] = None,
        plot_params: Optional[Dict[str, Any]] = None,
        benchmark: Optional[Any] = None,
        config: Optional[Union[Dict[str, Any], "BacktestConfig"]] = None,
        max_rows: Optional[int] = None,
    ) -> dict:
        rows = _to_rows(data, max_rows=max_rows)
        n = len(rows)

        if isinstance(signals, str):
            s = signals.strip().lower()
            if s in ("long", "buy"):
                sigs = [1.0] * n
            elif s in ("short", "sell"):
                sigs = [-1.0] * n
            elif s in ("flat", "none", "hold"):
                sigs = [0.0] * n
            else:
                raise CobwebError("signals string must be one of: long/short/flat")
        elif isinstance(signals, list):
            sigs = signals
            if len(sigs) != n:
                raise CobwebError(f"signals length must equal number of rows ({n}); got {len(sigs)}")
        else:
            raise CobwebError("signals must be a list[float] or one of: 'long'/'short'/'flat'")

        config_dict = config.to_dict() if isinstance(config, BacktestConfig) else (config or {})

        payload = {
            "data": {"rows": rows},
            "compute_features": compute_features,
            "feature_ids": feature_ids,
            "plot_ids": plot_ids,
            "plot_params": plot_params,
            "benchmark": self._prepare(benchmark, max_rows=max_rows) if benchmark is not None else None,
            "signals": sigs,
            "config": config_dict,
        }
        return self._call("/backtest", payload).json

    def plots(
        self,
        data: Any,
        backtest_result: dict,
        *,
        compute_features: bool = False,
        plot_ids: Optional[List[int]] = None,
        requested: Optional[List[str]] = None,
        plot_params: Optional[Dict[str, Any]] = None,
        benchmark: Optional[Any] = None,
        feature_ids: Optional[List[int]] = None,
        max_rows: Optional[int] = None,
    ) -> dict:
        payload = {
            "data": self._prepare(data, max_rows=max_rows),
            "compute_features": compute_features,
            "backtest_result": backtest_result,
            "feature_ids": feature_ids,
            "plot_ids": plot_ids,
            "plot_params": plot_params,
            "benchmark": self._prepare(benchmark, max_rows=max_rows) if benchmark is not None else None,
            "requested": requested,
        }
        return self._call("/plots", payload).json
