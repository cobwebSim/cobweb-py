from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

try:
    import pandas as pd  # optional
except Exception:  # pragma: no cover
    pd = None  # type: ignore


class CobwebError(RuntimeError):
    """Raised when the API request fails or input data is invalid."""


def _is_pathlike(x: Any) -> bool:
    return isinstance(x, (str, Path))


def _read_csv_simple(csv_path: Union[str, Path], max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read a CSV file and convert to the API's OHLCRow list."""
    if pd is None:
        raise CobwebError("pandas is required to read CSV. Install with: pip install pandas")

    p = Path(csv_path)
    if not p.exists():
        raise CobwebError(f"CSV not found: {p}")

    df = pd.read_csv(p)
    if max_rows:
        df = df.head(max_rows).copy()

    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in lower_map:
                return lower_map[n.lower()]
        return None

    c_open = pick("open", "Open")
    c_high = pick("high", "High")
    c_low = pick("low", "Low")
    c_close = pick("close", "Close")
    c_vol = pick("volume", "Volume", "vol")
    c_ts = pick("timestamp", "time", "date", "datetime", "Date", "Datetime")

    missing = [n for n, c in [("open", c_open), ("high", c_high), ("low", c_low), ("close", c_close)] if c is None]
    if missing:
        raise CobwebError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    for c in [c_open, c_high, c_low, c_close]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if c_vol is not None:
        df[c_vol] = pd.to_numeric(df[c_vol], errors="coerce")

    df = df.dropna(subset=[c_open, c_high, c_low, c_close]).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        row: Dict[str, Any] = {
            "open": float(r[c_open]),
            "high": float(r[c_high]),
            "low": float(r[c_low]),
            "close": float(r[c_close]),
        }
        if c_vol is not None and r.get(c_vol) == r.get(c_vol):
            row["volume"] = float(r[c_vol])
        if c_ts is not None and r.get(c_ts) == r.get(c_ts):
            row["timestamp"] = str(r[c_ts])
        rows.append(row)

    if len(rows) < 20:
        raise CobwebError("Need at least 20 rows for meaningful indicators like SMA/RSI.")

    return rows


_OHLCV_FIELDS = {"open", "high", "low", "close", "volume"}


def _normalize_ohlcv_keys(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lowercase any OHLCV keys that came back capitalised from the API (e.g. Open → open)."""
    if not rows:
        return rows
    if not any(k for k in rows[0] if k.lower() in _OHLCV_FIELDS and k != k.lower()):
        return rows  # nothing to fix — fast path
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

    if pd is not None and hasattr(data, "__class__") and data.__class__.__name__ == "DataFrame":
        df = data.copy()
        if max_rows:
            df = df.head(max_rows).copy()

        df.columns = [str(c).strip() for c in df.columns]
        lower_map = {c.lower(): c for c in df.columns}

        def pick(*names: str) -> Optional[str]:
            for n in names:
                if n.lower() in lower_map:
                    return lower_map[n.lower()]
            return None

        c_open = pick("open", "Open")
        c_high = pick("high", "High")
        c_low = pick("low", "Low")
        c_close = pick("close", "Close")
        c_vol = pick("volume", "Volume", "vol")
        c_ts = pick("timestamp", "time", "date", "datetime", "Date", "Datetime")

        missing = [n for n, c in [("open", c_open), ("high", c_high), ("low", c_low), ("close", c_close)] if c is None]
        if missing:
            raise CobwebError(f"DataFrame missing required columns: {missing}. Found: {list(df.columns)}")

        for c in [c_open, c_high, c_low, c_close]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if c_vol is not None:
            df[c_vol] = pd.to_numeric(df[c_vol], errors="coerce")

        df = df.dropna(subset=[c_open, c_high, c_low, c_close]).reset_index(drop=True)

        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            row: Dict[str, Any] = {
                "open": float(r[c_open]),
                "high": float(r[c_high]),
                "low": float(r[c_low]),
                "close": float(r[c_close]),
            }
            if c_vol is not None and r.get(c_vol) == r.get(c_vol):
                row["volume"] = float(r[c_vol])
            if c_ts is not None and r.get(c_ts) == r.get(c_ts):
                row["timestamp"] = str(r[c_ts])
            rows.append(row)

        if len(rows) < 20:
            raise CobwebError("Need at least 20 rows for meaningful indicators like SMA/RSI.")
        return rows

    raise CobwebError(
        "Unsupported data input. Use a CSV path, a pandas DataFrame, "
        "a list of OHLC row dicts, or a dict with {'rows': [...]}."
    )


@dataclass
class BacktestConfig:
    """
    Configuration for a backtest run.

    Pass to sim.backtest(config=...) — all fields are optional, defaults match the API.

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
        import dataclasses
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

    def features(self, data: Any, feature_ids: Optional[List[int]] = None, *, max_rows: Optional[int] = None) -> dict:
        """Alias for enrich()."""
        return self.enrich(data, feature_ids=feature_ids, max_rows=max_rows)

    # ✅ FIX 1: backtest is a proper CobwebSim method (indented under the class)
    # ✅ FIX 2: use `.json` (field) not `.json()` (method)
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

    # ✅ FIX 3: plots is its own method (not nested inside backtest)
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
