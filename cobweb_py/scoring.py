from __future__ import annotations

from typing import Any, List, Mapping, Dict, Optional, Union, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# Feature ID -> column name mapping
FEATURES: Dict[int, str] = {
    1: "ret_1d",
    2: "logret_1d",
    3: "ret_5d",
    4: "ret_20d",
    5: "ret_60d",
    6: "ret_252d",
    7: "ret_accel_5_20",
    8: "ret_accel_20_60",
    9: "mom_pct_20d_252w",
    10: "sma_10",
    11: "sma_20",
    12: "sma_50",
    13: "sma_200",
    14: "ema_12",
    15: "ema_26",
    16: "sma_20_slope",
    17: "sma_50_slope",
    18: "dist_sma_20",
    19: "dist_sma_20_pct",
    20: "dist_sma_50",
    21: "dist_sma_50_pct",
    22: "dist_sma_200",
    23: "dist_sma_200_pct",
    24: "dist_ema_12",
    25: "dist_ema_12_pct",
    26: "dist_ema_26",
    27: "dist_ema_26_pct",
    28: "ma50_ma200_spread",
    29: "ma50_ma200_spread_pct",
    30: "macd",
    31: "macd_signal",
    32: "macd_hist",
    33: "macd_slope_9",
    34: "macd_hist_slope_9",
    35: "macd_z_252",
    36: "rsi_14",
    37: "rsi_pct_252",
    38: "vol_20",
    39: "vol_60",
    40: "atr_14",
    41: "atr_pct_14",
    42: "vol20_pct_252",
    43: "atrpct14_pct_252",
    44: "vol_of_vol_20_60",
    45: "bb_mid_20",
    46: "bb_upper_20",
    47: "bb_lower_20",
    48: "bb_bw_20",
    49: "bb_pctb_20",
    50: "dist_bb_upper_20",
    51: "dist_bb_lower_20",
    52: "zscore_20",
    53: "dev_mean_20",
    54: "dev_mean_20_pct",
    55: "stoch_k_14",
    56: "stoch_d_14",
    57: "hl_range_ratio_14",
    58: "adv_20",
    59: "vol_shock_20",
    60: "vol_z_20",
    61: "vol_x_ret",
    62: "participation_rate",
    63: "dd_from_52w_high",
    64: "rolling_dd_63",
    65: "ddd_1d",
    66: "skew_60",
    67: "kurt_60",
    68: "beta_252",
    69: "corr_252",
    70: "trend_regime",
    71: "vol_regime",
}

# Fast set for MA-like features where users typically mean "signal", not raw level
_MA_COLS = {"sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26"}

# Plot ID -> name mapping (mirrors the CobwebSim server registry)
PLOTS: Dict[int, str] = {
    # Core performance
    1:  "equity_curve_linear",
    2:  "equity_curve_log",
    3:  "equity_vs_benchmark",
    4:  "daily_return_ts",
    5:  "rolling_sharpe",
    6:  "rolling_sortino",
    7:  "rolling_volatility",
    8:  "rolling_beta",
    9:  "rolling_alpha",
    10: "drawdown_underwater",
    11: "drawdown_duration",
    12: "drawdown_distribution",
    13: "var_over_time",
    14: "cvar_over_time",
    15: "tail_risk_histogram",
    # Trade analysis
    16: "trade_return_distribution",
    17: "win_loss_overlay",
    18: "expectancy_over_time",
    19: "trade_duration_vs_return",
    # Regime / context
    20: "performance_by_volatility_regime",
    21: "performance_by_trend_regime",
    22: "performance_bull_vs_bear",
    # Feature diagnostics
    23: "feature_correlation_heatmap",
    # Volume / execution
    24: "volume_ts",
    25: "volume_shock_20",
    26: "trade_participation_ts",
    27: "transaction_costs_ts",
}

# Category labels for features (id -> category string)
_FEATURE_CATEGORIES: Dict[int, str] = {
    **{i: "returns"        for i in range(1,  10)},
    **{i: "trend_ma"       for i in range(10, 30)},
    **{i: "macd"           for i in range(30, 36)},
    **{i: "rsi"            for i in range(36, 38)},
    **{i: "volatility"     for i in range(38, 45)},
    **{i: "bollinger"      for i in range(45, 52)},
    **{i: "mean_reversion" for i in range(52, 55)},
    **{i: "stochastic"     for i in range(55, 57)},
    **{i: "volume"         for i in range(57, 63)},
    **{i: "risk"           for i in range(63, 68)},
    **{i: "beta_corr"      for i in range(68, 70)},
    **{i: "regime"         for i in range(70, 72)},
}

# Scoring behaviour notes per feature ID
_FEATURE_NOTES: Dict[int, str] = {
    10: "score_by_id() converts to sma_signal = (Close - MA) / MA",
    11: "score_by_id() converts to sma_signal = (Close - MA) / MA",
    12: "score_by_id() converts to sma_signal = (Close - MA) / MA",
    13: "score_by_id() converts to sma_signal = (Close - MA) / MA",
    14: "score_by_id() converts to sma_signal = (Close - MA) / MA",
    15: "score_by_id() converts to sma_signal = (Close - MA) / MA",
    36: "auto-scaled to 0-1 range (÷100) when rsi_to_unit=True",
    37: "auto-scaled to 0-1 range (÷100) when rsi_to_unit=True",
}

# Category labels for plots (id -> category string)
_PLOT_CATEGORIES: Dict[int, str] = {
    **{i: "performance"       for i in range(1,  16)},
    **{i: "trades"            for i in range(16, 20)},
    **{i: "regime"            for i in range(20, 23)},
    23: "diagnostics",
    **{i: "volume_execution"  for i in range(24, 28)},
}

# Dependency / requirement notes per plot ID
_PLOT_NOTES: Dict[int, str] = {
    3:  "requires benchmark data",
    8:  "requires benchmark data",
    9:  "requires benchmark data",
    20: "requires feature_id=71 (vol_regime) — pass feature_ids=[71]",
    21: "requires feature_id=70 (trend_regime) — pass feature_ids=[70]",
    22: "requires bull/bear regime columns from backtest",
    23: "requires all feature_ids to be computed — pass feature_ids=[1..71]",
}

# Sorted lists of valid category strings — public so users can inspect them
FEATURE_CATS: List[str] = sorted(set(_FEATURE_CATEGORIES.values()))
PLOT_CATS:    List[str] = sorted(set(_PLOT_CATEGORIES.values()))

# Private aliases used internally (keeps internal references short)
_FEATURE_CATS = FEATURE_CATS
_PLOT_CATS    = PLOT_CATS


def show_categories() -> None:
    """
    Print all valid category names for features and plots.

    These are the exact strings accepted by list_features(), list_plots(),
    show_features(), and show_plots() — fuzzy variants also work
    (e.g. 'betacorr', 'vol', 'perf').

    Example:
        show_categories()
    """
    print("Feature categories  (use with list_features / show_features):")
    for c in FEATURE_CATS:
        print(f"  {c}")
    print()
    print("Plot categories  (use with list_plots / show_plots):")
    for c in PLOT_CATS:
        print(f"  {c}")


def _resolve_category(query: str, valid: List[str]) -> List[str]:
    """
    Fuzzy-match a user query against a list of known category strings.

    Matching order (most specific wins):
      1. Exact match after normalising (lowercase, strip _, - and spaces)
      2. Prefix match  — query is a prefix of one or more categories
      3. Substring match — query appears anywhere inside a category

    Returns a list of matched original category strings (may be >1 for
    ambiguous prefixes like "vol" → ["volatility", "volume"]).
    Returns [] when nothing matches.
    """
    def _norm(s: str) -> str:
        return s.lower().replace("_", "").replace("-", "").replace(" ", "")

    q = _norm(query)
    norm_map: Dict[str, str] = {_norm(c): c for c in valid}

    # 1. Exact match
    if q in norm_map:
        return [norm_map[q]]

    # 2. Prefix match
    prefix_hits = [norm_map[k] for k in norm_map if k.startswith(q)]
    if prefix_hits:
        return sorted(prefix_hits)

    # 3. Substring match
    sub_hits = [norm_map[k] for k in norm_map if q in k]
    return sorted(sub_hits)


def _resolve_plot_id(query: str) -> int:
    """
    Resolve a fuzzy plot name string to its integer plot ID.

    Uses the same normalisation (lowercase, strip _, - and spaces) as
    _resolve_category, then matches against PLOTS names (not categories).

    Raises ValueError when the query is ambiguous or unrecognised — the
    message includes all matching or available plot names so the caller
    can refine their input. Use show_plots() for a full reference.
    """
    def _norm(s: str) -> str:
        return s.lower().replace("_", "").replace("-", "").replace(" ", "")

    q = _norm(query)
    # normalised name -> plot id
    norm_map: Dict[str, int] = {_norm(name): pid for pid, name in PLOTS.items()}

    def _fmt(ids: List[int]) -> str:
        return ", ".join(f"{pid}={PLOTS[pid]}" for pid in sorted(ids))

    # 1. Exact match
    if q in norm_map:
        return norm_map[q]

    # 2. Prefix match
    prefix = [norm_map[k] for k in norm_map if k.startswith(q)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        raise ValueError(
            f"Ambiguous plot name '{query}' — {len(prefix)} matches: {_fmt(prefix)}. "
            f"Be more specific."
        )

    # 3. Substring match
    sub = [norm_map[k] for k in norm_map if q in k]
    if len(sub) == 1:
        return sub[0]
    if len(sub) > 1:
        raise ValueError(
            f"Ambiguous plot name '{query}' — {len(sub)} matches: {_fmt(sub)}. "
            f"Be more specific."
        )

    # No match at all
    raise ValueError(
        f"No plot matched '{query}'. Run show_plots() to see all available plot names."
    )


def list_features(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return all 71 feature IDs as a list of dicts: id, name, category.

    Args:
        category: optional fuzzy filter — exact, prefix, or partial match.
                  e.g. "rsi", "vol" (→ volatility + volume), "betacorr",
                  "mean", "stoch", "trend"

    Valid categories:
        returns, trend_ma, macd, rsi, volatility, bollinger,
        mean_reversion, stochastic, volume, risk, beta_corr, regime

    Example:
        list_features()                   # all 71
        list_features("rsi")              # exact
        list_features("vol")              # volatility + volume
        list_features("betacorr")         # beta_corr (no underscore needed)
        pd.DataFrame(list_features("ma")) # trend_ma
    """
    all_rows = [
        {
            "id":       fid,
            "name":     name,
            "category": _FEATURE_CATEGORIES.get(fid, ""),
        }
        for fid, name in sorted(FEATURES.items())
    ]
    if category is None:
        return all_rows
    matched = _resolve_category(category, _FEATURE_CATS)
    return [r for r in all_rows if r["category"] in matched]


def list_plots(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return all 27 plot IDs as a list of dicts: id, name, category.

    Args:
        category: optional fuzzy filter — exact, prefix, or partial match.
                  e.g. "trades", "perf" (→ performance), "exec", "diag"

    Valid categories:
        performance, trades, regime, diagnostics, volume_execution

    Example:
        list_plots()                      # all 27
        list_plots("trades")              # exact
        list_plots("perf")               # performance
        list_plots("exec")               # volume_execution
        pd.DataFrame(list_plots("regime"))
    """
    all_rows = [
        {
            "id":       pid,
            "name":     name,
            "category": _PLOT_CATEGORIES.get(pid, ""),
        }
        for pid, name in sorted(PLOTS.items())
    ]
    if category is None:
        return all_rows
    matched = _resolve_category(category, _PLOT_CATS)
    return [r for r in all_rows if r["category"] in matched]


def show_features(category: Optional[str] = None) -> None:
    """
    Print a formatted feature reference table to stdout (no pandas needed).

    Args:
        category: optional fuzzy filter (same as list_features)

    Example:
        show_features()
        show_features("rsi")
        show_features("vol")       # volatility + volume
        show_features("betacorr")  # beta_corr
    """
    rows = list_features(category=category)
    if not rows:
        valid = ", ".join(_FEATURE_CATS)
        print(f"No features matched '{category}'. Valid categories: {valid}")
        return

    w_name = max(len(r["name"]) for r in rows)
    w_cat  = max(len(r["category"]) for r in rows)

    print(f"{'ID':>4}  {'Name':<{w_name}}  {'Category':<{w_cat}}")
    print(f"{'─'*4}  {'─'*w_name}  {'─'*w_cat}")
    for r in rows:
        print(f"{r['id']:>4}  {r['name']:<{w_name}}  {r['category']:<{w_cat}}")


def show_plots(category: Optional[str] = None) -> None:
    """
    Print a formatted plot reference table to stdout (no pandas needed).

    Args:
        category: optional fuzzy filter (same as list_plots)

    Example:
        show_plots()
        show_plots("trades")
        show_plots("perf")   # performance
        show_plots("exec")   # volume_execution
    """
    rows = list_plots(category=category)
    if not rows:
        valid = ", ".join(_PLOT_CATS)
        print(f"No plots matched '{category}'. Valid categories: {valid}")
        return

    w_name = max(len(r["name"]) for r in rows)
    w_cat  = max(len(r["category"]) for r in rows)

    print(f"{'ID':>4}  {'Name':<{w_name}}  {'Category':<{w_cat}}")
    print(f"{'─'*4}  {'─'*w_name}  {'─'*w_cat}")
    for r in rows:
        print(f"{r['id']:>4}  {r['name']:<{w_name}}  {r['category']:<{w_cat}}")


def _to_df(rows_or_df: Any):
    if pd is None:
        raise RuntimeError("pandas is required for scoring helpers. Install with: pip install pandas")

    if hasattr(rows_or_df, "__class__") and rows_or_df.__class__.__name__ == "DataFrame":
        return rows_or_df.copy()
    if isinstance(rows_or_df, dict) and "rows" in rows_or_df:
        return pd.DataFrame(rows_or_df["rows"])
    if isinstance(rows_or_df, list):
        return pd.DataFrame(rows_or_df)
    raise ValueError("Provide a DataFrame, list of dict rows, or dict with {'rows': [...]}.")


def zscore(series):
    # ddof=0 is slightly faster and stable; fine for scoring
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if sd is None or sd == 0 or pd.isna(sd):
        return s * 0.0
    return (s - mu) / sd


def score(
    rows_or_df: Any,
    weights: Mapping[str, float],
    *,
    normalize: bool = True,
    rsi_to_unit: bool = True,
) -> List[float]:
    """
    Compute weighted scores using COLUMN NAMES (strings).
    """
    df = _to_df(rows_or_df)

    # derived MA signal if requested
    # Accept both "Close" (enriched API rows) and "close" (raw load_csv rows)
    _close_col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
    if "sma_signal" in weights:
        if _close_col and "sma_20" in df.columns:
            close = pd.to_numeric(df[_close_col], errors="coerce")
            sma = pd.to_numeric(df["sma_20"], errors="coerce")
            df["sma_signal"] = (close - sma) / sma
        elif _close_col and "sma_10" in df.columns:
            close = pd.to_numeric(df[_close_col], errors="coerce")
            sma = pd.to_numeric(df["sma_10"], errors="coerce")
            df["sma_signal"] = (close - sma) / sma
        else:
            df["sma_signal"] = 0.0

    _acc = None
    for col, w in weights.items():
        contrib = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(0.0, index=df.index)

        if rsi_to_unit and isinstance(col, str) and col.lower().startswith("rsi"):
            contrib = contrib / 100.0

        if normalize and not (rsi_to_unit and isinstance(col, str) and col.lower().startswith("rsi")):
            contrib = zscore(contrib)

        _acc = (w * contrib) if _acc is None else (_acc + w * contrib)

    if _acc is None:
        return [0.0] * len(df)
    return [float(x) for x in _acc.fillna(0.0).tolist()]


def auto_score(rows_or_df: Any) -> List[float]:
    """
    Default formula (name-based):
      score = 0.3*rsi_14 + 0.3*sma_signal + 0.4*ret_1d
    """
    weights = {"rsi_14": 0.3, "sma_signal": 0.3, "ret_1d": 0.4}
    return score(rows_or_df, weights, normalize=True, rsi_to_unit=True)


def score_by_id(
    rows_or_df: Any,
    weights_by_id: Mapping[Union[int, str], float],
    *,
    normalize: bool = True,
    rsi_to_unit: bool = True,
    ma_ids_use_signal: bool = True,
) -> List[float]:
    """
    Compute weighted scores using NUMERIC feature IDs.

    Example:
        weights_by_id = {36: 0.3, 11: 0.3, 1: 0.4}
    Behavior:
      - IDs are mapped via FEATURES[id] -> column name
      - If ma_ids_use_signal=True and the id is a MA feature (sma_*/ema_*),
        it contributes to a derived `sma_signal` term (Close - MA)/MA
      - RSI columns can be scaled to 0..1 via /100
      - Non-RSI columns are optionally z-scored (normalize=True)
    """
    df = _to_df(rows_or_df)

    # Build name-weights efficiently
    name_weights: Dict[str, float] = {}
    want_sma_signal = False

    for fid, w in weights_by_id.items():
        try:
            fid_int = int(fid)
        except Exception:
            continue
        col = FEATURES.get(fid_int)
        if not col:
            continue

        w = float(w)
        if ma_ids_use_signal and col in _MA_COLS:
            want_sma_signal = True
            name_weights["sma_signal"] = name_weights.get("sma_signal", 0.0) + w
        else:
            name_weights[col] = name_weights.get(col, 0.0) + w

    if want_sma_signal and "sma_signal" not in df.columns:
        # Accept both "Close" (enriched API rows) and "close" (raw load_csv rows)
        _close_col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
        # Prefer sma_20, fallback sma_10
        if _close_col and "sma_20" in df.columns:
            close = pd.to_numeric(df[_close_col], errors="coerce")
            ma = pd.to_numeric(df["sma_20"], errors="coerce")
            df["sma_signal"] = (close - ma) / ma
        elif _close_col and "sma_10" in df.columns:
            close = pd.to_numeric(df[_close_col], errors="coerce")
            ma = pd.to_numeric(df["sma_10"], errors="coerce")
            df["sma_signal"] = (close - ma) / ma
        else:
            df["sma_signal"] = 0.0

    # Compute score without extra conversions
    _acc = None
    for col, w in name_weights.items():
        contrib = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(0.0, index=df.index)

        if rsi_to_unit and col.lower().startswith("rsi"):
            contrib = contrib / 100.0
            # keep RSI un-zscored when scaled to unit (typical preference)
            if normalize and not (rsi_to_unit and col.lower().startswith("rsi")):
                contrib = zscore(contrib)
        else:
            if normalize:
                contrib = zscore(contrib)

        _acc = (w * contrib) if _acc is None else (_acc + w * contrib)

    if _acc is None:
        return [0.0] * len(df)
    # Reindex against df to guarantee output length == len(df) regardless of
    # any index drift that can occur when the API returns fewer rows than sent.
    return [float(x) for x in _acc.reindex(df.index).fillna(0.0).tolist()]


def auto_score_by_id(rows_or_df: Any) -> List[float]:
    """
    Default formula using IDs only:
      36=rsi_14, 11=sma_20 (as sma_signal), 1=ret_1d
    """
    return score_by_id(rows_or_df, {36: 0.3, 11: 0.3, 1: 0.4}, normalize=True, rsi_to_unit=True)
