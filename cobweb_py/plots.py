from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Union, Optional, Tuple
import json
import math

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
except Exception:  # pragma: no cover
    go = None  # type: ignore
    make_subplots = None  # type: ignore


# ----------------------------
# Labeling / naming helpers
# ----------------------------

# Canonical axis labels for common keys
_AXIS_LABELS: Dict[str, str] = {
    "t": "Time",
    "timestamp": "Timestamp",
    "time": "Time",
    "date": "Date",
    "datetime": "Datetime",
    "x": "X",
    "y": "Y",
    "bins": "Bins",
    "counts": "Count",
    "duration_bars": "Duration (bars)",
    "return": "Return",
    "regime": "Regime",
    "mean_ret": "Mean return",
    "std_ret": "Std. dev. (return)",
    "volume": "Volume",
    "adv": "Average daily volume (ADV)",
    "vol_shock": "Volume shock (vol/ADV)",
    "score": "Score",
    "equity": "Equity",
    "portfolio_value": "Portfolio value",
    "value": "Value",
    "nav": "Net asset value (NAV)",
    "cash": "Cash",
    "Close": "Close",
    "close": "Close",
}

# Chart title defaults by plot "shape"
_TITLE_DEFAULTS: Dict[str, str] = {
    "timeseries": "Time Series",
    "hist": "Histogram",
    "xy": "XY Plot",
    "dist": "Distribution",
    "scatter": "Scatter",
    "regime": "Regime Return Summary",
    "bullbear": "Bull vs Bear Mean Return",
    "winloss": "Wins vs Losses (Histogram Overlay)",
    "volshock": "Volume, ADV and Volume Shock",
    "corr": "Correlation Heatmap",
    "equity": "Equity Curve",
    "score": "Score Over Time",
    "price_score": "Price and Score",
}


def _label(name: str) -> str:
    """Pretty label for axis titles and trace names."""
    if not name:
        return ""
    key = name.strip()
    # exact match first
    if key in _AXIS_LABELS:
        return _AXIS_LABELS[key]
    # common casing variants
    low = key.lower()
    if low in _AXIS_LABELS:
        return _AXIS_LABELS[low]
    # fallback: humanize snake_case
    return key.replace("_", " ").strip().title()


def _make_title(*parts: str) -> str:
    s = " - ".join([p for p in parts if p and str(p).strip()])
    return s.strip() or "Plot"


def _ensure_plotly():
    if go is None:
        raise RuntimeError("plotly is required for HTML plots. Install with: pip install plotly")


# ----------------------------
# Dataframe helpers
# ----------------------------

def _to_df(obj: Any):
    if pd is None:
        raise RuntimeError("pandas is required for plotting helpers. Install with: pip install pandas")
    if hasattr(obj, "__class__") and obj.__class__.__name__ == "DataFrame":
        return obj.copy()
    if isinstance(obj, dict) and "rows" in obj:
        return pd.DataFrame(obj["rows"])
    if isinstance(obj, dict) and "equity_curve" in obj:
        return pd.DataFrame(obj["equity_curve"])
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    raise ValueError("Unsupported input type for plotting.")


def _pick_x(df):
    for c in ("timestamp", "time", "date", "Datetime", "Date", "index", "bar", "t"):
        if c in df.columns:
            return c
    return df.columns[0]


def _pick_equity_y(df):
    for c in ("equity", "Equity", "portfolio_value", "value", "nav", "cash"):
        if c in df.columns:
            return c
    return df.columns[-1]


# ----------------------------
# Generic plot writers
# ----------------------------

def save_line_plot(
    df_or_obj: Any,
    *,
    x: str,
    y: str,
    title: str,
    out_html: Union[str, Path],
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    trace_name: Optional[str] = None,
) -> str:
    _ensure_plotly()
    df = _to_df(df_or_obj)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y], name=trace_name or _label(y)))

    fig.update_layout(
        title=title,
        xaxis_title=x_title or _label(x),
        yaxis_title=y_title or _label(y),
        legend=dict(orientation="h"),
    )

    out_html = str(out_html)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs=True)
    return out_html


def _write_note_html(path: Union[str, Path], title: str, msg: str) -> None:
    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>{title}</title>
  </head>
  <body style="font-family: sans-serif; padding: 16px;">
    <h2>{title}</h2>
    <p>{msg}</p>
  </body>
</html>
"""
    Path(path).write_text(html, encoding="utf-8")


def _as_float_matrix(z: Any) -> List[List[float]]:
    """Convert nested lists to float matrix; non-finite/non-numeric => NaN."""
    out: List[List[float]] = []
    if not isinstance(z, list):
        return out
    for row in z:
        if not isinstance(row, list):
            continue
        r2: List[float] = []
        for v in row:
            try:
                fv = float(v)
                if not math.isfinite(fv):
                    fv = float("nan")
            except Exception:
                fv = float("nan")
            r2.append(fv)
        out.append(r2)
    return out


# ----------------------------
# Convenience plots
# ----------------------------

def save_equity_plot(backtest_result: Dict[str, Any], out_html: Union[str, Path] = "equity.html") -> str:
    _ensure_plotly()
    df = _to_df(backtest_result)
    x = _pick_x(df)
    y = _pick_equity_y(df)
    title = _make_title(_TITLE_DEFAULTS["equity"], _label(y))

    pos_col = "pos_units" if "pos_units" in df.columns else None

    if pos_col and make_subplots is not None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df[x], y=df[y], name=_label(y)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df[x], y=df[pos_col], name=_label(pos_col)), secondary_y=True)
        fig.update_yaxes(title_text=_label(y), secondary_y=False)
        fig.update_yaxes(title_text=_label(pos_col), secondary_y=True)
        fig.update_layout(
            title=title,
            xaxis_title=_label(x),
            legend=dict(orientation="h"),
        )
        out_html = str(out_html)
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs=True)
        return out_html

    return save_line_plot(df, x=x, y=y, title=title, out_html=out_html)


def save_score_plot(rows_or_df: Any, scores: Sequence[float], out_html: Union[str, Path] = "score.html") -> str:
    df = _to_df(rows_or_df).copy()
    df["score"] = list(scores)
    x = _pick_x(df)
    title = _TITLE_DEFAULTS["score"]
    return save_line_plot(df, x=x, y="score", title=title, out_html=out_html)


def save_price_and_score_plot(
    rows_or_df: Any,
    scores: Sequence[float],
    out_html: Union[str, Path] = "price_and_score.html",
) -> str:
    _ensure_plotly()

    df = _to_df(rows_or_df).copy()
    df["score"] = list(scores)

    x = _pick_x(df)
    # pick a sensible price column
    if "Close" in df.columns:
        price_col = "Close"
    elif "close" in df.columns:
        price_col = "close"
    else:
        # prefer "price" if present, else last column (excluding score)
        candidates = [c for c in df.columns if c != "score"]
        price_col = "price" if "price" in df.columns else candidates[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[price_col], name=_label(price_col)))
    fig.add_trace(go.Scatter(x=df[x], y=df["score"], name=_label("score"), yaxis="y2"))

    fig.update_layout(
        title=_TITLE_DEFAULTS["price_score"],
        xaxis=dict(title=_label(x)),
        yaxis=dict(title=_label(price_col)),
        yaxis2=dict(title=_label("score"), overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )

    out_html = str(out_html)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs=True)
    return out_html


# ----------------------------
# API payload plots -> HTML
# ----------------------------

def payload_to_figure(
    payload: Any,
    name: str = "",
    *,
    title_prefix: str = "",
    layout_overrides: Optional[Dict[str, Any]] = None,
) -> Optional["go.Figure"]:
    """
    Convert a single API plot payload to a Plotly ``go.Figure``.

    Returns ``None`` if the payload shape is unrecognised or contains
    insufficient data.

    Args:
        payload:          The payload dict (one entry from the ``"payloads"``
                          dict returned by the API ``/plots`` endpoint).
        name:             Payload name (used for axis labels and title).
        title_prefix:     Prepended to the auto-generated chart title.
        layout_overrides: Optional dict passed to ``fig.update_layout()``
                          after default layout is applied, letting callers
                          customise colours, fonts, margins, themes, etc.

    Example::

        pl = sim.plots(data, bt, plot_ids=[1])
        for pname, p in pl["payloads"].items():
            fig = payload_to_figure(p, pname)
            if fig:
                fig.show()
    """
    _ensure_plotly()
    p = payload
    base_title = _make_title(title_prefix.rstrip(), name)

    def _apply_overrides(fig: "go.Figure") -> "go.Figure":
        if layout_overrides:
            fig.update_layout(**layout_overrides)
        return fig

    # ---------- SPECIAL CASES FIRST (many include "t") ----------

    # Special: volume shock (plot 25) — MUST come before generic (t + series)
    if (
        isinstance(p, dict)
        and isinstance(p.get("t"), list)
        and ("volume" in p)
        and ("adv" in p)
        and ("vol_shock" in p)
    ):
        t = p.get("t", [])
        vol = p.get("volume", [])
        adv = p.get("adv", [])
        shock = p.get("vol_shock", [])

        if not t or not isinstance(vol, list) or len(vol) != len(t):
            return None

        fig = go.Figure()
        if isinstance(vol, list) and len(vol) == len(t):
            fig.add_trace(go.Scatter(x=t, y=vol, name=_label("volume"), yaxis="y"))
        if isinstance(adv, list) and len(adv) == len(t):
            fig.add_trace(go.Scatter(x=t, y=adv, name=_label("adv"), yaxis="y"))
        if isinstance(shock, list) and len(shock) == len(t):
            fig.add_trace(go.Scatter(x=t, y=shock, name=_label("vol_shock"), yaxis="y2"))

        if len(fig.data) == 0:
            return None

        fig.update_layout(
            title=_make_title(base_title, _TITLE_DEFAULTS["volshock"]),
            xaxis=dict(title=_label("t")),
            yaxis=dict(title="Volume / ADV"),
            yaxis2=dict(title=_label("vol_shock"), overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        return _apply_overrides(fig)

    # Special: win/loss overlay (plot 17)
    if isinstance(p, dict) and isinstance(p.get("wins_hist"), dict) and isinstance(p.get("losses_hist"), dict):
        w = p["wins_hist"]
        l = p["losses_hist"]
        wb, wc = w.get("bins", []), w.get("counts", [])
        lb, lc = l.get("bins", []), l.get("counts", [])

        if not wb or not wc or not lb or not lc:
            return None

        n1 = min(len(wb), len(wc))
        n2 = min(len(lb), len(lc))

        fig = go.Figure()
        fig.add_trace(go.Bar(x=wb[:n1], y=wc[:n1], name="Wins", opacity=0.6))
        fig.add_trace(go.Bar(x=lb[:n2], y=lc[:n2], name="Losses", opacity=0.6))
        fig.update_layout(
            title=_make_title(base_title, _TITLE_DEFAULTS["winloss"]),
            barmode="overlay",
            xaxis_title="Trade return",
            yaxis_title=_label("counts"),
            legend=dict(orientation="h"),
        )
        return _apply_overrides(fig)

    # Special: regime bars (plots 20 & 21)
    if (
        isinstance(p, dict)
        and isinstance(p.get("regime"), list)
        and isinstance(p.get("mean_ret"), list)
        and isinstance(p.get("std_ret"), list)
    ):
        x = p["regime"]
        y = p["mean_ret"]
        e = p["std_ret"]
        n = min(len(x), len(y), len(e))
        if n == 0:
            return None

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=x[:n], y=y[:n],
                error_y=dict(type="data", array=e[:n], visible=True),
                name=_label("mean_ret"),
            )
        )
        fig.update_layout(
            title=_make_title(base_title, _TITLE_DEFAULTS["regime"]),
            xaxis_title=_label("regime"),
            yaxis_title=_label("mean_ret"),
            legend=dict(orientation="h"),
        )
        return _apply_overrides(fig)

    # Special: bull vs bear summary (plot 22)
    if isinstance(p, dict) and ("bull_mean_ret" in p) and ("bear_mean_ret" in p):
        bull = p.get("bull_mean_ret")
        bear = p.get("bear_mean_ret")
        if bull is None and bear is None:
            return None

        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Bull", "Bear"], y=[bull, bear], name=_label("mean_ret")))
        fig.update_layout(
            title=_make_title(base_title, _TITLE_DEFAULTS["bullbear"]),
            xaxis_title=_label("regime"),
            yaxis_title=_label("mean_ret"),
            legend=dict(orientation="h"),
        )
        return _apply_overrides(fig)

    # Special: correlation heatmap (plot 23)
    if isinstance(p, dict) and isinstance(p.get("cols"), list) and isinstance(p.get("matrix"), list):
        cols = p.get("cols", [])
        z = _as_float_matrix(p.get("matrix", []))
        if not cols or not z or not z[0]:
            return None

        fig = go.Figure(data=go.Heatmap(z=z, x=[_label(c) for c in cols], y=[_label(c) for c in cols]))
        fig.update_layout(
            title=_make_title(base_title, _TITLE_DEFAULTS["corr"]),
            xaxis_title="Features",
            yaxis_title="Features",
        )
        return _apply_overrides(fig)

    # ---------- GENERIC CASES AFTER SPECIAL CASES ----------

    # Case 1: time-series payload (t + series)
    if isinstance(p, dict) and isinstance(p.get("t"), list):
        t = p["t"]
        sec_y_keys = set(p.get("_secondary_y", []))
        use_secondary = bool(sec_y_keys)

        if use_secondary and make_subplots is not None:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
        else:
            fig = go.Figure()

        for k, v in p.items():
            if k == "t" or k.startswith("_"):
                continue
            if isinstance(v, list) and len(v) == len(t):
                on_secondary = use_secondary and k in sec_y_keys
                fig.add_trace(
                    go.Scatter(x=t, y=v, name=_label(k)),
                    secondary_y=on_secondary if use_secondary else None,
                )

        if len(fig.data) == 0:
            return None

        if use_secondary and make_subplots is not None:
            primary_keys = [
                k for k, v in p.items()
                if k != "t" and not k.startswith("_") and k not in sec_y_keys
                and isinstance(v, list) and len(v) == len(t)
            ]
            sec_keys = [k for k in sec_y_keys if isinstance(p.get(k), list)]
            fig.update_yaxes(title_text=_label(primary_keys[0]) if primary_keys else _label("value"), secondary_y=False)
            fig.update_yaxes(title_text=_label(sec_keys[0]) if sec_keys else _label("value"), secondary_y=True)
        else:
            fig.update_layout(yaxis_title=_label("value"))

        fig.update_layout(
            title=_make_title(base_title, _TITLE_DEFAULTS["timeseries"]),
            xaxis_title=_label("t"),
            legend=dict(orientation="h"),
        )
        return _apply_overrides(fig)

    # Case 2: histogram payload (bins + counts)
    if isinstance(p, dict) and isinstance(p.get("bins"), list) and isinstance(p.get("counts"), list):
        bins = p.get("bins", [])
        counts = p.get("counts", [])
        n = min(len(bins), len(counts))
        if n == 0:
            return None

        # Context-aware x-axis label based on plot name
        name_l = str(name).lower()
        if "trade_return" in name_l or "win_loss" in name_l:
            x_title = "Trade return"
        elif "tail_risk" in name_l:
            x_title = "Return"
        elif "drawdown" in name_l:
            x_title = "Drawdown"
        else:
            x_title = "Value"

        fig = go.Figure()
        fig.add_trace(go.Bar(x=bins[:n], y=counts[:n], name="Count"))
        fig.update_layout(
            title=_make_title(base_title, _TITLE_DEFAULTS["hist"]),
            xaxis_title=x_title,
            yaxis_title="Count",
            legend=dict(orientation="h"),
        )
        return _apply_overrides(fig)

    # Case 3: generic x/y payload
    if isinstance(p, dict) and isinstance(p.get("x"), list) and isinstance(p.get("y"), list):
        x = p["x"]
        y = p["y"]
        n = min(len(x), len(y))
        if n == 0:
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x[:n], y=y[:n], name=_label(name)))
        fig.update_layout(
            title=_make_title(base_title, _TITLE_DEFAULTS["xy"]),
            xaxis_title=_label("x"),
            yaxis_title=_label("y"),
        )
        return _apply_overrides(fig)

    # Case 4: distribution list (e.g., {"drawdowns": [...]})
    if isinstance(p, dict):
        for dist_key in ("drawdowns", "returns", "values", "samples"):
            if isinstance(p.get(dist_key), list) and len(p[dist_key]) > 0:
                data = p[dist_key]
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=data, name=_label(dist_key)))
                fig.update_layout(
                    title=_make_title(base_title, _TITLE_DEFAULTS["dist"], _label(dist_key)),
                    xaxis_title=_label(dist_key),
                    yaxis_title=_label("counts"),
                )
                return _apply_overrides(fig)

    # Case 5: scatter (duration vs return)
    if isinstance(p, dict) and isinstance(p.get("duration_bars"), list) and isinstance(p.get("return"), list):
        x = p["duration_bars"]
        y = p["return"]
        n = min(len(x), len(y))
        if n == 0:
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x[:n], y=y[:n], mode="markers", name=_label("return")))
        fig.update_layout(
            title=_make_title(base_title, _TITLE_DEFAULTS["scatter"]),
            xaxis_title=_label("duration_bars"),
            yaxis_title=_label("return"),
        )
        return _apply_overrides(fig)

    # Unrecognised shape
    return None


def payloads_to_figures(
    payloads: Dict[str, Any],
    *,
    title_prefix: str = "",
    layout_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, "go.Figure"]:
    """
    Convert a dict of API plot payloads to a dict of Plotly figures.

    Skips payloads that produce ``None`` (unrecognised shape or insufficient
    data).

    Args:
        payloads:         Dict of ``{name: payload}``, as returned by the API
                          ``/plots`` endpoint in the ``"payloads"`` key.
        title_prefix:     Prepended to each chart title.
        layout_overrides: Optional dict passed to each figure's
                          ``update_layout()``.

    Example::

        pl = sim.plots(data, bt, plot_ids=[1, 5, 10])
        figs = payloads_to_figures(pl["payloads"])
        for name, fig in figs.items():
            fig.show()
    """
    _ensure_plotly()
    result: Dict[str, "go.Figure"] = {}
    for pname, p in (payloads or {}).items():
        fig = payload_to_figure(p, pname, title_prefix=title_prefix, layout_overrides=layout_overrides)
        if fig is not None:
            result[pname] = fig
    return result


def save_api_payloads_to_html(
    payloads: Dict[str, Any],
    out_dir: Union[str, Path] = "out/api_plots",
    *,
    title_prefix: str = "",
) -> List[str]:
    """
    Render API plot payloads into individual HTML files.

    Internally delegates to :func:`payload_to_figure` for figure construction,
    then writes each figure to an HTML file.  Payloads that cannot be converted
    to a figure are written as raw JSON.
    """
    _ensure_plotly()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[str] = []

    for name, p in (payloads or {}).items():
        out_path = out_dir / f"{name}.html"
        fig = payload_to_figure(p, name, title_prefix=title_prefix)

        if fig is not None:
            fig.write_html(str(out_path), include_plotlyjs=True)
        else:
            # Fallback: write payload as JSON for inspection
            base_title = _make_title(title_prefix.rstrip(), name)
            raw = json.dumps(p, indent=2, ensure_ascii=False) if not isinstance(p, str) else p
            out_path.write_text(
                f"<html><head><meta charset='utf-8'><title>{base_title}</title></head>"
                f"<body><h2>{base_title}</h2><pre style='white-space:pre-wrap;'>{raw}</pre></body></html>",
                encoding="utf-8",
            )

        written.append(str(out_path))

    return written


# ----------------------------
# API payload plots -> DataFrames
# (kept mostly the same, but cleaned up)
# ----------------------------

def save_api_payloads_to_dfs(
    payloads: Dict[str, Any],
    *,
    fallback: str = "json",  # "json" | "normalize" | "none"
) -> Dict[str, "pd.DataFrame"]:
    if pd is None:
        raise RuntimeError("pandas is required. Install with: pip install pandas")

    out: Dict[str, pd.DataFrame] = {}
    for name, p in (payloads or {}).items():
        out[name] = _payload_to_df(p, fallback=fallback)
    return out


def _payload_to_df(p: Any, *, fallback: str = "json") -> "pd.DataFrame":
    """
    Convert a plot payload to a pandas DataFrame.

    Rules:
    - No plotting / file writing / loop control here.
    - Order special-case payloads before generic ones (e.g., volume_shock before t+series).
    """
    if pd is None:
        raise RuntimeError("pandas is required. Install with: pip install pandas")

    # ---------- SPECIAL CASES FIRST ----------

    # Case 9: volume shock (t + volume + adv + vol_shock)
    if (
        isinstance(p, dict)
        and isinstance(p.get("t"), list)
        and ("volume" in p)
        and ("adv" in p)
        and ("vol_shock" in p)
    ):
        t = p.get("t", [])
        vol = p.get("volume", [])
        adv = p.get("adv", [])
        shock = p.get("vol_shock", [])
        n = min(
            len(t),
            len(vol) if isinstance(vol, list) else 0,
            len(adv) if isinstance(adv, list) else 0,
            len(shock) if isinstance(shock, list) else 0,
        )
        return pd.DataFrame(
            {
                "t": t[:n],
                "volume": vol[:n] if isinstance(vol, list) else [],
                "adv": adv[:n] if isinstance(adv, list) else [],
                "vol_shock": shock[:n] if isinstance(shock, list) else [],
            }
        )

    # Case 8: win/loss overlay (wins_hist + losses_hist)
    if isinstance(p, dict) and isinstance(p.get("wins_hist"), dict) and isinstance(p.get("losses_hist"), dict):
        w = p.get("wins_hist", {}) or {}
        l = p.get("losses_hist", {}) or {}

        wb, wc = (w.get("bins", []) or []), (w.get("counts", []) or [])
        lb, lc = (l.get("bins", []) or []), (l.get("counts", []) or [])

        n1 = min(len(wb), len(wc))
        n2 = min(len(lb), len(lc))

        rows = [{"side": "wins", "bin": wb[i], "count": wc[i]} for i in range(n1)]
        rows += [{"side": "losses", "bin": lb[i], "count": lc[i]} for i in range(n2)]
        return pd.DataFrame(rows)

    # Case 6: regime bars (mean +/- std) (+ optional count)
    if (
        isinstance(p, dict)
        and isinstance(p.get("regime"), list)
        and isinstance(p.get("mean_ret"), list)
        and isinstance(p.get("std_ret"), list)
    ):
        x = p.get("regime", [])
        y = p.get("mean_ret", [])
        e = p.get("std_ret", [])
        n = min(len(x), len(y), len(e))

        data: Dict[str, Any] = {"regime": x[:n], "mean_ret": y[:n], "std_ret": e[:n]}
        if isinstance(p.get("count"), list):
            c = p.get("count", [])
            data["count"] = c[: min(len(c), n)]
        return pd.DataFrame(data)

    # Case 7: bull vs bear summary
    if isinstance(p, dict) and ("bull_mean_ret" in p) and ("bear_mean_ret" in p):
        bull = p.get("bull_mean_ret")
        bear = p.get("bear_mean_ret")
        return pd.DataFrame({"regime": ["bull", "bear"], "mean_ret": [bull, bear]})

    # Case 10: correlation heatmap (cols + matrix)
    if isinstance(p, dict) and isinstance(p.get("cols"), list) and isinstance(p.get("matrix"), list):
        cols = p.get("cols", [])
        matrix = p.get("matrix", [])
        try:
            df = pd.DataFrame(matrix, columns=cols)
            # if square matrix, label rows too
            if len(cols) == len(df.index):
                df.index = cols
            return df
        except Exception:
            pass

    # ---------- GENERIC CASES AFTER SPECIAL CASES ----------

    # Case 1: t + series (generic)
    if isinstance(p, dict) and isinstance(p.get("t"), list):
        t = p.get("t", [])
        data: Dict[str, Any] = {"t": t}
        for k, v in p.items():
            if k == "t" or k.startswith("_"):
                continue
            if isinstance(v, list) and len(v) == len(t):
                data[k] = v
        return pd.DataFrame(data)

    # Case 2: histogram payload (bins + counts)
    # IMPORTANT: In DataFrame form, keep it semantic: "value" + "count"
    if isinstance(p, dict) and isinstance(p.get("bins"), list) and isinstance(p.get("counts"), list):
        bins = p.get("bins", [])
        counts = p.get("counts", [])
        n = min(len(bins), len(counts))
        return pd.DataFrame({"value": bins[:n], "count": counts[:n]})

    # Case 3: x/y
    if isinstance(p, dict) and isinstance(p.get("x"), list) and isinstance(p.get("y"), list):
        x = p.get("x", [])
        y = p.get("y", [])
        n = min(len(x), len(y))
        return pd.DataFrame({"x": x[:n], "y": y[:n]})

    # Case 4: distribution list
    if isinstance(p, dict):
        for dist_key in ("drawdowns", "returns", "values", "samples"):
            if isinstance(p.get(dist_key), list) and len(p[dist_key]) > 0:
                return pd.DataFrame({dist_key: p[dist_key]})

    # Case 5: scatter (duration vs return)
    if isinstance(p, dict) and isinstance(p.get("duration_bars"), list) and isinstance(p.get("return"), list):
        x = p.get("duration_bars", [])
        y = p.get("return", [])
        n = min(len(x), len(y))
        return pd.DataFrame({"duration_bars": x[:n], "return": y[:n]})

    # ---------- FALLBACKS ----------

    if fallback == "none":
        return pd.DataFrame()

    if fallback == "normalize":
        try:
            return pd.json_normalize(p) if isinstance(p, (dict, list)) else pd.DataFrame({"value": [p]})
        except Exception:
            pass

    try:
        raw = json.dumps(p, ensure_ascii=False, indent=2) if not isinstance(p, str) else p
    except Exception:
        raw = str(p)
    return pd.DataFrame({"payload_json": [raw]})
    

def save_features_table(
    rows_or_df: Any,
    *,
    out_html: Union[str, Path] = "features_table.html",
    out_csv: Union[str, Path] = "features_table.csv",
    title: str = "Features Table",
    max_rows: Optional[int] = None,
    float_precision: int = 6,
    index: bool = False,
) -> Tuple[str, str]:
    """
    Save a stylized HTML table + a CSV snapshot of the features rows.

    Mode A:
      Pins core columns first (timestamp/date + OHLCV + Volume), then all feature columns.
    """
    if pd is None:
        raise RuntimeError("pandas is required for table export. Install with: pip install pandas")

    df = _to_df(rows_or_df)

    if max_rows is not None:
        df = df.head(int(max_rows)).copy()

    # ----------------------------
    # Column ordering: core first
    # ----------------------------
    core_candidates = [
        # time
        "timestamp", "Timestamp", "date", "Date", "Datetime", "datetime", "time", "Time", "t",
        # ohlcv (common variants)
        "open", "Open",
        "high", "High",
        "low", "Low",
        "close", "Close",
        "volume", "Volume",
    ]
    core_cols = [c for c in core_candidates if c in df.columns]

    # Remaining columns: keep original order (stable) after core
    remaining_cols = [c for c in df.columns if c not in set(core_cols)]
    df = df[core_cols + remaining_cols]

    # ---- CSV ----
    out_csv = str(out_csv)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=index)

    # ---- HTML (stylized) ----
    out_html = str(out_html)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)

    styler = df.style.format(precision=int(float_precision), na_rep="")

    # Hide index (works across pandas versions)
    try:
        styler = styler.hide(axis="index")
    except Exception:  # pragma: no cover
        try:
            styler = styler.hide_index()
        except Exception:
            pass

    styler = styler.set_table_styles(
        [
            {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
            {"selector": "th", "props": [("position", "sticky"), ("top", "0"),
                                        ("background", "#0f172a"), ("color", "white"),
                                        ("text-align", "left"), ("padding", "10px"),
                                        ("font-weight", "600"), ("border-bottom", "1px solid #334155")]},
            {"selector": "td", "props": [("padding", "8px 10px"), ("border-bottom", "1px solid #e2e8f0")]},
            {"selector": "tr:nth-child(even) td", "props": [("background", "#f8fafc")]},
            {"selector": "tr:hover td", "props": [("background", "#eef2ff")]},
            {"selector": "caption", "props": [("caption-side", "top"), ("text-align", "left"),
                                              ("font-size", "18px"), ("font-weight", "700"),
                                              ("padding", "8px 0 12px 0")]},
        ]
    )

    try:
        styler = styler.set_caption(title)
    except Exception:
        pass

    table_html = styler.to_html()

    html_doc = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>{title}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 18px; }}
      .meta {{ color: #475569; font-size: 13px; margin-bottom: 10px; }}
      .wrap {{ overflow:auto; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px; }}
    </style>
  </head>
  <body>
    <div class="meta"></div>
    <div class="wrap">
      {table_html}
    </div>
  </body>
</html>
"""
    Path(out_html).write_text(html_doc, encoding="utf-8")
    return out_html, out_csv


# ----------------------------
# Backtest result tables
# ----------------------------

_METRIC_LABELS: Dict[str, str] = {
    "final_equity":               "Final Equity",
    "total_return":               "Total Return",
    "return_ann":                 "Annualised Return",
    "volatility_ann":             "Annualised Volatility",
    "sharpe_ann":                 "Sharpe Ratio",
    "sortino_ann":                "Sortino Ratio",
    "max_drawdown":               "Max Drawdown",
    "max_drawdown_duration_bars": "Max Drawdown Duration (bars)",
    "mean_return_bar":            "Mean Return (per bar)",
    "var_bar":                    "VaR (per bar)",
    "cvar_bar":                   "CVaR (per bar)",
    "var_hist_q":                 "VaR Quantile",
    "skew":                       "Skewness",
    "kurtosis":                   "Excess Kurtosis",
    "bars":                       "Total Bars",
    "trades":                     "Total Trades",
    "reward_sum":                 "Reward Sum (RL)",
}

_PCT_METRICS = {
    "total_return", "return_ann", "volatility_ann",
    "max_drawdown", "mean_return_bar", "var_bar", "cvar_bar",
}


def _fmt_metric(key: str, val: Any) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    if key == "final_equity":
        return f"${float(val):,.2f}"
    if key in _PCT_METRICS:
        return f"{float(val) * 100:.2f}%"
    if key in {"bars", "trades", "max_drawdown_duration_bars"}:
        return str(int(val))
    if key == "var_hist_q":
        return f"{float(val) * 100:.0f}th pct."
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def save_metrics_table(
    bt: Dict[str, Any],
    out_html: Union[str, Path] = "metrics.html",
) -> str:
    """
    Save a styled HTML summary of backtest metrics.

    Example:
        save_metrics_table(bt, out_html="out/metrics.html")
    """
    metrics = bt.get("metrics", {})
    out_html = str(out_html)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)

    rows_html = ""
    for key, label in _METRIC_LABELS.items():
        if key not in metrics:
            continue
        val_str = _fmt_metric(key, metrics[key])
        rows_html += (
            f"<tr>"
            f"<td style='padding:9px 14px;border-bottom:1px solid #e2e8f0;color:#475569;font-weight:500'>{label}</td>"
            f"<td style='padding:9px 14px;border-bottom:1px solid #e2e8f0;text-align:right;font-variant-numeric:tabular-nums'>{val_str}</td>"
            f"</tr>"
        )

    signal = bt.get("current_signal", "")
    exposure = bt.get("current_exposure", None)
    strength = bt.get("signal_strength", None)
    signal_color = {"buy": "#16a34a", "sell": "#dc2626"}.get(signal, "#64748b")
    signal_html = (
        f"<div style='margin-top:16px;padding:12px 16px;border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0'>"
        f"<span style='font-weight:700;color:{signal_color};font-size:15px'>{signal.upper() if signal else '—'}</span>"
        + (f" &nbsp;|&nbsp; exposure <b>{exposure:.2f}</b>" if exposure is not None else "")
        + (f" &nbsp;|&nbsp; strength <b>{strength:.2f}</b>" if strength is not None else "")
        + "</div>"
    ) if signal else ""

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Backtest Metrics</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
              margin: 18px; color: #1e293b; }}
      h2   {{ margin-bottom: 4px; }}
      table {{ border-collapse: collapse; min-width: 340px; }}
      tr:hover td {{ background: #eef2ff; }}
    </style>
  </head>
  <body>
    <h2>Backtest Metrics</h2>
    {signal_html}
    <table style="margin-top:12px">
      <tbody>{rows_html}</tbody>
    </table>
  </body>
</html>"""

    Path(out_html).write_text(html, encoding="utf-8")
    return out_html


def save_trades_table(
    bt: Dict[str, Any],
    out_html: Union[str, Path] = "trades.html",
) -> str:
    """
    Save a styled HTML table of all trades from a backtest result.

    Example:
        save_trades_table(bt, out_html="out/trades.html")
    """
    trades = bt.get("trades", [])
    out_html = str(out_html)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)

    _SIDE = {1: "BUY", -1: "SELL", 0: "FLAT"}
    _SIDE_COLOR = {1: "#16a34a", -1: "#dc2626", 0: "#64748b"}

    cols = ["t", "side", "fill", "units", "cost", "tca_cost", "target_exposure"]
    headers = ["Time", "Side", "Fill Price", "Units", "Fee", "TCA Cost", "Target Exp."]

    header_html = "".join(
        f"<th style='padding:10px 12px;text-align:{"right" if i > 1 else "left"};background:#0f172a;"
        f"color:white;font-weight:600;border-bottom:1px solid #334155;white-space:nowrap'>{h}</th>"
        for i, h in enumerate(headers)
    )

    rows_html = ""
    for i, trade in enumerate(trades):
        side_val = trade.get("side", 0)
        side_label = _SIDE.get(side_val, str(side_val))
        side_color = _SIDE_COLOR.get(side_val, "#64748b")
        bg = "#f8fafc" if i % 2 == 0 else "white"

        cells = []
        for j, col in enumerate(cols):
            val = trade.get(col, "")
            if col == "side":
                cell = f"<span style='color:{side_color};font-weight:600'>{side_label}</span>"
            elif col == "t":
                cell = str(val)
            elif isinstance(val, float):
                cell = f"{val:.4f}" if col == "fill" else f"{val:.6f}"
            else:
                cell = str(val) if val != "" else "—"
            align = "right" if j > 1 else "left"
            cells.append(
                f"<td style='padding:8px 12px;border-bottom:1px solid #e2e8f0;text-align:{align};"
                f"font-variant-numeric:tabular-nums'>{cell}</td>"
            )
        rows_html += f"<tr style='background:{bg}'>{''.join(cells)}</tr>"

    n = len(trades)
    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Trades ({n})</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body  {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
               margin: 18px; color: #1e293b; }}
      h2    {{ margin-bottom: 4px; }}
      .meta {{ color: #64748b; font-size: 13px; margin-bottom: 10px; }}
      .wrap {{ overflow: auto; border: 1px solid #e2e8f0; border-radius: 10px; }}
      table {{ border-collapse: collapse; width: 100%; }}
      tr:hover td {{ background: #eef2ff !important; }}
    </style>
  </head>
  <body>
    <h2>Trades</h2>
    <p class="meta">{n} trade{"s" if n != 1 else ""}</p>
    <div class="wrap">
      <table>
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
  </body>
</html>"""

    Path(out_html).write_text(html, encoding="utf-8")
    return out_html