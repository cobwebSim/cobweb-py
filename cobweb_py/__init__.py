"""cobweb-py - Python client for CobwebSim."""

from .client import CobwebSim, CobwebError, BacktestConfig
from .scoring import (
    score, score_by_id, auto_score, auto_score_by_id,
    list_features, list_plots,
    show_features, show_plots, show_categories,
    FEATURE_CATS, PLOT_CATS,
)
from .utils import (
    save_table,
    fix_timestamps,
    load_csv,
    align,
    to_signals,
    get_plot,
    save_all_plots,
    get_signal,
    print_signal,
)
from .plots import payload_to_figure, payloads_to_figures
from .easy import Pipeline

__all__ = [
    "CobwebSim",
    "CobwebError",
    "BacktestConfig",
    "score",
    "score_by_id",
    "auto_score",
    "auto_score_by_id",
    "list_features",
    "list_plots",
    "show_features",
    "show_plots",
    "show_categories",
    "FEATURE_CATS",
    "PLOT_CATS",
    "save_table",
    "fix_timestamps",
    "load_csv",
    "align",
    "to_signals",
    "get_plot",
    "save_all_plots",
    "get_signal",
    "print_signal",
    "payload_to_figure",
    "payloads_to_figures",
    "Pipeline",
]
