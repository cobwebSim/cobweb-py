"""cobweb-py - Python client for CobwebSim."""

__version__ = "0.3.2"

from .client import CobwebSim, CobwebError, BacktestConfig, from_yfinance, from_alpaca
from .scoring import (
    FEATURES, PLOTS,
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
    get_plot_df,
    save_all_plots,
    get_signal,
    print_signal,
    signal_label,
    format_metrics,
)
from .plots import payload_to_figure, payloads_to_figures, payload_to_df
from .easy import quickstart, Pipeline, DeployableResult
from .sweep import (
    WeightedStrategy,
    RuleStrategy,
    ModelStrategy,
    market_sweep,
    param_sweep,
    SweepRow,
    SweepResult,
    ParamRow,
    ParamSweepResult,
)
from .brokers import (
    AlpacaBroker,
    BaseBroker,
    BrokerOrder,
    PositionSizer,
    FullCash,
    FixedQty,
    PercentOfEquity,
    FixedDollar,
)
from .execution import deploy, get_execution_log, clear_signal_state

__all__ = [
    # Core
    "CobwebSim",
    "CobwebError",
    "BacktestConfig",
    "from_yfinance",
    "from_alpaca",
    # Scoring
    "FEATURES",
    "PLOTS",
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
    # Utilities
    "save_table",
    "fix_timestamps",
    "load_csv",
    "align",
    "to_signals",
    "get_plot",
    "get_plot_df",
    "save_all_plots",
    "get_signal",
    "print_signal",
    "signal_label",
    "format_metrics",
    # Plots
    "payload_to_figure",
    "payloads_to_figures",
    "payload_to_df",
    # Pipeline
    "quickstart",
    "Pipeline",
    "DeployableResult",
    # Brokers
    "AlpacaBroker",
    "BaseBroker",
    "BrokerOrder",
    # Position sizing
    "PositionSizer",
    "FullCash",
    "FixedQty",
    "PercentOfEquity",
    "FixedDollar",
    # Sweep
    "WeightedStrategy",
    "RuleStrategy",
    "ModelStrategy",
    "market_sweep",
    "param_sweep",
    "SweepRow",
    "SweepResult",
    "ParamRow",
    "ParamSweepResult",
    # Execution
    "deploy",
    "get_execution_log",
    "clear_signal_state",
]
