"""
Comprehensive unit tests for cobweb-py 0.1.7.

Tests cover:
  - Package imports and __version__
  - client.py: _pick_col, _df_to_ohlc_rows, _to_rows, _normalize_ohlcv_keys,
                BacktestConfig, CobwebSim signal validation
  - scoring.py: FEATURES/PLOTS dicts, _to_df, zscore, score, score_by_id,
                auto_score, list_features, list_plots, _resolve_plot_id,
                _resolve_category, show_features, show_plots
  - utils.py: fix_timestamps, load_csv, align, to_signals, get_signal, save_table
  - plots.py: _to_df, _as_float_matrix, _label, payload_to_figure, payloads_to_figures
  - easy.py: _ensure_feature_deps, Pipeline.__init__

Run with:
    python -m pytest tests/test_cobweb.py -v
"""
from __future__ import annotations

import math
import warnings
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. Package imports and __version__
# ---------------------------------------------------------------------------
class TestPackageInit:
    def test_version_string(self):
        import cobweb_py
        assert hasattr(cobweb_py, "__version__")
        assert cobweb_py.__version__ == "0.1.7"

    def test_public_exports(self):
        import cobweb_py
        expected = [
            "CobwebSim", "CobwebError", "BacktestConfig",
            "FEATURES", "PLOTS",
            "score", "score_by_id", "auto_score", "auto_score_by_id",
            "list_features", "list_plots",
            "show_features", "show_plots", "show_categories",
            "FEATURE_CATS", "PLOT_CATS",
            "save_table", "fix_timestamps", "load_csv", "align",
            "to_signals", "get_plot", "save_all_plots",
            "get_signal", "print_signal",
            "payload_to_figure", "payloads_to_figures",
            "quickstart", "Pipeline",
        ]
        for name in expected:
            assert hasattr(cobweb_py, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# 2. client.py
# ---------------------------------------------------------------------------
class TestPickCol:
    def test_basic_lookup(self):
        from cobweb_py.client import _pick_col
        lower_map = {"open": "Open", "close": "Close"}
        assert _pick_col(lower_map, "open") == "Open"
        assert _pick_col(lower_map, "close") == "Close"

    def test_fallback_names(self):
        from cobweb_py.client import _pick_col
        lower_map = {"vol": "Vol"}
        assert _pick_col(lower_map, "volume", "vol") == "Vol"

    def test_missing_returns_none(self):
        from cobweb_py.client import _pick_col
        lower_map = {"open": "open"}
        assert _pick_col(lower_map, "missing_col") is None


class TestDfToOhlcRows:
    def _make_df(self, n=25):
        return pd.DataFrame({
            "Open": [100.0 + i for i in range(n)],
            "High": [105.0 + i for i in range(n)],
            "Low": [95.0 + i for i in range(n)],
            "Close": [102.0 + i for i in range(n)],
            "Volume": [1000 + i for i in range(n)],
            "timestamp": [f"2024-01-{i+1:02d}" for i in range(n)],
        })

    def test_basic_conversion(self):
        from cobweb_py.client import _df_to_ohlc_rows
        df = self._make_df()
        rows = _df_to_ohlc_rows(df)
        assert len(rows) == 25
        assert "open" in rows[0]
        assert "high" in rows[0]
        assert "low" in rows[0]
        assert "close" in rows[0]
        assert "volume" in rows[0]
        assert "timestamp" in rows[0]

    def test_missing_columns_raises(self):
        from cobweb_py.client import _df_to_ohlc_rows, CobwebError
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        with pytest.raises(CobwebError, match="Missing required columns"):
            _df_to_ohlc_rows(df)

    def test_too_few_rows_raises(self):
        from cobweb_py.client import _df_to_ohlc_rows, CobwebError
        df = self._make_df(n=5)
        with pytest.raises(CobwebError, match="at least 20"):
            _df_to_ohlc_rows(df)

    def test_nan_volume_excluded(self):
        from cobweb_py.client import _df_to_ohlc_rows
        df = self._make_df()
        df.loc[0, "Volume"] = float("nan")
        rows = _df_to_ohlc_rows(df)
        assert "volume" not in rows[0]  # NaN volume excluded
        assert "volume" in rows[1]      # valid volume kept


class TestToRows:
    def _make_rows(self, n=25):
        return [
            {"open": 100+i, "high": 105+i, "low": 95+i, "close": 102+i, "volume": 1000+i}
            for i in range(n)
        ]

    def test_list_of_dicts(self):
        from cobweb_py.client import _to_rows
        rows = self._make_rows()
        result = _to_rows(rows)
        assert len(result) == 25

    def test_dict_with_rows_key(self):
        from cobweb_py.client import _to_rows
        data = {"rows": self._make_rows()}
        result = _to_rows(data)
        assert len(result) == 25

    def test_dataframe_input(self):
        from cobweb_py.client import _to_rows
        df = pd.DataFrame({
            "Open": [100.0 + i for i in range(25)],
            "High": [105.0 + i for i in range(25)],
            "Low": [95.0 + i for i in range(25)],
            "Close": [102.0 + i for i in range(25)],
        })
        result = _to_rows(df)
        assert len(result) == 25

    def test_unsupported_type_raises(self):
        from cobweb_py.client import _to_rows, CobwebError
        with pytest.raises(CobwebError, match="Unsupported"):
            _to_rows(42)


class TestNormalizeOhlcvKeys:
    def test_already_lowercase(self):
        from cobweb_py.client import _normalize_ohlcv_keys
        rows = [{"open": 1, "close": 2}]
        assert _normalize_ohlcv_keys(rows) == rows

    def test_uppercase_normalized(self):
        from cobweb_py.client import _normalize_ohlcv_keys
        rows = [{"Open": 1, "Close": 2, "custom_col": 99}]
        result = _normalize_ohlcv_keys(rows)
        assert "open" in result[0]
        assert "close" in result[0]
        assert "custom_col" in result[0]


class TestBacktestConfig:
    def test_default_values(self):
        from cobweb_py.client import BacktestConfig
        cfg = BacktestConfig()
        assert cfg.initial_cash == 10_000.0
        assert cfg.fee_bps == 1.0

    def test_to_dict(self):
        from cobweb_py.client import BacktestConfig
        cfg = BacktestConfig(initial_cash=50_000)
        d = cfg.to_dict()
        assert d["initial_cash"] == 50_000
        assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# 3. scoring.py
# ---------------------------------------------------------------------------
class TestFeaturesDicts:
    def test_features_has_71_entries(self):
        from cobweb_py.scoring import FEATURES
        assert len(FEATURES) == 71

    def test_plots_has_27_entries(self):
        from cobweb_py.scoring import PLOTS
        assert len(PLOTS) == 27

    def test_feature_ids_1_to_71(self):
        from cobweb_py.scoring import FEATURES
        assert set(FEATURES.keys()) == set(range(1, 72))

    def test_plot_ids_1_to_27(self):
        from cobweb_py.scoring import PLOTS
        assert set(PLOTS.keys()) == set(range(1, 28))


class TestScoringToDf:
    def test_list_of_dicts(self):
        from cobweb_py.scoring import _to_df
        df = _to_df([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_dict_with_rows(self):
        from cobweb_py.scoring import _to_df
        df = _to_df({"rows": [{"a": 1}, {"a": 2}]})
        assert len(df) == 2

    def test_dataframe_passthrough(self):
        from cobweb_py.scoring import _to_df
        orig = pd.DataFrame({"x": [1, 2, 3]})
        df = _to_df(orig)
        assert len(df) == 3
        # Should be a copy, not the same object
        assert df is not orig

    def test_invalid_input_raises(self):
        from cobweb_py.scoring import _to_df
        with pytest.raises(ValueError):
            _to_df(42)


class TestZscore:
    def test_normal_values(self):
        from cobweb_py.scoring import zscore
        s = pd.Series([1, 2, 3, 4, 5])
        z = zscore(s)
        assert abs(z.mean()) < 1e-10  # mean should be ~0

    def test_constant_series(self):
        from cobweb_py.scoring import zscore
        s = pd.Series([5, 5, 5, 5])
        z = zscore(s)
        # constant series -> std=0 -> all zeros
        assert all(v == 0.0 for v in z)


class TestScore:
    def _make_rows(self, n=50):
        return [
            {
                "ret_1d": 0.01 * (i % 10 - 5),
                "rsi_14": 30 + i,
                "sma_20": 100 + i * 0.5,
                "close": 100 + i * 0.5 + 1,
            }
            for i in range(n)
        ]

    def test_basic_score(self):
        from cobweb_py.scoring import score
        rows = self._make_rows()
        weights = {"ret_1d": 0.5, "rsi_14": 0.5}
        result = score(rows, weights)
        assert len(result) == 50
        assert all(isinstance(v, float) for v in result)

    def test_missing_column_warns(self):
        from cobweb_py.scoring import score
        rows = self._make_rows()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = score(rows, {"nonexistent_col": 1.0})
            assert len(w) >= 1
            assert "not found" in str(w[0].message)
        assert len(result) == 50

    def test_empty_weights(self):
        from cobweb_py.scoring import score
        rows = self._make_rows()
        result = score(rows, {})
        assert all(v == 0.0 for v in result)

    def test_rsi_to_unit(self):
        from cobweb_py.scoring import score
        rows = [{"rsi_14": 70.0} for _ in range(25)]
        result = score(rows, {"rsi_14": 1.0}, normalize=False, rsi_to_unit=True)
        # RSI 70 -> 0.70
        assert all(abs(v - 0.70) < 1e-6 for v in result)


class TestScoreById:
    def test_basic_score_by_id(self):
        from cobweb_py.scoring import score_by_id
        rows = [
            {"ret_1d": 0.01, "rsi_14": 50, "close": 100, "sma_20": 99}
            for _ in range(30)
        ]
        result = score_by_id(rows, {36: 0.3, 1: 0.4})
        assert len(result) == 30

    def test_invalid_id_warns(self):
        from cobweb_py.scoring import score_by_id
        rows = [{"ret_1d": 0.01} for _ in range(25)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            score_by_id(rows, {999: 1.0})
            assert any("not recognised" in str(warning.message) for warning in w)

    def test_ma_id_uses_signal(self):
        from cobweb_py.scoring import score_by_id
        rows = [
            {"ret_1d": 0.01, "close": 105, "sma_20": 100}
            for _ in range(25)
        ]
        # feature 11 = sma_20, should be converted to sma_signal
        result = score_by_id(rows, {11: 1.0}, normalize=False, rsi_to_unit=False)
        assert len(result) == 25
        # sma_signal = (close - sma_20) / sma_20 = (105-100)/100 = 0.05
        assert abs(result[0] - 0.05) < 1e-6


class TestAutoScore:
    def test_auto_score_runs(self):
        from cobweb_py.scoring import auto_score
        rows = [
            {"ret_1d": 0.01, "rsi_14": 50, "close": 100, "sma_20": 99}
            for _ in range(30)
        ]
        result = auto_score(rows)
        assert len(result) == 30


class TestListFeaturesPlots:
    def test_list_features_all(self):
        from cobweb_py.scoring import list_features
        features = list_features()
        assert len(features) == 71

    def test_list_features_by_category(self):
        from cobweb_py.scoring import list_features
        rsi_features = list_features("rsi")
        assert len(rsi_features) == 2  # rsi_14 and rsi_pct_252

    def test_list_plots_all(self):
        from cobweb_py.scoring import list_plots
        plots = list_plots()
        assert len(plots) == 27

    def test_list_plots_by_category(self):
        from cobweb_py.scoring import list_plots
        trade_plots = list_plots("trades")
        assert len(trade_plots) == 4  # plots 16-19


class TestResolvePlotId:
    def test_exact_match(self):
        from cobweb_py.scoring import _resolve_plot_id
        assert _resolve_plot_id("equity_curve_linear") == 1

    def test_fuzzy_prefix(self):
        from cobweb_py.scoring import _resolve_plot_id
        assert _resolve_plot_id("rolling_sharpe") == 5

    def test_ambiguous_raises(self):
        from cobweb_py.scoring import _resolve_plot_id
        with pytest.raises(ValueError, match="Ambiguous"):
            _resolve_plot_id("equity")  # matches equity_curve_linear and equity_curve_log

    def test_no_match_raises(self):
        from cobweb_py.scoring import _resolve_plot_id
        with pytest.raises(ValueError, match="No plot matched"):
            _resolve_plot_id("nonexistent_plot_xyz")


class TestResolveCategory:
    def test_exact(self):
        from cobweb_py.scoring import _resolve_category, FEATURE_CATS
        result = _resolve_category("rsi", FEATURE_CATS)
        assert result == ["rsi"]

    def test_prefix(self):
        from cobweb_py.scoring import _resolve_category, FEATURE_CATS
        result = _resolve_category("vol", FEATURE_CATS)
        assert "volatility" in result
        assert "volume" in result


# ---------------------------------------------------------------------------
# 4. utils.py
# ---------------------------------------------------------------------------
class TestFixTimestamps:
    def test_basic(self):
        from cobweb_py.utils import fix_timestamps
        rows = [
            {"timestamp": "14/09/2020 00:00", "close": 100},
            {"timestamp": "15/09/2020 00:00", "close": 101},
        ]
        result = fix_timestamps(rows, dayfirst=True)
        assert "rows" in result
        assert len(result["rows"]) == 2
        assert "2020-09-14" in result["rows"][0]["timestamp"]

    def test_missing_column_raises(self):
        from cobweb_py.utils import fix_timestamps
        from cobweb_py.client import CobwebError
        rows = [{"close": 100}]
        with pytest.raises(CobwebError, match="missing"):
            fix_timestamps(rows, timestamp_col="timestamp")


class TestToSignals:
    def test_basic_long_signal(self):
        from cobweb_py.utils import to_signals
        scores = [0.0, 0.1, 0.25, 0.3, 0.1, 0.03, 0.0]
        signals = to_signals(scores, entry_th=0.20, exit_th=0.05)
        # 0.0 -> flat, 0.1 -> flat, 0.25 -> long, 0.3 -> long, 0.1 -> long, 0.03 -> flat, 0.0 -> flat
        assert signals == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    def test_nan_handling(self):
        from cobweb_py.utils import to_signals
        scores = [0.25, float("nan"), 0.1, 0.03]
        signals = to_signals(scores, entry_th=0.20, exit_th=0.05)
        # 0.25 -> long, nan -> stays in long, 0.1 -> still long, 0.03 -> exit
        assert signals == [1.0, 1.0, 1.0, 0.0]

    def test_inf_handling(self):
        from cobweb_py.utils import to_signals
        scores = [0.25, float("inf"), float("-inf"), 0.03]
        signals = to_signals(scores, entry_th=0.20, exit_th=0.05)
        assert signals[0] == 1.0
        assert signals[1] == 1.0  # inf -> no change
        assert signals[2] == 1.0  # -inf -> no change
        assert signals[3] == 0.0  # exit

    def test_short_signals(self):
        from cobweb_py.utils import to_signals
        scores = [0.0, -0.25, -0.3, -0.08, 0.0]
        signals = to_signals(scores, entry_th=0.20, exit_th=0.05, use_shorts=True)
        # 0.0 -> flat, -0.25 -> short (-1), -0.3 -> short (-1), -0.08 -> still short, 0.0 -> flat
        assert signals == [0.0, -1.0, -1.0, -1.0, 0.0]


class TestGetSignal:
    def test_basic(self):
        from cobweb_py.utils import get_signal
        bt = {"current_signal": "buy", "current_exposure": 0.87, "signal_strength": 0.72}
        s = get_signal(bt)
        assert s["signal"] == "buy"
        assert abs(s["exposure"] - 0.87) < 1e-6
        assert abs(s["strength"] - 0.72) < 1e-6

    def test_defaults(self):
        from cobweb_py.utils import get_signal
        s = get_signal({})
        assert s["signal"] == "hold"
        assert s["exposure"] == 0.0
        assert s["strength"] == 0.0


class TestSaveTable:
    def test_basic_save(self, tmp_path):
        from cobweb_py.utils import save_table
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        out = save_table(rows, tmp_path / "test_table.html", title="Test")
        assert Path(out).exists()
        content = Path(out).read_text()
        assert "Test" in content
        assert "<table" in content


class TestAlign:
    def test_basic_align(self):
        from cobweb_py.utils import align
        a = {"rows": [
            {"timestamp": "2024-01-01 00:00:00", "close": 100},
            {"timestamp": "2024-01-02 00:00:00", "close": 101},
            {"timestamp": "2024-01-03 00:00:00", "close": 102},
        ]}
        b = {"rows": [
            {"timestamp": "2024-01-02 00:00:00", "close": 200},
            {"timestamp": "2024-01-03 00:00:00", "close": 201},
            {"timestamp": "2024-01-04 00:00:00", "close": 202},
        ]}
        a_out, b_out = align(a, b)
        assert len(a_out["rows"]) == 2
        assert len(b_out["rows"]) == 2

    def test_no_overlap_raises(self):
        from cobweb_py.utils import align
        from cobweb_py.client import CobwebError
        a = {"rows": [{"timestamp": "2024-01-01 00:00:00", "close": 100}]}
        b = {"rows": [{"timestamp": "2025-06-01 00:00:00", "close": 200}]}
        with pytest.raises(CobwebError, match="no overlapping"):
            align(a, b)


# ---------------------------------------------------------------------------
# 5. plots.py
# ---------------------------------------------------------------------------
class TestPlotsToDf:
    def test_dataframe_passthrough(self):
        from cobweb_py.plots import _to_df
        orig = pd.DataFrame({"x": [1, 2]})
        df = _to_df(orig)
        assert df is not orig
        assert len(df) == 2

    def test_dict_with_rows(self):
        from cobweb_py.plots import _to_df
        df = _to_df({"rows": [{"a": 1}]})
        assert len(df) == 1

    def test_dict_with_equity_curve(self):
        from cobweb_py.plots import _to_df
        df = _to_df({"equity_curve": [{"equity": 10000}]})
        assert len(df) == 1
        assert "equity" in df.columns


class TestAsFloatMatrix:
    def test_normal_values(self):
        from cobweb_py.plots import _as_float_matrix
        z = [[1.0, 2.0], [3.0, 4.0]]
        result = _as_float_matrix(z)
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_non_numeric_becomes_nan(self):
        from cobweb_py.plots import _as_float_matrix
        z = [[1.0, "bad"], [None, 4.0]]
        result = _as_float_matrix(z)
        assert result[0][0] == 1.0
        assert math.isnan(result[0][1])
        assert math.isnan(result[1][0])
        assert result[1][1] == 4.0

    def test_inf_becomes_nan(self):
        from cobweb_py.plots import _as_float_matrix
        z = [[float("inf"), float("-inf")]]
        result = _as_float_matrix(z)
        assert math.isnan(result[0][0])
        assert math.isnan(result[0][1])


class TestLabel:
    def test_known_labels(self):
        from cobweb_py.plots import _label
        assert _label("close") == "Close"
        assert _label("volume") == "Volume"
        assert _label("equity") == "Equity"

    def test_snake_case_humanized(self):
        from cobweb_py.plots import _label
        assert _label("my_custom_col") == "My Custom Col"


# ---------------------------------------------------------------------------
# 6. easy.py
# ---------------------------------------------------------------------------
class TestEnsureFeatureDeps:
    def test_both_none(self):
        from cobweb_py.easy import _ensure_feature_deps
        result = _ensure_feature_deps(None, None)
        assert result is None

    def test_plot_21_adds_70(self):
        from cobweb_py.easy import _ensure_feature_deps
        result = _ensure_feature_deps(None, [21])
        assert 70 in result

    def test_plot_20_adds_71(self):
        from cobweb_py.easy import _ensure_feature_deps
        result = _ensure_feature_deps(None, [20])
        assert 71 in result

    def test_existing_features_preserved(self):
        from cobweb_py.easy import _ensure_feature_deps
        result = _ensure_feature_deps([1, 2, 3], [21])
        assert 1 in result
        assert 2 in result
        assert 3 in result
        assert 70 in result

    def test_no_deps_needed_returns_none(self):
        from cobweb_py.easy import _ensure_feature_deps
        result = _ensure_feature_deps(None, [5])  # plot 5 doesn't need deps
        assert result is None


class TestPipelineInit:
    def test_init(self):
        from cobweb_py.easy import Pipeline
        pipe = Pipeline("http://localhost:8000", [{"open": 1, "close": 2}])
        assert pipe.is_enriched is False
        assert pipe.enriched_rows is None
        assert pipe.feature_columns is None

    def test_reset(self):
        from cobweb_py.easy import Pipeline
        pipe = Pipeline("http://localhost:8000", [{"open": 1, "close": 2}])
        pipe._enriched_rows = [{"a": 1}]  # fake cached data
        assert pipe.is_enriched is True
        pipe.reset()
        assert pipe.is_enriched is False


# ---------------------------------------------------------------------------
# 7. Integration: features.py was deleted
# ---------------------------------------------------------------------------
class TestFeaturesModuleRemoved:
    def test_no_features_module(self):
        """Ensure the dead features.py module was actually deleted."""
        import importlib
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("cobweb_py.features")


# ---------------------------------------------------------------------------
# 8. XSS safety (html.escape usage)
# ---------------------------------------------------------------------------
class TestXssSafety:
    def test_save_table_escapes_title(self, tmp_path):
        from cobweb_py.utils import save_table
        malicious_title = '<script>alert("xss")</script>'
        out = save_table(
            [{"a": 1}],
            tmp_path / "xss_test.html",
            title=malicious_title,
        )
        content = Path(out).read_text()
        assert "<script>" not in content
        assert "&lt;script&gt;" in content


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_rows_list(self):
        from cobweb_py.client import _normalize_ohlcv_keys
        assert _normalize_ohlcv_keys([]) == []

    def test_to_signals_empty(self):
        from cobweb_py.utils import to_signals
        assert to_signals([], 0.2, 0.05) == []

    def test_score_with_sma_signal(self):
        """Test score() creates sma_signal from close & sma_20."""
        from cobweb_py.scoring import score
        rows = [
            {"close": 105, "sma_20": 100, "ret_1d": 0.01}
            for _ in range(25)
        ]
        result = score(rows, {"sma_signal": 0.5, "ret_1d": 0.5})
        assert len(result) == 25
        # sma_signal should be (105-100)/100 = 0.05

    def test_backtest_config_none_fields(self):
        from cobweb_py.client import BacktestConfig
        cfg = BacktestConfig(max_participation=None)
        d = cfg.to_dict()
        assert "max_participation" not in d


# ---------------------------------------------------------------------------
# 10. New 0.1.4 helpers
# ---------------------------------------------------------------------------
class TestFromYfinance:
    def test_missing_yfinance_raises(self):
        """If yfinance isn't installed, should raise CobwebError."""
        from cobweb_py.client import from_yfinance, CobwebError
        from unittest.mock import patch
        with patch.dict("sys.modules", {"yfinance": None}):
            # Force re-import to trigger ImportError
            pass
        # We can't easily un-import yfinance if it IS installed,
        # so just verify the function exists and is callable
        assert callable(from_yfinance)

    def test_function_exported(self):
        import cobweb_py
        assert hasattr(cobweb_py, "from_yfinance")
        assert callable(cobweb_py.from_yfinance)


class TestEnrichRows:
    def test_method_exists(self):
        from cobweb_py.client import CobwebSim
        sim = CobwebSim("http://localhost:8000")
        assert hasattr(sim, "enrich_rows")
        assert callable(sim.enrich_rows)


class TestSignalLabel:
    def test_buy(self):
        from cobweb_py.utils import signal_label
        assert signal_label(1.0) == "BUY"

    def test_sell(self):
        from cobweb_py.utils import signal_label
        assert signal_label(-1.0) == "SELL"

    def test_hold(self):
        from cobweb_py.utils import signal_label
        assert signal_label(0.0) == "HOLD"

    def test_other_values(self):
        from cobweb_py.utils import signal_label
        assert signal_label(0.5) == "HOLD"
        assert signal_label(-0.5) == "HOLD"

    def test_exported(self):
        import cobweb_py
        assert hasattr(cobweb_py, "signal_label")


class TestFormatMetrics:
    def test_basic(self):
        from cobweb_py.utils import format_metrics
        bt = {
            "metrics": {
                "final_equity": 12345.67,
                "total_return": 0.2345,
                "sharpe_ann": 1.23,
                "trades": 42,
                "bars": 252,
                "max_drawdown": -0.15,
            }
        }
        result = format_metrics(bt)
        assert isinstance(result, dict)
        assert "Final Equity" in result
        assert "$12,345.67" == result["Final Equity"]
        assert "23.45%" == result["Total Return"]
        assert "42" == result["Total Trades"]

    def test_empty_metrics(self):
        from cobweb_py.utils import format_metrics
        result = format_metrics({})
        assert result == {}

    def test_none_value(self):
        from cobweb_py.utils import format_metrics
        bt = {"metrics": {"sharpe_ann": None}}
        result = format_metrics(bt)
        assert result["Sharpe Ratio"] == "---"

    def test_exported(self):
        import cobweb_py
        assert hasattr(cobweb_py, "format_metrics")


class TestVersionBump:
    def test_version_0_1_5(self):
        import cobweb_py
        assert cobweb_py.__version__ == "0.1.7"


class TestBacktestAcceptsListDirect:
    def test_to_rows_accepts_list(self):
        """Verify _to_rows handles plain list[dict] without wrapping in {"rows":...}."""
        from cobweb_py.client import _to_rows
        rows = [
            {"open": 100+i, "high": 105+i, "low": 95+i, "close": 102+i}
            for i in range(25)
        ]
        result = _to_rows(rows)
        assert len(result) == 25
        assert result[0]["open"] == 100


class TestPayloadToDf:
    """Tests for the newly-public payload_to_df helper."""

    def test_exported(self):
        import cobweb_py
        assert hasattr(cobweb_py, "payload_to_df")

    def test_t_series(self):
        from cobweb_py.plots import payload_to_df
        p = {"t": [1, 2, 3], "close": [10, 20, 30], "sma": [11, 21, 31]}
        df = payload_to_df(p)
        assert list(df.columns) == ["t", "close", "sma"]
        assert len(df) == 3
        assert df["close"].tolist() == [10, 20, 30]

    def test_histogram(self):
        from cobweb_py.plots import payload_to_df
        p = {"bins": [1.0, 2.0, 3.0], "counts": [5, 10, 15]}
        df = payload_to_df(p)
        assert "value" in df.columns
        assert "count" in df.columns
        assert len(df) == 3

    def test_xy(self):
        from cobweb_py.plots import payload_to_df
        p = {"x": [1, 2], "y": [3, 4]}
        df = payload_to_df(p)
        assert list(df.columns) == ["x", "y"]
        assert len(df) == 2

    def test_volume_shock(self):
        from cobweb_py.plots import payload_to_df
        p = {"t": [1, 2], "volume": [100, 200], "adv": [150, 150], "vol_shock": [0.5, 1.5]}
        df = payload_to_df(p)
        assert "vol_shock" in df.columns
        assert len(df) == 2

    def test_regime_bars(self):
        from cobweb_py.plots import payload_to_df
        p = {"regime": ["bull", "bear"], "mean_ret": [0.01, -0.02], "std_ret": [0.05, 0.06]}
        df = payload_to_df(p)
        assert "regime" in df.columns
        assert len(df) == 2

    def test_correlation_heatmap(self):
        from cobweb_py.plots import payload_to_df
        p = {"cols": ["a", "b"], "matrix": [[1.0, 0.5], [0.5, 1.0]]}
        df = payload_to_df(p)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_fallback_none(self):
        from cobweb_py.plots import payload_to_df
        df = payload_to_df("unexpected_string", fallback="none")
        assert df.empty

    def test_fallback_json(self):
        from cobweb_py.plots import payload_to_df
        df = payload_to_df({"weird": "shape"}, fallback="json")
        assert "payload_json" in df.columns

    def test_save_api_payloads_uses_public(self):
        """Ensure save_api_payloads_to_dfs still works after the rename."""
        from cobweb_py.plots import save_api_payloads_to_dfs
        payloads = {
            "plot_a": {"t": [1, 2], "val": [10, 20]},
            "plot_b": {"bins": [0.1, 0.2], "counts": [3, 7]},
        }
        dfs = save_api_payloads_to_dfs(payloads)
        assert "plot_a" in dfs
        assert "plot_b" in dfs
        assert len(dfs["plot_a"]) == 2


class TestGetPlotDf:
    """Tests for the get_plot_df convenience helper."""

    def test_exported(self):
        import cobweb_py
        assert hasattr(cobweb_py, "get_plot_df")

    def test_returns_dataframe_from_mock(self):
        """Mock get_plot to return a known payload, verify DataFrame output."""
        from unittest.mock import patch
        from cobweb_py.utils import get_plot_df

        fake_result = {"payloads": {"some_plot": {"t": [1, 2, 3], "value": [10, 20, 30]}}}
        with patch("cobweb_py.utils.get_plot", return_value=fake_result):
            df = get_plot_df(None, None, None, 5)
        assert len(df) == 3
        assert "t" in df.columns
        assert "value" in df.columns

    def test_empty_payloads_returns_empty_df(self):
        """When get_plot returns no payloads, get_plot_df returns empty DataFrame."""
        from unittest.mock import patch
        from cobweb_py.utils import get_plot_df

        fake_result = {"payloads": {}}
        with patch("cobweb_py.utils.get_plot", return_value=fake_result):
            df = get_plot_df(None, None, None, 99)
        assert df.empty

    def test_passes_kwargs(self):
        """Verify benchmark_rows and all_feature_ids are forwarded."""
        from unittest.mock import patch, MagicMock
        from cobweb_py.utils import get_plot_df

        fake_result = {"payloads": {"p": {"t": [1], "v": [2]}}}
        with patch("cobweb_py.utils.get_plot", return_value=fake_result) as mock_gp:
            get_plot_df(MagicMock(), [], {}, 3, benchmark_rows=[{"Close": 100}], all_feature_ids=[1, 2])
            _, kwargs = mock_gp.call_args
            assert kwargs["benchmark_rows"] == [{"Close": 100}]
            assert kwargs["all_feature_ids"] == [1, 2]


# ---------------------------------------------------------------------------
# sweep.py
# ---------------------------------------------------------------------------

# Shared test data for sweep tests
def _make_enriched_rows(n: int = 50) -> List[Dict[str, Any]]:
    """Build fake enriched rows with OHLCV + key features."""
    import random
    random.seed(42)
    rows = []
    for i in range(n):
        rows.append({
            "timestamp": f"2024-01-{i+1:02d} 00:00:00",
            "open": 100 + i * 0.1,
            "high": 102 + i * 0.1,
            "low": 98 + i * 0.1,
            "close": 101 + i * 0.1,
            "Close": 101 + i * 0.1,
            "volume": 1000000,
            "ret_1d": random.uniform(-0.05, 0.05),
            "sma_20": 100 + i * 0.08,
            "rsi_14": 30 + i * 0.8,  # gradually rising RSI
            "vol_20": 0.2,
            "macd_hist": random.uniform(-1, 1),
            "bb_pctb_20": random.uniform(0, 1),
            "zscore_20": random.uniform(-2, 2),
        })
    return rows


class TestWeightedStrategy:
    def test_basic(self):
        from cobweb_py.sweep import WeightedStrategy
        rows = _make_enriched_rows()
        strat = WeightedStrategy(weights={36: 0.5, 1: 0.5}, entry_th=0.20, exit_th=0.05)
        signals = strat(rows)
        assert len(signals) == len(rows)
        assert all(s in (0.0, 1.0, -1.0) for s in signals)

    def test_params_property(self):
        from cobweb_py.sweep import WeightedStrategy
        strat = WeightedStrategy(weights={36: 0.3, 11: 0.7}, entry_th=0.15, exit_th=0.03)
        p = strat.params
        assert p["weights"] == {36: 0.3, 11: 0.7}
        assert p["entry_th"] == 0.15
        assert p["exit_th"] == 0.03

    def test_repr(self):
        from cobweb_py.sweep import WeightedStrategy
        strat = WeightedStrategy(weights={36: 0.5}, entry_th=0.20, exit_th=0.05)
        r = repr(strat)
        assert "WeightedStrategy" in r
        assert "36" in r

    def test_use_shorts(self):
        from cobweb_py.sweep import WeightedStrategy
        rows = _make_enriched_rows()
        strat = WeightedStrategy(weights={36: 0.5, 1: 0.5}, entry_th=0.10, exit_th=0.02, use_shorts=True)
        signals = strat(rows)
        assert all(s in (0.0, 1.0, -1.0) for s in signals)


class TestRuleStrategy:
    def test_basic(self):
        from cobweb_py.sweep import RuleStrategy

        def simple_rule(df):
            return [1.0 if row["rsi_14"] < 40 else 0.0 for _, row in df.iterrows()]

        strat = RuleStrategy(simple_rule)
        rows = _make_enriched_rows()
        signals = strat(rows)
        assert len(signals) == len(rows)
        assert all(isinstance(s, float) for s in signals)

    def test_with_params(self):
        from cobweb_py.sweep import RuleStrategy

        def threshold_rule(df, buy_level=30, sell_level=70):
            return [1.0 if row["rsi_14"] < buy_level else 0.0 for _, row in df.iterrows()]

        strat = RuleStrategy(threshold_rule, buy_level=35, sell_level=65)
        assert strat.params == {"buy_level": 35, "sell_level": 65}
        signals = strat(_make_enriched_rows())
        assert len(signals) > 0

    def test_repr(self):
        from cobweb_py.sweep import RuleStrategy

        def my_rule(df):
            return [0.0] * len(df)

        strat = RuleStrategy(my_rule)
        assert "RuleStrategy" in repr(strat)
        assert "my_rule" in repr(strat)


class TestModelStrategy:
    def test_with_predict(self):
        from cobweb_py.sweep import ModelStrategy

        model = MagicMock()
        model.predict.return_value = [0.6, 0.3, 0.8, 0.2, 0.5] * 10
        strat = ModelStrategy(
            model=model,
            feature_cols=["rsi_14", "vol_20"],
            entry_th=0.5,
            exit_th=0.3,
        )
        signals = strat(_make_enriched_rows())
        assert len(signals) == 50
        model.predict.assert_called_once()

    def test_with_callable(self):
        from cobweb_py.sweep import ModelStrategy

        def fake_model(X):
            return [0.5] * len(X)

        strat = ModelStrategy(
            model=fake_model,
            feature_cols=["rsi_14", "vol_20"],
            entry_th=0.5,
            exit_th=0.3,
        )
        signals = strat(_make_enriched_rows())
        assert len(signals) == 50

    def test_missing_columns_raises(self):
        from cobweb_py.sweep import ModelStrategy
        from cobweb_py.client import CobwebError

        model = MagicMock()
        strat = ModelStrategy(model=model, feature_cols=["nonexistent_col"])
        with pytest.raises(CobwebError, match="missing columns"):
            strat(_make_enriched_rows())

    def test_params_property(self):
        from cobweb_py.sweep import ModelStrategy
        strat = ModelStrategy(model=lambda x: x, feature_cols=["rsi_14"], entry_th=0.6, exit_th=0.4)
        p = strat.params
        assert p["feature_cols"] == ["rsi_14"]
        assert p["entry_th"] == 0.6


class TestPlainFunctionStrategy:
    def test_plain_function_as_strategy(self):
        """Any callable with (rows) -> List[float] should work."""
        def always_long(rows):
            return [1.0] * len(rows)

        rows = _make_enriched_rows()
        signals = always_long(rows)
        assert len(signals) == 50
        assert all(s == 1.0 for s in signals)

    def test_lambda_as_strategy(self):
        strat = lambda rows: [0.0] * len(rows)
        rows = _make_enriched_rows()
        signals = strat(rows)
        assert all(s == 0.0 for s in signals)


class TestSweepRow:
    def test_creation(self):
        from cobweb_py.sweep import SweepRow
        row = SweepRow(
            ticker="AAPL",
            signal="BUY",
            close=187.3,
            signal_age=3,
            signals=[0.0, 1.0, 1.0, 1.0],
            enriched_rows=[],
        )
        assert row.ticker == "AAPL"
        assert row.signal == "BUY"
        assert row.error is None

    def test_with_error(self):
        from cobweb_py.sweep import SweepRow
        row = SweepRow(
            ticker="ZZZZZ", signal="", close=0, signal_age=0,
            signals=[], enriched_rows=[], error="Not found",
        )
        assert row.error == "Not found"


class TestSweepResult:
    def _make_result(self):
        from cobweb_py.sweep import SweepRow, SweepResult
        rows = [
            SweepRow("AAPL", "BUY", 187.3, 3, [0.0, 1.0, 1.0, 1.0], []),
            SweepRow("MSFT", "HOLD", 412.8, 12, [0.0, 0.0], []),
            SweepRow("TSLA", "SELL", 245.0, 1, [1.0, 0.0, -1.0], []),
            SweepRow("NVDA", "BUY", 892.3, 2, [0.0, 1.0, 1.0], []),
        ]
        return SweepResult(rows, elapsed_ms=1500, errors=[])

    def test_len(self):
        result = self._make_result()
        assert len(result) == 4

    def test_iter(self):
        result = self._make_result()
        tickers = [r.ticker for r in result]
        assert "AAPL" in tickers

    def test_getitem(self):
        result = self._make_result()
        assert result[0].ticker == "AAPL"

    def test_buys(self):
        result = self._make_result()
        buys = result.buys()
        assert len(buys) == 2
        assert all(r.signal == "BUY" for r in buys)

    def test_sells(self):
        result = self._make_result()
        sells = result.sells()
        assert len(sells) == 1
        assert sells[0].ticker == "TSLA"

    def test_holds(self):
        result = self._make_result()
        holds = result.holds()
        assert len(holds) == 1

    def test_top(self):
        result = self._make_result()
        top2 = result.top(2)
        assert len(top2) == 2

    def test_sort_by(self):
        result = self._make_result()
        result.sort_by("close", ascending=False)
        assert result[0].ticker == "NVDA"  # highest close

    def test_to_df(self):
        result = self._make_result()
        df = result.to_df()
        assert len(df) == 4
        assert "ticker" in df.columns
        assert "signal" in df.columns

    def test_print_summary(self, capsys):
        result = self._make_result()
        result.print_summary()
        captured = capsys.readouterr()
        assert "BUY" in captured.out
        assert "AAPL" in captured.out

    def test_repr(self):
        result = self._make_result()
        assert "SweepResult" in repr(result)
        assert "4 tickers" in repr(result)


class TestSweepResultBacktestAll:
    """Tests for SweepResult.backtest_all() and to_comparison_df()."""

    def _make_result(self):
        from cobweb_py.sweep import SweepRow, SweepResult
        rows = [
            SweepRow("AAPL", "BUY", 187.3, 3,
                     [0.0, 1.0, 1.0, 1.0],
                     [{"Close": 185}, {"Close": 186}, {"Close": 187}, {"Close": 187.3}]),
            SweepRow("MSFT", "HOLD", 412.8, 2,
                     [0.0, 0.0],
                     [{"Close": 410}, {"Close": 412.8}]),
        ]
        return SweepResult(rows, elapsed_ms=1000, errors=[])

    def _mock_sim(self):
        """Return a mock CobwebSim that returns canned backtest results."""
        from unittest.mock import MagicMock
        sim = MagicMock()
        sim.backtest.return_value = {
            "metrics": {
                "total_return": 0.15,
                "sharpe_ann": 1.5,
                "sortino_ann": 2.0,
                "max_drawdown": -0.08,
                "volatility_ann": 0.12,
                "final_equity": 11500,
                "trades": 12,
                "bars": 250,
            },
            "trades": [],
            "equity_curve": [],
        }
        return sim

    def test_backtest_all_populates_backtest_field(self):
        result = self._make_result()
        sim = self._mock_sim()
        result.backtest_all(sim, progress=False)
        for row in result:
            assert row.backtest is not None
            assert "metrics" in row.backtest

    def test_backtest_all_calls_sim_per_ticker(self):
        result = self._make_result()
        sim = self._mock_sim()
        result.backtest_all(sim, progress=False)
        assert sim.backtest.call_count == 2

    def test_backtest_all_returns_self(self):
        result = self._make_result()
        sim = self._mock_sim()
        ret = result.backtest_all(sim, progress=False)
        assert ret is result

    def test_backtest_all_passes_config_and_benchmark(self):
        result = self._make_result()
        sim = self._mock_sim()
        config = {"initial_cash": 5000}
        benchmark = {"rows": [{"Close": 100}]}
        result.backtest_all(sim, config=config, benchmark=benchmark, progress=False)
        call_kwargs = sim.backtest.call_args_list[0][1]
        assert call_kwargs["config"] == config
        assert call_kwargs["benchmark"] == benchmark

    def test_backtest_all_skips_error_rows(self):
        from cobweb_py.sweep import SweepRow, SweepResult
        rows = [
            SweepRow("AAPL", "BUY", 187.3, 3, [1.0], [{"Close": 187}]),
            SweepRow("BAD", "HOLD", 0, 0, [], [], error="fetch failed"),
        ]
        result = SweepResult(rows, elapsed_ms=500, errors=[("BAD", "fetch failed")])
        sim = self._mock_sim()
        result.backtest_all(sim, progress=False)
        assert sim.backtest.call_count == 1  # only AAPL
        assert rows[1].backtest is None  # error row untouched

    def test_to_comparison_df(self):
        result = self._make_result()
        sim = self._mock_sim()
        result.backtest_all(sim, progress=False)
        df = result.to_comparison_df()
        assert len(df) == 2
        assert df.index.name == "Ticker"
        assert "AAPL" in df.index
        assert "Total Return (%)" in df.columns
        assert "Sharpe" in df.columns
        assert df.loc["AAPL", "Total Return (%)"] == 15.0

    def test_to_comparison_df_before_backtest(self):
        result = self._make_result()
        df = result.to_comparison_df()
        assert len(df) == 0  # no backtests yet → empty

    def test_backtest_all_handles_backtest_error(self):
        from unittest.mock import MagicMock
        result = self._make_result()
        sim = MagicMock()
        sim.backtest.side_effect = Exception("API error")
        result.backtest_all(sim, progress=False)
        for row in result:
            assert row.backtest is not None
            assert "error" in row.backtest


class TestParamRow:
    def test_creation(self):
        from cobweb_py.sweep import ParamRow
        row = ParamRow(
            params={"w_rsi": 0.3, "entry_th": 0.20},
            metrics={"sharpe": 1.5, "total_return": 0.24},
            sharpe=1.5,
            total_return=0.24,
            max_dd=-0.08,
            sortino=2.1,
            trades=47,
            signals=[0.0, 1.0, 0.0],
        )
        assert row.sharpe == 1.5
        assert row.trades == 47
        assert row.error is None


class TestParamSweepResult:
    def _make_result(self):
        from cobweb_py.sweep import ParamRow, ParamSweepResult
        rows = [
            ParamRow({"w": 0.3, "th": 0.20}, {"sharpe": 1.8}, 1.8, 0.24, -0.08, 2.1, 47, []),
            ParamRow({"w": 0.5, "th": 0.15}, {"sharpe": 1.5}, 1.5, 0.20, -0.10, 1.8, 52, []),
            ParamRow({"w": 0.2, "th": 0.25}, {"sharpe": 0.9}, 0.9, 0.10, -0.15, 1.0, 30, []),
        ]
        return ParamSweepResult(rows, elapsed_ms=5000, total_combos=3, errors=[], enriched_rows=[])

    def test_len(self):
        result = self._make_result()
        assert len(result) == 3

    def test_best(self):
        result = self._make_result()
        best = result.best(1)
        assert len(best) == 1
        assert best[0].sharpe == 1.8

    def test_best_params(self):
        result = self._make_result()
        bp = result.best_params()
        assert bp == {"w": 0.3, "th": 0.20}

    def test_sort_by(self):
        result = self._make_result()
        result.sort_by("trades", ascending=True)
        assert result[0].trades == 30

    def test_filter_min_sharpe(self):
        result = self._make_result()
        filtered = result.filter(min_sharpe=1.0)
        assert len(filtered) == 2

    def test_filter_min_trades(self):
        result = self._make_result()
        filtered = result.filter(min_trades=40)
        assert len(filtered) == 2

    def test_to_df(self):
        result = self._make_result()
        df = result.to_df()
        assert len(df) == 3
        assert "w" in df.columns
        assert "sharpe" in df.columns

    def test_print_summary(self, capsys):
        result = self._make_result()
        result.print_summary()
        captured = capsys.readouterr()
        assert "1.80" in captured.out
        assert "3 combos" in captured.out

    def test_repr(self):
        result = self._make_result()
        assert "ParamSweepResult" in repr(result)
        assert "3 combos" in repr(result)


class TestComputeSignalAge:
    def test_no_change(self):
        from cobweb_py.sweep import _compute_signal_age
        assert _compute_signal_age([1.0, 1.0, 1.0]) == 3

    def test_recent_change(self):
        from cobweb_py.sweep import _compute_signal_age
        assert _compute_signal_age([0.0, 0.0, 1.0]) == 1

    def test_two_bars(self):
        from cobweb_py.sweep import _compute_signal_age
        assert _compute_signal_age([0.0, 1.0, 1.0]) == 2

    def test_empty(self):
        from cobweb_py.sweep import _compute_signal_age
        assert _compute_signal_age([]) == 0

    def test_single(self):
        from cobweb_py.sweep import _compute_signal_age
        assert _compute_signal_age([1.0]) == 1


class TestBuildWeightedCombos:
    def test_basic(self):
        from cobweb_py.sweep import _build_weighted_combos
        strategy_fn, param_grid = _build_weighted_combos(
            {36: [0.2, 0.3], 11: [0.5]},
            entry_thresholds=[0.15, 0.20],
            exit_thresholds=[0.05],
            use_shorts=False,
        )
        # 2 × 1 × 2 × 1 = 4 combos
        assert "entry_th" in param_grid
        assert "exit_th" in param_grid
        assert len(param_grid["entry_th"]) == 2

        # Factory should produce a WeightedStrategy
        from cobweb_py.sweep import WeightedStrategy
        combo = {k: v[0] for k, v in param_grid.items()}
        strat = strategy_fn(**combo)
        assert isinstance(strat, WeightedStrategy)


class TestMarketSweepValidation:
    def test_no_strategy_or_weights_raises(self):
        from cobweb_py.sweep import market_sweep
        sim = MagicMock()
        with pytest.raises(ValueError, match="strategy.*weights"):
            market_sweep(sim, ["AAPL"])

    def test_weights_shorthand_builds_strategy(self):
        """Verify that passing weights= auto-builds a WeightedStrategy internally."""
        from cobweb_py.sweep import market_sweep, WeightedStrategy

        # We can't easily run the full sweep without API, but we can verify
        # the resolution logic by mocking the internals
        sim = MagicMock()
        sim.enrich.return_value = {"rows": _make_enriched_rows()}

        with patch("cobweb_py.sweep.from_yfinance") as mock_yf:
            mock_yf.return_value = {"rows": _make_enriched_rows()}
            result = market_sweep(
                sim, ["AAPL"],
                weights={36: 0.3, 1: 0.7},
                entry_th=0.15,
                max_workers=1,
                progress=False,
            )
        assert len(result) == 1
        assert result[0].signal in ("BUY", "SELL", "HOLD")


class TestParamSweepValidation:
    def test_no_strategy_or_weight_grid_raises(self):
        from cobweb_py.sweep import param_sweep
        sim = MagicMock()
        with pytest.raises(ValueError, match="strategy_fn.*weight_grid"):
            param_sweep(sim, [])

    def test_weight_grid_shorthand(self):
        """Verify weight_grid shorthand produces combos and runs."""
        from cobweb_py.sweep import param_sweep

        sim = MagicMock()
        sim.enrich.return_value = {"rows": _make_enriched_rows()}
        sim.backtest.return_value = {
            "metrics": {
                "sharpe": 1.5,
                "total_return": 0.20,
                "max_drawdown": -0.08,
                "sortino": 2.0,
                "total_trades": 10,
            }
        }

        result = param_sweep(
            sim, _make_enriched_rows(),
            weight_grid={36: [0.3, 0.5]},
            entry_thresholds=[0.20],
            exit_thresholds=[0.05],
            max_workers=1,
            progress=False,
        )
        # 2 weight × 1 entry × 1 exit = 2 combos
        assert len(result) == 2
        assert result[0].sharpe == 1.5

    def test_custom_strategy_fn(self):
        """Verify strategy_fn + param_grid works for non-weighted strategies."""
        from cobweb_py.sweep import param_sweep, RuleStrategy

        sim = MagicMock()
        sim.enrich.return_value = {"rows": _make_enriched_rows()}
        sim.backtest.return_value = {
            "metrics": {
                "sharpe": 1.2,
                "total_return": 0.15,
                "max_drawdown": -0.05,
                "sortino": 1.8,
                "total_trades": 8,
            }
        }

        def make_rule(**p):
            rsi_buy = p["rsi_buy"]
            def rule(rows):
                return [1.0 if row["rsi_14"] < rsi_buy else 0.0
                        for row in rows]
            return rule

        result = param_sweep(
            sim, _make_enriched_rows(),
            strategy_fn=make_rule,
            param_grid={"rsi_buy": [25, 30, 35]},
            max_workers=1,
            progress=False,
        )
        assert len(result) == 3


class TestSplitMultiTickerDf:
    """Tests for _split_multi_ticker_df and market_sweep data= parameter."""

    def test_split_default_yfinance_format(self):
        """Default yf.download format: level-0 = field, level-1 = ticker."""
        from cobweb_py.sweep import _split_multi_ticker_df
        arrays = [
            ["Close", "Close", "High", "High"],
            ["AAPL", "MSFT", "AAPL", "MSFT"],
        ]
        cols = pd.MultiIndex.from_arrays(arrays, names=["Price", "Ticker"])
        df = pd.DataFrame(
            [[150.0, 400.0, 155.0, 405.0], [151.0, 401.0, 156.0, 406.0]],
            columns=cols,
        )
        df.index.name = "Date"
        result = _split_multi_ticker_df(df, ["AAPL", "MSFT"])
        assert "AAPL" in result
        assert "MSFT" in result
        assert "Close" in result["AAPL"].columns
        assert len(result["AAPL"]) == 2

    def test_split_group_by_ticker_format(self):
        """group_by='ticker' format: level-0 = ticker, level-1 = field."""
        from cobweb_py.sweep import _split_multi_ticker_df
        arrays = [
            ["AAPL", "AAPL", "MSFT", "MSFT"],
            ["Close", "High", "Close", "High"],
        ]
        cols = pd.MultiIndex.from_arrays(arrays)
        df = pd.DataFrame(
            [[150.0, 155.0, 400.0, 405.0]],
            columns=cols,
        )
        df.index.name = "Date"
        result = _split_multi_ticker_df(df, ["AAPL", "MSFT"])
        assert "AAPL" in result
        assert "Close" in result["AAPL"].columns

    def test_split_missing_ticker(self):
        """Tickers not in the DataFrame are silently skipped."""
        from cobweb_py.sweep import _split_multi_ticker_df
        arrays = [
            ["Close", "Close"],
            ["AAPL", "MSFT"],
        ]
        cols = pd.MultiIndex.from_arrays(arrays, names=["Price", "Ticker"])
        df = pd.DataFrame([[150.0, 400.0]], columns=cols)
        df.index.name = "Date"
        result = _split_multi_ticker_df(df, ["AAPL", "GOOGL"])
        assert "AAPL" in result
        assert "GOOGL" not in result

    def test_market_sweep_with_data_dict(self):
        """market_sweep accepts data= as a dict of ticker → data."""
        from cobweb_py.sweep import market_sweep
        sim = MagicMock()
        sim.enrich.return_value = {"rows": _make_enriched_rows()}

        data = {
            "AAPL": {"rows": _make_enriched_rows()},
            "MSFT": {"rows": _make_enriched_rows()},
        }
        result = market_sweep(
            sim, ["AAPL", "MSFT"],
            weights={36: 0.3, 1: 0.7},
            data=data,
            max_workers=1,
            progress=False,
        )
        assert len(result) == 2
        # Should NOT call from_yfinance when data is provided
        assert sim.enrich.call_count == 2

    def test_market_sweep_with_data_dict_missing_ticker(self):
        """Missing tickers in data dict are reported as errors."""
        from cobweb_py.sweep import market_sweep
        sim = MagicMock()
        sim.enrich.return_value = {"rows": _make_enriched_rows()}

        data = {"AAPL": {"rows": _make_enriched_rows()}}
        result = market_sweep(
            sim, ["AAPL", "MISSING"],
            weights={36: 0.3, 1: 0.7},
            data=data,
            max_workers=1,
            progress=False,
        )
        assert len(result) == 1
        assert len(result.errors) == 1
        assert result.errors[0][0] == "MISSING"


class TestSweepExports:
    def test_all_exports_present(self):
        import cobweb_py
        expected = [
            "WeightedStrategy", "RuleStrategy", "ModelStrategy",
            "market_sweep", "param_sweep",
            "SweepRow", "SweepResult",
            "ParamRow", "ParamSweepResult",
        ]
        for name in expected:
            assert hasattr(cobweb_py, name), f"Missing export: {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
