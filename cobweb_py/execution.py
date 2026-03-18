"""Signal state tracking, execution logging, and deployment logic."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .brokers import BaseBroker, BrokerOrder, FullCash, PositionSizer
from .client import CobwebError


# ---------------------------------------------------------------------------
# Signal state tracker
# ---------------------------------------------------------------------------

_STATE_DIR = Path.home() / ".cobweb"
_STATE_FILE = _STATE_DIR / "signal_state.json"


class SignalTracker:
    """
    Tracks the last deployed signal per symbol to avoid duplicate orders.

    State is persisted to ``~/.cobweb/signal_state.json`` so it survives
    script restarts.
    """

    def __init__(self):
        self._state: Dict[str, str] = self._load()

    def _load(self) -> Dict[str, str]:
        if _STATE_FILE.exists():
            try:
                return json.loads(_STATE_FILE.read_text())
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(self._state, indent=2))

    def last_signal(self, symbol: str) -> Optional[str]:
        """Return the last deployed signal for *symbol*, or None."""
        return self._state.get(symbol)

    def should_act(self, symbol: str, new_signal: str) -> bool:
        """True if *new_signal* differs from the last deployed signal."""
        return self._state.get(symbol) != new_signal

    def record(self, symbol: str, signal: str) -> None:
        """Record that *signal* was deployed for *symbol*."""
        self._state[symbol] = signal
        self._save()

    def clear(self, symbol: Optional[str] = None) -> None:
        """Clear state for one symbol or all."""
        if symbol:
            self._state.pop(symbol, None)
        else:
            self._state.clear()
        self._save()


# ---------------------------------------------------------------------------
# Execution logger
# ---------------------------------------------------------------------------

_LOG_FILE = _STATE_DIR / "executions.log"


class ExecutionLogger:
    """Append-only log of every deployment action to ``~/.cobweb/executions.log``."""

    def __init__(self):
        _STATE_DIR.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        symbol: str,
        signal: str,
        action: str,
        qty: int = 0,
        order_id: str = "",
        dry_run: bool = False,
        details: str = "",
    ) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        mode = "[DRY RUN] " if dry_run else ""
        line = (
            f"{ts} | {mode}{symbol} | signal={signal} | action={action} | "
            f"qty={qty} | order_id={order_id} | {details}\n"
        )
        with open(_LOG_FILE, "a") as f:
            f.write(line)

    def read(self, last_n: int = 50) -> List[str]:
        """Return the last *n* log lines."""
        if not _LOG_FILE.exists():
            return []
        lines = _LOG_FILE.read_text().strip().splitlines()
        return lines[-last_n:]


# ---------------------------------------------------------------------------
# Deploy function
# ---------------------------------------------------------------------------

_tracker = SignalTracker()
_logger = ExecutionLogger()


def deploy(
    signal_info: Dict[str, Any],
    broker: BaseBroker,
    symbol: str,
    *,
    price: Optional[float] = None,
    sizer: Optional[PositionSizer] = None,
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Deploy a trading signal to a broker.

    Args:
        signal_info: Dict from ``get_signal(bt)`` with keys:
                     signal ("buy"/"sell"/"hold"), exposure, strength.
        broker:      A connected ``BaseBroker`` instance (e.g. ``AlpacaBroker``).
        symbol:      Ticker symbol to trade (e.g. "AAPL").
        price:       Override price for sizing.  If None, fetched from broker.
        sizer:       Position sizing strategy.  Defaults to ``FullCash()``.
        dry_run:     If True, log what *would* happen without placing orders.
        force:       If True, skip duplicate signal check.

    Returns:
        Dict with keys: action, signal, symbol, qty, order (BrokerOrder or None).
    """
    if sizer is None:
        sizer = FullCash()

    signal = signal_info.get("signal", "hold")
    current_qty = broker.get_position(symbol)
    account = broker.get_account_info()
    buying_power = account["buying_power"]
    portfolio_value = account["portfolio_value"]

    # Duplicate signal check
    if not force and not _tracker.should_act(symbol, signal):
        msg = f"Signal unchanged ({signal}) — skipping. Use force=True to override."
        print(f"  {msg}")
        _logger.log(symbol, signal, "skip_duplicate", details=msg, dry_run=dry_run)
        return {"action": "skip_duplicate", "signal": signal, "symbol": symbol, "qty": 0, "order": None}

    # Get price for sizing
    if price is None:
        price = broker.get_last_price(symbol)

    result: Dict[str, Any] = {
        "signal": signal,
        "symbol": symbol,
        "qty": 0,
        "order": None,
    }

    # ── BUY logic ──
    if signal == "buy" and current_qty == 0:
        qty = sizer.calculate_qty(buying_power, portfolio_value, price)
        if qty <= 0:
            result["action"] = "insufficient_funds"
            print(f"  Not enough buying power to buy {symbol} at ${price:,.2f}")
            _logger.log(symbol, signal, "insufficient_funds", dry_run=dry_run)
            return result

        if dry_run:
            result["action"] = "dry_run_buy"
            result["qty"] = qty
            print(f"  [DRY RUN] Would BUY {qty} shares of {symbol} at ~${price:,.2f} (${qty * price:,.2f})")
            _logger.log(symbol, signal, "dry_run_buy", qty=qty, dry_run=True)
        else:
            order = broker.buy(symbol, qty)
            result["action"] = "buy"
            result["qty"] = qty
            result["order"] = order
            print(f"  BUY order placed: {qty} shares of {symbol}")
            print(f"  Order ID: {order.order_id}")
            _logger.log(symbol, signal, "buy", qty=qty, order_id=order.order_id)
            _tracker.record(symbol, signal)

    elif signal == "buy" and current_qty > 0:
        result["action"] = "already_long"
        print(f"  Already long {current_qty} shares of {symbol} — no action needed.")
        _logger.log(symbol, signal, "already_long", qty=current_qty, dry_run=dry_run)
        _tracker.record(symbol, signal)

    # ── SELL / HOLD logic ──
    elif signal in ("sell", "hold") and current_qty > 0:
        if dry_run:
            result["action"] = "dry_run_sell"
            result["qty"] = current_qty
            print(f"  [DRY RUN] Would SELL {current_qty} shares of {symbol}")
            _logger.log(symbol, signal, "dry_run_sell", qty=current_qty, dry_run=True)
        else:
            order = broker.close_position(symbol)
            result["action"] = "sell"
            result["qty"] = current_qty
            result["order"] = order
            print(f"  SELL order placed: closing {current_qty} shares of {symbol}")
            _logger.log(symbol, signal, "sell", qty=current_qty, order_id=order.order_id)
            _tracker.record(symbol, signal)

    else:
        result["action"] = "no_action"
        print(f"  No position and no buy signal — nothing to do.")
        _logger.log(symbol, signal, "no_action", dry_run=dry_run)
        _tracker.record(symbol, signal)

    return result


def get_execution_log(last_n: int = 50) -> List[str]:
    """Return the last *n* lines from the execution log."""
    return _logger.read(last_n)


def clear_signal_state(symbol: Optional[str] = None) -> None:
    """Clear the signal state tracker (for one symbol or all)."""
    _tracker.clear(symbol)
