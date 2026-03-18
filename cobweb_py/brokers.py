"""Broker adapters for live/paper trading deployment."""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .client import CobwebError


# ---------------------------------------------------------------------------
# Position sizers
# ---------------------------------------------------------------------------

class PositionSizer(ABC):
    """Base class for position sizing strategies."""

    @abstractmethod
    def calculate_qty(self, buying_power: float, portfolio_value: float, price: float) -> int:
        """Return the number of whole shares to buy."""


class FullCash(PositionSizer):
    """Use all available buying power (minus a 5% buffer)."""

    def calculate_qty(self, buying_power: float, portfolio_value: float, price: float) -> int:
        return math.floor(buying_power * 0.95 / price) if price > 0 else 0


class FixedQty(PositionSizer):
    """Always buy a fixed number of shares."""

    def __init__(self, qty: int):
        self.qty = qty

    def calculate_qty(self, buying_power: float, portfolio_value: float, price: float) -> int:
        affordable = math.floor(buying_power / price) if price > 0 else 0
        return min(self.qty, affordable)


class PercentOfEquity(PositionSizer):
    """Use a percentage of total portfolio value."""

    def __init__(self, pct: float):
        if not 0 < pct <= 1:
            raise ValueError("pct must be between 0 and 1 (e.g. 0.95 for 95%)")
        self.pct = pct

    def calculate_qty(self, buying_power: float, portfolio_value: float, price: float) -> int:
        target_dollars = portfolio_value * self.pct
        usable = min(target_dollars, buying_power)
        return math.floor(usable / price) if price > 0 else 0


class FixedDollar(PositionSizer):
    """Spend a fixed dollar amount."""

    def __init__(self, dollars: float):
        self.dollars = dollars

    def calculate_qty(self, buying_power: float, portfolio_value: float, price: float) -> int:
        usable = min(self.dollars, buying_power)
        return math.floor(usable / price) if price > 0 else 0


# ---------------------------------------------------------------------------
# Base broker
# ---------------------------------------------------------------------------

@dataclass
class BrokerOrder:
    """Standardised order result returned by all broker adapters."""
    order_id: str
    symbol: str
    side: str       # "buy" or "sell"
    qty: int
    status: str     # "submitted", "filled", "dry_run", etc.


class BaseBroker(ABC):
    """Abstract broker interface.  Subclass for Alpaca, IBKR, etc."""

    @abstractmethod
    def connect(self) -> None:
        """Validate credentials and connect to the broker."""

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Return account summary: buying_power, portfolio_value, status."""

    @abstractmethod
    def get_position(self, symbol: str) -> int:
        """Return current share count for *symbol* (0 if no position)."""

    @abstractmethod
    def buy(self, symbol: str, qty: int) -> BrokerOrder:
        """Submit a market buy order."""

    @abstractmethod
    def sell(self, symbol: str, qty: int) -> BrokerOrder:
        """Submit a market sell order (or close position)."""

    @abstractmethod
    def close_position(self, symbol: str) -> BrokerOrder:
        """Liquidate the entire position in *symbol*."""

    @abstractmethod
    def get_last_price(self, symbol: str) -> float:
        """Return the most recent trade price for *symbol*."""


# ---------------------------------------------------------------------------
# Alpaca broker
# ---------------------------------------------------------------------------

class AlpacaBroker(BaseBroker):
    """
    Alpaca broker adapter for paper or live trading.

    Credentials are read from the arguments or environment variables:
      - ``ALPACA_API_KEY``
      - ``ALPACA_SECRET_KEY``

    Example::

        broker = AlpacaBroker(paper=True)
        broker = AlpacaBroker(api_key="...", secret_key="...", paper=True)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self.paper = paper
        self._api: Any = None

        if not self.api_key or not self.secret_key:
            raise CobwebError(
                "Alpaca API keys not found.\n\n"
                "Set environment variables:\n"
                '  export ALPACA_API_KEY="your-key-id"\n'
                '  export ALPACA_SECRET_KEY="your-secret-key"\n\n'
                "Or pass directly:\n"
                '  cw.AlpacaBroker(api_key="...", secret_key="...", paper=True)\n\n'
                "Get keys at: https://app.alpaca.markets/paper/dashboard/overview"
            )

        self.connect()

    def connect(self) -> None:
        try:
            import alpaca_trade_api as tradeapi  # type: ignore
        except ImportError:
            raise CobwebError(
                "alpaca-trade-api is required for AlpacaBroker.\n"
                "Install with: pip install alpaca-trade-api"
            )

        base_url = (
            "https://paper-api.alpaca.markets"
            if self.paper
            else "https://api.alpaca.markets"
        )
        self._api = tradeapi.REST(
            self.api_key, self.secret_key, base_url, api_version="v2"
        )

        # Validate credentials
        try:
            acct = self._api.get_account()
            if acct.status != "ACTIVE":
                raise CobwebError(f"Alpaca account status is '{acct.status}', expected 'ACTIVE'.")
        except Exception as e:
            if "forbidden" in str(e).lower() or "unauthorized" in str(e).lower():
                raise CobwebError(
                    "Alpaca authentication failed. Check your API keys.\n"
                    "Get keys at: https://app.alpaca.markets/paper/dashboard/overview"
                )
            raise

        mode = "paper" if self.paper else "LIVE"
        print(f"  Connected to Alpaca ({mode} trading)")

    def get_account_info(self) -> Dict[str, Any]:
        acct = self._api.get_account()
        return {
            "status": acct.status,
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "cash": float(acct.cash),
        }

    def get_position(self, symbol: str) -> int:
        try:
            pos = self._api.get_position(symbol)
            return int(pos.qty)
        except Exception:
            return 0

    def buy(self, symbol: str, qty: int) -> BrokerOrder:
        order = self._api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
        )
        return BrokerOrder(
            order_id=order.id,
            symbol=symbol,
            side="buy",
            qty=qty,
            status=order.status,
        )

    def sell(self, symbol: str, qty: int) -> BrokerOrder:
        order = self._api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="market",
            time_in_force="day",
        )
        return BrokerOrder(
            order_id=order.id,
            symbol=symbol,
            side="sell",
            qty=qty,
            status=order.status,
        )

    def close_position(self, symbol: str) -> BrokerOrder:
        qty = self.get_position(symbol)
        self._api.close_position(symbol)
        return BrokerOrder(
            order_id="close",
            symbol=symbol,
            side="sell",
            qty=qty,
            status="submitted",
        )

    def get_last_price(self, symbol: str) -> float:
        try:
            snapshot = self._api.get_snapshot(symbol)
            return float(snapshot.latest_trade.price)
        except Exception:
            # Fallback: get last bar
            bars = self._api.get_bars(symbol, "1Day", limit=1)
            if bars:
                return float(bars[0].c)
            raise CobwebError(f"Could not get price for {symbol}")
