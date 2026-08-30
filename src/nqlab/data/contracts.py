"""Futures contract specifications.

Contract mechanics are encoded as data rather than scattered through the code
as magic numbers. A stop distance in "points" is meaningless until multiplied
by the contract's point value, and rounding a price to a valid tick is a
property of the instrument, not of the strategy using it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum


class RollMethod(StrEnum):
    """How a continuous series is stitched across contract expiries."""

    #: Raw front month. Price jumps at every roll — never use for modelling.
    NONE = "NONE"
    #: Back-adjust by the roll-gap difference. Preserves absolute point moves,
    #: so historical prices are NOT real historical prices.
    PANAMA = "PANAMA"
    #: Multiplicative adjustment. Preserves percentage returns.
    RATIO = "RATIO"


@dataclass(frozen=True)
class ContractSpec:
    """Immutable specification of a futures contract.

    Frozen because a spec changing underneath a backtest would silently
    revalue every trade in it.
    """

    symbol: str
    name: str
    exchange: str
    #: Dollar value of a one-point move in the underlying index.
    point_value: Decimal
    #: Minimum price increment, in index points.
    tick_size: Decimal
    #: Contract months as single-letter codes (H=Mar, M=Jun, U=Sep, Z=Dec).
    contract_months: tuple[str, ...]
    #: IANA timezone the exchange session is defined in.
    timezone: str
    #: Regular trading hours, local to `timezone`.
    rth_open: str
    rth_close: str
    #: Electronic session. Opens the prior calendar day.
    globex_open: str
    globex_close: str
    #: Round-turn commission per contract, in dollars. A default, not a promise —
    #: real cost is broker-specific and configurable.
    default_commission: Decimal
    #: Typical bid/ask spread in ticks under normal liquidity.
    typical_spread_ticks: Decimal

    @property
    def tick_value(self) -> Decimal:
        """Dollar value of one tick."""
        return self.tick_size * self.point_value

    def round_to_tick(self, price: float | Decimal) -> Decimal:
        """Snap a price to the nearest valid tick.

        Orders at invalid prices are rejected by the exchange, so a backtest
        that fills at an untradeable price is reporting a fill that could not
        have happened. Uses Decimal throughout: 0.25 is exact in binary but
        the accumulated float error over a long series is not.
        """
        p = Decimal(str(price))
        return (p / self.tick_size).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * self.tick_size

    def points_to_dollars(self, points: float | Decimal) -> Decimal:
        """Convert an index-point move to dollars per contract."""
        return Decimal(str(points)) * self.point_value

    def dollars_to_points(self, dollars: float | Decimal) -> Decimal:
        """Convert a dollar amount to index points per contract."""
        return Decimal(str(dollars)) / self.point_value

    def ticks_to_points(self, ticks: float | Decimal) -> Decimal:
        return Decimal(str(ticks)) * self.tick_size


#: E-mini Nasdaq-100. The primary instrument.
NQ = ContractSpec(
    symbol="NQ",
    name="E-mini Nasdaq-100",
    exchange="CME",
    point_value=Decimal("20"),
    tick_size=Decimal("0.25"),
    contract_months=("H", "M", "U", "Z"),
    timezone="America/New_York",
    rth_open="09:30",
    rth_close="16:00",
    globex_open="18:00",
    globex_close="17:00",
    default_commission=Decimal("4.00"),
    typical_spread_ticks=Decimal("1"),
)

#: Micro E-mini Nasdaq-100 — 1/10 the size. Matters for position sizing on
#: small accounts, where NQ's $20/point makes the minimum risk unit too large.
MNQ = ContractSpec(
    symbol="MNQ",
    name="Micro E-mini Nasdaq-100",
    exchange="CME",
    point_value=Decimal("2"),
    tick_size=Decimal("0.25"),
    contract_months=("H", "M", "U", "Z"),
    timezone="America/New_York",
    rth_open="09:30",
    rth_close="16:00",
    globex_open="18:00",
    globex_close="17:00",
    default_commission=Decimal("1.20"),
    typical_spread_ticks=Decimal("1"),
)

CONTRACTS: dict[str, ContractSpec] = {"NQ": NQ, "MNQ": MNQ}


def get_contract(symbol: str) -> ContractSpec:
    """Look up a contract spec.

    Raises rather than defaulting: silently falling back to NQ's $20 multiplier
    for an unknown symbol would misprice every trade by a factor of ten.
    """
    try:
        return CONTRACTS[symbol.upper()]
    except KeyError:
        raise KeyError(
            f"No contract specification for {symbol!r}. Known: {sorted(CONTRACTS)}"
        ) from None
