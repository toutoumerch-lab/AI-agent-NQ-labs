"""Position sizing and risk arithmetic.

Pure functions over Decimals. Nothing here touches a model, a data source or
the network — which is why it can be tested exhaustively, and why it is the one
part of the system whose correctness is not in question.

All money is Decimal. A sizing routine that accumulates float error across
thousands of backtest trades reports a P&L that does not reconcile, and the
discrepancy is invisible until someone adds up the trade list by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from ..data.contracts import ContractSpec


class RiskError(Exception):
    """Raised when a risk constraint is violated or a request is incoherent."""


@dataclass(frozen=True)
class RiskConfig:
    """Account-level risk settings. All configurable; none hardcoded."""

    account_size: Decimal
    #: Fraction of the account risked on one trade, e.g. 0.005 for 0.5%.
    risk_per_trade: Decimal
    #: Hard cap on contracts regardless of what sizing computes.
    max_contracts: int = 10
    #: Fraction of the account that halts trading for the day once lost.
    daily_loss_limit: Decimal = Decimal("0.02")
    #: Round-turn commission per contract. Overrides the contract default.
    commission_per_contract: Decimal | None = None
    #: Assumed slippage per side, in ticks.
    slippage_ticks: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        if self.account_size <= 0:
            raise RiskError("account_size must be positive")
        if not (Decimal("0") < self.risk_per_trade <= Decimal("0.1")):
            raise RiskError(
                "risk_per_trade must be in (0, 0.10]. Risking more than 10% of "
                "an account on a single trade is not a strategy."
            )
        if self.max_contracts <= 0:
            raise RiskError("max_contracts must be positive")

    @property
    def risk_dollars(self) -> Decimal:
        """Dollars at risk on one trade."""
        return self.account_size * self.risk_per_trade


@dataclass(frozen=True)
class PositionSize:
    contracts: int
    #: Dollars at risk if the stop fills exactly.
    risk_dollars: Decimal
    #: Stop distance in index points.
    stop_points: Decimal
    #: Why sizing produced this number — surfaced rather than inferred.
    reason: str

    @property
    def is_tradeable(self) -> bool:
        return self.contracts > 0


def stop_distance_points(entry: Decimal, stop: Decimal, spec: ContractSpec) -> Decimal:
    """Absolute stop distance, tick-aligned.

    Direction-agnostic: a long's stop is below entry and a short's above, and
    the distance is what sizing needs either way.
    """
    entry_t = spec.round_to_tick(entry)
    stop_t = spec.round_to_tick(stop)
    distance = abs(entry_t - stop_t)
    if distance == 0:
        raise RiskError(
            "Entry and stop round to the same tick. The stop must be at least "
            "one tick away or the position has undefined risk."
        )
    return distance


def size_position(
    *,
    entry: Decimal,
    stop: Decimal,
    spec: ContractSpec,
    config: RiskConfig,
) -> PositionSize:
    """Contracts to trade for a given stop distance.

    Rounds **down**. A fractional contract cannot be traded, and rounding up
    would exceed the configured risk on every trade that does not divide evenly
    — a small, systematic overshoot that compounds.

    Slippage and commission are charged against the risk budget, not ignored:
    a stop that fills a tick worse than intended plus a round-turn commission
    is real money, and a sizing routine that omits them under-reports risk on
    every trade.
    """
    stop_points = stop_distance_points(entry, stop, spec)

    # Cost per contract if the stop is hit: the stop distance, plus slippage on
    # both entry and exit, plus commission.
    slippage_points = config.slippage_ticks * spec.tick_size * 2
    commission = (
        config.commission_per_contract
        if config.commission_per_contract is not None
        else spec.default_commission
    )
    loss_per_contract = spec.points_to_dollars(stop_points + slippage_points) + commission

    if loss_per_contract <= 0:
        raise RiskError("Computed a non-positive loss per contract")

    raw = (config.risk_dollars / loss_per_contract).quantize(Decimal("1"), rounding=ROUND_DOWN)
    contracts = int(raw)

    if contracts <= 0:
        return PositionSize(
            contracts=0,
            risk_dollars=Decimal("0"),
            stop_points=stop_points,
            reason=(
                f"Stop of {stop_points} points risks "
                f"${loss_per_contract:.2f}/contract, which exceeds the "
                f"${config.risk_dollars:.2f} budget. Widen the account, tighten "
                f"the stop, or use a smaller contract (MNQ)."
            ),
        )

    capped = min(contracts, config.max_contracts)
    reason = f"{capped} contract(s) at ${loss_per_contract:.2f} risk each" + (
        f" (capped from {contracts} by max_contracts)" if capped < contracts else ""
    )

    return PositionSize(
        contracts=capped,
        risk_dollars=loss_per_contract * capped,
        stop_points=stop_points,
        reason=reason,
    )


def r_multiple(
    *,
    entry: Decimal,
    stop: Decimal,
    exit_price: Decimal,
    is_long: bool,
    spec: ContractSpec,
) -> Decimal:
    """Result in R — profit or loss as a multiple of the initial risk.

    R normalises across stop distances and account sizes, which is what makes
    trades from different volatility regimes comparable at all. Reporting
    results in dollars instead hides that a good month was three large positions.
    """
    stop_points = stop_distance_points(entry, stop, spec)
    entry_t = spec.round_to_tick(entry)
    exit_t = spec.round_to_tick(exit_price)

    move = (exit_t - entry_t) if is_long else (entry_t - exit_t)
    return move / stop_points


def target_from_r(
    *,
    entry: Decimal,
    stop: Decimal,
    r: Decimal,
    is_long: bool,
    spec: ContractSpec,
) -> Decimal:
    """Price that would realise `r` R, tick-aligned."""
    stop_points = stop_distance_points(entry, stop, spec)
    entry_t = spec.round_to_tick(entry)
    offset = stop_points * r
    return spec.round_to_tick(entry_t + offset if is_long else entry_t - offset)


def expected_value_r(
    *, win_probability: Decimal, reward_r: Decimal, loss_r: Decimal = Decimal("1")
) -> Decimal:
    """Expected value per trade in R.

    The number that decides whether a setup is worth taking. A 40% win rate at
    3R is better than a 70% win rate at 0.5R, and only EV shows that.
    """
    if not (Decimal("0") <= win_probability <= Decimal("1")):
        raise RiskError("win_probability must be in [0, 1]")
    return win_probability * reward_r - (Decimal("1") - win_probability) * loss_r


def breakeven_win_rate(reward_r: Decimal, loss_r: Decimal = Decimal("1")) -> Decimal:
    """Win rate at which a setup breaks even.

    Displayed next to any predicted probability, because "62%" means nothing
    until you know the setup needs 33% to break even.
    """
    if reward_r <= 0:
        raise RiskError("reward_r must be positive")
    return loss_r / (reward_r + loss_r)
