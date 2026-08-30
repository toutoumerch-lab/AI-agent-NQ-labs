"""Risk engine tests.

Position sizing is the component where an error costs real money and is least
likely to be noticed — an off-by-one in contracts is a 2x risk overshoot that
looks like a run of bad luck. So it is tested exhaustively.
"""

from __future__ import annotations

from decimal import Decimal as D

import pytest

from nqlab.data.contracts import MNQ, NQ
from nqlab.risk.sizing import (
    RiskConfig,
    RiskError,
    breakeven_win_rate,
    expected_value_r,
    r_multiple,
    size_position,
    stop_distance_points,
    target_from_r,
)


@pytest.fixture
def config() -> RiskConfig:
    return RiskConfig(account_size=D("100000"), risk_per_trade=D("0.01"))


class TestStopDistance:
    def test_is_direction_agnostic(self) -> None:
        long_stop = stop_distance_points(D("18000"), D("17950"), NQ)
        short_stop = stop_distance_points(D("18000"), D("18050"), NQ)
        assert long_stop == short_stop == D("50.00")

    def test_snaps_to_valid_ticks(self) -> None:
        # 17_949.87 rounds to 17_949.75; the distance is measured after rounding
        # because an order cannot rest at an invalid price.
        assert stop_distance_points(D("18000"), D("17949.87"), NQ) == D("50.25")

    def test_rejects_a_stop_on_the_entry_tick(self) -> None:
        with pytest.raises(RiskError, match="undefined risk"):
            stop_distance_points(D("18000"), D("18000.10"), NQ)


class TestPositionSizing:
    def test_never_exceeds_the_risk_budget(self, config: RiskConfig) -> None:
        """The core guarantee, swept across a wide range of stop distances."""
        for stop_points in range(1, 200, 3):
            size = size_position(
                entry=D("18000"),
                stop=D("18000") - D(stop_points),
                spec=NQ,
                config=config,
            )
            assert size.risk_dollars <= config.risk_dollars, (
                f"{stop_points}pt stop sized {size.contracts} contracts risking "
                f"${size.risk_dollars}, over the ${config.risk_dollars} budget"
            )

    def test_rounds_down_never_up(self) -> None:
        """Rounding up would overshoot risk on every trade that does not divide
        evenly — a small systematic error that compounds."""
        cfg = RiskConfig(account_size=D("100000"), risk_per_trade=D("0.01"))
        size = size_position(entry=D("18000"), stop=D("17990"), spec=NQ, config=cfg)

        loss_per_contract = NQ.points_to_dollars(D("10") + D("0.5")) + NQ.default_commission
        assert size.contracts * loss_per_contract <= cfg.risk_dollars
        assert (size.contracts + 1) * loss_per_contract > cfg.risk_dollars

    def test_refuses_to_trade_when_one_contract_is_too_much_risk(self) -> None:
        """Returns zero with a reason rather than rounding up to one.

        A $10k account at 0.5% has $50 to risk; one NQ contract on a 50-point
        stop risks over $1,000. Sizing must decline, not shrug and trade one.
        """
        tiny = RiskConfig(account_size=D("10000"), risk_per_trade=D("0.005"))
        size = size_position(entry=D("18000"), stop=D("17950"), spec=NQ, config=tiny)

        assert size.contracts == 0
        assert not size.is_tradeable
        assert "exceeds" in size.reason
        assert "MNQ" in size.reason, "the reason should name the actionable alternative"

    def test_micro_contract_makes_the_same_trade_feasible(self) -> None:
        """MNQ is 1/10 the size, so an account too small for NQ can still trade.

        The budget has to sit between the two contracts' per-contract risk for
        this to mean anything: at $250, a 50-point stop costs $1,002 on NQ and
        $102 on MNQ.
        """
        cfg = RiskConfig(account_size=D("25000"), risk_per_trade=D("0.01"))
        nq = size_position(entry=D("18000"), stop=D("17950"), spec=NQ, config=cfg)
        mnq = size_position(entry=D("18000"), stop=D("17950"), spec=MNQ, config=cfg)

        assert nq.contracts == 0, "NQ should be unaffordable at this budget"
        assert mnq.contracts >= 1, "MNQ should be affordable at the same budget"

    def test_micro_is_never_less_affordable_than_the_full_contract(self) -> None:
        """Swept: MNQ must size at least as many contracts as NQ, always."""
        cfg = RiskConfig(account_size=D("250000"), risk_per_trade=D("0.01"))
        for stop_points in range(5, 150, 5):
            stop = D("18000") - D(stop_points)
            nq = size_position(entry=D("18000"), stop=stop, spec=NQ, config=cfg)
            mnq = size_position(entry=D("18000"), stop=stop, spec=MNQ, config=cfg)
            assert mnq.contracts >= nq.contracts

    def test_respects_the_contract_cap(self) -> None:
        cfg = RiskConfig(account_size=D("10000000"), risk_per_trade=D("0.01"), max_contracts=5)
        size = size_position(entry=D("18000"), stop=D("17999"), spec=NQ, config=cfg)

        assert size.contracts == 5
        assert "capped" in size.reason

    def test_a_wider_stop_never_increases_size(self, config: RiskConfig) -> None:
        """Monotonicity. A violation means the arithmetic is inverted somewhere."""
        sizes = [
            size_position(
                entry=D("18000"), stop=D("18000") - D(p), spec=NQ, config=config
            ).contracts
            for p in range(5, 120, 5)
        ]
        assert sizes == sorted(sizes, reverse=True)

    def test_costs_are_charged_against_the_budget(self) -> None:
        """Sizing that ignores slippage and commission under-reports risk."""
        free = RiskConfig(
            account_size=D("100000"),
            risk_per_trade=D("0.01"),
            slippage_ticks=D("0"),
            commission_per_contract=D("0"),
        )
        costly = RiskConfig(
            account_size=D("100000"),
            risk_per_trade=D("0.01"),
            slippage_ticks=D("2"),
            commission_per_contract=D("10"),
        )
        a = size_position(entry=D("18000"), stop=D("17990"), spec=NQ, config=free)
        b = size_position(entry=D("18000"), stop=D("17990"), spec=NQ, config=costly)
        assert b.contracts < a.contracts


class TestRiskConfig:
    def test_rejects_absurd_risk_per_trade(self) -> None:
        with pytest.raises(RiskError, match="not a strategy"):
            RiskConfig(account_size=D("100000"), risk_per_trade=D("0.5"))

    def test_rejects_non_positive_account(self) -> None:
        with pytest.raises(RiskError):
            RiskConfig(account_size=D("0"), risk_per_trade=D("0.01"))


class TestRMultiple:
    def test_stop_out_is_exactly_minus_one_r(self) -> None:
        assert r_multiple(
            entry=D("18000"),
            stop=D("17950"),
            exit_price=D("17950"),
            is_long=True,
            spec=NQ,
        ) == D("-1")

    def test_target_at_two_r_returns_two(self) -> None:
        assert r_multiple(
            entry=D("18000"),
            stop=D("17950"),
            exit_price=D("18100"),
            is_long=True,
            spec=NQ,
        ) == D("2")

    def test_short_side_is_symmetric(self) -> None:
        assert r_multiple(
            entry=D("18000"),
            stop=D("18050"),
            exit_price=D("17900"),
            is_long=False,
            spec=NQ,
        ) == D("2")
        assert r_multiple(
            entry=D("18000"),
            stop=D("18050"),
            exit_price=D("18050"),
            is_long=False,
            spec=NQ,
        ) == D("-1")

    def test_round_trip_with_target_from_r(self) -> None:
        """target_from_r and r_multiple must be inverses."""
        for r in (D("1"), D("1.5"), D("2"), D("3.25")):
            for is_long in (True, False):
                target = target_from_r(
                    entry=D("18000"),
                    stop=D("17950") if is_long else D("18050"),
                    r=r,
                    is_long=is_long,
                    spec=NQ,
                )
                realised = r_multiple(
                    entry=D("18000"),
                    stop=D("17950") if is_long else D("18050"),
                    exit_price=target,
                    is_long=is_long,
                    spec=NQ,
                )
                assert realised == pytest.approx(float(r), abs=0.02)


class TestExpectedValue:
    def test_breakeven_win_rate_gives_zero_ev(self) -> None:
        for reward in (D("1"), D("2"), D("3")):
            wr = breakeven_win_rate(reward)
            assert expected_value_r(win_probability=wr, reward_r=reward) == pytest.approx(
                0, abs=1e-9
            )

    def test_a_low_win_rate_can_beat_a_high_one(self) -> None:
        """The point of EV: 40% at 3R beats 70% at 0.5R."""
        patient = expected_value_r(win_probability=D("0.40"), reward_r=D("3"))
        scalper = expected_value_r(win_probability=D("0.70"), reward_r=D("0.5"))
        assert patient > scalper

    def test_rejects_an_impossible_probability(self) -> None:
        with pytest.raises(RiskError):
            expected_value_r(win_probability=D("1.5"), reward_r=D("2"))

    def test_breakeven_win_rate_falls_as_reward_rises(self) -> None:
        rates = [breakeven_win_rate(D(str(r))) for r in (0.5, 1, 2, 3, 5)]
        assert rates == sorted(rates, reverse=True)
