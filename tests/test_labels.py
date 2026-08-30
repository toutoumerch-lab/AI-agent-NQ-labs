"""Triple-barrier labelling tests.

Hand-built price paths where the correct label is known by inspection. A
labeller that is subtly wrong produces a model that learns the wrong thing, and
nothing downstream will reveal it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqlab.features.labels import (
    BarrierConfig,
    Outcome,
    triple_barrier_labels,
)


def path(closes: list[float], *, highs=None, lows=None) -> pd.DataFrame:
    """Build bars from an explicit close path."""
    n = len(closes)
    index = pd.date_range("2026-01-05 09:30", periods=n, freq="5min", tz="UTC")
    close = np.array(closes, dtype="float64")
    return pd.DataFrame(
        {
            "open": close,
            "high": np.array(highs, dtype="float64") if highs else close,
            "low": np.array(lows, dtype="float64") if lows else close,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=index,
    )


def flat_atr(bars: pd.DataFrame, value: float = 10.0) -> pd.Series:
    """Constant ATR, so barrier distances are exact and hand-checkable."""
    return pd.Series(value, index=bars.index, dtype="float64")


#: No costs, 1 ATR stop, 2R target — barriers land on round numbers.
CLEAN = BarrierConfig(stop_atr=1.0, target_r=2.0, max_holding=10, cost_points=0.0)


class TestBarrierTouches:
    def test_reaching_the_target_first_is_a_win(self) -> None:
        # Entry 100, ATR 10 -> stop 90, target 120. Price walks up to 121.
        bars = path([100, 105, 110, 121, 121])
        result = triple_barrier_labels(bars, atr=flat_atr(bars), config=CLEAN)

        assert result.outcome.iloc[0] == Outcome.WIN
        assert result.holding_bars.iloc[0] == 3

    def test_reaching_the_stop_first_is_a_loss(self) -> None:
        bars = path([100, 96, 92, 89, 89])
        result = triple_barrier_labels(bars, atr=flat_atr(bars), config=CLEAN)

        assert result.outcome.iloc[0] == Outcome.LOSS
        assert result.holding_bars.iloc[0] == 3

    def test_touching_neither_barrier_is_a_timeout(self) -> None:
        bars = path([100] * 12)
        result = triple_barrier_labels(bars, atr=flat_atr(bars), config=CLEAN)

        assert result.outcome.iloc[0] == Outcome.TIMEOUT
        assert result.holding_bars.iloc[0] == CLEAN.max_holding

    def test_order_of_touch_decides_not_the_final_price(self) -> None:
        """The path-dependence that makes this label worth the complexity.

        Price ends far above the target, but hits the stop on the way. A
        directional forecast would call this a win; the trade was closed at a
        loss before the move happened.
        """
        bars = path([100, 89, 130, 130], highs=[100, 100, 130, 130], lows=[100, 89, 89, 130])
        result = triple_barrier_labels(bars, atr=flat_atr(bars), config=CLEAN)

        assert result.outcome.iloc[0] == Outcome.LOSS

    def test_both_barriers_in_one_bar_resolves_pessimistically(self) -> None:
        """OHLC cannot reveal intrabar order, so the loss is assumed.

        Assuming the favourable order is the single most common way a backtest
        is inflated without a line of dishonest code being written.
        """
        # Bar 1 spans 85 to 125 — through both the 90 stop and the 120 target.
        bars = path([100, 105], highs=[100, 125], lows=[100, 85])
        result = triple_barrier_labels(bars, atr=flat_atr(bars), config=CLEAN)

        assert result.outcome.iloc[0] == Outcome.LOSS


class TestShortSide:
    def test_short_target_is_below_entry(self) -> None:
        bars = path([100, 95, 90, 79])
        result = triple_barrier_labels(bars, atr=flat_atr(bars), direction=-1, config=CLEAN)

        assert result.outcome.iloc[0] == Outcome.WIN

    def test_short_stop_is_above_entry(self) -> None:
        bars = path([100, 105, 111, 111])
        result = triple_barrier_labels(bars, atr=flat_atr(bars), direction=-1, config=CLEAN)

        assert result.outcome.iloc[0] == Outcome.LOSS

    def test_a_path_is_a_win_short_exactly_when_its_mirror_is_a_win_long(self) -> None:
        up = path([100, 108, 121])
        down = path([100, 92, 79])

        long_result = triple_barrier_labels(up, atr=flat_atr(up), direction=1, config=CLEAN)
        short_result = triple_barrier_labels(down, atr=flat_atr(down), direction=-1, config=CLEAN)

        assert long_result.outcome.iloc[0] == short_result.outcome.iloc[0] == Outcome.WIN


class TestCosts:
    def test_costs_can_turn_a_marginal_win_into_a_non_win(self) -> None:
        """A gross win that does not clear costs must not be labelled a win.

        Gross target is 120. Price touches exactly 120 and stops there. Free of
        costs that is a win; with 2 points of cost the target is 122 and it
        is not.
        """
        bars = path([100, 110, 120, 120, 120, 120])
        free = triple_barrier_labels(bars, atr=flat_atr(bars), config=CLEAN)
        costly = triple_barrier_labels(
            bars,
            atr=flat_atr(bars),
            config=BarrierConfig(stop_atr=1.0, target_r=2.0, max_holding=10, cost_points=2.0),
        )

        assert free.outcome.iloc[0] == Outcome.WIN
        assert costly.outcome.iloc[0] != Outcome.WIN

    def test_costs_never_make_a_win_easier(self) -> None:
        """Swept invariant: raising costs can only ever remove wins.

        The direction that matters. An implementation that pulled the target in
        and pushed the stop out would inflate every win rate the system reports,
        and it would look like a rounding detail in review.
        """
        bars = path([100 + i for i in range(30)])
        wins = []
        for cost in (0.0, 1.0, 3.0, 6.0, 12.0):
            result = triple_barrier_labels(
                bars,
                atr=flat_atr(bars),
                config=BarrierConfig(stop_atr=1.0, target_r=2.0, max_holding=15, cost_points=cost),
            )
            wins.append(int((result.outcome == Outcome.WIN).sum()))

        assert wins == sorted(wins, reverse=True), f"win count rose as costs increased: {wins}"

    def test_the_stop_does_not_move_because_of_costs(self) -> None:
        """A resting stop order does not move because commission exists.

        The extra loss costs cause is charged by the backtester; folding it into
        the label would make stop-outs less likely, which is backwards.
        """
        bars = path([100, 95, 89.5, 89.5, 89.5, 89.5])
        free = triple_barrier_labels(bars, atr=flat_atr(bars), config=CLEAN)
        costly = triple_barrier_labels(
            bars,
            atr=flat_atr(bars),
            config=BarrierConfig(stop_atr=1.0, target_r=2.0, max_holding=10, cost_points=2.0),
        )

        assert free.outcome.iloc[0] == Outcome.LOSS
        assert costly.outcome.iloc[0] == Outcome.LOSS


class TestExcursions:
    def test_mfe_and_mae_are_measured_in_r(self) -> None:
        # Rises to 115 (1.5R) then falls to 95 (0.5R adverse) before resolving.
        bars = path([100, 115, 95, 100], highs=[100, 115, 115, 100], lows=[100, 100, 95, 100])
        result = triple_barrier_labels(bars, atr=flat_atr(bars), config=CLEAN)

        assert result.mfe_r.iloc[0] == pytest.approx(1.5)
        assert result.mae_r.iloc[0] == pytest.approx(0.5)

    def test_excursions_are_never_negative(self) -> None:
        bars = path([100] * 8)
        result = triple_barrier_labels(bars, atr=flat_atr(bars), config=CLEAN)
        assert (result.mfe_r.dropna() >= 0).all()
        assert (result.mae_r.dropna() >= 0).all()


class TestUnresolvedLabels:
    def test_the_tail_is_nan_not_timeout(self) -> None:
        """Bars without a full forward window must be NaN, not TIMEOUT.

        Recording them as non-events would bias the end of every sample toward
        whichever class TIMEOUT maps to.
        """
        bars = path([100] * 12)
        result = triple_barrier_labels(bars, atr=flat_atr(bars), config=CLEAN)

        # Only bars with 10 full bars ahead can resolve.
        assert result.outcome.iloc[:2].notna().all()
        assert result.outcome.iloc[2:].isna().all()

    def test_a_nan_atr_yields_a_nan_label(self) -> None:
        bars = path([100, 110, 121, 121, 121])
        atr = flat_atr(bars)
        atr.iloc[0] = np.nan
        result = triple_barrier_labels(bars, atr=atr, config=CLEAN)

        assert pd.isna(result.outcome.iloc[0])


class TestUniqueness:
    def test_uniqueness_is_a_fraction(self) -> None:
        bars = path([100 + i * 0.5 for i in range(60)])
        result = triple_barrier_labels(bars, atr=flat_atr(bars), config=CLEAN)
        values = result.uniqueness.dropna()

        assert len(values)
        assert ((values > 0) & (values <= 1)).all()

    def test_overlapping_labels_are_less_unique_than_isolated_ones(self) -> None:
        """The whole point: heavily overlapping labels are not independent."""
        overlapping = path([100] * 40)
        long_window = triple_barrier_labels(
            overlapping,
            atr=flat_atr(overlapping),
            config=BarrierConfig(stop_atr=1.0, target_r=2.0, max_holding=20, cost_points=0.0),
        )
        short_window = triple_barrier_labels(
            overlapping,
            atr=flat_atr(overlapping),
            config=BarrierConfig(stop_atr=1.0, target_r=2.0, max_holding=2, cost_points=0.0),
        )
        assert long_window.uniqueness.mean() < short_window.uniqueness.mean()
