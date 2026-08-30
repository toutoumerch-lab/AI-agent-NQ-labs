"""Triple-barrier labelling.

The target the system predicts. From a candidate entry at bar t, with a stop
distance s and a target distance k·s, does price touch the target *before* the
stop, within a maximum holding time H?

Three properties the naive "will the next close be higher" target lacks:

- **Path-dependent.** Order of arrival matters. A forecast that is directionally
  right but drawn through the stop first is a loss, and this label says so.
- **Cost-aware.** Barriers are set net of spread, slippage and commission, so a
  labelled win is a win after costs.
- **Horizon-bounded.** TIMEOUT is a distinct third outcome, not silently forced
  into win or loss.

Labels here are the ONLY place future bars are read, which is legitimate — a
label is by definition about the future. They are registered with `is_label=True`
so the causality tests exclude them, and so they can never be used as inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class Outcome(IntEnum):
    LOSS = -1
    TIMEOUT = 0
    WIN = 1


@dataclass(frozen=True)
class BarrierConfig:
    """Barrier geometry, in ATR units so it adapts to volatility.

    Fixed point distances make a stop that is trivially tight in a fast market
    and absurdly wide in a quiet one, so the label would mean something
    different in each regime.
    """

    #: Stop distance as a multiple of ATR.
    stop_atr: float = 1.0
    #: Target distance as a multiple of the stop. 2.0 means a 2R target.
    target_r: float = 2.0
    #: Maximum holding period, in bars.
    max_holding: int = 48
    #: Round-trip cost in points (spread + slippage + commission), expressed in
    #: index points. The target must clear it, so a labelled win is a win NET
    #: of costs rather than gross.
    cost_points: float = 1.0


@dataclass(frozen=True)
class LabelResult:
    """Labels and the diagnostics needed to weight and audit them."""

    outcome: pd.Series
    #: Bars until the barrier was touched. NaN where the label never resolved.
    holding_bars: pd.Series
    #: Maximum favourable / adverse excursion in R, for sizing research.
    mfe_r: pd.Series
    mae_r: pd.Series
    #: Fraction of this label's span not shared with other labels, in (0, 1].
    #: Overlapping labels are not independent observations, and treating them
    #: as such inflates every significance estimate built on them.
    uniqueness: pd.Series


def triple_barrier_labels(
    bars: pd.DataFrame,
    *,
    atr: pd.Series,
    direction: pd.Series | int = 1,
    config: BarrierConfig | None = None,
) -> LabelResult:
    """Label each bar by which barrier its forward path touches first.

    Args:
        bars: OHLCV with a monotonic index.
        atr: Volatility estimate aligned to `bars`, used to scale the barriers.
            Must itself be causal — a leaky ATR makes leaky labels.
        direction: +1 for long, -1 for short, or a Series of per-bar directions.
        config: Barrier geometry.

    The scan walks forward bar by bar and stops at the first touch. Where both
    barriers fall inside one bar's range, the outcome is recorded as a **LOSS**:
    intrabar order is unknowable from OHLC data, and assuming the favourable
    order is the single most common way backtests are inflated.
    """
    config = config or BarrierConfig()
    n = len(bars)

    directions = (
        pd.Series(direction, index=bars.index)
        if np.isscalar(direction)
        else pd.Series(direction).reindex(bars.index)
    )

    high = bars["high"].to_numpy(dtype="float64")
    low = bars["low"].to_numpy(dtype="float64")
    close = bars["close"].to_numpy(dtype="float64")
    atr_values = atr.to_numpy(dtype="float64")
    dir_values = directions.to_numpy(dtype="float64")

    outcome = np.full(n, np.nan)
    holding = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)

    for i in range(n):
        a = atr_values[i]
        d = dir_values[i]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(d) or d == 0:
            continue

        entry = close[i]
        stop_distance = config.stop_atr * a
        if stop_distance <= 0:
            continue

        # Costs must make winning HARDER, never easier. Price has to travel the
        # gross target PLUS the round-trip cost before the trade nets its
        # target. The stop stays where it was placed — a resting order does not
        # move because commission exists; the extra loss it incurs is charged
        # by the backtester, not hidden in the label.
        #
        # An earlier version pulled the target in and pushed the stop out,
        # which made both barriers easier and would have inflated every win
        # rate the system reported.
        target_distance = config.target_r * stop_distance + config.cost_points
        stop_distance_net = stop_distance
        if target_distance <= 0:
            continue

        if d > 0:
            target_price = entry + target_distance
            stop_price = entry - stop_distance_net
        else:
            target_price = entry - target_distance
            stop_price = entry + stop_distance_net

        last = min(i + config.max_holding, n - 1)
        best = 0.0
        worst = 0.0
        resolved = False

        for j in range(i + 1, last + 1):
            favourable = (high[j] - entry) if d > 0 else (entry - low[j])
            adverse = (entry - low[j]) if d > 0 else (high[j] - entry)
            best = max(best, favourable / stop_distance)
            worst = max(worst, adverse / stop_distance)

            hit_target = (high[j] >= target_price) if d > 0 else (low[j] <= target_price)
            hit_stop = (low[j] <= stop_price) if d > 0 else (high[j] >= stop_price)

            if hit_target and hit_stop:
                # Both barriers inside one bar. OHLC does not reveal which came
                # first, so the pessimistic branch is taken. Assuming the
                # favourable order here is the classic way a backtest is
                # inflated without anyone writing a line of dishonest code.
                outcome[i] = Outcome.LOSS
                holding[i] = j - i
                resolved = True
                break
            if hit_target:
                outcome[i] = Outcome.WIN
                holding[i] = j - i
                resolved = True
                break
            if hit_stop:
                outcome[i] = Outcome.LOSS
                holding[i] = j - i
                resolved = True
                break

        # Ran out of bars. Only a TIMEOUT if the full window was available;
        # otherwise the label is unknown and must stay NaN rather than being
        # recorded as a non-event, which would bias the tail of the sample.
        if not resolved and i + config.max_holding <= n - 1:
            outcome[i] = Outcome.TIMEOUT
            holding[i] = config.max_holding
        mfe[i] = best
        mae[i] = worst

    holding_series = pd.Series(holding, index=bars.index, name="holding_bars")
    return LabelResult(
        outcome=pd.Series(outcome, index=bars.index, name="outcome"),
        holding_bars=holding_series,
        mfe_r=pd.Series(mfe, index=bars.index, name="mfe_r"),
        mae_r=pd.Series(mae, index=bars.index, name="mae_r"),
        uniqueness=_label_uniqueness(holding_series),
    )


def _label_uniqueness(holding_bars: pd.Series) -> pd.Series:
    """Fraction of each label's span not shared with other labels.

    Triple-barrier labels overlap: a label at t spanning h bars shares outcome
    information with labels at t+1…t+h. Treating them as independent inflates
    the effective sample size and every p-value derived from it. This is used
    as a sample weight during training.
    """
    n = len(holding_bars)
    spans = holding_bars.to_numpy(dtype="float64")

    # How many labels are live at each bar.
    concurrency = np.zeros(n, dtype="float64")
    for i in range(n):
        h = spans[i]
        if not np.isfinite(h) or h <= 0:
            continue
        concurrency[i : min(i + int(h) + 1, n)] += 1.0

    uniqueness = np.full(n, np.nan)
    for i in range(n):
        h = spans[i]
        if not np.isfinite(h) or h <= 0:
            continue
        window = concurrency[i : min(i + int(h) + 1, n)]
        live = window[window > 0]
        if len(live):
            uniqueness[i] = float(np.mean(1.0 / live))

    return pd.Series(uniqueness, index=holding_bars.index, name="uniqueness")
