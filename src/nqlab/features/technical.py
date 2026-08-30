"""Price, volatility, trend, structure and volume features.

Every function here obeys one rule: **the value at index t may depend only on
rows 0…t**. No centred windows, no negative shifts, no full-series fitting.

Where pandas offers a convenient function that violates this, it is not used —
`rolling(...).mean()` is causal, `ewm(...).mean()` is causal, but
`rolling(..., center=True)` is not, and neither is anything that calls `.fit()`
on the whole series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import registry

# ─── Price ────────────────────────────────────────────────────────────────


@registry.register(
    "log_return_1",
    family="price",
    description="Log return over the previous bar.",
    warmup=1,
)
def log_return_1(bars: pd.DataFrame) -> pd.Series:
    """Log rather than simple returns: they are additive across time, which
    makes multi-bar aggregation a sum, and they are closer to symmetric."""
    return np.log(bars["close"] / bars["close"].shift(1))


@registry.register(
    "momentum_20",
    family="price",
    description="Log return over the previous 20 bars.",
    warmup=20,
)
def momentum_20(bars: pd.DataFrame) -> pd.Series:
    return np.log(bars["close"] / bars["close"].shift(20))


@registry.register(
    "range_position_20",
    family="price",
    description="Where the close sits inside the trailing 20-bar range, in [0,1].",
    warmup=20,
)
def range_position_20(bars: pd.DataFrame) -> pd.Series:
    """0 means the close is at the low of the window, 1 at the high.

    The window is trailing and *includes* the current bar, which is causal: the
    current bar's own high and low are known once it has closed.
    """
    high = bars["high"].rolling(20, min_periods=20).max()
    low = bars["low"].rolling(20, min_periods=20).min()
    span = high - low
    # A perfectly flat window has zero range. Return 0.5 (the midpoint) rather
    # than NaN or an infinity, so a rare flat window does not drop the row.
    return ((bars["close"] - low) / span).where(span > 0, 0.5)


# ─── Volatility ───────────────────────────────────────────────────────────


def _true_range(bars: pd.DataFrame) -> pd.Series:
    """True range, accounting for gaps against the previous close."""
    previous_close = bars["close"].shift(1)
    return pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous_close).abs(),
            (bars["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


@registry.register(
    "atr_14",
    family="volatility",
    description="14-bar average true range (Wilder smoothing).",
    warmup=14,
)
def atr_14(bars: pd.DataFrame) -> pd.Series:
    """Wilder's smoothing is an EWM with alpha = 1/n, which is causal and needs
    no full-series pass."""
    return _true_range(bars).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()


@registry.register(
    "atr_pct_14",
    family="volatility",
    description="ATR as a fraction of price — comparable across price levels.",
    warmup=14,
)
def atr_pct_14(bars: pd.DataFrame) -> pd.Series:
    """Raw ATR is not comparable across time: 50 points at NQ 5,000 is a very
    different market from 50 points at 20,000."""
    return atr_14(bars) / bars["close"]


@registry.register(
    "realised_vol_20",
    family="volatility",
    description="Standard deviation of the last 20 log returns.",
    warmup=21,
)
def realised_vol_20(bars: pd.DataFrame) -> pd.Series:
    return log_return_1(bars).rolling(20, min_periods=20).std()


@registry.register(
    "vol_percentile_100",
    family="volatility",
    description="Rank of current realised vol within the trailing 100 bars, in [0,1].",
    warmup=120,
)
def vol_percentile_100(bars: pd.DataFrame) -> pd.Series:
    """A trailing *rank*, not a z-score against the full sample.

    Ranking against the whole series would use future volatility to describe
    the past — one of the most common and least visible leaks in published
    strategy code.
    """
    vol = realised_vol_20(bars)
    return vol.rolling(100, min_periods=100).rank(pct=True)


# ─── Trend ────────────────────────────────────────────────────────────────


@registry.register(
    "ema_ratio_20_50",
    family="trend",
    description="Fast EMA over slow EMA, minus one. Positive means uptrend.",
    warmup=50,
)
def ema_ratio_20_50(bars: pd.DataFrame) -> pd.Series:
    fast = bars["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    slow = bars["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    return fast / slow - 1.0


@registry.register(
    "ema_slope_20",
    family="trend",
    description="Slope of the 20-bar EMA over 5 bars, normalised by price.",
    warmup=25,
)
def ema_slope_20(bars: pd.DataFrame) -> pd.Series:
    ema = bars["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    return (ema - ema.shift(5)) / bars["close"]


@registry.register(
    "trend_strength_20",
    family="trend",
    description="Net directional movement over 20 bars divided by total movement.",
    warmup=21,
)
def trend_strength_20(bars: pd.DataFrame) -> pd.Series:
    """An efficiency ratio in [0,1]: 1 is a straight line, 0 is pure chop.

    Distinguishes a trend from a market that travelled the same distance while
    oscillating — which momentum alone cannot.
    """
    net = (bars["close"] - bars["close"].shift(20)).abs()
    total = bars["close"].diff().abs().rolling(20, min_periods=20).sum()
    return (net / total).where(total > 0, 0.0)


# ─── Market structure ─────────────────────────────────────────────────────


@registry.register(
    "dist_from_high_50",
    family="structure",
    description="Distance below the trailing 50-bar high, in ATR units.",
    warmup=64,
)
def dist_from_high_50(bars: pd.DataFrame) -> pd.Series:
    """Measured in ATR rather than points so it is comparable across
    volatility regimes: 30 points from the high is near in a fast market and
    far in a quiet one."""
    high = bars["high"].rolling(50, min_periods=50).max()
    atr = atr_14(bars)
    return ((high - bars["close"]) / atr).where(atr > 0)


@registry.register(
    "dist_from_low_50",
    family="structure",
    description="Distance above the trailing 50-bar low, in ATR units.",
    warmup=64,
)
def dist_from_low_50(bars: pd.DataFrame) -> pd.Series:
    low = bars["low"].rolling(50, min_periods=50).min()
    atr = atr_14(bars)
    return ((bars["close"] - low) / atr).where(atr > 0)


@registry.register(
    "range_compression_20",
    family="structure",
    description="Recent 5-bar range over the 20-bar range. Low means coiling.",
    warmup=20,
)
def range_compression_20(bars: pd.DataFrame) -> pd.Series:
    recent = (
        bars["high"].rolling(5, min_periods=5).max() - bars["low"].rolling(5, min_periods=5).min()
    )
    wide = (
        bars["high"].rolling(20, min_periods=20).max()
        - bars["low"].rolling(20, min_periods=20).min()
    )
    return (recent / wide).where(wide > 0, 1.0)


# ─── Volume ───────────────────────────────────────────────────────────────


@registry.register(
    "relative_volume_20",
    family="volume",
    description="Volume over its trailing 20-bar mean.",
    warmup=20,
)
def relative_volume_20(bars: pd.DataFrame) -> pd.Series:
    mean = bars["volume"].rolling(20, min_periods=20).mean()
    return (bars["volume"] / mean).where(mean > 0, 1.0)


@registry.register(
    "volume_trend_correlation_20",
    family="volume",
    description="Rolling correlation of volume with absolute return.",
    warmup=21,
)
def volume_trend_correlation_20(bars: pd.DataFrame) -> pd.Series:
    """Whether moves are being confirmed by participation, or drifting on air."""
    return bars["volume"].rolling(20, min_periods=20).corr(log_return_1(bars).abs())
