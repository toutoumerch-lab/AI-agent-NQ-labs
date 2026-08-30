"""Meta-tests: prove the leakage detector actually detects leakage.

A test suite that passes on everything is worthless. These tests construct
features that leak in each of the ways real code leaks, and assert that the
causality checks in `test_leakage.py` fail on them.

Without this file, "all leakage tests pass" means nothing — it could equally
mean the checks are vacuous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars

# ─── Deliberately leaky features ──────────────────────────────────────────
# Each is a realistic mistake, not a strawman.


def leak_negative_shift(bars: pd.DataFrame) -> pd.Series:
    """Reads the next bar directly. The textbook look-ahead."""
    return bars["close"].shift(-1) / bars["close"] - 1


def leak_centred_window(bars: pd.DataFrame) -> pd.Series:
    """A centred rolling mean. Half of every window is in the future.

    Easy to write by accident — `center=True` reads as a smoothing choice
    rather than a time-travel one.
    """
    return bars["close"].rolling(20, center=True, min_periods=1).mean()


def leak_full_series_zscore(bars: pd.DataFrame) -> pd.Series:
    """Normalises by the mean and standard deviation of the whole series.

    The most common leak in published strategy code, because it looks like
    ordinary preprocessing rather than a peek at the future.
    """
    returns = np.log(bars["close"] / bars["close"].shift(1))
    return (returns - returns.mean()) / returns.std()


def leak_full_series_rank(bars: pd.DataFrame) -> pd.Series:
    """Ranks each value against the entire sample, including the future."""
    return bars["volume"].rank(pct=True)


def leak_expanding_max_of_future(bars: pd.DataFrame) -> pd.Series:
    """Distance from the series maximum — which may not have happened yet."""
    return bars["close"] / bars["close"].max() - 1


def leak_backfill(bars: pd.DataFrame) -> pd.Series:
    """Backward fill. Pulls a later value into an earlier gap."""
    series = np.log(bars["close"] / bars["close"].shift(1))
    series.iloc[::7] = np.nan
    return series.bfill()


LEAKY_FEATURES = {
    "negative_shift": leak_negative_shift,
    "centred_window": leak_centred_window,
    "full_series_zscore": leak_full_series_zscore,
    "full_series_rank": leak_full_series_rank,
    "max_of_future": leak_expanding_max_of_future,
    "backfill": leak_backfill,
}


# ─── The checks under test, extracted so they can be applied to any fn ────


def truncation_check(fn, *, n: int = 400, seed: int = 11) -> bool:
    """True if `fn` is causal under truncation. Mirrors the real test.

    Sweeps every cut point rather than sampling: a sparse sweep missed the
    backfill leak entirely, because none of the sampled indices happened to
    land on a gap row.
    """
    full = make_bars(n=n, seed=seed)
    full_values = fn(full)

    for cut in range(3, n + 1):
        truncated = fn(full.iloc[:cut])
        expected, actual = full_values.iloc[cut - 1], truncated.iloc[cut - 1]
        if pd.isna(expected) and pd.isna(actual):
            continue
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True):
            return False
    return True


def append_check(fn, *, n: int = 300, seed: int = 13) -> bool:
    """True if appending a bar leaves history unchanged."""
    extended = make_bars(n=n + 1, seed=seed)
    base = extended.iloc[:n]
    before = fn(base).to_numpy(dtype="float64")
    after = fn(extended).iloc[:n].to_numpy(dtype="float64")
    return np.allclose(before, after, rtol=1e-12, atol=1e-12, equal_nan=True)


# ─── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(LEAKY_FEATURES))
def test_detector_catches_every_leak(name: str) -> None:
    """Every deliberately leaky feature must fail at least one causality check.

    If a leak passes both, the detector has a blind spot and the suite's green
    status is not evidence of anything.
    """
    fn = LEAKY_FEATURES[name]
    causal_under_truncation = truncation_check(fn)
    stable_under_append = append_check(fn)

    assert not (causal_under_truncation and stable_under_append), (
        f"The leakage detector FAILED to catch {name!r}. This is a hole in the "
        "causality guarantee: a feature using future information passed both "
        "the truncation and append checks."
    )


@pytest.mark.parametrize("name", sorted(LEAKY_FEATURES))
def test_leak_report_identifies_which_check_fired(name: str) -> None:
    """Record which check catches which leak, so coverage is visible.

    Not every leak trips every check — a full-series rank changes history on
    append but a negative shift is caught by truncation. Knowing the mapping is
    what tells you the two checks are complementary rather than redundant.
    """
    fn = LEAKY_FEATURES[name]
    caught_by = []
    if not truncation_check(fn):
        caught_by.append("truncation")
    if not append_check(fn):
        caught_by.append("append")

    assert caught_by, f"{name} escaped both checks"
    print(f"\n  {name:22} caught by: {', '.join(caught_by)}")


def test_a_genuinely_causal_feature_passes_both_checks() -> None:
    """The converse: a correct feature must NOT trip the detector.

    Without this, a detector that returned False unconditionally would pass
    every test above while being useless.
    """

    def causal_sma(bars: pd.DataFrame) -> pd.Series:
        return bars["close"].rolling(20, min_periods=20).mean()

    assert truncation_check(causal_sma)
    assert append_check(causal_sma)
