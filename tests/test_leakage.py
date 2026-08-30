"""Causality and leakage tests.

This is the most important file in the repository. Every claim the system makes
about out-of-sample performance rests on the guarantee tested here: **no feature
value at bar t depends on any bar after t**.

The central test is a truncation property. For a feature f, a dataset D and a
cut point t:

    f(D[0:t]).iloc[-1] == f(D).iloc[t-1]

Computing a feature on history truncated at t must give the identical value at
t-1 as computing it on the full series. Any peek at the future — a centred
window, a negative shift, a full-series fit, a rank against the whole sample —
breaks this equality.

It runs over every feature in the registry, so a newly registered feature is
covered automatically and a leaky one cannot pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars

import nqlab.features.technical  # noqa: F401 — populates the registry
from nqlab.features.registry import registry

ALL_FEATURES = [f.name for f in registry.inputs()]


@pytest.mark.parametrize("feature_name", ALL_FEATURES)
def test_feature_is_causal_under_truncation(feature_name: str) -> None:
    """A feature computed on truncated data must match the full-series value.

    This is the leakage test. If a feature uses information from after bar t,
    truncating the data at t changes its value at t-1, and this fails.
    """
    feature = registry.get(feature_name)
    full = make_bars(n=400, seed=11)
    full_values = feature.fn(full)

    # Every cut point past the warm-up, not a handful of samples.
    #
    # A sparse sweep was the original design and it had a hole: a leak that
    # only manifests at particular indices — a backward fill over periodic
    # gaps, say — slips between the samples. The meta-test in
    # test_leakage_detector_works.py caught that, so the sweep is now dense.
    warmup = max(feature.warmup, 1)
    for cut in range(warmup + 2, len(full) + 1):
        truncated_values = feature.fn(full.iloc[:cut])
        expected = full_values.iloc[cut - 1]
        actual = truncated_values.iloc[cut - 1]

        if pd.isna(expected) and pd.isna(actual):
            continue

        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12), (
            f"{feature_name} leaks future information: value at bar {cut - 1} is "
            f"{actual} when computed on data truncated at {cut}, but {expected} "
            f"when computed on the full series."
        )


@pytest.mark.parametrize("feature_name", ALL_FEATURES)
def test_feature_appending_a_bar_does_not_change_history(feature_name: str) -> None:
    """Appending a future bar must not alter any earlier value.

    The same guarantee approached from the other side, and the one that matters
    in live operation: the value shown at 10:05 must still be that value after
    the 10:10 bar arrives. A feature that silently revises history would make
    every backtest a fiction.
    """
    feature = registry.get(feature_name)
    # The extended series must be a true prefix-extension. Generating
    # make_bars(300) and make_bars(301) separately does NOT give that: the
    # generator draws OHLC, wiggle and volume in sequence, so a different `n`
    # shifts every later draw's position in the stream. Slicing one series is
    # the only way to get a genuine prefix.
    extended = make_bars(n=301, seed=13)
    base = extended.iloc[:300]

    before = feature.fn(base)
    after = feature.fn(extended).iloc[:300]

    pd.testing.assert_series_equal(
        before,
        after,
        check_names=False,
        obj=f"{feature_name} revised historical values when a new bar arrived",
    )


@pytest.mark.parametrize("feature_name", ALL_FEATURES)
def test_feature_output_is_aligned_and_finite(feature_name: str) -> None:
    """Shape and sanity: aligned index, no infinities.

    An infinity reaches a model as a silent NaN after scaling, or as a value
    that dominates every split in a tree. Division guards in the feature code
    exist for this; the test is what keeps them.
    """
    feature = registry.get(feature_name)
    frame = make_bars(n=400, seed=17)
    values = feature.fn(frame)

    assert isinstance(values, pd.Series)
    assert len(values) == len(frame)
    assert values.index.equals(frame.index)

    finite = values.to_numpy(dtype="float64")
    assert not np.isinf(finite).any(), f"{feature_name} produced infinite values"


@pytest.mark.parametrize("feature_name", ALL_FEATURES)
def test_feature_warmup_is_declared_honestly(feature_name: str) -> None:
    """The declared warm-up must cover the actual NaN prefix.

    An understated warm-up leaves NaNs in what the pipeline believes is valid
    data; the row is then dropped or imputed somewhere downstream, and the
    training set quietly differs from what the config says it is.
    """
    feature = registry.get(feature_name)
    frame = make_bars(n=400, seed=19)
    values = feature.fn(frame)

    valid = values.notna().to_numpy()
    if not valid.any():
        pytest.fail(f"{feature_name} is entirely NaN on 400 bars")

    first_valid = int(np.argmax(valid))
    assert first_valid <= feature.warmup, (
        f"{feature_name} declares warmup={feature.warmup} but its first non-NaN "
        f"value is at index {first_valid}. Training would begin on NaN rows."
    )


def test_feature_values_do_not_depend_on_unseen_future_scale() -> None:
    """A regime shift in the future must not change values in the past.

    Catches full-series normalisation specifically. A feature that divides by
    the standard deviation of the whole series will change every historical
    value when a volatile period is appended — a leak that the truncation test
    also catches, but which this states in the terms it actually occurs in.
    """
    quiet = make_bars(n=300, seed=23, vol=0.0005)
    # Same first 300 bars, then a violent regime.
    violent_tail = make_bars(n=100, seed=29, vol=0.02, start_price=float(quiet["close"].iloc[-1]))
    violent_tail.index = pd.date_range(
        quiet.index[-1] + pd.Timedelta("5min"), periods=100, freq="5min", tz=quiet.index.tz
    )
    combined = pd.concat([quiet, violent_tail])

    for feature in registry.inputs():
        before = feature.fn(quiet)
        after = feature.fn(combined).iloc[:300]
        pd.testing.assert_series_equal(
            before,
            after,
            check_names=False,
            obj=f"{feature.name} changed historical values when a volatile future was appended",
        )


def test_registry_has_no_unregistered_label_used_as_input() -> None:
    """Labels must never be reachable as model inputs."""
    for feature in registry.inputs():
        assert not feature.is_label
        assert "label" not in feature.name.lower(), (
            f"{feature.name} looks like a label but is registered as an input"
        )
