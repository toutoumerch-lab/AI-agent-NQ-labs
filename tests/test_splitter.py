"""Walk-forward splitter tests.

These check the properties that, if violated, make every downstream metric
meaningless — not that the code runs.
"""

from __future__ import annotations

import pytest

from nqlab.backtest.splitter import (
    Fold,
    WalkForwardSplitter,
    assert_no_overlap,
    embargo_is_sufficient,
)


def test_folds_are_chronological_and_train_precedes_test() -> None:
    splitter = WalkForwardSplitter(train_size=100, test_size=20, val_size=20, embargo=5)
    folds = list(splitter.split(300))

    assert folds
    for fold in folds:
        assert fold.train_start < fold.train_end <= fold.val_start < fold.val_end
        assert fold.val_end < fold.test_start, "training must end before test begins"
        assert fold.test_start < fold.test_end


def test_test_folds_never_overlap() -> None:
    """Overlapping test folds double-count trades and shrink apparent variance."""
    splitter = WalkForwardSplitter(train_size=100, test_size=25, embargo=3)
    folds = list(splitter.split(400))
    assert_no_overlap(folds)

    starts = [f.test_start for f in folds]
    assert starts == sorted(starts)


def test_embargo_separates_training_from_test() -> None:
    """The embargo must be at least the label horizon, or labels leak across."""
    horizon = 12
    splitter = WalkForwardSplitter(train_size=200, test_size=50, embargo=horizon)
    folds = list(splitter.split(600))

    assert embargo_is_sufficient(folds, horizon)
    for fold in folds:
        assert fold.test_start - fold.val_end == horizon


def test_insufficient_embargo_is_detected() -> None:
    splitter = WalkForwardSplitter(train_size=200, test_size=50, embargo=2)
    folds = list(splitter.split(600))
    assert not embargo_is_sufficient(folds, label_horizon=20)


def test_anchored_training_window_grows() -> None:
    splitter = WalkForwardSplitter(train_size=100, test_size=20, embargo=0, anchored=True)
    folds = list(splitter.split(300))

    assert all(f.train_start == 0 for f in folds)
    sizes = [f.val_end - f.train_start for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_rolling_training_window_stays_fixed() -> None:
    splitter = WalkForwardSplitter(train_size=100, test_size=20, embargo=0, anchored=False)
    folds = list(splitter.split(300))

    sizes = [f.val_end - f.train_start for f in folds]
    assert len(set(sizes)) == 1, "a rolling window must keep a constant span"


def test_no_folds_when_data_is_too_short() -> None:
    """Silently returning nothing is correct; inventing a fold is not."""
    splitter = WalkForwardSplitter(train_size=500, test_size=100, embargo=10)
    assert list(splitter.split(200)) == []


def test_every_test_index_appears_at_most_once() -> None:
    """Across all folds, no observation may be tested twice."""
    splitter = WalkForwardSplitter(train_size=80, test_size=15, embargo=4)
    folds = list(splitter.split(500))

    tested: set[int] = set()
    for fold in folds:
        indices = set(range(fold.test_start, fold.test_end))
        assert not (tested & indices), f"fold {fold.index} retests earlier observations"
        tested |= indices


def test_training_never_includes_a_test_index() -> None:
    """The property everything else depends on."""
    splitter = WalkForwardSplitter(train_size=120, test_size=30, val_size=20, embargo=8)
    folds = list(splitter.split(600))

    for fold in folds:
        train_indices = set(range(fold.train_start, fold.val_end))
        test_indices = set(range(fold.test_start, fold.test_end))
        assert not (train_indices & test_indices)


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        WalkForwardSplitter(train_size=0, test_size=10)
    with pytest.raises(ValueError):
        WalkForwardSplitter(train_size=100, test_size=0)
    with pytest.raises(ValueError):
        WalkForwardSplitter(train_size=100, test_size=10, val_size=100)
    with pytest.raises(ValueError):
        WalkForwardSplitter(train_size=100, test_size=10, embargo=-1)


def test_fold_rejects_inverted_ranges() -> None:
    with pytest.raises(ValueError):
        Fold(
            index=0,
            train_start=0,
            train_end=50,
            val_start=40,
            val_end=60,
            test_start=70,
            test_end=80,
        )
    with pytest.raises(ValueError):
        Fold(
            index=0,
            train_start=0,
            train_end=50,
            val_start=50,
            val_end=80,
            test_start=70,
            test_end=90,
        )
