"""Walk-forward splitting with embargo.

Time-series cross-validation is not k-fold with a different shuffle. Three
things must hold, and each is a separate way results get inflated:

1. **Chronological order.** Training on a future fold to predict a past one is
   not a mistake you can bound — it is unbounded optimism.
2. **An embargo.** Path-dependent labels (triple-barrier) resolve up to H bars
   after their entry. Without a gap, the last training labels resolve using
   bars inside the test fold.
3. **Test folds are touched once.** Selection on test is the slowest and most
   convincing way to fool yourself, because every individual step looks
   reasonable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold. Index positions, not labels."""

    index: int
    train_start: int
    train_end: int
    #: Validation slice, carved from the END of training. Hyperparameters are
    #: chosen here — never on test.
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start

    def __post_init__(self) -> None:
        if not (self.train_start <= self.train_end <= self.val_start <= self.val_end):
            raise ValueError(f"Fold {self.index}: train/val ranges overlap or invert")
        if self.val_end > self.test_start:
            raise ValueError(f"Fold {self.index}: validation overlaps test")
        if self.test_start >= self.test_end:
            raise ValueError(f"Fold {self.index}: empty test slice")


class WalkForwardSplitter:
    """Anchored or rolling walk-forward splits.

    ```
    fold 1  ├── train ──┤─val─┤▒embargo▒├─ test ─┤
    fold 2  ├──── train ────┤─val─┤▒embargo▒├─ test ─┤
    ```

    Args:
        train_size: Bars in the initial training window.
        test_size: Bars in each test fold.
        val_size: Bars reserved from the end of training for model selection.
        embargo: Bars discarded between validation and test. **Must be at least
            the maximum label horizon**, or labels leak across the boundary.
        anchored: True grows the training window from a fixed start (more data,
            assumes old regimes still inform). False slides a fixed window
            (adapts, discards history). Both are testable; neither is obviously
            right, which is why it is a parameter and not a decision.
    """

    def __init__(
        self,
        *,
        train_size: int,
        test_size: int,
        val_size: int = 0,
        embargo: int = 0,
        anchored: bool = True,
    ) -> None:
        if train_size <= 0 or test_size <= 0:
            raise ValueError("train_size and test_size must be positive")
        if val_size < 0 or embargo < 0:
            raise ValueError("val_size and embargo must be non-negative")
        if val_size >= train_size:
            raise ValueError("val_size must be smaller than train_size")

        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.embargo = embargo
        self.anchored = anchored

    def split(self, n_samples: int) -> Iterator[Fold]:
        """Yield folds covering `n_samples` observations in time order."""
        fold_index = 0
        # Start of the first test fold.
        test_start = self.train_size + self.embargo

        while test_start + self.test_size <= n_samples:
            train_end_total = test_start - self.embargo
            train_start = 0 if self.anchored else max(0, train_end_total - self.train_size)

            val_start = train_end_total - self.val_size
            fold = Fold(
                index=fold_index,
                train_start=train_start,
                train_end=val_start,
                val_start=val_start,
                val_end=train_end_total,
                test_start=test_start,
                test_end=test_start + self.test_size,
            )
            yield fold

            fold_index += 1
            test_start += self.test_size

    def n_splits(self, n_samples: int) -> int:
        return sum(1 for _ in self.split(n_samples))


def assert_no_overlap(folds: list[Fold]) -> None:
    """Assert test folds are disjoint and chronologically ordered.

    Overlapping test folds double-count trades, which inflates trade counts and
    shrinks apparent variance — making a strategy look more reliable than it is.
    """
    for earlier, later in pairwise(folds):
        if later.test_start < earlier.test_end:
            raise ValueError(
                f"Test folds {earlier.index} and {later.index} overlap: "
                f"{earlier.test_end} > {later.test_start}"
            )


def embargo_is_sufficient(folds: list[Fold], label_horizon: int) -> bool:
    """True if every train/test boundary is separated by at least the horizon."""
    return all(fold.test_start - fold.val_end >= label_horizon for fold in folds)
