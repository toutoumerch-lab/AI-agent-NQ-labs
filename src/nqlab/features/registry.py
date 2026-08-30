"""Feature registry.

Every feature is registered here. That is not bookkeeping — it is what makes
the leakage guarantee mechanical rather than aspirational.

The truncation property test (`tests/test_leakage.py`) iterates the registry,
so a feature is covered by the causality check the moment it is registered.
There is no way to add a leaky feature and leave the suite green, and no
reviewer has to remember to add a test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import pandas as pd

#: A feature takes the bar frame and returns a Series aligned to its index.
FeatureFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Feature:
    name: str
    fn: FeatureFn
    #: Family, for ablation testing (see research/07_ablation.md).
    family: str
    description: str
    #: Bars of history needed before the value is meaningful. Leading rows are
    #: NaN and must be dropped before training rather than imputed — imputing
    #: a warm-up value invents data.
    warmup: int = 0
    #: Set True only for features that legitimately need future bars (labels).
    #: Registered features with this flag are excluded from the causality test
    #: and may never be used as model inputs.
    is_label: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)


class FeatureRegistry:
    def __init__(self) -> None:
        self._features: dict[str, Feature] = {}

    def register(
        self,
        name: str,
        *,
        family: str,
        description: str,
        warmup: int = 0,
        is_label: bool = False,
        tags: Iterable[str] = (),
    ) -> Callable[[FeatureFn], FeatureFn]:
        """Decorator registering a feature function."""

        def decorator(fn: FeatureFn) -> FeatureFn:
            if name in self._features:
                raise ValueError(f"Feature {name!r} is already registered")
            self._features[name] = Feature(
                name=name,
                fn=fn,
                family=family,
                description=description,
                warmup=warmup,
                is_label=is_label,
                tags=tuple(tags),
            )
            return fn

        return decorator

    def get(self, name: str) -> Feature:
        try:
            return self._features[name]
        except KeyError:
            raise KeyError(f"Unknown feature {name!r}") from None

    def all(self) -> list[Feature]:
        return list(self._features.values())

    def inputs(self) -> list[Feature]:
        """Features usable as model inputs — everything that is not a label."""
        return [f for f in self._features.values() if not f.is_label]

    def by_family(self, family: str) -> list[Feature]:
        return [f for f in self._features.values() if f.family == family]

    def families(self) -> list[str]:
        return sorted({f.family for f in self._features.values()})

    def build(self, bars: pd.DataFrame, *, names: Iterable[str] | None = None) -> pd.DataFrame:
        """Compute features into a frame aligned to `bars`.

        Warm-up rows are left as NaN rather than filled. Dropping them is the
        caller's decision and must happen before splitting, so that a fold
        boundary never lands inside a warm-up window.
        """
        selected = [self.get(n) for n in names] if names is not None else self.inputs()
        out = pd.DataFrame(index=bars.index)
        for feature in selected:
            out[feature.name] = feature.fn(bars)
        return out

    @property
    def max_warmup(self) -> int:
        """Longest warm-up across registered features."""
        return max((f.warmup for f in self._features.values()), default=0)


#: The global registry. Importing `nqlab.features` populates it.
registry = FeatureRegistry()
