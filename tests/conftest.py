"""Shared fixtures.

The bar generator here produces *synthetic* series. They are used only to
exercise engine correctness — causality, fills, arithmetic. No result derived
from synthetic data is ever reported as a finding about the market.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_bars(
    n: int = 500,
    *,
    seed: int = 0,
    start: str = "2026-01-05 09:30",
    freq: str = "5min",
    tz: str = "America/New_York",
    drift: float = 0.0,
    vol: float = 0.0015,
    start_price: float = 18_000.0,
) -> pd.DataFrame:
    """Synthetic OHLCV bars satisfying every schema invariant.

    A geometric random walk. Deliberately *not* meant to resemble NQ — its only
    job is to be valid, deterministic, and free of the degeneracies that would
    make a correctness test pass vacuously.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(steps))

    # Build OHLC around the close path so that high >= max(o,c) and
    # low <= min(o,c) hold by construction.
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    wiggle = np.abs(rng.normal(0, vol / 2, n)) * close
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    volume = rng.integers(500, 5_000, n).astype(float)

    index = pd.date_range(start, periods=n, freq=freq, tz=tz)
    index.name = "timestamp"
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture
def long_bars() -> pd.DataFrame:
    return make_bars(n=2_000, seed=7)
