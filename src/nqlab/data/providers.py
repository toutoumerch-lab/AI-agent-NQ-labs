"""Market data providers.

The system depends on the `MarketDataProvider` protocol, never on a concrete
source, so the data vendor can change without touching features, models or
backtests. Three implementations ship:

- `CsvProvider`     — files you supply. Works offline, and is the reference
                      implementation the others are checked against.
- `YFinanceProvider`— free vendor data. Requires network access.
- `ReplayProvider`  — wraps another provider and yields bars one at a time,
                      refusing to return anything after the current cursor.
                      This is the causality harness: a component that leaks
                      cannot be fed by it without raising.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

#: The canonical bar schema. Every provider returns exactly these columns,
#: indexed by a timezone-aware DatetimeIndex named "timestamp".
BAR_COLUMNS = ("open", "high", "low", "close", "volume")


class DataQualityError(Exception):
    """Raised when data violates an invariant that must hold before modelling."""


@dataclass(frozen=True)
class BarRequest:
    symbol: str
    #: Pandas offset alias: "1min", "5min", "1h", "1D".
    interval: str
    start: datetime | None = None
    end: datetime | None = None


class MarketDataProvider(ABC):
    """Source of OHLCV bars.

    Implementations must return bars that are sorted, unique, timezone-aware
    (UTC), and free of the obvious corruptions checked by `validate_bars`.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def get_bars(self, request: BarRequest) -> pd.DataFrame:
        """Return bars for the request, or an empty frame with the right schema."""


class HistoricalDataProvider(MarketDataProvider):
    """Marker for providers serving a fixed historical range."""


class LiveDataProvider(MarketDataProvider):
    """Marker for providers serving data up to now."""

    @abstractmethod
    def get_latest_bar(self, symbol: str, interval: str) -> pd.Series | None: ...


def empty_bars() -> pd.DataFrame:
    """An empty frame with the canonical schema.

    Returned instead of None so callers never branch on nullness, and so an
    empty result still has the right dtypes for downstream concatenation.
    """
    frame = pd.DataFrame(columns=list(BAR_COLUMNS), dtype="float64")
    frame.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return frame


def validate_bars(bars: pd.DataFrame, *, symbol: str = "?") -> None:
    """Assert the invariants every downstream component relies on.

    Raises rather than repairing. Silent repair of bad market data is how a
    corrupt series reaches a model without anyone noticing; the caller should
    decide explicitly (see `clean_bars`).
    """
    missing = set(BAR_COLUMNS) - set(bars.columns)
    if missing:
        raise DataQualityError(f"{symbol}: missing columns {sorted(missing)}")

    if len(bars) == 0:
        return

    if not isinstance(bars.index, pd.DatetimeIndex):
        raise DataQualityError(f"{symbol}: index must be a DatetimeIndex")

    if bars.index.tz is None:
        raise DataQualityError(
            f"{symbol}: index must be timezone-aware — naive timestamps are "
            "ambiguous across DST transitions"
        )

    if not bars.index.is_monotonic_increasing:
        raise DataQualityError(f"{symbol}: bars are not sorted by time")

    if bars.index.has_duplicates:
        dupes = bars.index[bars.index.duplicated()][:5]
        raise DataQualityError(f"{symbol}: duplicate timestamps, e.g. {list(dupes)}")

    # OHLC consistency. A bar whose high is below its open is not a bar; it is
    # a vendor error, and feeding it to an ATR calculation produces a negative
    # true range that then propagates into every volatility feature.
    bad_high = bars["high"] < bars[["open", "close", "low"]].max(axis=1)
    bad_low = bars["low"] > bars[["open", "close", "high"]].min(axis=1)
    if bad_high.any() or bad_low.any():
        n = int(bad_high.sum() + bad_low.sum())
        first = bars.index[bad_high | bad_low][0]
        raise DataQualityError(
            f"{symbol}: {n} bars violate high >= max(o,c,l) or low <= min(o,c,h), first at {first}"
        )

    if (bars[["open", "high", "low", "close"]] <= 0).to_numpy().any():
        raise DataQualityError(f"{symbol}: non-positive prices present")

    if (bars["volume"] < 0).any():
        raise DataQualityError(f"{symbol}: negative volume present")


def normalise_bars(frame: pd.DataFrame, *, tz: str = "UTC") -> pd.DataFrame:
    """Coerce a raw vendor frame into the canonical schema.

    Handles the three things every vendor gets differently: column naming,
    timezone, and duplicate timestamps.
    """
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    aliases = {"vol": "volume", "o": "open", "h": "high", "l": "low", "c": "close"}
    out = out.rename(columns={k: v for k, v in aliases.items() if k in out.columns})

    missing = set(BAR_COLUMNS) - set(out.columns)
    if missing:
        raise DataQualityError(f"Cannot normalise: missing {sorted(missing)}")

    out = out[list(BAR_COLUMNS)].astype("float64")

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    out.index = out.index.tz_localize(tz) if out.index.tz is None else out.index.tz_convert(tz)
    out.index.name = "timestamp"

    # Keep the last of any duplicate timestamps: vendors emit a provisional bar
    # and then a corrected one, and the correction arrives second.
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


class CsvProvider(HistoricalDataProvider):
    """Bars from CSV or Parquet files on disk.

    Expects `{root}/{symbol}/{interval}.csv` (or `.parquet`). This is the
    provider to use when you have your own data — it needs no network and no
    credentials, and it is the reference the others are validated against.
    """

    def __init__(self, root: str | Path, *, tz: str = "UTC") -> None:
        self.root = Path(root)
        self.tz = tz

    @property
    def name(self) -> str:
        return f"csv:{self.root}"

    def _path(self, symbol: str, interval: str) -> Path:
        base = self.root / symbol.upper()
        for suffix in (".parquet", ".csv"):
            candidate = base / f"{interval}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No data for {symbol} {interval} under {base}. "
            f"Expected {base / (interval + '.csv')} or .parquet"
        )

    def get_bars(self, request: BarRequest) -> pd.DataFrame:
        path = self._path(request.symbol, request.interval)

        if path.suffix == ".parquet":
            raw = pd.read_parquet(path)
        else:
            raw = pd.read_csv(path, index_col=0, parse_dates=True)

        bars = normalise_bars(raw, tz=self.tz)

        if request.start is not None:
            bars = bars[bars.index >= pd.Timestamp(request.start, tz=self.tz)]
        if request.end is not None:
            bars = bars[bars.index <= pd.Timestamp(request.end, tz=self.tz)]

        validate_bars(bars, symbol=request.symbol)
        return bars


class YFinanceProvider(HistoricalDataProvider):
    """Bars from Yahoo Finance via `yfinance`.

    Free and convenient, with real limitations that are stated rather than
    discovered later: intraday history is capped (roughly 60 days at 1-minute),
    volume is unreliable for futures, and the continuous "NQ=F" series is
    stitched by the vendor with an undocumented method — so it is adequate for
    development and unsuitable as the basis of a published result.
    """

    def __init__(self, *, tz: str = "UTC") -> None:
        self.tz = tz

    @property
    def name(self) -> str:
        return "yfinance"

    def get_bars(self, request: BarRequest) -> pd.DataFrame:
        try:
            import yfinance  # imported lazily: the package is optional
        except ImportError as exc:
            raise ImportError(
                "YFinanceProvider requires `yfinance`. Install it, or use "
                "CsvProvider with your own data."
            ) from exc

        ticker = yfinance.Ticker(request.symbol)
        raw = ticker.history(
            interval=request.interval.replace("min", "m"),
            start=request.start,
            end=request.end,
            auto_adjust=False,
        )
        if raw.empty:
            return empty_bars()

        bars = normalise_bars(raw, tz=self.tz)
        validate_bars(bars, symbol=request.symbol)
        return bars


class ReplayProvider(MarketDataProvider):
    """Serves a historical dataset one bar at a time, refusing to look ahead.

    This is the causality harness. A component fed by a `ReplayProvider` cannot
    accidentally read a future bar, because future bars are not reachable —
    asking for them raises rather than returning data. Any strategy, feature
    pipeline or agent can be run against it to demonstrate that it never
    consults information it would not have had at decision time.
    """

    def __init__(self, bars: pd.DataFrame, *, symbol: str = "NQ") -> None:
        validate_bars(bars, symbol=symbol)
        self._bars = bars
        self._symbol = symbol
        self._cursor = 0

    @property
    def name(self) -> str:
        return f"replay:{self._symbol}"

    @property
    def cursor(self) -> int:
        """Index of the next bar to be revealed."""
        return self._cursor

    @property
    def current_time(self) -> pd.Timestamp | None:
        """Timestamp of the most recently revealed bar."""
        if self._cursor == 0:
            return None
        return self._bars.index[self._cursor - 1]

    def advance(self) -> pd.Series | None:
        """Reveal the next bar, or None when the dataset is exhausted."""
        if self._cursor >= len(self._bars):
            return None
        bar = self._bars.iloc[self._cursor]
        self._cursor += 1
        return bar

    def __iter__(self) -> Iterator[pd.Series]:
        while (bar := self.advance()) is not None:
            yield bar

    def get_bars(self, request: BarRequest) -> pd.DataFrame:
        """Return only bars already revealed.

        `request.end` beyond the cursor is silently clamped rather than raising,
        because asking for "everything up to now" is the normal case. Asking for
        a window that *starts* in the future is a bug, and raises.
        """
        visible = self._bars.iloc[: self._cursor]

        if request.start is not None:
            start = pd.Timestamp(request.start)
            if self.current_time is not None and start > self.current_time:
                raise DataQualityError(
                    f"Requested window starts at {start}, which is after the current "
                    f"replay time {self.current_time}. This is a look-ahead."
                )
            visible = visible[visible.index >= start]
        if request.end is not None:
            visible = visible[visible.index <= pd.Timestamp(request.end)]

        return visible

    def reset(self) -> None:
        self._cursor = 0
