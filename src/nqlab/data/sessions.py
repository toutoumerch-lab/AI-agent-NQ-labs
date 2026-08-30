"""Trading session classification.

NQ trades nearly around the clock, but the character of the market differs
sharply between segments: the RTH open carries most of the day's volume and
volatility, the overnight session is thin and gap-prone, and 17:00–18:00 ET is
a scheduled halt rather than missing data.

Getting this wrong corrupts features in ways that are hard to see. An "opening
range" computed on Globex bars is not an opening range, and a "missing bar"
alarm firing every day at 17:00 will train you to ignore the alarm.
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd

from .contracts import ContractSpec


class Session(StrEnum):
    """Which segment of the trading day a bar belongs to."""

    #: 09:30–16:00 ET. Most volume, tightest spreads.
    RTH = "RTH"
    #: 18:00–09:30 ET. Thin, gap-prone, driven by overseas markets.
    OVERNIGHT = "OVERNIGHT"
    #: 16:00–17:00 ET. Post-RTH, still open, liquidity falling away.
    POST_CLOSE = "POST_CLOSE"
    #: 17:00–18:00 ET. Exchange closed. No bars should exist here.
    HALT = "HALT"


def _minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


def classify_sessions(index: pd.DatetimeIndex, spec: ContractSpec) -> pd.Series:
    """Label each timestamp with its trading session.

    The index must be timezone-aware. A naive timestamp is ambiguous across DST
    transitions — 01:30 occurs twice on the November changeover — and silently
    guessing produces two bars that claim the same instant.
    """
    if index.tz is None:
        raise ValueError(
            "Timestamps must be timezone-aware before session classification. "
            "Naive timestamps are ambiguous across DST transitions."
        )

    local = index.tz_convert(spec.timezone)
    minute_of_day = local.hour * 60 + local.minute

    rth_open = _minutes(spec.rth_open)
    rth_close = _minutes(spec.rth_close)
    globex_open = _minutes(spec.globex_open)
    globex_close = _minutes(spec.globex_close)

    labels = pd.Series(Session.OVERNIGHT.value, index=index, dtype="object")
    labels[(minute_of_day >= rth_open) & (minute_of_day < rth_close)] = Session.RTH.value
    labels[(minute_of_day >= rth_close) & (minute_of_day < globex_close)] = Session.POST_CLOSE.value
    labels[(minute_of_day >= globex_close) & (minute_of_day < globex_open)] = Session.HALT.value

    return labels


def trading_date(index: pd.DatetimeIndex, spec: ContractSpec) -> pd.Series:
    """Map each timestamp to the *trading* date it belongs to.

    A futures trading day starts at the Globex open (18:00 ET) on the previous
    calendar day. Bars from Sunday 18:00 through Monday 17:00 are all Monday's
    session. Using the calendar date instead splits every overnight session in
    half, which breaks "previous day high", overnight range, and any daily
    aggregate built from intraday bars.
    """
    if index.tz is None:
        raise ValueError("Timestamps must be timezone-aware.")

    local = index.tz_convert(spec.timezone)
    minute_of_day = local.hour * 60 + local.minute
    globex_open = _minutes(spec.globex_open)

    dates = pd.Series(local.date, index=index)
    # At or after the Globex open, the bar belongs to the NEXT calendar day's
    # trading session.
    rolls_forward = minute_of_day >= globex_open
    dates[rolls_forward] = (local[rolls_forward] + pd.Timedelta(days=1)).date

    return dates


def is_rth(index: pd.DatetimeIndex, spec: ContractSpec) -> pd.Series:
    return classify_sessions(index, spec) == Session.RTH.value
