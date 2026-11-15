"""Jalali / Gregorian date-time parsing and plausibility checks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import jdatetime

from .normalize import normalize_text

TEHRAN = timezone(timedelta(hours=3, minutes=30))  # Iran has no DST since 2022

_DATE_RE = re.compile(r"(?<!\d)(\d{2,4})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{1,4})(?!\d)")
_TIME_RE = re.compile(r"(?<![\d/])([01]?\d|2[0-3])\s*:\s*([0-5]\d)(?:\s*:\s*([0-5]\d))?(?!\d)")


@dataclass(frozen=True)
class ParsedDate:
    raw: str
    jalali: str               # "1405/08/24 11:42" (as printed, normalized digits)
    calendar: str             # "jalali" | "gregorian"
    iso: Optional[str]        # ISO-8601 with +03:30, only when the date (and time) parsed
    dt: Optional[datetime]    # aware datetime, None when time is missing
    date_only: bool


def _mk_jalali(y: int, m: int, d: int) -> Optional[jdatetime.date]:
    try:
        return jdatetime.date(y, m, d)
    except ValueError:
        return None


def parse_datetime(text: Optional[str]) -> Optional[ParsedDate]:
    """Find the first date (and the time on the same text, if any).

    Handles 1405/08/24, 1405-8-24, 05/08/24 (two-digit Jalali year), 24/08/1405
    (day first), and Gregorian 2026/11/15. The original string is kept.
    """
    if not text:
        return None
    t = normalize_text(text)
    for m in _DATE_RE.finditer(t):
        p = _parse_match(t, m)
        if p is not None:
            return p
    return None


def _parse_match(t: str, m: re.Match) -> Optional[ParsedDate]:
    a, b, c = m.group(1), m.group(2), m.group(3)
    y = mo = d = None
    if len(a) == 4:
        y, mo, d = int(a), int(b), int(c)
    elif len(c) == 4:
        y, mo, d = int(c), int(b), int(a)
    elif len(a) == 2 and len(c) <= 2:
        y, mo, d = 1400 + int(a), int(b), int(c)
    if y is None:
        return None

    tm = _TIME_RE.search(t[m.end():]) or _TIME_RE.search(t[: m.start()])
    hh = mm = ss = None
    if tm:
        hh, mm = int(tm.group(1)), int(tm.group(2))
        ss = int(tm.group(3)) if tm.group(3) else 0

    if 1300 <= y <= 1499:
        jd = _mk_jalali(y, mo, d)
        if jd is None:
            return None
        gd = jd.togregorian()
        calendar = "jalali"
        printed = f"{y:04d}/{mo:02d}/{d:02d}"
    elif 1990 <= y <= 2100:
        try:
            gd = datetime(y, mo, d).date()
        except ValueError:
            return None
        calendar = "gregorian"
        jd = jdatetime.date.fromgregorian(date=gd)
        printed = f"{jd.year:04d}/{jd.month:02d}/{jd.day:02d}"
    else:
        return None

    if hh is not None:
        dt = datetime(gd.year, gd.month, gd.day, hh, mm, ss, tzinfo=TEHRAN)
        jalali = f"{printed} {hh:02d}:{mm:02d}"
        iso = dt.isoformat()
        date_only = False
    else:
        dt = None
        jalali = printed
        iso = gd.isoformat()
        date_only = True
    return ParsedDate(raw=m.group(0) + (" " + tm.group(0) if tm else ""), jalali=jalali,
                      calendar=calendar, iso=iso, dt=dt, date_only=date_only)


def is_plausible(p: ParsedDate, now: datetime, max_age_days: int,
                 future_skew: timedelta = timedelta(minutes=15)) -> bool:
    """Not in the future and not older than max_age_days."""
    if p.dt is not None:
        ref = p.dt
    else:
        d = datetime.fromisoformat(p.iso).date()
        ref = datetime(d.year, d.month, d.day, 23, 59, tzinfo=TEHRAN)
        # date-only: compare whole days
        if d > now.astimezone(TEHRAN).date():
            return False
        return (now.astimezone(TEHRAN).date() - d).days <= max_age_days
    if ref > now + future_skew:
        return False
    return now - ref <= timedelta(days=max_age_days)
