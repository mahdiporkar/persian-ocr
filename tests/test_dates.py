from receipt_reader.dates import is_plausible, parse_datetime

from .conftest import NOW


def test_jalali_with_time():
    p = parse_datetime("تاریخ: ۱۴۰۵/۰۸/۲۴ ساعت ۱۱:۴۲")
    assert p.jalali == "1405/08/24 11:42"
    assert p.iso == "2026-11-15T11:42:00+03:30"
    assert p.calendar == "jalali"


def test_variants():
    assert parse_datetime("1405-8-24 09:05").jalali == "1405/08/24 09:05"
    assert parse_datetime("05/08/24").jalali == "1405/08/24"
    assert parse_datetime("24/08/1405").jalali == "1405/08/24"
    g = parse_datetime("2026/11/15 11:42")
    assert g.calendar == "gregorian" and g.jalali == "1405/08/24 11:42"


def test_invalid():
    assert parse_datetime("1405/13/40") is None
    assert parse_datetime("no date") is None


def test_esfand_leap():
    assert parse_datetime("1403/12/30") is not None  # 1403 is a leap year


def test_plausibility():
    assert is_plausible(parse_datetime("1405/08/24 11:42"), NOW, 3)
    assert not is_plausible(parse_datetime("1405/08/24 18:00"), NOW, 3)   # future
    assert not is_plausible(parse_datetime("1405/08/10 10:00"), NOW, 3)   # too old
    assert is_plausible(parse_datetime("1405/08/21 13:00"), NOW, 3)
    assert is_plausible(parse_datetime("1405/08/22"), NOW, 3)              # date only
    assert not is_plausible(parse_datetime("1405/08/25"), NOW, 3)          # tomorrow
