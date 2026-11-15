import pytest

from receipt_reader.normalize import (find_labels, format_fa_number, normalize_name, normalize_text,
                                      parse_amount_digits, to_ascii_digits)


def test_persian_and_arabic_digits():
    assert to_ascii_digits("۰۱۲۳۴۵۶۷۸۹") == "0123456789"
    assert to_ascii_digits("٠١٢٣٤٥٦٧٨٩") == "0123456789"
    assert to_ascii_digits("مبلغ ۴۵۰ و ٤٥٠") == "مبلغ 450 و 450"


def test_arabic_letters_zwnj_tatweel_bidi():
    assert normalize_text("علي كريمي") == "علی کریمی"
    assert normalize_text("می‌شود") == "میشود"
    assert normalize_text("پیگیــــری") == "پیگیری"
    assert normalize_text("‏IR12‎  34") == "IR12 34"


@pytest.mark.parametrize("raw,val", [
    ("4,500,000,000", 4_500_000_000),
    ("۴٬۵۰۰٬۰۰۰٬۰۰۰", 4_500_000_000),
    ("٤٬٥٠٠٬٠٠٠٬٠٠٠", 4_500_000_000),
    ("4.500.000.000", 4_500_000_000),
    ("4 500 000 000", 4_500_000_000),
    ("450000000", 450_000_000),
    ("450,000,000 ریال", 450_000_000),
    ("450,000,000.00", 450_000_000),
])
def test_parse_amount_digits(raw, val):
    assert parse_amount_digits(raw) == val


@pytest.mark.parametrize("raw", ["4,50,000", "12,34", "abc", "", None, "4.5", "1,0000"])
def test_parse_amount_rejects_ambiguous(raw):
    assert parse_amount_digits(raw) is None


def test_labels():
    assert {"tracking"} <= find_labels("شماره پیگیری: 12345678")
    assert "failed" in find_labels("وضعیت: ناموفق")
    assert "success" not in find_labels("وضعیت: ناموفق")
    assert "success" in find_labels("تراکنش موفق")
    assert {"amount", "amount_words"} <= find_labels("مبلغ به حروف: چهارصد میلیون ریال")
    assert "balance" in find_labels("مانده حساب: 1,000")
    assert "fee" in find_labels("کارمزد 5,000 ریال")
    assert "satna" in find_labels("فرم درخواست ساتنا")


def test_format_and_names():
    assert format_fa_number(450000000) == "۴۵۰٬۰۰۰٬۰۰۰"
    assert normalize_name("آقای علي  كريمي") == normalize_name("علی کریمی")
