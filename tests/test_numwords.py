import pytest

from receipt_reader.numwords import parse_persian_number_words as p


@pytest.mark.parametrize("text,value", [
    ("چهارصد و پنجاه میلیون ریال", 450_000_000),
    ("چهار میلیارد و پانصد میلیون ریال", 4_500_000_000),
    ("یک میلیارد و پانصد میلیون", 1_500_000_000),
    ("یکصد و بیست و پنج هزار تومان", 125_000),
    ("صد و بیست و پنج هزار", 125_000),
    ("دویست و پنجاه هزار ریال", 250_000),
    ("سیصد و شصت و پنج", 365),
    ("هفتصد و هشتاد و نه میلیون و ششصد و پنجاه و چهار هزار و سیصد و بیست و یک ریال", 789_654_321),
    ("دوازده میلیون", 12_000_000),
    ("چهار صد و پنجاه میلیون", 450_000_000),          # «چهار صد» written apart
    ("چهارصدوپنجاه میلیون ریال", 450_000_000),        # run together
    ("چهارصد‌و‌پنجاه میلیون", 450_000_000),  # with ZWNJ
    ("۴۵۰ میلیون ریال", 450_000_000),                  # mixed digits
    ("450 میلیون ریال", 450_000_000),
    ("دو میلیارد ریال", 2_000_000_000),
    ("مبلغ به حروف: پانصد میلیون ریال", 500_000_000),
    ("فقط چهل و پنج میلیون تومان تمام", 45_000_000),
    ("صد هزار میلیون", 100_000_000_000),
    ("هزار", 1000),
    ("نود و نه", 99),
    ("نوزده هزار", 19_000),
    ("سي ميليون", 30_000_000),                        # Arabic ي
])
def test_words(text, value):
    r = p(text)
    assert r is not None, text
    assert r.value == value


def test_currency():
    assert p("پانصد میلیون ریال").currency == "rial"
    assert p("پانصد میلیون تومان").currency == "toman"
    assert p("پانصد میلیون").currency is None


@pytest.mark.parametrize("text", ["", None, "علی رضایی", "پانصد میلیون ریال به حساب علی",
                                  "پانصد ریال و صد تومان", "میلیون هزار میلیون"])
def test_rejects_garbage(text):
    assert p(text) is None


def test_the_real_failure_is_caught():
    # handwritten 1,500,000,000 vs words «چهارصد و پنجاه میلیون»: the words parser must disagree
    assert p("چهارصد و پنجاه میلیون ریال").value != 1_500_000_000
